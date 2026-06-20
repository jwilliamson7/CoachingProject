#!/usr/bin/env python
"""
Regression baseline for coach tenure (Reviewer 1.4): predict CONTINUOUS tenure
(years) with XGBoost, leakage-free, then discretize the predictions into the
three tenure classes and compare ordinal metrics against the ordinal classifier.

Validates the reviewer's suggestion to "use both methods." Expectation (and
finding): regressing continuous tenure and binning generalizes WORSE than the
purpose-built ordinal classifier.

Design (identical protocol to the classifier so the comparison is fair):
  - Modern-era (1970+) population, 171 features, same coach-level stratified
    splits across 50 seeds (stratified on the tenure CLASS, as the classifier).
  - Continuous target = reconstructed tenure years (validated 97.9% vs stored class).
  - SVD imputation fit on each split's training partition only.
  - Same SHAP-selected top-K feature set and tuned hyperparameters as the classifier.

Usage:
    python scripts/train_tenure_regression.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor

from model import ordinal_metrics
from model.config import MODEL_CONFIG, ORDINAL_CONFIG, OPTIMIZED_XGBOOST_PARAMS
from model.pipeline import load_modeling_data, shap_feature_ranking, best_k, leakage_free_split
from scripts.data.engineer_career_features import (
    load_history, hiring_team_info, canon_employer, classify_role, is_nfl,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_SEEDS = 50


def tenure_class(t):
    return 0 if t <= 2 else (1 if t <= 4 else 2)


def reconstruct_tenure(coach, hire_year):
    """Consecutive HC seasons at the hiring franchise starting at hire_year."""
    h = load_history(coach)
    if h is None:
        return None
    name, level = hiring_team_info(h, hire_year)
    if not name:
        return None
    fkey = canon_employer(name, level, hire_year)
    hc_years = set()
    for _, r in h.iterrows():
        y = int(r["Year"])
        if (y >= hire_year and is_nfl(r.get("Level"))
                and classify_role(r.get("Role", "")) == "HC"
                and canon_employer(r.get("Employer", ""), r.get("Level"), y) == fkey):
            hc_years.add(y)
    t, y = 0, hire_year
    while y in hc_years:
        t += 1
        y += 1
    return t if t > 0 else None


def reg_params():
    p = {k: v for k, v in OPTIMIZED_XGBOOST_PARAMS.items()}
    p.update(objective="reg:squarederror", n_jobs=-1, tree_method="hist", random_state=42)
    return p


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    n = len(a); m = a.mean(); se = a.std(ddof=1) / np.sqrt(n)
    return m, m - stats.t.ppf(0.975, n - 1) * se, m + stats.t.ppf(0.975, n - 1) * se


def main():
    df, X, y_class = load_modeling_data()
    y_cont = pd.Series([reconstruct_tenure(c, int(yr)) for c, yr in zip(df["Coach Name"], df["Year"])],
                       index=df.index)
    keep = y_cont.notna()
    print(f"Regression on {keep.sum()} instances (continuous tenure reconstructed)")

    idx = shap_feature_ranking()[:best_k()]
    bk = len(idx)
    print(f"Using top-{bk} SHAP features and tuned params (reg:squarederror)")

    reg_metrics = {"r2": [], "mae_years": [], "spearman": [], "pearson": []}
    ord_from_reg = {"qwk": [], "mae": [], "adjacent_accuracy": [], "macro_f1": []}

    for seed in range(N_SEEDS):
        # Shared leakage-free split (stratified on the tenure CLASS, like the
        # classifier); the continuous target is aligned via the split indices.
        split = leakage_free_split(df, X, y_class, seed, feature_indices=idx)
        ytr = y_cont.loc[split.train_index].values.astype(float)
        yte = y_cont.loc[split.test_index].values.astype(float)
        tr_ok = ~np.isnan(ytr); te_ok = ~np.isnan(yte)

        Xtr_i = split.X_train[tr_ok]
        Xte_i = split.X_test[te_ok]

        reg = XGBRegressor(**reg_params())
        reg.fit(Xtr_i, ytr[tr_ok])
        pred = reg.predict(Xte_i)
        yte_v = yte[te_ok]

        reg_metrics["r2"].append(1 - np.sum((yte_v - pred) ** 2) / np.sum((yte_v - yte_v.mean()) ** 2))
        reg_metrics["mae_years"].append(np.mean(np.abs(yte_v - pred)))
        reg_metrics["spearman"].append(stats.spearmanr(yte_v, pred).statistic)
        reg_metrics["pearson"].append(np.corrcoef(yte_v, pred)[0, 1])

        # discretize predictions and true years into classes, score ordinal metrics
        yp_cls = np.array([tenure_class(max(1, round(p))) for p in pred])
        yt_cls = np.array([tenure_class(t) for t in yte_v])
        m = ordinal_metrics(yt_cls, yp_cls, None, ORDINAL_CONFIG["class_names"])
        for k in ord_from_reg:
            ord_from_reg[k].append(m[k])

    print("\n" + "=" * 70)
    print("REGRESSION (continuous tenure) -- leakage-free, 50 seeds")
    print("=" * 70)
    for k, lab in [("r2", "R^2"), ("mae_years", "MAE (years)"),
                   ("pearson", "Pearson r"), ("spearman", "Spearman rho")]:
        m, lo, hi = ci(reg_metrics[k])
        print(f"  {lab:<14} {m:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n" + "=" * 70)
    print("ORDINAL METRICS: regression+discretize vs the ordinal classifier")
    print("=" * 70)
    print(f"{'metric':<14}{'regression->bin':>22}{'ordinal clf':>16}")
    # Ordinal classifier reference: pull the locked headline straight from the
    # parsimony results at its best-K (single source of truth), not a hardcode.
    with open(os.path.join(project_root, "analysis", "parsimony_results.pkl"), "rb") as f:
        _pres = pickle.load(f)["results"]
    _bk = max(_pres, key=lambda k: _pres[k]["qwk"]["mean"])
    clf_ref = {m: _pres[_bk][m]["mean"] for m in ["qwk", "mae", "adjacent_accuracy", "macro_f1"]}
    for k, lab in [("qwk", "QWK"), ("mae", "MAE"), ("adjacent_accuracy", "Adj.Acc"), ("macro_f1", "Macro F1")]:
        m, lo, hi = ci(ord_from_reg[k])
        print(f"{lab:<14}{m:>10.3f} [{lo:.3f},{hi:.3f}]{clf_ref[k]:>16.3f}")

    out = os.path.join(project_root, "analysis", "regression_results.pkl")
    with open(out, "wb") as f:
        pickle.dump({"reg_metrics": reg_metrics, "ord_from_reg": ord_from_reg, "best_k": bk}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
