#!/usr/bin/env python
"""
Survival-native feature selection: the bake-off (survival_models.py) handicapped
the survival models by feeding them the feature set chosen for the CLASSIFIER
(SHAP-ranked for bucket classification, K tuned for the classifier's QWK). The
hazard of being fired need not be driven by the same features that separate the
tenure buckets. Here we give the survival model its OWN selection -- mirroring
the classifier's pipeline -- and measure the lift:

  1. Rank all 171 features by XGBoost-AFT importance (TreeSHAP if available, else
     gain) from a model fit on the full known-tenure population.
  2. Sweep K with leakage-free coach-level CV (SVD imputation fit per fold),
     selecting K by the survival-native concordance index.
  3. Re-evaluate the survival models on their own top-K and compare, on the same
     50 leakage-free splits and the same ordinal KPIs, against (a) the classifier
     feature set and (b) the ordinal classifier itself.

Hyperparameters are reused from the bake-off's tuning (analysis/survival_models.pkl)
so this isolates the effect of the feature set.

Usage:
    python scripts/survival_feature_selection.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from model.cross_validation import CoachLevelStratifiedKFold
from model.pipeline import (
    load_modeling_data, shap_feature_ranking, best_k, leakage_free_split,
)
from scripts.survival_analysis import global_max_season, build_survival_targets, tci
from scripts.survival_models import (
    XGBAFTModel, cox_builder, xgbaft_builder, weibull_builder, eval_model, METRICS,
)

warnings.filterwarnings("ignore")

N_SEEDS = 50
K_GRID = [5, 10, 15, 20, 30, 40, 60, 80, 120, 171]


def survival_importance(Ximp, T, E, params):
    """Rank all features by XGB-AFT importance (TreeSHAP, fallback to gain)."""
    m = XGBAFTModel(dist="normal", scale=params["scale"],
                    num_boost_round=params["num_boost_round"], eta=params["eta"],
                    max_depth=params["max_depth"], subsample=params["subsample"],
                    colsample_bytree=params["colsample_bytree"],
                    min_child_weight=params["min_child_weight"],
                    reg_lambda=params["reg_lambda"]).fit(Ximp, T, E)
    p = Ximp.shape[1]
    try:
        import shap
        sv = shap.TreeExplainer(m.bst).shap_values(Ximp)
        imp = np.abs(sv).mean(axis=0)
        src = "TreeSHAP"
    except Exception:
        score = m.bst.get_score(importance_type="gain")  # {'f12': gain, ...}
        imp = np.zeros(p)
        for k, v in score.items():
            imp[int(k[1:])] = v
        src = "gain"
    return np.argsort(imp)[::-1], src


def full_cv_folds(df, X, y, dur, evt, n_splits=5, seed=42):
    """Leakage-free CV folds with FULL imputed feature matrices (no selection)."""
    from scripts.data.matrix_factorization_imputation import SVDImputer
    cv = CoachLevelStratifiedKFold(n_splits=n_splits, random_state=seed)
    groups = df["Coach Name"].values
    Xv, yv = X.values, y.values
    Tv, Ev = dur.values.astype(float), evt.values.astype(int)
    folds = []
    for tr, va in cv.split(Xv, yv, groups):
        imp = SVDImputer().fit(Xv[tr])
        folds.append((imp.transform(Xv[tr]), Tv[tr], Ev[tr],
                      imp.transform(Xv[va]), Tv[va], Ev[va]))
    return folds


def cv_cindex_at_k(builder, params, folds, idx):
    scores = []
    for Xtr, Ttr, Etr, Xva, Tva, Eva in folds:
        try:
            m = builder(params).fit(Xtr[:, idx], Ttr, Etr)
            scores.append(concordance_index(Tva, m.surv_score(Xva[:, idx]), Eva))
        except Exception:
            return -1.0
    return float(np.mean(scores))


def main():
    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]

    clf_idx = shap_feature_ranking()[:best_k()]          # classifier feature set
    bake = pickle.load(open(os.path.join(project_root, "analysis",
                                         "survival_models.pkl"), "rb"))
    xgb_params = bake["tuned"]["XGB-AFT"]["params"]
    cox_params = bake["tuned"]["Cox"]["params"]
    wb_params = bake["tuned"]["Weibull AFT"]["params"]

    # ---- rank features for survival, on full imputed known population ----
    from scripts.data.matrix_factorization_imputation import SVDImputer
    Ximp = SVDImputer().fit(X.values).transform(X.values)
    rank, src = survival_importance(Ximp, dur.values.astype(float),
                                    evt.values.astype(int), xgb_params)
    print(f"Survival feature ranking via XGB-AFT {src}: top-10 indices = "
          f"{list(rank[:10])}")
    overlap = len(set(rank[:20]) & set(clf_idx))
    print(f"Overlap of survival top-20 with classifier top-20: {overlap}/20\n")

    # ---- sweep K by CV C-index (leakage-free) for XGB-AFT and Cox ----
    folds = full_cv_folds(df, X, y, dur, evt)
    print(f"{'K':>5}{'XGB-AFT C-idx':>16}{'Cox C-idx':>12}")
    print("-" * 33)
    sweep = {}
    for K in K_GRID:
        idx = rank[:K]
        cx = cv_cindex_at_k(xgbaft_builder, xgb_params, folds, idx)
        cc = cv_cindex_at_k(cox_builder, cox_params, folds, idx)
        sweep[K] = {"xgb": cx, "cox": cc}
        print(f"{K:>5}{cx:>16.3f}{cc:>12.3f}")
    best_k_xgb = max(sweep, key=lambda k: sweep[k]["xgb"])
    best_k_cox = max(sweep, key=lambda k: sweep[k]["cox"])
    print(f"\nBest K -> XGB-AFT: {best_k_xgb} (C-idx {sweep[best_k_xgb]['xgb']:.3f}); "
          f"Cox: {best_k_cox} (C-idx {sweep[best_k_cox]['cox']:.3f})")

    # ---- 50-seed eval on survival-selected features vs classifier features ----
    surv_idx_xgb = rank[:best_k_xgb]
    surv_idx_cox = rank[:best_k_cox]

    print("\nEvaluating on survival-selected features (50 splits)...")
    runs = {
        f"Cox (surv top-{best_k_cox})":
            eval_model(cox_builder, cox_params, df, X, y, dur, evt, surv_idx_cox),
        f"XGB-AFT (surv top-{best_k_xgb})":
            eval_model(xgbaft_builder, xgb_params, df, X, y, dur, evt, surv_idx_xgb),
        f"Weibull AFT (surv top-{best_k_cox})":
            eval_model(weibull_builder, wb_params, df, X, y, dur, evt, surv_idx_cox),
    }

    # classifier-feature rows straight from the bake-off (already computed)
    bs = bake["summary"]

    def line(name, summ_or_run):
        if isinstance(summ_or_run, dict) and "qwk" in summ_or_run and isinstance(
                summ_or_run["qwk"], (list, np.ndarray)):
            q = tci(summ_or_run["qwk"]); mf = tci(summ_or_run["macro_f1"])[0]
            aa = tci(summ_or_run["adjacent_accuracy"])[0]; au = tci(summ_or_run["auroc"])[0]
            ci = tci(summ_or_run["c_index"])[0] if not np.all(np.isnan(summ_or_run["c_index"])) else np.nan
        else:  # bake summary: dict of (mean,lo,hi) tuples + c_index float
            q = summ_or_run["qwk"]; mf = summ_or_run["macro_f1"][0]
            aa = summ_or_run["adjacent_accuracy"][0]; au = summ_or_run["auroc"][0]
            ci = summ_or_run.get("c_index", np.nan)
        cis = f"{ci:.3f}" if not (ci is None or np.isnan(ci)) else "   -"
        print(f"{name:<30}{q[0]:>7.3f} [{q[1]:.3f},{q[2]:.3f}]{mf:>9.3f}{aa:>8.3f}{au:>8.3f}{cis:>8}")

    print("\n" + "=" * 90)
    print("FEATURE SET COMPARISON (classifier top-20  vs  survival-selected)")
    print("=" * 90)
    print(f"{'model':<30}{'QWK':>19}{'MacroF1':>9}{'AdjAcc':>8}{'AUROC':>8}{'C-idx':>8}")
    print("-" * 90)
    line("Ordinal classifier", bs["Ordinal classifier"])
    line("Cox (clf top-20)", bs["Cox"])
    line(f"Cox (surv top-{best_k_cox})", runs[f"Cox (surv top-{best_k_cox})"])
    line("XGB-AFT (clf top-20)", bs["XGB-AFT"])
    line(f"XGB-AFT (surv top-{best_k_xgb})", runs[f"XGB-AFT (surv top-{best_k_xgb})"])
    line("Weibull AFT (clf top-20)", bs["Weibull AFT"])
    line(f"Weibull AFT (surv top-{best_k_cox})", runs[f"Weibull AFT (surv top-{best_k_cox})"])
    print("-" * 90)

    out = os.path.join(project_root, "analysis", "survival_feature_selection.pkl")
    with open(out, "wb") as f:
        pickle.dump({"ranking": rank.tolist(), "ranking_src": src,
                     "sweep": sweep, "best_k_xgb": int(best_k_xgb),
                     "best_k_cox": int(best_k_cox),
                     "overlap_top20": int(overlap),
                     "runs": runs, "n_seeds": N_SEEDS}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
