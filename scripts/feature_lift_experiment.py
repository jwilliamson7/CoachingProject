#!/usr/bin/env python
"""
A/B feature-lift experiment (leakage-free, modern-era 1970+).

Compares three configurations to separate the era-filter effect from the
career-path/rank feature contribution:
  A) all-era, 150 original features      -> known: QWK ~0.292 (parsimony_results.pkl)
  B) 1970+,   150 original features      -> isolates the era-filter effect
  C) 1970+,   150 + 21 engineered feats  -> isolates the new-feature contribution

Everything is leakage-free: SHAP ranking is computed on a feature-only,
fit-on-sample SVD imputation; every reported metric uses imputation fit on each
split's TRAINING partition only (reusing parsimonious_model.train_and_evaluate_single_seed).

Usage:
    python scripts/feature_lift_experiment.py
"""

import os
import sys
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scipy import stats

from model import CoachTenureModel
from scripts.data.matrix_factorization_imputation import SVDImputer
from scripts.parsimonious_model import train_and_evaluate_single_seed
from scripts.shap_analysis import compute_shap_values, compute_aggregated_shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

EXT = os.path.join(project_root, "data", "master_data_extended.csv")
N_SEEDS = 50
COUNTS = [5, 10, 20, 30, 40, 50, 60, 80, 100, 130, 150]
METRICS = ["qwk", "mae", "macro_f1", "adjacent_accuracy", "exact_accuracy"]


def load_known():
    df = pd.read_csv(EXT, index_col=0)
    df = df[df["Coach Tenure Class"] != -1].copy()
    X = df.iloc[:, 2:-2]            # all engineered features (150 original + 21 new)
    y = df["Coach Tenure Class"]
    return df, X, y


def shap_ranking(df, X, y):
    """Leakage-free global SHAP ranking over the columns of X (feature-only impute)."""
    imp = SVDImputer().fit(X.values)
    Xi = imp.transform(X.values)
    m = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    m.fit(pd.DataFrame(Xi), y, verbose=0)
    names = list(X.columns)
    sd = compute_shap_values(m, Xi, names)
    agg = compute_aggregated_shap(sd, names)
    mean_abs = np.abs(agg.values).mean(axis=0)
    return np.argsort(mean_abs)[::-1], mean_abs


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    n = len(a)
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se


def sweep(df, X, y, ranking, counts):
    res = {}
    for K in counts:
        if K > X.shape[1]:
            continue
        idx = ranking[:K]
        acc = {k: [] for k in METRICS}
        for s in range(N_SEEDS):
            mm = train_and_evaluate_single_seed(df, X, y, feature_indices=idx, random_state=s)
            for k in METRICS:
                acc[k].append(mm[k])
        res[K] = {k: np.array(v) for k, v in acc.items()}
    return res


def best_k(res):
    return max(res, key=lambda K: res[K]["qwk"].mean())


def report(name, res, X=None, ranking=None):
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"{'K':>4} {'QWK [95% CI]':>24} {'MAE':>8} {'F1':>8} {'Adj':>7} {'Exact':>7}")
    for K in sorted(res):
        q = ci(res[K]["qwk"]);
        print(f"{K:>4} {q[0]:>7.3f} [{q[1]:.3f},{q[2]:.3f}]"
              f" {res[K]['mae'].mean():>8.3f} {res[K]['macro_f1'].mean():>8.3f}"
              f" {res[K]['adjacent_accuracy'].mean():>7.3f} {res[K]['exact_accuracy'].mean():>7.3f}")
    bk = best_k(res)
    q = ci(res[bk]["qwk"]); f = ci(res[bk]["macro_f1"])
    print(f"  -> BEST K={bk}: QWK {q[0]:.3f} [{q[1]:.3f}, {q[2]:.3f}], F1 {f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}]")
    if X is not None and ranking is not None:
        topidx = ranking[:bk]
        new = [c for c in X.columns[topidx] if str(c).startswith(("cf_", "rf_"))]
        print(f"     new features in top-{bk}: {len(new)} -> {new}")
    return bk, res[bk]


def main():
    df, Xfull, y = load_known()
    n_new = sum(str(c).startswith(("cf_", "rf_")) for c in Xfull.columns)
    print(f"1970+ known instances: {len(df)} | total features: {Xfull.shape[1]} "
          f"(150 original + {n_new} engineered)")
    print("class dist:", dict(y.value_counts().sort_index()))

    # original 150 columns are Feature 1..150 (first 150 of the feature block)
    X150 = Xfull.iloc[:, :150]

    import pickle
    cache_path = os.path.join(project_root, "analysis", "feature_lift_results.pkl")
    # The 150-feature baseline (B) is invariant across feature-engineering
    # iterations (first 150 columns are the original features), so reuse it.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        resB, rank150 = cached["resB"], cached["rank150"]
        print("\nReusing cached 150-feature baseline B (first-150 cols unchanged).")
    else:
        print("\nComputing 150-feature baseline B...")
        rank150, _ = shap_ranking(df, X150, y)
        resB = sweep(df, X150, y, rank150, COUNTS)

    print("Computing extended SHAP ranking + sweep (50 seeds)...")
    rankFull, _ = shap_ranking(df, Xfull, y)
    countsC = COUNTS + [Xfull.shape[1]]
    resC = sweep(df, Xfull, y, rankFull, countsC)

    bkB, _ = report("B) 1970+, 150 ORIGINAL features", resB)
    bkC, _ = report("C) 1970+, 150 + engineered features", resC, Xfull, rankFull)

    qB = ci(resB[bkB]["qwk"]); qC = ci(resC[bkC]["qwk"])
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"A) all-era,  150 feat : QWK 0.292 (prior leakage-free run)")
    print(f"B) 1970+,    150 feat : QWK {qB[0]:.3f} [{qB[1]:.3f}, {qB[2]:.3f}]  (era-filter effect)")
    print(f"C) 1970+, +engineered : QWK {qC[0]:.3f} [{qC[1]:.3f}, {qC[2]:.3f}]  (feature effect)")
    print(f"   feature lift (C - B): {qC[0]-qB[0]:+.3f} QWK")

    import pickle
    out = os.path.join(project_root, "analysis", "feature_lift_results.pkl")
    with open(out, "wb") as fh:
        pickle.dump({"resB": resB, "resC": resC, "rank150": rank150,
                     "rankFull": rankFull, "columns": list(Xfull.columns)}, fh)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
