#!/usr/bin/env python
"""
Ordinal vs. multiclass comparison with proper significance testing
(Reviewers 1.2 and 1.3): the prior paper compared the two models with a naive
paired t-test that treats the 50 resampled train/test splits as independent.
They are NOT independent (the training sets overlap), which deflates the
variance and inflates significance. This script reports, for every metric:

  - paired t-test (the original, naive test, kept for reference)
  - Wilcoxon signed-rank test (distribution-free; R1.2)
  - Shapiro-Wilk normality test on the per-seed differences (R1.2: checks
    whether the paired-t normality assumption even holds)
  - Nadeau-Bengio corrected resampled t-test (R1.3: corrects the variance for
    the train/test overlap across the 50 resamples)

Protocol is identical to the parsimony headline so the numbers are comparable:
modern-era (1970+) population, 171 features, the same 50 coach-level stratified
splits, SVD imputation fit on each split's training partition only, and the same
top-K SHAP-ranked feature subset (best K read from the parsimony results).

Usage:
    python scripts/compare_ordinal_multiclass.py
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

from model import ordinal_metrics
from model.config import MODEL_CONFIG, ORDINAL_CONFIG
from model.pipeline import (
    load_modeling_data, shap_feature_ranking, best_k, leakage_free_split, fit_model,
    ordinal_model, multiclass_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_SEEDS = 50
# Metrics where a HIGHER value is better. MAE is the only "lower is better".
HIGHER_BETTER = {"qwk", "adjacent_accuracy", "exact_accuracy", "macro_f1",
                 "auroc", "class0_f1", "class1_f1", "class2_f1"}


def metrics_dict(y_true, y_pred, y_proba):
    m = ordinal_metrics(y_true, y_pred, y_proba, ORDINAL_CONFIG["class_names"])
    out = {k: m[k] for k in ["mae", "qwk", "adjacent_accuracy", "exact_accuracy", "macro_f1"]}
    out["auroc"] = m.get("auroc", np.nan)
    # per-class F1 keyed by class index
    for cls_idx, (cls_name, cm) in enumerate(m["per_class"].items()):
        out[f"class{cls_idx}_f1"] = cm["f1"]
    return out


def nadeau_bengio_ttest(diffs, test_frac):
    """Nadeau-Bengio (2003) corrected resampled t-test.

    For J random train/test resamples with a fixed test fraction rho, the
    naive variance of the mean difference, var/J, ignores the correlation
    induced by overlapping training sets. The corrected variance multiplies
    the sample variance by (1/J + rho/(1-rho)); rho/(1-rho) = n_test/n_train.
    df = J - 1.
    """
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    J = len(d)
    mean = d.mean()
    var = d.var(ddof=1)
    ratio = test_frac / (1.0 - test_frac)          # n_test / n_train
    corrected_var = (1.0 / J + ratio) * var
    if corrected_var <= 0:
        return mean, np.nan, np.nan
    t_stat = mean / np.sqrt(corrected_var)
    p = 2.0 * stats.t.sf(abs(t_stat), df=J - 1)     # two-sided
    return mean, t_stat, p


def main():
    df, X, y = load_modeling_data()
    idx = shap_feature_ranking()[:best_k()]
    bk = len(idx)
    test_frac = MODEL_CONFIG["test_size"]
    print(f"Ordinal vs multiclass: {len(df)} instances, top-{bk} SHAP features, "
          f"{N_SEEDS} coach-level splits (test_frac={test_frac})")

    ord_runs, mc_runs = [], []
    for seed in range(N_SEEDS):
        # Same leakage-free split feeds BOTH models so the comparison is paired.
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)
        Xte = pd.DataFrame(split.X_test)
        yte_a = split.y_test

        om = fit_model(split, ordinal_model, seed)
        ord_runs.append(metrics_dict(yte_a, om.predict(Xte), om.predict_proba(Xte)))

        mm = fit_model(split, multiclass_model, seed)
        mc_runs.append(metrics_dict(yte_a, mm.predict(Xte), mm.predict_proba(Xte)))

        if (seed + 1) % 10 == 0:
            print(f"  ...{seed + 1}/{N_SEEDS} seeds")

    metrics = ["qwk", "mae", "adjacent_accuracy", "exact_accuracy", "macro_f1",
               "auroc", "class0_f1", "class1_f1", "class2_f1"]

    def tci(a):
        a = np.asarray(a, float); a = a[~np.isnan(a)]
        n = len(a); m = a.mean(); se = a.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        return m, m - h, m + h

    rows = {}
    print("\n" + "=" * 110)
    print("ORDINAL vs MULTICLASS (means with t-dist 95% CI, and per-seed difference tests)")
    print("=" * 110)
    hdr = (f"{'metric':<18}{'ordinal':>20}{'multiclass':>20}{'mean diff':>11}"
           f"{'t-test p':>10}{'Wilcoxon p':>12}{'Shapiro p':>11}{'N-B p':>9}")
    print(hdr)
    print("-" * 110)

    for k in metrics:
        ov = np.array([r[k] for r in ord_runs], float)
        mv = np.array([r[k] for r in mc_runs], float)
        diffs = ov - mv
        d = diffs[~np.isnan(diffs)]

        om_m, om_lo, om_hi = tci(ov)
        mm_m, mm_lo, mm_hi = tci(mv)

        # Naive paired t-test (two-sided)
        t_p = stats.ttest_rel(ov, mv, nan_policy="omit").pvalue
        # Wilcoxon signed-rank (two-sided); guard the all-zero / tiny case
        try:
            w_p = stats.wilcoxon(d, zero_method="wilcox").pvalue
        except ValueError:
            w_p = np.nan
        # Shapiro-Wilk normality of the per-seed differences
        try:
            sh_p = stats.shapiro(d).pvalue
        except ValueError:
            sh_p = np.nan
        # Nadeau-Bengio corrected resampled t-test
        nb_mean, nb_t, nb_p = nadeau_bengio_ttest(diffs, test_frac)

        rows[k] = {
            "ordinal_mean": om_m, "ordinal_ci": (om_lo, om_hi),
            "multiclass_mean": mm_m, "multiclass_ci": (mm_lo, mm_hi),
            "mean_diff": float(np.nanmean(diffs)),
            "ord_better": bool((np.nanmean(diffs) > 0) == (k in HIGHER_BETTER)),
            "ord_wins": int(np.sum((d > 0) if k in HIGHER_BETTER else (d < 0))),
            "n": int(len(d)),
            "ttest_p": float(t_p), "wilcoxon_p": float(w_p),
            "shapiro_p": float(sh_p), "nb_p": float(nb_p), "nb_t": float(nb_t),
            "ord_per_seed": ov.tolist(), "mc_per_seed": mv.tolist(),
        }

        def pstr(p):
            return "  nan" if np.isnan(p) else f"{p:.4f}"
        print(f"{k:<18}{om_m:>9.3f} [{om_lo:.3f},{om_hi:.3f}]"
              f"{mm_m:>9.3f} [{mm_lo:.3f},{mm_hi:.3f}]"
              f"{np.nanmean(diffs):>+11.4f}{pstr(t_p):>10}{pstr(w_p):>12}"
              f"{pstr(sh_p):>11}{pstr(nb_p):>9}")

    print("-" * 110)
    print("Notes: p-values are two-sided. Shapiro p<0.05 => per-seed diffs are non-normal,")
    print("       so the distribution-free Wilcoxon and the overlap-corrected Nadeau-Bengio")
    print("       tests are the appropriate ones to report. 'ord_wins' (in the pickle) counts")
    print("       seeds where ordinal beat multiclass on that metric.")

    out = os.path.join(project_root, "analysis", "ordinal_vs_multiclass.pkl")
    with open(out, "wb") as f:
        pickle.dump({"results": rows, "best_k": bk, "n_seeds": N_SEEDS,
                     "test_frac": test_frac}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
