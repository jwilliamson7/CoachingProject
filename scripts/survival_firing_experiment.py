#!/usr/bin/env python
"""
Firing-survival experiment on the cleaned 357-stint population with FIRING-aware
events (analysis/event_labels_final.csv). Answers three questions for the JQAS
pivot, with NO ordinal-classifier comparison:

  1. What drives firing? Survival-native SHAP ranking (XGB-AFT TreeSHAP) over all
     181 features, flagging the new structural-missingness indicators
     (cf_ever_nfl_oc/dc) and the re-tested tier 5 (org instability) / tier 6
     (roster talent) blocks.
  2. Do the new features help? Ablation: best leakage-free CV C-index for
     ALL features vs ALL-minus-tier56 vs ALL-minus-flags vs base, each with its
     own SHAP-ranked top-K (Cox).
  3. Best firing model: 50-seed leakage-free C-index for Cox / Weibull AFT /
     XGB-AFT at the full-pool best K.

Reuses the bake-off + feature-selection primitives (no duplicated split/impute/
fit logic). Hyperparameters tuned once on the full feature set (coach-level CV,
C-index objective) and locked.

Usage:
    python scripts/survival_firing_experiment.py [--quick]
"""

import os
import sys
import pickle
import argparse
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from model.pipeline import load_modeling_data
from scripts.data.matrix_factorization_imputation import SVDImputer
from scripts.survival_analysis import global_max_season, build_survival_targets, tci
from scripts.survival_models import (
    cox_builder, weibull_builder, xgbaft_builder, random_search, precompute_cv_folds,
    eval_model, COX_SPACE, XGBAFT_SPACE, N_SEEDS,
)
from scripts.survival_feature_selection import (
    survival_importance, full_cv_folds, cv_cindex_at_k,
)

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)        # FIRING events
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]
    names = list(X.columns)
    T, E = dur.values.astype(float), evt.values.astype(int)
    print(f"Firing-survival experiment: {len(df)} stints, {X.shape[1]} features, "
          f"events(fired)={int(E.sum())}, censored={int((E==0).sum())}")

    tier56 = {i for i, n in enumerate(names)
              if n.startswith("org_") or n.startswith("hire_")}
    flags = {i for i, n in enumerate(names)
             if n in ("cf_ever_nfl_oc", "cf_ever_nfl_dc")}
    print(f"tier5/6 cols: {len(tier56)} | structural flags: {len(flags)}")

    # ---- tune XGB-AFT + Cox once on the full feature set (leakage-free CV) ----
    idx_all = np.arange(X.shape[1])
    folds_sel = precompute_cv_folds(df, X, y, dur, evt, idx_all)
    rng = np.random.default_rng(0)
    n_xgb = 12 if args.quick else 25
    xgb_p, xgb_cv = random_search(xgbaft_builder, XGBAFT_SPACE, folds_sel, n_xgb, rng)
    cox_p, cox_cv = random_search(cox_builder, COX_SPACE, folds_sel, 12, rng)
    print(f"\nTuned XGB-AFT (full): CV C-index={xgb_cv:.3f} {xgb_p}")
    print(f"Tuned Cox    (full): CV C-index={cox_cv:.3f} {cox_p}")

    # ---- survival-native SHAP ranking over ALL features ----
    Ximp = SVDImputer().fit(X.values).transform(X.values)
    rank, src = survival_importance(Ximp, T, E, xgb_p)
    print(f"\nSurvival SHAP ranking ({src}) -- top 25 firing drivers:")
    for r, i in enumerate(rank[:25], 1):
        tag = "[tier5/6]" if i in tier56 else ("[flag]" if i in flags else "")
        print(f"  {r:>2}. {names[i]:<34} {tag}")
    flag_ranks = {names[i]: int(np.where(rank == i)[0][0]) + 1 for i in flags}
    t56_ranks = sorted((int(np.where(rank == i)[0][0]) + 1, names[i]) for i in tier56)
    print(f"\nstructural flag ranks: {flag_ranks}")
    print(f"tier5/6 ranks (best 5): {t56_ranks[:5]}")

    # ---- ablation: best CV C-index per feature pool (Cox), own top-K ----
    folds_full = full_cv_folds(df, X, y, dur, evt)
    K_GRID = [5, 10, 15, 20, 30, 40, 60, 80]
    pools = {
        "ALL (181)": set(idx_all.tolist()),
        "no tier5/6": set(idx_all.tolist()) - tier56,
        "no OC/DC flags": set(idx_all.tolist()) - flags,
        "base (no tier5/6 + no flags)": set(idx_all.tolist()) - tier56 - flags,
    }
    print("\n" + "=" * 64)
    print("ABLATION -- best leakage-free CV C-index (Cox), SHAP-ranked top-K")
    print("=" * 64)
    abl = {}
    for label, pool in pools.items():
        pool_rank = [i for i in rank if i in pool]
        best_k, best_s = None, -1.0
        for K in K_GRID + [len(pool_rank)]:
            idx = np.array(pool_rank[:K])
            s = cv_cindex_at_k(cox_builder, cox_p, folds_full, idx)
            if s > best_s:
                best_k, best_s = K, s
        abl[label] = {"best_k": best_k, "cv_cindex": best_s}
        print(f"  {label:<32} bestK={best_k:>3}  CV C-index={best_s:.3f}")

    # ---- headline bake-off at full-pool best K (50-seed C-index) ----
    full_rank = [i for i in rank]
    bk = abl["ALL (181)"]["best_k"]
    idx_best = np.array(full_rank[:bk])
    print(f"\n" + "=" * 64)
    print(f"FIRING-MODEL BAKE-OFF (50-seed C-index) at top-{bk} features")
    print("=" * 64)
    bake = {}
    for name, builder, params in [
        ("Cox", cox_builder, cox_p),
        ("Weibull AFT", weibull_builder, {"penalizer": cox_p["penalizer"]}),
        ("XGB-AFT", xgbaft_builder, xgb_p),
    ]:
        res = eval_model(builder, params, df, X, y, dur, evt, idx_best)
        m, lo, hi = tci(res["c_index"])
        bake[name] = {"c_index": (m, lo, hi)}
        print(f"  {name:<14} C-index {m:.3f} [{lo:.3f}, {hi:.3f}]")
    print("=" * 64)

    out = os.path.join(project_root, "analysis", "survival_firing_experiment.pkl")
    with open(out, "wb") as f:
        pickle.dump({"ranking": [names[i] for i in rank],
                     "flag_ranks": flag_ranks, "tier56_ranks": t56_ranks,
                     "ablation": abl, "bakeoff": bake,
                     "xgb_params": xgb_p, "cox_params": cox_p,
                     "n_stints": len(df), "n_fired": int(E.sum())}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
