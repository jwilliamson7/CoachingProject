#!/usr/bin/env python
"""
Two robustness checks for the firing-survival pivot (JQAS):

  1. TUNING DEPTH -- is the XGB-AFT search deep enough? Re-tune with 150 random
     draws (vs the bake-off's 25-40) on coach-level CV; compare CV C-index. Cox/AFT
     grids are tiny (effectively exhaustive already), so only XGB-AFT is at issue.

  2. IMPUTATION ROBUSTNESS -- does imputing the structurally-absent OC/DC/HC blocks
     drive results? Compare, on the SAME 50 leakage-free splits and top-20 features:
       - Cox            (SVD-imputed; lifelines needs complete data)
       - XGB-AFT        (SVD-imputed; status quo)
       - XGB-AFT native (RAW, no imputation at all -- xgboost handles NaN natively)
     If native-NaN XGB-AFT matches the imputed version, the SVD imputation is not
     distorting the firing results -- the reviewer's missing-data rebuttal.

Reuses the bake-off primitives (no duplicated split/fit logic).

Usage:
    python scripts/survival_imputation_tuning.py
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

from model.pipeline import load_modeling_data, leakage_free_split
from scripts.survival_analysis import global_max_season, build_survival_targets, tci
from scripts.survival_models import (
    CoxModel, XGBAFTModel, cox_builder, xgbaft_builder, random_search,
    precompute_cv_folds, COX_SPACE, XGBAFT_SPACE, N_SEEDS,
)

warnings.filterwarnings("ignore")

TOP_K = 20


def main():
    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]
    names = list(X.columns)

    # top-20 survival-SHAP features from the firing experiment
    pkl = pickle.load(open(os.path.join(project_root, "analysis",
                                        "survival_firing_experiment.pkl"), "rb"))
    rank_names = pkl["ranking"][:TOP_K]
    idx = np.array([names.index(n) for n in rank_names])
    n_block_nan = int(np.isnan(X.values[:, idx]).any(axis=0).sum())
    print(f"{len(df)} stints | top-{TOP_K} features "
          f"({n_block_nan} of them carry structural NaN) | "
          f"fired={int(evt.sum())}, censored={int((evt==0).sum())}")

    # ---- 1. tuning depth ----
    folds = precompute_cv_folds(df, X, y, dur, evt, idx)
    rng = np.random.default_rng(0)
    print("\n[1] TUNING DEPTH (coach-level CV C-index)")
    cox_p, cox_cv = random_search(cox_builder, COX_SPACE, folds, 15, rng)
    print(f"  Cox     (15/15 exhaustive):  CV C-index={cox_cv:.4f}  {cox_p}")
    for n_it in (25, 75, 150):
        p, s = random_search(xgbaft_builder, XGBAFT_SPACE, folds, n_it,
                             np.random.default_rng(1))
        print(f"  XGB-AFT ({n_it:>3} draws):         CV C-index={s:.4f}")
        xgb_p = p  # keep the deepest
    print(f"  -> XGB-AFT best params: {xgb_p}")

    # ---- 2. imputation robustness (50-seed C-index) ----
    print("\n[2] IMPUTATION ROBUSTNESS (50-seed leakage-free C-index)")
    cox_c, xgi_c, xgr_c = [], [], []
    for seed in range(N_SEEDS):
        sp = leakage_free_split(df, X, y, seed, feature_indices=idx)
        Ttr = dur.loc[sp.train_index].values.astype(float)
        Etr = evt.loc[sp.train_index].values.astype(int)
        Tte = dur.loc[sp.test_index].values.astype(float)
        Ete = evt.loc[sp.test_index].values.astype(int)

        # imputed (status quo)
        cm = CoxModel(**cox_p).fit(sp.X_train, Ttr, Etr)
        cox_c.append(concordance_index(Tte, cm.surv_score(sp.X_test), Ete))
        xi = XGBAFTModel(dist="normal", scale=xgb_p["scale"],
                         num_boost_round=xgb_p["num_boost_round"], eta=xgb_p["eta"],
                         max_depth=xgb_p["max_depth"], subsample=xgb_p["subsample"],
                         colsample_bytree=xgb_p["colsample_bytree"],
                         min_child_weight=xgb_p["min_child_weight"],
                         reg_lambda=xgb_p["reg_lambda"]).fit(sp.X_train, Ttr, Etr)
        xgi_c.append(concordance_index(Tte, xi.surv_score(sp.X_test), Ete))

        # raw, NO imputation (xgboost native NaN)
        Xtr_raw = X.loc[sp.train_index].values[:, idx]
        Xte_raw = X.loc[sp.test_index].values[:, idx]
        xr = XGBAFTModel(dist="normal", scale=xgb_p["scale"],
                         num_boost_round=xgb_p["num_boost_round"], eta=xgb_p["eta"],
                         max_depth=xgb_p["max_depth"], subsample=xgb_p["subsample"],
                         colsample_bytree=xgb_p["colsample_bytree"],
                         min_child_weight=xgb_p["min_child_weight"],
                         reg_lambda=xgb_p["reg_lambda"]).fit(Xtr_raw, Ttr, Etr)
        xgr_c.append(concordance_index(Tte, xr.surv_score(Xte_raw), Ete))

    for label, arr in [("Cox (SVD-imputed)", cox_c),
                       ("XGB-AFT (SVD-imputed)", xgi_c),
                       ("XGB-AFT (raw, NO imputation)", xgr_c)]:
        m, lo, hi = tci(arr)
        print(f"  {label:<30} C-index {m:.3f} [{lo:.3f}, {hi:.3f}]")
    dm = tci(np.array(xgr_c) - np.array(xgi_c))
    print(f"  delta (native - imputed XGB): {dm[0]:+.4f} [{dm[1]:+.4f}, {dm[2]:+.4f}]")

    out = os.path.join(project_root, "analysis", "survival_imputation_tuning.pkl")
    with open(out, "wb") as f:
        pickle.dump({"cox_imputed": cox_c, "xgb_imputed": xgi_c,
                     "xgb_native_nan": xgr_c, "xgb_params": xgb_p,
                     "cox_params": cox_p, "top_k": TOP_K}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
