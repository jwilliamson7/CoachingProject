#!/usr/bin/env python
"""
Test collapsing the four parallel performance blocks into ONE role-agnostic
"unit performance" block (JQAS; user's idea).

The four 33-stat blocks are Feature 9-140:
  OC  = Feature 9-41   (your OFFENSE's stats        -> "your side")
  DC  = Feature 42-74  (OPPONENT offense you allowed -> "opponent side")
  HC  = Feature 75-107 (your team's stats as HC      -> "your side")
  opp = Feature 108-140(opponents' stats as HC       -> "opponent side")
All are z-scores. Verified by corr with win%: PF__oc +0.17, PF__hc +0.16 (your
side, positive=good) but PF__dc -0.12, PF__opp__hc -0.24 (opponent side,
positive=BAD). So a raw average would cancel; we ORIENT each stat to
"unit quality, positive=good" first:

  unit_S = nanmean( sign_S*OC_S, sign_S*HC_S, -sign_S*DC_S )

where sign_S = -1 for "bad" stats (turnovers/interceptions/penalties), +1 else.
This yields one 33-feature block defined for any coach who held OC/DC/HC -- no
structural block-missingness. The HC-opp block (schedule strength) is kept
separate. We compare firing C-index: FULL (181 feats) vs COLLAPSED.

Usage:
    python scripts/survival_combined_features.py
"""

import os
import re
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

import data_constants as dc
from model.pipeline import load_modeling_data
from scripts.data.engineer_career_features import load_history, classify_role, is_nfl
from scripts.data.matrix_factorization_imputation import SVDImputer
from scripts.survival_analysis import global_max_season, build_survival_targets, tci
from scripts.survival_models import (
    cox_builder, xgbaft_builder, random_search, precompute_cv_folds, eval_model,
    COX_SPACE, XGBAFT_SPACE, N_SEEDS,
)
from scripts.survival_feature_selection import survival_importance, full_cv_folds, cv_cindex_at_k
from scripts.compare_ordinal_multiclass import nadeau_bengio_ttest
from model.config import MODEL_CONFIG

warnings.filterwarnings("ignore")
# Finer K sweep through 10-20 (collapsed model selected K=10 at the boundary),
# plus anchors below/above so FULL is not artificially capped.
K_GRID = [5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40]

# Expanded hyperparameter search (wider grids + more draws than the bake-off
# defaults) now that the collapsed model is the paper's primary representation.
COX_SPACE_WIDE = {
    "penalizer": [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
    "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
}  # 45 combos -> searched exhaustively
N_COX_DRAWS = 45
XGBAFT_SPACE_WIDE = {
    "num_boost_round": [30, 50, 80, 120, 200],
    "eta": [0.02, 0.03, 0.05, 0.1, 0.15],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.6, 0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5, 8],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    "scale": [0.5, 0.8, 1.0, 1.2, 1.5],
}
N_XGB_DRAWS = 200


def fcol(n):
    return f"Feature {n}"


def _role_weights(df):
    """Per stint, count pre-hire NFL seasons in each role -> combine weights.
    Weighting the (already per-season-mean) block values by these counts is
    equivalent to pooling all of the coach's role-seasons (the create_data route),
    up to seasons where league data was missing."""
    w_oc, w_dc, w_hc = (np.zeros(len(df)) for _ in range(3))
    for j, (coach, yr) in enumerate(zip(df["Coach Name"], df["Year"])):
        h = load_history(coach)
        hy = int(yr)
        if h is None:
            continue
        for _, r in h.iterrows():
            if int(r["Year"]) < hy and is_nfl(r.get("Level")):
                rr = classify_role(r.get("Role", ""))
                if rr == "OC":
                    w_oc[j] += 1
                elif rr == "DC":
                    w_dc[j] += 1
                elif rr == "HC":
                    w_hc[j] += 1
    return w_oc, w_dc, w_hc


def build_collapsed(df, X):
    """Replace OC/DC/HC blocks with one orientation-corrected, season-WEIGHTED
    unit_* block; keep opp + the rest. unit_S = weighted mean (by role-season
    count) of sign_S*OC_S, sign_S*HC_S, -sign_S*DC_S, all oriented positive=good."""
    names = dc.get_all_feature_names()             # 140 semantic names
    stat = [names[8 + k].replace("__oc", "") for k in range(33)]
    sign = np.array([-1.0 if any(t in s for t in ("TO", "Int", "Pen")) else 1.0
                     for s in stat])
    w_oc, w_hc, w_dc = _role_weights(df)
    unit = {}
    for k in range(33):
        s = sign[k]
        vals = np.vstack([s * X[fcol(9 + k)].values,        # OC (your offense)
                          s * X[fcol(75 + k)].values,       # HC (your team)
                          -s * X[fcol(42 + k)].values])     # DC (flip: allowed)
        wts = np.vstack([w_oc, w_hc, w_dc]).astype(float)
        mask = ~np.isnan(vals)
        wts = wts * mask
        num = np.where(mask, vals, 0.0) * wts
        den = wts.sum(0)
        unit["unit_" + stat[k]] = np.where(den > 0, num.sum(0) / den, np.nan)
    unit_df = pd.DataFrame(unit, index=X.index)
    core = [fcol(n) for n in range(1, 9)]
    opp = [fcol(108 + k) for k in range(33)]
    named = [c for c in X.columns if not re.fullmatch(r"Feature \d+", c)]
    Xc = pd.concat([X[core], unit_df, X[opp], X[named]], axis=1)
    return Xc


def best_cindex(df, Xmat, y, dur, evt):
    """Tune (Cox exhaustive + XGB), SHAP-rank, sweep K, 50-seed C-index for both."""
    idx_all = np.arange(Xmat.shape[1])
    folds_sel = precompute_cv_folds(df, Xmat, y, dur, evt, idx_all)
    rng = np.random.default_rng(0)
    xgb_p, _ = random_search(xgbaft_builder, XGBAFT_SPACE_WIDE, folds_sel, N_XGB_DRAWS, rng)
    cox_p, _ = random_search(cox_builder, COX_SPACE_WIDE, folds_sel, N_COX_DRAWS, rng)
    Ximp = SVDImputer().fit(Xmat.values).transform(Xmat.values)
    rank, _ = survival_importance(Ximp, dur.values.astype(float),
                                  evt.values.astype(int), xgb_p)
    folds_full = full_cv_folds(df, Xmat, y, dur, evt)
    bk, bs, curve = None, -1, {}
    for K in K_GRID + [Xmat.shape[1]]:
        s = cv_cindex_at_k(cox_builder, cox_p, folds_full, rank[:K])
        curve[K] = s
        if s > bs:
            bk, bs = K, s
    idx = rank[:bk]
    cox = eval_model(cox_builder, cox_p, df, Xmat, y, dur, evt, idx)
    xgb = eval_model(xgbaft_builder, xgb_p, df, Xmat, y, dur, evt, idx)
    return {"best_k": bk, "n_feats": Xmat.shape[1],
            "cox": tci(cox["c_index"]), "xgb": tci(xgb["c_index"]),
            "cox_raw": np.asarray(cox["c_index"]),
            "xgb_raw": np.asarray(xgb["c_index"]),
            "cox_params": cox_p, "xgb_params": xgb_p, "k_curve": curve,
            "top": [Xmat.columns[i] for i in idx[:bk]]}


def main():
    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]

    Xc = build_collapsed(df, X)
    print(f"{len(df)} stints | FULL {X.shape[1]} feats | "
          f"COLLAPSED {Xc.shape[1]} feats (OC/DC/HC -> weighted unit/33; opp/33 kept)")

    res = {}
    for label, Xm in [("FULL (181)", X), ("COLLAPSED (unit+opp)", Xc)]:
        r = best_cindex(df, Xm, y, dur, evt)
        res[label] = r
        print(f"\n{label}: {r['n_feats']} feats, bestK={r['best_k']}")
        print(f"   Cox     C-index {r['cox'][0]:.3f} [{r['cox'][1]:.3f}, {r['cox'][2]:.3f}]")
        print(f"   XGB-AFT C-index {r['xgb'][0]:.3f} [{r['xgb'][1]:.3f}, {r['xgb'][2]:.3f}]")
        print("   K-curve (CV C-index): "
              + "  ".join(f"{k}:{v:.3f}" for k, v in r["k_curve"].items()))
        print(f"   Cox params: {r['cox_params']}")
        print(f"   XGB params: {r['xgb_params']}")
        print(f"   selected ({r['best_k']}): {r['top']}")

    # paired Nadeau-Bengio test on COLLAPSED - FULL (same 50 coach splits per seed,
    # each model at its own CV-selected K) -- the standard corrected comparison.
    full, coll = res["FULL (181)"], res["COLLAPSED (unit+opp)"]
    tf = MODEL_CONFIG["test_size"]
    print("\nPAIRED Nadeau-Bengio (COLLAPSED - FULL, test_frac="
          f"{tf}, J={N_SEEDS}):")
    for mdl, key in [("Cox", "cox_raw"), ("XGB-AFT", "xgb_raw")]:
        m, t, p = nadeau_bengio_ttest(coll[key] - full[key], tf)
        print(f"   {mdl:<8} delta C-index {m:+.4f}  t={t:+.3f}  p={p:.4g}")

    out = os.path.join(project_root, "analysis", "survival_combined_features.pkl")
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
