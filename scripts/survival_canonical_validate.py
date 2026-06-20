#!/usr/bin/env python
"""
Validate the CANONICAL unit-block dataset (collapse now baked into create_data /
data_constants, exact per-season pooling with impute-after-collapse) reproduces
the firing-survival C-index found post-hoc.

The modeling matrix from load_modeling_data() is now natively the collapsed
representation (8 core + 33 unit + 33 opp + 10 hiring + 31 career/recent = 115
features); no on-the-fly build_collapsed is needed. We run the same tune ->
SHAP-rank -> K-sweep -> N_SEEDS C-index protocol (reused from
survival_combined_features.best_cindex) and compare to the post-hoc COLLAPSED
result (Cox ~0.616 at K=18, 50-seed; expect ~equal at 25-seed here).

Usage:
    python scripts/survival_canonical_validate.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season, build_survival_targets
from scripts.survival_combined_features import best_cindex
from scripts.survival_models import N_SEEDS

warnings.filterwarnings("ignore")


def main():
    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]

    unit_cols = [c for c in X.columns if "unit" in c.lower()]
    print(f"{len(df)} stints | CANONICAL {X.shape[1]} feats (native unit-block) | "
          f"fired={int(evt.sum())}, censored={int((evt==0).sum())} | "
          f"seeds={N_SEEDS}")
    print(f"unit_* feature columns present: {len([c for c in X.columns if '__unit' in c])}")

    r = best_cindex(df, X, y, dur, evt)
    print(f"\nCANONICAL: {r['n_feats']} feats, bestK={r['best_k']}")
    print(f"   Cox     C-index {r['cox'][0]:.3f} [{r['cox'][1]:.3f}, {r['cox'][2]:.3f}]")
    print(f"   XGB-AFT C-index {r['xgb'][0]:.3f} [{r['xgb'][1]:.3f}, {r['xgb'][2]:.3f}]")
    print("   K-curve (CV C-index): "
          + "  ".join(f"{k}:{v:.3f}" for k, v in r["k_curve"].items()))
    print(f"   Cox params: {r['cox_params']}")
    print(f"   selected ({r['best_k']}): {r['top']}")

    out = os.path.join(project_root, "analysis", "survival_canonical_validate.pkl")
    with open(out, "wb") as f:
        pickle.dump(r, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
