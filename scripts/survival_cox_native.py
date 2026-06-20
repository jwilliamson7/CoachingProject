#!/usr/bin/env python
"""
Fully Cox-native firing-survival pipeline (JQAS interpretability variant).

Selection, prediction, and the hazard-ratio interpretation are all the SAME
model class (Cox), so there is no cross-model importance disagreement to
adjudicate (XGB-SHAP vs Cox coefficients). This reuses survival_methods entirely;
the only addition over survival_definitive is the importance_fn=cox_importance
selector and a correct Feature-N -> real-name resolver.

  - Stability selection with Cox ridge |z-coef| ranking (Meinshausen-Buhlmann).
  - Cox Harrell C on the stable set (leakage-free, N_SEEDS).
  - Cause-specific firing Cox with CLUSTER-ROBUST SEs -> named hazard ratios.
  - Grambsch-Therneau PH test.

Usage:
    python scripts/survival_cox_native.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from model.pipeline import load_modeling_data
from model.config import MODEL_CONFIG
from scripts.survival_analysis import global_max_season, build_survival_targets, tci
from scripts.survival_models import cox_builder, N_SEEDS
from scripts.data.matrix_factorization_imputation import SVDImputer
from data_constants import get_all_feature_names, HIRING_TEAM_FEATURES
from scripts.survival_methods import (
    build_competing_targets, firing_event, cox_hazard_ratios, ph_test,
    survival_eval, stability_selection, cox_importance, FIRED, VOLUNTARY,
)

warnings.filterwarnings("ignore")

STAB_K = 15
STAB_BOOTS = 100      # bump to 200 for camera-ready
STAB_THR = 0.5
CANON_PKL = os.path.join(project_root, "analysis", "survival_canonical_validate.pkl")

# Feature 1..51 are positional placeholders for the 8 core + 33 unit + 10 hiring
# columns; everything else (cf_/org_/hire_/tq_/rf_) already carries a real name.
_PLACEHOLDER_NAMES = get_all_feature_names() + HIRING_TEAM_FEATURES


def nm(c):
    if isinstance(c, str) and c.startswith("Feature "):
        i = int(c.split()[1]) - 1
        if 0 <= i < len(_PLACEHOLDER_NAMES):
            return _PLACEHOLDER_NAMES[i]
    return c


def main():
    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, cause = dur.loc[keep], cause.loc[keep]
    durb, _ = build_survival_targets(df, boundary)
    durb, evt = durb.loc[keep], pd.Series(firing_event(cause), index=keep)
    cols = list(X.columns)
    n_fire = int((cause == FIRED).sum())
    n_vol = int((cause == VOLUNTARY).sum())
    n_cens = len(cause) - n_fire - n_vol
    print(f"{len(df)} stints | {X.shape[1]} feats | fired={n_fire} "
          f"voluntary={n_vol} active-censored={n_cens} | seeds={N_SEEDS}\n")

    canon = pickle.load(open(CANON_PKL, "rb"))
    cox_p = canon.get("cox_params", {"penalizer": 0.5, "l1_ratio": 0.25})
    print(f"Cox params {cox_p}\n")

    # ---- Cox-native stability selection ----
    print(f"Cox-native stability selection: {STAB_BOOTS} subsamples, top-{STAB_K}...")
    freq = stability_selection(df, X, dur, evt, K=STAB_K, n_boot=STAB_BOOTS,
                               subsample=0.5, seed=0, importance_fn=cox_importance)
    stable = list(freq[freq >= STAB_THR].index)
    stable_idx = [cols.index(c) for c in stable]
    print(f"stable set ({len(stable)} feats, freq>= {STAB_THR}):")
    for c in stable:
        print(f"  {nm(c):<28} {c:<14} {freq[c]:.2f}")
    print(f"  freq>=0.6: {int((freq>=0.6).sum())} | >=0.7: {int((freq>=0.7).sum())}")
    print("\n  selection frequencies (top 18):")
    for c, v in freq.head(18).items():
        print(f"    {nm(c):<28} {v:.2f}")

    # ---- Cox Harrell C on the stable set (leakage-free, N_SEEDS) ----
    ev = survival_eval(cox_builder, cox_p, df, X, y, durb, evt, stable_idx, N_SEEDS)
    h = tci(ev["harrell"])
    u = tci(ev["uno"])
    print(f"\nCox stable-set discrimination: Harrell C {h[0]:.3f} [{h[1]:.3f},{h[2]:.3f}]"
          f"  Uno C {u[0]:.3f} [{u[1]:.3f},{u[2]:.3f}]")

    # ---- named hazard ratios (cluster-robust) + PH test ----
    Ximp_full = SVDImputer().fit(X.values).transform(X.values)
    Xsel = pd.DataFrame(Ximp_full[:, stable_idx], columns=stable, index=X.index)
    hr, _ = cox_hazard_ratios(Xsel, dur, cause, df["Coach Name"].values)
    pht = ph_test(Xsel, dur, cause)
    hr_named = hr.copy()
    hr_named.index = [nm(c) for c in hr.index]
    print("\nHazard ratios (cause-specific firing, cluster-robust SE on coach):")
    print(hr_named.round(3).to_string())
    print(f"\nPH test: all p>0.05? {bool((pht['p'] > 0.05).all())} "
          f"(min p={pht['p'].min():.3f})")

    out = os.path.join(project_root, "analysis", "survival_cox_native.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "stab_freq": freq, "stable": stable,
            "stable_named": [nm(c) for c in stable],
            "harrell": h, "uno": u, "harrell_raw": np.array(ev["harrell"]),
            "hazard_ratios": hr_named, "ph_test": pht,
            "cox_params": cox_p, "n_seeds": N_SEEDS,
        }, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
