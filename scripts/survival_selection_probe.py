#!/usr/bin/env python
"""
Selection probe for the internal-hire firing premium (JSE reframe).

The headline cause-specific Cox model (survival_definitive.py) reports an
internal-hire hazard ratio of 1.52. An economic reviewer will ask whether that
premium reflects the internal channel itself or the distressed franchises that
tend to promote from within (selection on the hiring situation). This script
gives the identification posture its observable-component test: refit the
headline specification with observable franchise-distress / inherited-quality
covariates added, and report whether the internal-hire HR survives.

This is NOT a robustness check (the estimate's reliability is settled by the
NB-resampled bake-off and stability selection). It is an identification probe:
it absorbs the OBSERVABLE part of the "which teams promote internally" story so
the residual premium can be read as conditional on it, leaving only selection on
unobservables to concede in prose.

Everything reuses the frozen pipeline: same data prep, same SVD imputation, same
cluster-robust `cox_hazard_ratios` as the headline. Model 1 reproduces the
published table exactly; Model 2 is Model 1 + the five distress proxies. No
re-selection, no new split/impute machinery.

    Model 1 (frozen headline): cf_internal_hire, age, num_yr_nfl_pos, tq_dsrs, Y/P__unit
    Model 2 (+ distress proxies): + org_unique_hc_10yr, org_prev_coach_tenure,
                                    tq_osrs, hire_roster_av, tq_srs_traj

Usage:
    python scripts/survival_selection_probe.py
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
from scripts.survival_analysis import global_max_season
from scripts.survival_methods import (
    build_competing_targets, drop_redundant_features, cox_hazard_ratios, ph_test,
    FIRED, VOLUNTARY,
)
from scripts.data.matrix_factorization_imputation import SVDImputer
from data_constants import get_all_feature_names, HIRING_TEAM_FEATURES

warnings.filterwarnings("ignore")

_PLACEHOLDER_NAMES = get_all_feature_names() + HIRING_TEAM_FEATURES

# Observable franchise-distress / inherited-quality proxies for the
# "which teams promote internally" selection story. All measured strictly
# pre-hire; all survive drop_redundant_features; tq_dsrs is already in the
# headline model (defensive side), so we add the offensive side (tq_osrs) and
# the fitted 3-season SRS slope (tq_srs_traj, least-squares annual rate).
PROXIES = ["org_unique_hc_10yr",   # HC churn over the prior decade (instability)
           "org_prev_coach_tenure",  # previous coach's tenure (near-orthogonal to internal: r=0.10)
           "tq_osrs",               # inherited offensive quality (def already in model)
           "hire_roster_av",        # inherited roster talent
           "tq_srs_traj"]           # fitted 3-season SRS slope (franchise trending down)


def nm(c):
    """Resolve a 'Feature N' placeholder to its real variable name."""
    if isinstance(c, str) and c.startswith("Feature "):
        i = int(c.split()[1]) - 1
        if 0 <= i < len(_PLACEHOLDER_NAMES):
            return _PLACEHOLDER_NAMES[i]
    return c


def vif_table(Xdf):
    """Manual VIF (1/(1-R^2) of each column on the others), standardized design."""
    cols = list(Xdf.columns)
    X = Xdf.values.astype(float)
    X = (X - X.mean(0)) / X.std(0)
    out = {}
    for j, name in enumerate(cols):
        yj = X[:, j]
        Z = np.delete(X, j, axis=1)
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z1, yj, rcond=None)
        r2 = 1 - ((yj - Z1 @ beta) ** 2).sum() / ((yj - yj.mean()) ** 2).sum()
        out[nm(name)] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(out)


def main():
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X = df.loc[keep], X.loc[keep]
    dur, cause = dur.loc[keep], cause.loc[keep]
    X = drop_redundant_features(X)
    cols = list(X.columns)

    # SVD imputation on the full sample, exactly as the headline HR path does.
    Ximp = SVDImputer().fit(X.values).transform(X.values)
    Xf = pd.DataFrame(Ximp, columns=cols, index=X.index)
    coach_ids = df["Coach Name"].values

    n_fire = int((cause == FIRED).sum())
    n_vol = int((cause == VOLUNTARY).sum())
    n_cens = int(len(cause) - n_fire - n_vol)
    print(f"{len(df)} stints | fired={n_fire} voluntary={n_vol} active-censored={n_cens}")

    # frozen headline stable set (from the camera-ready definitive run)
    defn = pickle.load(open(os.path.join(project_root, "analysis",
                                          "survival_definitive.pkl"), "rb"))
    stable = list(defn["stable"])
    missing = [c for c in stable + PROXIES if c not in Xf.columns]
    if missing:
        raise SystemExit(f"columns absent after prune/impute: {missing}")
    print("Model 1 (headline):", [nm(c) for c in stable])
    print("Model 2 adds      :", [nm(c) for c in PROXIES])
    print(f"Model 2: {len(stable) + len(PROXIES)} covariates, {n_fire} events "
          f"(EPV ~ {n_fire / (len(stable) + len(PROXIES)):.0f})\n")

    # ---- Model 1: reproduce the frozen headline ----
    hr1, _ = cox_hazard_ratios(Xf[stable], dur, cause, coach_ids, penalizer=0.0)
    hr1.index = [nm(c) for c in hr1.index]
    print("=== Model 1: frozen headline (reproduction check) ===")
    print(hr1.round(3).to_string(), "\n")

    # ---- Model 2: headline + distress proxies ----
    aug = stable + PROXIES
    hr2, _ = cox_hazard_ratios(Xf[aug], dur, cause, coach_ids, penalizer=0.0)
    pht2 = ph_test(Xf[aug], dur, cause, penalizer=0.0)
    hr2.index = [nm(c) for c in hr2.index]
    pht2.index = [nm(c) for c in pht2.index]
    print("=== Model 2: + observable distress proxies (cluster-robust) ===")
    print(hr2.round(3).to_string())
    print(f"\nPH test (Grambsch-Therneau) min p = {pht2['p'].min():.3f} "
          f"(all > 0.05? {bool((pht2['p'] > 0.05).all())})")

    # ---- VIF on the augmented design ----
    vif = vif_table(Xf[aug])
    print("\nVIF (augmented design):")
    print(vif.round(2).to_string())

    # ---- headline comparison ----
    ih1 = hr1.loc["cf_internal_hire"]
    ih2 = hr2.loc["cf_internal_hire"]
    print("\n=== internal-hire HR: headline vs selection-adjusted ===")
    print(f"  Model 1 (headline)        : HR {ih1['HR']:.3f} "
          f"[{ih1['CI_low']:.3f}, {ih1['CI_high']:.3f}]  p={ih1['p']:.3f}")
    print(f"  Model 2 (+distress proxies): HR {ih2['HR']:.3f} "
          f"[{ih2['CI_low']:.3f}, {ih2['CI_high']:.3f}]  p={ih2['p']:.3f}")
    pct = 100 * (ih2["HR"] - ih1["HR"]) / ih1["HR"]
    print(f"  change in HR: {pct:+.1f}%   "
          f"({'persists' if ih2['CI_low'] > 1 else 'CI now spans 1 -- attenuated'})")

    out = os.path.join(project_root, "analysis", "survival_selection_probe.pkl")
    with open(out, "wb") as f:
        pickle.dump({"hr_model1": hr1, "hr_model2": hr2, "ph_model2": pht2,
                     "vif": vif, "proxies": PROXIES, "stable": stable,
                     "n_fire": n_fire, "n_vol": n_vol, "n_cens": n_cens}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
