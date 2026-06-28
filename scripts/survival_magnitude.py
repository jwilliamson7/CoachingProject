#!/usr/bin/env python
"""
Economic-magnitude read-off for the internal-hire firing premium (JSE).

The headline cause-specific Cox (survival_definitive.py) reports the internal-hire
effect as a hazard ratio (1.52). An economics reader wants that translated into an
interpretable magnitude: how much sooner does an internal promotion get fired? This
script reads that off the SAME frozen five-feature fit -- no re-estimation, no new
split/impute machinery. It imports the frozen pipeline exactly as the selection
probe does, fits the published unpenalized HR model via `cox_hazard_ratios`, then
contrasts two covariate profiles that differ only in the internal-hire indicator,
holding the other four covariates at their sample means:

    predicted median firing-free tenure (seasons), internal vs external
    predicted P(fired within 4 seasons), internal vs external

Usage:
    python scripts/survival_magnitude.py
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
    build_competing_targets, drop_redundant_features, cox_hazard_ratios,
)
from scripts.data.matrix_factorization_imputation import SVDImputer

warnings.filterwarnings("ignore")

INTERNAL = "cf_internal_hire"


def _surv_at(cph, profile_df, t):
    """P(T > t) for each profile row, read off the fitted baseline survival."""
    sf = cph.predict_survival_function(profile_df, times=[t])
    return sf.iloc[0].values  # one row (time t), one column per profile


def main():
    # ---- frozen headline path: same data prep / impute as the HR model ----
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X = df.loc[keep], X.loc[keep]
    dur, cause = dur.loc[keep], cause.loc[keep]
    X = drop_redundant_features(X)
    cols = list(X.columns)

    Ximp = SVDImputer().fit(X.values).transform(X.values)
    Xf = pd.DataFrame(Ximp, columns=cols, index=X.index)
    coach_ids = df["Coach Name"].values

    defn = pickle.load(open(os.path.join(project_root, "analysis",
                                         "survival_definitive.pkl"), "rb"))
    stable = list(defn["stable"])
    if INTERNAL not in stable:
        raise SystemExit(f"{INTERNAL} not in stable set {stable}")

    hr, cph = cox_hazard_ratios(Xf[stable], dur, cause, coach_ids, penalizer=0.0)

    # ---- two profiles: differ only in the internal-hire indicator ----
    base = Xf[stable].mean()
    prof = pd.DataFrame([base, base], index=["external", "internal"])
    prof[INTERNAL] = [0.0, 1.0]

    med = cph.predict_median(prof)                 # median firing-free tenure
    s4 = _surv_at(cph, prof, 4.0)                   # P(T > 4)
    p4 = 1.0 - s4                                   # P(fired within 4 seasons)

    # ---- transparency: the per-season baseline is NOT constant ----
    sf = cph.predict_survival_function(prof, times=[1, 2, 3, 4, 5, 6, 7, 8])
    print("predicted firing-free survival S(t)=P(T>t) by season:")
    print("  season:        " + "  ".join(f"{int(t):>5d}" for t in sf.index))
    for who in ("external", "internal"):
        s = sf[who].values
        print(f"  S(t) {who[:3]}:     " + "  ".join(f"{v:5.3f}" for v in s))
        cond = [s[0]] + [s[i] / s[i - 1] for i in range(1, len(s))]
        print(f"  per-season {who[:3]}:" + "  ".join(f"{v:5.3f}" for v in cond))
    print()

    internal_share = float((Xf[INTERNAL] > 0.5).mean())
    hr_ih = hr.loc[INTERNAL]

    print(f"internal-hire HR        : {hr_ih['HR']:.3f} "
          f"[{hr_ih['CI_low']:.3f}, {hr_ih['CI_high']:.3f}]  p={hr_ih['p']:.3f}")
    print(f"internal-hire share     : {internal_share*100:.1f}% of stints\n")
    print("profile (other 4 covariates at sample means):")
    print(f"  median firing-free tenure  external={med['external']:.2f}  "
          f"internal={med['internal']:.2f}  seasons")
    print(f"  P(fired within 4 seasons)  external={p4[0]*100:.1f}%  "
          f"internal={p4[1]*100:.1f}%")
    print(f"  tenure gap                 {med['external'] - med['internal']:.2f} "
          f"fewer seasons for the internal promotion")

    out = os.path.join(project_root, "analysis", "survival_magnitude.pkl")
    with open(out, "wb") as f:
        pickle.dump({"median_external": float(med["external"]),
                     "median_internal": float(med["internal"]),
                     "p4_external": float(p4[0]), "p4_internal": float(p4[1]),
                     "internal_share": internal_share,
                     "hr": float(hr_ih["HR"])}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
