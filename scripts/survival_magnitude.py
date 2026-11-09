#!/usr/bin/env python
"""
Economic-magnitude read-off for the internal-hire firing premium (JSE).

The headline cause-specific Cox (survival_definitive.py) reports the internal-hire
effect as a hazard ratio. An economics reader wants that translated into an
interpretable magnitude: how much sooner does an internal promotion get fired? This
script reads that off the SAME frozen headline fit -- no re-estimation, no new
split/impute machinery. It imports the frozen pipeline exactly as the selection
probe does, fits the published unpenalized HR model via `cox_hazard_ratios`, and
contrasts internal vs external hires by MARGINAL STANDARDIZATION: every stint is
evaluated once with the internal-hire indicator set to 0 and once set to 1, keeping
its own other covariates, and the predictions are averaged. (A "mean profile" would
plug in impossible values such as 0.3 of a structural indicator.)

    predicted P(fired within 4 seasons), internal vs external -- a competing-risks
        cumulative incidence (a voluntary exit removes a stint from firing risk)
    1 - S(4), the firing-only complement, for reference
    median firing-free tenure (seasons) from the standardized survival curves

Durations are whole seasons, so the curves use the discrete product-limit form,
S(t|x) = prod_{t_j <= t} (1 - min(1, dH0_j exp(x b))), not exp(-H): with large
per-season hazards exp(-H) overstates survival. dH0_j are lifelines' Breslow
increments, which are estimated at the training covariate MEANS, so they are paired
with lifelines' centered partial hazard exp((x - xbar) b) (predict_partial_hazard).
The firing CIF is CIF_f(t|x) = sum_{s<=t} S(s-|x) h_f(s|x), with S the survival from
EITHER exit. The voluntary hazard h_v(s) comes from a covariate-free Nelson-Aalen
(d_v(s)/Y(s)): there are too few voluntary exits to model their covariates, and the
quantity of interest is the firing hazard. A self-check compares the model's sample-
mean S(4) and CIF_f(4) with the Kaplan-Meier and Aalen-Johansen estimates.

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
from lifelines import KaplanMeierFitter

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season
from scripts.survival_methods import (
    build_competing_targets, drop_redundant_features, cox_hazard_ratios, firing_event,
    aalen_johansen_cif, cif_at, with_structural_indicators, FIRED, VOLUNTARY,
    cox_discrete_hazards as firing_hazards,
)
from model.pipeline import make_imputer

warnings.filterwarnings("ignore")

INTERNAL = "cf_internal_hire"
HORIZON = 4                      # seasons: "fired within four seasons"
DEF_PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")
OUT_PKL = os.path.join(project_root, "analysis", "survival_magnitude.pkl")


def voluntary_hazard(dur, cause, seasons):
    """Covariate-free Nelson-Aalen increment of the voluntary-exit hazard,
    d_v(s) / Y(s), at each season s (Y = stints still at risk entering s)."""
    T = np.asarray(dur, float)
    vol = np.asarray(cause) == VOLUNTARY
    return np.array([(vol & (T == s)).sum() / max(int((T >= s).sum()), 1)
                     for s in seasons])


def firing_curves(h_f, h_v):
    """(firing-free survival S_f, firing CIF) per row, from discrete hazards.

    S_f(t) = prod_{s<=t} (1 - h_f(s)) is the product-limit, cause-specific survival.
    CIF_f(t) = sum_{s<=t} S(s-) h_f(s), with S the survival from EITHER exit,
    prod_{s<=t} (1 - h_f(s) - h_v(s)). h_v is capped so the two cannot exceed one.
    """
    h_v = np.minimum(h_v[None, :], 1.0 - h_f)
    s_any = np.cumprod(1.0 - h_f - h_v, axis=1)
    s_prev = np.hstack([np.ones((len(h_f), 1)), s_any[:, :-1]])
    return np.cumprod(1.0 - h_f, axis=1), np.cumsum(s_prev * h_f, axis=1)


def at(seasons, curve, t):
    """Value of a step curve on `seasons` through the end of season t (t >= the
    first season on the grid)."""
    return float(curve[np.searchsorted(seasons, t, side="right") - 1])


def median_season(seasons, surv):
    """Median firing-free tenure: the first season at which S(t) <= 0.5."""
    hit = np.flatnonzero(surv <= 0.5)
    return float(seasons[hit[0]]) if len(hit) else np.inf


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

    Ximp = make_imputer().fit(X.values).transform(X.values)
    Xf = pd.DataFrame(Ximp, columns=cols, index=X.index)
    coach_ids = df["Coach Name"].values

    defn = pickle.load(open(DEF_PKL, "rb"))
    stable = list(defn.get("model_features") or with_structural_indicators(defn["stable"], X))
    if INTERNAL not in stable:
        raise SystemExit(f"{INTERNAL} not in stable set {stable}")

    hr, cph = cox_hazard_ratios(Xf[stable], dur, cause, coach_ids, penalizer=0.0)

    # ---- marginal standardization: every stint as external, then as internal ----
    seasons, _ = firing_hazards(cph, Xf[stable])
    h_v = voluntary_hazard(dur, cause, seasons)
    std = {}
    for who, val in (("external", 0.0), ("internal", 1.0)):
        Xc = Xf[stable].copy()
        Xc[INTERNAL] = val
        s_f, cif = firing_curves(firing_hazards(cph, Xc)[1], h_v)
        std[who] = {"S": s_f.mean(axis=0), "CIF": cif.mean(axis=0)}

    p4 = {w: at(seasons, std[w]["CIF"], HORIZON) for w in std}       # P(fired by 4)
    p4_1ms = {w: 1.0 - at(seasons, std[w]["S"], HORIZON) for w in std}
    med = {w: median_season(seasons, std[w]["S"]) for w in std}

    # ---- self-check: the model at the observed covariates reproduces the
    # nonparametric curves (product-limit S vs Kaplan-Meier; CIF vs Aalen-Johansen) ----
    s_obs, cif_obs = firing_curves(firing_hazards(cph, Xf[stable])[1], h_v)
    km = KaplanMeierFitter().fit(dur.values, firing_event(cause))
    aj = aalen_johansen_cif(dur, cause, FIRED)
    s_mean, s_km = at(seasons, s_obs.mean(axis=0), HORIZON), float(km.predict(HORIZON))
    c_mean, c_aj = at(seasons, cif_obs.mean(axis=0), HORIZON), cif_at(aj.cumulative_density_, HORIZON)
    s_expH = float(cph.predict_survival_function(Xf[stable], times=[HORIZON]).values.mean())

    # ---- transparency: the per-season hazard is NOT constant ----
    show = seasons[(seasons >= 1) & (seasons <= 8)]
    cols_show = [np.flatnonzero(seasons == s)[0] for s in show]
    print("standardized firing-free survival S(t)=P(no firing by t) and firing CIF:")
    print("  season:          " + "  ".join(f"{int(t):>5d}" for t in show))
    for who in ("external", "internal"):
        s = std[who]["S"][cols_show]
        cond = [s[0]] + [s[i] / s[i - 1] for i in range(1, len(s))]
        print(f"  S(t) {who[:3]}:       " + "  ".join(f"{v:5.3f}" for v in s))
        print(f"  per-season {who[:3]}: " + "  ".join(f"{v:5.3f}" for v in cond))
        print(f"  CIF_f(t) {who[:3]}:   " + "  ".join(f"{v:5.3f}" for v in std[who]["CIF"][cols_show]))
    print()

    internal_share = float((Xf[INTERNAL] > 0.5).mean())
    hr_ih = hr.loc[INTERNAL]
    print(f"internal-hire HR        : {hr_ih['HR']:.3f} "
          f"[{hr_ih['CI_low']:.3f}, {hr_ih['CI_high']:.3f}]  p={hr_ih['p']:.3f}")
    print(f"internal-hire share     : {internal_share*100:.1f}% of stints\n")
    print(f"marginal standardization over all {len(Xf)} stints (each evaluated as an "
          f"external and as an internal hire, other covariates as observed):")
    rows = [(f"P(fired within {HORIZON} seasons), competing risks",
             f"{p4['external']*100:.1f}%", f"{p4['internal']*100:.1f}%"),
            (f"1 - S({HORIZON}) (firing only, for reference)",
             f"{p4_1ms['external']*100:.1f}%", f"{p4_1ms['internal']*100:.1f}%"),
            ("median firing-free tenure (seasons)",
             f"{med['external']:.0f}", f"{med['internal']:.0f}")]
    for lab, e, i in rows:
        print(f"  {lab:<46} external={e:>6}  internal={i:>6}")
    print(f"\nself-check at the observed covariates, season {HORIZON}:")
    print(f"  sample-mean S({HORIZON}) product-limit {s_mean:.3f} vs Kaplan-Meier {s_km:.3f}"
          f"   (exp(-H) form: {s_expH:.3f})")
    print(f"  sample-mean CIF_f({HORIZON}) {c_mean:.3f} vs Aalen-Johansen {c_aj:.3f}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"median_external": med["external"], "median_internal": med["internal"],
                     "p4_external": p4["external"], "p4_internal": p4["internal"],
                     "p4_1mS_external": p4_1ms["external"],
                     "p4_1mS_internal": p4_1ms["internal"],
                     "horizon": HORIZON, "seasons": seasons,
                     "std_curves": std, "voluntary_hazard": h_v,
                     "self_check": {"S_mean": s_mean, "S_km": s_km, "S_expH": s_expH,
                                    "CIF_mean": c_mean, "CIF_aj": c_aj},
                     "method": "marginal standardization; product-limit Cox (firing) + "
                               "Nelson-Aalen (voluntary) competing risks",
                     "internal_share": internal_share,
                     "hr": float(hr_ih["HR"])}, f)
    print(f"\nSaved {OUT_PKL}")


if __name__ == "__main__":
    main()
