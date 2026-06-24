#!/usr/bin/env python
"""
Validate every numerical claim in the survival paper against its source of truth.

This is a REGRESSION GUARD, not a recompute-and-paste tool. The numbers quoted in
latex/2026-Williamson-NFL-Coach-Dismissal-Survival.tex are transcribed into the
PAPER block below (one place to update when the prose changes); the script pulls
the corresponding truth from the cached analysis artifacts and, for the few
quantities not stored with their confidence intervals (the cumulative incidence
function, the Kaplan-Meier median CI, the population and feature-family counts),
recomputes them live from the same pipeline functions the paper's figures use.

Each check compares the paper's value to the truth within half a unit of the last
quoted decimal, so it catches transcription drift and stale numbers while
tolerating ordinary rounding. Run after any pipeline re-run:

    python scripts/validate_paper_numbers.py

Exit code is the number of failing checks (0 = all clear).

Cached sources:
  analysis/survival_definitive.pkl       population, KM, era log-rank, stability,
                                         bake-off, Brier, hazard ratios, PH test
  analysis/survival_null_baseline.pkl    single-block / null ablations (Table 4)
  analysis/threshold_winrate.pkl         outcome-validity win rate (Appendix D)
Live recompute (model.pipeline + scripts.survival_methods):
  distinct-coach and feature-family counts, KM median CI, competing-risks CIF.
"""

import os
import sys
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

# --------------------------------------------------------------------------- #
# PAPER block: every number quoted in the .tex, at the precision quoted.
# Update this when the prose changes; the script checks it against the truth.
# --------------------------------------------------------------------------- #
PAPER = {
    # --- Population and event coding (Abstract, Sec 3.1, Table 1) ---
    "n_stints": 371,
    "n_distinct_coaches": 264,
    "n_multi_stint_coaches": 86,
    "n_fire": 312, "share_fire": 84.1,
    "n_vol": 27, "share_vol": 7.3,
    "n_cens": 32, "share_cens": 8.6,

    # --- Firing-time distribution (Sec 4.1) ---
    "km_median": 4, "km_median_ci": [3, 4],
    "era_logrank_p": 0.56,

    # --- Competing-risks incidence (Sec 4.2) ---  [point, lo, hi] by season
    "cif_fire_s2": [0.29, 0.25, 0.34], "cif_vol_s2": [0.01, 0.00, 0.02],
    "cif_fire_s4": [0.62, 0.57, 0.67], "cif_vol_s4": [0.04, 0.02, 0.06],
    "cif_fire_s8": [0.82, 0.77, 0.86], "cif_vol_s8": [0.06, 0.04, 0.09],

    # --- Feature pool and families (Sec 3.3, Table 2, Appendix B) ---
    "n_candidate_features": 74,
    "n_features_dropped": 20, "n_unit_dropped": 7, "n_nonunit_dropped": 13,
    "n_unit_total": 33, "n_unit_candidate": 26,
    "families": {  # name: count in candidate set
        "Core experience": 8, "Career structure": 16,
        "Prior unit performance": 26, "Recent form": 5,
        "Hiring-team results": 3, "Inherited team quality": 4,
        "Inherited roster": 10, "Org instability": 2,
    },

    # --- Stability selection (Sec 4.3, Table 3) ---  feature: {q: freq}
    "stability": {
        "cf_internal_hire": {10: 0.82, 15: 0.89, 20: 0.92},
        "age":              {10: 0.53, 15: 0.64, 20: 0.74},
        "Y/P__unit":        {10: 0.44, 15: 0.59, 20: 0.71},
        "num_yr_nfl_pos":   {10: 0.48, 15: 0.60, 20: 0.68},
        "tq_dsrs":          {10: 0.44, 15: 0.57, 20: 0.67},
    },
    "stable_vs_topk_delta": 0.002, "stable_vs_topk_p": 0.53,

    # --- Predictive performance (Sec 4.4, Table 5) ---  Harrell+Uno [mean, lo, hi]
    "bakeoff": {
        "Cox":            {"h": [0.602, 0.560, 0.645], "u": [0.631, 0.511, 0.751]},
        "Weibull AFT":    {"h": [0.615, 0.573, 0.657], "u": [0.643, 0.526, 0.760]},
        "LogNormal AFT":  {"h": [0.615, 0.576, 0.654], "u": [0.641, 0.525, 0.758]},
        "LogLogistic AFT":{"h": [0.615, 0.575, 0.654], "u": [0.640, 0.524, 0.757]},
        "XGB-AFT":        {"h": [0.608, 0.567, 0.648], "u": [0.634, 0.515, 0.753]},
    },
    "bakeoff_max_gap_from_cox": 0.013,  # "no more than about 0.013 in concordance"

    # --- Ablation (Sec 4.4, Table 4) ---  [mean, lo, hi]
    "ablation": {
        "KM null (no covariates)":     [0.500, 0.500, 0.500],
        "Coach unit performance (33)": [0.526, 0.480, 0.573],
        "Inherited situation (30)":    [0.494, 0.448, 0.539],
        "Coach age only":              [0.560, 0.515, 0.604],
        "Internal hire only":          [0.564, 0.532, 0.596],
        "FULL CV-selected (5)":        [0.602, 0.560, 0.645],
    },

    # --- Calibration (Sec 4.4) ---
    "ibs": [0.176, 0.160, 0.191], "brier_skill": 0.30,

    # --- Hazard ratios (Sec 4.5, Table 6) ---  HR, CI_low, CI_high, p
    "hazard": {
        "cf_internal_hire": [1.524, 1.150, 2.021, 0.003],
        "age":              [1.021, 1.003, 1.039, 0.021],
        "Y/P__unit":        [0.739, 0.614, 0.891, 0.001],
        "num_yr_nfl_pos":   [1.021, 0.989, 1.054, 0.198],
        "tq_dsrs":          [1.024, 0.992, 1.057, 0.147],
    },
    "ph_min_p": 0.20,

    # --- Outcome validity (Appendix D) ---
    "winrate": {0: 0.31, 1: 0.41, 2: 0.54},

    # --- Hyperparameters (Appendix E, Table 10) ---
    "cox_params": {"penalizer": 0.5, "l1_ratio": 0.25},
    "xgb_params": {"num_boost_round": 50, "eta": 0.02, "max_depth": 4,
                   "subsample": 1.0, "colsample_bytree": 1.0,
                   "min_child_weight": 3, "reg_lambda": 1.0, "scale": 1.5},
}

# Name aliases: the Table-3 / Table-6 display names map to internal columns.
HR_LABEL = {"cf_internal_hire": "Internal hire", "age": "Age at hire",
            "Y/P__unit": "Prior-unit yards/play", "num_yr_nfl_pos": "Years NFL pos coach",
            "tq_dsrs": "Inherited def SRS"}

# --------------------------------------------------------------------------- #
RESULTS = []


def _tol(dp):
    return 0.5 * 10 ** (-dp) + 1e-9


def check(name, paper, actual, dp=3):
    """Pass if |paper - actual| within half the last quoted decimal."""
    ok = abs(float(paper) - float(actual)) <= _tol(dp)
    RESULTS.append((ok, name, f"paper={paper}  actual={float(actual):.{dp+1}f}"))


def check_eq(name, paper, actual):
    ok = paper == actual
    RESULTS.append((ok, name, f"paper={paper}  actual={actual}"))


def check_triplet(name, paper, actual, dp=3):
    for tag, p, a in zip(("pt", "lo", "hi"), paper, actual):
        check(f"{name} [{tag}]", p, a, dp)


def check_lt(name, actual, bound):
    ok = float(actual) < bound
    RESULTS.append((ok, name, f"actual={float(actual):.2e} < {bound}"))


# --------------------------------------------------------------------------- #
def main():
    A = os.path.join(project_root, "analysis")
    D = pickle.load(open(os.path.join(A, "survival_definitive.pkl"), "rb"))
    NB = pickle.load(open(os.path.join(A, "survival_null_baseline.pkl"), "rb"))
    WR = pickle.load(open(os.path.join(A, "threshold_winrate.pkl"), "rb"))

    # ---- population / event coding ----
    check_eq("n_stints", PAPER["n_stints"], D["n_stints"])
    check_eq("n_fire", PAPER["n_fire"], D["n_fire"])
    check_eq("n_vol", PAPER["n_vol"], D["n_vol"])
    check_eq("n_cens", PAPER["n_cens"], D["n_cens"])
    N = D["n_stints"]
    check("share_fire", PAPER["share_fire"], 100 * D["n_fire"] / N, 1)
    check("share_vol", PAPER["share_vol"], 100 * D["n_vol"] / N, 1)
    check("share_cens", PAPER["share_cens"], 100 * D["n_cens"] / N, 1)

    # ---- firing-time distribution ----
    check_eq("km_median", PAPER["km_median"], int(D["km_median"]))
    check("era_logrank_p", PAPER["era_logrank_p"], D["era_logrank_p"], 2)

    # ---- stability (Table 3) ----
    named_q = {q: _stab_named(D, q) for q in (10, 15, 20)}
    for feat, qmap in PAPER["stability"].items():
        for q, pv in qmap.items():
            check(f"stability {feat} q={q}", pv, named_q[q].get(feat), 2)
    check("stable_vs_topk delta", PAPER["stable_vs_topk_delta"],
          D["nb_stable_vs_topk"]["delta"], 3)
    check("stable_vs_topk p", PAPER["stable_vs_topk_p"],
          D["nb_stable_vs_topk"]["p"], 2)

    # ---- bake-off (Table 5) ----
    key = {"Cox": "Cox", "Weibull AFT": "Weibull AFT", "LogNormal AFT": "LogNormal AFT",
           "LogLogistic AFT": "LogLogistic AFT", "XGB-AFT": "XGB-AFT"}
    cox_h = D["bakeoff"]["Cox"]["harrell"][0]
    max_gap = 0.0
    for m, vals in PAPER["bakeoff"].items():
        bh = D["bakeoff"][key[m]]["harrell"]
        bu = D["bakeoff"][key[m]]["uno"]
        check_triplet(f"bakeoff {m} Harrell", vals["h"], bh, 3)
        check_triplet(f"bakeoff {m} Uno", vals["u"], bu, 3)
        max_gap = max(max_gap, abs(bh[0] - cox_h))
    check("bakeoff max gap from Cox", PAPER["bakeoff_max_gap_from_cox"], max_gap, 3)

    # ---- ablation (Table 4) ----
    for label, vals in PAPER["ablation"].items():
        s = NB["summary"][label]
        check_triplet(f"ablation {label}", vals, [s["mean"], s["lo"], s["hi"]], 3)

    # ---- calibration ----
    check_triplet("integrated Brier", PAPER["ibs"], D["ibs_ci"], 3)
    check("Brier skill score", PAPER["brier_skill"],
          (0.25 - D["ibs_ci"][0]) / 0.25, 2)

    # ---- hazard ratios (Table 6) ----
    hr = D["hazard_ratios"]
    for feat, (HR, lo, hi, p) in PAPER["hazard"].items():
        r = hr.loc[feat]
        lbl = HR_LABEL[feat]
        check(f"HR {lbl} value", HR, r["HR"], 3)
        check(f"HR {lbl} CI_low", lo, r["CI_low"], 3)
        check(f"HR {lbl} CI_high", hi, r["CI_high"], 3)
        check(f"HR {lbl} p", p, r["p"], 3)
    check("PH-test min p", PAPER["ph_min_p"], D["ph_test"]["p"].min(), 2)

    # ---- hyperparameters ----
    for k, v in PAPER["cox_params"].items():
        check(f"cox_param {k}", v, D["cox_params"][k], 4)
    for k, v in PAPER["xgb_params"].items():
        check(f"xgb_param {k}", v, D["xgb_params"][k], 4)

    # ---- outcome validity (Appendix D) ----
    for cls, mean in PAPER["winrate"].items():
        check(f"winrate class {cls}", mean, WR["primary"][cls]["mean"], 2)
    check_lt("winrate Kruskal-Wallis p", WR["kruskal"]["p"], 0.001)
    check_lt("winrate MW 0v1 p", WR["pairwise_p"]["0v1"], 0.001)
    check_lt("winrate MW 1v2 p", WR["pairwise_p"]["1v2"], 0.001)

    # ---- LIVE recompute: population, families, KM CI, CIF ----
    _live_checks()

    # ---- report ----
    print(f"\n{'='*72}\n  PAPER NUMBER VALIDATION\n{'='*72}")
    fails = [r for r in RESULTS if not r[0]]
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"  FAIL  {name:42} {detail}")
    print(f"\n  {len(RESULTS)-len(fails)}/{len(RESULTS)} checks passed.")
    if fails:
        print(f"  {len(fails)} FAILED (listed above).")
    else:
        print("  All clear.")
    return len(fails)


def _stab_named(D, q):
    """stab_freq_multi[q] keyed by placeholder; remap to display names via the
    same alignment the named series uses at q=10."""
    placeholder = D["stab_freq_multi"][q]
    named10 = D["stab_freq_named"]              # q=10 named order
    base10 = D["stab_freq_multi"][10]
    remap = dict(zip(base10.index, named10.index))
    return placeholder.rename(index=lambda c: remap.get(c, c))


def _live_checks():
    from model.pipeline import load_modeling_data
    from scripts.survival_analysis import global_max_season
    from scripts.survival_methods import (
        build_competing_targets, aalen_johansen_cif, firing_event,
        drop_redundant_features, DROP_REDUNDANT_FEATURES, FIRED, VOLUNTARY)
    from scripts.survival_definitive import nm, _PLACEHOLDER_NAMES
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times
    from collections import Counter

    df, X, y = load_modeling_data(known_only=False)

    check_eq("distinct coaches", PAPER["n_distinct_coaches"], df["Coach Name"].nunique())
    vc = df["Coach Name"].value_counts()
    check_eq("multi-stint coaches", PAPER["n_multi_stint_coaches"], int((vc > 1).sum()))

    # feature pool + redundancy
    Xd = drop_redundant_features(X.copy())
    check_eq("candidate features", PAPER["n_candidate_features"], Xd.shape[1])
    check_eq("features dropped", PAPER["n_features_dropped"], X.shape[1] - Xd.shape[1])
    du = [c for c in DROP_REDUNDANT_FEATURES if str(c).endswith("__unit")]
    check_eq("unit dropped", PAPER["n_unit_dropped"], len(du))
    check_eq("non-unit dropped", PAPER["n_nonunit_dropped"],
             len(DROP_REDUNDANT_FEATURES) - len(du))
    check_eq("unit candidate (33-7)", PAPER["n_unit_candidate"],
             PAPER["n_unit_total"] - len(du))

    # family decomposition
    core = _PLACEHOLDER_NAMES[:8]
    hiring_ctx = _PLACEHOLDER_NAMES[41:51]

    def fam(n):
        n = str(n)
        if n in core: return "Core experience"
        if n.endswith("__unit"): return "Prior unit performance"
        if n in hiring_ctx: return "Hiring-team results"
        if n.startswith("cf_"): return "Career structure"
        if n.startswith("rf_"): return "Recent form"
        if n.startswith("tq_"): return "Inherited team quality"
        if n.startswith("hire_"): return "Inherited roster"
        if n.startswith("org_"): return "Org instability"
        return "OTHER"
    counts = Counter(fam(nm(c)) for c in Xd.columns)
    for k, v in PAPER["families"].items():
        check_eq(f"family {k}", v, counts.get(k, 0))

    # KM median CI
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    evt = firing_event(cause)
    kmf = KaplanMeierFitter().fit(dur.values, evt)
    ci = median_survival_times(kmf.confidence_interval_).values.ravel()
    check_eq("KM median CI", PAPER["km_median_ci"], [int(ci[0]), int(ci[1])])

    # competing-risks CIF at the end of each season (read at s+0.5; see
    # generate_survival_figures._step_at_seasons for the jitter rationale)
    def cif_band(ev, t):
        aj = aalen_johansen_cif(dur, cause, ev)
        c, q = aj.cumulative_density_, aj.confidence_interval_

        def at(frame, col):
            m = frame.index <= t + 0.5
            return float(frame[m].iloc[-1, col]) if m.any() else 0.0
        return [at(c, 0), at(q, 0), at(q, 1)]
    for s in (2, 4, 8):
        check_triplet(f"CIF firing s{s}", PAPER[f"cif_fire_s{s}"], cif_band(FIRED, s), 2)
        check_triplet(f"CIF voluntary s{s}", PAPER[f"cif_vol_s{s}"], cif_band(VOLUNTARY, s), 2)


if __name__ == "__main__":
    sys.exit(main())
