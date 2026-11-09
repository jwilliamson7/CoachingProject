#!/usr/bin/env python
"""
Validate every numerical claim in the survival paper against its source of truth.

This is a REGRESSION GUARD, not a recompute-and-paste tool. The numbers quoted in
papers/jse/2026-NFL-Coach-Dismissal-JSE.tex are transcribed into the
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

from scripts.survival_methods import nm, cif_at

ANALYSIS_DIR = os.path.join(project_root, "analysis")   # the cached artifacts read below

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
        "XGB-AFT":        {"h": [0.621, 0.585, 0.657], "u": [0.649, 0.535, 0.764]},
    },
    "bakeoff_max_gap_from_cox": 0.019,  # "no more than about 0.02 in concordance" (XGB-AFT 0.0185)

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
    "xgb_params": {"num_boost_round": 50, "eta": 0.03, "max_depth": 4,
                   "subsample": 0.6, "colsample_bytree": 0.6,
                   "min_child_weight": 5, "reg_lambda": 1.0, "scale": 0.8},
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


def check_set(name, paper, actual):
    """Pass if the paper's feature set equals the fitted one; report each side's extras."""
    p, a = set(paper), set(actual)
    detail = (f"{len(p)} features" if p == a else
              f"paper-only={sorted(p - a)}  pkl-only={sorted(a - p)}")
    RESULTS.append((p == a, name, detail))


def missing(name, why):
    """Record a check that cannot run (absent row/feature/key) as a failure."""
    RESULTS.append((False, name, why))


# --------------------------------------------------------------------------- #
def main():
    A = ANALYSIS_DIR
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
    # era_logrank_p is a dict of tests (salary_cap, rooney_rule, three_era); the paper
    # value may be the same dict or a single legacy p, which has no counterpart
    era, pe = D["era_logrank_p"], PAPER["era_logrank_p"]
    if not isinstance(era, dict):
        check("era_logrank_p", pe, era, 2)
    elif isinstance(pe, dict):
        for k, v in pe.items():
            if k in era:
                check(f"era_logrank_p {k}", v, era[k], 2)
            else:
                missing(f"era_logrank_p {k}", f"no such test in pkl ({sorted(era)})")
    else:
        missing("era_logrank_p", f"paper quotes one p ({pe}); pkl has "
                + ", ".join(f"{k}={v:.3f}" for k, v in era.items()))

    # ---- stability (Table 3): the paper's set must be the fitted stable set ----
    stable_named = D.get("stable_named") or [nm(c) for c in D["stable"]]
    check_set("stability feature set = stable", PAPER["stability"], stable_named)
    named_q = {q: _stab_named(D, q) for q in D["stab_freq_multi"]}
    for feat, qmap in PAPER["stability"].items():
        for q, pv in qmap.items():
            v = named_q[q].get(feat) if q in named_q else None
            if v is None:
                missing(f"stability {feat} q={q}", "feature or q not in pkl")
            else:
                check(f"stability {feat} q={q}", pv, v, 2)
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
        if key[m] not in D["bakeoff"]:   # now in the nested comparison (own selection)
            check(f"bakeoff {m} present in survival_definitive", 1, 0, 0)
            continue
        bh = D["bakeoff"][key[m]]["harrell"]
        bu = D["bakeoff"][key[m]]["uno"]
        check_triplet(f"bakeoff {m} Harrell", vals["h"], bh, 3)
        check_triplet(f"bakeoff {m} Uno", vals["u"], bu, 3)
        max_gap = max(max_gap, abs(bh[0] - cox_h))
    check("bakeoff max gap from Cox", PAPER["bakeoff_max_gap_from_cox"], max_gap, 3)

    # ---- ablation (Table 4) ----
    for label, vals in PAPER["ablation"].items():
        s = NB["summary"].get(label)
        if s is None:
            missing(f"ablation {label}", f"row not in pkl ({list(NB['summary'])})")
            continue
        check_triplet(f"ablation {label}", vals, [s["mean"], s["lo"], s["hi"]], 3)

    # ---- calibration: skill relative to the covariate-free Kaplan-Meier null on the
    # same splits (survival_methods.km_null_brier), not to a 0.25 coin flip ----
    check_triplet("integrated Brier", PAPER["ibs"], D["ibs_ci"], 3)
    if "ibs_km_ci" in D:
        check("Brier skill vs KM null", PAPER["brier_skill"],
              1.0 - D["ibs_ci"][0] / D["ibs_km_ci"][0], 2)
    else:
        missing("Brier skill vs KM null", "pkl has no ibs_km_ci (rerun survival_definitive)")

    # ---- hazard ratios (Table 6): the paper's rows must be the fitted model ----
    hr = D["hazard_ratios"]
    model_named = D.get("model_features_named") or [nm(c) for c in D["model_features"]]
    check_set("HR feature set = model_features", PAPER["hazard"], model_named)
    for feat, (HR, lo, hi, p) in PAPER["hazard"].items():
        lbl = HR_LABEL.get(feat, feat)
        if feat not in hr.index:
            missing(f"HR {lbl}", "paper feature not in the fitted model")
            continue
        r = hr.loc[feat]
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
    """stab_freq_multi[q] (keyed by column) re-keyed by variable name. Each column is
    resolved directly with nm(), so this is right whatever primary q (D["stab_k"])
    the run used; the old remap aligned positions against stab_freq_named, which is
    the stab_k series, and was right only when stab_k happened to be 10."""
    return D["stab_freq_multi"][q].rename(index=nm)


def _live_checks():
    from model.pipeline import load_modeling_data
    from scripts.survival_analysis import global_max_season
    from scripts.survival_methods import (
        build_competing_targets, aalen_johansen_cif, firing_event,
        drop_redundant_features, _PLACEHOLDER_NAMES, FIRED, VOLUNTARY)
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times
    from collections import Counter

    # the analysis sample: every stint with a reconstructed tenure and outcome
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    df, X = df.loc[dur.index], X.loc[dur.index]

    check_eq("distinct coaches", PAPER["n_distinct_coaches"], df["Coach Name"].nunique())
    vc = df["Coach Name"].value_counts()
    check_eq("multi-stint coaches", PAPER["n_multi_stint_coaches"], int((vc > 1).sum()))

    # feature pool + redundancy, counted from the columns actually present and removed
    Xd = drop_redundant_features(X.copy())
    dropped = [nm(c) for c in X.columns if c not in Xd.columns]
    du = [c for c in dropped if c.endswith("__unit")]
    check_eq("candidate features", PAPER["n_candidate_features"], Xd.shape[1])
    check_eq("features dropped", PAPER["n_features_dropped"], len(dropped))
    check_eq("unit dropped", PAPER["n_unit_dropped"], len(du))
    check_eq("non-unit dropped", PAPER["n_nonunit_dropped"], len(dropped) - len(du))
    check_eq("unit total", PAPER["n_unit_total"],
             sum(nm(c).endswith("__unit") for c in X.columns))
    check_eq("unit candidate", PAPER["n_unit_candidate"],
             sum(nm(c).endswith("__unit") for c in Xd.columns))

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
    evt = firing_event(cause)
    kmf = KaplanMeierFitter().fit(dur.values, evt)
    ci = median_survival_times(kmf.confidence_interval_).values.ravel()
    check_eq("KM median CI", PAPER["km_median_ci"], [int(ci[0]), int(ci[1])])

    # competing-risks CIF at the end of each season (survival_methods.cif_at)
    def cif_band(ev, t):
        aj = aalen_johansen_cif(dur, cause, ev)
        c, q = aj.cumulative_density_, aj.confidence_interval_
        return [cif_at(c, t), cif_at(q, t, 0), cif_at(q, t, 1)]
    for s in (2, 4, 8):
        check_triplet(f"CIF firing s{s}", PAPER[f"cif_fire_s{s}"], cif_band(FIRED, s), 2)
        check_triplet(f"CIF voluntary s{s}", PAPER[f"cif_vol_s{s}"], cif_band(VOLUNTARY, s), 2)


if __name__ == "__main__":
    sys.exit(main())
