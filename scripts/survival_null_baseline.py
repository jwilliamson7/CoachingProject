#!/usr/bin/env python
"""
Null and naive baselines for the firing-survival C-index (JQAS).

Contextualizes the full model's ~0.60 firing C-index so a reviewer can see it is
real signal over a null, not a strong absolute number sold as one. Every row uses
the IDENTICAL leakage-free protocol (coach-level splits, per-fold SVD imputation,
firing-aware concordance) via scripts.survival_models.eval_model -- no new
split/impute/fit code (DRY).

Baselines, weakest to strongest:
  1. Kaplan-Meier null (no covariates): C-index = 0.5 by construction (everyone
     shares the marginal hazard, so every comparable pair ties). The floor.
  2. Single obvious covariates, one Cox model each (age / internal-hire /
     inherited team SRS / inherited defensive SRS / roster turnover): what one
     variable a naive analyst would reach for actually buys.
  3. Conceptual blocks:
       - INHERITED SITUATION (the team/org walked into, not the coach):
         original hiring context + org history + roster state + team SRS family.
       - COACH RESUME (who the coach is, not the team): core career + career-path
         features + recent-form percentiles.
       - COACH UNIT PERFORMANCE (the oriented prior on-field unit block).
  4. FULL CV-selected model (the headline top-K), recomputed here on the same
     seeds so the whole table is strictly comparable.

Usage:
    python scripts/survival_null_baseline.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from lifelines.utils import concordance_index

from model.pipeline import load_modeling_data, leakage_free_split
from model.config import MODEL_CONFIG
from scripts.survival_analysis import global_max_season, build_survival_targets, nb_tci
from scripts.survival_models import cox_builder, eval_model, N_SEEDS
from scripts.survival_methods import drop_redundant_features

warnings.filterwarnings("ignore")

# A naive analyst's default Cox: a light ridge, no tuning. The baselines are not
# meant to be optimized; the point is the floor. (The full model row uses its own
# CV-tuned penalizer, loaded from the canonical-validate artifact.)
BASELINE_COX = {"penalizer": 0.1, "l1_ratio": 0.0}
DEF_PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")


def km_null_cindex(df, X, y, dur, evt):
    """C-index of a covariate-free model on the same seeds: constant risk for
    everyone -> every comparable pair ties -> concordance = 0.5 each seed."""
    scores = []
    for seed in range(N_SEEDS):
        split = leakage_free_split(df, X, y, seed)
        Tte = dur.loc[split.test_index].values.astype(float)
        Ete = evt.loc[split.test_index].values.astype(int)
        scores.append(concordance_index(Tte, np.zeros(len(Tte)), Ete))
    return scores


def cox_cindex(df, X, y, dur, evt, idx, params=BASELINE_COX):
    """Firing C-index over N_SEEDS leakage-free splits for a fixed feature set."""
    return eval_model(cox_builder, params, df, X, y, dur, evt, list(idx))["c_index"]


def main():
    # known_only=False to match survival_definitive: the still-active coaches are
    # right-censored observations, not dropped (see survival_definitive.py).
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]
    cols = list(X.columns)
    pos = {c: cols.index(c) for c in cols}
    print(f"{len(df)} stints | {X.shape[1]} feats | fired={int((evt==1).sum())} "
          f"censored={int((evt==0).sum())} | seeds={N_SEEDS}\n")

    # ---- conceptual blocks by column position ----
    core = [pos[f"Feature {n}"] for n in range(1, 9)]            # career core
    unit = [pos[f"Feature {n}"] for n in range(9, 42)]           # oriented unit perf
    hiring_orig = [pos[f"Feature {n}"] for n in range(42, 52)]   # original hiring ctx
    cf = [p for c, p in pos.items() if c.startswith("cf_")]
    org = [p for c, p in pos.items() if c.startswith("org_")]
    hire = [p for c, p in pos.items() if c.startswith("hire_")]
    tq = [p for c, p in pos.items() if c.startswith("tq_")]
    rf = [p for c, p in pos.items() if c.startswith("rf_")]

    inherited = sorted(hiring_orig + org + hire + tq)   # the situation walked into
    resume = sorted(core + cf + rf)                      # who the coach is
    # unit = the coach's prior on-field performance

    # ---- full selected model: use the SAME stable set + params as the headline
    # (survival_definitive.pkl) so the ablation's full row equals the bake-off ----
    with open(DEF_PKL, "rb") as f:
        defn = pickle.load(f)
    full_feats = defn["stable"]
    full_params = defn.get("cox_params", BASELINE_COX)
    # The full row must equal survival_definitive's headline exactly, so it uses
    # the SAME redundancy-pruned matrix the selection + hazard model used (the
    # construct ablations above intentionally stay on the full feature set).
    Xp = drop_redundant_features(X)
    posp = {c: i for i, c in enumerate(Xp.columns)}
    full_idx = [posp[c] for c in full_feats]

    rows = []  # (label, scores)
    rows.append(("KM null (no covariates)", km_null_cindex(df, X, y, dur, evt)))

    singles = [
        ("Coach age only", "Feature 1"),
        ("Internal hire only", "cf_internal_hire"),
        ("Inherited team SRS only", "tq_srs"),
        ("Inherited defensive SRS only", "tq_dsrs"),
        ("Roster turnover only", "hire_roster_turnover"),
    ]
    for label, name in singles:
        rows.append((label, cox_cindex(df, X, y, dur, evt, [pos[name]])))

    rows.append((f"Inherited situation ({len(inherited)})",
                 cox_cindex(df, X, y, dur, evt, inherited)))
    rows.append((f"Coach resume ({len(resume)})",
                 cox_cindex(df, X, y, dur, evt, resume)))
    rows.append((f"Coach unit performance ({len(unit)})",
                 cox_cindex(df, X, y, dur, evt, unit)))
    rows.append((f"FULL CV-selected ({len(full_idx)})",
                 cox_cindex(df, Xp, y, dur, evt, full_idx, full_params)))

    tf = MODEL_CONFIG["test_size"]
    print("=" * 64)
    print(f"{'baseline':<34}{'C-index':>10}{'NB 95% CI':>20}")
    print("-" * 64)
    summary = {}
    for label, sc in rows:
        m, lo, hi = nb_tci(sc, tf)
        summary[label] = {"mean": m, "lo": lo, "hi": hi, "raw": np.asarray(sc)}
        print(f"{label:<34}{m:>10.3f}   [{lo:.3f}, {hi:.3f}]")
    print("=" * 64)
    print("\nFull-model selected features:", full_feats)
    print("Full-model Cox params:", full_params)

    out = os.path.join(project_root, "analysis", "survival_null_baseline.pkl")
    with open(out, "wb") as f:
        pickle.dump({"summary": summary, "n_seeds": N_SEEDS,
                     "n_stints": int(len(df)), "boundary": boundary,
                     "full_feats": full_feats, "full_params": full_params}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
