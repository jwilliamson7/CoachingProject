#!/usr/bin/env python
"""
Definitive firing-survival analysis (JQAS) on the canonical + enriched dataset,
pruned to a non-redundant candidate pool for selection and inference (see
survival_methods.drop_redundant_features). Supersedes every old 4-block
survival_*.pkl. Produces the numbers and curves the paper reports, with the full
reviewer-grade methods stack:

  - FULLY COX-NATIVE pipeline: feature ranking, stability selection, prediction,
    and hazard-ratio interpretation all use the Cox model class, so there is no
    cross-model importance disagreement (XGB-SHAP vs Cox coef) to adjudicate.
    Feature set by STABILITY SELECTION on Cox ridge |z-coef| (coach-level
    subsampling, Meinshausen & Buhlmann) -> a tight, high-frequency stable core,
    plus a paired Nadeau-Bengio check it is not worse than the CV-argmax set.
  - Model bake-off (Cox primary + Weibull/LogNormal/LogLogistic AFT + XGB-AFT),
    each scored with Harrell's C AND Uno's censoring-robust C.
  - CALIBRATION: IPCW integrated Brier score + Brier-by-horizon (Cox), against the
    covariate-free Kaplan-Meier null on the same splits.
  - HAZARD RATIOS: cause-specific firing Cox with CLUSTER-ROBUST SEs on coach id,
    plus the Grambsch-Therneau PH test.
  - COMPETING RISKS: Aalen-Johansen cumulative incidence (firing vs voluntary).
  - KM median tenure + log-rank era test.
  - The null/naive baseline table (loaded from survival_null_baseline.pkl).

Iterates at N_SEEDS (50) + STAB_BOOTS (2000): the locked camera-ready settings.

--selection-only runs the data prep and the stability selection exactly as the full
run does (same B, q grid, seed), writes every candidate's selection-frequency
distribution to analysis/survival_selection_distribution.pkl, prints it, and stops
before the bake-off (survival_definitive.pkl is not written). It was the input to the
retention rule (survival_methods.choose_stable).

Usage:
    python scripts/survival_definitive.py [--selection-only]
"""

import os
import sys
import pickle
import argparse
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season, build_survival_targets, nb_tci
from scripts.survival_models import (
    cox_builder, weibull_builder, lognormal_builder, loglogistic_builder,
    xgbaft_builder, N_SEEDS, COX_GRID,
)
from scripts.survival_feature_selection import full_cv_folds, cv_cindex_at_k, tune_cox_penalty
from scripts.survival_combined_features import K_GRID
from scripts.compare_ordinal_multiclass import nadeau_bengio_ttest
from model.config import MODEL_CONFIG
from model.pipeline import make_imputer
from scripts.survival_methods import (
    build_competing_targets, firing_event, aalen_johansen_cif, cif_at, cox_hazard_ratios,
    ph_test, survival_eval, cox_calibration, stability_selection_multi, cox_importance,
    drop_redundant_features, with_structural_indicators, structural_companions,
    selectable, choose_stable, nm, friendly_name, _PLACEHOLDER_NAMES,
    STRUCTURAL_INDICATORS, STAB_THR, BRIER_GRID, ERA_BREAKS, FIRED, VOLUNTARY,
)

warnings.filterwarnings("ignore")

STAB_K = 10          # top-K (q) per subsample. Primary value; STAB_K_GRID below
                     # records the K-sensitivity from the same subsamples so the
                     # choice is shown to be immaterial. With p~75 candidates, K=10
                     # and STAB_THR=0.6 give a valid Meinshausen-Buhlmann error bound
                     # E[V] <= (1/(2*thr-1)) * K^2/p ~= 6.7 (undefined at thr=0.5).
STAB_K_GRID = (10, 15, 20)  # K-sensitivity grid (one bootstrap pass, shared draws)
STAB_BOOTS = 2000    # coach-level subsamples (camera-ready). 2000 drives the
                     # Monte-Carlo SE of each selection frequency down to ~0.011,
                     # so features near the 0.6 retention line are placed reliably
                     # rather than by subsampling noise.
# STAB_THR (the retention threshold) and BRIER_GRID live in survival_methods, next to
# choose_stable and uno_cindex, which use them.
CANON_PKL = os.path.join(project_root, "analysis", "survival_canonical_validate.pkl")
OUT_PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")
SELECTION_PKL = os.path.join(project_root, "analysis", "survival_selection_distribution.pkl")
AFT_P = {"penalizer": 0.01}


def run_stability(df, X, dur, evt):
    """Cox-native stability selection at the locked settings. One bootstrap pass scores
    top-K membership for every K in STAB_K_GRID, so the K choice is shown to be
    immaterial (apples-to-apples, shared subsamples). Returns {K: freq_series}."""
    print(f"Stability selection: {STAB_BOOTS} coach-level subsamples, "
          f"top-K K in {STAB_K_GRID} (primary K={STAB_K}, threshold {STAB_THR})...")
    return stability_selection_multi(df, X, dur, evt, Ks=STAB_K_GRID,
                                     n_boot=STAB_BOOTS, subsample=0.5, seed=0,
                                     importance_fn=cox_importance)


def save_selection_distribution(freq_multi, X, n_stints):
    """Write and print every candidate's selection frequency at each q, with its
    Monte-Carlo SE sqrt(p(1-p)/B), the structural indicators it would bring into a
    model, and its display name: the evidence for choosing the retention rule."""
    cands = [c for c in X.columns if c not in STRUCTURAL_INDICATORS]
    tab = pd.DataFrame(index=pd.Index(cands, name="column"))
    tab["variable"] = [nm(c) for c in cands]
    tab["name"] = [friendly_name(c) for c in cands]
    for q in STAB_K_GRID:
        p = freq_multi[q].reindex(cands).astype(float)
        tab[f"freq_q{q}"] = p
        tab[f"se_q{q}"] = np.sqrt(p * (1.0 - p) / STAB_BOOTS)
    tab["companions"] = [structural_companions([c], X) for c in cands]
    tab = tab.sort_values(f"freq_q{STAB_K}", ascending=False, kind="stable")

    with open(SELECTION_PKL, "wb") as f:
        pickle.dump({
            "table": tab,
            "freq": {q: tab[f"freq_q{q}"] for q in STAB_K_GRID},
            "se": {q: tab[f"se_q{q}"] for q in STAB_K_GRID},
            "companions": tab["companions"].to_dict(),
            "names": tab["name"].to_dict(),
            "n_boot": STAB_BOOTS, "q_grid": STAB_K_GRID, "primary_q": STAB_K,
            "thr": STAB_THR, "subsample": 0.5, "seed": 0,
            "n_stints": n_stints, "n_candidates": len(cands),
        }, f)

    print(f"\nSelection-frequency distribution: {len(cands)} candidates "
          f"(structural indicators excluded), B={STAB_BOOTS}; "
          f"Monte-Carlo SE = sqrt(p(1-p)/B)")
    qhdr = "".join(f"{f'q={q}  (SE)':>16}" for q in STAB_K_GRID)
    print(f"  {'#':>3}  {'feature':<38}{'variable':<28}{qhdr}  companions")
    for r, (c, row) in enumerate(tab.iterrows(), 1):
        cells = "".join(f"{row[f'freq_q{q}']:>9.3f} ({row[f'se_q{q}']:.3f})"
                        for q in STAB_K_GRID)
        comp = ", ".join(row["companions"]) or "-"
        print(f"  {r:>3}  {row['name'][:37]:<38}{row['variable'][:27]:<28}{cells}  {comp}")
    print("\n  candidates at or above each frequency, by q:")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"    freq >= {thr:.1f}: " + "  ".join(
            f"q={q}: {int((tab[f'freq_q{q}'] >= thr).sum()):>2}" for q in STAB_K_GRID))
    print(f"\nSaved {SELECTION_PKL}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-only", action="store_true",
                    help="write the selection-frequency distribution and stop")
    args = ap.parse_args(argv)

    # known_only=False: include the still-active coaches (tenure class -1). The
    # classification framing had to drop them (an ongoing tenure has no class);
    # survival analysis uses them correctly as right-censored observations, which
    # is the whole point of the framework. They code as cause=0 (active) and bring
    # the sample to the full set of currently-employed head coaches.
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, cause = dur.loc[keep], cause.loc[keep]
    # Collapse collinear / double-measured features before selection + inference
    # (raw hiring-team box scores duplicated by schedule-adjusted SRS; the
    # tq_srs/tq_mov linear identities; same-construct near-duplicates). See
    # survival_methods.DROP_REDUNDANT_FEATURES. Construct ablations keep the full set.
    n_before = X.shape[1]
    X = drop_redundant_features(X)
    print(f"redundancy prune: {n_before} -> {X.shape[1]} feats "
          f"({n_before - X.shape[1]} dropped)\n")
    durb, evtb = build_survival_targets(df, boundary)  # binary firing target
    durb, evt = durb.loc[keep], pd.Series(firing_event(cause), index=keep)
    cols = list(X.columns)
    n_fire = int((cause == FIRED).sum())
    n_vol = int((cause == VOLUNTARY).sum())
    n_cens = int(len(cause) - n_fire - n_vol)
    print(f"{len(df)} stints | {X.shape[1]} feats | fired={n_fire} "
          f"voluntary={n_vol} active-censored={n_cens} | seeds={N_SEEDS}\n")

    if args.selection_only:
        save_selection_distribution(run_stability(df, X, dur, evt), X, len(df))
        return

    # reuse the tuned params from the canonical validate (robust; camera-ready re-tunes)
    canon = pickle.load(open(CANON_PKL, "rb"))
    cox_p = canon.get("cox_params", {"penalizer": 0.5, "l1_ratio": 0.25})
    xgb_p = canon["xgb_params"]
    print(f"Cox params {cox_p}\nXGB params {xgb_p}\n")

    # ---- 1. Cox-native rank + CV-argmax K (reference set) ----
    Ximp_full = make_imputer().fit(X.values).transform(X.values)
    rank, _ = cox_importance(Ximp_full, durb.values.astype(float),
                             evt.values.astype(int))
    rank = selectable(rank, cols)   # structural indicators enter only as companions
    folds_full = full_cv_folds(df, X, y, durb, evt)
    kcurve = {K: cv_cindex_at_k(cox_builder, cox_p, folds_full, rank[:K])
              for K in K_GRID}
    cv_k = max(kcurve, key=kcurve.get)
    topk_idx = list(rank[:cv_k])
    print(f"CV-argmax K={cv_k} (CV C-index {kcurve[cv_k]:.3f}); "
          f"K-curve {{{', '.join(f'{k}:{v:.3f}' for k,v in kcurve.items())}}}\n")

    # ---- 2. Cox-native stability selection -> tight stable core ----
    freq_multi = run_stability(df, X, dur, evt)
    save_selection_distribution(freq_multi, X, len(df))   # the supplement's table
    freq = freq_multi[STAB_K]
    # Retention rule: survival_methods.choose_stable (freq >= STAB_THR, no floor).
    stable = choose_stable(freq, STAB_THR)
    if not stable:
        raise SystemExit(f"no candidate reached selection frequency {STAB_THR} at "
                         f"K={STAB_K}; see --selection-only")
    stable_idx = [cols.index(c) for c in stable]
    print(f"stable set ({len(stable)} feats, freq>= {STAB_THR}, K={STAB_K}):")
    for c in stable:
        print(f"  {nm(c):<28} {c:<14} {freq[c]:.2f}")
    print(f"  freq>=0.6: {int((freq>=0.6).sum())} | >=0.7: {int((freq>=0.7).sum())}")
    # Structural indicators enter every model with the features whose constant-filled
    # values they mark (make_imputer; CLAUDE.md "Data policies"), so each feature's
    # coefficient comes only from rows where it exists.
    model_feats = with_structural_indicators(stable, X)
    model_idx = [cols.index(c) for c in model_feats]
    topk_model_idx = [cols.index(c) for c in
                      with_structural_indicators([cols[i] for i in topk_idx], X)]
    added = [c for c in model_feats if c not in stable]
    print(f"  structural indicators added to the model: {added if added else 'none'}")
    # post-selection collinearity guard: redundant measurements are pruned upstream,
    # but flag any residual stable-set pair with |r|>0.7 so a suppression artifact
    # (two collinear survivors with unstable joint coefficients) cannot slip through.
    if len(stable_idx) > 1:
        cstab = pd.DataFrame(Ximp_full[:, stable_idx], columns=stable).corr().abs().values
        hot = [(stable[i], stable[j], cstab[i, j])
               for i in range(len(stable)) for j in range(i + 1, len(stable))
               if cstab[i, j] > 0.7]
        if hot:
            print("  WARNING stable-set collinearity |r|>0.7:")
            for a, b, r in hot:
                print(f"      {nm(a)} ~ {nm(b)}: r={r:.2f}")
        else:
            print("  collinearity guard: no stable-set pair |r|>0.7 (clean)")
    # K-sensitivity table: every feature that clears the threshold at ANY K, with
    # its frequency at each K (a star marks membership in the primary-K stable set).
    sens = sorted({c for K in STAB_K_GRID for c in choose_stable(freq_multi[K], STAB_THR)},
                  key=lambda c: -freq_multi[STAB_K].get(c, 0.0))
    print(f"  K-sensitivity (freq at K in {STAB_K_GRID}, threshold {STAB_THR}):")
    for c in sens:
        cells = "  ".join(f"K{K}={freq_multi[K][c]:.2f}" for K in STAB_K_GRID)
        star = "*" if c in stable else " "
        print(f"    {star} {nm(c):<26} {cells}")
    print("  selection frequencies (top 20, primary K):")
    for c, v in freq.head(20).items():
        print(f"    {nm(c):<28} {v:.2f}")
    print()

    # The canonical penalty (cox_p) was tuned on the full candidate matrix, for the
    # CV-argmax sets above; the stable set gets its own, tuned the same way on its
    # columns (tune_cox_penalty; ties go to the lightest penalty). It is used for the
    # stable set's bake-off row and calibration.
    stable_p = tune_cox_penalty(folds_full, model_idx, COX_GRID)
    print(f"stable-set Cox penalty (coach-level CV, ties to the lightest): {stable_p}\n")

    # ---- 3. paired NB: stable set vs CV-argmax set (Cox) ----
    ev_stable = survival_eval(cox_builder, stable_p, df, X, y, durb, evt, model_idx, N_SEEDS)
    ev_topk = survival_eval(cox_builder, cox_p, df, X, y, durb, evt, topk_model_idx, N_SEEDS)
    tf = MODEL_CONFIG["test_size"]
    d_ch = np.array(ev_stable["harrell"]) - np.array(ev_topk["harrell"])
    nb_m, nb_t, nb_p = nadeau_bengio_ttest(d_ch, tf)
    print(f"NB stable(K={len(stable)}) - CV-argmax(K={cv_k}) Cox Harrell: "
          f"delta={nb_m:+.4f} t={nb_t:+.3f} p={nb_p:.3g}  "
          f"(stable {nb_tci(ev_stable['harrell'], tf)[0]:.3f} vs "
          f"topk {nb_tci(ev_topk['harrell'], tf)[0]:.3f})\n")

    # ---- 4. concordance of the stable-set Cox (Harrell + Uno). The comparison across
    #      model classes, each choosing its own features, is nested:
    #      survival_nested_eval.py. ----
    h, u = nb_tci(ev_stable["harrell"], tf), nb_tci(ev_stable["uno"], tf)
    bakeoff = {"Cox": {"harrell": h, "uno": u,
                       "harrell_raw": np.array(ev_stable["harrell"]),
                       "uno_raw": np.array(ev_stable["uno"])}}
    print("Stable-set Cox (Nadeau-Bengio 95% CI): "
          f"Harrell {h[0]:.3f} [{h[1]:.3f},{h[2]:.3f}]  Uno {u[0]:.3f} [{u[1]:.3f},{u[2]:.3f}]")

    # ---- 5. calibration (Cox) against the covariate-free Kaplan-Meier null ----
    # The reference is the training KM curve scored on the same held-out stints, not
    # 0.25: a model with no covariates already knows the marginal firing curve.
    cal = cox_calibration(df, X, y, durb, evt, model_idx, stable_p, BRIER_GRID, N_SEEDS)
    ibs, ibs_km = cal["ibs"], cal["ibs_km"]
    ibs_ci, ibs_km_ci = nb_tci(ibs, tf), nb_tci(ibs_km, tf)
    skill = ibs_km - ibs                       # paired, per split; > 0 = Cox better
    skill_ci = nb_tci(skill, tf)
    _, skill_t, skill_p = nadeau_bengio_ttest(skill, tf)
    print(f"\nCalibration (Nadeau-Bengio 95% CI): integrated Brier "
          f"Cox {ibs_ci[0]:.4f} [{ibs_ci[1]:.4f},{ibs_ci[2]:.4f}]   "
          f"KM null {ibs_km_ci[0]:.4f} [{ibs_km_ci[1]:.4f},{ibs_km_ci[2]:.4f}]")
    print(f"  paired skill KM - Cox = {skill_ci[0]:+.4f} [{skill_ci[1]:+.4f},{skill_ci[2]:+.4f}]"
          f"  t={skill_t:.2f} p={skill_p:.3g}; relative skill vs KM "
          f"{1 - ibs_ci[0] / ibs_km_ci[0]:.3f}")
    print("  Cox Brier by horizon:", {int(g): round(float(b), 3)
                                     for g, b in zip(BRIER_GRID, cal["curve"])})
    print("  KM  Brier by horizon:", {int(g): round(float(b), 3)
                                     for g, b in zip(BRIER_GRID, cal["curve_km"])})

    # ---- 6. hazard ratios (cluster-robust) + PH test ----
    # UNPENALIZED for the reported inference: the stable set is small and the events
    # many, so the model is stable without a ridge, and penalized coefficients have no
    # nominal frequentist coverage (ridge vs unpenalized: survival_robustness_check.py).
    Xsel = pd.DataFrame(Ximp_full[:, model_idx], columns=model_feats, index=X.index)
    hr, cph = cox_hazard_ratios(Xsel, dur, cause, df["Coach Name"].values, penalizer=0.0)
    pht = ph_test(Xsel, dur, cause, penalizer=0.0)
    hr.index = [nm(c) for c in hr.index]
    print("\nHazard ratios (cause-specific firing, cluster-robust SE on coach):")
    print(hr.round(3).to_string())
    print(f"PH test: all p>0.05? {bool((pht['p'] > 0.05).all())} "
          f"(min p={pht['p'].min():.3f})")

    # ---- 7. competing-risks CIF ----
    aj_f = aalen_johansen_cif(dur, cause, FIRED)
    aj_v = aalen_johansen_cif(dur, cause, VOLUNTARY)
    print("\nCumulative incidence (competing risks):")
    for t in (2, 4, 6, 8):
        print(f"  by season {t}: fired={cif_at(aj_f.cumulative_density_, t):.3f}  "
              f"voluntary={cif_at(aj_v.cumulative_density_, t):.3f}")

    # ---- 8. KM median + era log-rank at the labor-market breaks ----
    kmf = KaplanMeierFitter().fit(dur.values, evt.values)
    med = float(kmf.median_survival_time_)
    yr = df["Year"].values
    era_p = {}
    for name, brk in ERA_BREAKS.items():
        early = yr < brk
        era_p[name] = float(logrank_test(dur.values[early], dur.values[~early],
                                         evt.values[early], evt.values[~early]).p_value)
    era = np.digitize(yr, sorted(ERA_BREAKS.values()))
    era_p["three_era"] = float(multivariate_logrank_test(dur.values, era, evt.values).p_value)
    print(f"\nKM median firing-free tenure = {med:.0f} seasons; era log-rank p: "
          + ", ".join(f"{k} {v:.3f}" for k, v in era_p.items()))

    # ---- save ----
    with open(OUT_PKL, "wb") as f:
        pickle.dump({
            "n_stints": len(df), "n_fire": n_fire, "n_vol": n_vol, "n_cens": n_cens,
            "n_seeds": N_SEEDS, "boundary": boundary,
            "cox_params": cox_p, "cox_params_stable": stable_p, "xgb_params": xgb_p,
            "kcurve": kcurve, "cv_k": cv_k, "topk": [cols[i] for i in topk_idx],
            "stab_freq": freq, "stab_freq_named": freq.rename(index=nm),
            "stab_freq_multi": freq_multi, "stab_k": STAB_K, "stab_k_grid": STAB_K_GRID,
            "stab_boots": STAB_BOOTS,
            "stable": stable, "stable_named": [nm(c) for c in stable],
            "model_features": model_feats, "model_features_named": [nm(c) for c in model_feats],
            "stab_thr": STAB_THR,
            "nb_stable_vs_topk": {"delta": nb_m, "t": nb_t, "p": nb_p},
            "bakeoff": bakeoff,
            "ibs": np.array(ibs), "ibs_ci": ibs_ci, "brier_grid": BRIER_GRID,
            "brier_curve": cal["curve"],
            "ibs_km": np.array(ibs_km), "ibs_km_ci": ibs_km_ci,
            "brier_curve_km": cal["curve_km"],
            "ibs_skill": skill, "ibs_skill_ci": skill_ci,
            "ibs_skill_test": {"t": skill_t, "p": skill_p},
            "hazard_ratios": hr, "ph_test": pht,
            "cif_fired": aj_f.cumulative_density_, "cif_vol": aj_v.cumulative_density_,
            "km_median": med, "era_logrank_p": era_p, "era_breaks": ERA_BREAKS,
        }, f)
    print(f"\nSaved {OUT_PKL}")


if __name__ == "__main__":
    main()
