#!/usr/bin/env python
"""
Definitive firing-survival analysis (JQAS) on the canonical + enriched dataset
(94 features, pruned to a non-redundant pool of ~74 for selection and inference;
see survival_methods.drop_redundant_features). Supersedes every old 4-block
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
  - CALIBRATION: IPCW integrated Brier score + Brier-by-horizon (Cox).
  - HAZARD RATIOS: cause-specific firing Cox with CLUSTER-ROBUST SEs on coach id,
    plus the Grambsch-Therneau PH test.
  - COMPETING RISKS: Aalen-Johansen cumulative incidence (firing vs voluntary).
  - KM median tenure + log-rank era test.
  - The null/naive baseline table (loaded from survival_null_baseline.pkl).

Iterates at N_SEEDS (50) + STAB_BOOTS (2000): the locked camera-ready settings.

Usage:
    python scripts/survival_definitive.py
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
from lifelines.statistics import logrank_test

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season, build_survival_targets, nb_tci
from scripts.survival_models import (
    cox_builder, weibull_builder, lognormal_builder, loglogistic_builder,
    xgbaft_builder, N_SEEDS,
)
from scripts.survival_feature_selection import full_cv_folds, cv_cindex_at_k
from scripts.survival_combined_features import K_GRID
from scripts.compare_ordinal_multiclass import nadeau_bengio_ttest
from model.config import MODEL_CONFIG
from scripts.data.matrix_factorization_imputation import SVDImputer
from data_constants import get_all_feature_names, HIRING_TEAM_FEATURES
from scripts.survival_methods import (
    build_competing_targets, firing_event, aalen_johansen_cif, cox_hazard_ratios,
    ph_test, survival_eval, cox_calibration, stability_selection_multi, cox_importance,
    drop_redundant_features, FIRED, VOLUNTARY,
)

warnings.filterwarnings("ignore")

STAB_K = 10          # top-K (q) per subsample. Primary value; STAB_K_GRID below
                     # records the K-sensitivity from the same subsamples so the
                     # choice is shown to be immaterial. With p~74, K=10 and
                     # STAB_THR=0.6 give a valid Meinshausen-Buhlmann error bound
                     # E[V] <= (1/(2*thr-1)) * K^2/p ~= 6.8 (undefined at thr=0.5).
STAB_K_GRID = (10, 15, 20)  # K-sensitivity grid (one bootstrap pass, shared draws)
STAB_BOOTS = 2000    # coach-level subsamples (camera-ready). 2000 drives the
                     # Monte-Carlo SE of each selection frequency down to ~0.011,
                     # so features near the 0.6 retention line are placed reliably
                     # rather than by subsampling noise.
STAB_THR = 0.6       # selection-frequency threshold (Meinshausen-Buhlmann floor)
BRIER_GRID = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10.])
CANON_PKL = os.path.join(project_root, "analysis", "survival_canonical_validate.pkl")
AFT_P = {"penalizer": 0.01}

# Feature 1..51 are positional placeholders for the 8 core + 33 unit + 10 hiring
# columns; cf_/org_/hire_/tq_/rf_ columns already carry real names.
_PLACEHOLDER_NAMES = get_all_feature_names() + HIRING_TEAM_FEATURES


def nm(c):
    """Resolve a 'Feature N' placeholder to its real variable name."""
    if isinstance(c, str) and c.startswith("Feature "):
        i = int(c.split()[1]) - 1
        if 0 <= i < len(_PLACEHOLDER_NAMES):
            return _PLACEHOLDER_NAMES[i]
    return c


def main():
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

    # reuse the tuned params from the canonical validate (robust; camera-ready re-tunes)
    canon = pickle.load(open(CANON_PKL, "rb"))
    cox_p = canon.get("cox_params", {"penalizer": 0.5, "l1_ratio": 0.25})
    xgb_p = canon["xgb_params"]
    print(f"Cox params {cox_p}\nXGB params {xgb_p}\n")

    # ---- 1. Cox-native rank + CV-argmax K (reference set) ----
    Ximp_full = SVDImputer().fit(X.values).transform(X.values)
    rank, _ = cox_importance(Ximp_full, durb.values.astype(float),
                             evt.values.astype(int))
    folds_full = full_cv_folds(df, X, y, durb, evt)
    kcurve = {K: cv_cindex_at_k(cox_builder, cox_p, folds_full, rank[:K])
              for K in K_GRID}
    cv_k = max(kcurve, key=kcurve.get)
    topk_idx = list(rank[:cv_k])
    print(f"CV-argmax K={cv_k} (CV C-index {kcurve[cv_k]:.3f}); "
          f"K-curve {{{', '.join(f'{k}:{v:.3f}' for k,v in kcurve.items())}}}\n")

    # ---- 2. Cox-native stability selection -> tight stable core ----
    # One bootstrap pass scores top-K membership for every K in STAB_K_GRID, so the
    # K choice is shown to be immaterial (apples-to-apples, shared subsamples).
    print(f"Stability selection: {STAB_BOOTS} coach-level subsamples, "
          f"top-K K in {STAB_K_GRID} (primary K={STAB_K}, threshold {STAB_THR})...")
    freq_multi = stability_selection_multi(df, X, dur, evt, Ks=STAB_K_GRID,
                                           n_boot=STAB_BOOTS, subsample=0.5, seed=0,
                                           importance_fn=cox_importance)
    freq = freq_multi[STAB_K]
    stable = list(freq[freq >= STAB_THR].index)
    floored = len(stable) < 3
    if floored:                               # degenerate-run floor only
        stable = list(freq.head(5).index)
    stable_idx = [cols.index(c) for c in stable]
    print(f"stable set ({len(stable)} feats, freq>= {STAB_THR}, K={STAB_K}"
          f"{', FLOORED to top-5' if floored else ''}):")
    for c in stable:
        print(f"  {nm(c):<28} {c:<14} {freq[c]:.2f}")
    print(f"  freq>=0.6: {int((freq>=0.6).sum())} | >=0.7: {int((freq>=0.7).sum())}")
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
    sens = sorted({c for K in STAB_K_GRID for c in freq_multi[K][freq_multi[K] >= STAB_THR].index},
                  key=lambda c: -freq_multi[STAB_K].get(c, 0.0))
    print(f"  K-sensitivity (freq at K in {STAB_K_GRID}, threshold {STAB_THR}):")
    for c in sens:
        cells = "  ".join(f"K{K}={freq_multi[K][c]:.2f}" for K in STAB_K_GRID)
        star = "*" if freq[c] >= STAB_THR else " "
        print(f"    {star} {nm(c):<26} {cells}")
    print("  selection frequencies (top 20, primary K):")
    for c, v in freq.head(20).items():
        print(f"    {nm(c):<28} {v:.2f}")
    print()

    # ---- 3. paired NB: stable set vs CV-argmax set (Cox) ----
    ev_stable = survival_eval(cox_builder, cox_p, df, X, y, durb, evt, stable_idx, N_SEEDS)
    ev_topk = survival_eval(cox_builder, cox_p, df, X, y, durb, evt, topk_idx, N_SEEDS)
    tf = MODEL_CONFIG["test_size"]
    d_ch = np.array(ev_stable["harrell"]) - np.array(ev_topk["harrell"])
    m, t, p = nadeau_bengio_ttest(d_ch, tf)
    print(f"NB stable(K={len(stable)}) - CV-argmax(K={cv_k}) Cox Harrell: "
          f"delta={m:+.4f} t={t:+.3f} p={p:.3g}  "
          f"(stable {nb_tci(ev_stable['harrell'], tf)[0]:.3f} vs "
          f"topk {nb_tci(ev_topk['harrell'], tf)[0]:.3f})\n")

    # ---- 4. model bake-off on the stable set (Harrell + Uno) ----
    builders = [("Cox", cox_builder, cox_p),
                ("Weibull AFT", weibull_builder, AFT_P),
                ("LogNormal AFT", lognormal_builder, AFT_P),
                ("LogLogistic AFT", loglogistic_builder, AFT_P),
                ("XGB-AFT", xgbaft_builder, xgb_p)]
    bakeoff = {}
    print("Model bake-off on the stable set (Nadeau-Bengio 95% CI):")
    print(f"  {'model':<18}{'Harrell C':>20}{'Uno C':>20}")
    for name, b, p_ in builders:
        ev = ev_stable if name == "Cox" else survival_eval(
            b, p_, df, X, y, durb, evt, stable_idx, N_SEEDS)
        h, u = nb_tci(ev["harrell"], tf), nb_tci(ev["uno"], tf)
        bakeoff[name] = {"harrell": h, "uno": u,
                         "harrell_raw": np.array(ev["harrell"]),
                         "uno_raw": np.array(ev["uno"])}
        print(f"  {name:<18}{h[0]:>9.3f} [{h[1]:.3f},{h[2]:.3f}]"
              f"{u[0]:>9.3f} [{u[1]:.3f},{u[2]:.3f}]")

    # ---- 5. calibration (Cox) ----
    ibs, bs_curve = cox_calibration(df, X, y, durb, evt, stable_idx, cox_p,
                                    BRIER_GRID, N_SEEDS)
    ibs_ci = nb_tci(ibs, tf)
    print(f"\nCalibration (Cox, Nadeau-Bengio 95% CI): integrated Brier {ibs_ci[0]:.4f} "
          f"[{ibs_ci[1]:.4f},{ibs_ci[2]:.4f}]  (0.25 = uninformative)")
    print("  Brier by horizon:", dict(zip(BRIER_GRID.astype(int), np.round(bs_curve, 3))))

    # ---- 6. hazard ratios (cluster-robust) + PH test ----
    # UNPENALIZED for the reported inference: with ~5 near-uncorrelated features
    # and EPV ~64 (319 events) the model is stable without a ridge, and penalized
    # coefficients have no nominal frequentist coverage. Verified in
    # survival_robustness_check.py: penalizer 0.1->0 leaves p-values/CIs
    # materially unchanged and pushes the internal-hire HR away from 1 (1.55->1.62).
    Xsel = pd.DataFrame(Ximp_full[:, stable_idx], columns=stable, index=X.index)
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

    def cif_at(aj, t):
        c = aj.cumulative_density_
        return float(c[c.index <= t].iloc[-1, 0]) if (c.index <= t).any() else 0.0
    print("\nCumulative incidence (competing risks):")
    for t in (2, 4, 6, 8):
        print(f"  by season {t}: fired={cif_at(aj_f,t):.3f}  voluntary={cif_at(aj_v,t):.3f}")

    # ---- 8. KM median + era log-rank ----
    kmf = KaplanMeierFitter().fit(dur.values, evt.values)
    med = float(kmf.median_survival_time_)
    yr = df["Year"].values
    early = yr < 2000
    lr = logrank_test(dur.values[early], dur.values[~early],
                      evt.values[early], evt.values[~early])
    print(f"\nKM median firing-free tenure = {med:.0f} seasons; "
          f"era log-rank p = {lr.p_value:.3f}")

    # ---- save ----
    out = os.path.join(project_root, "analysis", "survival_definitive.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "n_stints": len(df), "n_fire": n_fire, "n_vol": n_vol, "n_cens": n_cens,
            "n_seeds": N_SEEDS, "boundary": boundary,
            "cox_params": cox_p, "xgb_params": xgb_p,
            "kcurve": kcurve, "cv_k": cv_k, "topk": [cols[i] for i in topk_idx],
            "stab_freq": freq, "stab_freq_named": freq.rename(index=nm),
            "stab_freq_multi": freq_multi, "stab_k": STAB_K, "stab_k_grid": STAB_K_GRID,
            "stable": stable, "stable_named": [nm(c) for c in stable],
            "stab_thr": STAB_THR,
            "nb_stable_vs_topk": {"delta": m, "t": t, "p": p},
            "bakeoff": bakeoff,
            "ibs": np.array(ibs), "ibs_ci": ibs_ci, "brier_grid": BRIER_GRID,
            "brier_curve": bs_curve,
            "hazard_ratios": hr, "ph_test": pht,
            "cif_fired": aj_f.cumulative_density_, "cif_vol": aj_v.cumulative_density_,
            "km_median": med, "era_logrank_p": float(lr.p_value),
        }, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
