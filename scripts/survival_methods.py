#!/usr/bin/env python
"""
Methodology additions for the definitive firing-survival analysis (JQAS).

These close the gaps a survival-literate reviewer expects in a Cox-primary,
recurrent-subject, multi-exit-type paper. Everything here REUSES the existing
leakage-free pipeline and model builders; nothing re-implements the
split/impute/select/fit block.

Pieces:
  build_competing_targets   competing-risks event coding (0=admin-censored,
                            1=fired, 2=voluntary exit) -- voluntary departures
                            are a competing event, not plain censoring.
  aalen_johansen_cif        cumulative incidence of firing vs voluntary exit
                            (the correct absolute-risk curve; replaces 1-KM).
  cox_hazard_ratios         final cause-specific Cox on the selected features
                            with CLUSTER-ROBUST SEs on coach id (repeat coaches)
                            -> hazard ratios + 95% CI + p.
  ph_test                   Grambsch-Therneau proportional-hazards test on that
                            model (scaled Schoenfeld residuals), per-covariate
                            and global.
  ipcw_brier / integrated_brier   IPCW (Graf) Brier score + integrated Brier
                            score for CALIBRATION (not just discrimination).
  uno_cindex                Uno's censoring-robust concordance (complements
                            Harrell's C; matters under non-trivial censoring).

Refs: Fine & Gray 1999; Putter et al. 2007 (competing risks); Schoenfeld 1982,
Grambsch & Therneau 1994 (PH test); Lin & Wei 1989, Therneau & Grambsch 2000
(robust/cluster SE); Graf et al. 1999 (Brier); Harrell 1996, Uno et al. 2011 (C).
"""

import os
import sys
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, AalenJohansenFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

from model.pipeline import leakage_free_split
from scripts.survival_analysis import _firing_event_labels
from scripts.data.engineer_career_features import reconstruct_tenure

warnings.filterwarnings("ignore")

# cause codes
CENSORED, FIRED, VOLUNTARY = 0, 1, 2


# --------------------------------------------------------------------------- #
# Competing-risks event coding
# --------------------------------------------------------------------------- #
def build_competing_targets(df, boundary):
    """duration (seasons) + 3-state cause indicator.

    cause = 1  involuntary non-retention (firing) -- the event of interest
    cause = 2  voluntary-while-viable exit (moved team, retired on top, health)
               -- a COMPETING event: this stint can never end in a firing, so
               treating it as plain censoring overstates firing incidence
    cause = 0  still active at the data boundary -- legitimate right-censoring

    The split of the 38 non-firings is exact: an event=0 label at last_year >=
    boundary is administrative censoring (active coach), otherwise it is a
    documented voluntary departure (the competing risk).
    """
    labels = _firing_event_labels()
    dur, cause = {}, {}
    for c, yr in zip(df["Coach Name"], df["Year"]):
        t = reconstruct_tenure(c, int(yr))
        if t is None:
            continue
        last = int(yr) + t - 1
        idx = df.index[(df["Coach Name"] == c) & (df["Year"] == yr)][0]
        ev = labels.get((c, last), 0 if last >= boundary else 1)
        dur[idx] = t
        if ev == 1:
            cause[idx] = FIRED
        else:
            cause[idx] = CENSORED if last >= boundary else VOLUNTARY
    return pd.Series(dur), pd.Series(cause)


def firing_event(cause):
    """Binary firing indicator (1=fired, 0 otherwise) for cause-specific Cox /
    Harrell C: both voluntary exits and active coaches are censored for the
    'what drives firing' hazard."""
    return (np.asarray(cause) == FIRED).astype(int)


# --------------------------------------------------------------------------- #
# Cumulative incidence (Aalen-Johansen) -- correct absolute firing risk
# --------------------------------------------------------------------------- #
def aalen_johansen_cif(dur, cause, event_of_interest=FIRED, seed=42):
    """Cumulative incidence function for one cause under competing risks.

    Returns the fitted AalenJohansenFitter. Integer-season ties are broken by
    lifelines' internal jitter (fixed seed for reproducibility)."""
    ajf = AalenJohansenFitter(seed=seed, calculate_variance=True)
    ajf.fit(np.asarray(dur, float), np.asarray(cause, int),
            event_of_interest=event_of_interest)
    return ajf


# --------------------------------------------------------------------------- #
# Final cause-specific Cox with cluster-robust SEs -> hazard ratios
# --------------------------------------------------------------------------- #
def cox_hazard_ratios(Xsel_df, dur, cause, coach_ids,
                      penalizer=0.1, l1_ratio=0.0):
    """Cause-specific Cox for FIRING on the selected (already imputed) features.

    Inference is what we report, so we use a small RIDGE (stable with correlated
    features, keeps valid SEs -- unlike lasso) and CLUSTER-ROBUST sandwich SEs on
    coach id, since coaches recur across stints and violate independence.
    Returns (summary_df, fitted_model). summary_df has HR, 95% CI, robust p.
    """
    d = Xsel_df.copy()
    d["T"] = np.asarray(dur, float)
    d["E"] = firing_event(cause)
    d["coach"] = np.asarray(coach_ids)
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(d, duration_col="T", event_col="E", cluster_col="coach", robust=True)
    s = cph.summary
    out = pd.DataFrame({
        "HR": s["exp(coef)"],
        "CI_low": s["exp(coef) lower 95%"],
        "CI_high": s["exp(coef) upper 95%"],
        "p": s["p"],
    })
    return out.sort_values("p"), cph


def ph_test(Xsel_df, dur, cause, penalizer=0.1):
    """Grambsch-Therneau scaled-Schoenfeld PH test on the selected features.

    Fits a plain (non-cluster) cause-specific Cox: the PH assumption concerns the
    covariate effects, not the SE estimator, and the cluster_col used for the
    robust HR model breaks proportional_hazard_test's residual lookup. Returns a
    per-covariate DataFrame (test statistic + p). p>0.05 = no detected violation.
    """
    d = Xsel_df.copy()
    d["T"] = np.asarray(dur, float)
    d["E"] = firing_event(cause)
    m = CoxPHFitter(penalizer=penalizer)
    m.fit(d, duration_col="T", event_col="E")
    return proportional_hazard_test(m, d, time_transform="rank").summary


# --------------------------------------------------------------------------- #
# Calibration: IPCW (Graf) Brier score + integrated Brier score
# --------------------------------------------------------------------------- #
def _km_censoring(dur_train, evt_train):
    """KM estimate of the CENSORING survival function G(t)=P(C>t) from training
    (event indicator flipped). Used for IPCW weights in the Brier score."""
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(dur_train, float), 1 - np.asarray(evt_train, int))
    return kmf


def _G(kmf_cens, t):
    g = kmf_cens.predict(t)
    return float(max(g, 1e-8))


def ipcw_brier(t, surv_t, dur_test, evt_test, kmf_cens):
    """Graf (1999) IPCW Brier score at horizon t.

    surv_t[i] = model's predicted P(T_i > t). evt_test = firing indicator.
    Subjects fired by t are weighted 1/G(T_i); subjects surviving past t by
    1/G(t); subjects censored before t contribute 0 (their weight is undefined).
    """
    dur_test = np.asarray(dur_test, float)
    evt_test = np.asarray(evt_test, int)
    surv_t = np.asarray(surv_t, float)
    bs = 0.0
    for i in range(len(dur_test)):
        Ti, di, Si = dur_test[i], evt_test[i], surv_t[i]
        if Ti <= t and di == 1:
            bs += (0.0 - Si) ** 2 / _G(kmf_cens, Ti)
        elif Ti > t:
            bs += (1.0 - Si) ** 2 / _G(kmf_cens, t)
        # censored before t -> contributes 0
    return bs / len(dur_test)


def integrated_brier(times, surv_matrix, dur_test, evt_test, kmf_cens):
    """Integrated Brier score over `times` (trapezoidal). surv_matrix is
    n_test x len(times) of predicted P(T>t). Lower is better (0.25 = useless)."""
    times = np.asarray(times, float)
    bs = np.array([ipcw_brier(times[j], surv_matrix[:, j], dur_test, evt_test,
                              kmf_cens) for j in range(len(times))])
    return float(np.trapz(bs, times) / (times[-1] - times[0])), bs


# --------------------------------------------------------------------------- #
# Uno's censoring-robust C-index
# --------------------------------------------------------------------------- #
def uno_cindex(risk, dur_test, evt_test, kmf_cens, tau=None):
    """Uno et al. (2011) IPCW concordance. `risk` higher = shorter survival
    (e.g. Cox partial hazard). Comparable pairs are those where the earlier
    time is a firing; each is weighted 1/G(T_i)^2. Complements Harrell's C
    (which is biased under heavy censoring; here censoring is light so the two
    should nearly agree -- reporting both pre-empts the objection)."""
    dur_test = np.asarray(dur_test, float)
    evt_test = np.asarray(evt_test, int)
    risk = np.asarray(risk, float)
    if tau is None:
        tau = dur_test[evt_test == 1].max()
    num = den = 0.0
    n = len(dur_test)
    for i in range(n):
        if evt_test[i] != 1 or dur_test[i] > tau:
            continue
        w = 1.0 / _G(kmf_cens, dur_test[i]) ** 2
        for j in range(n):
            if dur_test[j] > dur_test[i]:
                den += w
                if risk[i] > risk[j]:
                    num += w
                elif risk[i] == risk[j]:
                    num += 0.5 * w
    return num / den if den > 0 else np.nan


# --------------------------------------------------------------------------- #
# Survival-native per-seed evaluation: Harrell C + Uno C (any builder)
# --------------------------------------------------------------------------- #
def survival_eval(builder, params, df, X, y, dur, evt, idx, n_seeds):
    """Leakage-free N_SEEDS evaluation returning Harrell's C and Uno's C per seed
    for any survival builder (Cox / AFT / XGB-AFT). surv_score is oriented so
    higher = LONGER survival; Uno's risk argument is its negation."""
    out = {"harrell": [], "uno": []}
    for seed in range(n_seeds):
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)
        Ttr = dur.loc[split.train_index].values.astype(float)
        Etr = evt.loc[split.train_index].values.astype(int)
        Tte = dur.loc[split.test_index].values.astype(float)
        Ete = evt.loc[split.test_index].values.astype(int)
        m = builder(params).fit(split.X_train, Ttr, Etr)
        score = np.asarray(m.surv_score(split.X_test)).ravel()
        kmf = _km_censoring(Ttr, Etr)
        out["harrell"].append(concordance_index(Tte, score, Ete))
        out["uno"].append(uno_cindex(-score, Tte, Ete, kmf))
    return out


def cox_calibration(df, X, y, dur, evt, idx, params, grid, n_seeds):
    """Per-seed IPCW integrated Brier score for the Cox primary model, plus the
    mean Brier-by-horizon curve (for the calibration figure)."""
    ibs_list, bs_rows = [], []
    cols = [f"f{i}" for i in range(len(idx))]
    for seed in range(n_seeds):
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)
        Ttr = dur.loc[split.train_index].values.astype(float)
        Etr = evt.loc[split.train_index].values.astype(int)
        Tte = dur.loc[split.test_index].values.astype(float)
        Ete = evt.loc[split.test_index].values.astype(int)
        tr = pd.DataFrame(split.X_train, columns=cols)
        tr["T"], tr["E"] = Ttr, Etr
        m = CoxPHFitter(penalizer=params.get("penalizer", 0.1),
                        l1_ratio=params.get("l1_ratio", 0.0))
        m.fit(tr, "T", "E")
        sf = m.predict_survival_function(
            pd.DataFrame(split.X_test, columns=cols), times=grid).T.values
        kmf = _km_censoring(Ttr, Etr)
        ibs, bs = integrated_brier(grid, sf, Tte, Ete, kmf)
        ibs_list.append(ibs)
        bs_rows.append(bs)
    return ibs_list, np.array(bs_rows).mean(axis=0)


# --------------------------------------------------------------------------- #
# Stability selection (Meinshausen & Buhlmann 2010)
# --------------------------------------------------------------------------- #
def cox_importance(Ximp, T, E, penalizer=0.1):
    """Cox-native feature ranking: |standardized coefficient| from a ridge Cox.
    The interpretable counterpart to XGB-SHAP -- with this as the stability-
    selection criterion, selection, prediction, and the hazard-ratio
    interpretation are all the SAME (Cox) model class, so there is no
    cross-model importance disagreement to adjudicate. Features are z-scored so
    the penalized coefficients are on a comparable scale."""
    Ximp = np.asarray(Ximp, float)
    mu, sd = Ximp.mean(0), Ximp.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (Ximp - mu) / sd
    d = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
    d["T"], d["E"] = np.asarray(T, float), np.asarray(E, int)
    m = CoxPHFitter(penalizer=penalizer).fit(d, "T", "E")
    imp = np.abs(m.params_.values)
    return np.argsort(imp)[::-1], imp


def stability_selection(df, X, dur, evt, xgb_params=None, K=15, n_boot=100,
                        subsample=0.5, seed=0, importance_fn=None):
    """Coach-level subsampling stability selection. On each of n_boot subsamples
    (half the coaches, drawn without replacement so no coach leaks across the
    split), re-impute (fit on the subsample only), re-rank features, and record
    top-K membership. Returns per-feature selection frequency -- the data-driven,
    plateau-robust way to choose a feature set larger than the CV argmax.

    importance_fn(Ximp, T, E) -> (rank, imp). Default is XGB-AFT survival
    importance (TreeSHAP); pass `cox_importance` for a fully Cox-native pipeline.
    """
    from scripts.data.matrix_factorization_imputation import SVDImputer
    if importance_fn is None:
        from scripts.survival_feature_selection import survival_importance
        def importance_fn(Z, T, E):
            return survival_importance(Z, T, E, xgb_params)

    rng = np.random.default_rng(seed)
    coaches = df["Coach Name"].unique()
    cols = list(X.columns)
    counts = np.zeros(len(cols))
    Tv = dur.values.astype(float)
    Ev = evt.values.astype(int)
    n_take = max(1, int(len(coaches) * subsample))
    for _ in range(n_boot):
        samp = rng.choice(coaches, size=n_take, replace=False)
        mask = df["Coach Name"].isin(samp).values
        Ximp = SVDImputer().fit(X.values[mask]).transform(X.values[mask])
        rank, _ = importance_fn(Ximp, Tv[mask], Ev[mask])
        for i in rank[:K]:
            counts[i] += 1
    return pd.Series(counts / n_boot, index=cols).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def main():
    from model.pipeline import load_modeling_data
    from scripts.survival_analysis import global_max_season

    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X = df.loc[keep], X.loc[keep]
    counts = cause.value_counts().to_dict()
    print(f"boundary={boundary} | n={len(dur)} | "
          f"fired={counts.get(FIRED,0)} active-censored={counts.get(CENSORED,0)} "
          f"voluntary={counts.get(VOLUNTARY,0)}")
    assert counts.get(FIRED) == 319 and counts.get(CENSORED) == 11 \
        and counts.get(VOLUNTARY) == 27, "competing-risk split drifted!"

    # CIF: firing vs voluntary
    aj_fire = aalen_johansen_cif(dur, cause, FIRED)
    aj_vol = aalen_johansen_cif(dur, cause, VOLUNTARY)
    for t in (2, 4, 6, 8):
        cf = float(aj_fire.cumulative_density_.iloc[
            (aj_fire.cumulative_density_.index <= t)].iloc[-1, 0])
        cv = float(aj_vol.cumulative_density_.iloc[
            (aj_vol.cumulative_density_.index <= t)].iloc[-1, 0])
        print(f"  by season {t}: CIF(fired)={cf:.3f}  CIF(voluntary)={cv:.3f}")
    print("survival_methods self-test OK")


if __name__ == "__main__":
    main()
