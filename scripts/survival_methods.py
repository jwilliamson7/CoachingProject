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
                            (the correct absolute-risk curve; replaces 1-KM),
                            read through the end of a season with cif_at.
  cox_hazard_ratios         final cause-specific Cox on the selected features
                            with CLUSTER-ROBUST SEs on coach id (repeat coaches)
                            -> hazard ratios + 95% CI + p.
  ph_test                   Grambsch-Therneau proportional-hazards test on that
                            model (scaled Schoenfeld residuals), per-covariate
                            and global.
  ipcw_brier / integrated_brier   IPCW (Graf) Brier score + integrated Brier
                            score for CALIBRATION (not just discrimination);
                            cox_brier / km_null_brier score a Cox model and its
                            covariate-free Kaplan-Meier null on the same split.
  uno_cindex                Uno's censoring-robust concordance (complements
                            Harrell's C; matters under non-trivial censoring).
  choose_stable             the stability-selection retention rule (one place).

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

from model.pipeline import leakage_free_split, varying_columns, load_modeling_data, make_imputer
from scripts.survival_analysis import _firing_event_labels, _active_labels, global_max_season
from scripts.data.engineer_career_features import (reconstruct_tenure, load_history,
                                                   hiring_team_info, canon_employer)

warnings.filterwarnings("ignore")

# cause codes
CENSORED, FIRED, VOLUNTARY = 0, 1, 2


# --------------------------------------------------------------------------- #
# Redundant-feature prune (collinearity / double-measurement removal)
# --------------------------------------------------------------------------- #
# The canonical feature set measures a few constructs twice, which produced a
# multicollinearity/suppression problem in the multivariable hazard model:
#   (1) the hiring team's pre-hire quality is captured BOTH as raw box scores
#       (hiring_team_* points/yards) AND as schedule-adjusted SRS (tq_*). The SRS
#       version is preferred -- it adjusts for strength of schedule -- so the raw
#       scoring/yardage duplicates are dropped.
#   (2) tq_srs = tq_osrs + tq_dsrs and tq_mov = tq_srs - tq_sos are EXACT linear
#       identities (R^2=1.0), so keeping all six tq_ columns is perfect collinearity;
#       the non-redundant basis is {tq_osrs, tq_dsrs, tq_sos, tq_srs_traj}.
#   (3) same-construct near-duplicates elsewhere (|r|>=0.85): penalties vs penalty
#       yards, rushing yards vs rushing first downs, etc. -- one representative each.
# This prune feeds FEATURE SELECTION and the FINAL HAZARD MODEL only; the
# construct-level ablations in survival_null_baseline keep the full block (a
# construct's predictive ceiling legitimately uses all of its measurements).
# Pairs that LOOK correlated but are conceptually distinct are intentionally KEPT:
#   num_times_hc vs num_yr_nfl_hc (count of stints vs accumulated HC years),
#   rf_final_unit_pctl vs rf_final_winpct_pctl (unit quality vs team success).
DROP_REDUNDANT_FEATURES = [
    # Tier 1 -- inherited-team double measurement + SRS linear identities
    "tq_srs", "tq_mov",
    "hiring_team_win_pct", "hiring_team_points_scored", "hiring_team_points_allowed",
    "hiring_team_yards_offense", "hiring_team_yards_allowed",
    "hiring_team_yards_per_play", "hiring_team_yards_per_play_allowed",
    # Tier 2 -- same-construct near-duplicates (keep one representative each)
    "Yds__unit", "NY/A Passing__unit", "Int Passing__unit", "1stD Rushing__unit",
    "Yds Penalties__unit", "Pts Average Drive__unit", "PF (Points For)__unit",
    "cf_pre_side_def", "hire_starter_age", "hire_roster_age",
    # overall roster retention is the pooled version of the offense/defense split
    # (r = 0.81 / 0.83 with them); keep the split, which carries the side information
    "hire_roster_retention",
    # Removed for a proportional-hazards violation: when selected it was the sole
    # Grambsch-Therneau PH failure (p<0.001) while contributing nothing (HR p=0.88);
    # a null covariate whose PH "violation" is residual noise. Its sibling
    # rf_final_unit_pctl (unit quality, not team record) stays in the pool.
    "rf_final_winpct_pctl",
]

from data_constants import get_all_feature_names as _gafn, HIRING_TEAM_FEATURES as _htf
# Feature 1..51 are positional placeholders for the 8 core + 33 unit + 10 hiring
# columns; cf_/org_/hire_/tq_/rf_ columns already carry real names.
_PLACEHOLDER_NAMES = list(_gafn()) + list(_htf)
# friendly name -> positional 'Feature N' label (base features only; engineered
# features such as tq_/cf_/hire_ carry their real column label already).
_FRIENDLY_TO_COL = {n: f"Feature {i+1}" for i, n in enumerate(_PLACEHOLDER_NAMES)}


def nm(c):
    """Resolve a 'Feature N' placeholder to its real variable name."""
    if isinstance(c, str) and c.startswith("Feature "):
        i = int(c.split()[1]) - 1
        if 0 <= i < len(_PLACEHOLDER_NAMES):
            return _PLACEHOLDER_NAMES[i]
    return c


# Variable name (as nm() returns it) -> manuscript display label. Covers every column of
# the modeling data, pruned ones included, so no table or figure falls back to a raw key.
# Wording follows supplement Table S1 / Table S2 (analysis/feature_audit.md); the labels
# already used in the main-text tables and figures are kept as they were.
FEATURE_LABELS = {
    # core experience
    "age": "Age at hire",
    "num_times_hc": "Prior HC stints",
    "num_yr_col_pos": "College position-coach years",
    "num_yr_col_coor": "College coordinator years",
    "num_yr_col_hc": "College head-coach years",
    "num_yr_nfl_pos": "Years as NFL position coach",
    "num_yr_nfl_coor": "NFL coordinator years",
    "num_yr_nfl_hc": "NFL head-coach years",
    # pooled prior-unit performance (Table S2)
    "PF (Points For)__unit": "Prior-unit points scored",
    "Yds__unit": "Prior-unit total yards",
    "Y/P__unit": "Prior-unit yards-per-play efficiency",
    "TO__unit": "Prior-unit turnovers",
    "1stD__unit": "Prior-unit first downs",
    "Cmp Passing__unit": "Prior-unit completions",
    "Att Passing__unit": "Prior-unit pass attempts",
    "Yds Passing__unit": "Prior-unit passing yards",
    "TD Passing__unit": "Prior-unit passing touchdowns",
    "Int Passing__unit": "Prior-unit interceptions",
    "NY/A Passing__unit": "Prior-unit net yards per attempt",
    "1stD Passing__unit": "Prior-unit passing first downs",
    "Att Rushing__unit": "Prior-unit rush attempts",
    "Yds Rushing__unit": "Prior-unit rushing yards",
    "TD Rushing__unit": "Prior-unit rushing touchdowns",
    "Y/A Rushing__unit": "Prior-unit yards per carry",
    "1stD Rushing__unit": "Prior-unit rushing first downs",
    "Pen__unit": "Prior-unit penalties",
    "Yds Penalties__unit": "Prior-unit penalty yards",
    "1stPy__unit": "Prior-unit first downs by penalty",
    "#Dr__unit": "Prior-unit drives",
    "Sc%__unit": "Prior-unit scoring percentage",
    "TO%__unit": "Prior-unit turnover percentage",
    "Time Average Drive__unit": "Prior-unit time per drive",
    "Plays Average Drive__unit": "Prior-unit plays per drive",
    "Yds Average Drive__unit": "Prior-unit yards per drive",
    "Pts Average Drive__unit": "Prior-unit points per drive",
    "3DAtt__unit": "Prior-unit third-down attempts",
    "3D%__unit": "Prior-unit third-down rate",
    "4DAtt__unit": "Prior-unit fourth-down attempts",
    "4D%__unit": "Prior-unit fourth-down rate",
    "RZAtt__unit": "Prior-unit red-zone attempts",
    "RZPct__unit": "Prior-unit red-zone rate",
    # hiring-team results (prior two seasons)
    "hiring_team_win_pct": "Hiring-team win percentage",
    "hiring_team_points_scored": "Hiring-team points scored",
    "hiring_team_points_allowed": "Hiring-team points allowed",
    "hiring_team_yards_offense": "Hiring-team yards gained",
    "hiring_team_yards_allowed": "Hiring-team yards allowed",
    "hiring_team_yards_per_play": "Hiring-team yards per play",
    "hiring_team_yards_per_play_allowed": "Hiring-team yards per play allowed",
    "hiring_team_turnovers_forced": "Hiring-team turnovers forced",
    "hiring_team_turnovers_committed": "Hiring-team turnovers committed",
    "hiring_team_num_playoff_appearances": "Playoff appearances (prior 2 seasons)",
    # career structure
    "cf_num_nfl_employers": "NFL employers",
    "cf_num_college_employers": "College employers",
    "cf_nfl_share": "NFL career share",
    "cf_age_first_job": "Age at first job",
    "cf_age_first_nfl": "Age at first NFL job",
    "cf_gap_before_hire": "Gap before hire",
    "cf_internal_hire": "Internal hire",
    "cf_years_at_hiring_org": "Years at hiring org",
    "cf_pre_OC": "Hired from OC role",
    "cf_pre_DC": "Hired from DC role",
    "cf_pre_POS": "Hired from position-coach role",
    "cf_pre_HC": "Hired from head-coach role",
    "cf_pre_side_off": "Hired from offensive role",
    "cf_pre_side_def": "Hired from defensive role",
    "cf_pre_level_nfl": "Hired from NFL-level role",
    "cf_ever_nfl_oc": "Ever NFL OC",
    "cf_ever_nfl_dc": "Ever NFL DC",
    "cf_ever_interim_hc": "Ever interim HC",
    "cf_no_unit_record": "No NFL unit record",
    "cf_no_drive_record": "No drive-era record",
    # organizational instability
    "org_prev_coach_tenure": "Predecessor tenure",
    "org_unique_hc_10yr": "Head coaches in prior decade",
    "org_expansion": "Expansion franchise",
    "org_prev_fired": "Predecessor fired",
    # inherited roster
    "hire_qb_av": "Inherited starting-QB value",
    "hire_roster_av": "Inherited roster value",
    "hire_starter_av": "Inherited starter value",
    "hire_starter_age": "Inherited starter age",
    "hire_roster_age": "Inherited roster age",
    "hire_starter_exp": "Inherited starter experience",
    "hire_roster_exp": "Inherited roster experience",
    "hire_recent_r1_picks": "Recent first-round picks",
    "hire_premium_draft": "Premium draft capital",
    "hire_roster_retention": "Roster retention",
    "hire_off_retention": "Offensive retention",
    "hire_def_retention": "Defensive retention",
    "hire_no_roster_data": "No roster data",
    "hire_no_retention_data": "No retention data",
    # inherited team quality (season before the hire)
    "tq_srs": "Inherited SRS",
    "tq_osrs": "Inherited offensive SRS",
    "tq_dsrs": "Inherited defensive SRS",
    "tq_mov": "Inherited margin of victory",
    "tq_sos": "Inherited strength of schedule",
    "tq_srs_traj": "Inherited SRS trajectory",
    # recent form
    "rf_final_unit_pctl": "Final unit percentile",
    "rf_avg_unit_pctl": "Average unit percentile",
    "rf_unit_traj": "Unit trajectory",
    "rf_final_winpct_pctl": "Final team-success percentile",
    "rf_avg_winpct_pctl": "Average team-success percentile",
    "rf_n_seasons_rank": "Ranked seasons",
    "rf_no_trend": "No trend",
}


def friendly_name(c):
    """Display label for a column key or variable name (the name itself if unlisted)."""
    return FEATURE_LABELS.get(nm(c), nm(c))


# Order matters for the covering rule below: broader causes first, and the retention
# flag (1970-71) before the roster flag (1970), whose rows it contains.
STRUCTURAL_INDICATORS = ["cf_no_unit_record", "cf_no_drive_record", "org_expansion",
                         "hire_no_retention_data", "hire_no_roster_data", "rf_no_trend"]
# None of these is ever a selection candidate. Each marks rows where a feature is
# structurally undefined (never led an NFL unit, record predates drive stats, expansion
# franchise, rosters not yet recorded, a single ranked season), so it is a nuisance term
# that identifies its feature's coefficient from the observed rows, not a finding: it
# enters a model only as the companion of a selected feature it marks
# (with_structural_indicators). Ranking one as a candidate would let a data-coverage
# artifact (several are hire-year dummies created by when our sources begin) or a
# definitional zero compete with the substantive features for a slot.

# Head-coach labor-market era breaks (first season of the new regime): the salary cap
# (1994) and the Rooney Rule (adopted December 2002, so first binding on 2003 hires).
ERA_BREAKS = {"salary_cap": 1994, "rooney_rule": 2003}

# Stability-selection retention threshold pi (Meinshausen & Buhlmann 2010); see
# choose_stable for the rule that applies it.
STAB_THR = 0.6
# Horizons (seasons since hire) for the IPCW Brier score. The last one also fixes the
# truncation horizon of Uno's C, so discrimination and calibration are both scored over
# the same first ten seasons, identically in every split.
BRIER_GRID = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10.])
UNO_TAU = float(BRIER_GRID.max())


def selectable(rank, cols):
    """`rank` (column indices, most important first) without the structural indicators,
    which enter models only as companions (see STRUCTURAL_INDICATORS)."""
    return [i for i in rank if cols[i] not in STRUCTURAL_INDICATORS]


def choose_stable(freq, thr=STAB_THR):
    """The stable set: every feature whose stability-selection frequency is at least
    `thr`, most frequent first.

    There is deliberately no floor (no "top-5 if too few clear the bar"), so the result
    can be small, or empty in a resampled training partition. This is the ONE place the
    rule lives; the headline (survival_definitive) and the nested evaluation both call
    it. The rule was fixed after reviewing the frequency distribution written by
    `survival_definitive.py --selection-only` (author decision, 2026-10-29): the
    Meinshausen-Buhlmann bound needs pi > 0.5, and the next-ranked candidates sat on a
    plateau below 0.5, so no lower cutoff could be defended except post hoc.
    """
    return list(freq[freq >= thr].sort_values(ascending=False, kind="stable").index)


# Values set by definition rather than left missing, so their indicator cannot be
# found from NaNs: a one-season trajectory is 0, flagged by rf_no_trend.
DEFINED_VALUE_INDICATORS = {"rf_unit_traj": ["rf_no_trend"]}


def _companions_for(feature, X):
    """Indicators that explain `feature`'s structurally missing values: each one marks
    rows that are ALL missing the feature, and covers missing rows no earlier
    indicator in STRUCTURAL_INDICATORS already covers."""
    need = list(DEFINED_VALUE_INDICATORS.get(feature, []))
    if feature not in X.columns or feature in STRUCTURAL_INDICATORS:
        return need
    miss = X[feature].isna()
    covered = pd.Series(False, index=X.index)
    for ind in STRUCTURAL_INDICATORS:
        if ind not in X.columns:
            continue
        flagged = X[ind] == 1
        if flagged.any() and miss[flagged].all() and (miss & flagged & ~covered).any():
            need.append(ind)
            covered |= flagged
    return need


def structural_companions(features, X):
    """Structural indicators that must enter a model alongside `features`.

    Structurally missing values are constant-filled (make_imputer), so a feature's
    coefficient is identified from observed rows only if the indicator marking the
    filled rows is in the same model (computed from the unimputed X). Returns the
    indicators not already in `features`, in a stable order.
    """
    need = []
    for f in features:
        for ind in _companions_for(f, X):
            if ind not in features and ind not in need:
                need.append(ind)
    return need


def with_structural_indicators(features, X):
    """`features` plus the structural indicators they require."""
    return list(features) + structural_companions(features, X)


def drop_redundant_features(X):
    """Return X without the columns in DROP_REDUNDANT_FEATURES.

    Names resolve by friendly label: engineered features carry their real column
    name; base features are positional 'Feature N' placeholders mapped via the
    canonical name list. Absent/unknown names are skipped silently.
    """
    cols = set(X.columns)
    drop = []
    for f in DROP_REDUNDANT_FEATURES:
        c = f if f in cols else _FRIENDLY_TO_COL.get(f)
        if c in cols:
            drop.append(c)
    return X.drop(columns=drop)


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

    The non-firings split on the labels' explicit `active` flag: a stint still
    running at the start of the season after the boundary is administrative
    censoring; every other non-firing is a documented voluntary departure (the
    competing risk), including one that happened in the boundary season itself.
    """
    labels = _firing_event_labels()
    active = _active_labels()
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
            cause[idx] = CENSORED if active.get((c, last), last >= boundary) else VOLUNTARY
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

    Returns the fitted AalenJohansenFitter; read it with cif_at. Durations are integer
    seasons, so exits tie. lifelines breaks ties by jittering the EVENT times only
    (fixed seed): harmless among events, but a firing jittered just past a censoring
    in the same season sees a risk set short by that censoring. Censored (cause 0)
    durations are therefore shifted by +0.5 before fitting, the usual convention that
    events precede censorings at a tied time. It is also literally right here: a coach
    active at the boundary was observed through the whole of his last season. With the
    shift the estimate equals the exact tie-aware Aalen-Johansen calculation.
    """
    dur = np.asarray(dur, float)
    cause = np.asarray(cause, int)
    dur = np.where(cause == CENSORED, dur + 0.5, dur)
    ajf = AalenJohansenFitter(seed=seed, calculate_variance=True)
    ajf.fit(dur, cause, event_of_interest=event_of_interest)
    return ajf


def cif_at(frame, t, col=0):
    """Value of an Aalen-Johansen step function (cumulative_density_ or
    confidence_interval_) through the END of integer season t.

    The jittered event times sit at t +/- epsilon, so reading at exactly t would miss
    about half of season t. Reading at t + 0.5 captures every exit with integer time
    <= t and none from season t + 1. (Censorings shifted to t + 0.5 are not jumps of
    the CIF, so including them does not change the value.)"""
    m = frame.index <= t + 0.5
    return float(frame[m].iloc[-1, col]) if m.any() else 0.0


# --------------------------------------------------------------------------- #
# Final cause-specific Cox with cluster-robust SEs -> hazard ratios
# --------------------------------------------------------------------------- #
def cox_hazard_ratios(Xsel_df, dur, cause, coach_ids,
                      penalizer=0.1, l1_ratio=0.0, strata=None):
    """Cause-specific Cox for FIRING on the selected (already imputed) features.

    Inference is what we report, so we use a small RIDGE (stable with correlated
    features, keeps valid SEs -- unlike lasso) and CLUSTER-ROBUST sandwich SEs on
    coach id, since coaches recur across stints and violate independence.
    `strata` (one label per row) gives each stratum its own baseline hazard.
    Returns (summary_df, fitted_model). summary_df has HR, 95% CI, robust p.
    """
    d = Xsel_df.copy()
    d["T"] = np.asarray(dur, float)
    d["E"] = firing_event(cause)
    d["coach"] = np.asarray(coach_ids)
    if strata is not None:
        d["stratum"] = np.asarray(strata)
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(d, duration_col="T", event_col="E", cluster_col="coach", robust=True,
            strata=None if strata is None else ["stratum"])
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
# Inference sample for models outside the selection pipeline
# --------------------------------------------------------------------------- #
def inference_sample():
    """Every stint with a known duration, prepared as the hazard-ratio path prepares it.

    Returns (df, X, Xf, dur, cause): X is the unimputed feature matrix (which
    structural_companions reads), Xf the same matrix filled by make_imputer on the full
    sample (a structural constant fill; nothing is estimated), dur and cause from
    build_competing_targets. All share one index.
    """
    df, X, _ = load_modeling_data(known_only=False)
    dur, cause = build_competing_targets(df, global_max_season())
    keep = dur.index
    df, X, dur, cause = df.loc[keep], X.loc[keep], dur.loc[keep], cause.loc[keep]
    Xf = pd.DataFrame(make_imputer().fit(X.values).transform(X.values),
                      columns=X.columns, index=X.index)
    return df, X, Xf, dur, cause


def stint_franchise(df):
    """Canonical franchise key of each stint's hiring team, for franchise strata.
    Relocation-aware (the Oilers and Titans are one franchise). As PFR keys them, the
    Browns are one franchise across their 1996-98 suspension and the Ravens a new one
    from 1996 (unlike the predecessor features, which give the 1996 Ravens the 1995
    Browns' history: INHERITED_FRANCHISES)."""
    keys = []
    for c, y in zip(df["Coach Name"], df["Year"]):
        name, level = hiring_team_info(load_history(c), int(y))
        keys.append(canon_employer(name, level, int(y)))
    return pd.Series(keys, index=df.index)


# --------------------------------------------------------------------------- #
# Calibration: IPCW (Graf) Brier score + integrated Brier score
# --------------------------------------------------------------------------- #
def _km_censoring(dur_train, evt_train):
    """KM estimate of the CENSORING survival function G(t)=P(C>t) from training
    (event indicator flipped). Used for IPCW weights in the Brier score."""
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(dur_train, float), 1 - np.asarray(evt_train, int))
    return kmf


def _G(kmf_cens, t, left=False, floor=1e-8):
    """Training censoring survival G(t) = P(C > t), or with `left` its left limit
    G(t-) = P(C >= t): the probability a stint ending at t was still uncensored when
    it ended, which is the IPCW weight an observed firing at t needs. Exact step-
    function read (no interpolation), floored at `floor` to avoid division by 0."""
    sf = kmf_cens.survival_function_
    k = np.searchsorted(sf.index.values, t, side="left" if left else "right") - 1
    g = float(sf.values[k, 0]) if k >= 0 else 1.0
    return max(g, floor)


def ipcw_brier(t, surv_t, dur_test, evt_test, kmf_cens):
    """Graf (1999) IPCW Brier score at horizon t.

    surv_t[i] = model's predicted P(T_i > t). evt_test = firing indicator.
    Subjects fired by t are weighted 1/G(T_i-) (left limit: a censoring in the same
    season as the firing did not precede it); subjects surviving past t by 1/G(t);
    subjects censored before t contribute 0 (their weight is undefined).
    """
    dur_test = np.asarray(dur_test, float)
    evt_test = np.asarray(evt_test, int)
    surv_t = np.asarray(surv_t, float)
    bs = 0.0
    for i in range(len(dur_test)):
        Ti, di, Si = dur_test[i], evt_test[i], surv_t[i]
        if Ti <= t and di == 1:
            bs += (0.0 - Si) ** 2 / _G(kmf_cens, Ti, left=True)
        elif Ti > t:
            bs += (1.0 - Si) ** 2 / _G(kmf_cens, t)
        # censored before t -> contributes 0
    return bs / len(dur_test)


def integrated_brier(times, surv_matrix, dur_test, evt_test, kmf_cens):
    """Integrated Brier score over `times` (trapezoidal). surv_matrix is
    n_test x len(times) of predicted P(T>t). Lower is better; the reference to beat
    is the covariate-free Kaplan-Meier curve (km_null_brier), not 0.25."""
    times = np.asarray(times, float)
    bs = np.array([ipcw_brier(times[j], surv_matrix[:, j], dur_test, evt_test,
                              kmf_cens) for j in range(len(times))])
    return float(np.trapz(bs, times) / (times[-1] - times[0])), bs


def cox_discrete_hazards(cph, X):
    """Per-season discrete hazard h(s|x) = min(1, dH0(s) exp((x - xbar) b)), one row per
    row of X (a DataFrame with the model's columns), on the baseline's season grid.
    Returns (seasons, n x T).

    Survival is then the product-limit prod_{s<=t} (1 - h(s|x)), not lifelines'
    exp(-H(t|x)). Exits are counted in whole seasons, so they are heavily tied, and
    exp(-H) runs above the Kaplan-Meier curve (S(4) 0.38 vs 0.33 on one training
    partition). The product-limit form reduces exactly to Kaplan-Meier when b = 0, so a
    Cox model is scored against the KM null like for like."""
    dH0 = cph.baseline_hazard_.iloc[:, 0]                      # at the covariate means
    r = np.asarray(cph.predict_partial_hazard(X[cph.params_.index]), float)  # centered
    return dH0.index.values.astype(float), np.minimum(1.0, np.outer(r, dH0.values))


def cox_brier(params, Xtr, Ttr, Etr, Xte, Tte, Ete, grid, kmf_cens):
    """(integrated Brier, Brier-by-horizon) of a Cox model fit on one training
    partition and scored on its held-out partition, from product-limit survival
    (cox_discrete_hazards). Columns constant in the training partition (a rare flag)
    are left out of the fit and of the prediction."""
    keep = varying_columns(Xtr)
    cols = [f"f{i}" for i in range(int(keep.sum()))]
    d = pd.DataFrame(np.asarray(Xtr)[:, keep], columns=cols)
    d["T"], d["E"] = Ttr, Etr
    m = CoxPHFitter(penalizer=params.get("penalizer", 0.1),
                    l1_ratio=params.get("l1_ratio", 0.0)).fit(d, "T", "E")
    seasons, h = cox_discrete_hazards(
        m, pd.DataFrame(np.asarray(Xte)[:, keep], columns=cols))
    surv = np.cumprod(1.0 - h, axis=1)
    k = np.searchsorted(seasons, np.asarray(grid, float), side="right") - 1
    sf = np.where(k >= 0, surv[:, np.clip(k, 0, None)], 1.0)  # through the end of t
    return integrated_brier(grid, sf, Tte, Ete, kmf_cens)


def km_null_brier(Ttr, Etr, Tte, Ete, grid, kmf_cens):
    """(integrated Brier, Brier-by-horizon) of the covariate-free null: the training
    partition's Kaplan-Meier curve predicted for every held-out stint. This is the
    reference a calibrated model has to beat. 0.25 (a coin flip at S = 0.5) is not:
    the marginal firing curve is already known, and the KM null scores well below
    0.25 at short and long horizons where S(t) is far from 0.5."""
    km = KaplanMeierFitter().fit(Ttr, Etr)
    s_km = np.array([float(km.predict(g)) for g in grid])
    return integrated_brier(grid, np.tile(s_km, (len(Tte), 1)), Tte, Ete, kmf_cens)


# --------------------------------------------------------------------------- #
# Uno's censoring-robust C-index
# --------------------------------------------------------------------------- #
def uno_cindex(risk, dur_test, evt_test, kmf_cens, tau=UNO_TAU):
    """Uno et al. (2011) IPCW concordance, truncated at tau. `risk` higher = shorter
    survival (e.g. Cox partial hazard). Comparable pairs are those where the earlier
    time T_i <= tau is a firing; each is weighted 1/G(T_i-)^2, the left limit of the
    training censoring survival. Complements Harrell's C (which is biased under heavy
    censoring; here censoring is light so the two should nearly agree -- reporting
    both pre-empts the objection).

    tau is a FIXED horizon (UNO_TAU, the last Brier horizon) passed identically in
    every split, not the largest test event time: beyond the last training time with
    G > 0 the weight is undefined, and flooring G there let a single pair decide a
    whole split (two of 50 Cox splits scored ~0). As a guard, if the training G is 0
    at tau, tau is lowered to the largest time at which G is still positive.
    """
    dur_test = np.asarray(dur_test, float)
    evt_test = np.asarray(evt_test, int)
    risk = np.asarray(risk, float)
    if _G(kmf_cens, tau, floor=0.0) <= 0:
        sf = kmf_cens.survival_function_.iloc[:, 0]
        tau = float(sf.index[sf.values > 0].max())
    num = den = 0.0
    for i in np.flatnonzero((evt_test == 1) & (dur_test <= tau)):
        w = 1.0 / _G(kmf_cens, dur_test[i], left=True) ** 2
        later = dur_test > dur_test[i]
        den += w * later.sum()
        num += w * ((risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum())
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
        out["uno"].append(uno_cindex(-score, Tte, Ete, kmf, tau=UNO_TAU))
    return out


def cox_calibration(df, X, y, dur, evt, idx, params, grid, n_seeds):
    """Per-seed IPCW integrated Brier score for the Cox primary model AND for the
    covariate-free Kaplan-Meier null on the same splits (the paired reference), plus
    each one's mean Brier-by-horizon curve (for the calibration figure).

    Returns {"ibs", "ibs_km"} (per seed) and {"curve", "curve_km"} (mean over seeds).
    """
    ibs, ibs_km, bs, bs_km = [], [], [], []
    for seed in range(n_seeds):
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)
        Ttr = dur.loc[split.train_index].values.astype(float)
        Etr = evt.loc[split.train_index].values.astype(int)
        Tte = dur.loc[split.test_index].values.astype(float)
        Ete = evt.loc[split.test_index].values.astype(int)
        kmf = _km_censoring(Ttr, Etr)
        i, b = cox_brier(params, split.X_train, Ttr, Etr, split.X_test, Tte, Ete, grid, kmf)
        ik, bk = km_null_brier(Ttr, Etr, Tte, Ete, grid, kmf)
        ibs.append(i); bs.append(b)
        ibs_km.append(ik); bs_km.append(bk)
    return {"ibs": np.array(ibs), "ibs_km": np.array(ibs_km),
            "curve": np.mean(bs, axis=0), "curve_km": np.mean(bs_km, axis=0)}


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
    keep = varying_columns(Ximp)   # a column constant in this subsample ranks last
    Z = (Ximp[:, keep] - Ximp[:, keep].mean(0)) / Ximp[:, keep].std(0)
    d = pd.DataFrame(Z, columns=[f"f{i}" for i in range(Z.shape[1])])
    d["T"], d["E"] = np.asarray(T, float), np.asarray(E, int)
    m = CoxPHFitter(penalizer=penalizer).fit(d, "T", "E")
    imp = np.zeros(Ximp.shape[1])
    imp[keep] = np.abs(m.params_.values)
    return np.argsort(imp)[::-1], imp


def aft_importance(fitter_cls, Ximp, T, E, penalizer=0.1):
    """An AFT model's own feature ranking, the counterpart of cox_importance:
    |standardized coefficient| on the location parameter (Weibull lambda_, log-normal
    mu_, log-logistic alpha_) of a ridge fit with every column, z-scored."""
    Ximp = np.asarray(Ximp, float)
    keep = varying_columns(Ximp)
    Z = (Ximp[:, keep] - Ximp[:, keep].mean(0)) / Ximp[:, keep].std(0)
    cols = [f"f{i}" for i in range(Z.shape[1])]
    d = pd.DataFrame(Z, columns=cols)
    d["T"], d["E"] = np.asarray(T, float), np.asarray(E, int)
    m = fitter_cls(penalizer=penalizer).fit(d, "T", "E")
    imp = np.zeros(Ximp.shape[1])
    imp[keep] = np.abs(m.params_.loc[m._primary_parameter_name].reindex(cols).values)
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
    from model.pipeline import make_imputer
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
        Ximp = make_imputer().fit(X.values[mask]).transform(X.values[mask])
        rank, _ = importance_fn(Ximp, Tv[mask], Ev[mask])
        rank = selectable(rank, cols)
        for i in rank[:K]:
            counts[i] += 1
    return pd.Series(counts / n_boot, index=cols).sort_values(ascending=False)


def stability_selection_multi(df, X, dur, evt, Ks=(10, 15, 20), xgb_params=None,
                              n_boot=100, subsample=0.5, seed=0, importance_fn=None):
    """Stability selection recording top-K membership for SEVERAL K from the SAME
    subsamples in ONE bootstrap pass.

    Identical procedure to stability_selection (coach-level subsampling, per-sample
    re-imputation + re-ranking), but each subsample's full rank is scored against
    every K in ``Ks`` at once. This makes the K-sensitivity apples-to-apples (same
    draws) and costs one pass instead of len(Ks) passes. Returns {K: freq_series}.
    """
    from model.pipeline import make_imputer
    if importance_fn is None:
        from scripts.survival_feature_selection import survival_importance
        def importance_fn(Z, T, E):
            return survival_importance(Z, T, E, xgb_params)

    rng = np.random.default_rng(seed)
    coaches = df["Coach Name"].unique()
    cols = list(X.columns)
    counts = {K: np.zeros(len(cols)) for K in Ks}
    Tv = dur.values.astype(float)
    Ev = evt.values.astype(int)
    n_take = max(1, int(len(coaches) * subsample))
    for _ in range(n_boot):
        samp = rng.choice(coaches, size=n_take, replace=False)
        mask = df["Coach Name"].isin(samp).values
        Ximp = make_imputer().fit(X.values[mask]).transform(X.values[mask])
        rank, _ = importance_fn(Ximp, Tv[mask], Ev[mask])
        rank = selectable(rank, cols)
        for K in Ks:
            for i in rank[:K]:
                counts[K][i] += 1
    return {K: pd.Series(counts[K] / n_boot, index=cols).sort_values(ascending=False)
            for K in Ks}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def main():
    from model.pipeline import load_modeling_data
    from scripts.survival_analysis import global_max_season

    from scripts.survival_analysis import build_survival_targets

    # every stint, active coaches included (the survival sample; see survival_definitive)
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X = df.loc[keep], X.loc[keep]
    counts = cause.value_counts().to_dict()
    print(f"boundary={boundary} | n={len(dur)} | "
          f"fired={counts.get(FIRED,0)} active-censored={counts.get(CENSORED,0)} "
          f"voluntary={counts.get(VOLUNTARY,0)}")
    # consistency, not hard-coded counts: three causes partition the sample, and the
    # binary target used for prediction agrees with the competing-risks coding
    assert sum(counts.values()) == len(dur) and set(counts) <= {CENSORED, FIRED, VOLUNTARY}
    durb, evtb = build_survival_targets(df, boundary)
    assert (durb.loc[keep] == dur).all(), "duration drift between target builders"
    assert (evtb.loc[keep].values == firing_event(cause)).all(), "firing label drift"

    # CIF: firing vs voluntary, through the end of each season. The two CIFs must add
    # up to one minus the Kaplan-Meier curve of any exit, which holds only when ties
    # are handled exactly (aalen_johansen_cif's censoring shift).
    aj_fire = aalen_johansen_cif(dur, cause, FIRED)
    aj_vol = aalen_johansen_cif(dur, cause, VOLUNTARY)
    km_any = KaplanMeierFitter().fit(dur.values, (cause != CENSORED).values)
    for t in (2, 4, 6, 8):
        cf = cif_at(aj_fire.cumulative_density_, t)
        cv = cif_at(aj_vol.cumulative_density_, t)
        gap = cf + cv - (1.0 - float(km_any.predict(t)))
        print(f"  by season {t}: CIF(fired)={cf:.4f}  CIF(voluntary)={cv:.4f}  "
              f"(sum - (1-KM_any) = {gap:+.1e})")
        assert abs(gap) < 1e-9, "Aalen-Johansen ties mishandled"
    print("survival_methods self-test OK")


if __name__ == "__main__":
    main()
