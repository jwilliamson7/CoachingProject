#!/usr/bin/env python
"""
Engineer additional career-path (Tier 1) and rank recency/trajectory (Tier 2)
features for NFL head-coach tenure prediction.

Motivation: after fixing the imputation leak, the existing 150 features predict
tenure only weakly (QWK ~0.29), and the surviving signal is concentrated in the
always-present experience features. The current pipeline compresses each coach's
full career history (Coaches/<name>/all_coaching_history.csv) into 8 counters and
averages away the per-season ranks (all_coaching_ranks.csv). This script recovers
the discarded structure.

CRITICAL (no leakage): every feature for a hire instance (coach, hire_year) is
computed using ONLY seasons strictly before hire_year. The hire-year row is used
solely to identify the hiring franchise (the team where the coach's role that year
is Head Coach), which is information available at hiring time.

Output: data/master_data_extended.csv -- the original master_data.csv with the new
feature columns inserted immediately before 'Avg 2Y Win Pct' and 'Coach Tenure
Class', so the existing pipeline (features = columns 2:-2) picks them up
automatically.

Usage:
    python scripts/data/engineer_career_features.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

COACHES_DIR = project_root / "Coaches"
# Single canonical modeling dataset, written by scripts/data/create_data.py (which
# builds the base instances then calls build_modeling_dataset()). There is no longer
# a separate raw/_extended split -- one file, one build path.
MASTER_DATA = project_root / "data" / "master_data.csv"
# Tier 6 inherited-roster/talent source (another project; team-year keyed, 1970-2024)
COACH_WAR_COMBINED = Path(
    r"C:\Users\jonwi\Documents\Projects\Coach_WAR\data\final\combined_final_dataset.csv"
)

# Single source of truth for role exclusions: the same list create_data uses (via
# data_constants). Previously this module kept its OWN copy that had drifted (it was
# missing "Passing Game Coordinator", "Pass Gm. Coord." and "Associate Head Coach"),
# so the base-builder and the feature-engineer disagreed on 28 role-seasons. Import,
# don't duplicate, so the two classifiers can never drift again.
from data_constants import EXCLUDED_ROLE_KEYWORDS

OFF_KW = ["offensive", "quarterback", "running back", "wide receiver", "tight end",
          "o-line", "passing game", "pass game"]
DEF_KW = ["defensive", "linebacker", "defensive back", "secondary", "cornerback",
          "safet", "d-line", "nickel", "pass rush", "run game coordinator/def"]


def classify_role(role):
    """Return one of: HC, OC, DC, STC, POS, NONE (mirrors create_data logic)."""
    if not isinstance(role, str) or not role.strip():
        return "NONE"
    for kw in EXCLUDED_ROLE_KEYWORDS:
        if kw in role:
            return "NONE"
    if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
        return "NONE"
    if "Head Coach" in role and "Ass" not in role and "Interim" not in role:
        return "HC"
    if "Coordinator" in role:
        if "Offensive Coordinator" in role and "Interim O" not in role:
            return "OC"
        if "Defensive Coordinator" in role and "Interim D" not in role:
            return "DC"
        if "Special" in role and "Interim S" not in role:
            return "STC"
        return "POS"
    return "POS"


def role_side(role):
    """Return 'off', 'def', or 'other' based on the role string."""
    r = str(role).lower()
    is_off = any(k in r for k in OFF_KW)
    is_def = any(k in r for k in DEF_KW)
    if is_off and not is_def:
        return "off"
    if is_def and not is_off:
        return "def"
    return "other"


# Relocation/rename handling for employer counts -- consistent with Coach_WAR.
#
# Franchise identity is resolved exactly as the Coach_WAR project does, via
# data_constants.standardize_team_abbreviation(abbr, year): a (team-abbreviation,
# season-year) pair maps to a canonical PFR franchise key, with year-based logic
# for the abbreviations that meant two franchises across eras (BAL, HOU, STL).
# Coach histories store full team NAMES, so we bridge full-name -> abbreviation
# data-drivenly by pairing each season's history Employer with the abbreviation
# the ranks file uses for that same coach-year, then apply the year-based
# standardization. A nickname fallback covers the rare name never seen in ranks.
_NICK_RENAME = {"oilers": "titans", "redskins": "commanders"}


def build_name_to_abbr():
    """Full team NAME -> ranks abbreviation, learned by pairing history Employer
    with the ranks 'Tm' for the same coach-year. Abbreviations stay era-ambiguous
    (e.g. 'BAL'); standardize_team_abbreviation resolves them by year."""
    from collections import Counter
    n2a = {}
    for d in COACHES_DIR.iterdir():
        if not d.is_dir():
            continue
        h, r = load_history(d.name), load_ranks(d.name)
        if h is None or r is None:
            continue
        yr2abbr = {}
        for _, rr in r.iterrows():
            yr2abbr.setdefault(int(rr["Year"]), str(rr.get("Tm", "")).strip())
        for _, hr in h.iterrows():
            if is_nfl(hr.get("Level")) and int(hr["Year"]) in yr2abbr:
                nm = str(hr.get("Employer", "")).strip().lower()
                ab = yr2abbr[int(hr["Year"])]
                if nm and ab:
                    n2a.setdefault(nm, Counter())[ab] += 1
    return {nm: c.most_common(1)[0][0] for nm, c in n2a.items()}


_NAME2ABBR = None


def name2abbr():
    global _NAME2ABBR
    if _NAME2ABBR is None:
        _NAME2ABBR = build_name_to_abbr()
    return _NAME2ABBR


def canon_employer(employer, level, year=None):
    """Canonical franchise key so relocations/renames count as one franchise."""
    from data_constants import standardize_team_abbreviation
    name = str(employer).strip().lower()
    if is_nfl(level):
        abbr = name2abbr().get(name)
        if abbr:
            return "nfl:" + standardize_team_abbreviation(abbr, year)
        # fallback: nickname (Colts/Raiders/etc. keep nickname across relocations)
        if "football team" in name:
            return "nfl:commanders"
        toks = name.replace(".", "").split()
        nick = toks[-1] if toks else name
        return "nfl:" + _NICK_RENAME.get(nick, nick)
    return ("col:" if "College" in str(level) else "oth:") + name


def is_nfl(level):
    s = str(level)
    return ("NFL" in s and s != "NFL Europe")


def _slope(values):
    """Least-squares slope of values vs 0..n-1.

    Returns NaN when fewer than 2 valid points exist: a single-season coach has
    an UNDEFINED trajectory, which must not be conflated with a genuinely flat
    (slope 0) multi-season trajectory. NaN is then imputed downstream.
    """
    v = [x for x in values if x == x]  # drop NaN
    n = len(v)
    if n < 2:
        return np.nan
    x = np.arange(n)
    return float(np.polyfit(x, np.asarray(v, float), 1)[0])


def _dedupe_history(df):
    """Collapse PFR sub-role duplicate rows so the whole module sees one season as
    one row -- matching create_data._dedupe_career_rows. PFR sometimes lists a
    single season as two rows with different role text (e.g. 'Defensive
    Coordinator' and 'Defensive Coordinator/Linebackers'); without this, row-count
    features such as cf_nfl_share double-count those seasons. Keep one row per
    (Year, level-class, role-class); same-year role CHANGES (different role-class)
    and non-coaching rows are preserved. reconstruct_tenure is set-based and so is
    unaffected -- this only de-duplicates, never drops a distinct season.
    """
    if df is None or df.empty:
        return df
    def lvl_tag(l):
        return "nfl" if is_nfl(l) else ("col" if "College" in str(l) else "oth")
    seen, keep = set(), []
    for idx, r in df.iterrows():
        role = classify_role(r.get("Role", ""))
        if role == "NONE":
            keep.append(idx)
            continue
        key = (int(r["Year"]), lvl_tag(r.get("Level")), role)
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    return df.loc[keep]


def load_history(coach):
    p = COACHES_DIR / coach / "all_coaching_history.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        return _dedupe_history(df)
    except Exception:
        return None


def load_ranks(coach):
    p = COACHES_DIR / coach / "all_coaching_ranks.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        return df
    except Exception:
        return None


def _pctl_from_rank(rank, n_teams):
    """Convert a rank (1=best) to a performance percentile in [0,1] (1=best)."""
    try:
        r = float(rank); n = float(n_teams)
        if not (r == r) or not (n == n) or n <= 1:
            return np.nan
        return 1.0 - (r - 1.0) / (n - 1.0)
    except Exception:
        return np.nan


# cf_hire_year dropped: within the 1970+ window its corr with tenure is ~0
# (the era signal was a pre/post-merger artifact that the filter already removes).
TIER1_KEYS = ["cf_num_nfl_employers", "cf_num_college_employers",
              "cf_nfl_share", "cf_age_first_job", "cf_age_first_nfl",
              "cf_gap_before_hire", "cf_internal_hire", "cf_years_at_hiring_org",
              "cf_pre_OC", "cf_pre_DC", "cf_pre_POS", "cf_pre_HC", "cf_pre_side_off",
              "cf_pre_side_def", "cf_pre_level_nfl",
              # Structural-missingness indicators: did the coach EVER hold an NFL
              # OC/DC role? If 0, the __oc / __dc performance blocks are
              # structurally absent (not missing-at-random) and any imputed values
              # there are meaningless -- the flag lets the model condition on that.
              # (An NFL-HC indicator already exists: Feature 2 num_times_hc / Feature
              # 8 num_yr_nfl_hc.) Addresses the JQAS missing-data critique.
              "cf_ever_nfl_oc", "cf_ever_nfl_dc"]

# Restrict the modeling population to the modern era (AFL-NFL merger onward).
MIN_HIRE_YEAR = 1970


def tier1_features(hist, hire_year):
    """Career-path features from history rows strictly before hire_year.

    Redundant-with-existing or collinear features removed (ever-NFL-HC ~ Feature 2;
    ever-college-HC ~ Feature 5; total_seasons/career_span collinear with the six
    year-count features). Employer counts are relocation-aware and split into NFL
    and college; total employers dropped as their sum.
    """
    f = {k: np.nan for k in TIER1_KEYS}

    pre = hist[hist["Year"] < hire_year].copy().sort_values("Year")
    pre["rtype"] = pre["Role"].apply(classify_role)
    coaching = pre[pre["rtype"] != "NONE"]
    if len(coaching) == 0:
        return f

    years = coaching["Year"].values
    nfl_rows = coaching[coaching["Level"].apply(is_nfl)]
    col_rows = coaching[coaching["Level"].astype(str).str.contains("College")]

    # structural-missingness indicators for the OC / DC performance blocks
    f["cf_ever_nfl_oc"] = int(any(classify_role(rr) == "OC" for rr in nfl_rows["Role"]))
    f["cf_ever_nfl_dc"] = int(any(classify_role(rr) == "DC" for rr in nfl_rows["Role"]))

    # Relocation-aware employer counts (year-based franchise identity for NFL)
    nfl_keys = {canon_employer(e, l, y)
                for e, l, y in zip(nfl_rows["Employer"], nfl_rows["Level"], nfl_rows["Year"])}
    col_keys = {canon_employer(e, l, y)
                for e, l, y in zip(col_rows["Employer"], col_rows["Level"], col_rows["Year"])}
    f["cf_num_nfl_employers"] = int(len(nfl_keys))
    f["cf_num_college_employers"] = int(len(col_keys))
    f["cf_nfl_share"] = len(nfl_rows) / len(coaching)

    if "Age" in coaching.columns:
        ages = pd.to_numeric(coaching["Age"], errors="coerce")
        f["cf_age_first_job"] = float(ages.min()) if ages.notna().any() else np.nan
        nfl_ages = pd.to_numeric(nfl_rows["Age"], errors="coerce") if len(nfl_rows) else pd.Series(dtype=float)
        # NaN is meaningful: coach had no prior NFL coaching (college-to-NFL jump)
        f["cf_age_first_nfl"] = float(nfl_ages.min()) if len(nfl_ages) and nfl_ages.notna().any() else np.nan
    f["cf_gap_before_hire"] = hire_year - int(years.max()) - 1

    # hiring franchise: employer in the hire-year row whose role is Head Coach
    hire_row = hist[hist["Year"] == hire_year]
    hiring_team = hiring_level = None
    for _, r in hire_row.iterrows():
        if classify_role(r.get("Role", "")) == "HC":
            hiring_team, hiring_level = str(r.get("Employer", "")).strip(), r.get("Level", "")
            break
    if hiring_team is None and len(hire_row):
        hiring_team = str(hire_row.iloc[0].get("Employer", "")).strip()
        hiring_level = hire_row.iloc[0].get("Level", "")
    hiring_key = canon_employer(hiring_team, hiring_level, hire_year) if hiring_team else None

    # internal hire: employed by the hiring franchise in the IMMEDIATE prior season
    prior = coaching[coaching["Year"] == hire_year - 1]
    prior_keys = {canon_employer(e, l, y)
                  for e, l, y in zip(prior["Employer"], prior["Level"], prior["Year"])}
    f["cf_internal_hire"] = int(hiring_key in prior_keys) if hiring_key else 0
    # consecutive seasons at the hiring franchise immediately before the hire
    yrs_at_org = 0
    if hiring_key:
        org_years = {int(y) for e, l, y in zip(coaching["Employer"], coaching["Level"], coaching["Year"])
                     if canon_employer(e, l, y) == hiring_key}
        yr = hire_year - 1
        while yr in org_years:
            yrs_at_org += 1
            yr -= 1
    f["cf_years_at_hiring_org"] = yrs_at_org

    # most recent prior coaching role
    last = coaching[coaching["Year"] == int(years.max())].iloc[-1]
    rt, side = last["rtype"], role_side(last["Role"])
    f["cf_pre_OC"] = int(rt == "OC")
    f["cf_pre_DC"] = int(rt == "DC")
    f["cf_pre_POS"] = int(rt == "POS")
    f["cf_pre_HC"] = int(rt == "HC")
    f["cf_pre_side_off"] = int(side == "off")
    f["cf_pre_side_def"] = int(side == "def")
    f["cf_pre_level_nfl"] = int(is_nfl(last["Level"]))
    return f


# Rank columns: offense block has 'Pts' (points scored); defense block has
# 'Pts_2' (points allowed); 'Pts±' is net point differential; lower rank = better.
def _unit_col(role):
    r = str(role).upper()
    if "OC" in r:
        return "Pts"      # offensive points scored rank
    if "DC" in r:
        return "Pts_2"    # defensive points allowed rank
    return "Pts±"   # HC -> net point differential rank


def tier2_features(ranks, hire_year):
    """Side-appropriate unit performance + team success from prior ranked seasons.

    For each prior season the coach's OWN unit is used: offensive points rank when
    they were OC, DEFENSIVE points-allowed rank when DC, net point differential
    when HC. Ranks are converted to percentiles (1=best) normalized by league size.
    """
    f = {k: np.nan for k in ["rf_final_unit_pctl", "rf_avg_unit_pctl",
                             "rf_unit_traj", "rf_final_winpct_pctl",
                             "rf_avg_winpct_pctl", "rf_n_seasons_rank"]}
    if ranks is None:
        f["rf_n_seasons_rank"] = 0
        return f
    pre = ranks[ranks["Year"] < hire_year].sort_values("Year")
    f["rf_n_seasons_rank"] = int(len(pre))
    if len(pre) == 0:
        return f

    tms = pd.to_numeric(pre.get("Tms"), errors="coerce").tolist()

    # team win% percentile
    wl_rk = pd.to_numeric(pre.get("WL%"), errors="coerce").tolist()
    wl = [_pctl_from_rank(r, n) for r, n in zip(wl_rk, tms)]
    wl = [x for x in wl if x == x]
    if wl:
        f["rf_final_winpct_pctl"] = wl[-1]
        f["rf_avg_winpct_pctl"] = float(np.mean(wl))

    # side-appropriate unit percentile (per-season, by that season's role)
    unit = []
    for (_, row), n in zip(pre.iterrows(), tms):
        col = _unit_col(row.get("Role", ""))
        rk = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        unit.append(_pctl_from_rank(rk, n))
    unit_valid = [x for x in unit if x == x]
    if unit_valid:
        f["rf_final_unit_pctl"] = unit_valid[-1]
        f["rf_avg_unit_pctl"] = float(np.mean(unit_valid))
        f["rf_unit_traj"] = _slope([x for x in unit[-3:] if x == x])
    return f


# ----------------------------------------------------------------------------
# Tier 5: organizational instability (hiring franchise's coaching history)
# Tier 6: inherited roster/talent (hiring franchise at season hire_year - 1)
# ----------------------------------------------------------------------------

def hiring_team_info(hist, hire_year):
    """Return (employer_name, level) for the hire-year Head Coach row."""
    hire_row = hist[hist["Year"] == hire_year]
    for _, r in hire_row.iterrows():
        if classify_role(r.get("Role", "")) == "HC":
            return str(r.get("Employer", "")).strip(), r.get("Level", "")
    if len(hire_row):
        return str(hire_row.iloc[0].get("Employer", "")).strip(), hire_row.iloc[0].get("Level", "")
    return None, None


def reconstruct_tenure(coach, hire_year):
    """Relocation-aware tenure = consecutive seasons the coach is HC of the hiring
    franchise starting at hire_year. A coach who reaches season N (even via a
    mid-season exit) has tenure in (N-1, N], so this count, binned with the
    standard thresholds, correctly handles partial seasons (e.g. 2.5 -> class 1)
    and spans relocations/renames (Oilers->Titans, etc.). Returns None if unknown.
    """
    h = load_history(coach)
    if h is None:
        return None
    name, level = hiring_team_info(h, hire_year)
    if not name:
        return None
    fkey = canon_employer(name, level, hire_year)
    hc_years = set()
    for _, r in h.iterrows():
        y = int(r["Year"])
        if (y >= hire_year and is_nfl(r.get("Level"))
                and classify_role(r.get("Role", "")) == "HC"
                and canon_employer(r.get("Employer", ""), r.get("Level"), y) == fkey):
            hc_years.add(y)
    t, y = 0, hire_year
    while y in hc_years:
        t += 1
        y += 1
    return t if t > 0 else None


def tenure_class(n):
    """Bin a (relocation/partial-aware) season count into the ordinal class."""
    return 0 if n <= 2 else (1 if n <= 4 else 2)


# A coach who leaves the profession at or after this age is treated as a
# retirement (right-censored), not a firing. Sensitivity-checked in the pivot.
DEFAULT_RETIRE_AGE = 62


def _age_in_season(h, year):
    """Coach's age in a given season from the history Age column (NaN if absent)."""
    for _, r in h[h["Year"] == year].iterrows():
        a = r.get("Age")
        if pd.notna(a):
            try:
                return int(a)
            except (ValueError, TypeError):
                return np.nan
    return np.nan


def classify_exit(coach, hire_year, boundary, retire_age=DEFAULT_RETIRE_AGE):
    """Tenure duration plus a firing-vs-retirement event label for survival.

    Addresses the JQAS objection to lumping firings and retirements. Returns
    dict(duration, event, exit_type, age_at_exit, coached_after) or None.

    event = 1 marks an involuntary exit (FIRING) -- the event of interest;
    event = 0 marks a right-censored observation (voluntary retirement, or a
    coach still active at the data boundary). A coach who held ANY coaching role
    after leaving the head-coaching job did not retire then, so that HC exit is a
    firing/dismissal. A coach who never coached again is a retirement only if old
    enough (>= retire_age); otherwise it is a firing that pushed him out of the
    league. Still active at the boundary -> censored.
    """
    t = reconstruct_tenure(coach, hire_year)
    if t is None:
        return None
    h = load_history(coach)
    last = hire_year + t - 1
    age_last = _age_in_season(h, last)
    if np.isnan(age_last):
        a0 = _age_in_season(h, hire_year)
        age_last = a0 + (last - hire_year) if not np.isnan(a0) else np.nan

    coached_after = any(
        int(r["Year"]) > last and classify_role(r.get("Role", "")) != "NONE"
        for _, r in h.iterrows())

    if last >= boundary:
        event, exit_type = 0, "active"
    elif coached_after:
        event, exit_type = 1, "fired_or_moved"
    elif not np.isnan(age_last) and age_last >= retire_age:
        event, exit_type = 0, "retired"
    else:
        event, exit_type = 1, "fired_out"

    return {"duration": t, "event": event, "exit_type": exit_type,
            "age_at_exit": (None if np.isnan(age_last) else int(age_last)),
            "coached_after": coached_after}


# ----------------------------------------------------------------------------
# Population hygiene: drop mid-season interim caretakers and re-anchor
# interim->permanent hires to their first season-opening year. Reuses the
# Coach_WAR project's solved multi-coach-season resolution (Primary_Coach per
# team-year) rather than re-deriving it (see memory reference-coachwar-primary-coach).
#
# As of the single-builder consolidation, create_data._process_coach_career applies
# this hygiene AT THE SOURCE (via _classify_hire): interim-only caretakers are never
# emitted, and an interim->permanent promotion is built directly at the season-opening
# year with the interim partial HC season FOLDED INTO prior experience (its unit
# performance + the HC-year count + age/context as-of the anchored year). So there is
# no longer any one-season feature staleness. clean_population below now runs only as a
# VERIFICATION GUARD on the already-clean data -- it reuses the same classify_hire_
# instance logic and is expected to report "dropped 0, re-anchored 0".
# ----------------------------------------------------------------------------
COACH_WAR_HC_TABLE = Path(
    r"C:\Users\jonwi\Documents\Projects\Coach_WAR\data\processed\coaching"
    r"\team_year_head_coaches.csv")
# Spurious duplicate hire instances (same real stint recorded twice in master_data).
# Now empty: the upstream stint detector (create_data._process_coach_career) resolves
# franchise identity with the year-aware canonical key plus the head-coaching employer
# name, so relocations/renames and same-year demotion-then-rehire fragments no longer
# create a duplicate instance to remove here. Kept as a guard hook only.
DUPLICATE_HIRE_INSTANCES = []


def _primary_coach_index():
    """year -> list of (set(coach names sharing the team-year), primary coach).
    Primary = season-opening / prior-year-continuity coach (Coach_WAR rule)."""
    hc = pd.read_csv(COACH_WAR_HC_TABLE)
    by_year = {}
    for _, r in hc.iterrows():
        names = {n.strip() for n in str(r["Combined_Coach"]).split("/")}
        by_year.setdefault(int(r["Year"]), []).append(
            (names, str(r["Primary_Coach"]).strip()))
    return by_year


def _is_primary(by_year, coach, year):
    """True/False if the coach opened the season as primary HC that year;
    None if the team-year is absent from the table (coach not found)."""
    for names, prim in by_year.get(year, []):
        if coach in names:
            return coach == prim
    return None


def classify_hire_instance(by_year, coach, hire_year):
    """Return ('clean'|'interim_only'|'reanchor', anchored_hire_year).

    interim_only: never the season-opening primary during the stint (a mid-season
        caretaker, not a real hire) -> drop.
    reanchor: took over mid-season then was kept as the permanent coach -> move
        the hire year to the first season he opened as primary.
    """
    t = reconstruct_tenure(coach, hire_year)
    if t is None:
        return "clean", hire_year
    flags = [(y, _is_primary(by_year, coach, y))
             for y in range(hire_year, hire_year + t)]
    flags = [(y, f) for y, f in flags if f is not None]
    if not flags:                        # coach absent from table -> keep as-is
        return "clean", hire_year
    if not any(f for _, f in flags):     # never primary -> interim caretaker
        return "interim_only", hire_year
    if any(y == hire_year and f for y, f in flags):
        return "clean", hire_year
    return "reanchor", min(y for y, f in flags if f)


def clean_population(df):
    """Drop interim-only fragments + duplicate instances and re-anchor
    interim->permanent hires. Returns the cleaned frame; prints a summary."""
    by_year = _primary_coach_index()
    drop, reanchored = [], []
    for i, r in df.iterrows():
        coach, hy = r["Coach Name"], int(r["Year"])
        if (coach, hy) in DUPLICATE_HIRE_INSTANCES:
            drop.append(i)
            continue
        kind, new_hy = classify_hire_instance(by_year, coach, hy)
        if kind == "interim_only":
            drop.append(i)
        elif kind == "reanchor":
            df.at[i, "Year"] = new_hy
            reanchored.append((coach, hy, new_hy))
    out = df.drop(index=drop).copy()
    print(f"Population hygiene (Coach_WAR primary-coach resolution): "
          f"dropped {len(drop)} interim/duplicate, re-anchored {len(reanchored)}")
    for c, a, b in reanchored:
        print(f"    re-anchor {c}: {a} -> {b}")
    return out


def build_franchise_hc_history():
    """franchise_key -> {year: set(coach names)} for all NFL head-coach seasons."""
    hc = {}
    for d in COACHES_DIR.iterdir():
        if not d.is_dir():
            continue
        h = load_history(d.name)
        if h is None:
            continue
        for _, r in h.iterrows():
            if is_nfl(r.get("Level")) and classify_role(r.get("Role", "")) == "HC":
                fkey = canon_employer(r.get("Employer", ""), r.get("Level"), int(r["Year"]))
                hc.setdefault(fkey, {}).setdefault(int(r["Year"]), set()).add(d.name)
    return hc


def tier5_features(hist, hire_year, hc_hist):
    """Org instability: previous coach's tenure + distinct HCs in the prior decade."""
    f = {"org_prev_coach_tenure": np.nan, "org_unique_hc_10yr": np.nan}
    name, level = hiring_team_info(hist, hire_year)
    if not name:
        return f
    fkey = canon_employer(name, level, hire_year)
    years_hc = hc_hist.get(fkey)
    if not years_hc:
        return f

    # distinct HCs in [hire_year-10, hire_year-1]
    uniq = set()
    for y in range(hire_year - 10, hire_year):
        uniq |= years_hc.get(y, set())
    f["org_unique_hc_10yr"] = len(uniq)  # 0 is meaningful (e.g., expansion team)

    # previous coach tenure: HC in season Y-1, counted consecutively backwards
    prev = years_hc.get(hire_year - 1)
    if prev:
        pc = sorted(prev)[0]
        t, y = 0, hire_year - 1
        while pc in years_hc.get(y, set()):
            t += 1
            y -= 1
        f["org_prev_coach_tenure"] = t
    return f


_ROSTER_LKUP = None
_RET_COLS = None


def _roster_lookup():
    """(Team, Year) -> row dict from the Coach_WAR combined team-year dataset."""
    global _ROSTER_LKUP, _RET_COLS
    if _ROSTER_LKUP is None:
        cw = pd.read_csv(COACH_WAR_COMBINED)
        _RET_COLS = [c for c in cw.columns
                     if c.endswith("_Retention_Rate_Pct") and not c.endswith("_crosstab")]
        _ROSTER_LKUP = cw.set_index(["Team", "Year"]).to_dict("index")
    return _ROSTER_LKUP


# Position groups for offense/defense retention splits (Coach_WAR convention)
_OFF_POS = ["QB", "RB", "WR", "TE", "OL"]
_DEF_POS = ["DL", "LB", "CB", "S"]


def tier6_features(hist, hire_year):
    """Inherited roster/talent from the hiring franchise's PRIOR season (Y-1).

    All columns are point-in-time season Y-1 snapshots (the inherited roster under
    the OLD regime) or churn ENTERING Y-1 (the Y-2->Y-1 transition; Coach_WAR keys
    retention at Year_To, so Y-1 holds Y-2->Y-1) -- never the new coach's offseason.
    Restricted to full-coverage (1970+) fields; cap/injury (2009+ only) excluded.
    """
    from data_constants import standardize_team_abbreviation
    keys = ["hire_qb_av", "hire_roster_av", "hire_starter_av", "hire_starter_age",
            "hire_roster_age", "hire_starter_exp", "hire_roster_exp",
            "hire_recent_r1_picks", "hire_premium_draft",
            "hire_roster_turnover", "hire_off_retention", "hire_def_retention"]
    f = {k: np.nan for k in keys}
    name, _ = hiring_team_info(hist, hire_year)
    if not name:
        return f
    abbr = name2abbr().get(name.lower())
    if not abbr:
        return f
    py = hire_year - 1
    team = standardize_team_abbreviation(abbr, py).upper()
    row = _roster_lookup().get((team, py))
    if row is None:
        return f
    # roster talent / demographics (full 1970+ coverage)
    f["hire_qb_av"] = row.get("Avg_Starter_AV_QB", np.nan)
    f["hire_roster_av"] = row.get("Avg_Roster_AV", np.nan)
    f["hire_starter_av"] = row.get("Avg_Starter_AV", np.nan)
    f["hire_starter_age"] = row.get("Avg_Starter_Age", np.nan)
    f["hire_roster_age"] = row.get("Avg_Roster_Age", np.nan)
    f["hire_starter_exp"] = row.get("Avg_Starter_Experience", np.nan)
    f["hire_roster_exp"] = row.get("Avg_Roster_Experience", np.nan)
    # draft capital: R1 (recent) and premium R1+R2 over the prior 3 years
    r1 = 0.0
    for c in ["Current_Round_1_Picks", "Prev_1Yr_Round_1_Picks", "Prev_2Yr_Round_1_Picks"]:
        v = row.get(c)
        r1 += v if v == v and v is not None else 0.0
    f["hire_recent_r1_picks"] = r1
    prem, any_prem = 0.0, False
    for c in [f"Prev_{n}Yr_Round_{r}_Picks" for n in (1, 2, 3) for r in (1, 2)]:
        v = row.get(c)
        if v == v and v is not None:
            prem += v
            any_prem = True
    f["hire_premium_draft"] = prem if any_prem else np.nan
    # roster churn entering Y-1 (overall + offense/defense split)
    rets = [row[c] for c in _RET_COLS if c in row and row[c] == row[c]]
    f["hire_roster_turnover"] = float(np.mean(rets)) if rets else np.nan
    offv = [row.get(f"{p}_Retention_Rate_Pct") for p in _OFF_POS]
    offv = [x for x in offv if x == x and x is not None]
    defv = [row.get(f"{p}_Retention_Rate_Pct") for p in _DEF_POS]
    defv = [x for x in defv if x == x and x is not None]
    f["hire_off_retention"] = float(np.mean(offv)) if offv else np.nan
    f["hire_def_retention"] = float(np.mean(defv)) if defv else np.nan
    return f


# ----------------------------------------------------------------------------
# Hiring-team STATE: schedule-adjusted quality from PFR team records
# (Teams/<abbr>/team_record.csv: SRS/OSRS/DSRS/MoV/SoS, full 1970+ coverage) plus
# a 3-season SRS trajectory. Measured at hire_year-1 and earlier (the old regime's
# last seasons), never the hiring season -- no leakage.
# ----------------------------------------------------------------------------
TEAMS_DIR = project_root / "Teams"
_TEAM_REC = None


def _team_record_lookup():
    """(abbr, year) -> row from every Teams/<abbr>/team_record.csv."""
    global _TEAM_REC
    if _TEAM_REC is None:
        _TEAM_REC = {}
        for d in TEAMS_DIR.iterdir():
            ff = d / "team_record.csv"
            if not d.is_dir() or not ff.exists():
                continue
            try:
                tr = pd.read_csv(ff)
            except Exception:
                continue
            for _, r in tr.iterrows():
                try:
                    y = int(r["Year"])
                except (ValueError, TypeError):
                    continue
                _TEAM_REC[(d.name.lower(), y)] = r
    return _TEAM_REC


def _hiring_franchise_abbrs(hist, hire_year):
    """Candidate Teams/ folder abbreviations for the hiring franchise (relocation-aware)."""
    from data_constants import standardize_team_abbreviation, TEAM_FRANCHISE_MAPPINGS
    name, _ = hiring_team_info(hist, hire_year)
    if not name:
        return []
    abbr = name2abbr().get(name.lower())
    if not abbr:
        return []
    cands = []
    for a in [abbr, standardize_team_abbreviation(abbr, hire_year)] + \
            list(TEAM_FRANCHISE_MAPPINGS.get(abbr, [])):
        a = str(a).lower()
        if a not in cands:
            cands.append(a)
    return cands


def _team_rec_row(cands, year):
    lk = _team_record_lookup()
    for a in cands:
        if (a, year) in lk:
            return lk[(a, year)]
    return None


def team_quality_features(hist, hire_year):
    """Schedule-adjusted hiring-team quality at hire_year-1 + 3-season SRS slope."""
    keys = ["tq_srs", "tq_osrs", "tq_dsrs", "tq_mov", "tq_sos", "tq_srs_traj"]
    f = {k: np.nan for k in keys}
    cands = _hiring_franchise_abbrs(hist, hire_year)
    if not cands:
        return f
    row = _team_rec_row(cands, hire_year - 1)
    if row is not None:
        for k, col in [("tq_srs", "SRS"), ("tq_osrs", "OSRS"), ("tq_dsrs", "DSRS"),
                       ("tq_mov", "MoV"), ("tq_sos", "SoS")]:
            v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            f[k] = float(v) if v == v else np.nan
    srs_seq = []
    for y in (hire_year - 3, hire_year - 2, hire_year - 1):
        r = _team_rec_row(cands, y)
        v = (pd.to_numeric(pd.Series([r.get("SRS")]), errors="coerce").iloc[0]
             if r is not None else np.nan)
        srs_seq.append(v)
    f["tq_srs_traj"] = _slope([v for v in srs_seq if v == v])
    return f


def build_modeling_dataset(df, verbose=True):
    """Build the final modeling dataset from the all-era base hiring-instance table.

    This is the single source of the canonical `data/master_data.csv`: it takes the
    raw all-era hiring instances (produced in-memory by create_data's
    CoachingDataProcessor), restricts to the modern era (1970+), applies population
    hygiene (drops mid-season interim caretakers, re-anchors interim->permanent
    hires), corrects the tenure labels relocation/partial-season-aware, and appends
    the engineered career-path / rank / org / roster / team-quality features.

    Returns the final DataFrame. The previous two-file split (master_data.csv base +
    master_data_extended.csv final) is gone: create_data.main() calls this and writes
    the single result. Kept as an importable function so the build has ONE code path.
    """
    n_all = len(df)
    df = df[df["Year"] >= MIN_HIRE_YEAR].copy()
    if verbose:
        print(f"Loaded {n_all} instances; kept {len(df)} from {MIN_HIRE_YEAR}+ (modern era)")

    # Population hygiene: remove mid-season interim caretakers + duplicate hire
    # instances and re-anchor interim->permanent hires to their season-opening
    # year (reuses Coach_WAR's Primary_Coach resolution). Must precede the tenure
    # recompute and feature build so both use the anchored hire year.
    df = clean_population(df)

    # Correct the target labels: the original pipeline truncated tenure at franchise
    # relocations/renames and undercounted mid-season exits. Recompute the class
    # relocation-aware and partial-season-aware (season count -> standard bins).
    # Active recent hires (class -1) are left censored.
    fixed = 0
    for i, r in df.iterrows():
        if int(r["Coach Tenure Class"]) == -1:
            continue
        t = reconstruct_tenure(r["Coach Name"], int(r["Year"]))
        if t is not None:
            newc = tenure_class(t)
            if newc != int(r["Coach Tenure Class"]):
                fixed += 1
            df.at[i, "Coach Tenure Class"] = newc
    known_all = (df["Coach Tenure Class"] != -1).sum()
    print(f"  known-tenure instances {MIN_HIRE_YEAR}+: {known_all} | corrected labels: {fixed}")
    print(f"  corrected class dist: {dict(df[df['Coach Tenure Class']!=-1]['Coach Tenure Class'].value_counts().sort_index())}")

    # Tier 5 (org instability) and Tier 6 (inherited roster/talent) were built and
    # tested but found to carry no predictive signal for tenure and to slightly
    # HURT performance (see scripts/feature_lift_experiment.py /
    # analysis/feature_lift_results.pkl). They are therefore EXCLUDED from the
    # canonical modeling dataset. Set INCLUDE_TIER56=True to regenerate the
    # 179-feature variant used for that null-result analysis.
    # Tier 5 (org instability) + Tier 6 (inherited roster/talent) were null for
    # tenure CLASSIFICATION, but are being re-tested for the FIRING-survival pivot
    # (a churn-prone franchise plausibly raises firing hazard even if it does not
    # shift tenure class). Included so feature selection can evaluate them.
    INCLUDE_TIER56 = True
    hc_hist = build_franchise_hc_history() if INCLUDE_TIER56 else None

    rows = []
    missing_hist = 0
    for _, row in df.iterrows():
        coach = row["Coach Name"]
        hire_year = int(row["Year"])
        hist = load_history(coach)
        ranks = load_ranks(coach)
        if hist is None:
            missing_hist += 1
            feat = {}
        else:
            feat = tier1_features(hist, hire_year)
            if INCLUDE_TIER56:
                feat.update(tier5_features(hist, hire_year, hc_hist))
                feat.update(tier6_features(hist, hire_year))
                feat.update(team_quality_features(hist, hire_year))
        feat.update(tier2_features(ranks, hire_year))
        rows.append(feat)

    new_df = pd.DataFrame(rows, index=df.index)
    print(f"Engineered {new_df.shape[1]} new features ({missing_hist} coaches missing history)"
          f"{' [incl. Tier5/6]' if INCLUDE_TIER56 else ' [Tier1+2 only]'}")

    # Insert new feature columns before the last two target columns
    target_cols = ["Avg 2Y Win Pct", "Coach Tenure Class"]
    feat_cols = [c for c in df.columns if c not in target_cols]
    ext = pd.concat([df[feat_cols], new_df, df[target_cols]], axis=1)
    if verbose:
        _report_feature_coverage(ext, new_df)
    return ext


def _report_feature_coverage(ext, new_df):
    """Coverage/correlation diagnostics + spot checks printed during a build."""
    known = ext[ext["Coach Tenure Class"] != -1]
    y = known["Coach Tenure Class"]
    print("\nNew-feature coverage and correlation with tenure class (known instances):")
    print(f"{'feature':<28} {'%nonNaN':>8} {'corr(y)':>9}")
    for c in new_df.columns:
        v = pd.to_numeric(known[c], errors="coerce")
        cov = 100 * v.notna().mean()
        corr = v.corr(y) if v.notna().sum() > 10 else np.nan
        print(f"{c:<28} {cov:>7.0f}% {corr:>9.3f}")

    for name, yr in [('Andy Reid', 1999), ('Jeff Fisher', 1995)]:
        idx = ext[(ext['Coach Name'] == name) & (ext['Year'] == yr)]
        if len(idx):
            r = idx.iloc[0]
            print(f"\nSpot check ({name} {yr}):")
            for c in ['cf_num_nfl_employers','cf_num_college_employers','cf_internal_hire',
                      'cf_years_at_hiring_org','cf_pre_OC','cf_pre_DC','cf_pre_side_def',
                      'cf_pre_level_nfl','rf_final_unit_pctl','rf_avg_winpct_pctl']:
                print(f"  {c} = {r[c]}")


if __name__ == "__main__":
    # The single dataset builder is scripts/data/create_data.py, which constructs
    # the base instances and calls build_modeling_dataset(). This module is the
    # shared feature/tenure/hygiene library imported by the survival scripts;
    # running it directly just delegates to that one canonical builder.
    from scripts.data.create_data import main as build_master_data
    build_master_data()
