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
import re
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
from data_constants import coaching_title, get_all_feature_names

# Drive and situational unit statistics, which PFR records only from about the 1999
# season; a coach whose record predates them has a unit record but no drive record.
DRIVE_ERA_STATS = {"#Dr", "Sc%", "TO%", "Time Average Drive", "Plays Average Drive",
                   "Yds Average Drive", "Pts Average Drive", "3DAtt", "3D%", "4DAtt",
                   "4D%", "RZAtt", "RZPct"}


def unit_block_columns():
    """(all pooled-unit columns, drive-era subset) as the generic 'Feature N' names
    the base build writes, located by their data_constants feature names."""
    names = get_all_feature_names()
    unit = [f"Feature {i + 1}" for i, n in enumerate(names) if n.endswith("__unit")]
    drive = [f"Feature {i + 1}" for i, n in enumerate(names)
             if n.endswith("__unit") and n[:-len("__unit")] in DRIVE_ERA_STATS]
    return unit, drive

OFF_KW = ["offensive", "quarterback", "running back", "receiver", "tight end",
          "o-line", "passing game", "pass game"]
DEF_KW = ["defensive", "linebacker", "defensive back", "secondary", "cornerback",
          "safet", "d-line", "nickel", "pass rush", "run game coordinator/def"]
# Position abbreviations, matched as whole words ("TE", "WR Coach", "LBs")
OFF_ABBR = {"qb", "qbs", "rb", "rbs", "wr", "wrs", "te", "tes", "ol"}
DEF_ABBR = {"lb", "lbs", "db", "dbs", "dl", "cb", "cbs"}


def classify_role(role):
    """Return one of: HC, OC, DC, STC, POS, NONE (mirrors create_data logic)."""
    if not isinstance(role, str) or not role.strip():
        return "NONE"
    role = coaching_title(role)   # drop excluded duties from compound titles
    if not role:
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
    words = set(re.findall(r"[a-z]+", r))
    is_off = any(k in r for k in OFF_KW) or bool(words & OFF_ABBR)
    is_def = any(k in r for k in DEF_KW) or bool(words & DEF_ABBR)
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
    """Major-league (NFL-equivalent) level: the NFL, plus the AFL of 1960-69, whose
    ten franchises are NFL franchises today. AFL seasons therefore count toward
    coaching experience, unit performance, stint continuity (so AFL incumbents in
    1970 are continuing stints, not hires) and franchise history. Minor leagues
    (NFL Europe, USFL, WFL, CFL) do not count."""
    s = str(level)
    return ("NFL" in s and s != "NFL Europe") or s == "AFL"


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


def load_history(coach, dedupe=True):
    """Coach history rows. dedupe=False keeps PFR's split same-season rows, which
    interim detection needs: a season listed as "Defensive Line" and "Defensive
    Line/Interim HC" dedupes to the first row and would lose the interim title."""
    p = COACHES_DIR / coach / "all_coaching_history.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df = df.dropna(subset=["Year"])
        df["Year"] = df["Year"].astype(int)
        return _dedupe_history(df) if dedupe else df
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

# Restrict the modeling population to hires from the 1970 merger on. AFL seasons
# count as major-league seasons (is_nfl), so AFL incumbents in 1970 (Stram, Ewbank,
# Lemm, Saban, P. Brown, Madden, Rush, Rauch, Waller) are continuing stints that began
# before the window, exactly like NFL incumbents such as Landry, and are not hires.
# Data a 1970-71 hire lacks (rosters begin in 1970) is flagged, not imputed.
MIN_HIRE_YEAR = 1970
# Inherited-roster data (Coach_WAR combined team-year file) begins with this season:
# a 1970 hire has no prior-season roster, and 1970-71 hires have no prior-season
# retention (which compares two consecutive rosters).
ROSTER_DATA_START = 1970

# Expansion franchises' first head coaches inherit no predecessor, record, team
# quality or roster. A known list, not a data rule: a data rule misreads franchises
# whose prior season is merely absent from a source (e.g. the AFL). The 1996 Ravens
# are not an expansion team; they inherit the 1995 Browns (INHERITED_FRANCHISES).
EXPANSION_FIRST_SEASONS = {("Seattle Seahawks", 1976), ("Tampa Bay Buccaneers", 1976),
                           ("Carolina Panthers", 1995), ("Jacksonville Jaguars", 1995),
                           ("Cleveland Browns", 1999), ("Houston Texans", 2002)}

# First season covered by the Coach_WAR Primary_Coach table used for interim
# resolution. Distinct from MIN_HIRE_YEAR: 1970-71 hires are still resolved (so a
# 1971 mid-season takeover is folded and re-anchored to 1972) before the window cut.
PRIMARY_COACH_START = 1970

# Mid-season appointments hired from OUTSIDE the organization, keyed by anchored hire
# year. PFR lists only "Head Coach" for a partial season (earlier same-season roles are
# not recorded), so the data cannot separate these from staff or front-office insiders
# promoted mid-season, and the prior-season rule codes all of them internal. All 18
# such takeovers were checked against contemporaneous sources
# (analysis/midseason_takeovers_summary.md): six were verified insiders (Khayat 1971,
# Svare 1971, Gillman 1973, McVay 1976, Hudspeth 1976, Rice 1978) and stay internal;
# these four were outside appointments and are recoded external. J.D. Roberts (Saints,
# took over in November 1970; anchored 1971 once PRIMARY_COACH_OVERRIDES restores Fears
# as the 1970 opener) stays internal: he came from the Richmond Roadrunners, the Saints'
# minor-league affiliate, after two seasons as a Saints scout and linebackers coach, so
# he was inside the organization (author decision, 2026-10-29).
MIDSEASON_OUTSIDE_HIRES = {("Don Coryell", 1979), ("Raymond Berry", 1985),
                           ("Marv Levy", 1987), ("Ron Meyer", 1987)}

_INTERIM_HC = re.compile(r"Interim\s*(?:Head Coach|HC|Co-HC)")
_INTERIM_TITLE = {"HC": _INTERIM_HC,
                  "OC": re.compile(r"Interim\s*(?:OC|Offensive Coordinator)"),
                  "DC": re.compile(r"Interim\s*(?:DC|Defensive Coordinator)")}


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
    # Career-path measures (ages, employers, NFL share, years at the hiring
    # organization) count every job in the record, including the entry-level titles
    # (quality control, graduate or generic assistant) that the role-year counters
    # skip: the age at a coach's first NFL job is the age at which the NFL first
    # appears in his record. Role measures (most recent role, gap since the last
    # coaching season) use coaching seasons only.
    career = pre[pre["Role"].astype(str).str.strip().ne("") & pre["Role"].notna()]
    coaching = pre[pre["rtype"] != "NONE"]
    if len(career) == 0:
        return f

    nfl_rows = coaching[coaching["Level"].apply(is_nfl)]
    nfl_career = career[career["Level"].apply(is_nfl)]
    col_career = career[career["Level"].astype(str).str.contains("College")]

    # structural-missingness indicators for the OC / DC performance blocks
    f["cf_ever_nfl_oc"] = int(any(classify_role(rr) == "OC" for rr in nfl_rows["Role"]))
    f["cf_ever_nfl_dc"] = int(any(classify_role(rr) == "DC" for rr in nfl_rows["Role"]))

    # Relocation-aware employer counts (year-based franchise identity for NFL)
    nfl_keys = {canon_employer(e, l, y)
                for e, l, y in zip(nfl_career["Employer"], nfl_career["Level"], nfl_career["Year"])}
    col_keys = {canon_employer(e, l, y)
                for e, l, y in zip(col_career["Employer"], col_career["Level"], col_career["Year"])}
    f["cf_num_nfl_employers"] = int(len(nfl_keys))
    f["cf_num_college_employers"] = int(len(col_keys))
    # share of career SEASONS spent in the NFL (a season can carry several titles)
    f["cf_nfl_share"] = nfl_career["Year"].nunique() / career["Year"].nunique()

    if "Age" in career.columns:
        ages = pd.to_numeric(career["Age"], errors="coerce")
        f["cf_age_first_job"] = float(ages.min()) if ages.notna().any() else np.nan
        nfl_ages = pd.to_numeric(nfl_career["Age"], errors="coerce") if len(nfl_career) else pd.Series(dtype=float)
        if len(nfl_ages) and nfl_ages.notna().any():
            f["cf_age_first_nfl"] = float(nfl_ages.min())
        else:
            # no prior NFL job (college-to-NFL jump): this hire IS the first NFL
            # job, so the value is the age at hire (a definition, not an imputation)
            hire_age = pd.to_numeric(hist.loc[hist["Year"] == hire_year, "Age"], errors="coerce")
            f["cf_age_first_nfl"] = float(hire_age.min()) if hire_age.notna().any() else np.nan
    if len(coaching) == 0:
        return f
    years = coaching["Year"].values
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

    # internal hire: employed by the hiring franchise in the IMMEDIATE prior season.
    # For a re-anchored mid-season takeover the prior season is the partial HC season
    # itself, which correctly codes staff (and front-office) promotions as internal.
    # The verified mid-season OUTSIDE appointments are recoded in
    # build_modeling_dataset via MIDSEASON_OUTSIDE_HIRES.
    prior = career[career["Year"] == hire_year - 1]
    prior_keys = {canon_employer(e, l, y)
                  for e, l, y in zip(prior["Employer"], prior["Level"], prior["Year"])}
    f["cf_internal_hire"] = int(hiring_key in prior_keys) if hiring_key else 0
    # consecutive seasons at the hiring franchise immediately before the hire, in any job
    yrs_at_org = 0
    if hiring_key:
        org_years = {int(y) for e, l, y in zip(career["Employer"], career["Level"], career["Year"])
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


def interim_seasons(coach, raw_hist, ranks=None):
    """Interim years, whose HEAD-COACHING side never counts toward a coach's unit
    statistics, in either the base unit block (create_data) or the recent-form features
    (tier2). A coordinator title held outright that season still counts, on its own side
    of the ball: a defensive coordinator named interim head coach (Fisher 1994) keeps
    that season's defense and gets no offense. Every other title counts as experience
    only. A season is an interim year when the coach
      - held an interim title in it ("Defensive Coordinator/Interim HC",
        "QB Coach/Interim OC");
      - is credited with a head-coaching season in PFR's ranks table that his history
        does not show him holding outright (Lynn 2016, a one-game caretaker stint
        listed under his OC title); or
      - took over as head coach mid-season, so the season was not his to open
        (Coach_WAR Primary_Coach from 1970, with PRIMARY_COACH_OVERRIDES; before the
        table, _took_over_pre_table). Most of that season was coached by his
        predecessor; or
      - was suspended for it (Payton 2012: the history keeps his head-coach title for a
        season he did not coach).
    Needs the un-deduplicated history, whose split rows can carry the interim title."""
    out = set()
    if raw_hist is None or not len(raw_hist):
        return out
    years = lambda d: pd.to_numeric(d["Year"], errors="coerce")
    raw_hist = raw_hist[years(raw_hist).notna()].assign(Year=lambda d: years(d).astype(int))
    if ranks is not None:
        ranks = ranks[years(ranks).notna()].assign(Year=lambda d: years(d).astype(int))
    nfl = raw_hist[raw_hist["Level"].apply(is_nfl)]
    for y, g in nfl.groupby("Year"):
        roles = g["Role"].astype(str)
        if any(roles.str.contains(pat).any() for pat in _INTERIM_TITLE.values()):
            out.add(int(y))
        if roles.str.contains("suspended", case=False).any():
            out.add(int(y))
    hc_years = {int(y) for y, r in zip(nfl["Year"], nfl["Role"]) if classify_role(r) == "HC"}
    if ranks is not None:
        nfl_years = set(nfl["Year"].astype(int))
        for y, t in zip(ranks["Year"].astype(int), ranks["Role"].astype(str).str.upper()):
            if t == "HC" and y not in hc_years and y in nfl_years:
                out.add(y)
    for y in hc_years:
        took_over = (_is_primary(_primary_idx(), coach, y) is False if y >= PRIMARY_COACH_START
                     else _took_over_pre_table(coach, y))
        if took_over:
            out.add(y)
    return out


def _coordinator_held(hist, year):
    """'OC' or 'DC' if the coach held that coordinator title outright in an NFL season
    (offense first if both), else None."""
    if hist is None:
        return None
    rows = hist[(hist["Year"] == year) & hist["Level"].apply(is_nfl)]
    held = {classify_role(r) for r in rows["Role"]}
    return next((r for r in ("OC", "DC") if r in held), None)


_FRANCHISE_HC = None


def _franchise_hc():
    """build_franchise_hc_history(), built once per process."""
    global _FRANCHISE_HC
    if _FRANCHISE_HC is None:
        _FRANCHISE_HC = build_franchise_hc_history()
    return _FRANCHISE_HC


def _took_over_pre_table(coach, year):
    """Before the Primary_Coach table: True if the coach took over as head coach during
    `year`. He shared the club's season with another head coach and had not led the
    club the season before, and either one of the others had (the continuing coach
    opened the season: Gillman for the 1969 AFL Chargers, whom Waller replaced; Collier
    for the 1968 Bills, whom Harvey Johnson replaced, though Johnson was not kept) or,
    when no one continued, he was the one kept the next season. A coach who opened a
    season and was fired during it is not a takeover."""
    h = load_history(coach)
    if h is None:
        return False
    rows = h[(h["Year"] == year) & h["Level"].apply(is_nfl)]
    rows = rows[[classify_role(r) == "HC" for r in rows["Role"]]]
    if rows.empty:
        return False
    r = rows.iloc[0]
    by_year = _franchise_hc().get(canon_employer(r["Employer"], r["Level"], year), {})
    cur, prev = by_year.get(year, set()), by_year.get(year - 1, set())
    if len(cur) < 2 or coach in prev:
        return False
    return bool((cur - {coach}) & prev) or coach in by_year.get(year + 1, set())


def _team_rank_row(employer, year, role):
    """A ranks-table-style row for a season PFR's coach ranks omit (e.g. Tobin's
    1994-95 Colts defenses), built from the team's own season record: points scored,
    points allowed and net-points ranks, league size, and a win-pct rank within that
    season's league. None if the team-season cannot be located."""
    from data_constants import standardize_team_abbreviation
    abbr = name2abbr().get(str(employer).strip().lower())
    if not abbr:
        return None
    lk = _team_record_lookup()
    r = lk.get((standardize_team_abbreviation(abbr, year).lower(), year))
    if r is None:
        return None
    num = lambda v, c: float(pd.to_numeric(pd.Series([v.get(c)]), errors="coerce").fillna(0).iloc[0])
    wp = lambda v: (num(v, "W") + 0.5 * num(v, "T")) / max(num(v, "W") + num(v, "L") + num(v, "T"), 1)
    peers = [v for (_, y), v in lk.items() if y == year and v.get("Lg") == r.get("Lg")]
    return {"Year": year, "Role": role, "Tms": r.get("out of"),
            "WL%": 1 + sum(wp(v) > wp(r) for v in peers),
            "Pts": r.get("Pts"), "Pts_2": r.get("Pts Def Rank"), "Pts±": r.get("Pts±")}


def _unranked_unit_seasons(ranks, hist, hire_year, interim=()):
    """Ranks-style rows for prior NFL seasons the history shows as an outright HC, OC or
    DC season but PFR's coach ranks table lacks (in an interim year, a coordinator
    title only)."""
    if hist is None:
        return []
    have = set() if ranks is None else set(ranks["Year"].astype(int))
    out = []
    pre = hist[(hist["Year"] < hire_year) & hist["Level"].apply(is_nfl)]
    for y, g in pre.groupby("Year"):
        if int(y) in have:
            continue
        held = {classify_role(r) for r in g["Role"]}
        if int(y) in interim:
            held.discard("HC")
        role = next((r for r in ("HC", "OC", "DC") if r in held), None)
        if role:
            emp = g.loc[g["Role"].apply(classify_role) == role, "Employer"].iloc[0]
            row = _team_rank_row(emp, int(y), role)
            if row is not None:
                out.append(row)
    return out


def interim_features(raw_hist, hire_year, coach=None, ranks=None):
    """Served as an interim head coach in a prior NFL season: an interim head-coach
    title, or a head-coaching season that interim_seasons marks as an interim year (a
    mid-season takeover, or a caretaker stint PFR credits without the title, as with
    Lynn in 2016). PFR titles many interim coaches plain "Head Coach" (Garrett and
    Frazier in 2010), so the title alone misses them. A suspension (Payton 2012) is not
    interim service, nor is the takeover season of a mid-season hire appointed
    permanently from outside (MIDSEASON_OUTSIDE_HIRES). Interim seasons never count as
    head-coaching or unit-leading seasons anywhere in the feature set; this flag records
    the experience instead. Needs the un-deduplicated history."""
    if raw_hist is None:
        return {"cf_ever_interim_hc": np.nan}
    year = pd.to_numeric(raw_hist["Year"], errors="coerce")
    pre = raw_hist[(year < hire_year) & raw_hist["Level"].apply(is_nfl)]
    if pre["Role"].astype(str).str.contains(_INTERIM_HC).any():
        return {"cf_ever_interim_hc": 1}
    roles = list(zip(pd.to_numeric(pre["Year"]).astype(int), pre["Role"].astype(str)))
    hc = {y for y, r in roles if classify_role(r) == "HC"}
    if ranks is not None:
        hc |= {int(y) for y, t in zip(pd.to_numeric(ranks["Year"], errors="coerce"),
                                      ranks["Role"].astype(str).str.upper())
               if t == "HC" and y == y and y < hire_year}
    suspended = {y for y, r in roles if "suspended" in r.lower()}
    held = {y for y in interim_seasons(coach, raw_hist, ranks) if y < hire_year} & hc
    held -= suspended
    if (coach, hire_year) in MIDSEASON_OUTSIDE_HIRES:
        held.discard(hire_year - 1)
    return {"cf_ever_interim_hc": int(bool(held))}


def tier2_features(ranks, hire_year, hist=None, coach=None):
    """Side-appropriate unit performance + team success from prior ranked seasons.

    For each prior season the coach's OWN unit is used: offensive points rank when
    they were OC, DEFENSIVE points-allowed rank when DC, net point differential
    when HC. Ranks are converted to percentiles (1=best) normalized by league size.
    In an interim year (interim_seasons) a row is scored on the coordinator title held
    outright that season and dropped if there is none, exactly as in the base unit
    block; outright unit seasons missing from PFR's coach ranks come from the team's
    season record.
    """
    f = {k: np.nan for k in ["rf_final_unit_pctl", "rf_avg_unit_pctl",
                             "rf_unit_traj", "rf_final_winpct_pctl",
                             "rf_avg_winpct_pctl", "rf_n_seasons_rank"]}
    f["rf_no_trend"] = 0
    interim = interim_seasons(coach, hist, ranks)
    pre = ranks[ranks["Year"] < hire_year].copy() if ranks is not None else pd.DataFrame()
    if len(pre):
        pre["Role"] = [_coordinator_held(hist, int(y)) if int(y) in interim else r
                       for y, r in zip(pre["Year"], pre["Role"])]
        pre = pre[pre["Role"].notna()]
    extra = _unranked_unit_seasons(ranks, hist, hire_year, interim)
    if extra:
        pre = pd.concat([pre, pd.DataFrame(extra)], ignore_index=True)
    if len(pre):
        pre = pre.sort_values("Year")
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
        if f["rf_unit_traj"] != f["rf_unit_traj"]:
            # one ranked unit season: the trajectory is undefined. Set it to 0 and
            # flag it, so the model estimates these coaches' hazard separately
            # instead of treating them as a genuinely flat multi-season trend.
            f["rf_unit_traj"] = 0.0
            f["rf_no_trend"] = 1
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
# year with the interim partial HC season FOLDED INTO prior experience (the HC-year
# count + age/context as-of the anchored year; as an interim year it adds no
# head-coaching unit statistics, see interim_seasons). So there is
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


# Corrections to the Coach_WAR Primary_Coach table, keyed (season, a coach named in
# that team-season) -> the coach who actually opened the season. Coach_WAR breaks a
# tie in games coached by prior-year continuity, which is unavailable in the table's
# first season: Tom Fears opened the 1970 Saints and was fired after a 1-5-1 start,
# and J.D. Roberts coached the last seven games.
PRIMARY_COACH_OVERRIDES = {(1970, "J.D. Roberts"): "Tom Fears"}


def _primary_coach_index():
    """year -> list of (set(coach names sharing the team-year), primary coach).
    Primary = season-opening / prior-year-continuity coach (Coach_WAR rule)."""
    hc = pd.read_csv(COACH_WAR_HC_TABLE)
    by_year = {}
    for _, r in hc.iterrows():
        names = {n.strip() for n in str(r["Combined_Coach"]).split("/")}
        prim = str(r["Primary_Coach"]).strip()
        for (y, member), opener in PRIMARY_COACH_OVERRIDES.items():
            if int(r["Year"]) == y and member in names:
                prim = opener
        by_year.setdefault(int(r["Year"]), []).append((names, prim))
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


_PRIMARY_IDX = None


def _primary_idx():
    """Cached _primary_coach_index ({} if the Coach_WAR table is unavailable)."""
    global _PRIMARY_IDX
    if _PRIMARY_IDX is None:
        try:
            _PRIMARY_IDX = _primary_coach_index()
        except Exception:
            _PRIMARY_IDX = {}
    return _PRIMARY_IDX


def _franchise_hc_years(hc_hist, fkey, hire_year):
    """A franchise's head-coach seasons as its ORGANIZATION experienced them: the
    Ravens include the Browns' seasons through 1995, and the reactivated Browns
    (1999 on) exclude them (data_constants.INHERITED_FRANCHISES)."""
    from data_constants import INHERITED_FRANCHISES
    years = dict(hc_hist.get(fkey) or {})
    fr = fkey.split(":", 1)[-1]
    if fr in INHERITED_FRANCHISES:
        src, last = INHERITED_FRANCHISES[fr]
        years = {**{y: v for y, v in (hc_hist.get("nfl:" + src) or {}).items() if y <= last}, **years}
    for heir, (src, last) in INHERITED_FRANCHISES.items():
        if fr == src and hire_year > last + 1:
            years = {y: v for y, v in years.items() if y > last}
    return years


def _predecessor(names, year, years_hc, exclude):
    """The coach a hire replaced: the franchise's SEASON-OPENING head coach in the
    prior season (Coach_WAR Primary_Coach), never the hire himself. When a season had
    two head coaches (a mid-season change), picking one arbitrarily could return the
    hire's own partial season or a one-game caretaker. Falls back to the
    longest-serving remaining coach if the table lacks the team-year."""
    cands = set(names) - {exclude}
    for team_names, prim in _primary_idx().get(year, []):
        if team_names & set(names) and prim in cands:
            return prim

    def run(c):
        t, y = 0, year
        while c in years_hc.get(y, set()):
            t, y = t + 1, y - 1
        return t
    return max(sorted(cands), key=run) if cands else None


def tier5_features(hist, hire_year, hc_hist, coach=None):
    """Org instability: previous coach's tenure + distinct HCs in the prior decade
    (a re-anchored mid-season takeover's own partial season is not counted)."""
    f = {"org_prev_coach_tenure": np.nan, "org_unique_hc_10yr": np.nan, "org_expansion": np.nan,
         "org_prev_fired": np.nan}
    name, level = hiring_team_info(hist, hire_year)
    if not name:
        return f
    fkey = canon_employer(name, level, hire_year)
    f["org_expansion"] = int((name, hire_year) in EXPANSION_FIRST_SEASONS)
    years_hc = _franchise_hc_years(hc_hist, fkey, hire_year)
    if not years_hc:
        return f

    # distinct HCs in [hire_year-10, hire_year-1]
    # (the hire's own partial takeover season at hire_year - 1 belongs to this stint
    # and is excluded; earlier stints of a returning coach, e.g. Bud Grant, still count)
    uniq = set()
    for y in range(hire_year - 10, hire_year):
        s = years_hc.get(y, set())
        uniq |= (s - {coach}) if y == hire_year - 1 else s
    f["org_unique_hc_10yr"] = len(uniq)  # 0 is meaningful (e.g., expansion team)

    # previous coach tenure: the coach replaced. A stint in the sample uses its own
    # anchored tenure, so a predecessor's partial takeover season is folded into his
    # prior experience as everywhere else (Garrett's Dallas stint is 9 seasons, not
    # 10); a stint that began before the window is counted consecutively backwards.
    prev = years_hc.get(hire_year - 1)
    pc = _predecessor(prev, hire_year - 1, years_hc, coach) if prev else None
    if pc:
        fired, tenure = _predecessor_exit(pc, hire_year - 1)
        if tenure != tenure:
            tenure, y = 0, hire_year - 1
            while pc in years_hc.get(y, set()):
                tenure += 1
                y -= 1
        f["org_prev_coach_tenure"] = tenure
        f["org_prev_fired"] = fired
    return f


_PRED_EXITS = None


def _predecessor_exit(coach, last_season):
    """(fired, tenure) for the predecessor's stint ending in last_season. fired is 1
    if it ended in involuntary non-retention, 0 if he left voluntarily while still
    wanted (same protocol as the outcome). Read from the outcome labels
    (analysis/event_labels_final.csv, which also give the anchored tenure) for stints
    in the sample, and from analysis/predecessor_exits_prewindow.csv (tenure NaN) for
    stints that began before 1970. The labels file is built from master_data, so after
    a change to stints or outcomes rebuild the labels and then rerun create_data."""
    global _PRED_EXITS
    if _PRED_EXITS is None:
        _PRED_EXITS = {}
        pw = project_root / "analysis" / "predecessor_exits_prewindow.csv"
        if pw.exists():
            t = pd.read_csv(pw)
            _PRED_EXITS.update({(c, int(y)): (int(e), np.nan)
                                for c, y, e in zip(t["predecessor"], t["last_season"], t["event"])})
        lab = project_root / "analysis" / "event_labels_final.csv"
        if lab.exists():
            t = pd.read_csv(lab)
            _PRED_EXITS.update({(c, int(y)): (int(e), int(d)) for c, y, e, d
                                in zip(t["coach"], t["last_year"], t["event"], t["duration"])})
    return _PRED_EXITS.get((coach, int(last_season)), (np.nan, np.nan))


_ROSTER_LKUP = None


def _roster_lookup():
    """(Team, Year) -> row dict from the Coach_WAR combined team-year dataset."""
    global _ROSTER_LKUP
    if _ROSTER_LKUP is None:
        cw = pd.read_csv(COACH_WAR_COMBINED)
        _ROSTER_LKUP = cw.set_index(["Team", "Year"]).to_dict("index")
    return _ROSTER_LKUP


# Roster retention is computed here from the raw PFR team rosters rather than taken
# from the Coach_WAR combined file, whose position-group retention mapped starters'
# side-specific codes (RDE, LCB, LOLB, ...) to no group and so reported 0% where the
# rosters show otherwise (1984->85 Vikings defense: 0% reported, 13 of 29 defenders
# returned). Coach_WAR's mapping is now fixed, but its rates still count a player as
# retained only within his position group. For the inherited roster the question is
# continuity of personnel, so here a player is retained if he is on the next season's
# roster at ANY position; position (first listed code, previous season) only assigns
# him to the offense or defense split.
COACH_WAR_ROSTERS = COACH_WAR_COMBINED.parent.parent / "raw" / "Rosters"
_OFF_CODES = {"QB", "RB", "HB", "FB", "TB", "WR", "FL", "SE", "TE", "C", "G", "T", "OL",
              "OG", "OT", "LT", "RT", "LG", "RG"}
_DEF_CODES = {"DE", "DT", "NT", "DL", "LDE", "RDE", "LDT", "RDT", "LB", "MLB", "ILB",
              "OLB", "LLB", "RLB", "LOLB", "ROLB", "LILB", "RILB", "DB", "CB", "LCB",
              "RCB", "S", "SS", "FS", "SAF"}
_ROSTERS = {}


def _roster(team, year):
    """(player name, side) for a franchise-season's PFR roster; None if absent."""
    key = (team, year)
    if key not in _ROSTERS:
        f = COACH_WAR_ROSTERS / team / f"{team.lower()}_{year}_roster.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["Player", "Pos"])
            code = r["Pos"].astype(str).str.upper().str.split("/").str[0].str.strip()
            side = np.where(code.isin(_OFF_CODES), "off", np.where(code.isin(_DEF_CODES), "def", "other"))
            _ROSTERS[key] = pd.DataFrame({"name": r["Player"].astype(str).str.strip().str.upper(),
                                          "side": side})
        else:
            _ROSTERS[key] = None
    return _ROSTERS[key]


def roster_retention(team_from, team_to, year_to):
    """Percent of the year_to-1 roster still on the year_to roster: overall, offense,
    defense (sides by the earlier season's position)."""
    a, b = _roster(team_from, year_to - 1), _roster(team_to, year_to)
    if a is None or b is None or not len(a):
        return np.nan, np.nan, np.nan
    kept = a["name"].isin(set(b["name"]))
    pct = lambda m: float(100 * kept[m].mean()) if m.any() else np.nan
    return float(100 * kept.mean()), pct(a["side"] == "off"), pct(a["side"] == "def")


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
            "hire_roster_retention", "hire_off_retention", "hire_def_retention"]
    f = {k: np.nan for k in keys}
    name, _ = hiring_team_info(hist, hire_year)
    if not name:
        return f
    abbr = name2abbr().get(name.lower())
    if not abbr:
        return f
    from data_constants import prior_season_franchise
    py = hire_year - 1
    team = prior_season_franchise(standardize_team_abbreviation(abbr, hire_year), py).upper()
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
    # draft capital over the three drafts before the hire (Y-1, Y-2, Y-3). In the Y-1
    # row, "Current" is the Y-1 draft and "Prev_kYr" the Y-1-k draft. First-round
    # picks, and first- plus second-round ("premium") picks over the SAME window
    # (premium previously used Y-2..Y-4, skipping the most recent draft).
    window = ["Current", "Prev_1Yr", "Prev_2Yr"]

    def picks(rounds):
        vals = [row.get(f"{w}_Round_{r}_Picks") for w in window for r in rounds]
        vals = [v for v in vals if v is not None and v == v]
        return float(sum(vals)) if vals else np.nan
    f["hire_recent_r1_picks"] = picks((1,))
    f["hire_premium_draft"] = picks((1, 2))
    # roster retention entering Y-1 (the Y-2 -> Y-1 transition under the old regime):
    # overall, offense and defense, from the raw rosters (see roster_retention)
    team_y2 = prior_season_franchise(standardize_team_abbreviation(abbr, hire_year), py - 1).upper()
    (f["hire_roster_retention"], f["hire_off_retention"],
     f["hire_def_retention"]) = roster_retention(team_y2, team, py)
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


def _inherited_cands(cands, hire_year, season):
    """Team-file candidates for a prior season, following an inherited franchise
    (the Ravens' seasons before 1996 come from the Browns; data_constants)."""
    from data_constants import standardize_team_abbreviation, prior_season_franchise
    for a in cands:
        own = standardize_team_abbreviation(a, hire_year)
        src = prior_season_franchise(own, season)
        if src != own:
            return [src]
    return cands


def team_quality_features(hist, hire_year):
    """Schedule-adjusted hiring-team quality at hire_year-1 + 3-season SRS slope."""
    keys = ["tq_srs", "tq_osrs", "tq_dsrs", "tq_mov", "tq_sos", "tq_srs_traj"]
    f = {k: np.nan for k in keys}
    cands = _hiring_franchise_abbrs(hist, hire_year)
    if not cands:
        return f
    row = _team_rec_row(_inherited_cands(cands, hire_year, hire_year - 1), hire_year - 1)
    if row is not None:
        for k, col in [("tq_srs", "SRS"), ("tq_osrs", "OSRS"), ("tq_dsrs", "DSRS"),
                       ("tq_mov", "MoV"), ("tq_sos", "SoS")]:
            v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            f[k] = float(v) if v == v else np.nan
    srs_seq = []
    for y in (hire_year - 3, hire_year - 2, hire_year - 1):
        r = _team_rec_row(_inherited_cands(cands, hire_year, y), y)
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
    # one season of slack so a mid-season takeover in MIN_HIRE_YEAR - 1 can be
    # re-anchored into the window by the hygiene step before the final cut; never
    # before the Primary_Coach table starts, where hygiene would misclassify
    df = df[df["Year"] >= max(MIN_HIRE_YEAR - 1, PRIMARY_COACH_START)].copy()

    # Population hygiene: remove mid-season interim caretakers + duplicate hire
    # instances and re-anchor interim->permanent hires to their season-opening
    # year (reuses Coach_WAR's Primary_Coach resolution). Must precede the tenure
    # recompute and feature build so both use the anchored hire year.
    df = clean_population(df)
    df = df[df["Year"] >= MIN_HIRE_YEAR].copy()
    if verbose:
        print(f"Loaded {n_all} instances; kept {len(df)} hires from {MIN_HIRE_YEAR}+")

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
            if (coach, hire_year) in MIDSEASON_OUTSIDE_HIRES:
                feat["cf_internal_hire"] = 0
                feat["cf_years_at_hiring_org"] = 0
            if INCLUDE_TIER56:
                feat.update(tier5_features(hist, hire_year, hc_hist, coach))
                feat.update(tier6_features(hist, hire_year))
                feat.update(team_quality_features(hist, hire_year))
        raw_hist = load_history(coach, dedupe=False)
        feat.update(interim_features(raw_hist, hire_year, coach, ranks))
        feat.update(tier2_features(ranks, hire_year, raw_hist, coach))
        rows.append(feat)

    new_df = pd.DataFrame(rows, index=df.index)

    # Structural indicators for the pooled unit block. A coach who never led an NFL
    # unit (no OC, DC or non-interim HC season) has no unit record: the statistics
    # are undefined, not unobserved. A coach whose record predates PFR's drive data
    # has a unit record but no drive record. Paired with a constant fill these let
    # the model estimate each group's hazard separately, so the filled values carry
    # no information and the unit coefficients come only from observed records.
    unit_cols, drive_cols = unit_block_columns()
    no_unit = df[unit_cols].isna().all(axis=1)
    new_df["cf_no_unit_record"] = no_unit.astype(int)
    new_df["cf_no_drive_record"] = (~no_unit & df[drive_cols].isna().all(axis=1)).astype(int)

    # Coverage flags for inherited-roster data, by the known coverage rule (rosters
    # begin in ROSTER_DATA_START), so any other missing roster value still surfaces.
    # Expansion hires inherit no roster at all and are covered by org_expansion.
    not_exp = new_df["org_expansion"] != 1
    new_df["hire_no_roster_data"] = (not_exp & (df["Year"] - 1 < ROSTER_DATA_START)).astype(int)
    new_df["hire_no_retention_data"] = (not_exp & (df["Year"] - 2 < ROSTER_DATA_START)).astype(int)
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
