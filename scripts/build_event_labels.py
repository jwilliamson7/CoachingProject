#!/usr/bin/env python
"""
Build firing-vs-voluntary event labels for the survival pivot.

New event definition (supersedes the fired/retired heuristic in classify_exit):
the survival EVENT is INVOLUNTARY NON-RETENTION -- the coach failed to keep the
job. The nominal exit label (fired / resigned / retired) is cosmetic and gameable
(teams launder firings as "mutual partings"; a 62-year-old whose contract quietly
lapses after a 5-12 season "retired" only on paper). So the default outcome is
EVENT=1. The ONLY censoring (event=0, "survived") cases are VOLUNTARY-WHILE-VIABLE
departures:
  - still active at the data boundary,
  - voluntarily left to coach another team (rare in the NFL, common in college),
  - walked away on top (genuinely retired while still wanted/winning),
  - left for non-football reasons (health, family, death in office).

This script attaches every deterministic signal to each stint, assigns a
best-effort auto-label with a confidence level, and flags the cases that need
human/web validation. Cases resolved with high confidence by record alone:
  - losing final season that ended the spell      -> EVENT  (non-retention)
  - PFR Notes say "fired"                          -> EVENT
  - still active at boundary                       -> CENSOR
Everything with a non-losing final season that ended the spell is genuinely
ambiguous (voluntary move / retire-on-top vs fired-despite-winning, e.g. Lovie
Smith 10-6) and is flagged needs_validation=True for the research pass.

Output: analysis/event_labels.csv

Usage:
    python scripts/build_event_labels.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season
from scripts.data.engineer_career_features import (
    reconstruct_tenure, load_history, classify_role, canon_employer,
    hiring_team_info, is_nfl, COACHES_DIR,
)

OUT = project_root / "analysis" / "event_labels.csv"

FIRED_KW = ["fired", "dismiss", "relieved", "let go", "terminated", "released"]
RESIGN_KW = ["resign", "stepped down", "step down", "mutual", "parted", "quit"]
# Notes that signal a genuinely voluntary / non-football exit
VOLUNTARY_NOTE_KW = ["retire", "health", "illness", "died", "death", "leave of absence"]


def load_results(coach):
    p = COACHES_DIR / coach / "all_coaching_results.csv"
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


def _abbr_to_fkey(abbr, year):
    from data_constants import standardize_team_abbreviation
    if not isinstance(abbr, str) or not abbr.strip():
        return None
    return "nfl:" + standardize_team_abbreviation(abbr.strip(), year)


def final_season_row(res, last, fkey):
    """The hiring-franchise result row for the stint's final season."""
    rows = res[res["Year"] == last]
    if len(rows) == 0:
        return None
    if len(rows) > 1 and fkey is not None:
        for _, r in rows.iterrows():
            if _abbr_to_fkey(str(r.get("Tm", "")), last) == fkey:
                return r
    return rows.iloc[-1]


def parse_winpct(row):
    """Final-season win pct from W-L% (fallback to W/L/T)."""
    v = pd.to_numeric(pd.Series([row.get("W-L%")]), errors="coerce").iloc[0]
    if pd.notna(v):
        return float(v)
    w = pd.to_numeric(pd.Series([row.get("W")]), errors="coerce").iloc[0]
    l = pd.to_numeric(pd.Series([row.get("L")]), errors="coerce").iloc[0]
    t = pd.to_numeric(pd.Series([row.get("T")]), errors="coerce").iloc[0]
    g = (w or 0) + (l or 0) + (t or 0)
    return float((w + 0.5 * (t or 0)) / g) if g else np.nan


def has_kw(note, kws):
    s = str(note).lower()
    return any(k in s for k in kws)


def career_signals(coach, last, fkey):
    """Post-exit coaching signals from the full history."""
    h = load_history(coach)
    coached_after = hc_elsewhere_next = hc_elsewhere_after = False
    if h is not None:
        for _, r in h.iterrows():
            y = int(r["Year"])
            if y <= last:
                continue
            role = classify_role(r.get("Role", ""))
            if role != "NONE":
                coached_after = True
            if (role == "HC" and is_nfl(r.get("Level"))):
                k = canon_employer(r.get("Employer", ""), r.get("Level"), y)
                if k != fkey:
                    hc_elsewhere_after = True
                    if y == last + 1:
                        hc_elsewhere_next = True
    return coached_after, hc_elsewhere_next, hc_elsewhere_after


def label_stint(coach, hire_year, boundary):
    t = reconstruct_tenure(coach, hire_year)
    if t is None:
        return None
    last = hire_year + t - 1
    name, level = hiring_team_info(load_history(coach), hire_year)
    fkey = canon_employer(name, level, hire_year) if name else None

    res = load_results(coach)
    fr = final_season_row(res, last, fkey) if res is not None else None
    winpct = parse_winpct(fr) if fr is not None else np.nan
    notes = str(fr.get("Notes", "")) if fr is not None else ""
    notes = "" if notes == "nan" else notes
    age = (pd.to_numeric(pd.Series([fr.get("Age")]), errors="coerce").iloc[0]
           if fr is not None else np.nan)

    coached_after, hc_next, hc_after = career_signals(coach, last, fkey)

    note_fired = has_kw(notes, FIRED_KW)
    note_resign = has_kw(notes, RESIGN_KW)
    note_voluntary = has_kw(notes, VOLUNTARY_NOTE_KW)
    losing = pd.notna(winpct) and winpct < 0.500

    # ---- decision tree under the new default-event rule ----
    if last >= boundary:
        label, etype, conf, flag = "censor", "active", "high", False
    elif note_fired:
        label, etype, conf, flag = "event", "fired_notes", "high", False
    elif note_resign:
        # forced/face-saving resignation defaults to EVENT, but verify (could be voluntary)
        label, etype, conf, flag = "event", "resigned_notes", "low", True
    elif losing:
        label, etype, conf, flag = "event", "losing_finale", "high", False
    else:
        # non-losing final season that ended the spell: the censoring candidates.
        # provisional guess from the move signal, but always flag for validation.
        if hc_next:
            label, etype, conf, flag = "censor", "moved_team?", "low", True
        elif note_voluntary:
            label, etype, conf, flag = "censor", "voluntary_note?", "low", True
        else:
            label, etype, conf, flag = "censor", "nonlosing_left?", "low", True

    return {
        "coach": coach, "hire_year": hire_year, "duration": t, "last_year": last,
        "final_winpct": round(winpct, 3) if pd.notna(winpct) else np.nan,
        "final_record": (f"{int(fr.get('W'))}-{int(fr.get('L'))}"
                         f"{('-' + str(int(fr.get('T')))) if pd.notna(fr.get('T')) and int(fr.get('T'))>0 else ''}"
                         if fr is not None and pd.notna(fr.get("W")) else ""),
        "age_at_exit": int(age) if pd.notna(age) else np.nan,
        "notes": notes,
        "coached_after": int(coached_after),
        "hc_elsewhere_next": int(hc_next),
        "hc_elsewhere_after": int(hc_after),
        "auto_label": label, "exit_type": etype, "confidence": conf,
        "needs_validation": flag,
        "resolved_label": "", "resolved_reason": "",  # filled by research pass
    }


def main():
    df, X, y = load_modeling_data()          # known-tenure, 1970+, class != -1
    boundary = global_max_season()
    print(f"Censoring boundary season = {boundary}")

    rows = []
    for c, yr in zip(df["Coach Name"], df["Year"]):
        r = label_stint(c, int(yr), boundary)
        if r is not None:
            rows.append(r)
    out = pd.DataFrame(rows).sort_values(["needs_validation", "last_year"],
                                         ascending=[False, True])

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    n = len(out)
    ev = (out["auto_label"] == "event").sum()
    ce = (out["auto_label"] == "censor").sum()
    flag = out["needs_validation"].sum()
    print(f"\n{n} stints | event(default) {ev} | censor {ce} | "
          f"censor rate {ce/n:.3f}")
    print(f"needs_validation flagged: {flag}")
    print("\nby exit_type:")
    print(out.groupby(["auto_label", "exit_type", "confidence"]).size()
          .to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
