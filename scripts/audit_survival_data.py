#!/usr/bin/env python
"""
Stint-integrity audit for the firing-survival dataset.

Diagnostic only: this does NOT touch the modeling pipeline. It cross-checks the
survival stint table (data/master_data_extended.csv + reconstruct_tenure + the
event labels) against the original ordinal table (data/master_data.csv) to find
data-derivation problems before any camera-ready re-run.

Checks
  1. Overlapping stints      same coach, two stints whose [start, end] seasons
                             overlap -> a double-counted tenure.
  2. Forward over-count      reconstruct_tenure for a stint runs past the start
                             of that same coach's NEXT stint -> duration merges
                             two records.
  3. Relocation spanning     stint active in a season the franchise relocated
                             (the split trigger).
  4. Continuation flag       stint coded internal_hire=1 while the coach was the
                             real head coach the prior season -> a relocation /
                             mid-season split miscoded as a fresh internal hire.
  5. Duration vs original    reconstruct_tenure class disagrees with the original
                             per-segment Coach Tenure Class.
  6. Event coding            firing / voluntary / active-censored counts + the
                             boundary rule, for sanity.
  7. Composition             378 vs 357 (known) vs 21 (class -1) vs original 656.

Usage:  python scripts/audit_survival_data.py
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season
from scripts.survival_methods import build_competing_targets, FIRED, VOLUNTARY, CENSORED
from scripts.data.engineer_career_features import reconstruct_tenure, load_history

# Single canonical dataset (the former master_data.csv base + master_data_extended.csv
# final were merged into one file). EXT and ORIG now point at the same table, so
# check [5] becomes a stored-class vs reconstruct_tenure consistency check.
EXT = os.path.join(project_root, "data", "master_data.csv")
ORIG = os.path.join(project_root, "data", "master_data.csv")

# Season the franchise first played in its new city (relocation split trigger).
RELOCATIONS = {
    1982: "Raiders Oakland->LA", 1984: "Colts Baltimore->Indianapolis",
    1988: "Cardinals St.Louis->Phoenix", 1995: "Rams LA->St.Louis / Raiders LA->Oakland",
    1997: "Oilers Houston->Tennessee", 2016: "Rams St.Louis->LA",
    2017: "Chargers San Diego->LA", 2020: "Raiders Oakland->Las Vegas",
}


def real_hc_prior(coach, yr):
    h = load_history(coach)
    if h is None:
        return False
    prev = h[h["Year"] == yr - 1]
    for r in prev["Role"]:
        s = str(r).strip()
        if "Assistant" in s or "Asst" in s or "Interim" in s:
            continue
        if s == "Head Coach" or s.startswith("Head Coach/") or s.startswith("Head Coach "):
            return True
    return False


def main():
    ext = pd.read_csv(EXT, index_col=0)
    orig = pd.read_csv(ORIG, index_col=0)

    # Build the stint table with durations.
    rec = []
    for _, r in ext.iterrows():
        c, yr = r["Coach Name"], int(r["Year"])
        dur = reconstruct_tenure(c, yr)
        if dur is None:
            continue
        rec.append({"coach": c, "start": yr, "dur": int(dur),
                    "end": yr + int(dur) - 1,
                    "internal": int(r["cf_internal_hire"]),
                    "tclass": int(r["Coach Tenure Class"])})
    s = pd.DataFrame(rec)

    print(f"survival stint rows: {len(s)}")
    print("=" * 70)

    # 1. Overlapping stints --------------------------------------------------
    print("\n[1] OVERLAPPING STINTS (double-counted tenures)")
    overlaps = []
    for c, g in s.groupby("coach"):
        rows = g.sort_values("start").to_dict("records")
        for a, b in zip(rows, rows[1:]):
            if b["start"] <= a["end"]:
                overlaps.append((c, a["start"], a["end"], b["start"], b["end"], b["internal"]))
    for c, as_, ae, bs, be, ih in overlaps:
        print(f"  {c:22s} [{as_}-{ae}] overlaps [{bs}-{be}]  dup.internal={ih}")
    print(f"  -> {len(overlaps)} overlapping pairs")

    # 2. Forward over-count past next record --------------------------------
    print("\n[2] DURATION RUNS PAST NEXT STINT (forward over-count)")
    over = []
    for c, g in s.groupby("coach"):
        rows = g.sort_values("start").to_dict("records")
        for a, b in zip(rows, rows[1:]):
            if a["end"] >= b["start"]:
                over.append((c, a["start"], a["end"], b["start"]))
    for c, as_, ae, bs in over:
        print(f"  {c:22s} stint {as_} runs to {ae}, but next stint starts {bs}")
    print(f"  -> {len(over)} over-counts")

    # 3. Relocation spanning -------------------------------------------------
    print("\n[3] STINTS ACTIVE IN A RELOCATION SEASON")
    reloc = []
    for _, r in s.iterrows():
        for yr, desc in RELOCATIONS.items():
            if r["start"] <= yr <= r["end"]:
                reloc.append((r["coach"], r["start"], r["end"], yr, desc))
    for c, st, en, yr, desc in reloc:
        print(f"  {c:22s} [{st}-{en}] spans {yr} ({desc})")
    print(f"  -> {len(reloc)} stint-relocation spans")

    # 4. Continuation flag ---------------------------------------------------
    print("\n[4] internal_hire=1 WHILE ALREADY HEAD COACH THE PRIOR SEASON")
    cont = []
    for _, r in s.iterrows():
        if r["internal"] == 1 and real_hc_prior(r["coach"], r["start"]):
            cont.append((r["coach"], r["start"]))
    for c, yr in cont:
        print(f"  {c:22s} {yr}")
    print(f"  -> {len(cont)} suspicious internal-hire stints")

    # 5. Duration vs original class -----------------------------------------
    print("\n[5] reconstruct_tenure CLASS != original Coach Tenure Class")
    def cls(d):
        return 0 if d <= 2 else (1 if d <= 4 else 2)
    mism = []
    for _, r in s.iterrows():
        o = orig[(orig["Coach Name"] == r["coach"]) & (orig["Year"] == r["start"])]
        if not len(o):
            continue
        oc = int(o["Coach Tenure Class"].iloc[0])
        if oc >= 0 and cls(r["dur"]) != oc:
            mism.append((r["coach"], r["start"], oc, r["dur"], cls(r["dur"])))
    for c, yr, oc, dur, rc in mism:
        print(f"  {c:22s} {yr}  orig_class={oc}  recon_dur={dur} (class {rc})")
    print(f"  -> {len(mism)} class disagreements")

    # 6. Event coding --------------------------------------------------------
    print("\n[6] EVENT CODING (competing risks)")
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    vc = cause.value_counts().to_dict()
    print(f"  boundary season = {boundary}")
    print(f"  fired (1)            = {vc.get(FIRED, 0)}")
    print(f"  voluntary (2)        = {vc.get(VOLUNTARY, 0)}")
    print(f"  active-censored (0)  = {vc.get(CENSORED, 0)}")
    print(f"  total                = {sum(vc.values())}")

    # 7. Composition ---------------------------------------------------------
    print("\n[7] SAMPLE COMPOSITION")
    print(f"  extended rows (known_only=False) : {len(ext)}")
    print(f"  class -1 (recent active)         : {int((ext['Coach Tenure Class'] == -1).sum())}")
    print(f"  known (known_only=True)          : {int((ext['Coach Tenure Class'] != -1).sum())}")
    print(f"  original master_data.csv rows    : {len(orig)}")
    print(f"  original NFL HC rows (class>=-1)  : {int((orig['Coach Tenure Class'] >= -1).sum())}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  overlapping/double-counted stints : {len(overlaps)}")
    print(f"  forward over-counts               : {len(over)}")
    print(f"  duration/class disagreements      : {len(mism)}")
    print(f"  suspicious internal-hire stints   : {len(cont)}")


if __name__ == "__main__":
    main()
