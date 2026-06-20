#!/usr/bin/env python
"""
Merge the web-research verdicts (analysis/event_verdicts.csv) into the
deterministic label file (analysis/event_labels.csv) to produce the final
survival event labels for the cleaned ~357-stint population.

Verdicts are keyed by (coach, last_year) -- stable under the interim->permanent
re-anchoring (which shifts hire_year but never last_year). Includes the user's
two confirmed overrides (Marrone 2014 -> event; Gruden 2001 -> censor) and keeps
Groh/Arians/Payton as censor per review.

event = 1 involuntary non-retention (firing/forced-out/pushed retirement);
event = 0 censored (voluntary-while-viable: moved team, retired on top, health,
or still active at the boundary).

Output: analysis/event_labels_final.csv

Usage:
    python scripts/merge_event_labels.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

ANALYSIS = project_root / "analysis"
BASE = ANALYSIS / "event_labels.csv"
VERDICTS = ANALYSIS / "event_verdicts.csv"
OUT = ANALYSIS / "event_labels_final.csv"


def main():
    base = pd.read_csv(BASE)
    verd = pd.read_csv(VERDICTS)
    vmap = {(r.coach, int(r.last_year)): r for r in verd.itertuples()}

    rows, missing = [], []
    for r in base.itertuples():
        key = (r.coach, int(r.last_year))
        if key in vmap:
            v = vmap[key]
            label = str(v.label).strip()
            rows.append({
                "label_source": "research",
                "resolved_label": label,
                "event": 1 if label == "event" else 0,
                "confidence": str(v.confidence).strip(),
                "reason": str(v.reason).strip(),
                "source": str(v.source).strip(),
            })
        else:
            label = r.auto_label  # deterministic: losing_finale / fired_notes / active
            rows.append({
                "label_source": "deterministic",
                "resolved_label": label,
                "event": 1 if label == "event" else 0,
                "confidence": "high",
                "reason": r.exit_type,
                "source": "record/Notes",
            })
            # any non-losing exit that ended a spell must have a verdict; flag gaps
            if bool(r.needs_validation):
                missing.append(key)

    add = pd.DataFrame(rows, index=base.index)
    keep = ["coach", "hire_year", "duration", "last_year", "final_record",
            "final_winpct", "age_at_exit", "notes", "hc_elsewhere_next",
            "hc_elsewhere_after"]
    final = pd.concat([base[keep], add], axis=1)
    final = final.sort_values(["event", "last_year"]).reset_index(drop=True)
    final.to_csv(OUT, index=False)

    n = len(final)
    ev = int((final.event == 1).sum())
    ce = int((final.event == 0).sum())
    print(f"{n} stints | EVENT {ev} ({ev/n:.1%}) | CENSOR {ce} ({ce/n:.1%})")
    print(f"label source: {dict(final['label_source'].value_counts())}")
    print("censor breakdown (event=0):")
    print(final[final.event == 0].groupby(["label_source", "resolved_label"])
          .size().to_string())
    if missing:
        print(f"\n[WARN] {len(missing)} flagged stints without a verdict "
              f"(fell back to deterministic):")
        for k in missing:
            print("   ", k)
    else:
        print("\nAll flagged (non-losing-exit) stints have a research verdict.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
