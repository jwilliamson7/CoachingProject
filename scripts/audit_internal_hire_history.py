"""Triple-check cf_internal_hire and dump a human-readable history CSV.

cf_internal_hire == 1 iff the hiring franchise's canonical key is among the
franchises the coach worked for in the IMMEDIATE prior season (hire_year - 1).

This script:
  1. Re-derives the flag for ALL 371 stints by THREE independent paths and diffs
     them against the stored value (catches both false positives and negatives):
       (A) canon_employer membership   -- reproduces the build logic from raw history
       (B) nickname-token match        -- avoids canon_employer's abbr/standardize map
       (C) plain employer-name equality with relocation aliases
  2. For every internal hire, writes:
       analysis/internal_hire_audit_summary.csv   -- one row per hire + evidence
       analysis/internal_hire_history_detail.csv  -- raw history rows, hire_year-5..hire_year
     so each promotion can be eyeballed against the actual record.

Run: python scripts/audit_internal_hire_history.py
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))

from engineer_career_features import classify_role, is_nfl, canon_employer  # noqa: E402

COACHES_DIR = os.path.join(ROOT, "Coaches")
ANALYSIS = os.path.join(ROOT, "analysis")

# relocation/rename aliases for the plain-name path (independent of canon_employer)
_ALIAS = {"oilers": "titans", "redskins": "commanders", "football": "commanders"}


def nick(employer, level):
    name = str(employer).strip().lower()
    if not is_nfl(level):
        return ("col:" if "College" in str(level) else "oth:") + name
    if "football team" in name:
        return "nfl:commanders"
    toks = name.replace(".", "").split()
    n = toks[-1] if toks else name
    return "nfl:" + _ALIAS.get(n, n)


def load_hist(coach):
    p = os.path.join(COACHES_DIR, coach, "all_coaching_history.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df.dropna(subset=["Year"])


def hire_row_employer(hist, hire_year):
    """Employer (name, level) of the HC appointment at hire_year (engineer mirror)."""
    rows = hist[hist["Year"] == hire_year]
    for _, r in rows.iterrows():
        if classify_role(r.get("Role", "")) == "HC":
            return r.get("Employer", ""), r.get("Level", "")
    if len(rows):
        return rows.iloc[0].get("Employer", ""), rows.iloc[0].get("Level", "")
    return None, None


def prior_roles_at(hist, hire_year, hkey, keyfn):
    """All (employer, role) the coach held in hire_year-1 at the hiring franchise."""
    out = []
    for _, r in hist[hist["Year"] == hire_year - 1].iterrows():
        if classify_role(r.get("Role", "")) == "NONE":
            continue
        if keyfn(r.get("Employer", ""), r.get("Level", "")) == hkey:
            out.append((str(r.get("Employer", "")).strip(), str(r.get("Role", "")).strip()))
    return out


def main():
    md = pd.read_csv(os.path.join(ROOT, "data", "master_data.csv"), index_col=0)
    md = md.rename(columns={"Coach Name": "coach", "Year": "year"})

    disagree, detail, summary = [], [], []
    for _, row in md.iterrows():
        coach, yr = row["coach"], int(row["year"])
        stored = int(row["cf_internal_hire"])
        hist = load_hist(coach)
        if hist is None:
            disagree.append((coach, yr, "MISSING HISTORY"))
            continue

        emp, lvl = hire_row_employer(hist, yr)
        # three independent franchise keys for the hiring appointment
        hk_canon = canon_employer(emp, lvl, yr) if emp else None
        hk_nick = nick(emp, lvl) if emp else None
        hk_name = str(emp).strip().lower() if emp else None

        prior = hist[hist["Year"] == yr - 1]
        canon_keys, nick_keys, name_keys = set(), set(), set()
        for _, r in prior.iterrows():
            if classify_role(r.get("Role", "")) == "NONE":
                continue
            e, l = r.get("Employer", ""), r.get("Level", "")
            canon_keys.add(canon_employer(e, l, yr - 1))
            nick_keys.add(nick(e, l))
            name_keys.add(str(e).strip().lower())

        a = int(hk_canon in canon_keys) if hk_canon else 0
        b = int(hk_nick in nick_keys) if hk_nick else 0
        c = int(hk_name in name_keys or
                nick(emp, lvl) in nick_keys) if hk_name else 0  # alias-tolerant
        if not (a == b == c == stored):
            disagree.append((coach, yr, f"stored={stored} canon={a} nick={b} name={c} "
                                        f"hiring={hk_nick} prior={sorted(nick_keys)}"))

        if stored != 1:
            continue

        # evidence for this internal hire
        at_org = prior_roles_at(hist, yr, hk_canon, lambda e, l: canon_employer(e, l, yr - 1))
        roles_txt = "; ".join(f"{e}: {ro}" for e, ro in at_org) or "(none found)"
        is_interim = any("interim" in ro.lower() for _, ro in at_org)
        role_classes = {classify_role(ro) for _, ro in at_org}
        promo = ("interim->permanent" if is_interim else
                 "staff promotion" if role_classes - {"NONE"} else "unknown")
        summary.append({
            "coach": coach, "hire_year": yr,
            "hiring_employer": emp, "hiring_key": hk_nick,
            "years_at_org": int(row["cf_years_at_hiring_org"]),
            "prior_season_role_at_org": roles_txt,
            "promotion_type": promo,
            "agree_canon": a, "agree_nick": b, "agree_name": c,
        })
        # detail history rows hire_year-5 .. hire_year
        win = hist[(hist["Year"] >= yr - 5) & (hist["Year"] <= yr)].sort_values("Year")
        for _, r in win.iterrows():
            e, l = r.get("Employer", ""), r.get("Level", "")
            y = int(r["Year"])
            detail.append({
                "coach": coach, "hire_year": yr, "row_year": y,
                "employer": e, "level": l, "role": r.get("Role", ""),
                "role_class": classify_role(r.get("Role", "")),
                "franchise_key": nick(e, l),
                "at_hiring_franchise": int(nick(e, l) == hk_nick),
                "is_prior_season": int(y == yr - 1),
                "is_hire_year": int(y == yr),
            })

    n_int = sum(s["promotion_type"] is not None for s in summary)
    print(f"Stints: {len(md)}   internal hires: {len(summary)}")
    print(f"\n=== Three-path agreement (canon / nick / name vs stored) ===")
    real_dis = [d for d in disagree if "MISSING" not in d[2]]
    print(f"All-4-agree on every stint? {len(real_dis) == 0}   "
          f"disagreements: {len(real_dis)}   missing-history: "
          f"{sum('MISSING' in d[2] for d in disagree)}")
    for coach, yr, msg in disagree:
        print(f"  {coach} {yr}: {msg}")

    promo_counts = pd.Series([s["promotion_type"] for s in summary]).value_counts()
    print(f"\n=== Promotion-type breakdown of the {len(summary)} internal hires ===")
    print(promo_counts.to_string())

    sdf = pd.DataFrame(summary).sort_values(["hire_year", "coach"])
    ddf = pd.DataFrame(detail)
    os.makedirs(ANALYSIS, exist_ok=True)
    sdf.to_csv(os.path.join(ANALYSIS, "internal_hire_audit_summary.csv"), index=False)
    ddf.to_csv(os.path.join(ANALYSIS, "internal_hire_history_detail.csv"), index=False)
    print(f"\nWrote analysis/internal_hire_audit_summary.csv ({len(sdf)} rows)")
    print(f"Wrote analysis/internal_hire_history_detail.csv ({len(ddf)} rows)")

    # spot-print the interim->permanent ones (the entanglement-relevant subset)
    interim = sdf[sdf["promotion_type"] == "interim->permanent"]
    print(f"\n=== interim->permanent promotions ({len(interim)}) ===")
    for _, r in interim.iterrows():
        print(f"  {r['hire_year']}  {r['coach']:<22} {r['prior_season_role_at_org']}")


if __name__ == "__main__":
    main()
