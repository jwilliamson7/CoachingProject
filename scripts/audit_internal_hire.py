"""Validation probe for cf_internal_hire (the paper's headline predictor).

Two independent checks:
  (A) Re-derive an internal-hire flag for every stint using a nickname-based
      franchise normalizer that does NOT share canon_employer's abbr+standardize
      path, then diff against the stored cf_internal_hire. Disagreements are the
      relocation/rename edge cases (or bugs) to inspect by hand.
  (B) Curated ground-truth: known internal-promotion vs external hires.

Run: python scripts/audit_internal_hire.py
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))

from engineer_career_features import classify_role, is_nfl  # noqa: E402

COACHES_DIR = os.path.join(ROOT, "Coaches")

# Independent franchise key: collapse a full team name to a single stable
# nickname token, applying relocation/rename aliases so one franchise == one key.
# This deliberately avoids canon_employer's learned name->abbr map + year
# standardization, so agreement is real corroboration.
_ALIAS = {
    "oilers": "titans",
    "redskins": "commanders",
    "football": "commanders",   # "Washington Football Team"
}


def nick_key(employer, level):
    name = str(employer).strip().lower()
    if not is_nfl(level):
        return ("col:" if "College" in str(level) else "oth:") + name
    if "football team" in name:
        return "nfl:commanders"
    toks = name.replace(".", "").split()
    nick = toks[-1] if toks else name
    return "nfl:" + _ALIAS.get(nick, nick)


def load_hist(coach):
    p = os.path.join(COACHES_DIR, coach, "all_coaching_history.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df.dropna(subset=["Year"])


def hiring_nick(hist, hire_year):
    """Nickname key of the franchise this coach was hired by (mirror engineer's
    HC-row selection, but normalize via nick_key)."""
    rows = hist[hist["Year"] == hire_year]
    for _, r in rows.iterrows():
        if classify_role(r.get("Role", "")) == "HC":
            return nick_key(r.get("Employer", ""), r.get("Level", ""))
    if len(rows):
        return nick_key(rows.iloc[0].get("Employer", ""), rows.iloc[0].get("Level", ""))
    return None


def prior_nicks(hist, hire_year):
    """Nickname keys for all coaching roles in season hire_year-1."""
    rows = hist[hist["Year"] == hire_year - 1]
    keys = set()
    for _, r in rows.iterrows():
        if classify_role(r.get("Role", "")) != "NONE":
            keys.add(nick_key(r.get("Employer", ""), r.get("Level", "")))
    return keys


def main():
    md = pd.read_csv(os.path.join(ROOT, "data", "master_data.csv"), index_col=0)
    md = md.rename(columns={"Coach Name": "coach", "Year": "year"})
    n_internal = int(md["cf_internal_hire"].sum())
    print(f"Stints: {len(md)}   stored cf_internal_hire==1: {n_internal} "
          f"({100*n_internal/len(md):.1f}%)\n")

    mismatches, missing_hist, internal_rows = [], [], []
    for _, row in md.iterrows():
        coach, yr = row["coach"], int(row["year"])
        stored = int(row["cf_internal_hire"])
        hist = load_hist(coach)
        if hist is None:
            missing_hist.append(coach)
            continue
        hk = hiring_nick(hist, yr)
        indep = int(hk in prior_nicks(hist, yr)) if hk else 0
        if indep != stored:
            mismatches.append((coach, yr, stored, indep, hk,
                               sorted(prior_nicks(hist, yr))))
        if stored == 1:
            internal_rows.append((coach, yr, hk, int(row["cf_years_at_hiring_org"])))

    print(f"=== (A) Independent nickname re-derivation vs stored ===")
    print(f"Agreements: {len(md)-len(mismatches)-len(missing_hist)} / {len(md)}"
          f"   mismatches: {len(mismatches)}   missing history: {len(missing_hist)}")
    if missing_hist:
        print("  missing history files:", missing_hist)
    for coach, yr, stored, indep, hk, pk in mismatches:
        print(f"  MISMATCH {coach} {yr}: stored={stored} indep={indep} "
              f"hiring={hk} prior_season_keys={pk}")

    print(f"\n=== All {len(internal_rows)} stored internal hires "
          f"(coach, year, franchise, yrs_at_org) ===")
    for coach, yr, hk, yrs in sorted(internal_rows, key=lambda t: t[1]):
        print(f"  {yr}  {coach:<22} {hk:<18} yrs_at_org={yrs}")

    # (B) curated ground truth -------------------------------------------------
    # expected internal promotions (was on staff the prior season)
    GT = {
        ("Mike Tomlin", 2007): 0,        # from Vikings DC -> Steelers
        ("Bill Belichick", 2000): 0,     # from Jets -> Patriots
        ("Andy Reid", 1999): 0,          # from Packers -> Eagles
        ("Pete Carroll", 2010): 0,       # from USC -> Seahawks
        ("Sean McVay", 2017): 0,         # from Washington OC -> Rams
        ("Kyle Shanahan", 2017): 0,      # from Falcons OC -> 49ers
        ("Mike McCarthy", 2006): 0,      # from 49ers OC -> Packers
        ("Jason Garrett", 2011): 1,      # Cowboys interim 2010 -> permanent
        ("Mike Munchak", 2011): 1,       # Titans OL coach -> HC
        ("Raheem Morris", 2009): 1,      # Buccaneers DC -> HC
        ("Frank Reich", 2018): 0,        # Eagles OC -> Colts (external)
        ("Doug Pederson", 2016): 0,      # Chiefs OC -> Eagles (external)
        ("Anthony Lynn", 2017): 0,       # Bills interim/OC -> Chargers (external)
        ("Dan Quinn", 2015): 0,          # Seahawks DC -> Falcons (external)
        ("Zac Taylor", 2019): 0,         # Rams QB coach -> Bengals (external)
    }
    print("\n=== (B) Curated ground-truth checks ===")
    bad = 0
    idx = {(r["coach"], int(r["year"])): int(r["cf_internal_hire"])
           for _, r in md.iterrows()}
    for (coach, yr), exp in GT.items():
        got = idx.get((coach, yr))
        tag = "n/a (not in sample)" if got is None else ("OK" if got == exp else "*** WRONG ***")
        if got is not None and got != exp:
            bad += 1
        print(f"  {coach:<18} {yr}: expected={exp} got={got}  {tag}")
    print(f"\nGround-truth failures: {bad}")


if __name__ == "__main__":
    main()
