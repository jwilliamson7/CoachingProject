#!/usr/bin/env python
"""
In-tenure performance and the internal-hire firing premium (mechanism check, JSE).

Does the premium run through worse results (insiders' teams play worse: the
match-quality channel, P2) or through a shorter leash (insiders are fired sooner at
equal results: the threshold channel, P3)? We follow every stint season by season and
fit a discrete-time proportional-hazards model for firing (complementary log-log link
with its own baseline hazard for each tenure year: the grouped-time version of the Cox
model, Prentice & Gloeckler 1978), with the team's Simple Rating System that season as
a time-varying covariate. Standard errors are clustered on the coach.

SRS (Pro-Football-Reference) is a team's average scoring margin plus the average SRS of
its opponents: points per game better or worse than a league-average team, adjusted for
schedule. A coach's SRS in a season is his team's SRS that season. Its lag in his first
season is the franchise's SRS the season before he was hired (tq_srs); an expansion
team has none (set to 0 and flagged).

Descriptive, not causal: in-tenure SRS is partly the coach's own output, a mediator of
the hiring channel, so conditioning on it splits the premium into a part that travels
with results and a part that does not; it does not identify either causally. A coach
fired mid-season is scored on his team's full-season SRS, which includes his
successor's games, so model (5) uses only the season before (known when the season
starts) as a check.

Team records run through 2024, so the panel ends with the 2024 season: a stint still
running then is censored there, and 2025 hires drop out.

  (1) internal + tenure-year baselines                (reproduces the stint-level HR)
  (2) + SRS this season
  (3) + SRS this season + SRS last season
  (4) (3) + internal x SRS this season                 (a different slope for insiders)
  (5) internal + SRS last season only                  (predetermined performance)
  (6) internal in tenure years 1-2 vs 3+, without and with (3)'s SRS terms (P2's early
      concentration)
  (7) internal after a fired vs a voluntarily departed predecessor, without and with
      (3)'s SRS terms (exploratory: 19 firings follow internal hires after a voluntary
      exit)
  OLS: SRS this season on internal + SRS last season + tenure year: do insiders' teams
      play worse at equal starting points?

Usage:
    python scripts/survival_tenure_srs.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scripts.survival_methods import inference_sample, cox_hazard_ratios, FIRED, CENSORED
from scripts.data.engineer_career_features import (load_history, _hiring_franchise_abbrs,
                                                   _team_rec_row)

warnings.filterwarnings("ignore")

OUT_PKL = os.path.join(project_root, "analysis", "survival_tenure_srs.pkl")
PANEL_LAST = 2024        # last season in Teams/<abbr>/team_record.csv
EARLY = 2                # tenure years counted as "early" in model (6)
K_CAP = 8                # tenure years 8 and later share one baseline


def build_panel(df, dur, cause, last=PANEL_LAST):
    """One row per stint-season through `last`: SRS that season and the season before,
    and whether the coach was fired at the end of it."""
    rows = []
    for i in dur.index:
        coach, hy = df.at[i, "Coach Name"], int(df.at[i, "Year"])
        T, c = int(dur[i]), int(cause[i])
        hist = load_history(coach)
        lag = df.at[i, "tq_srs"]
        for k in range(1, T + 1):
            s = hy + k - 1
            if s > last:
                break
            rec = _team_rec_row(_hiring_franchise_abbrs(hist, s), s)
            srs = pd.to_numeric(pd.Series([None if rec is None else rec.get("SRS")]),
                                errors="coerce").iloc[0]
            rows.append({"stint": i, "coach": coach, "hire_year": hy, "season": s, "k": k,
                         "internal": int(df.at[i, "cf_internal_hire"]),
                         "srs": srs, "srs_lag": lag,
                         "fired": int(c == FIRED and k == T)})
            lag = srs
    p = pd.DataFrame(rows)
    p["no_lag"] = p["srs_lag"].isna().astype(int)
    p["srs_lag"] = p["srs_lag"].fillna(0.0)
    return p


def _design(p, cols):
    fe = pd.get_dummies(p["k"].clip(upper=K_CAP), prefix="year", drop_first=True, dtype=float)
    return sm.add_constant(pd.concat([p[cols].astype(float), fe], axis=1))


def cloglog(p, cols):
    """Grouped-time PH model for firing; exp(coef) is a hazard ratio."""
    groups = pd.factorize(p["coach"])[0]
    fam = sm.families.Binomial(link=sm.families.links.CLogLog())
    return sm.GLM(p["fired"], _design(p, cols), family=fam).fit(
        cov_type="cluster", cov_kwds={"groups": groups})


def hr_row(m, v):
    b, se = m.params[v], m.bse[v]
    return {"HR": np.exp(b), "CI_low": np.exp(b - 1.96 * se),
            "CI_high": np.exp(b + 1.96 * se), "p": m.pvalues[v]}


def show(label, r, unit=""):
    print(f"    {label:<40} HR {r['HR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}]"
          f"  p={r['p']:.3f}{unit}")


def main():
    df, X, Xf, dur, cause = inference_sample()
    p = build_panel(df, dur, cause)
    # how the predecessor left (expansion hires have none, and none is internal)
    p["prev_fired"] = p["stint"].map(X["org_prev_fired"].fillna(0)).astype(int)
    miss = p["srs"].isna()
    print(f"panel: {len(p)} coach-seasons from {p['stint'].nunique()} stints "
          f"({p['coach'].nunique()} coaches), seasons {p['season'].min()}-{p['season'].max()}; "
          f"{int(p['fired'].sum())} firings; SRS missing in {int(miss.sum())} seasons; "
          f"{int(p['no_lag'].sum())} first seasons with no prior SRS (expansion)")
    if miss.any():
        print(p[miss][["coach", "season"]].to_string())
        raise SystemExit("in-tenure SRS missing; fix the franchise lookup first")
    print(f"SRS: mean {p['srs'].mean():+.2f}, sd {p['srs'].std():.2f} (points per game)")
    by = p.groupby("internal")[["srs", "fired"]].mean()
    print(f"  mean SRS: internal {by.loc[1, 'srs']:+.2f}, external {by.loc[0, 'srs']:+.2f}; "
          f"firing rate per season: internal {by.loc[1, 'fired']:.3f}, "
          f"external {by.loc[0, 'fired']:.3f}\n")

    # stint-level Cox on the same window, for comparison with model (1)
    st = p.groupby("stint").agg(T=("k", "max"), E=("fired", "max"),
                                internal=("internal", "first"), coach=("coach", "first"))
    hr_cox, _ = cox_hazard_ratios(st[["internal"]], st["T"],
                                  np.where(st["E"] == 1, FIRED, CENSORED), st["coach"].values,
                                  penalizer=0.0)

    p["internal_x_srs"] = p["internal"] * p["srs"]
    p["internal_early"] = p["internal"] * (p["k"] <= EARLY)
    p["internal_late"] = p["internal"] * (p["k"] > EARLY)
    p["internal_after_fired"] = p["internal"] * p["prev_fired"]
    p["internal_after_departed"] = p["internal"] * (1 - p["prev_fired"])
    lags = ["srs_lag", "no_lag"]
    specs = {"(1)": ["internal"],
             "(2)": ["internal", "srs"],
             "(3)": ["internal", "srs"] + lags,
             "(4)": ["internal", "srs", "internal_x_srs"] + lags,
             "(5)": ["internal"] + lags,
             "(6a)": ["internal_early", "internal_late"],
             "(6b)": ["internal_early", "internal_late", "srs"] + lags,
             "(7a)": ["internal_after_fired", "internal_after_departed", "prev_fired"],
             "(7b)": ["internal_after_fired", "internal_after_departed", "prev_fired",
                      "srs"] + lags}
    fits = {k: cloglog(p, v) for k, v in specs.items()}

    print("=" * 100)
    print("Firing hazard per coach-season (cloglog, tenure-year baselines, clustered on coach)")
    print("=" * 100)
    r = hr_cox.loc["internal"]
    print(f"  stint-level Cox, same window: internal HR {r['HR']:.3f} "
          f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]")
    labels = {"internal": "internal hire", "srs": "SRS this season (per point)",
              "srs_lag": "SRS last season (per point)",
              "internal_x_srs": "internal x SRS this season",
              "internal_early": f"internal, tenure years 1-{EARLY}",
              "internal_late": f"internal, tenure years {EARLY + 1}+",
              "internal_after_fired": "internal, predecessor fired",
              "internal_after_departed": "internal, predecessor left voluntarily",
              "prev_fired": "predecessor fired"}
    out = {}
    for k, m in fits.items():
        print(f"  {k}  n={int(m.nobs)}  log-likelihood {m.llf:.1f}")
        out[k] = {v: hr_row(m, v) for v in specs[k] if v != "no_lag"}
        for v in specs[k]:
            if v != "no_lag":
                show(labels[v], out[k][v])
        for a, b_, tag in (("internal_early", "internal_late", "early vs late"),
                           ("internal_after_fired", "internal_after_departed",
                            "after fired vs after departed")):
            if a in specs[k]:
                t = m.t_test(f"{a} - {b_} = 0")
                out[k][f"{tag} p"] = float(t.pvalue)
                print(f"    {tag + ' (ratio)':<40} HR {np.exp(float(t.effect)):.3f}"
                      f"  p={float(t.pvalue):.3f}")
    m4 = fits["(4)"]
    for s_ in (-6.0, 0.0, 6.0):
        b = m4.params["internal"] + s_ * m4.params["internal_x_srs"]
        print(f"    (4) internal HR at SRS {s_:+.0f}: {np.exp(b):.3f}")

    # do insiders' teams play worse at equal starting points?
    print("\n" + "=" * 100)
    print("OLS: SRS this season on internal hire (tenure-year effects, clustered on coach)")
    print("=" * 100)
    ols = {}
    for tag, sub, cols in [("all seasons", p, ["internal"] + lags),
                           ("first season only", p[p["k"] == 1], ["internal"] + lags),
                           ("all seasons, no lag", p, ["internal"])]:
        g = pd.factorize(sub["coach"])[0]
        Xd = _design(sub, cols)
        Xd = Xd.loc[:, Xd.std() > 0].assign(const=1.0)
        m = sm.OLS(sub["srs"].astype(float), Xd).fit(cov_type="cluster", cov_kwds={"groups": g})
        ci = m.conf_int().loc["internal"]
        ols[tag] = {"b": m.params["internal"], "CI_low": ci[0], "CI_high": ci[1],
                    "p": m.pvalues["internal"], "n": int(m.nobs)}
        print(f"  {tag:<22} n={int(m.nobs):>5}  internal: {m.params['internal']:+.2f} points/game "
              f"[{ci[0]:+.2f}, {ci[1]:+.2f}]  p={m.pvalues['internal']:.3f}"
              + (f"  (SRS last season {m.params['srs_lag']:+.2f})" if "srs_lag" in cols else ""))

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"panel": p, "hazard": out, "ols": ols, "cox_same_window": hr_cox,
                     "panel_last": PANEL_LAST, "early": EARLY, "k_cap": K_CAP}, f)
    print(f"\nSaved {OUT_PKL}")


if __name__ == "__main__":
    main()
