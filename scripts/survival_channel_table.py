#!/usr/bin/env python
"""
Hiring-channel regressions for the JSE revision (economics-style tables).

The headline model (survival_definitive.py) is the one stability selection keeps: the
internal-hire indicator alone. This script asks how that estimate moves as blocks of
controls and franchise baselines are added, and whether it holds in the subsamples and
alternative codings the referees raised. Inference only: no selection, no splits. Every
fit is the cause-specific Cox of survival_methods.cox_hazard_ratios, unpenalized, with
standard errors clustered on the coach; a feature's structural indicators enter with it
(with_structural_indicators), and a column that is constant in a subsample is dropped
from that fit.

  Table A (blocks): (1) internal hire; (2) + the coach's credentials; (3) + the
      inherited situation; (4) + franchise strata (a baseline hazard per franchise,
      conditioned out of the partial likelihood, so 32 franchises cost no parameters);
      (5) + labor-market era indicators (salary cap 1994, Rooney Rule 2003).
  Table B (robustness of the internal-hire HR, bare and with the column-3 controls):
      franchise strata, era strata, first stint per coach (the recurrence check, in
      place of a shared frailty), first-time NFL head coaches, hires from 1994 and from
      2003, the four mid-season outside hires coded internal as PFR's titles imply, any
      exit as the event, and a voluntary exit as the event.
  Table C (succession, P4): the internal-hire HR after a fired predecessor and after
      one who left voluntarily.
  Table D (hiring side, descriptive logits clustered on franchise): which situations
      promote from within (P5), and whether a defensive coach follows a weak defense.

Usage:
    python scripts/survival_channel_table.py
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

from model.pipeline import varying_columns
from scripts.survival_methods import (
    inference_sample, stint_franchise, cox_hazard_ratios, with_structural_indicators,
    friendly_name, nm, _FRIENDLY_TO_COL, ERA_BREAKS, CENSORED, FIRED, VOLUNTARY,
)
from scripts.data.engineer_career_features import MIDSEASON_OUTSIDE_HIRES

warnings.filterwarnings("ignore")

OUT_PKL = os.path.join(project_root, "analysis", "survival_channel_table.pkl")
INTERNAL = "cf_internal_hire"
# The coach's record at hire. Years at the hiring organization is left out: it is
# positive for every internal hire by construction.
CREDENTIALS = ["age", "num_times_hc", "num_yr_nfl_coor", "num_yr_col_hc",
               "cf_pre_side_off", "rf_avg_unit_pctl"]
# The situation he inherits, all measured before the hire.
INHERITED = ["tq_srs", "tq_srs_traj", "org_prev_fired", "org_unique_hc_10yr",
             "hire_roster_av"]


def col(name):
    """Column label of a variable (base features are 'Feature N' placeholders)."""
    return _FRIENDLY_TO_COL.get(name, name)


def fmt(r):
    return f"{r['HR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}]"


class Fitter:
    """Cause-specific Cox fits on the shared inference sample."""

    def __init__(self):
        self.df, self.X, self.Xf, self.dur, self.cause = inference_sample()
        self.coach = self.df["Coach Name"].values
        self.franchise = stint_franchise(self.df).values
        self.year = self.df["Year"].values.astype(int)

    def fit(self, feats, rows=None, strata=None, cause=None, Xf=None):
        """HR table + fit facts for `feats` (variable names) plus their indicators."""
        rows = np.ones(len(self.df), bool) if rows is None else np.asarray(rows, bool)
        cols = with_structural_indicators([col(f) for f in feats], self.X.loc[rows])
        Xm = (self.Xf if Xf is None else Xf).loc[rows, cols]
        Xm = Xm.loc[:, varying_columns(Xm.values)]
        cause = self.cause if cause is None else cause
        hr, cph = cox_hazard_ratios(Xm, self.dur[rows], cause[rows], self.coach[rows],
                                    penalizer=0.0,
                                    strata=None if strata is None else np.asarray(strata)[rows])
        hr.index = [nm(c) for c in hr.index]
        facts = {"n": int(rows.sum()), "events": int((np.asarray(cause)[rows] == FIRED).sum()),
                 "clusters": int(pd.Series(self.coach[rows]).nunique()),
                 "loglik": float(cph.log_likelihood_),
                 "covariates": [nm(c) for c in Xm.columns]}
        return hr, facts


def table_a(F):
    """Blocks of controls, then franchise strata, then era indicators."""
    post = {name: (F.year >= brk).astype(int) for name, brk in ERA_BREAKS.items()}
    Xf_era = F.Xf.assign(**post)
    specs = [("(1)", [INTERNAL], None, False),
             ("(2)", [INTERNAL] + CREDENTIALS, None, False),
             ("(3)", [INTERNAL] + CREDENTIALS + INHERITED, None, False),
             ("(4)", [INTERNAL] + CREDENTIALS + INHERITED, F.franchise, False),
             ("(5)", [INTERNAL] + CREDENTIALS + INHERITED + list(ERA_BREAKS), F.franchise, True)]
    res = {}
    for lab, feats, strata, era in specs:
        res[lab] = F.fit(feats, strata=strata, Xf=Xf_era if era else None)
        res[lab][1].update(franchise_strata=strata is not None, era=era)

    rows = [INTERNAL] + CREDENTIALS + INHERITED + list(ERA_BREAKS)
    width = 24
    print("=" * 150)
    print("TABLE A. Cause-specific Cox for firing: hazard ratios [95% CI, clustered on coach]")
    print("=" * 150)
    print(f"  {'':<34}" + "".join(f"{lab:^{width}}" for lab in res))
    for v in rows:
        cells = []
        for lab, (hr, _) in res.items():
            cells.append(fmt(hr.loc[v]) if v in hr.index else "")
        label = friendly_name(col(v)) if v not in ERA_BREAKS else f"Hired {ERA_BREAKS[v]} or later"
        print(f"  {label[:34]:<34}" + "".join(f"{c:^{width}}" for c in cells))
    for key, lab in [("n", "Stints"), ("events", "Firings"), ("clusters", "Coaches (clusters)")]:
        print(f"  {lab:<34}" + "".join(f"{res[m][1][key]:^{width}}" for m in res))
    print(f"  {'Log partial likelihood':<34}"
          + "".join(f"{res[m][1]['loglik']:^{width}.1f}" for m in res))
    print(f"  {'Franchise strata':<34}"
          + "".join(f"{('yes' if res[m][1]['franchise_strata'] else 'no'):^{width}}" for m in res))
    print(f"  {'Era indicators':<34}"
          + "".join(f"{('yes' if res[m][1]['era'] else 'no'):^{width}}" for m in res))
    ind = sorted({c for m in res for c in res[m][1]["covariates"]} - set(rows))
    print(f"  Structural indicators entered with their features (not shown): {ind}")
    print("  Note: partial likelihoods of stratified and unstratified fits use different risk "
          "sets and are not comparable across that line.")
    return res


def table_b(F):
    """The internal-hire HR under alternative samples, strata, codings and events."""
    ctrl = [INTERNAL] + CREDENTIALS + INHERITED
    era = np.digitize(F.year, sorted(ERA_BREAKS.values()))
    first = F.year == pd.Series(F.year).groupby(F.coach).transform("min").values
    first_time = F.Xf[col("num_times_hc")].values == 0
    pfr = F.Xf.copy()
    recode = [(c, y) in MIDSEASON_OUTSIDE_HIRES for c, y in zip(F.coach, F.year)]
    pfr.loc[recode, INTERNAL] = 1
    c = np.asarray(F.cause)
    any_exit = pd.Series(np.where(c == VOLUNTARY, FIRED, c), index=F.cause.index)
    voluntary = pd.Series(np.select([c == VOLUNTARY, c == FIRED], [FIRED, VOLUNTARY], CENSORED),
                          index=F.cause.index)
    specs = [
        ("Baseline", {}),
        ("Franchise strata", {"strata": F.franchise}),
        ("Era strata (1994, 2003)", {"strata": era}),
        ("First stint per coach", {"rows": first}),
        ("First-time NFL head coaches", {"rows": first_time}),
        ("Hired 1994 or later", {"rows": F.year >= ERA_BREAKS["salary_cap"]}),
        ("Hired 2003 or later", {"rows": F.year >= ERA_BREAKS["rooney_rule"]}),
        (f"{sum(recode)} mid-season outside hires coded internal", {"Xf": pfr}),
        ("Event: any exit", {"cause": any_exit}),
        ("Event: voluntary exit", {"cause": voluntary}),
    ]
    # 29 voluntary exits cannot carry the 14 column-3 covariates (about two events each)
    bare_only = {"Event: voluntary exit"}
    print("\n" + "=" * 150)
    print("TABLE B. Internal-hire HR [95% CI] across samples, strata, codings and events")
    print("=" * 150)
    print(f"  {'specification':<44}{'n':>5}{'events':>8}{'internal':>10}"
          f"{'no controls':^28}{'p':>7}{'column-3 controls':^28}{'p':>7}")
    res = {}
    for lab, kw in specs:
        hr0, f0 = F.fit([INTERNAL], **kw)
        hr1, f1 = (None, None) if lab in bare_only else F.fit(ctrl, **kw)
        rows = kw.get("rows", np.ones(len(F.df), bool))
        n_int = int((kw.get("Xf", F.Xf)[INTERNAL].values[np.asarray(rows, bool)] == 1).sum())
        res[lab] = {"bare": (hr0, f0), "controls": (hr1, f1), "n_internal": n_int}
        a = hr0.loc[INTERNAL]
        right = (f"{'(too few events)':^28}{'':>7}" if hr1 is None else
                 f"{fmt(hr1.loc[INTERNAL]):^28}{hr1.loc[INTERNAL, 'p']:>7.3f}")
        print(f"  {lab:<44}{f0['n']:>5}{f0['events']:>8}{n_int:>10}"
              f"{fmt(a):^28}{a['p']:>7.3f}" + right)
    print("  'events' counts the event of that row (firings unless stated).")
    return res


def table_c(F):
    """Internal hire after a fired vs a voluntarily departed predecessor (P4)."""
    pf_raw = F.X["org_prev_fired"]
    rows = pf_raw.notna().values                  # hires with a predecessor
    pf = pf_raw.fillna(0).values.astype(int)
    ih = F.Xf[INTERNAL].values.astype(int)
    Xc = F.Xf.assign(internal_after_fired=ih * pf, internal_after_departed=ih * (1 - pf),
                     internal_x_prev_fired=ih * pf)
    print("\n" + "=" * 150)
    print("TABLE C. Succession: internal-hire HR by how the predecessor left (hires with a predecessor)")
    print("=" * 150)
    for g, lab in [(1, "predecessor fired"), (0, "predecessor left voluntarily")]:
        m = rows & (pf == g)
        print(f"  {lab:<30} hires {int(m.sum()):>4}  internal {int(ih[m].sum()):>3}  "
              f"firings {int((np.asarray(F.cause)[m] == FIRED).sum()):>4}")
    res = {}
    for tag, extra in [("no controls", []), ("column-3 controls", CREDENTIALS + INHERITED)]:
        others = [v for v in extra if v != "org_prev_fired"]
        hr_s, f_s = F.fit(["internal_after_fired", "internal_after_departed", "org_prev_fired"]
                          + others, rows=rows, Xf=Xc)
        hr_i, _ = F.fit([INTERNAL, "org_prev_fired", "internal_x_prev_fired"] + others,
                        rows=rows, Xf=Xc)
        res[tag] = {"split": (hr_s, f_s), "interaction": hr_i}
        print(f"  [{tag}] n={f_s['n']} firings={f_s['events']}")
        for v, lab in [("internal_after_fired", "internal | predecessor fired"),
                       ("internal_after_departed", "internal | predecessor left voluntarily")]:
            r = hr_s.loc[v]
            print(f"    {lab:<42} HR {fmt(r)}  p={r['p']:.3f}")
        r = hr_i.loc["internal_x_prev_fired"]
        print(f"    {'ratio of the two (interaction)':<42} HR {fmt(r)}  p={r['p']:.3f}")
    return res


def _logit(y, Xd, groups):
    m = sm.Logit(y, sm.add_constant(Xd)).fit(disp=0, cov_type="cluster",
                                              cov_kwds={"groups": groups})
    ci = np.exp(m.conf_int())
    out = pd.DataFrame({"OR": np.exp(m.params), "CI_low": ci[0], "CI_high": ci[1],
                        "p": m.pvalues, "sd": Xd.std()})
    return out.reindex(list(Xd.columns))


def table_d(F):
    """Hiring-side logits (descriptive), clustered on franchise."""
    fr_codes = pd.factorize(F.franchise)[0]
    print("\n" + "=" * 150)
    print("TABLE D. Hiring side (descriptive logits, odds ratios per unit [95% CI], "
          "clustered on franchise; complete cases)")
    print("=" * 150)
    raw = F.X.assign(**{name: (F.year >= brk).astype(int) for name, brk in ERA_BREAKS.items()})
    sel = INHERITED + list(ERA_BREAKS)
    d = raw[sel + [INTERNAL]].dropna()
    g = fr_codes[raw.index.get_indexer(d.index)]
    which = _logit(d[INTERNAL].astype(int), d[sel], g)
    print(f"  (i) Who promotes from within? y = internal hire; n={len(d)}, "
          f"internal={int(d[INTERNAL].sum())}, franchises={len(set(g))}")
    for v, r in which.iterrows():
        label = friendly_name(col(v)) if v not in ERA_BREAKS else f"Hired {ERA_BREAKS[v]} or later"
        print(f"    {label:<34} OR {r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}]"
              f"  p={r['p']:.3f}  (sd {r['sd']:.2f})")

    sided = (F.X["cf_pre_side_off"] + F.X["cf_pre_side_def"]) == 1
    d2 = F.X.loc[sided, ["cf_pre_side_def", "tq_dsrs", "tq_osrs"]].dropna()
    g2 = fr_codes[F.X.index.get_indexer(d2.index)]
    side = _logit(d2["cf_pre_side_def"].astype(int), d2[["tq_dsrs", "tq_osrs"]], g2)
    print(f"  (ii) Does a defensive coach follow a weak defense? y = hired from a defensive "
          f"role; hires from a side-specific role, n={len(d2)}, "
          f"defensive={int(d2['cf_pre_side_def'].sum())}")
    for v, r in side.iterrows():
        print(f"    {friendly_name(v):<34} OR {r['OR']:.3f} [{r['CI_low']:.3f}, {r['CI_high']:.3f}]"
              f"  p={r['p']:.3f}  (sd {r['sd']:.2f})")
    return {"who_promotes": which, "n_who": len(d), "side_match": side, "n_side": len(d2)}


def main():
    F = Fitter()
    c = np.asarray(F.cause)
    print(f"{len(F.df)} stints, {int((c == FIRED).sum())} fired, "
          f"{int((c == VOLUNTARY).sum())} voluntary, {int((c == CENSORED).sum())} active; "
          f"{pd.Series(F.coach).nunique()} coaches, {pd.Series(F.franchise).nunique()} franchises\n")
    out = {"A": table_a(F), "B": table_b(F), "C": table_c(F), "D": table_d(F),
           "credentials": CREDENTIALS, "inherited": INHERITED, "era_breaks": ERA_BREAKS}
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved {OUT_PKL}")


if __name__ == "__main__":
    main()
