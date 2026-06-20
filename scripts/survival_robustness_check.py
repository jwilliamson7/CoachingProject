#!/usr/bin/env python
"""
Two pre-submission robustness checks for the definitive firing-survival analysis.
Reuses the existing loaders + cox_hazard_ratios; does NOT re-run selection or
duplicate the split/impute/select/fit block (the locked stable set is read from
analysis/survival_definitive.pkl).

  (a) durb == dur : the duration used by the PREDICTIVE eval (build_survival_targets)
      is provably identical, row for row, to the duration used by the INFERENTIAL
      model (build_competing_targets) on the shared index.
  (b) Unpenalized HR table : refit the cause-specific cluster-robust Cox at
      penalizer=0 and compare to the reported penalizer=0.1 ridge. The reported
      p-values/CIs should be materially unchanged (EPV ~64, ~uncorrelated feats).

Usage:
    python scripts/survival_robustness_check.py
"""

import os
import sys
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season, build_survival_targets
from scripts.data.matrix_factorization_imputation import SVDImputer
from scripts.survival_methods import build_competing_targets, cox_hazard_ratios, ph_test
from scripts.survival_definitive import nm

PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")


def main():
    # ---- reproduce the definitive data prep exactly ----
    df0, X0, y0 = load_modeling_data()
    boundary = global_max_season()
    dur, cause = build_competing_targets(df0, boundary)
    keep = dur.index
    df, X = df0.loc[keep], X0.loc[keep]
    dur, cause = dur.loc[keep], cause.loc[keep]
    durb, _ = build_survival_targets(df, boundary)
    durb = durb.loc[keep]

    # ---- check (a): durations identical ----
    aligned = durb.reindex(keep)
    same = bool((aligned.values.astype(float) == dur.values.astype(float)).all())
    n_mismatch = int((aligned.values.astype(float) != dur.values.astype(float)).sum())
    print("=" * 64)
    print("(a) duration identity: predictive (durb) vs inferential (dur)")
    print("=" * 64)
    print(f"  n rows compared : {len(keep)}")
    print(f"  identical       : {same}  (mismatches: {n_mismatch})")
    if not same:
        bad = keep[aligned.values.astype(float) != dur.values.astype(float)]
        print("  MISMATCH rows:")
        for i in bad[:10]:
            print(f"    {df.loc[i,'Coach Name']:<22} yr={df.loc[i,'Year']}"
                  f"  durb={aligned.loc[i]}  dur={dur.loc[i]}")

    # ---- check (b): unpenalized vs ridge HR table ----
    stable = pickle.load(open(PKL, "rb"))["stable"]
    cols = list(X.columns)
    stable_idx = [cols.index(c) for c in stable]
    Ximp = SVDImputer().fit(X.values).transform(X.values)
    Xsel = pd.DataFrame(Ximp[:, stable_idx], columns=stable, index=X.index)
    coach = df["Coach Name"].values

    hr_ridge, _ = cox_hazard_ratios(Xsel, dur, cause, coach, penalizer=0.1)
    hr_unpen, _ = cox_hazard_ratios(Xsel, dur, cause, coach, penalizer=0.0)
    hr_ridge.index = [nm(c) for c in hr_ridge.index]
    hr_unpen.index = [nm(c) for c in hr_unpen.index]
    hr_unpen = hr_unpen.reindex(hr_ridge.index)

    print("\n" + "=" * 64)
    print("(b) HR table: ridge (penalizer=0.1) vs unpenalized (penalizer=0)")
    print("=" * 64)
    hdr = f"  {'feature':<22}{'HR_ridge':>9}{'p_ridge':>9}{'HR_unpen':>10}{'p_unpen':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for f in hr_ridge.index:
        r, u = hr_ridge.loc[f], hr_unpen.loc[f]
        print(f"  {f:<22}{r['HR']:>9.3f}{r['p']:>9.3f}"
              f"{u['HR']:>10.3f}{u['p']:>9.3f}")

    print("\n  unpenalized 95% CI (cluster-robust):")
    for f in hr_unpen.index:
        u = hr_unpen.loc[f]
        print(f"    {f:<22} HR {u['HR']:.3f}  [{u['CI_low']:.3f}, {u['CI_high']:.3f}]"
              f"  p={u['p']:.3f}")

    # PH test at penalizer=0 for completeness
    pht = ph_test(Xsel, dur, cause, penalizer=0.0)
    print(f"\n  PH test (unpenalized): all p>0.05? {bool((pht['p']>0.05).all())}"
          f"  (min p={pht['p'].min():.3f})")

    # magnitude of drift
    drift = (hr_unpen["HR"] - hr_ridge["HR"]).abs()
    print(f"\n  max |HR drift| ridge->unpen: {drift.max():.3f}"
          f"  ({drift.idxmax()})")

    # ---- check (c): collinearity among the stable features (justifies penalizer=0) ----
    # If features are ~uncorrelated and VIF is low, the unpenalized MLE is stable
    # and the ridge has no legitimate stabilizing role for the reported inference.
    print("\n" + "=" * 64)
    print("(c) collinearity of the 5 stable features (unpenalized MLE safe?)")
    print("=" * 64)
    Z = Xsel.values.astype(float)
    names = [nm(c) for c in Xsel.columns]
    corr = np.corrcoef(Z, rowvar=False)
    off = corr - np.eye(len(names))
    iu = np.triu_indices(len(names), k=1)
    amax = np.abs(off[iu]).max()
    apair = (iu[0][np.abs(off[iu]).argmax()], iu[1][np.abs(off[iu]).argmax()])
    print("  pairwise |correlation| matrix:")
    print("      " + "".join(f"{j:>8}" for j in range(len(names))))
    for i, n in enumerate(names):
        print(f"  {i} {n[:14]:<14}" + "".join(f"{abs(corr[i,j]):>8.2f}" for j in range(len(names))))
    print(f"  max |off-diagonal corr| = {amax:.3f}  "
          f"({names[apair[0]]} vs {names[apair[1]]})")
    vif = []
    for j in range(Z.shape[1]):
        yj = Z[:, j]
        Xo = np.column_stack([np.ones(len(yj)), np.delete(Z, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xo, yj, rcond=None)
        ss_res = ((yj - Xo @ beta) ** 2).sum()
        ss_tot = ((yj - yj.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif.append(np.inf if r2 >= 1 else 1.0 / (1.0 - r2))
    print("  variance inflation factors (VIF; >5 = concerning, >10 = severe):")
    for n, v in zip(names, vif):
        print(f"    {n:<22} {v:>6.2f}")
    print(f"  max VIF = {max(vif):.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
