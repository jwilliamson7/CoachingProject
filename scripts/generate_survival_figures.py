#!/usr/bin/env python
"""
Generate the four JQAS firing-survival figures from analysis/survival_definitive.pkl
(the Cox-native pipeline output). Saved to ijcss/figures/ at 150 dpi to match the
existing survival_km.png convention.

  1. survival_stability.png    Cox stability-selection frequency bar (the steep
                               drop-off that justifies the tight stable core).
  2. survival_forest.png       Cause-specific firing hazard-ratio forest plot
                               (cluster-robust 95% CI; significant vs not).
  3. survival_calibration.png  IPCW time-dependent Brier curve + integrated Brier.
  4. survival_cif.png          Competing-risks cumulative incidence: firing vs
                               voluntary exit (Aalen-Johansen, 95% CI bands).

Run AFTER survival_definitive.py (which writes the pkl).

Usage:
    python scripts/generate_survival_figures.py
"""

import os
import sys
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from model.pipeline import load_modeling_data
from scripts.survival_analysis import global_max_season
from scripts.survival_methods import (
    build_competing_targets, aalen_johansen_cif, FIRED, VOLUNTARY,
)

FIGDIR = os.path.join(project_root, "ijcss", "figures")
PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")
SIG, NS = "#1f77b4", "#9aa0a6"     # significant / not-significant colors
REF = "#d62728"                    # reference-line red


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def fig_stability(d, top=18):
    freq = d["stab_freq_named"].head(top)[::-1]   # largest on top in barh
    thr = d["stab_thr"]
    colors = [SIG if v >= thr else NS for v in freq.values]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.barh(range(len(freq)), freq.values, color=colors)
    ax.set_yticks(range(len(freq)))
    ax.set_yticklabels(freq.index, fontsize=8)
    ax.axvline(thr, ls="--", color=REF, alpha=0.8,
               label=f"stable threshold = {thr:g}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Selection frequency (200 coach-level subsamples)")
    ax.set_title("Cox-native stability selection")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "survival_stability.png")


def fig_forest(d):
    hr = d["hazard_ratios"].sort_values("HR")
    n = len(hr)
    fig, ax = plt.subplots(figsize=(7, 0.62 * n + 1.6))
    for i, (name, r) in enumerate(hr.iterrows()):
        c = SIG if r["p"] < 0.05 else NS
        ax.plot([r["CI_low"], r["CI_high"]], [i, i], color=c, lw=2.2, zorder=2)
        ax.plot(r["HR"], i, "o", color=c, ms=8, zorder=3)
        ax.text(1.02, i, f"{r['HR']:.2f} [{r['CI_low']:.2f}, {r['CI_high']:.2f}]"
                         f"  p={r['p']:.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color="black")
    ax.axvline(1.0, ls="--", color=REF, alpha=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.7, 1.0, 1.5, 2.0])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    lo = min(hr["CI_low"].min() * 0.9, 0.5)
    hi = max(hr["CI_high"].max() * 1.1, 2.0)
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(n))
    ax.set_yticklabels(hr.index, fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Hazard ratio for firing (95% CI, cluster-robust)")
    ax.set_title("Cause-specific firing hazard ratios")
    _save(fig, "survival_forest.png")


def fig_calibration(d):
    g = np.asarray(d["brier_grid"], float)
    bs = np.asarray(d["brier_curve"], float)
    ibs = d["ibs_ci"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(g, bs, "o-", color=SIG, lw=2, label="IPCW Brier(t)")
    ax.axhline(0.25, ls="--", color=REF, alpha=0.8, label="uninformative (0.25)")
    ax.set_xlabel("Horizon (seasons since hire)")
    ax.set_ylabel("IPCW Brier score (lower = better)")
    ax.set_ylim(0, 0.30)
    ax.set_xlim(g.min(), g.max())
    ax.set_title(f"Time-dependent prediction error\n"
                 f"integrated Brier = {ibs[0]:.3f} [{ibs[1]:.3f}, {ibs[2]:.3f}]")
    ax.legend(loc="upper right", frameon=False)
    _save(fig, "survival_calibration.png")


def fig_cif():
    df, X, y = load_modeling_data()
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    dur, cause = dur.loc[keep], cause.loc[keep]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ev, color, lab in [(FIRED, REF, "Firing"),
                           (VOLUNTARY, SIG, "Voluntary exit")]:
        aj = aalen_johansen_cif(dur, cause, ev)
        cdf = aj.cumulative_density_
        ci = aj.confidence_interval_
        t = cdf.index.values
        ax.step(t, cdf.iloc[:, 0].values, where="post", color=color, lw=2, label=lab)
        ax.fill_between(ci.index.values, ci.iloc[:, 0].values, ci.iloc[:, 1].values,
                        step="post", color=color, alpha=0.15)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seasons since hire")
    ax.set_ylabel("Cumulative incidence")
    ax.set_title("Competing-risks cumulative incidence")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "survival_cif.png")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    d = pickle.load(open(PKL, "rb"))
    fig_stability(d)
    fig_forest(d)
    fig_calibration(d)
    fig_cif()
    print("\nAll four survival figures generated.")


if __name__ == "__main__":
    main()
