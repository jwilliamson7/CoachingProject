#!/usr/bin/env python
"""
Nested (selection-inside-the-split) evaluation of the firing-survival model (JSE).

survival_definitive.py chooses the feature set ONCE on all stints (Cox-native
stability selection, and the CV-argmax comparison set is ranked on the full data
too), then scores that fixed set over the 50 coach-level splits. The held-out
stints therefore helped pick the features they are scored on, so its concordance
and Brier numbers are optimistic (Hastie et al., ESL 7.10.2; Ambroise &
McLachlan 2002; Varma & Simon 2006). This script reruns every data-dependent step
inside each split's TRAINING partition only:

  - stability selection: same B, q, pi, and retention rule (choose_stable) as the
    headline. With no floor, a training partition can select nothing; that split's
    model is then the covariate-free null, scored as such (C = 0.5, the Kaplan-Meier
    Brier score), which is what the rule would have shipped.
  - Cox penalty for the selected set: tuned by inner coach-level 5-fold CV over
    COX_SPACE, ties to the lightest penalty (tune_cox_penalty, as in the headline)
  - the model comparison, in which each model class chooses its own features on the
    training partition: Cox and the three parametric AFTs rank the candidates by their
    own ridge fit (cox_importance, aft_importance) and keep the top K, with K and the
    model's penalty chosen jointly by inner CV over K_GRID x its penalty grid (COX_GRID,
    AFT_SPACE; each set with its structural indicators); an elastic-net Cox and XGB-AFT
    take every candidate. Internal hire alone is the reference row.

and then scores the held-out partition: Harrell and Uno C for every comparison row,
the IPCW integrated Brier score of the stability-selected Cox, and the same Brier for a
Kaplan-Meier null. The reported final model (its features and hazard ratios) is
unchanged; choosing it on all the data is correct. Only the performance estimate has to
be nested.

The all-candidate rows keep the headline settings, tuned once on the full data (the
canonical elastic net for Cox, the XGB-AFT hyperparameters), so they are nested in
everything but that tuning. Splits run in parallel, one process per split.

Usage:
    python scripts/survival_nested_eval.py [--boots 2000] [--workers 14]
"""

import os

# One BLAS thread per worker process (set before numpy loads, so children inherit it).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import pickle
import argparse
import warnings
from multiprocessing import Pool

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index

from model.pipeline import load_modeling_data, leakage_free_split
from model.config import MODEL_CONFIG
from scripts.survival_analysis import global_max_season, build_survival_targets, nb_tci
from scripts.survival_models import (
    cox_builder, weibull_builder, lognormal_builder, loglogistic_builder,
    xgbaft_builder, N_SEEDS, COX_GRID, AFT_SPACE,
)
from scripts.survival_feature_selection import full_cv_folds, cv_cindex_at_k, tune_cox_penalty
from scripts.survival_combined_features import K_GRID
from scripts.compare_ordinal_multiclass import nadeau_bengio_ttest
from scripts.survival_methods import (
    build_competing_targets, firing_event, drop_redundant_features, cox_importance,
    stability_selection_multi, uno_cindex, _km_censoring, cox_brier, km_null_brier,
    with_structural_indicators, selectable, choose_stable, nm, aft_importance,
    STAB_THR, BRIER_GRID, UNO_TAU,
)
from scripts.survival_definitive import STAB_K, AFT_P

warnings.filterwarnings("ignore")

DEF_PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")
OUT_PKL = os.path.join(project_root, "analysis", "survival_nested_eval.pkl")

_D = {}  # per-process data, filled by _init


def _init():
    """Load and prepare data exactly as survival_definitive.main() does."""
    warnings.filterwarnings("ignore")
    df, X, y = load_modeling_data(known_only=False)
    boundary = global_max_season()
    dur, cause = build_competing_targets(df, boundary)
    keep = dur.index
    df, X, y, dur, cause = df.loc[keep], X.loc[keep], y.loc[keep], dur.loc[keep], cause.loc[keep]
    X = drop_redundant_features(X)
    durb, _ = build_survival_targets(df, boundary)
    durb = durb.loc[keep]
    evt = pd.Series(firing_event(cause), index=keep)
    defn = pickle.load(open(DEF_PKL, "rb"))
    _D.update(df=df, X=X, y=y, dur=dur, durb=durb, evt=evt,
              cox_fixed=defn["cox_params"], xgb_p=defn["xgb_params"])


# The model comparison: (result key, label). Each model class chooses its own features.
ROWS = [("internal", "Internal hire alone (Cox)"),
        ("Cox", "Cox, stability selection (headline)"),
        ("CV-argmax Cox", "Cox, own rank, K by inner CV"),
        ("Cox all", "Cox, all candidates (elastic net)"),
        ("Weibull AFT", "Weibull AFT, own rank, K by inner CV"),
        ("LogNormal AFT", "LogNormal AFT, own rank, K by inner CV"),
        ("LogLogistic AFT", "LogLogistic AFT, own rank, K by inner CV"),
        ("XGB-AFT", "XGB-AFT, all candidates")]
AFT_FITTERS = {"Weibull AFT": (weibull_builder, WeibullAFTFitter),
               "LogNormal AFT": (lognormal_builder, LogNormalAFTFitter),
               "LogLogistic AFT": (loglogistic_builder, LogLogisticAFTFitter)}
AFT_GRID = [{"penalizer": p} for p in AFT_SPACE["penalizer"]]


def _own_k(builder, grid, folds, rank, cols, Xtr_raw):
    """(K, penalty, column indices) chosen jointly by inner coach-level CV on a model's
    own ranking, over K_GRID x `grid` (ties to the smaller K, then the lighter
    penalty); each candidate set enters with its structural indicators."""
    sets = {K: [cols.index(c) for c in
                with_structural_indicators([cols[i] for i in rank[:K]], Xtr_raw)]
            for K in K_GRID}
    scored = [(round(cv_cindex_at_k(builder, p, folds, idx), 6), -K, -p["penalizer"], K, j)
              for K, idx in sets.items() for j, p in enumerate(grid)]
    *_, K, j = max(scored)
    return K, grid[j], sets[K]


def _score(builder, params, Xtr, Ttr, Etr, Xte, Tte, Ete, kmf):
    m = builder(params)
    if hasattr(m, "params") and isinstance(m.params, dict):
        m.params["nthread"] = 1                       # XGB-AFT: stay on one core
    m.fit(Xtr, Ttr, Etr)
    s = np.asarray(m.surv_score(Xte)).ravel()         # higher = longer survival
    return concordance_index(Tte, s, Ete), uno_cindex(-s, Tte, Ete, kmf, tau=UNO_TAU)


def run_split(args):
    seed, n_boot = args
    t0 = time.time()
    df, X, y, dur, durb, evt = (_D[k] for k in ("df", "X", "y", "dur", "durb", "evt"))
    cols = list(X.columns)
    sp = leakage_free_split(df, X, y, seed)           # imputer fit on train only
    tr, te = sp.train_index, sp.test_index
    Ttr, Etr = durb.loc[tr].values.astype(float), evt.loc[tr].values.astype(int)
    Tte, Ete = durb.loc[te].values.astype(float), evt.loc[te].values.astype(int)

    # 1. stability selection on the training partition (headline rule)
    freq = stability_selection_multi(df.loc[tr], X.loc[tr], dur.loc[tr], evt.loc[tr],
                                     Ks=(STAB_K,), n_boot=n_boot, subsample=0.5, seed=seed,
                                     importance_fn=cox_importance)[STAB_K]
    stable = choose_stable(freq, STAB_THR)          # the headline's rule, no floor
    model_feats = with_structural_indicators(stable, X.loc[tr])
    s_idx = [cols.index(c) for c in model_feats]

    # 2. Cox penalty by inner coach-level CV on the training partition (none when
    #    nothing was selected: there is no model to tune)
    folds = full_cv_folds(df.loc[tr], X.loc[tr], y.loc[tr], durb.loc[tr], evt.loc[tr])
    cox_tuned = tune_cox_penalty(folds, s_idx, COX_GRID) if s_idx else None

    # 3. each comparison model class chooses its own features on the training partition
    #    (its own rank, structural indicators only as companions; K by inner CV)
    sets, ks, pars = {}, {}, {}
    rank = selectable(cox_importance(sp.X_train, Ttr, Etr)[0], cols)
    ks["CV-argmax Cox"], pars["CV-argmax Cox"], sets["CV-argmax Cox"] = _own_k(
        cox_builder, COX_GRID, folds, rank, cols, X.loc[tr])
    for name, (builder, fitter) in AFT_FITTERS.items():
        rank = selectable(aft_importance(fitter, sp.X_train, Ttr, Etr)[0], cols)
        ks[name], pars[name], sets[name] = _own_k(builder, AFT_GRID, folds, rank, cols,
                                                  X.loc[tr])
    sets["Cox all"] = sets["XGB-AFT"] = list(range(len(cols)))
    sets["internal"] = [cols.index("cf_internal_hire")]
    pars.update({"internal": COX_GRID[0], "Cox all": _D["cox_fixed"], "XGB-AFT": _D["xgb_p"]})

    # 4. score the held-out partition
    kmf = _km_censoring(Ttr, Etr)
    Xs_tr, Xs_te = sp.X_train[:, s_idx], sp.X_test[:, s_idx]
    out = {"seed": seed, "stable": stable, "cox_tuned": cox_tuned,
           "k_best": ks["CV-argmax Cox"], "k": ks, "params": pars}
    out["ibs_km"], out["bs_km"] = km_null_brier(Ttr, Etr, Tte, Ete, BRIER_GRID, kmf)
    # the null model ranks no one: every pair is a tie, scored 0.5 by both indices
    out["Cox"] = (_score(cox_builder, cox_tuned, Xs_tr, Ttr, Etr, Xs_te, Tte, Ete, kmf)
                  if s_idx else (0.5, 0.5))
    for name, b in [("internal", cox_builder), ("CV-argmax Cox", cox_builder),
                    ("Cox all", cox_builder), ("Weibull AFT", weibull_builder),
                    ("LogNormal AFT", lognormal_builder),
                    ("LogLogistic AFT", loglogistic_builder), ("XGB-AFT", xgbaft_builder)]:
        idx = sets[name]
        out[name] = _score(b, pars[name], sp.X_train[:, idx], Ttr, Etr, sp.X_test[:, idx],
                           Tte, Ete, kmf)
    if s_idx:
        out["ibs_cox"], out["bs_cox"] = cox_brier(cox_tuned, Xs_tr, Ttr, Etr, Xs_te, Tte,
                                                  Ete, BRIER_GRID, kmf)
    else:
        out["ibs_cox"], out["bs_cox"] = out["ibs_km"], out["bs_km"]
    out["secs"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    tf = MODEL_CONFIG["test_size"]
    print(f"Nested evaluation: {a.seeds} splits x {a.boots} stability subsamples, "
          f"{a.workers} workers\n", flush=True)

    t0, res = time.time(), []
    with Pool(a.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(run_split, [(s, a.boots) for s in range(a.seeds)]):
            res.append(r)
            print(f"  split {r['seed']:2d}: Cox C={r['Cox'][0]:.3f}  IBS={r['ibs_cox']:.4f} "
                  f"(KM {r['ibs_km']:.4f})  internal={'cf_internal_hire' in r['stable']}  "
                  f"K={r['k_best']}  [{r['secs']:.0f}s; {len(res)}/{a.seeds} at {time.time()-t0:.0f}s]",
                  flush=True)
    res.sort(key=lambda r: r["seed"])

    defn = pickle.load(open(DEF_PKL, "rb"))
    print("\nModel comparison, NESTED: each model class chooses its own features inside every "
          "training partition (Nadeau-Bengio 95% CI)")
    print(f"  {'model':<42}{'Harrell C':>22}{'Uno C':>22}{'vs internal alone':>26}{'K (median)':>12}")
    base = np.array([r["internal"][0] for r in res])
    for key, label in ROWS:
        hc = np.array([r[key][0] for r in res])
        h, u = nb_tci(hc, tf), nb_tci([r[key][1] for r in res], tf)
        if key == "internal":
            vs = ""
        else:
            dm, _, dp = nadeau_bengio_ttest(hc - base, tf)
            vs = f"{dm:+.3f} (p={dp:.2f})"
        kk = [r["k"][key] for r in res if key in r["k"]]
        kmed = f"{int(np.median(kk))}" if kk else ""
        print(f"  {label:<42}{h[0]:>9.3f} [{h[1]:.3f},{h[2]:.3f}]{u[0]:>9.3f} "
              f"[{u[1]:.3f},{u[2]:.3f}]{vs:>26}{kmed:>12}")

    cx = np.array([r["Cox"][0] for r in res])
    ck = np.array([r["CV-argmax Cox"][0] for r in res])
    m, t, p = nadeau_bengio_ttest(cx - ck, tf)
    print(f"\nStable set vs CV-argmax set (both nested), Cox Harrell: delta={m:+.4f} t={t:+.3f} p={p:.3g}"
          f"  (CV-argmax {nb_tci(ck, tf)[0]:.3f}; K chosen: median {int(np.median([r['k_best'] for r in res]))})")

    ibs_c, ibs_k = np.array([r["ibs_cox"] for r in res]), np.array([r["ibs_km"] for r in res])
    c, k = nb_tci(ibs_c, tf), nb_tci(ibs_k, tf)
    m, t, p = nadeau_bengio_ttest(ibs_k - ibs_c, tf)
    print(f"\nIntegrated Brier, NESTED: Cox {c[0]:.4f} [{c[1]:.4f},{c[2]:.4f}]   KM null {k[0]:.4f} "
          f"[{k[1]:.4f},{k[2]:.4f}]   (published Cox {defn['ibs_ci'][0]:.4f})")
    print(f"  KM - Cox = {m:+.4f}, t={t:.2f}, p={p:.3g}; skill vs KM {1 - c[0] / k[0]:.3f}")
    for lab, key in (("Cox", "bs_cox"), ("KM ", "bs_km")):
        curve = np.mean([r[key] for r in res], axis=0)
        print(f"  {lab} Brier by horizon:",
              {int(g): round(float(b), 3) for g, b in zip(BRIER_GRID, curve)})

    sel = pd.Series([nm(c) for r in res for c in r["stable"]]).value_counts()
    n_null = sum(not r["stable"] for r in res)
    print(f"\nFeatures selected across {len(res)} training partitions "
          f"({n_null} selected nothing and were scored as the null model):")
    for f, n in sel.head(10).items():
        print(f"  {f:<28}{n:>3}")
    pen = pd.Series([str(r["cox_tuned"]) for r in res if r["cox_tuned"] is not None]
                    ).value_counts()
    print("\nInner-CV Cox penalty chosen:")
    for p_, n in pen.items():
        print(f"  {p_:<44}{n:>3}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump({"splits": res, "n_boot": a.boots, "test_size": tf}, f)
    print(f"\nSaved {OUT_PKL}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
