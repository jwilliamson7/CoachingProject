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

  - stability selection: same B, q, pi, and floor rule as the headline
  - Cox penalty: tuned by inner coach-level 5-fold CV over COX_SPACE
  - the CV-argmax comparison set: Cox-native rank + inner-CV sweep over K_GRID

and then scores the held-out partition: Harrell and Uno C for the bake-off models,
the IPCW integrated Brier score, and the same Brier for a Kaplan-Meier null. The
reported final model (its features and hazard ratios) is unchanged; choosing it
on all the data is correct. Only the performance estimate has to be nested.

The AFT penalty and the XGB-AFT hyperparameters stay at the headline values
(tuned once on the full data), so those two bake-off rows are nested in selection
only. Splits run in parallel, one process per split.

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
from itertools import product
from multiprocessing import Pool

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

from model.pipeline import load_modeling_data, leakage_free_split
from model.config import MODEL_CONFIG
from scripts.survival_analysis import global_max_season, build_survival_targets, nb_tci
from scripts.survival_models import (
    cox_builder, weibull_builder, lognormal_builder, loglogistic_builder,
    xgbaft_builder, N_SEEDS, COX_SPACE,
)
from scripts.survival_feature_selection import full_cv_folds, cv_cindex_at_k
from scripts.survival_combined_features import K_GRID
from scripts.compare_ordinal_multiclass import nadeau_bengio_ttest
from scripts.survival_methods import (
    build_competing_targets, firing_event, drop_redundant_features, cox_importance,
    stability_selection_multi, integrated_brier, uno_cindex, _km_censoring,
)
from scripts.survival_definitive import STAB_K, STAB_THR, BRIER_GRID, AFT_P, nm

warnings.filterwarnings("ignore")

DEF_PKL = os.path.join(project_root, "analysis", "survival_definitive.pkl")
COX_GRID = [{"penalizer": p, "l1_ratio": l}
            for p, l in product(COX_SPACE["penalizer"], COX_SPACE["l1_ratio"])]

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


def _score(builder, params, Xtr, Ttr, Etr, Xte, Tte, Ete, kmf):
    m = builder(params)
    if hasattr(m, "params") and isinstance(m.params, dict):
        m.params["nthread"] = 1                       # XGB-AFT: stay on one core
    m.fit(Xtr, Ttr, Etr)
    s = np.asarray(m.surv_score(Xte)).ravel()         # higher = longer survival
    return concordance_index(Tte, s, Ete), uno_cindex(-s, Tte, Ete, kmf)


def _cox_brier(params, Xtr, Ttr, Etr, Xte, Tte, Ete, kmf):
    cols = [f"f{i}" for i in range(Xtr.shape[1])]
    d = pd.DataFrame(Xtr, columns=cols)
    d["T"], d["E"] = Ttr, Etr
    m = CoxPHFitter(penalizer=params["penalizer"], l1_ratio=params["l1_ratio"]).fit(d, "T", "E")
    sf = m.predict_survival_function(pd.DataFrame(Xte, columns=cols), times=BRIER_GRID).T.values
    return integrated_brier(BRIER_GRID, sf, Tte, Ete, kmf)


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
    stable = list(freq[freq >= STAB_THR].index)
    if len(stable) < 3:
        stable = list(freq.head(5).index)
    s_idx = [cols.index(c) for c in stable]

    # 2. Cox penalty by inner coach-level CV on the training partition
    folds = full_cv_folds(df.loc[tr], X.loc[tr], y.loc[tr], durb.loc[tr], evt.loc[tr])
    inner = [(cv_cindex_at_k(cox_builder, p, folds, s_idx), i) for i, p in enumerate(COX_GRID)]
    cox_tuned = COX_GRID[max(inner)[1]]

    # 3. CV-argmax comparison set: rank on training, choose K by inner CV
    rank, _ = cox_importance(sp.X_train, Ttr, Etr)
    kcurve = {K: cv_cindex_at_k(cox_builder, cox_tuned, folds, list(rank[:K])) for K in K_GRID}
    k_best = max(kcurve, key=kcurve.get)
    k_idx = list(rank[:k_best])

    # 4. score the held-out partition
    kmf = _km_censoring(Ttr, Etr)
    Xs_tr, Xs_te = sp.X_train[:, s_idx], sp.X_test[:, s_idx]
    out = {"seed": seed, "stable": stable, "cox_tuned": cox_tuned, "k_best": k_best}
    for name, b, p in [("Cox", cox_builder, cox_tuned),
                       ("Cox (headline penalty)", cox_builder, _D["cox_fixed"]),
                       ("Weibull AFT", weibull_builder, AFT_P),
                       ("LogNormal AFT", lognormal_builder, AFT_P),
                       ("LogLogistic AFT", loglogistic_builder, AFT_P),
                       ("XGB-AFT", xgbaft_builder, _D["xgb_p"])]:
        out[name] = _score(b, p, Xs_tr, Ttr, Etr, Xs_te, Tte, Ete, kmf)
    out["CV-argmax Cox"] = _score(cox_builder, cox_tuned, sp.X_train[:, k_idx], Ttr, Etr,
                                  sp.X_test[:, k_idx], Tte, Ete, kmf)
    out["ibs_cox"], out["bs_cox"] = _cox_brier(cox_tuned, Xs_tr, Ttr, Etr, Xs_te, Tte, Ete, kmf)
    km = KaplanMeierFitter().fit(Ttr, Etr)
    s_km = np.array([float(km.predict(g)) for g in BRIER_GRID])
    out["ibs_km"], out["bs_km"] = integrated_brier(BRIER_GRID, np.tile(s_km, (len(Tte), 1)),
                                                   Tte, Ete, kmf)
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
    print("\nModel bake-off, NESTED (Nadeau-Bengio 95% CI)   vs   published (selection on all data)")
    print(f"  {'model':<24}{'Harrell C':>22}{'Uno C':>22}{'published Harrell':>22}")
    for name in ["Cox", "Cox (headline penalty)", "Weibull AFT", "LogNormal AFT",
                 "LogLogistic AFT", "XGB-AFT"]:
        h = nb_tci([r[name][0] for r in res], tf)
        u = nb_tci([r[name][1] for r in res], tf)
        pub = defn["bakeoff"].get(name.replace(" (headline penalty)", ""), {}).get("harrell")
        pub_s = f"{pub[0]:.3f}" if pub is not None and "headline" not in name else ""
        print(f"  {name:<24}{h[0]:>9.3f} [{h[1]:.3f},{h[2]:.3f}]{u[0]:>9.3f} [{u[1]:.3f},{u[2]:.3f}]{pub_s:>14}")

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
    print("  Cox Brier by horizon:", dict(zip(BRIER_GRID.astype(int), np.round(np.mean([r['bs_cox'] for r in res], 0), 3))))
    print("  KM  Brier by horizon:", dict(zip(BRIER_GRID.astype(int), np.round(np.mean([r['bs_km'] for r in res], 0), 3))))

    sel = pd.Series([nm(c) for r in res for c in r["stable"]]).value_counts()
    print(f"\nFeatures selected across {len(res)} training partitions:")
    for f, n in sel.head(10).items():
        print(f"  {f:<28}{n:>3}")
    pen = pd.Series([str(r["cox_tuned"]) for r in res]).value_counts()
    print("\nInner-CV Cox penalty chosen:")
    for p_, n in pen.items():
        print(f"  {p_:<44}{n:>3}")

    out = os.path.join(project_root, "analysis", "survival_nested_eval.pkl")
    with open(out, "wb") as f:
        pickle.dump({"splits": res, "n_boot": a.boots, "test_size": tf}, f)
    print(f"\nSaved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
