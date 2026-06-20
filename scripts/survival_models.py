#!/usr/bin/env python
"""
Survival model bake-off (Reviewer 1.1, extended): now that a plain Cox model
proved competitive with the ordinal classifier, we test whether a better survival
model earns the spotlight. We compare, on the SAME leakage-free protocol and the
SAME ordinal KPIs as the classifier:

  - Cox proportional hazards (lifelines)
  - Weibull / log-normal / log-logistic accelerated failure time (lifelines)
  - XGBoost AFT (objective survival:aft) -- our own model family made
    censoring-aware, so TreeSHAP still applies if we lean into survival

Two design choices address the failure mode we diagnosed (the S(2)/S(4) bucketing
collapses the narrow 3-4 year middle class, hurting Macro F1):

  1. CLASS DECISION by binning each coach's predicted *median* survival time
     (populates the middle naturally), while the probability vector used for
     AUROC still comes from the survival curve at the cutpoints S(2), S(4).
  2. Each model's hyperparameters are tuned by RANDOMIZED SEARCH over coach-level
     CV folds with a concordance-index objective (leakage-free: SVD imputation is
     fit per fold), then locked for the 50-seed evaluation -- the same
     tune-once / resample-evaluate protocol used for the classifier.

We also test the survival-only advantage: re-training with the censored recent
hires (tenure class -1) appended to the TRAINING set (they carry partial,
right-censored tenure the classifier must discard).

Usage:
    python scripts/survival_models.py
    python scripts/survival_models.py --quick   # smaller search, for iteration
"""

import os
import sys
import pickle
import argparse
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import xgboost as xgb
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index

from model import ordinal_metrics
from model.config import ORDINAL_CONFIG, MODEL_CONFIG
from model.cross_validation import CoachLevelStratifiedKFold
from model.pipeline import (
    FSTART, FEND, load_modeling_data, load_recent_hires, shap_feature_ranking,
    best_k, leakage_free_split, fit_model, ordinal_model, transform_select,
)
from scripts.survival_analysis import (
    global_max_season, build_survival_targets, surv_to_class_proba, tci, CUTPOINTS,
)
from scripts.data.engineer_career_features import reconstruct_tenure

warnings.filterwarnings("ignore")

# Iteration default: 50 resamples (camera-ready JQAS). With the Nadeau-Bengio
# correction the variance is (1/J + rho/(1-rho))*s^2 with rho=test_frac=0.2, so the
# rho/(1-rho)=0.25 term is irreducible and seeds only shrink the 1/J part (just 7.4%
# of total variance at J=50). J=25 (the prior default) widens CIs only ~6% vs J=50;
# J=50 is the locked final-tables setting.
N_SEEDS = 50
T2, T4 = CUTPOINTS
INF_TIME = 99.0  # cap for unreachable (S never crosses 0.5) median predictions


def bin_time(t):
    """Bin a (continuous) predicted tenure time into the ordinal class."""
    t = min(float(t), INF_TIME)
    return 0 if t <= 2 else (1 if t <= 4 else 2)


# --------------------------------------------------------------------------- #
# Model wrappers: uniform fit / pred_class / proba / surv_score interface.
# surv_score is oriented so HIGHER = LONGER survival (for the concordance index).
# --------------------------------------------------------------------------- #
class _LifelinesSurv:
    """Common logic for lifelines Cox / AFT fitters."""
    def __init__(self, fitter):
        self.f = fitter

    def fit(self, X, T, E):
        self.cols = [f"f{i}" for i in range(X.shape[1])]
        d = pd.DataFrame(X, columns=self.cols)
        d["T"], d["E"] = T, E
        self.f.fit(d, duration_col="T", event_col="E")
        return self

    def _Xdf(self, X):
        return pd.DataFrame(X, columns=self.cols)

    def proba(self, X):
        return surv_to_class_proba(self.f, self._Xdf(X))

    def pred_class(self, X):
        med = np.asarray(self.f.predict_median(self._Xdf(X))).ravel()
        med = np.where(np.isinf(med), INF_TIME, med)
        return np.array([bin_time(t) for t in med])

    def surv_score(self, X):
        med = np.asarray(self.f.predict_median(self._Xdf(X))).ravel()
        return np.where(np.isinf(med), INF_TIME, med)


class CoxModel(_LifelinesSurv):
    def __init__(self, penalizer=0.1, l1_ratio=0.0):
        super().__init__(CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio))

    def surv_score(self, X):  # partial hazard is better-behaved than Cox median
        return -np.asarray(self.f.predict_partial_hazard(self._Xdf(X))).ravel()


class AFTModel(_LifelinesSurv):
    def __init__(self, fitter_cls, penalizer=0.01):
        super().__init__(fitter_cls(penalizer=penalizer))


class XGBAFTModel:
    """XGBoost survival:aft. Predicts a survival time; we reconstruct S(t) from
    the assumed log-location/scale family for the probability vector."""
    def __init__(self, dist="normal", scale=1.0, num_boost_round=60, **params):
        self.dist, self.scale, self.nbr = dist, scale, num_boost_round
        self.params = {
            "objective": "survival:aft", "eval_metric": "aft-nloglik",
            "aft_loss_distribution": dist, "aft_loss_distribution_scale": scale,
            "tree_method": "hist", "verbosity": 0, "nthread": -1, **params,
        }

    def fit(self, X, T, E):
        d = xgb.DMatrix(np.asarray(X))
        lower = np.asarray(T, float)
        upper = np.where(np.asarray(E) == 1, lower, np.inf)
        d.set_float_info("label_lower_bound", lower)
        d.set_float_info("label_upper_bound", upper)
        self.bst = xgb.train(self.params, d, num_boost_round=self.nbr)
        return self

    def _time(self, X):
        return self.bst.predict(xgb.DMatrix(np.asarray(X)))

    def _S(self, t, eta):
        z = (np.log(t) - eta) / self.scale
        if self.dist == "normal":
            return 1.0 - stats.norm.cdf(z)
        if self.dist == "logistic":
            return 1.0 - expit(z)
        return np.exp(-np.exp(z))  # extreme (Gumbel min)

    def proba(self, X):
        eta = np.log(np.clip(self._time(X), 1e-6, None))
        S2, S4 = self._S(T2, eta), self._S(T4, eta)
        P = np.vstack([1 - S2, S2 - S4, S4]).T
        P = np.clip(P, 0, None)
        return P / P.sum(axis=1, keepdims=True)

    def pred_class(self, X):
        return np.array([bin_time(t) for t in self._time(X)])

    def surv_score(self, X):
        return self._time(X)


# --------------------------------------------------------------------------- #
# Tuning: randomized search over coach-level CV folds, concordance objective.
# --------------------------------------------------------------------------- #
def precompute_cv_folds(df, X, y, dur, evt, idx, n_splits=5, seed=42):
    """Leakage-free CV folds: impute (fit on fold-train), select top-K, once."""
    cv = CoachLevelStratifiedKFold(n_splits=n_splits, random_state=seed)
    groups = df["Coach Name"].values
    Xv, yv = X.values, y.values
    Tv, Ev = dur.values.astype(float), evt.values.astype(int)
    from scripts.data.matrix_factorization_imputation import SVDImputer
    folds = []
    for tr, va in cv.split(Xv, yv, groups):
        imp = SVDImputer().fit(Xv[tr])
        Xtr = imp.transform(Xv[tr])[:, idx]
        Xva = imp.transform(Xv[va])[:, idx]
        folds.append((Xtr, Tv[tr], Ev[tr], Xva, Tv[va], Ev[va]))
    return folds


def cv_cindex(builder, params, folds):
    scores = []
    for Xtr, Ttr, Etr, Xva, Tva, Eva in folds:
        try:
            m = builder(params).fit(Xtr, Ttr, Etr)
            scores.append(concordance_index(Tva, m.surv_score(Xva), Eva))
        except Exception:
            return -1.0
    return float(np.mean(scores))


def random_search(builder, space, folds, n_iter, rng):
    """Sample n_iter param dicts from `space` (dict of lists); return best by CV C-index."""
    keys = list(space)
    best, best_s = None, -np.inf
    seen = set()
    tries = 0
    while len(seen) < n_iter and tries < n_iter * 5:
        tries += 1
        cand = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        sig = tuple(cand[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        s = cv_cindex(builder, cand, folds)
        if s > best_s:
            best, best_s = cand, s
    return best, best_s


# --------------------------------------------------------------------------- #
# Builders + search spaces
# --------------------------------------------------------------------------- #
def cox_builder(p):
    return CoxModel(penalizer=p["penalizer"], l1_ratio=p["l1_ratio"])

def weibull_builder(p):
    return AFTModel(WeibullAFTFitter, penalizer=p["penalizer"])

def lognormal_builder(p):
    return AFTModel(LogNormalAFTFitter, penalizer=p["penalizer"])

def loglogistic_builder(p):
    return AFTModel(LogLogisticAFTFitter, penalizer=p["penalizer"])

def xgbaft_builder(p):
    return XGBAFTModel(
        dist="normal", scale=p["scale"], num_boost_round=p["num_boost_round"],
        eta=p["eta"], max_depth=p["max_depth"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
        reg_lambda=p["reg_lambda"])

COX_SPACE = {"penalizer": [0.01, 0.05, 0.1, 0.5, 1.0], "l1_ratio": [0.0, 0.5, 1.0]}
AFT_SPACE = {"penalizer": [0.0, 0.001, 0.01, 0.1, 0.5]}
XGBAFT_SPACE = {
    "num_boost_round": [30, 50, 80, 120], "eta": [0.03, 0.05, 0.1, 0.15],
    "max_depth": [2, 3, 4], "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 3, 5],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0], "scale": [0.5, 0.8, 1.0, 1.2],
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
METRICS = ["qwk", "mae", "macro_f1", "adjacent_accuracy", "auroc"]


def eval_model(builder, params, df, X, y, dur, evt, idx,
               recent=None, rec_dur=None):
    """50-seed leakage-free evaluation; returns dict of metric lists (+c_index).
    If `recent` is given, its censored rows are appended to each split's TRAIN set
    (coach-leakage-guarded) using that split's fitted imputer."""
    out = {m: [] for m in METRICS}
    out["c_index"] = []
    for seed in range(N_SEEDS):
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)
        Xtr, ytr = split.X_train, split.y_train
        Ttr = dur.loc[split.train_index].values.astype(float)
        Etr = evt.loc[split.train_index].values.astype(int)
        Tte = dur.loc[split.test_index].values.astype(float)
        Ete = evt.loc[split.test_index].values.astype(int)

        if recent is not None:
            test_coaches = set(df.loc[split.test_index, "Coach Name"])
            keep = ~recent["Coach Name"].isin(test_coaches)
            r = recent[keep]
            if len(r):
                Xr = transform_select(r.iloc[:, FSTART:FEND], split.imputer, idx)
                Xtr = np.vstack([Xtr, Xr])
                Ttr = np.concatenate([Ttr, rec_dur.loc[r.index].values.astype(float)])
                Etr = np.concatenate([Etr, np.zeros(len(r), int)])  # all censored

        m = builder(params).fit(Xtr, Ttr, Etr)
        proba = m.proba(split.X_test)
        pred = m.pred_class(split.X_test)
        r = ordinal_metrics(split.y_test, pred, proba, ORDINAL_CONFIG["class_names"])
        for k in METRICS:
            out[k].append(r[k])
        out["c_index"].append(concordance_index(Tte, m.surv_score(split.X_test), Ete))
    return out


def classifier_reference(df, X, y, idx):
    out = {m: [] for m in METRICS}
    for seed in range(N_SEEDS):
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)
        om = fit_model(split, ordinal_model, seed)
        Xte = pd.DataFrame(split.X_test)
        r = ordinal_metrics(split.y_test, om.predict(Xte), om.predict_proba(Xte),
                            ORDINAL_CONFIG["class_names"])
        for k in METRICS:
            out[k].append(r[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller search for iteration")
    args = ap.parse_args()
    n_iter_xgb = 12 if args.quick else 40
    n_iter_aft = 3 if args.quick else 5

    df, X, y = load_modeling_data()
    idx = shap_feature_ranking()[:best_k()]
    bk = len(idx)
    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    df, X, y = df.loc[keep], X.loc[keep], y.loc[keep]
    dur, evt = dur.loc[keep], evt.loc[keep]
    print(f"Survival bake-off: {len(df)} stints, top-{bk} features, {N_SEEDS} splits, "
          f"censored={int((evt==0).sum())}")

    folds = precompute_cv_folds(df, X, y, dur, evt, idx)
    rng = np.random.default_rng(0)

    print("\nTuning (randomized search, coach-level CV, C-index objective):")
    tuned = {}
    for name, builder, space, n_it in [
        ("Cox", cox_builder, COX_SPACE, 15),
        ("Weibull AFT", weibull_builder, AFT_SPACE, n_iter_aft),
        ("LogNormal AFT", lognormal_builder, AFT_SPACE, n_iter_aft),
        ("LogLogistic AFT", loglogistic_builder, AFT_SPACE, n_iter_aft),
        ("XGB-AFT", xgbaft_builder, XGBAFT_SPACE, n_iter_xgb),
    ]:
        p, s = random_search(builder, space, folds, n_it, rng)
        tuned[name] = {"builder": builder, "params": p, "cv_cindex": s}
        print(f"  {name:<16} CV C-index={s:.3f}  params={p}")

    print("\nEvaluating tuned models on 50 leakage-free splits...")
    results = {}
    for name, t in tuned.items():
        results[name] = eval_model(t["builder"], t["params"], df, X, y, dur, evt, idx)
        print(f"  done: {name}")

    # survival-only advantage: XGB-AFT trained WITH censored recent hires
    recent = load_recent_hires()
    rec_dur = pd.Series(
        {i: reconstruct_tenure(c, int(yr))
         for i, c, yr in zip(recent.index, recent["Coach Name"], recent["Year"])})
    recent = recent.loc[rec_dur.dropna().index]
    rec_dur = rec_dur.dropna()
    print(f"\nCensored recent hires available for augmentation: {len(recent)}")
    best_surv = max(tuned, key=lambda k: tuned[k]["cv_cindex"])
    aug_name = f"XGB-AFT +censored"
    results[aug_name] = eval_model(
        tuned["XGB-AFT"]["builder"], tuned["XGB-AFT"]["params"],
        df, X, y, dur, evt, idx, recent=recent, rec_dur=rec_dur)
    print(f"  done: {aug_name}")

    print("\nClassifier reference on identical splits...")
    clf = classifier_reference(df, X, y, idx)
    results["Ordinal classifier"] = {**clf, "c_index": [np.nan]}

    # ---- report ----
    print("\n" + "=" * 100)
    print("MODEL COMPARISON (mean over 50 splits; t-dist 95% CI on QWK)")
    print("=" * 100)
    print(f"{'model':<22}{'QWK':>20}{'MacroF1':>9}{'AdjAcc':>8}{'AUROC':>8}{'MAE':>8}{'C-idx':>8}")
    print("-" * 100)
    summary = {}
    order = ["Ordinal classifier", "Cox", "Weibull AFT", "LogNormal AFT",
             "LogLogistic AFT", "XGB-AFT", aug_name]
    for name in order:
        r = results[name]
        qm, ql, qh = tci(r["qwk"])
        ci = tci(r["c_index"])[0] if not np.all(np.isnan(r["c_index"])) else np.nan
        summary[name] = {k: tci(r[k]) for k in METRICS}
        summary[name]["c_index"] = ci
        cis = f"{ci:.3f}" if not np.isnan(ci) else "   -"
        print(f"{name:<22}{qm:>9.3f} [{ql:.3f},{qh:.3f}]{tci(r['macro_f1'])[0]:>9.3f}"
              f"{tci(r['adjacent_accuracy'])[0]:>8.3f}{tci(r['auroc'])[0]:>8.3f}"
              f"{tci(r['mae'])[0]:>8.3f}{cis:>8}")
    print("-" * 100)

    out = os.path.join(project_root, "analysis", "survival_models.pkl")
    with open(out, "wb") as f:
        pickle.dump({"summary": summary, "raw": results,
                     "tuned": {k: {"params": v["params"], "cv_cindex": v["cv_cindex"]}
                               for k, v in tuned.items()},
                     "best_k": bk, "n_seeds": N_SEEDS, "boundary": boundary,
                     "n_censored_train": int((evt == 0).sum()),
                     "n_recent_aug": int(len(recent))}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
