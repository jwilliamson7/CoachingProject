#!/usr/bin/env python
"""
Survival analysis (Reviewer 1.1): coaching tenure is fundamentally a time-to-event
outcome with right-censoring (coaches still active at the data boundary have not
yet "failed"). Classification mislabels those cases and regression must drop or
distort them; a survival model uses them correctly. We therefore add a Cox
proportional-hazards model as a third modeling lens and -- crucially -- score it
on the SAME ordinal KPIs as the classifier so the comparison is apples-to-apples.

How a survival model is made comparable to the ordinal classifier:
  A Cox model predicts each coach's survival curve S(t) = P(tenure > t seasons).
  We read it at the class cutpoints t=2 and t=4 to get class probabilities
      P(class 0) = P(T <= 2) = 1 - S(2)
      P(class 1) = P(2 < T <= 4) = S(2) - S(4)
      P(class 2) = P(T > 4)      = S(4)
  argmax gives the predicted class, so QWK / MAE / Macro F1 / adjacent accuracy /
  AUROC are computed exactly as for the classifier, on the same 50 leakage-free
  coach-level splits and the same top-K SHAP features. We also report the native
  concordance index (C-index), survival's own discrimination KPI.

Protocol matches the headline: modern-era population, top-K SHAP features, SVD
imputation fit on each split's training partition only, 50 coach-level splits.
The ordinal classifier is fit on the identical splits for a paired comparison.

Usage:
    python scripts/survival_analysis.py
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from model import ordinal_metrics
from model.config import ORDINAL_CONFIG, MODEL_CONFIG
from model.pipeline import (
    load_modeling_data, shap_feature_ranking, best_k, leakage_free_split,
    fit_model, ordinal_model,
)
from scripts.data.engineer_career_features import (
    reconstruct_tenure, load_history, COACHES_DIR, is_nfl, classify_role,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_SEEDS = 50
PENALIZER = 0.1          # small L2 ridge for Cox stability with correlated features
CUTPOINTS = (2.0, 4.0)   # class boundaries: <=2 / 3-4 / 5+


def global_max_season():
    """Last HC season present anywhere in the coach histories (the censoring
    boundary): a stint whose final season equals this is right-censored."""
    mx = 0
    for d in COACHES_DIR.iterdir():
        if not d.is_dir():
            continue
        h = load_history(d.name)
        if h is None:
            continue
        for _, r in h.iterrows():
            if is_nfl(r.get("Level")) and classify_role(r.get("Role", "")) == "HC":
                mx = max(mx, int(r["Year"]))
    return mx


EVENT_LABELS_FILE = os.path.join(project_root, "analysis", "event_labels_final.csv")
_EVENT_LABELS = None


def _firing_event_labels():
    """(coach, last_year) -> firing-aware event (1=involuntary non-retention,
    0=voluntary/active). Built by build_event_labels.py + merge_event_labels.py;
    keyed by last_year so it is stable under interim->permanent re-anchoring."""
    global _EVENT_LABELS
    if _EVENT_LABELS is None:
        lab = pd.read_csv(EVENT_LABELS_FILE)
        _EVENT_LABELS = {(r.coach, int(r.last_year)): int(r.event)
                         for r in lab.itertuples()}
    return _EVENT_LABELS


def build_survival_targets(df, boundary):
    """duration (seasons) and FIRING-aware event indicator.

    event = 1 marks involuntary non-retention (firing / forced-out / pushed
    "retirement"); event = 0 marks a voluntary-while-viable departure (moved
    team, retired on top, health) or a coach still active at the boundary --
    these are right-censored. This replaces the earlier lumped definition
    (any tenure end = event) per the JQAS pivot. Labels come from
    analysis/event_labels_final.csv; the boundary rule is a safety fallback for
    any stint missing a label.
    """
    labels = _firing_event_labels()
    dur, evt = {}, {}
    for c, yr in zip(df["Coach Name"], df["Year"]):
        t = reconstruct_tenure(c, int(yr))
        if t is None:
            continue
        last = int(yr) + t - 1
        idx = df.index[(df["Coach Name"] == c) & (df["Year"] == yr)][0]
        dur[idx] = t
        evt[idx] = labels.get((c, last), 0 if last >= boundary else 1)
    return pd.Series(dur), pd.Series(evt)


def surv_to_class_proba(cph, X_df):
    """Map a fitted Cox model's survival curve at the cutpoints to 3-class probs."""
    t2, t4 = CUTPOINTS
    sf = cph.predict_survival_function(X_df, times=[t2, t4])  # index=[t2,t4]
    S2 = sf.iloc[0].values
    S4 = sf.iloc[1].values
    P0 = 1.0 - S2
    P1 = S2 - S4
    P2 = S4
    proba = np.vstack([P0, P1, P2]).T
    proba = np.clip(proba, 0.0, None)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


def tci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    n = len(a); m = a.mean(); se = a.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return m, m - h, m + h


def kaplan_meier_figure(dur, evt, df, boundary):
    """Overall KM curve + KM stratified by hire era (non-circular) with log-rank."""
    kmf = KaplanMeierFitter()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    kmf.fit(dur.values, evt.values, label="All HC stints")
    kmf.plot_survival_function(ax=ax1, ci_show=True)
    med = kmf.median_survival_time_
    for t in CUTPOINTS:
        ax1.axvline(t, ls="--", color="C3", alpha=0.5)
    ax1.set_title(f"NFL head-coach tenure survival (median = {med:.0f} seasons)")
    ax1.set_xlabel("Seasons"); ax1.set_ylabel("P(still head coach)")
    ax1.set_xlim(0, 15)

    yr = df.loc[dur.index, "Year"].values
    early = yr < 2000
    for mask, lab in [(early, "Hired <2000"), (~early, "Hired >=2000")]:
        kmf.fit(dur.values[mask], evt.values[mask], label=lab)
        kmf.plot_survival_function(ax=ax2, ci_show=False)
    lr = logrank_test(dur.values[early], dur.values[~early],
                      evt.values[early], evt.values[~early])
    ax2.set_title(f"Tenure by hiring era (log-rank p = {lr.p_value:.3f})")
    ax2.set_xlabel("Seasons"); ax2.set_ylabel("P(still head coach)")
    ax2.set_xlim(0, 15)
    fig.tight_layout()

    figdir = os.path.join(project_root, "ijcss", "figures")
    os.makedirs(figdir, exist_ok=True)
    figpath = os.path.join(figdir, "survival_km.png")
    fig.savefig(figpath, dpi=150)
    print(f"Saved figure {figpath}")
    return float(med), float(lr.p_value)


def main():
    df, X, y = load_modeling_data()
    idx = shap_feature_ranking()[:best_k()]
    bk = len(idx)
    feat_cols = [f"f{i}" for i in range(bk)]

    boundary = global_max_season()
    dur, evt = build_survival_targets(df, boundary)
    keep = dur.index
    print(f"Survival analysis: {len(keep)} stints, top-{bk} SHAP features, "
          f"{N_SEEDS} coach-level splits")
    print(f"Censoring boundary = {boundary}; censored (still active) = "
          f"{int((evt == 0).sum())} ({(evt == 0).mean() * 100:.1f}%)\n")

    med, lr_p = kaplan_meier_figure(dur, evt, df, boundary)
    print(f"Kaplan-Meier median tenure = {med:.0f} seasons; "
          f"era log-rank p = {lr_p:.3f}\n")

    # Per-seed paired comparison: Cox (bucketed) vs ordinal classifier.
    metrics = ["qwk", "mae", "macro_f1", "adjacent_accuracy", "auroc"]
    cox = {m: [] for m in metrics}
    cox["c_index"] = []
    clf = {m: [] for m in metrics}

    for seed in range(N_SEEDS):
        split = leakage_free_split(df, X, y, seed, feature_indices=idx)

        # align survival targets to the split's train/test rows
        Ttr = dur.loc[split.train_index].values.astype(float)
        Etr = evt.loc[split.train_index].values.astype(int)
        Tte = dur.loc[split.test_index].values.astype(float)
        Ete = evt.loc[split.test_index].values.astype(int)

        tr_df = pd.DataFrame(split.X_train, columns=feat_cols)
        tr_df["T"], tr_df["E"] = Ttr, Etr
        te_df = pd.DataFrame(split.X_test, columns=feat_cols)

        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(tr_df, duration_col="T", event_col="E")

        proba = surv_to_class_proba(cph, te_df)
        pred = proba.argmax(axis=1)
        m = ordinal_metrics(split.y_test, pred, proba, ORDINAL_CONFIG["class_names"])
        for k in metrics:
            cox[k].append(m[k])
        # native C-index: higher partial hazard = higher risk = shorter tenure
        risk = cph.predict_partial_hazard(te_df).values
        cox["c_index"].append(concordance_index(Tte, -risk, Ete))

        # ordinal classifier on the identical split (paired reference)
        om = fit_model(split, ordinal_model, seed)
        Xte = pd.DataFrame(split.X_test)
        mc = ordinal_metrics(split.y_test, om.predict(Xte), om.predict_proba(Xte),
                             ORDINAL_CONFIG["class_names"])
        for k in metrics:
            clf[k].append(mc[k])

        if (seed + 1) % 10 == 0:
            print(f"  ...{seed + 1}/{N_SEEDS} seeds")

    print("\n" + "=" * 78)
    print("COX SURVIVAL (bucketed to classes) vs ORDINAL CLASSIFIER -- same splits")
    print("=" * 78)
    print(f"{'metric':<18}{'Cox survival':>22}{'ordinal clf':>22}{'mean diff':>12}")
    print("-" * 78)
    rows = {}
    for k in metrics:
        cm, cl, ch = tci(cox[k])
        om_, ol, oh = tci(clf[k])
        diff = np.mean(np.array(cox[k]) - np.array(clf[k]))
        rows[k] = {"cox": (cm, cl, ch), "clf": (om_, ol, oh), "diff": float(diff)}
        print(f"{k:<18}{cm:>9.3f} [{cl:.3f},{ch:.3f}]"
              f"{om_:>9.3f} [{ol:.3f},{oh:.3f}]{diff:>+12.4f}")
    ci_m, ci_l, ci_h = tci(cox["c_index"])
    rows["c_index"] = {"cox": (ci_m, ci_l, ci_h)}
    print("-" * 78)
    print(f"{'c_index (Cox)':<18}{ci_m:>9.3f} [{ci_l:.3f},{ci_h:.3f}]"
          f"{'(survival-native; no classifier analog)':>44}")
    print("\nNote: C-index 0.5 = no discrimination, 1.0 = perfect risk ranking.")

    out = os.path.join(project_root, "analysis", "survival_results.pkl")
    with open(out, "wb") as f:
        pickle.dump({"rows": rows, "cox_raw": cox, "clf_raw": clf,
                     "best_k": bk, "n_seeds": N_SEEDS, "boundary": boundary,
                     "censored": int((evt == 0).sum()), "median_tenure": med,
                     "era_logrank_p": lr_p}, f)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
