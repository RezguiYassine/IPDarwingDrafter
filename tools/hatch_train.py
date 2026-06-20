#!/usr/bin/env python3
"""De-risk gate for the learned hatch detector (Phase 0).

Reads the labelled pilot pack, builds (features, label) rows, and runs a
by-FIGURE cross-validated classifier. Reports ROC-AUC + precision at high recall
+ feature importances. This answers the key question BEFORE any heavy investment:

    AUC >= ~0.90  -> region re-classifier is viable; proceed to Phase 1.
    AUC low       -> region features don't separate; go to Phase 2 (pixel CNN).

    python -m tools.hatch_train --pack output/PatentData/hatch_pilot

Falls back to a logistic-regression baseline if gradient boosting is unavailable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
import hatch_features as hf  # noqa: E402


def load(pack):
    X, y, groups = [], [], []
    for fi, f in enumerate(sorted(glob.glob(os.path.join(pack, "*_regions.json")))):
        doc = json.load(open(f))
        for r in doc["regions"]:
            if r["label"] is None:
                continue
            X.append(hf.feature_row(r["features"]))
            y.append(1 if r["label"] else 0)
            groups.append(fi)
    return np.array(X, float), np.array(y, int), np.array(groups, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    X, y, groups = load(a.pack)
    if len(y) == 0:
        print("no labelled regions yet — label first with tools/hatch_label.py"); return
    n_fig = len(set(groups))
    print(f"labelled regions: {len(y)}  (hatch={int(y.sum())}, not={int((1-y).sum())}) "
          f"across {n_fig} figures")
    if y.sum() < 5 or (1 - y).sum() < 5 or n_fig < a.folds:
        print("too few labels / figures for a reliable CV estimate — label more.")
        return

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier as Clf
        mk = lambda: Clf(max_iter=300, learning_rate=0.06, max_depth=3,
                         l2_regularization=1.0)
        model_name = "HistGradientBoosting"
    except Exception:
        from sklearn.linear_model import LogisticRegression as Clf
        mk = lambda: Clf(max_iter=2000, class_weight="balanced")
        model_name = "LogisticRegression"
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, precision_recall_curve

    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=a.folds)
    for tr, te in gkf.split(X, y, groups):
        m = mk(); m.fit(X[tr], y[tr])
        oof[te] = (m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba")
                   else m.decision_function(X[te]))
    auc = roc_auc_score(y, oof)
    prec, rec, thr = precision_recall_curve(y, oof)
    # precision achievable at recall >= 0.8
    ok = rec[:-1] >= 0.80
    p_at_r80 = float(prec[:-1][ok].max()) if ok.any() else 0.0

    print(f"\nmodel: {model_name} | by-figure {a.folds}-fold CV")
    print(f"  ROC-AUC                : {auc:.3f}")
    print(f"  precision @ recall>=0.80: {p_at_r80:.3f}")
    verdict = ("VIABLE — proceed to Phase 1" if auc >= 0.90 else
               "MARGINAL — more labels or richer features" if auc >= 0.80 else
               "WEAK — region features don't separate; go to Phase 2 (pixel CNN)")
    print(f"  verdict                : {verdict}")

    # feature importances (permutation-free: full-fit gain for HGB via a quick proxy)
    try:
        from sklearn.inspection import permutation_importance
        m = mk(); m.fit(X, y)
        imp = permutation_importance(m, X, y, n_repeats=8, random_state=0)
        order = np.argsort(imp.importances_mean)[::-1]
        print("\n  top features (permutation importance):")
        for idx in order[:8]:
            print(f"    {hf.FEATURE_NAMES[idx]:20} {imp.importances_mean[idx]:+.4f}")
    except Exception as exc:
        print("  (importance unavailable:", exc, ")")


if __name__ == "__main__":
    main()
