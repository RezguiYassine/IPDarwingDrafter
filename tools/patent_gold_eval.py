"""
Gold patent keypoint eval — scorer  (accuracy roadmap — step #1)
================================================================

Scores keypoint detectors against the hand-corrected gold labels produced by
`tools.patent_gold_sample`. For each reviewed `<id>.json` it runs each detector
on `<id>_skeleton.png` and computes per-class precision / recall / F1 by greedy
nearest matching within `--radius` px:

  CN        crossing-number (endpoints/junctions only — no corners)
  CNN-D2C   models/puhachov_d2c.pth
  CNN-FT    the patent-fine-tuned weight (if given via --weights-ft)

This is the direct measure of whether fine-tuning improved patent keypoint
detection — the thing the intrinsic output metrics can't see.

Usage:
    python -m tools.patent_gold_eval --gold output/PatentData/gold_eval \
        --weights-ft models/puhachov_d2c_patentft.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage2_strokeextraction"))
import stage2_stroke_extract as s2  # noqa: E402

CLASSES = ("endpoint", "junction", "corner")


def _match(gold, pred, radius):
    """Greedy nearest matching within radius. Returns (tp, fp, fn)."""
    if not gold and not pred:
        return 0, 0, 0
    if not gold:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gold)
    tree = cKDTree(np.array(gold, float))
    used = set()
    tp = 0
    for p in pred:
        d, i = tree.query(p)
        if d <= radius and i not in used:
            used.add(i); tp += 1
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def _classical(sk):
    return s2._classical_keypoints(sk)


def _cnn(model, sk, conf):
    return model.detect(sk, conf, 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(PROJECT_ROOT / "output/PatentData/gold_eval"))
    ap.add_argument("--weights-d2c", default=str(PROJECT_ROOT / "models/puhachov_d2c.pth"))
    ap.add_argument("--weights-ft", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    gold_dir = Path(args.gold)
    jsons = sorted(gold_dir.glob("*.json"))
    labeled = []
    for j in jsons:
        d = json.loads(j.read_text())
        if d.get("reviewed"):
            labeled.append((j.stem, d))
    if not labeled:
        print(f"No reviewed gold labels in {gold_dir} "
              f"(set \"reviewed\": true after correcting). "
              f"{len(jsons)} unreviewed file(s) present.")
        return 1
    print(f"Scoring on {len(labeled)} reviewed sketches (radius={args.radius}px)")

    dets = {"CN": None}
    dets["CNN-D2C"] = s2.PuhachovKeypointDetector(args.weights_d2c, device=args.device)
    if args.weights_ft:
        dets["CNN-FT"] = s2.PuhachovKeypointDetector(args.weights_ft, device=args.device)

    # accumulate tp/fp/fn per detector per class; only score classes the gold
    # actually labels (gold may be corner-only).
    acc = {name: {c: [0, 0, 0] for c in CLASSES} for name in dets}
    scored = set()
    for stem, d in labeled:
        sk = cv2.imread(str(gold_dir / f"{stem}_skeleton.png"), 0)
        if sk is None:
            continue
        lc = d.get("labeled_classes", list(CLASSES))
        scored.update(lc)
        gold_by = defaultdict(list)
        for k in d["keypoints"]:
            gold_by[k["type"]].append((k["x"], k["y"]))
        for name, model in dets.items():
            kps = _classical(sk) if name == "CN" else _cnn(model, sk, args.conf)
            pred_by = defaultdict(list)
            for k in kps:
                pred_by[k["type"]].append((k["x"], k["y"]))
            for c in lc:
                tp, fp, fn = _match(gold_by.get(c, []), pred_by.get(c, []), args.radius)
                acc[name][c][0] += tp; acc[name][c][1] += fp; acc[name][c][2] += fn
    score_classes = [c for c in CLASSES if c in scored]

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f = 2 * p * r / (p + r) if p and r and not (np.isnan(p) or np.isnan(r)) else float("nan")
        return p, r, f

    print(f"\nScored classes: {score_classes}")
    print(f"\n{'detector':10s}{'class':10s}{'P':>7s}{'R':>7s}{'F1':>7s}   (tp/fp/fn)")
    print("-" * 56)
    for name in dets:
        for c in score_classes:
            tp, fp, fn = acc[name][c]
            p, r, f = prf(tp, fp, fn)
            print(f"{name:10s}{c:10s}{p:7.3f}{r:7.3f}{f:7.3f}   ({tp}/{fp}/{fn})")
    print("\n(CN emits no corners — for a corner-only gold its corner row is all "
          "FN. The comparison that matters is CNN-D2C vs CNN-FT.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
