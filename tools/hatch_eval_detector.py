#!/usr/bin/env python3
"""Score the hatch detector's candidate regions (output/PatentData/hatch_pilot)
against the hand-drawn ground truth (output/PatentData/hatch_gt), and
auto-derive region labels for tools/hatch_train.py's de-risk AUC gate.

Why: the original Phase 0 plan (docs/hatch_detector_scope.md) had a human
click yes/no on each of the detector's candidate regions. That pass only
covers *precision* (is a proposed region real hatching?) and was never
finished by hand (9/213). The ground-truth masks (drawn independently of the
detector, via tools/hatch_mask_label.py) cover the true hatch areas directly,
so matching candidates against them gives both real precision AND recall
(including hatching the detector never proposed at all) -- for free, no more
manual clicking.

For each figure:
    1. Rasterize the candidate regions and the combined ground-truth polygons
       onto the TIF's actual pixel grid.
    2. label(region) = True iff >= --overlap-threshold of the region's own
       area is covered by ground-truth ink (this is what hatch_train.py's
       "label" field means: accept/reject this proposal).
    3. Aggregate pixel-level precision/recall across the whole pack, plus a
       recall *ceiling*: the max recall any Phase-1 classifier could ever
       reach, since it can only accept/reject existing proposals, never
       invent new ones.

Writes auto-derived labels back into hatch_pilot/*_regions.json (label +
label_overlap_frac fields, doc-level label_source: "gt_overlap_v1") so
tools/hatch_train.py works unchanged afterward.

    python -m tools.hatch_eval_detector --pack-source output/PatentData/hatch_pilot \
        --gt output/PatentData/hatch_gt --tif-root data/PatentData/ReorganisedData
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

_DEFAULT_TIF_ROOT = "data/PatentData/ReorganisedData"


def _fill(mask, polygons):
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)


def _load_gt_by_figure(gt_dir):
    """{(patent, sketch): [polygon, ...]} in full-TIF coords, all *_mask.json
    for that figure combined (including any legacy pre-split file -- its
    polygons are still in full-TIF coords, just from before sub-drawing
    splitting existed)."""
    by_fig = defaultdict(list)
    legacy_used = []
    for path in sorted(glob.glob(os.path.join(gt_dir, "*_mask.json"))):
        d = json.load(open(path))
        if not d.get("reviewed"):
            continue
        key = (d["patent"], d["sketch"])
        by_fig[key].extend(d.get("polygons", []))
        if "crop_box" not in d and d.get("polygons"):
            legacy_used.append(os.path.basename(path))
    return by_fig, legacy_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-source", default="output/PatentData/hatch_pilot")
    ap.add_argument("--gt", default="output/PatentData/hatch_gt")
    ap.add_argument("--tif-root", default=_DEFAULT_TIF_ROOT)
    ap.add_argument("--overlap-threshold", type=float, default=0.5,
                     help="fraction of a candidate region's own area that must be "
                          "covered by ground truth for it to be labelled hatch=True")
    ap.add_argument("--dry-run", action="store_true",
                     help="report only, don't write labels back to hatch_pilot")
    a = ap.parse_args()

    gt_by_fig, legacy_used = _load_gt_by_figure(a.gt)
    if legacy_used:
        print(f"note: including {len(legacy_used)} legacy (pre-split) ground-truth "
              f"file(s) with real polygons: {', '.join(legacy_used)}")

    region_files = sorted(glob.glob(os.path.join(a.pack_source, "*_regions.json")))
    if not region_files:
        print("no *_regions.json in", a.pack_source); return

    n_pos = n_neg = n_no_gt_figure = 0
    overlaps_pos, overlaps_neg = [], []
    ambiguous = []  # (frac near threshold) -- boundary-quality signal
    tot_inter = tot_cand_area = tot_gt_area = tot_gt_covered = 0

    for path in region_files:
        doc = json.load(open(path))
        patent, sketch = doc["patent"], doc["sketch"]
        gt_polys = gt_by_fig.get((patent, sketch))
        tif = os.path.join(a.tif_root, patent, f"{patent}_{sketch}.tif")
        gray = cv2.imread(tif, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip {patent}__{sketch}: tif not found at {tif}")
            continue
        H, W = gray.shape

        gt_mask = np.zeros((H, W), np.uint8)
        if gt_polys:
            _fill(gt_mask, gt_polys)
        else:
            n_no_gt_figure += 1  # figure has candidates but zero drawn ground truth

        any_cand_mask = np.zeros((H, W), np.uint8)
        region_masks = []
        for r in doc["regions"]:
            m = np.zeros((H, W), np.uint8)
            _fill(m, [r["boundary"]])
            region_masks.append(m)
            any_cand_mask |= m

        for r, m in zip(doc["regions"], region_masks):
            cand_area = int(m.sum())
            inter = int((m & gt_mask).sum())
            frac = (inter / cand_area) if cand_area > 0 else 0.0
            label = frac >= a.overlap_threshold
            r["label"] = bool(label)
            r["label_overlap_frac"] = round(frac, 4)
            if label:
                n_pos += 1; overlaps_pos.append(frac)
            else:
                n_neg += 1; overlaps_neg.append(frac)
            if 0.15 <= frac <= 0.85:
                ambiguous.append((patent, sketch, r["id"], frac))
            tot_inter += inter
            tot_cand_area += cand_area

        gt_area = int(gt_mask.sum())
        gt_covered = int((gt_mask & any_cand_mask).sum())
        tot_gt_area += gt_area
        tot_gt_covered += gt_covered

        doc["label_source"] = "gt_overlap_v1"
        doc["label_overlap_threshold"] = a.overlap_threshold
        if not a.dry_run:
            json.dump(doc, open(path, "w"), indent=2)

    n_total = n_pos + n_neg
    precision_proxy = (tot_inter / tot_cand_area) if tot_cand_area else 0.0
    recall_ceiling = (tot_gt_covered / tot_gt_area) if tot_gt_area else 0.0

    print(f"\n{'DRY RUN — ' if a.dry_run else ''}{n_total} candidate regions scored "
          f"across {len(region_files)} figures")
    print(f"  label=True (hatch):     {n_pos}  (mean overlap {np.mean(overlaps_pos):.2f})" if overlaps_pos
          else "  label=True (hatch):     0")
    print(f"  label=False (not hatch): {n_neg}  (mean overlap {np.mean(overlaps_neg):.2f})" if overlaps_neg
          else "  label=False (not hatch): 0")
    print(f"  figures with candidates but 0 drawn ground truth: {n_no_gt_figure} "
          f"(all their regions auto-labelled False)")
    print(f"  ambiguous regions (overlap 0.15-0.85, boundary-quality signal): {len(ambiguous)}")
    print(f"\n  pixel-level precision (candidate-area that is real hatch): {precision_proxy:.1%}")
    print(f"  pixel-level RECALL CEILING (real hatch area covered by *any* "
          f"candidate): {recall_ceiling:.1%}")
    print("    -> this is the max recall any Phase-1 classifier can reach: it can "
          "only accept/reject existing proposals, never invent new ones.")
    if ambiguous:
        print("\n  sample ambiguous regions (figure, region id, overlap frac):")
        for patent, sketch, rid, frac in ambiguous[:8]:
            print(f"    {patent}__{sketch} region {rid}: {frac:.2f}")

    if not a.dry_run:
        print(f"\nlabels written -> {a.pack_source}")
        print(f"Run the de-risk AUC gate with:  python -m tools.hatch_train --pack {a.pack_source}")


if __name__ == "__main__":
    main()
