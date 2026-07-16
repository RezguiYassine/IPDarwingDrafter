#!/usr/bin/env python3
"""Auto-save empty ground-truth masks for negative figures selected in hatch_select.py.

Reads hatch_selected_negatives.json, and for each figure that is not yet
in hatch_gt, creates a reviewed *_mask.json with zero polygons.  This
saves the negative label without any manual drawing step.

    python -m tools.hatch_finalize_negatives \\
        --negatives output/PatentData/hatch_selected_negatives.json \\
        --out output/PatentData/hatch_gt
"""
from __future__ import annotations

import argparse
import json
import os

import cv2

_DEFAULT_GT = "output/PatentData/hatch_gt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", default="output/PatentData/hatch_selected_negatives.json",
                     help="JSON list produced by hatch_select.py (negative figures)")
    ap.add_argument("--out", default=_DEFAULT_GT,
                     help="hatch_gt directory to write empty masks into")
    ap.add_argument("--redo", action="store_true",
                     help="overwrite even if a reviewed mask already exists")
    a = ap.parse_args()

    if not os.path.exists(a.negatives):
        print(f"not found: {a.negatives}"); return

    negatives = json.load(open(a.negatives))
    if not negatives:
        print("negatives list is empty"); return

    os.makedirs(a.out, exist_ok=True)

    saved = skipped = 0
    for fig in negatives:
        patent = fig["patent"]
        sketch = fig["sketch"]
        tif_path = fig.get("tif", "")

        stem     = f"{patent}__{sketch}__d0"
        out_path = os.path.join(a.out, f"{stem}_mask.json")

        if os.path.exists(out_path) and not a.redo:
            existing = json.load(open(out_path))
            if existing.get("reviewed"):
                skipped += 1
                continue

        # Load TIF to get image dimensions
        gray = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE) if tif_path else None
        if gray is None:
            # fallback: try ReorganisedData
            alt = f"data/PatentData/ReorganisedData/{patent}/{patent}_{sketch}.tif"
            gray = cv2.imread(alt, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  warning: TIF not found for {patent}__{sketch}, skipping")
            continue

        H, W = gray.shape
        json.dump({
            "patent":           patent,
            "sketch":           sketch,
            "subdrawing_index": 0,
            "crop_box":         [0, 0, int(W), int(H)],
            "reviewed":         True,
            "polygons":         [],
        }, open(out_path, "w"), indent=2)
        print(f"  saved {stem} (negative, {W}x{H}) -> {out_path}")
        saved += 1

    print(f"\n{saved} empty mask(s) saved, {skipped} already existed -> {a.out}")


if __name__ == "__main__":
    main()
