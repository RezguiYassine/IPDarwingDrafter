#!/usr/bin/env python3
"""Interactive labeller for the hatch-region pilot pack.

For each candidate region it shows a ZOOMED crop (region boundary highlighted
in red) in a reused matplotlib window, one region at a time. You judge it
purely visually:

    HATCH = regular parallel / cross-hatch lines filling an area, bounded by
            an outline (typical of cross-section views).
    NOT   = mechanical line clusters, dashed outlines, texture, anything not
            a bounded regular fill.

No feature numbers, no IDs to type, no manual line-counting -- those stats
were classifier inputs, not labeling instructions; they're gone from the
default view because reading them couldn't help you decide.

Progress is saved after every region, so you can quit ('q') and resume any
time.

    python -m tools.hatch_label --pack output/PatentData/hatch_pilot
    python -m tools.hatch_label --pack output/PatentData/hatch_pilot --no-display

Commands at the prompt (one region at a time):
    y    hatch
    n    not hatch
    s    skip (leave unlabelled)
    b    go back one region
    q    save + quit
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

try:
    import cv2
    import matplotlib.pyplot as plt
    _HAVE_GUI = True
except ImportError:
    _HAVE_GUI = False

_viewer = {"fig": None, "ax": None}
_DEFAULT_TIF_ROOT = "data/PatentData/ReorganisedData"
_PAD_FRAC = 0.4   # context padding around the region bbox, as a fraction of bbox size
_MIN_PAD = 25


def _crop_region(tif_root, patent, sketch, boundary):
    """Zoomed crop around `boundary` (source/tif pixel coords) with it outlined."""
    tif = os.path.join(tif_root, patent, f"{patent}_{sketch}.tif")
    base = cv2.imread(tif, cv2.IMREAD_GRAYSCALE)
    if base is None:
        return None
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    poly = np.asarray(boundary, dtype=np.int32)
    x0, y0 = poly[:, 0].min(), poly[:, 1].min()
    x1, y1 = poly[:, 0].max(), poly[:, 1].max()
    pad = max(_MIN_PAD, int(_PAD_FRAC * max(x1 - x0, y1 - y0)))
    H, W = img.shape[:2]
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    cv2.polylines(img, [poly], True, (0, 0, 255), 3)
    crop = img[cy0:cy1, cx0:cx1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _show(img_rgb, title):
    if img_rgb is None:
        return
    try:
        if _viewer["fig"] is None:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
            _viewer["fig"], _viewer["ax"] = fig, ax
        fig, ax = _viewer["fig"], _viewer["ax"]
        ax.clear()
        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.2)
        try:
            fig.canvas.manager.window.activateWindow()
            fig.canvas.manager.window.raise_()
        except Exception:
            pass
    except Exception as e:
        print(f"  [display failed: {e!r}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--tif-root", default=_DEFAULT_TIF_ROOT)
    ap.add_argument("--redo", action="store_true", help="re-label already-labelled regions")
    ap.add_argument("--no-display", action="store_true", help="don't pop up the crop window")
    a = ap.parse_args()
    use_gui = _HAVE_GUI and not a.no_display

    files = sorted(glob.glob(os.path.join(a.pack, "*_regions.json")))
    if not files:
        print("no *_regions.json in", a.pack); return

    print("HATCH = regular parallel/cross-hatch fill bounded by an outline (cross-section "
          "shading). Anything else (line clusters, texture, dashed outlines) = NOT.\n")

    # flatten to a (file_idx, region_idx) list so 'b' can step back across figures too
    queue = []
    docs = []
    for fi, path in enumerate(files):
        doc = json.load(open(path))
        docs.append(doc)
        for ri in range(len(doc["regions"])):
            queue.append((fi, ri))

    def region_done(fi, ri):
        return docs[fi]["regions"][ri]["label"] is not None

    qi = 0
    while 0 <= qi < len(queue):
        fi, ri = queue[qi]
        doc, r = docs[fi], docs[fi]["regions"][ri]
        if region_done(fi, ri) and not a.redo:
            qi += 1; continue
        stem = f"{doc['patent']}__{doc['sketch']}"
        n_lab = sum(1 for d in docs for rr in d["regions"] if rr["label"] is not None)
        title = f"[{n_lab}/{len(queue)} labelled]  {stem}  region {r['id']}"
        print(f"\n{title}")
        if use_gui:
            crop = _crop_region(a.tif_root, doc["patent"], doc["sketch"], r["boundary"])
            if crop is None:
                print("  (tif not found, can't crop -- judging blind, prefer 's')")
            _show(crop, title)
        ans = input("  hatch? [y/n/s/b/q] : ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            qi = max(0, qi - 1); continue
        if ans == "s":
            qi += 1; continue
        if ans not in ("y", "n"):
            print("  ? not understood (use y/n/s/b/q), skipping"); continue
        r["label"] = (ans == "y")
        json.dump(doc, open(files[fi], "w"), indent=2)
        qi += 1

    if _viewer["fig"] is not None:
        plt.close(_viewer["fig"])

    pos = neg = unl = 0
    for d in docs:
        for r in d["regions"]:
            if r["label"] is True: pos += 1
            elif r["label"] is False: neg += 1
            else: unl += 1
    print(f"\nlabelled: {pos} hatch, {neg} not-hatch, {unl} unlabelled. "
          f"Train de-risk AUC with:  python -m tools.hatch_train --pack {a.pack}")


if __name__ == "__main__":
    main()
