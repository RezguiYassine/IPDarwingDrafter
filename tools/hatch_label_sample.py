#!/usr/bin/env python3
"""Build a hatch-region labelling pack (Phase 0 of the learned hatch detector).

Consumes Stage-2 graphs (which carry `removed_hachures`), aggregates candidate
hatch regions, computes a feature vector per region, and writes a labelling pack:

    <out>/<patent>__<sketch>_overlay.png   original drawing + numbered region polys
    <out>/<patent>__<sketch>_regions.json  {regions:[{id,boundary,angles,spacing,
                                            features,label:null}]} in source coords
    <out>/index.csv                         one row per figure

YOU then mark each region hatch / not-hatch (tools/hatch_label.py, or edit the
JSON / labels.csv). tools/hatch_train.py reads the labelled pack and reports the
de-risk ROC-AUC.

Usage:
    python -m tools.hatch_label_sample --batch-dir /tmp/hatch_pilot \
        --tif-root data/PatentData/ReorganisedData --out output/PatentData/hatch_pilot
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "stage2_strokeextraction"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
import stage2_stroke_extract as s2        # noqa: E402
import hatch_features as hf               # noqa: E402

_REGION_CFG = {"hachure_region_dilate": 5, "hachure_region_min_lines": 6,
               "hachure_crosshatch_ratio": 0.45}


def _members_for_region(region, removed, angle_tol=12.0):
    """Removed-hachure edges whose centroid is in the region and angle matches."""
    hull = np.asarray(region["boundary"], dtype=np.int32)
    ang = region["angles"][0] if region.get("angles") else None
    out = []
    for e in removed:
        pix = e.get("pixels") or []
        if len(pix) < 2:
            continue
        a = np.asarray(pix, dtype=np.float64)
        cx, cy = float(a[:, 0].mean()), float(a[:, 1].mean())
        if cv2.pointPolygonTest(hull, (cx, cy), False) < 0:
            continue
        la = hf._line_angle(pix)
        if ang is None or (la is not None and
                           min(hf._angle_delta(la, ra) for ra in region["angles"]) <= angle_tol):
            out.append(a)
    return out


def _ink_masks(graph, H, W):
    """Binary ink of all edges and of main (non-hachure) edges, in working coords."""
    ink_all = np.zeros((H, W), np.uint8)
    ink_main = np.zeros((H, W), np.uint8)
    for e in graph.get("edges", []):
        for x, y in e.get("pixels", []):
            if 0 <= y < H and 0 <= x < W:
                ink_all[y, x] = 1; ink_main[y, x] = 1
    for e in graph.get("removed_hachures", []):
        for x, y in e.get("pixels", []):
            if 0 <= y < H and 0 <= x < W:
                ink_all[y, x] = 1
    return ink_all, ink_main


def process_figure(graph_path, tif_root, out_dir):
    g = json.load(open(graph_path))
    removed = g.get("removed_hachures") or []
    if not removed:
        return None
    H, W = g["image_shape"]
    oH, oW = g.get("original_image_shape", [H, W])
    sx, sy = oW / float(W), oH / float(H)
    patent = os.path.basename(os.path.dirname(os.path.dirname(graph_path)))
    sketch = os.path.basename(graph_path).replace("_graph.json", "")

    regions = s2._aggregate_hachure_regions(removed, (H, W), _REGION_CFG)
    if not regions:
        return None
    ink_all, ink_main = _ink_masks(g, H, W)

    tif = os.path.join(tif_root, patent, f"{patent}_{sketch}.tif")
    base = cv2.imread(tif, cv2.IMREAD_GRAYSCALE)
    if base is None:
        base = (1 - ink_all) * 255  # fallback: working-coord canvas
        base = cv2.resize(base.astype(np.uint8), (oW, oH))
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    out_regions = []
    for i, r in enumerate(regions):
        members = _members_for_region(r, removed)
        feats = hf.region_feature_vector(r, members, ink_all, ink_main)
        # boundary in SOURCE coords for the overlay + stored JSON
        src_b = [[int(round(p[0] * sx)), int(round(p[1] * sy))] for p in r["boundary"]]
        poly = np.asarray(src_b, dtype=np.int32)
        col = (0, 0, 220) if r.get("double") else (220, 60, 0)
        cv2.polylines(vis, [poly], True, col, 3)
        cx, cy = int(poly[:, 0].mean()), int(poly[:, 1].mean())
        cv2.putText(vis, str(i), (cx - 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 130, 0), 4, cv2.LINE_AA)
        out_regions.append({
            "id": i, "boundary": src_b, "angles": r["angles"],
            "spacing": r["spacing"], "double": bool(r.get("double")),
            "n_lines": r["n_lines"], "features": feats, "label": None,
        })

    stem = f"{patent}__{sketch}"
    scale = 1100.0 / max(vis.shape[1], 1)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), vis)
    json.dump({"patent": patent, "sketch": sketch, "regions": out_regions},
              open(os.path.join(out_dir, f"{stem}_regions.json"), "w"), indent=2)
    return {"figure": stem, "n_regions": len(out_regions),
            "n_double": sum(1 for r in out_regions if r["double"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True, help="Stage-2 output root (<patent>/graphs/*.json)")
    ap.add_argument("--tif-root", default="data/PatentData/ReorganisedData")
    ap.add_argument("--out", default="output/PatentData/hatch_pilot")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    graphs = sorted(glob.glob(os.path.join(a.batch_dir, "*", "graphs", "*_graph.json")))
    rows, tot_r, tot_d = [], 0, 0
    for gp in graphs:
        try:
            r = process_figure(gp, a.tif_root, a.out)
        except Exception as exc:
            print(f"skip {gp}: {exc}", file=sys.stderr); continue
        if r:
            rows.append(r); tot_r += r["n_regions"]; tot_d += r["n_double"]
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "n_regions", "n_double"])
        w.writeheader(); w.writerows(rows)
    print(f"pack: {len(rows)} figures, {tot_r} candidate regions ({tot_d} cross-hatch) -> {a.out}")
    print("Next: label with  python -m tools.hatch_label --pack", a.out)


if __name__ == "__main__":
    main()
