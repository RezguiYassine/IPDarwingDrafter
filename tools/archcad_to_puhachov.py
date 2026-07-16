#!/usr/bin/env python3
"""Convert ArchCAD vector CAD → Puhachov keypoint training data (.npz).

ArchCAD (https://huggingface.co/datasets/jackluoluo/ArchCAD) ships explicit
primitive-level JSON (LINE / ARC / CIRCLE / ELLIPSE with full params) in a
980×980 raster frame.  This tool turns each drawing into the exact npz schema
the Puhachov trainer consumes:

    skeleton : uint8 (H, W)   0/255 1-px skeleton
    kps      : int32 (N, 3)   columns = x, y, type   (0 endpoint / 1 junction / 2 corner)

Pipeline per sample:
  1. Convert every JSON entity to a polyline (subpath) in native coords.
  2. Rasterize all polylines → binary → skeletonize (skimage) to 1-px.
  3. derive_keypoints() on the subpaths  (degree 1 → endpoint, ≥3 → junction,
     degree-2 sharp turn → corner) — REUSED from tools.d2c_keypoint_labels so
     the labels are byte-consistent with the D2C training pipeline.
  4. snap_keypoints() onto the skeleton; save {skeleton, kps} npz per split.

The domain is architectural (out-of-distribution vs mechanical/patent), but a
line/arc/junction/corner is geometrically domain-agnostic — floor plans are in
fact junction- and corner-rich, which targets Puhachov's weak corner channel.

Usage:
    python -m tools.archcad_to_puhachov \
        --json-dir data/ArchCAD/data/json \
        --out output/ArchCAD/kp_labels \
        --limit 4000 --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize as _skeletonize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tools.d2c_keypoint_labels import (  # noqa: E402
    derive_keypoints, snap_keypoints, KP_ENDPOINT, KP_JUNCTION, KP_CORNER,
)

logger = logging.getLogger("archcad_kp")

RENDER_DEFAULT = 980   # ArchCAD native raster frame


# ─── Primitive → polyline ─────────────────────────────────────────────────────

def _arc_points(cx, cy, r, a0_deg, a1_deg):
    """CCW arc from a0 to a1 (DXF convention), ~1-px segment sampling."""
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    if a1 <= a0:
        a1 += 2 * math.pi
    n = max(4, int(r * (a1 - a0)))          # ≈ 1 px per segment
    return [(cx + r * math.cos(a), cy + r * math.sin(a))
            for a in np.linspace(a0, a1, n)]


def entity_to_polyline(e: dict) -> list[tuple[float, float]] | None:
    """Return a polyline (subpath) for one ArchCAD JSON entity, or None."""
    t = e.get("type")
    if t == "LINE":
        return [tuple(e["start"]), tuple(e["end"])]
    if t == "ARC":
        return _arc_points(e["center"][0], e["center"][1], e["radius"],
                           e["start_angle"], e["end_angle"])
    if t == "CIRCLE":
        cx, cy, r = e["center"][0], e["center"][1], e["radius"]
        n = max(8, int(2 * math.pi * r))
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a))
               for a in np.linspace(0, 2 * math.pi, n)]
        return pts                                      # closed (first≈last)
    if t == "ELLIPSE":
        cx, cy = e["center"]
        ra, rb = e.get("radius_a"), e.get("radius_b")
        if ra is None or rb is None:
            return None
        theta = math.radians(e.get("theta", 0.0))
        ct, st = math.cos(theta), math.sin(theta)
        p0 = float(e.get("start_param", 0.0)); p1 = float(e.get("end_param", 2 * math.pi))
        if p1 <= p0:
            p1 += 2 * math.pi
        n = max(8, int(max(ra, rb) * (p1 - p0)))
        pts = []
        for a in np.linspace(p0, p1, n):
            x, y = ra * math.cos(a), rb * math.sin(a)
            pts.append((cx + x * ct - y * st, cy + x * st + y * ct))
        return pts
    return None


def rasterize(subpaths, render: int) -> np.ndarray:
    """Draw polylines into a `render`×`render` binary and skeletonize to 1-px."""
    canvas = np.zeros((render, render), np.uint8)
    for sp in subpaths:
        if len(sp) < 2:
            continue
        pts = np.round(np.asarray(sp)).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=1)
    if not canvas.any():
        return canvas
    return (_skeletonize(canvas > 0).astype(np.uint8) * 255)


def skeleton_endpoints_junctions(skel: np.ndarray,
                                 cluster_dist: int = 4) -> tuple[list, list]:
    """Endpoints (degree-1) and junctions (degree-≥3) from the *skeleton* itself.

    ArchCAD floor-plan walls are independent LINE entities that cross without a
    shared vertex, so vector-topology misses those junctions.  The rasterized
    skeleton, by contrast, shows every crossing as a degree-≥3 pixel — the exact
    signal the CNN sees and what classical crossing-number detection uses.
    Junction pixels are clustered (one keypoint per intersection).
    """
    s = (skel > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    deg = cv2.filter2D(s, -1, k, borderType=cv2.BORDER_CONSTANT) * s

    ys, xs = np.where(deg == 1)
    endpoints = [(int(x), int(y), KP_ENDPOINT) for x, y in zip(xs, ys)]

    junc_mask = (deg >= 3).astype(np.uint8)
    junctions: list[tuple[int, int, int]] = []
    if junc_mask.any():
        # dilate so a thick intersection collapses to one component, then take
        # each component centroid snapped back onto a skeleton pixel.
        d = cv2.dilate(junc_mask, np.ones((cluster_dist, cluster_dist), np.uint8))
        n, _, _, cents = cv2.connectedComponentsWithStats(d, connectivity=8)
        for cx, cy in cents[1:]:                       # skip background label 0
            xi, yi = int(round(cx)), int(round(cy))
            if 0 <= yi < s.shape[0] and 0 <= xi < s.shape[1] and s[yi, xi] == 0:
                # snap to nearest skeleton pixel in the neighbourhood
                y0, y1 = max(0, yi - cluster_dist), yi + cluster_dist
                x0, x1 = max(0, xi - cluster_dist), xi + cluster_dist
                sub = s[y0:y1, x0:x1]
                if sub.any():
                    sy, sx = np.argwhere(sub)[0]
                    xi, yi = x0 + sx, y0 + sy
            junctions.append((xi, yi, KP_JUNCTION))
    return endpoints, junctions


# ─── Per-sample worker ────────────────────────────────────────────────────────

_ARGS: dict | None = None


def _init(a: dict) -> None:
    global _ARGS
    _ARGS = a


def _process_one(job: tuple) -> dict:
    json_path, out_path = job
    a = _ARGS
    try:
        ents = json.load(open(json_path)).get("entities", [])
        subpaths = [pl for e in ents if (pl := entity_to_polyline(e)) and len(pl) >= 2]
        if not subpaths:
            return {"file": os.path.basename(json_path), "status": "empty"}

        # Clip to the render frame so out-of-bounds geometry doesn't wrap.
        R = a["render"]
        subpaths = [[(min(max(x, 0), R - 1), min(max(y, 0), R - 1)) for x, y in sp]
                    for sp in subpaths]

        skeleton = rasterize(subpaths, R)
        if not skeleton.any():
            return {"file": os.path.basename(json_path), "status": "empty_skeleton"}

        # Endpoints + junctions from the skeleton (captures wall crossings that
        # share no vector vertex); corners from vector geometry (clean angles).
        endpoints, junctions = skeleton_endpoints_junctions(skeleton)
        vector_kps = derive_keypoints(subpaths, scale=1.0,
                                      corner_angle_deg=a["corner_angle"],
                                      merge_tol=a["merge_tol"])
        corners = [(x, y, t) for (x, y, t) in vector_kps if t == KP_CORNER]
        corners_snapped, n_dropped = snap_keypoints(corners, skeleton, a["snap_radius"])

        # Drop corners that coincide with a junction (a crossing dominates a bend).
        jset = {(x, y) for x, y, _ in junctions}
        rad = a["snap_radius"]
        corners_snapped = [
            (x, y, t) for (x, y, t) in corners_snapped
            if not any(abs(x - jx) <= rad and abs(y - jy) <= rad for jx, jy in jset)
        ]

        all_kps = endpoints + junctions + corners_snapped
        kp_arr = (np.array(all_kps, dtype=np.int32)
                  if all_kps else np.zeros((0, 3), np.int32))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, skeleton=skeleton.astype(np.uint8), kps=kp_arr)

        by = np.bincount(kp_arr[:, 2], minlength=3) if len(kp_arr) else [0, 0, 0]
        return {"file": os.path.basename(json_path), "status": "ok",
                "n_entities": len(ents), "n_subpaths": len(subpaths),
                "n_kps": int(len(kp_arr)), "n_endpoint": int(by[0]),
                "n_junction": int(by[1]), "n_corner": int(by[2]),
                "n_dropped": int(n_dropped),
                "n_skel_px": int((skeleton > 0).sum())}
    except Exception as exc:
        return {"file": os.path.basename(json_path), "status": f"error:{type(exc).__name__}:{exc}"}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="data/ArchCAD/data/json")
    ap.add_argument("--out",      default="output/ArchCAD/kp_labels")
    ap.add_argument("--limit",    type=int, default=0, help="0 = all files")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--render",   type=int, default=RENDER_DEFAULT)
    ap.add_argument("--corner-angle", type=float, default=35.0)
    ap.add_argument("--merge-tol",    type=float, default=0.5)
    ap.add_argument("--snap-radius",  type=int,   default=5)
    ap.add_argument("--seed",     type=int, default=42)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    files = sorted(Path(a.json_dir).glob("*.json"))
    if a.limit:
        rng = random.Random(a.seed); rng.shuffle(files); files = sorted(files[:a.limit])
    if not files:
        raise SystemExit(f"No JSON under {a.json_dir}")

    rng = random.Random(a.seed)
    jobs = []
    for f in files:
        split = "validation" if rng.random() < a.val_frac else "train"
        out = Path(a.out) / split / f"{f.stem}.npz"
        jobs.append((str(f), str(out)))

    logger.info(f"ArchCAD → Puhachov labels: {len(jobs)} files → {a.out}  "
                f"(render={a.render}, corner_angle={a.corner_angle}, "
                f"merge_tol={a.merge_tol}, snap_radius={a.snap_radius})")

    cfg = dict(render=a.render, corner_angle=a.corner_angle,
               merge_tol=a.merge_tol, snap_radius=a.snap_radius)
    stats = {"ok": 0, "empty": 0, "empty_skeleton": 0, "error": 0}
    agg = {k: 0 for k in ("n_endpoint", "n_junction", "n_corner", "n_kps", "n_dropped")}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(cfg,)) as ex:
        futs = [ex.submit(_process_one, j) for j in jobs]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="convert"):
            r = fu.result()
            s = r["status"]
            key = "error" if s.startswith("error") else s
            stats[key] = stats.get(key, 0) + 1
            if s == "ok":
                for k in agg:
                    agg[k] += r[k]

    n_ok = max(1, stats["ok"])
    logger.info(f"\ndone: {stats}")
    logger.info(f"per-drawing avg keypoints: endpoint={agg['n_endpoint']/n_ok:.1f} "
                f"junction={agg['n_junction']/n_ok:.1f} corner={agg['n_corner']/n_ok:.1f} "
                f"total={agg['n_kps']/n_ok:.1f}  (dropped {agg['n_dropped']} off-skeleton)")
    logger.info(f"labels → {a.out}/(train|validation)")


if __name__ == "__main__":
    main()
