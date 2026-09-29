#!/usr/bin/env python3
"""Measure how well Stage 0 removes the leader lines it claims to remove.

Owner's audit remark on release 1: leader lines come out duplicated, most
visibly where the original is curved -- the reconstruction shows a straight
thin stroke beside the curve. That happens when the removal mask covers a
straight chord while the leader's ink follows a curve: the curve survives
into Stage 2 and is vectorized, and Stage 4 reinjects a straight leader next
to it.

This tool measures that directly, without trusting the pipeline's own trace.
For every removed leader it re-derives the leader's true path from the
source ink -- the shortest skeleton path between the two points Stage 0
recorded -- and asks how much of that path is still ink in the
reference-free image Stage 1 consumes.

    chord_on_ink   share of the straight chord lying on source ink; low means curved
    sinuosity      true path length / chord length
    ink_left       share of the true path still present after removal

The same measurement runs before and after a change, so the numbers compare.

    python -m tools.leader_audit --run output/RELEASE_2026-09-28 --limit 250
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402
from skimage.morphology import skeletonize   # noqa: E402

SCHEMA = "ap3-leader-audit-v1"
CURVED_CHORD_ON_INK = 0.60      # the population definition used on 2026-09-29
STRAIGHT_CHORD_ON_INK = 0.90


def _near(mask: np.ndarray, x: float, y: float, r: int) -> bool:
    xi, yi = int(round(x)), int(round(y))
    return bool(mask[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1].any())


def chord_on_ink(ink: np.ndarray, a, b, samples: int = 60, tolerance: int = 2) -> float:
    points = np.linspace(np.asarray(a, float), np.asarray(b, float), samples)
    return float(np.mean([_near(ink, x, y, tolerance) for x, y in points]))


def true_path(ink: np.ndarray, a, b, *, pad: int = 12, snap: int = 4):
    """Shortest 8-connected skeleton path between a and b, in figure coordinates.

    Returns None when the two points are not joined by ink within the local
    window -- a leader the audit cannot reconstruct is left out rather than
    guessed at.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    h, w = ink.shape
    x0 = int(max(0, min(a[0], b[0]) - pad)); x1 = int(min(w, max(a[0], b[0]) + pad + 1))
    y0 = int(max(0, min(a[1], b[1]) - pad)); y1 = int(min(h, max(a[1], b[1]) + pad + 1))
    crop = ink[y0:y1, x0:x1]
    if crop.size == 0 or not crop.any():
        return None
    skel = skeletonize(crop > 0)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return None

    def nearest(p):
        d2 = (xs - (p[0] - x0)) ** 2 + (ys - (p[1] - y0)) ** 2
        k = int(np.argmin(d2))
        return (int(xs[k]), int(ys[k])) if d2[k] <= snap * snap else None

    start, goal = nearest(a), nearest(b)
    if start is None or goal is None:
        return None
    # breadth-first over skeleton pixels; the skeleton is thin, so this is cheap
    previous = {start: None}
    frontier = collections.deque([start])
    while frontier:
        cx, cy = frontier.popleft()
        if (cx, cy) == goal:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not dx and not dy:
                    continue
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < skel.shape[1] and 0 <= ny < skel.shape[0]
                        and skel[ny, nx] and (nx, ny) not in previous):
                    previous[(nx, ny)] = (cx, cy)
                    frontier.append((nx, ny))
    if goal not in previous:
        return None
    path, node = [], goal
    while node is not None:
        path.append((node[0] + x0, node[1] + y0))
        node = previous[node]
    return np.array(path[::-1], float)


def audit_leader(ink: np.ndarray, left: np.ndarray, leader: dict) -> dict | None:
    a = leader.get("label_endpoint") or leader.get("p1")
    b = leader.get("leader_to") or leader.get("p2")
    if not a or not b:
        return None
    chord = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if chord < 25:
        return None
    path = true_path(ink, a, b)
    if path is None or len(path) < 5:
        return None
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    remaining = float(np.mean([_near(left, x, y, 1) for x, y in path]))
    return {"chord_px": chord, "path_px": float(steps.sum()),
            "sinuosity": float(steps.sum() / chord),
            "chord_on_ink": chord_on_ink(ink, a, b),
            "ink_left": remaining,
            "has_stored_path": bool(leader.get("path"))}


def geometry_along(path: np.ndarray, tree, tolerance: float = 3.0) -> float:
    """Share of a leader's true path that has vector geometry beside it.

    After removal the leader's ink should be gone, so Stage 3 has nothing to
    fit there; geometry running along the path is the ghost stroke the owner
    saw. The leader's own reinjected annotation is not in the primitives, so
    it is not counted.
    """
    if tree is None or len(path) == 0:
        return 0.0
    distances, _ = tree.query(path)
    return float((distances <= tolerance).mean())


def _primitive_tree(primitives_path: Path):
    from scipy.spatial import cKDTree
    from tools import geometry_validation as gv
    if not primitives_path.is_file():
        return None
    samples = []
    for primitive in json.loads(primitives_path.read_text()).get("primitives") or []:
        try:
            sampled, _ = gv.sample_primitive(primitive, 1.0, gv.POLICY)
        except Exception:
            continue
        samples.append(np.asarray(sampled, float))
    return cKDTree(np.vstack(samples)) if samples else None


def audit_run(run: Path, limit: int | None = None) -> list[dict]:
    rows = []
    documents = sorted(run.glob("*/references/*_references.json"))
    for path in documents[:limit] if limit else documents:
        doc = json.loads(path.read_text())
        source = doc.get("source_path")
        sketch = path.stem.replace("_references", "")
        norefs = path.parent / f"{sketch}_norefs.png"
        if not source or not norefs.is_file():
            continue
        gray = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        cleaned = cv2.imread(str(norefs), cv2.IMREAD_GRAYSCALE)
        if gray is None or cleaned is None:
            continue
        ink, left = gray < 128, cleaned < 128
        tree = _primitive_tree(path.parent.parent / "primitives" / f"{sketch}_primitives.json")
        for label in doc.get("reference_labels") or []:
            for leader in label.get("leader_lines") or []:
                if not leader.get("removed", True):
                    continue
                result = audit_leader(ink, left, leader)
                if result:
                    result["figure"] = f"{path.parent.parent.name}/{sketch}"
                    a = leader.get("label_endpoint") or leader.get("p1")
                    b = leader.get("leader_to") or leader.get("p2")
                    true = true_path(ink, a, b)
                    result["geometry_along"] = (geometry_along(true, tree)
                                                if true is not None and tree is not None else None)
                    rows.append(result)
    return rows


def summarise(rows: list[dict]) -> dict:
    def population(selected):
        if not selected:
            return {"n": 0}
        left = np.array([r["ink_left"] for r in selected])
        ghost = np.array([r["geometry_along"] for r in selected
                          if r.get("geometry_along") is not None])
        return {"n": len(selected), "ink_left_median": float(np.median(left)),
                "ink_left_mean": float(left.mean()),
                "leaders_with_over_half_left": int((left > 0.5).sum()),
                "ghost_median": float(np.median(ghost)) if len(ghost) else None,
                "leaders_with_ghost": int((ghost > 0.5).sum()) if len(ghost) else None}
    curved = [r for r in rows if r["chord_on_ink"] < CURVED_CHORD_ON_INK]
    straight = [r for r in rows if r["chord_on_ink"] >= STRAIGHT_CHORD_ON_INK]
    return {"schema": SCHEMA, "leaders": len(rows),
            "curved_share": len(curved) / max(1, len(rows)),
            "straight_share": len(straight) / max(1, len(rows)),
            "curved": population(curved), "straight": population(straight),
            "all": population(rows),
            "with_stored_path": sum(r["has_stored_path"] for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="reference documents to read")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rows = audit_run(args.run, args.limit)
    summary = summarise(rows)
    print(f"leaders reconstructed from source ink: {summary['leaders']}")
    print(f"  curved   (chord on ink < {CURVED_CHORD_ON_INK:.0%}): {summary['curved_share']:.1%}")
    print(f"  straight (chord on ink >= {STRAIGHT_CHORD_ON_INK:.0%}): {summary['straight_share']:.1%}")
    for name in ("curved", "straight", "all"):
        s = summary[name]
        if s["n"]:
            print(f"  ink left behind, {name:8s}: median {s['ink_left_median']:.2f}  "
                  f"mean {s['ink_left_mean']:.2f}  over half left on {s['leaders_with_over_half_left']}")
            if s.get("ghost_median") is not None:
                print(f"  geometry along leader, {name:8s}: median {s['ghost_median']:.2f}  "
                      f"leaders with a ghost (>50% of path) {s['leaders_with_ghost']}")
    print(f"  leaders carrying a stored path: {summary['with_stored_path']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "leaders": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
