"""
Regression tests for the Stage 2 keypoint → topology plumbing.

Background (Puhachov roadmap, Phase 3): `_extract_topology` now *consumes*
keypoint clusters passed as an argument instead of recomputing the crossing-
number (CN) topology internally. These tests lock in two invariants:

  1. Contract equivalence — feeding explicit CN clusters
     (`_cn_keypoint_clusters`) produces a byte-identical graph to letting
     `_extract_topology` default to internal CN seeding. This is what keeps the
     classical path unchanged while a learned detector can be swapped in.

  2. Canonical-shape topology — a closed rectangle, a line, a T-junction, and a
     circle-with-chord extract the expected nodes/edges (including the
     parallel-edge walk between two junctions).

Runs under pytest, or standalone with no test dependency:

    python tests/test_stage2_topology.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "stage2_strokeextraction"))

import stage2_stroke_extract as s2  # noqa: E402


# ─── Skeleton fixtures ───────────────────────────────────────────────────────

def _skeleton(draw, size: int = 256) -> np.ndarray:
    """Draw a shape on a black canvas and return its 1-px Zhang-Suen skeleton."""
    img = np.zeros((size, size), np.uint8)
    draw(img)
    return s2._skeletonize(img > 0).astype(np.uint8) * 255


def _rect(img):
    cv2.rectangle(img, (40, 40), (200, 160), 255, 2)


def _line(img):
    cv2.line(img, (30, 90), (220, 150), 255, 2)


def _tee(img):
    cv2.line(img, (30, 128), (220, 128), 255, 2)
    cv2.line(img, (128, 128), (128, 220), 255, 2)


def _two_arc_circle(img):
    # circle + horizontal chord through the centre → two junctions joined by
    # the top arc, the bottom arc, and the chord (three parallel edges).
    cv2.circle(img, (128, 128), 70, 255, 2)
    cv2.line(img, (58, 128), (198, 128), 255, 2)


_SHAPES = {
    "rect": _rect,
    "line": _line,
    "tee": _tee,
    "two_arc_circle": _two_arc_circle,
}


def _canon(nodes, edges) -> str:
    return json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True)


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_explicit_cn_clusters_match_default():
    """Explicit CN clusters → byte-identical graph vs default internal seeding."""
    for name, draw in _SHAPES.items():
        sk = _skeleton(draw)
        n_def, e_def = s2._extract_topology(sk)                            # default
        n_exp, e_exp = s2._extract_topology(sk, s2._cn_keypoint_clusters(sk))
        assert _canon(n_def, e_def) == _canon(n_exp, e_exp), \
            f"explicit CN clusters diverged from internal seeding on '{name}'"


def test_rectangle_is_single_closed_loop():
    nodes, edges = s2._extract_topology(_skeleton(_rect))
    closed = [e for e in edges if e["is_closed"]]
    assert len(closed) == 1, f"expected 1 closed loop, got {len(closed)}"
    assert not any(n["type"] == s2.KP_ENDPOINT for n in nodes), \
        "a closed rectangle must have no endpoints"


def test_line_has_two_endpoints_one_open_edge():
    nodes, edges = s2._extract_topology(_skeleton(_line))
    endpoints = [n for n in nodes if n["type"] == s2.KP_ENDPOINT]
    assert len(endpoints) == 2
    assert len(edges) == 1 and not edges[0]["is_closed"]


def test_tee_has_one_junction_three_endpoints():
    nodes, edges = s2._extract_topology(_skeleton(_tee))
    junctions = [n for n in nodes if n["type"] == s2.KP_JUNCTION]
    endpoints = [n for n in nodes if n["type"] == s2.KP_ENDPOINT]
    assert len(junctions) == 1, f"expected 1 junction, got {len(junctions)}"
    assert len(endpoints) == 3, f"expected 3 endpoints, got {len(endpoints)}"
    assert all(not e["is_closed"] for e in edges)


def test_two_arc_circle_keeps_parallel_edges():
    """Circle + chord → 2 junctions with multiple edges between them."""
    nodes, edges = s2._extract_topology(_skeleton(_two_arc_circle))
    junctions = [n for n in nodes if n["type"] == s2.KP_JUNCTION]
    assert len(junctions) == 2, f"expected 2 junctions, got {len(junctions)}"
    pair = frozenset(n["id"] for n in junctions)
    between = [e for e in edges
              if frozenset((e["source"], e["target"])) == pair and not e["is_closed"]]
    assert len(between) >= 2, \
        f"parallel-edge walk lost arcs: only {len(between)} edges between junctions"


def test_clusters_from_points_snap_onto_skeleton():
    """Off-skeleton keypoints snap to the nearest foreground pixel within radius."""
    sk = _skeleton(_tee)
    binary = sk > 0
    pts = [{"x": 130, "y": 130, "type": s2.KP_JUNCTION, "confidence": 0.9}]
    clusters = s2._clusters_from_points(pts, sk, snap_radius=5)
    assert len(clusters) == 1
    x, y = clusters[0]["pixels"][0]
    assert binary[y, x], "snapped core pixel must lie on the skeleton"


def test_clusters_from_points_dedupes_and_drops_far_points():
    sk = _skeleton(_line)
    # two points snapping to the same pixel → one cluster; a far point → dropped
    pts = [
        {"x": 30, "y": 90, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 31, "y": 91, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 5,  "y": 5,  "type": s2.KP_ENDPOINT, "confidence": 1.0},  # far off-skeleton
    ]
    clusters = s2._clusters_from_points(pts, sk, snap_radius=3)
    assert 1 <= len(clusters) <= 2, f"expected dedupe + far-point drop, got {len(clusters)}"


# ─── Standalone runner (no pytest dependency) ────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
