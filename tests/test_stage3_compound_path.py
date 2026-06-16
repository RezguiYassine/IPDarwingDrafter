"""
Regression tests for the Stage 3 compound-path fitter.

Background: ~21% of patent primitives used to be raw jagged polylines (the
"could not fit a clean line/arc" fallback); 100% of the *curved* ones had
single-arc fit confidence < 0.45. Stage 3 now corner-splits the dense skeleton
and fits each piece line → arc → cubic Bézier, emitting one compound `path`
primitive. These tests lock in the two invariants that must not regress:

  1. Sharp corners stay sharp — an L-shape / zig-zag becomes a path of straight
     LINE segments split at the corners, never a single rounded curve.
  2. Smooth curves stay smooth — an S-curve becomes Bézier(s), a quarter circle
     an arc; neither degrades into a many-segment "line-soup" polyline.

Plus: a clean single line/arc still returns as a single top-level primitive
(the compound path is strictly a fallback — no regression on easy edges).

Runs under pytest, or standalone:

    python tests/test_stage3_compound_path.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage3_primitivesfitting"))
import stage3_primitive_fit as s3   # noqa: E402


def _edge(pts):
    pts = [[float(x), float(y)] for x, y in pts]
    return {"id": "t", "is_closed": False, "pixels": pts, "smooth_pts": pts}


def _seg_types(result):
    return [s["type"] for s in result["segments"]]


def test_lshape_two_lines_sharp_corner():
    pts = [(x, 0) for x in range(0, 60)] + [(60, y) for y in range(1, 60)]
    r = s3.fit_edge_ransac(_edge(pts))
    assert r["type"] == "path", r["type"]
    types = _seg_types(r)
    assert types == ["line", "line"], types          # 2 straight arms, sharp corner
    assert all(t != "bezier" for t in types)          # corner NOT rounded


def test_zigzag_three_lines():
    pts = ([(x, 0) for x in range(0, 30)]
           + [(30 + x, x) for x in range(0, 30)]
           + [(60 + x, 30 - x) for x in range(0, 30)])
    r = s3.fit_edge_ransac(_edge(pts))
    assert r["type"] == "path"
    assert _seg_types(r) == ["line", "line", "line"]


def test_quarter_circle_is_arc():
    th = np.linspace(0, math.pi / 2, 80)
    pts = np.column_stack([50 * np.cos(th), 50 * np.sin(th)])
    r = s3.fit_edge_ransac(_edge(pts))
    assert r["type"] == "arc", r["type"]


def test_scurve_is_smooth_not_linesoup():
    t = np.linspace(0, 2 * math.pi, 200)
    pts = np.column_stack([t * 30, 40 * np.sin(t)])
    r = s3.fit_edge_ransac(_edge(pts))
    assert r["type"] == "path"
    types = _seg_types(r)
    assert any(t == "bezier" for t in types), types    # smooth, not line-soup
    # a smooth S must not be shattered into many straight pieces
    assert sum(1 for t in types if t == "line") <= 1


def test_straight_line_stays_single_line():
    pts = [(x, 2 * x) for x in range(0, 80)]
    r = s3.fit_edge_ransac(_edge(pts))
    assert r["type"] == "line", r["type"]              # top-level, not a path


def test_compound_path_is_fallback_only():
    # A clean arc must return a single top-level 'arc', proving the compound
    # path never pre-empts a good single-primitive fit.
    th = np.linspace(0.2, 1.3, 60)
    pts = np.column_stack([100 + 40 * np.cos(th), 100 + 40 * np.sin(th)])
    r = s3.fit_edge_ransac(_edge(pts))
    assert r["type"] in ("arc", "ellipse"), r["type"]


def _run_standalone():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_standalone() else 1)
