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


def test_ill_conditioned_bezier_handles_stay_local_to_source_chain():
    # Real D2C outlier 0002/00021249 FrontTopRight. The normal equations are
    # nearly singular here and previously produced a control point at
    # (3026, 2479) for this 57 px chain.
    points = np.asarray([
        [571, 316], [574, 321], [575, 322], [577, 322], [578, 322],
        [585, 329], [586, 330], [586, 331], [587, 332], [588, 333],
        [612, 354], [613, 355],
    ], dtype=np.float64)

    segment, _ = s3._fit_bezier_segment(points)

    controls = np.asarray(segment["points"], dtype=np.float64)
    source_length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    nearest_source = np.min(
        np.linalg.norm(controls[:, None, :] - points[None, :, :], axis=2),
        axis=1,
    )
    assert np.isfinite(controls).all()
    assert nearest_source.max() <= 2.0 * source_length


def test_compound_path_rejects_one_bad_segment_hidden_by_good_average(monkeypatch):
    points = np.column_stack([np.arange(20), np.zeros(20)])
    edge = _edge(points)
    bad_segment = {
        "type": "bezier",
        "points": [[0, 0], [5000, 5000], [-5000, -5000], [19, 0]],
    }
    monkeypatch.setattr(
        s3,
        "_fit_subsegment",
        lambda _sub: (bad_segment, 0.99),
    )

    assert s3._fit_compound_path(edge, "bad", require_fidelity=True) is None


def test_compound_path_uses_configurable_endpoint_fidelity(monkeypatch):
    points = np.column_stack([np.arange(20), np.zeros(20)])
    edge = _edge(points)
    shifted_segment = {
        "type": "line",
        "p1": [0.5, 0.0],
        "p2": [18.5, 0.0],
    }
    monkeypatch.setattr(
        s3,
        "_fit_subsegment",
        lambda _sub: (shifted_segment, 0.99),
    )

    accepted = s3._fit_compound_path(
        edge,
        "guarded",
        require_fidelity=True,
        max_endpoint_error=0.5,
    )
    rejected = s3._fit_compound_path(
        edge,
        "guarded",
        require_fidelity=True,
        max_endpoint_error=0.49,
    )

    assert accepted is not None
    assert accepted["fit_fidelity"]["target_endpoint_error"] == 0.5
    assert rejected is None


def test_compound_promotion_cannot_discard_a_short_endpoint_hook():
    pixels = (
        [(255, y) for y in range(510, 514)]
        + [(256, 514), (257, 514)]
        + [(x, 515) for x in range(258, 766)]
    )
    edge = {
        "id": "hook",
        "is_closed": False,
        "pixels": [[float(x), float(y)] for x, y in pixels],
        "smooth_pts": [[255.0, 510.0], [255.0, 513.0],
                       [258.0, 515.0], [765.0, 515.0]],
    }

    result = s3.fit_edge_ransac(edge, prefer_compound_over_weak=True)

    assert result["type"] == "line"
    assert result["confidence"] < 0.6


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


def test_opt_in_compound_path_can_replace_a_weak_single_fit():
    x = np.linspace(0, 120, 121)
    pts = np.column_stack([x, 3 * np.sin(x / 120 * 2 * math.pi)])
    edge = _edge(pts)

    baseline = s3.fit_edge_ransac(edge)
    promoted = s3.fit_edge_ransac(
        edge,
        prefer_compound_over_weak=True,
        weak_compound_max_path_atoms=64,
    )

    assert baseline["type"] == "line"
    assert 0.2 < baseline["confidence"] < 0.6
    assert promoted["type"] == "path"
    assert promoted["confidence"] >= 0.6
    assert promoted["fit_metadata"]["strategy"] == "compound_over_weak"
    assert promoted["fit_metadata"]["weak_type"] == "line"
    assert promoted["fit_metadata"]["path_atoms"] <= 64


def test_opt_in_compound_path_respects_complexity_cap():
    x = np.linspace(0, 120, 121)
    pts = np.column_stack([x, 3 * np.sin(x / 120 * 2 * math.pi)])
    edge = _edge(pts)

    result = s3.fit_edge_ransac(
        edge,
        prefer_compound_over_weak=True,
        weak_compound_max_path_atoms=1,
    )

    assert result["type"] == "line"
    assert "fit_metadata" not in result


def test_opt_in_closed_trace_replaces_raw_fixed_confidence_fallback():
    theta = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    radius = 60 + 8 * np.sin(8 * theta)
    points = np.rint(np.column_stack([
        100 + radius * np.cos(theta),
        100 + radius * np.sin(theta),
    ])).astype(int)
    # Stage 2 closed-loop pixels are not guaranteed to be traversal-ordered.
    points = np.unique(points, axis=0)
    edge = {
        "id": "closed",
        "is_closed": True,
        "pixels": points.tolist(),
        "smooth_pts": [],
    }

    baseline = s3.fit_edge_ransac(edge)
    simplified = s3.fit_edge_ransac(edge, simplify_closed_fallback=True)

    assert baseline["type"] == "polyline"
    assert baseline["confidence"] == 0.3
    assert simplified["type"] == "polygon"
    assert simplified["confidence"] >= 0.6
    metadata = simplified["fit_metadata"]
    assert metadata["strategy"] == "closed_simplified_trace"
    assert metadata["replaced_type"] == "polyline"
    assert 3 <= metadata["output_vertices"] <= 128
    assert metadata["source_points"] > metadata["output_vertices"]
    assert metadata["residual_rms"] < 2.0


def test_clean_closed_polygon_does_not_expand_to_bounded_trace():
    points = (
        [(x, 0) for x in range(40)]
        + [(39, y) for y in range(1, 30)]
        + [(x, 29) for x in range(38, -1, -1)]
        + [(0, y) for y in range(28, 0, -1)]
    )
    edge = {
        "id": "rectangle",
        "is_closed": True,
        "pixels": (points[::3] + points[1::3] + points[2::3]),
        "smooth_pts": [],
    }

    result = s3.fit_edge_ransac(edge, simplify_closed_fallback=True)

    assert result["type"] == "polygon"
    assert result["confidence"] >= 0.6
    assert result.get("fit_metadata", {}).get("strategy") != (
        "closed_simplified_trace"
    )


def test_closed_trace_rejects_a_candidate_that_misses_fidelity_target():
    theta = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    radius = 50 + 12 * np.sin(10 * theta)
    points = np.column_stack([
        80 + radius * np.cos(theta),
        80 + radius * np.sin(theta),
    ])

    result = s3._fit_closed_simplified_trace(
        points,
        "strict",
        vertex_caps=(3,),
        target_confidence=0.99,
        target_p95=0.1,
    )

    assert result is None


def test_closed_trace_fidelity_rejection_preserves_exact_raw_fallback():
    theta = np.linspace(0, 2 * math.pi, 360, endpoint=False)
    radius = 50 + 12 * np.sin(10 * theta)
    points = np.unique(np.rint(np.column_stack([
        80 + radius * np.cos(theta),
        80 + radius * np.sin(theta),
    ])).astype(int), axis=0)
    edge = {
        "id": "strict",
        "is_closed": True,
        "pixels": points.tolist(),
        "smooth_pts": [],
    }

    result = s3.fit_edge_ransac(
        edge,
        simplify_closed_fallback=True,
        closed_trace_vertex_caps=(3,),
        closed_trace_target_confidence=0.99,
        closed_trace_target_p95=0.1,
    )

    assert result["type"] == "polyline"
    assert result["confidence"] == 0.3
    assert "fit_metadata" not in result
    assert result["points"][0] == result["points"][-1]


def test_shuffled_closed_loop_uses_exact_local_traversal():
    points = (
        [(x, 0) for x in range(20)]
        + [(19, y) for y in range(1, 12)]
        + [(x, 11) for x in range(18, -1, -1)]
        + [(0, y) for y in range(10, 0, -1)]
    )
    shuffled = points[::3] + points[1::3] + points[2::3]

    ordered = s3._reorder_loop_pixels(shuffled)

    assert len(ordered) == len(points)
    assert {tuple(point) for point in ordered.astype(int)} == set(points)
    closed = np.vstack([ordered, ordered[0]])
    steps = np.abs(np.diff(closed, axis=0))
    assert np.all(np.max(steps, axis=1) == 1)


def test_branched_pseudo_loop_has_linear_bounded_adjacency_trace():
    points = {(x, 20) for x in range(5000)}
    points.update({(2500, y) for y in range(41)})

    ordered = s3._reorder_loop_pixels(sorted(points, reverse=True))

    assert set(map(tuple, ordered.astype(int))) == points
    assert len(ordered) <= 2 * len(points) - 1
    steps = np.abs(np.diff(ordered, axis=0))
    assert np.all(np.max(steps, axis=1) == 1)


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
