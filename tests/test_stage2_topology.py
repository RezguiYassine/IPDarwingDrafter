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


def test_unclaimed_rectangle_is_marked_as_simple_cycle():
    skeleton = np.zeros((32, 32), np.uint8)
    cv2.rectangle(skeleton, (6, 7), (25, 24), 255, 1)

    _nodes, edges = s2._extract_topology(skeleton, [], unclaimed_mode="all")

    assert len(edges) == 1
    assert edges[0]["topology_origin"] == "unclaimed_component"
    assert edges[0]["is_simple_cycle"] is True


def test_unclaimed_branched_network_is_split_into_open_paths():
    skeleton = np.zeros((32, 32), np.uint8)
    skeleton[16, 4:28] = 255
    skeleton[4:28, 16] = 255

    _nodes, edges = s2._extract_topology(skeleton, [], unclaimed_mode="all")

    assert len(edges) == 4
    assert all(edge["topology_origin"] == "recovered_residual" for edge in edges)
    assert all(not edge["is_closed"] for edge in edges)
    assert {tuple(p) for e in edges for p in e["pixels"]} == {
        (int(x), int(y)) for y, x in zip(*np.nonzero(skeleton))}


def test_directional_walk_routes_unseeded_crossing_straight_through():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    points = [
        {"x": 8, "y": 32, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 56, "y": 32, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 32, "y": 8, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 32, "y": 56, "type": s2.KP_ENDPOINT, "confidence": 1.0},
    ]
    clusters = s2._clusters_from_points(points, skeleton, snap_radius=1)

    _nodes, edges = s2._extract_topology(
        skeleton,
        clusters,
        unclaimed_mode="none",
        directional_walk=True,
        directional_walk_baseline=6.0,
    )

    assert len(edges) == 2
    horizontal = [
        edge for edge in edges
        if all(pixel[1] == 32 for pixel in edge["pixels"])
    ]
    vertical = [
        edge for edge in edges
        if all(pixel[0] == 32 for pixel in edge["pixels"])
    ]
    assert len(horizontal) == 1
    assert len(vertical) == 1
    assert len(horizontal[0]["pixels"]) >= 45
    assert len(vertical[0]["pixels"]) >= 45


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


def test_closed_loop_smoothing_keeps_unordered_pixels_untouched():
    """Closed CC pixels are unordered and must never enter RDP/splprep."""
    _nodes, edges = s2._extract_topology(_skeleton(_rect))
    closed = next(edge for edge in edges if edge["is_closed"])
    original = json.dumps(closed["pixels"])
    out = s2._smooth_edges(edges)
    smoothed = next(edge for edge in out if edge["is_closed"])
    assert json.dumps(smoothed["pixels"]) == original
    assert smoothed["smooth_pts"] == []


def test_simplify_graph_preserves_explicit_corner_split():
    """A learned corner must remain a Stage-3 primitive boundary."""
    img = np.zeros((256, 256), np.uint8)
    cv2.line(img, (40, 180), (128, 180), 255, 2)
    cv2.line(img, (128, 180), (128, 60), 255, 2)
    sk = s2._skeletonize(img > 0).astype(np.uint8) * 255
    points = [
        {"x": 40, "y": 180, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 128, "y": 180, "type": s2.KP_CORNER, "confidence": 1.0},
        {"x": 128, "y": 60, "type": s2.KP_ENDPOINT, "confidence": 1.0},
    ]
    clusters = s2._clusters_from_points(points, sk, snap_radius=4)
    nodes, edges = s2._extract_topology(sk, clusters)
    nodes, edges = s2._simplify_graph(nodes, edges, junction_merge_radius=0.0)
    assert len(edges) == 2, f"corner was dissolved into {len(edges)} edge(s)"
    assert any(node["type"] == s2.KP_CORNER for node in nodes)


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


def _grid(img):
    # A grid of lines: many junctions joined by degree-2 chains — the structure
    # that triggered the dangling-reference bug (a once-per-pass static
    # adjacency popped a node an in-pass merge had just made an edge endpoint).
    for y in range(40, 220, 30):
        cv2.line(img, (40, y), (200, y), 255, 2)
    for x in range(40, 220, 30):
        cv2.line(img, (x, 40), (x, 190), 255, 2)


def test_simplify_graph_has_no_dangling_references():
    """Every edge endpoint must exist as a node after simplification."""
    sk = _skeleton(_grid, size=256)
    nodes, edges = s2._extract_topology(sk, kp_clusters=None)
    for radius in (0.0, 4.0):           # both the plain and hairball-merge paths
        n2, e2 = s2._simplify_graph(
            nodes, edges, spur_min_len=6.0, collinear_max_angle=28.0,
            junction_merge_radius=radius,
        )
        ids = {n["id"] for n in n2}
        dangling = [e for e in e2
                    if e["source"] not in ids or e["target"] not in ids]
        assert not dangling, (
            f"radius={radius}: {len(dangling)}/{len(e2)} edges reference a "
            f"missing node")


def test_hachure_pixel_subtraction_preserves_shared_crossing():
    skeleton = np.zeros((32, 32), np.uint8)
    skeleton[16, 4:28] = 255
    skeleton[4:28, 16] = 255
    kept = [{"pixels": [[x, 16] for x in range(4, 28)]}]
    removed = [{"pixels": [[16, y] for y in range(4, 28)]}]

    cleaned, count = s2._skeleton_without_hachure_edges(
        skeleton, kept, removed
    )

    assert count == 23
    assert np.all(cleaned[16, 4:28] > 0)
    assert np.count_nonzero(cleaned[4:28, 16]) == 1


def test_hachure_gap_repair_restores_contour_but_not_hatch_direction():
    original = np.zeros((65, 65), np.uint8)
    original[32, 8:57] = 255
    original[8:57, 32] = 255
    cleaned = original.copy()
    cleaned[32, 29:36] = 0
    cleaned[29:36, 32] = 0
    removed_hachures = [{
        "id": 0,
        "source": 0,
        "target": 1,
        "pixels": [[32, y] for y in range(8, 57)],
        "is_closed": False,
    }]
    config = {
        "hachure_gap_repair_max_path_length": 12.0,
        "hachure_gap_repair_max_angle": 20.0,
        "hachure_gap_repair_tangent_support": 10.0,
        "hachure_gap_repair_hatch_angle_margin": 15.0,
        "hachure_gap_repair_hatch_context_radius": 64.0,
    }

    repaired, repairs = s2._repair_hachure_crossing_gaps(
        original, cleaned, removed_hachures, config
    )

    assert len(repairs) == 1
    assert np.all(repaired[32, 8:57] > 0)
    assert np.count_nonzero(repaired[29:36, 32]) == 1
    assert repairs[0]["hatch_angle_delta"] > 15.0


def test_hachure_gap_repair_does_not_invent_missing_raster_geometry():
    original = np.zeros((65, 65), np.uint8)
    original[32, 8:28] = 255
    original[32, 37:57] = 255
    cleaned = original.copy()

    repaired, repairs = s2._repair_hachure_crossing_gaps(
        original,
        cleaned,
        [],
        {"hachure_gap_repair_max_path_length": 14.0},
    )

    assert not repairs
    assert np.array_equal(repaired, cleaned)


def test_hachure_topology_prepass_removes_hatch_branch_before_main_graph():
    skeleton = np.zeros((64, 64), np.uint8)
    skeleton[32, 8:56] = 255
    hatch_xs = (20, 28, 36, 44)
    for x in hatch_xs:
        skeleton[8:56, x] = 255
    # The CNN predicts a filled hatch region, not individual hatch strokes.
    hatch_mask = np.ones_like(skeleton)
    config = {
        "remove_hachures": True,
        "hachure_cnn_inside_frac": 0.60,
        "hachure_cnn_min_length": 1.0,
        "hachure_cnn_max_removed_edge_ratio": 0.92,
        "hachure_min_graph_edges": 0,
        "hachure_topology_prepass_max_removed_pixel_ratio": 0.95,
        "simplify_graph": True,
        "spur_min_length": 0.0,
        "merge_collinear_max_angle": 28.0,
        "junction_merge_radius": 0.0,
    }

    cleaned, removed, removed_pixels = s2._run_hachure_topology_prepass(
        skeleton, hatch_mask, config, max_search_radius=60
    )

    assert removed
    assert removed_pixels > 0
    assert all(
        edge["hachure"]["pass"] == "cnn_topology_prepass_geometric"
        for edge in removed
    )
    assert np.all(cleaned[32, 8:56] > 0)
    for x in hatch_xs:
        assert np.count_nonzero(cleaned[8:56, x]) <= 3


def test_cnn_geometric_intersection_keeps_long_contour():
    skeleton = np.zeros((64, 64), np.uint8)
    skeleton[32, 8:56] = 255
    hatch_xs = (20, 28, 36, 44)
    for x in hatch_xs:
        skeleton[8:56, x] = 255
    nodes, edges = s2._extract_topology(
        skeleton, s2._cn_keypoint_clusters(skeleton)
    )
    nodes, edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=0.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )
    config = {
        "remove_hachures": True,
        "hachure_min_graph_edges": 0,
        "hachure_min_length": 5.0,
        "hachure_max_length": 80.0,
        "hachure_min_straightness": 0.7,
        "hachure_max_residual_rms": 2.2,
        "hachure_angle_tolerance": 12.0,
        "hachure_cluster_radius": 95.0,
        "hachure_min_cluster_edges": 4,
        "hachure_min_cluster_total_length": 35.0,
        "hachure_cnn_inside_frac": 0.60,
        "hachure_cnn_min_length": 1.0,
        "hachure_cnn_geometric_max_candidate_ratio": 0.95,
    }

    _nodes, kept, removed = s2._remove_hachures_cnn_geometric(
        nodes,
        edges,
        np.ones_like(skeleton),
        config,
        pass_name="test_cnn_geometric",
    )

    assert len(removed) == len(hatch_xs)
    assert all(
        edge["hachure"]["pass"] == "test_cnn_geometric"
        for edge in removed
    )
    assert len(kept) == 1
    assert all(pixel[1] == 32 for pixel in kept[0]["pixels"])
    assert s2._chain_length(kept[0]["pixels"]) >= 44.0


def test_hatch_additive_mask_excludes_region_evidence():
    region = np.zeros((8, 8), np.uint8)
    region[1:5, 1:5] = 255
    stroke = np.zeros_like(region)
    stroke[3:7, 3:7] = 1

    separated_region, additive = s2._separate_hatch_additive_masks(
        region, stroke
    )

    assert np.array_equal(separated_region, region > 0)
    assert not np.any(additive[region > 0])
    assert np.array_equal(additive > 0, (stroke > 0) & (region == 0))


def test_hatch_additive_mask_supports_missing_region():
    stroke = np.zeros((8, 8), np.uint8)
    stroke[2:6, 3] = 1

    region, additive = s2._separate_hatch_additive_masks(None, stroke)

    assert not np.any(region)
    assert np.array_equal(additive, stroke)


def test_hough_hachure_mask_selects_periodic_family_not_single_contour():
    skeleton = np.zeros((256, 256), np.uint8)
    cv2.line(skeleton, (16, 128), (240, 128), 255, 1)
    hatch_only = np.zeros_like(skeleton)
    for offset in range(-176, 177, 16):
        cv2.line(hatch_only, (20, 20 + offset), (236, 236 + offset), 255, 1)
    skeleton |= hatch_only
    region = np.zeros_like(skeleton)
    region[8:248, 8:248] = 1
    config = {
        "hachure_hough_min_region_area": 500,
        "hachure_hough_threshold": 8,
        "hachure_hough_min_line_length": 10.0,
        "hachure_hough_max_line_gap": 3.0,
        "hachure_hough_angle_step": 5.0,
        "hachure_hough_angle_tolerance": 7.5,
        "hachure_hough_rho_tolerance": 3.0,
        "hachure_hough_min_family_lines": 6,
        "hachure_hough_max_families": 1,
        "hachure_hough_mask_thickness": 3,
        "hachure_hough_interior_margin": 0,
    }

    stroke_mask, families = s2._hough_hachure_stroke_mask(
        skeleton, region, config
    )

    assert len(families) == 1
    assert s2._angle_delta_deg(families[0]["angle_deg"], 45.0) <= 7.5
    hatch_pixels = hatch_only > 0
    assert np.count_nonzero((stroke_mask > 0) & hatch_pixels) > 0.5 * np.count_nonzero(
        hatch_pixels
    )
    horizontal_support = np.flatnonzero(stroke_mask[128, :] > 0)
    runs = np.split(
        horizontal_support,
        np.where(np.diff(horizontal_support) > 1)[0] + 1,
    )
    assert max((len(run) for run in runs), default=0) < 10


def test_hough_hachure_mask_rejects_two_parallel_structural_lines():
    skeleton = np.zeros((128, 128), np.uint8)
    cv2.line(skeleton, (12, 40), (116, 40), 255, 1)
    cv2.line(skeleton, (12, 88), (116, 88), 255, 1)

    stroke_mask, families = s2._hough_hachure_stroke_mask(
        skeleton,
        np.ones_like(skeleton),
        {
            "hachure_hough_min_region_area": 100,
            "hachure_hough_threshold": 8,
            "hachure_hough_min_line_length": 10.0,
            "hachure_hough_min_family_lines": 6,
            "hachure_hough_interior_margin": 0,
        },
    )

    assert not families
    assert not np.any(stroke_mask)


def test_simplification_restores_closed_cycle_split_by_temporary_nodes():
    nodes = [
        {"id": 0, "x": 8, "y": 8, "type": s2.KP_JUNCTION},
        {"id": 1, "x": 24, "y": 8, "type": s2.KP_JUNCTION},
        {"id": 2, "x": 16, "y": 24, "type": s2.KP_JUNCTION},
    ]
    edges = [
        {"id": 0, "source": 0, "target": 1,
         "pixels": [[8, 8], [24, 8]], "is_closed": False},
        {"id": 1, "source": 1, "target": 2,
         "pixels": [[24, 8], [16, 24]], "is_closed": False},
        {"id": 2, "source": 2, "target": 0,
         "pixels": [[16, 24], [8, 8]], "is_closed": False},
    ]

    simplified_nodes, simplified_edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=0.0,
        collinear_max_angle=0.0,
        junction_merge_radius=0.0,
    )

    assert len(simplified_nodes) == 1
    assert len(simplified_edges) == 1
    assert simplified_edges[0]["source"] == simplified_edges[0]["target"]
    assert simplified_edges[0]["is_closed"]


def test_longer_tangent_baseline_routes_through_noisy_junction():
    nodes = [
        {"id": 0, "x": 50, "y": 50, "type": s2.KP_JUNCTION},
        {"id": 1, "x": 20, "y": 50, "type": s2.KP_ENDPOINT},
        {"id": 2, "x": 80, "y": 50, "type": s2.KP_ENDPOINT},
        {"id": 3, "x": 50, "y": 20, "type": s2.KP_ENDPOINT},
    ]
    left = [[50, 50], [49, 51], [48, 52], [47, 53], [46, 54],
            [45, 55], [44, 56], [40, 55], [35, 53], [30, 51], [20, 50]]
    right = [[50, 50], [51, 51], [52, 52], [53, 53], [54, 54],
             [55, 55], [56, 56], [60, 55], [65, 53], [70, 51], [80, 50]]
    branch = [[50, y] for y in range(50, 19, -1)]
    edges = [
        {"id": 0, "source": 0, "target": 1, "pixels": left,
         "is_closed": False},
        {"id": 1, "source": 0, "target": 2, "pixels": right,
         "is_closed": False},
        {"id": 2, "source": 0, "target": 3, "pixels": branch,
         "is_closed": False},
    ]

    _short_nodes, short_edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=0.0,
        collinear_max_angle=28.0,
        collinear_tangent_baseline=8.0,
        junction_merge_radius=0.0,
    )
    _long_nodes, long_edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=0.0,
        collinear_max_angle=28.0,
        collinear_tangent_baseline=24.0,
        junction_merge_radius=0.0,
    )

    assert len(short_edges) == 3
    assert len(long_edges) == 2
    through = max(long_edges, key=lambda edge: s2._chain_length(edge["pixels"]))
    assert {through["source"], through["target"]} == {1, 2}


def test_hatch_bridge_junction_preserves_structural_crossing_continuity():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    hatch_mask = np.zeros_like(skeleton)
    hatch_mask[8:57, 30:35] = 1
    structural = [
        {
            "x": 8,
            "y": 32,
            "type": s2.KP_ENDPOINT,
            "confidence": 1.0,
            "pixels": [(8, 32)],
            "structural_seed": True,
        },
        {
            "x": 56,
            "y": 32,
            "type": s2.KP_ENDPOINT,
            "confidence": 1.0,
            "pixels": [(56, 32)],
            "structural_seed": True,
        },
    ]

    bridges = s2._hatch_bridge_junction_clusters(
        skeleton, hatch_mask, structural
    )
    nodes, edges = s2._extract_topology(
        skeleton,
        structural + bridges,
        unclaimed_mode="closed_only",
    )
    nodes, edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=6.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )

    assert len(bridges) == 1
    assert len(nodes) == 2
    assert len(edges) == 1
    assert not edges[0]["is_closed"]
    assert all(pixel[1] == 32 for pixel in edges[0]["pixels"])
    assert len(edges[0]["pixels"]) >= 45


def test_hatch_cn_endpoint_recovery_tracks_only_masked_stroke():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    hatch_stroke_mask = np.zeros_like(skeleton)
    hatch_stroke_mask[8:57, 32] = 1
    existing = [{
        "x": 32,
        "y": 8,
        "type": s2.KP_ENDPOINT,
        "confidence": 1.0,
        "pixels": [(32, 8)],
    }]

    endpoints = s2._hatch_cn_endpoint_clusters(
        skeleton,
        hatch_stroke_mask,
        existing,
        dedup_radius=2.0,
    )

    assert [(cluster["x"], cluster["y"]) for cluster in endpoints] == [
        (32, 56),
    ]
    assert not endpoints[0]["structural_seed"]
    assert endpoints[0]["topology_origin"] == "hatch_endpoint_cn"


def test_mixed_hatch_junction_filter_rejects_hatch_crossing_and_mask_touch():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255

    vertical_hatch = np.zeros_like(skeleton)
    vertical_hatch[8:57, 32] = 1
    candidates = s2._hatch_bridge_junction_clusters(
        skeleton, vertical_hatch, [], dedup_radius=0.0
    )
    vertical_family = [{"angle_deg": 90.0, "bbox": [0, 0, 65, 65]}]
    fractions = s2._junction_arm_mask_fractions(
        skeleton, vertical_hatch, candidates[0]
    )
    mixed = s2._mixed_hatch_junction_clusters(
        skeleton, vertical_hatch, candidates, {}, vertical_family
    )
    assert sorted(fractions) == [0.0, 0.0, 1.0, 1.0]
    assert len(mixed) == 1

    all_hatch = (skeleton > 0).astype(np.uint8)
    all_hatch_candidates = s2._hatch_bridge_junction_clusters(
        skeleton, all_hatch, [], dedup_radius=0.0
    )
    assert not s2._mixed_hatch_junction_clusters(
        skeleton,
        all_hatch,
        all_hatch_candidates,
        {},
        [
            {"angle_deg": 0.0, "bbox": [0, 0, 65, 65]},
            {"angle_deg": 90.0, "bbox": [0, 0, 65, 65]},
        ],
    )

    centre_touch = np.zeros_like(skeleton)
    centre_touch[32, 32] = 1
    touch_candidates = s2._hatch_bridge_junction_clusters(
        skeleton, centre_touch, [], dedup_radius=0.0
    )
    assert len(touch_candidates) == 1
    assert not s2._mixed_hatch_junction_clusters(
        skeleton, centre_touch, touch_candidates, {}, vertical_family
    )

    hairball = skeleton.copy()
    cv2.line(hairball, (32, 32), (56, 56), 255, 1)
    hairball_candidates = s2._hatch_bridge_junction_clusters(
        hairball, vertical_hatch, [], dedup_radius=0.0
    )
    assert not s2._mixed_hatch_junction_clusters(
        hairball,
        vertical_hatch,
        hairball_candidates,
        {},
        vertical_family,
    )


def test_hough_routing_splits_cross_before_hatch_cleanup():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    hatch_stroke_mask = np.zeros_like(skeleton)
    hatch_stroke_mask[8:57, 32] = 1
    structural_points = [
        {"x": 8, "y": 32, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 56, "y": 32, "type": s2.KP_ENDPOINT, "confidence": 1.0},
    ]
    clusters = s2._clusters_from_points(
        structural_points, skeleton, snap_radius=1
    )
    bridge_candidates = s2._hatch_bridge_junction_clusters(
        skeleton, hatch_stroke_mask, clusters, dedup_radius=1.0
    )
    bridges = s2._mixed_hatch_junction_clusters(
        skeleton,
        hatch_stroke_mask,
        bridge_candidates,
        {},
        [{"angle_deg": 90.0, "bbox": [0, 0, 65, 65]}],
    )
    hatch_endpoints = s2._hatch_cn_endpoint_clusters(
        skeleton,
        hatch_stroke_mask,
        clusters + bridges,
        dedup_radius=1.0,
    )

    nodes, edges = s2._extract_topology(
        skeleton,
        clusters + bridges + hatch_endpoints,
        unclaimed_mode="all",
    )
    nodes, edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=0.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )
    nodes, kept, removed = s2._remove_hachures_cnn(
        nodes,
        edges,
        hatch_stroke_mask,
        {
            "hachure_cnn_inside_frac": 0.30,
            "hachure_cnn_min_length": 1.0,
            "hachure_cnn_max_removed_edge_ratio": 0.90,
        },
        pass_name="cnn",
    )

    assert len(bridges) == 1
    assert len(hatch_endpoints) == 2
    assert len(kept) == 1
    assert len(removed) == 1
    assert all(pixel[1] == 32 for pixel in kept[0]["pixels"])
    assert all(pixel[0] == 32 for pixel in removed[0]["pixels"])
    assert s2._chain_length(kept[0]["pixels"]) >= 45.0


def test_hough_routing_keeps_true_tee_contour_after_branch_cleanup():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[32:57, 32] = 255
    hatch_stroke_mask = np.zeros_like(skeleton)
    hatch_stroke_mask[32:57, 32] = 1
    structural_points = [
        {"x": 8, "y": 32, "type": s2.KP_ENDPOINT, "confidence": 1.0},
        {"x": 56, "y": 32, "type": s2.KP_ENDPOINT, "confidence": 1.0},
    ]
    clusters = s2._clusters_from_points(
        structural_points, skeleton, snap_radius=1
    )
    bridge_candidates = s2._hatch_bridge_junction_clusters(
        skeleton, hatch_stroke_mask, clusters, dedup_radius=1.0
    )
    bridges = s2._mixed_hatch_junction_clusters(
        skeleton,
        hatch_stroke_mask,
        bridge_candidates,
        {},
        [{"angle_deg": 90.0, "bbox": [0, 0, 65, 65]}],
    )
    hatch_endpoints = s2._hatch_cn_endpoint_clusters(
        skeleton,
        hatch_stroke_mask,
        clusters + bridges,
        dedup_radius=1.0,
    )

    nodes, edges = s2._extract_topology(
        skeleton,
        clusters + bridges + hatch_endpoints,
        unclaimed_mode="all",
    )
    nodes, edges = s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=0.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )
    _nodes, kept, removed = s2._remove_hachures_cnn(
        nodes,
        edges,
        hatch_stroke_mask,
        {
            "hachure_cnn_inside_frac": 0.30,
            "hachure_cnn_min_length": 1.0,
            "hachure_cnn_max_removed_edge_ratio": 0.90,
        },
        pass_name="cnn",
    )

    assert len(bridges) == 1
    assert len(hatch_endpoints) == 1
    assert len(kept) == 1
    assert len(removed) == 1
    assert all(pixel[1] == 32 for pixel in kept[0]["pixels"])
    assert all(pixel[0] == 32 for pixel in removed[0]["pixels"])
    assert s2._chain_length(kept[0]["pixels"]) >= 45.0


def test_cn_endpoint_recovery_excludes_hatch_and_deduplicates_learned_seed():
    skeleton = np.zeros((65, 65), np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    hatch_stroke_mask = np.zeros_like(skeleton)
    hatch_stroke_mask[8:57, 31:34] = 1
    structural = [{
        "x": 8,
        "y": 32,
        "type": s2.KP_ENDPOINT,
        "confidence": 1.0,
        "pixels": [(8, 32)],
        "structural_seed": True,
    }]

    recovered = s2._structural_cn_endpoint_clusters(
        skeleton,
        hatch_stroke_mask,
        structural,
        dedup_radius=3.0,
    )

    assert [(cluster["x"], cluster["y"]) for cluster in recovered] == [
        (56, 32),
    ]
    assert recovered[0]["structural_seed"]
    assert recovered[0]["topology_origin"] == "structural_cn_endpoint_recovery"


def test_closed_only_unclaimed_keeps_cycle_and_drops_open_hatch():
    skeleton = np.zeros((65, 65), np.uint8)
    cv2.circle(skeleton, (32, 32), 15, 255, 1)
    cycle_nodes, cycle_edges = s2._extract_topology(
        skeleton, [], unclaimed_mode="closed_only"
    )

    open_hatch = np.zeros_like(skeleton)
    open_hatch[32, 8:57] = 255
    hatch_nodes, hatch_edges = s2._extract_topology(
        open_hatch, [], unclaimed_mode="closed_only"
    )

    assert len(cycle_nodes) == 1
    assert len(cycle_edges) == 1
    assert cycle_edges[0]["is_closed"]
    assert hatch_nodes == []
    assert hatch_edges == []


def test_hachure_side_layer_dedup_keeps_crossing_distinct_lines():
    edges = [
        {
            "id": 0,
            "pixels": [[x, 10] for x in range(2, 21)],
            "hachure": {"pass": "cnn_topology_prepass"},
        },
        {
            "id": 1,
            "pixels": [[x, 10] for x in range(4, 19)],
            "hachure": {"pass": "cnn"},
        },
        {
            "id": 2,
            "pixels": [[10, y] for y in range(2, 21)],
            "hachure": {"pass": "cnn"},
        },
    ]

    deduplicated = s2._deduplicate_hachure_edges(edges, 0.75)

    assert [edge["id"] for edge in deduplicated] == [0, 2]


def test_cnn_hachure_removal_combines_seed_and_strong_mask_evidence():
    curved_pixels = []
    for angle in np.linspace(0.0, np.pi, 40):
        pixel = [
            int(round(32 + 15 * np.cos(angle))),
            int(round(32 + 15 * np.sin(angle))),
        ]
        if not curved_pixels or curved_pixels[-1] != pixel:
            curved_pixels.append(pixel)
    nodes = [
        {"id": 0, "structural_seed": True},
        {"id": 1, "structural_seed": True},
        {"id": 2},
        {"id": 3},
        {"id": 4, "structural_seed": True},
        {"id": 5, "structural_seed": True},
    ]
    edges = [
        {
            "id": 0,
            "source": 0,
            "target": 1,
            "pixels": [[x, 10] for x in range(2, 20)],
            "is_closed": False,
        },
        {
            "id": 1,
            "source": 2,
            "target": 3,
            "pixels": [[x, 20] for x in range(2, 20)],
            "is_closed": False,
        },
        {
            "id": 2,
            "source": 0,
            "target": 2,
            "pixels": [[x, 25] for x in range(2, 20)],
            "is_closed": False,
        },
        {
            "id": 3,
            "source": 4,
            "target": 5,
            "pixels": curved_pixels,
            "is_closed": False,
        },
    ]
    hatch_mask = np.ones((64, 64), np.uint8)
    config = {
        "hachure_cnn_inside_frac": 0.60,
        "hachure_cnn_min_length": 1.0,
        "hachure_cnn_max_removed_edge_ratio": 1.0,
        "hachure_cnn_protect_structural_seed_edges": True,
        "hachure_cnn_main_require_line_like": True,
    }

    _nodes, kept, removed = s2._remove_hachures_cnn(
        nodes, edges, hatch_mask, config, pass_name="cnn"
    )

    assert [edge["id"] for edge in kept] == [3]
    assert [edge["id"] for edge in removed] == [0, 1, 2]


def test_multilabel_hatch_prepass_preserves_crossing_structure():
    skeleton = np.zeros((65, 65), dtype=np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    probabilities = np.zeros((2, 65, 65), dtype=np.float32)
    probabilities[0, 32, 8:57] = 0.99
    probabilities[0, 8:57, 32] = 0.01
    probabilities[0, 32, 32] = 0.99
    probabilities[1, 8:57, 32] = 0.99
    probabilities[1, 32, 8:57] = 0.99

    cleaned, side_edges, stats = s2._hatch_stroke_prepass_from_probabilities(
        skeleton,
        probabilities,
        {
            "hachure_stroke_hatch_high_threshold": 0.8,
            "hachure_stroke_structural_low_threshold": 0.2,
            "hachure_stroke_max_removed_pixel_ratio": 0.75,
            "simplify_graph": True,
        },
        max_search_radius=60,
    )

    assert stats["applied"]
    assert stats["safe_removed_pixels"] == 48
    assert stats["side_layer_pixels"] <= 51
    assert np.all(cleaned[32, 8:57] > 0)
    assert not np.any(cleaned[8:32, 32])
    assert not np.any(cleaned[33:57, 32])
    assert side_edges
    assert all(edge["is_hachure"] for edge in side_edges)


def test_multilabel_hatch_prepass_guard_rejects_destructive_mask():
    skeleton = np.zeros((65, 65), dtype=np.uint8)
    skeleton[32, 8:57] = 255
    probabilities = np.zeros((2, 65, 65), dtype=np.float32)
    probabilities[1, 32, 8:57] = 0.99

    cleaned, side_edges, stats = s2._hatch_stroke_prepass_from_probabilities(
        skeleton,
        probabilities,
        {"hachure_stroke_max_removed_pixel_ratio": 0.25},
        max_search_radius=60,
    )

    assert not stats["applied"]
    assert stats["guard"] == "maximum_removed_pixel_ratio"
    assert np.array_equal(cleaned, skeleton)
    assert side_edges == []


def test_multilabel_hatch_postmask_is_non_destructive_and_structural_vetoed():
    skeleton = np.zeros((65, 65), dtype=np.uint8)
    skeleton[32, 8:57] = 255
    skeleton[8:57, 32] = 255
    original = skeleton.copy()
    probabilities = np.zeros((2, 65, 65), dtype=np.float32)
    probabilities[0, 32, 8:57] = 0.99
    probabilities[0, 8:57, 32] = 0.01
    probabilities[0, 32, 32] = 0.99
    probabilities[1, 8:57, 32] = 0.99
    probabilities[1, 32, 8:57] = 0.99

    edge_mask, stats = s2._hatch_stroke_postmask_from_probabilities(
        skeleton,
        probabilities,
        {
            "hachure_stroke_hatch_high_threshold": 0.8,
            "hachure_stroke_structural_low_threshold": 0.2,
            "hachure_stroke_max_removed_pixel_ratio": 0.75,
        },
    )

    assert stats["applied"]
    assert stats["mode"] == "post_topology"
    assert stats["safe_candidate_pixels"] == 48
    assert stats["safe_removed_pixels"] == 0
    assert not stats["pixel_subtraction_applied"]
    assert np.array_equal(skeleton, original)
    assert edge_mask is not None
    assert not np.any(edge_mask[32, 8:57])
    assert np.count_nonzero(edge_mask[8:57, 32]) == 48


def test_hatch_region_residue_is_preserved_as_side_layer():
    diagonal = [[value, value] for value in range(10, 51)]
    horizontal = [[value, 32] for value in range(10, 51)]
    nodes = [
        {"id": 0, "x": 10, "y": 10, "type": "endpoint"},
        {"id": 1, "x": 50, "y": 50, "type": "endpoint"},
        {"id": 2, "x": 10, "y": 32, "type": "endpoint"},
        {"id": 3, "x": 50, "y": 32, "type": "endpoint"},
    ]
    edges = [
        {
            "id": 0,
            "source": 0,
            "target": 1,
            "pixels": diagonal,
            "is_closed": False,
        },
        {
            "id": 1,
            "source": 2,
            "target": 3,
            "pixels": horizontal,
            "is_closed": False,
        },
    ]
    regions = [{
        "boundary": [[5, 5], [58, 5], [58, 58], [5, 58]],
        "angles": [45.0],
    }]

    kept_nodes, kept, removed = s2._cleanup_hatch_residue(
        nodes,
        edges,
        regions,
        (64, 64),
        {"hachure_cleanup_angle_tol": 5.0},
    )

    assert [edge["id"] for edge in kept] == [1]
    assert {node["id"] for node in kept_nodes} == {2, 3}
    assert [edge["id"] for edge in removed] == [0]
    assert removed[0]["is_hachure"]
    assert removed[0]["hachure"]["pass"] == "region_residue"


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
