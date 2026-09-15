from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.batch_run import stage2_stroke_extract as s2, stage3_primitive_fit as s3
from tools.geometry_validation import check_primitive


def _pixels(edges):
    return {tuple(point) for edge in edges for point in edge["pixels"]}


def _legacy(pixels):
    return ([{"id": 40, "x": 0, "y": 0, "type": s2.KP_LOOP_ANCHOR}],
            [{"id": 70, "source": 40, "target": 40, "pixels": pixels,
              "is_closed": True, "is_simple_cycle": False,
              "topology_origin": "unclaimed_component", "smooth_pts": []}])


@pytest.mark.parametrize("shape", ["arc", "cross", "tee", "staircase", "loop_and_branch", "grid"])
def test_repair_preserves_pixels_and_each_local_adjacency_once(shape):
    image = np.zeros((96, 96), np.uint8)
    if shape == "arc":
        cv2.ellipse(image, (45, 45), (30, 30), 0, 10, 110, 255, 1)
    elif shape == "staircase":
        for i in range(10, 50):
            image[i, i:i+2] = 255
    elif shape == "loop_and_branch":
        cv2.rectangle(image, (20, 20), (60, 60), 255, 1)
        image[60:81, 40] = 255
    elif shape == "grid":
        image[10:81:10, 10:81] = 255
        image[10:81, 10:81:10] = 255
    else:
        image[40, 10:81] = 255
        image[10 if shape == "cross" else 40:81, 40] = 255
    y, x = np.nonzero(image)
    nodes, edges = _legacy(np.column_stack([x, y]).tolist())
    before = deepcopy((nodes, edges))
    repaired_nodes, repaired = s2.repair_noncycle_residuals(nodes, edges)
    assert (nodes, edges) == before
    assert _pixels(repaired) == _pixels(edges)
    adjacency, _ = s2._residual_chains(edges[0]["pixels"])
    links = Counter(tuple(sorted((tuple(a), tuple(b)))) for edge in repaired
                    for a, b in zip(edge["pixels"], edge["pixels"][1:]))
    assert set(links) == {tuple(sorted((a, b))) for a, near in adjacency.items() for b in near}
    assert all(count == 1 for count in links.values())
    assert len({edge["id"] for edge in repaired}) == len(repaired)
    ids = {node["id"] for node in repaired_nodes}
    for edge in repaired:
        assert edge["source"] in ids and edge["target"] in ids
        assert edge["is_closed"] == (edge["pixels"][0] == edge["pixels"][-1])
    assert s2.repair_noncycle_residuals(repaired_nodes, repaired) == (repaired_nodes, repaired)


def test_simplification_keeps_short_recovered_branch_and_long_continuation():
    points = [[x, 20] for x in range(10, 91)] + [[50, 21], [50, 22], [50, 23]]
    nodes, edges = s2.repair_noncycle_residuals(*_legacy(points))
    nodes, edges = s2._simplify_graph(nodes, edges, spur_min_len=10, junction_merge_radius=0)
    assert _pixels(edges) == {tuple(p) for p in points}
    assert len(edges) == 2
    assert max(len(edge["pixels"]) for edge in edges) == 81
    assert all(edge["residual_parent_edge_ids"] == [70] for edge in edges)


def test_untouched_graph_is_an_exact_noop():
    nodes, edges = _legacy([[10, 10], [11, 10]])
    edges[0].pop("topology_origin")
    assert s2.repair_noncycle_residuals(nodes, edges) == (nodes, edges)


def test_confirmed_patent_false_circles_become_supported_arcs():
    fixture = json.loads((Path(__file__).parent / "fixtures/unsupported_patent_circles.json").read_text())
    edges = fixture["graph"]["edges"]
    nodes = [{"id": edge["source"], "x": 0, "y": 0, "type": s2.KP_LOOP_ANCHOR} for edge in edges]
    _, repaired = s2.repair_noncycle_residuals(nodes, edges)
    assert _pixels(repaired) == _pixels(edges)
    assert len(repaired) == 2
    for edge in repaired:
        primitive = s3.fit_edge_ransac(edge)
        assert primitive["type"] == "arc"
        assert check_primitive(primitive, edge)["status"] == "pass"


@pytest.mark.parametrize("kind", ["circle", "ellipse"])
@pytest.mark.parametrize("closed", [False, True])
def test_partial_curve_cannot_promote_a_full_circle_or_ellipse(kind, closed, monkeypatch):
    t = np.radians(np.linspace(15, 45, 201))
    a, b = 80, 80 if kind == "circle" else 35
    points = np.column_stack([200+a*np.cos(t), 200+b*np.sin(t)])
    edge = {"id": 5, "pixels": points.tolist(), "smooth_pts": [], "is_closed": closed}
    candidate = {"type": kind, "center": [200, 200], "confidence": 0.999}
    candidate.update({"radius": a} if kind == "circle" else {"a": a, "b": b, "angle": 0})
    if kind == "ellipse":
        monkeypatch.setattr(s3, "_fit_circle_ransac", lambda pts: {"type": "circle", "center": [0, 0], "radius": 1, "confidence": 0})
    monkeypatch.setattr(s3, "_fit_" + kind + "_ransac", lambda pts: dict(candidate))
    result = s3.fit_edge_ransac(edge, prefer_compound_over_weak=True, simplify_closed_fallback=True)
    assert result["type"] not in {"circle", "ellipse"}


@pytest.mark.parametrize("kind", ["circle", "ellipse"])
@pytest.mark.parametrize("angle", [0, 35, 90])
def test_full_supported_curves_keep_their_analytic_representation(kind, angle):
    t = np.linspace(0, 2*np.pi, 801)
    a, b = 60, 60 if kind == "circle" else 30
    points = np.column_stack([a*np.cos(t), b*np.sin(t)])
    r = np.radians(angle)
    points = points @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]]) + [200, 200]
    edge = {"id": 5, "pixels": points.tolist(), "smooth_pts": [], "is_closed": True}
    result = s3.fit_edge_ransac(edge)
    assert result["type"] == kind
    assert check_primitive(result, edge)["status"] == "pass"


def test_unrepaired_legacy_network_never_enters_closed_shape_cascade():
    points = [[x, 20] for x in range(10, 51)] + [[30, y] for y in range(21, 41)]
    _, edges = _legacy(points)
    result = s3.fit_edge_ransac(edges[0])
    assert result["type"] == "polyline"
    assert {tuple(p) for p in result["points"]} == {tuple(p) for p in points}
    assert result["fit_metadata"]["strategy"] == "unrepaired_noncycle_trace"


def test_recovered_long_line_survives_truncated_smoothing():
    edge = {"id": 7, "pixels": [[x, 20] for x in range(10, 151)],
            "smooth_pts": [[148, 20], [149, 20], [150, 20]],
            "is_closed": False, "residual_parent_edge_ids": [70]}
    result = s3.fit_edge_ransac(edge)
    assert result["type"] == "line"
    assert check_primitive(result, edge)["status"] == "pass"
    assert np.linalg.norm(np.array(result["p1"])-result["p2"]) == pytest.approx(140)
    assert result["fit_metadata"]["strategy"] == "source_supported_residual_refit"


def test_recovered_bent_chain_uses_supported_trace_when_smoothing_cuts_corner():
    points = [[x, 20] for x in range(10, 61)] + [[60, y] for y in range(21, 71)]
    edge = {"id": 7, "pixels": points, "smooth_pts": [[10, 20], [60, 70]],
            "is_closed": False, "residual_parent_edge_ids": [70]}
    result = s3.fit_edge_ransac(edge)
    assert result["type"] == "path"
    assert check_primitive(result, edge)["status"] == "pass"
    assert result["fit_metadata"]["strategy"] == "source_supported_residual_anchored_path"


def test_recovered_arc_refits_raw_source_instead_of_shifted_spline():
    t = np.radians(np.linspace(20, 130, 121))
    points = np.column_stack([150+70*np.cos(t), 150+70*np.sin(t)])
    edge = {"id": 7, "pixels": points.tolist(), "smooth_pts": (points+[5, 5]).tolist(),
            "is_closed": False, "residual_parent_edge_ids": [70]}
    result = s3.fit_edge_ransac(edge)
    assert result["type"] == "arc"
    assert result["center"] == pytest.approx([150, 150])
    assert check_primitive(result, edge)["status"] == "pass"


def test_source_guard_also_repairs_unsupported_nonresidual_fits():
    edge = {"id": 7, "pixels": [[x, x] for x in range(10, 21)], "is_closed": False}
    candidate = {"type": "line", "p1": [500, 500], "p2": [510, 510], "confidence": 0.5}
    result = s3._guard_source_primitive(edge, candidate)
    assert result is not candidate
    assert check_primitive(result, edge)["status"] == "pass"


@pytest.mark.parametrize("legacy", [False, True])
def test_recovered_hatch_region_preserves_explicit_strokes_without_hull_fill(legacy):
    edges = [{"id": 4, "pixels": [[x, y] for x in range(10, 51)],
              "residual_parent_edge_ids": [70]} for y in (10, 20)]
    region = {"boundary": [[0, 0], [100, 0], [100, 100], [0, 100]],
              "angles": [0], "spacing": 10, "source_hachure_indices": [0, 1]}
    if legacy:
        region.pop("source_hachure_indices")
    graph = {"removed_hachures": edges, "hachure_regions": [region]}
    before = deepcopy(graph)
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert graph == before
    assert [p["type"] for p in primitives] == ["line", "line"]
    assert [p["source_hachure_indices"] for p in primitives] == [[0], [1]]
    assert coverage["region_edges"] == 0
    assert coverage["represented_edges"] == coverage["source_edges"] == 2
    assert coverage["recovered_regions_traced"] == [0]
    for primitive, edge in zip(primitives, edges):
        assert check_primitive(primitive, edge)["status"] == "pass"


def test_recovered_hatch_without_region_preserves_bent_source():
    edge = {"id": 4, "pixels": [[10, 10], [20, 10], [30, 10], [30, 20], [30, 30]],
            "residual_parent_edge_ids": [70]}
    primitives, coverage = s3.fit_hachure_layer({"removed_hachures": [edge]})
    assert primitives[0]["type"] == "polyline"
    assert primitives[0]["points"] == edge["pixels"]
    assert coverage["represented_edges"] == 1
