from copy import deepcopy

import numpy as np
import pytest

from tools.batch_run import stage2_stroke_extract as s2, stage3_primitive_fit as s3
from tools import geometry_validation as geometry


def edge(ident, points, *, recovered=False, **kwargs):
    return dict(id=ident, source=2*ident, target=2*ident+1, pixels=points,
                smooth_pts=deepcopy(points), is_closed=False,
                topology_origin="coverage_recovery" if recovered else "traced", **kwargs)


def integrate(edges, hatches=(), extra=()):
    nodes = [dict(id=e[key], x=p[0], y=p[1], type="endpoint") for e in edges
             for key, p in (("source", e["pixels"][0]), ("target", e["pixels"][-1]))]
    skeleton = np.zeros((100, 100), np.uint8)
    for p in list(extra) + [p for e in edges+list(hatches) for p in e["pixels"]]:
        skeleton[p[1], p[0]] = 255
    before = deepcopy((nodes, edges, hatches)), skeleton.copy()
    result = s2.integrate_recovered_connections(skeleton, nodes, edges, hatches)
    assert (nodes, edges, hatches) == before[0]
    assert np.array_equal(skeleton, before[1])
    n, out, report = result
    assert {tuple(p) for e in out for p in e["pixels"]} == {tuple(p) for e in edges for p in e["pixels"]}
    by_id = {e["id"]: e for e in out}
    for e in edges:
        owned = {tuple(p) for ident in report["input_to_output_edge_ids"][str(e["id"]) ]
                 for p in by_id[ident]["pixels"]}
        assert set(map(tuple, e["pixels"])) <= owned
    n2, out2, _ = s2.integrate_recovered_connections(skeleton, n, out, hatches)
    assert n2 == n and out2 == out
    return result


@pytest.mark.parametrize("reverse_left,reverse_right", [(False, False), (False, True), (True, False), (True, True)])
def test_splices_connector_between_endpoints_with_orientation_and_ownership(reverse_left, reverse_right):
    left = [[x, 20] for x in range(10, 20)]
    right = [[x, 20] for x in range(21, 31)]
    edges = [edge(2, left[::-1] if reverse_left else left),
             edge(5, right[::-1] if reverse_right else right),
             edge(9, [[19, 20], [20, 20], [21, 20]], recovered=True)]
    nodes, out, report = integrate(edges)
    assert len(out) == 1 and report["endpoint_joins"] == 2
    assert out[0]["pixels"] == [[x, 20] for x in range(10, 31)]
    assert out[0]["coverage_parent_edge_ids"] == [2, 5, 9]
    assert out[0]["smooth_pts"] == []
    assert geometry.check_primitive(s3.fit_edge_ransac(out[0]), out[0])["status"] == "pass"
    by_id = {n["id"]: n for n in nodes}
    for key, p in (("source", out[0]["pixels"][0]), ("target", out[0]["pixels"][-1])):
        assert (by_id[out[0][key]]["x"], by_id[out[0][key]]["y"]) == tuple(p)


def test_missing_interior_pixel_is_inserted_without_fragmenting_old_stroke():
    original = [[x, 20] for x in range(10, 31) if x != 20]
    _, out, report = integrate([edge(0, original), edge(1, [[19, 20], [20, 20]], recovered=True),
                                edge(2, [[20, 20], [21, 20]], recovered=True)])
    assert len(out) == 1
    assert out[0]["pixels"] == [[x, 20] for x in range(10, 31)]
    assert report["inserted_pixel_occurrences"] == 1 and report["retired_recovery_edges"] == 2


def test_diagonal_insertion_keeps_a_real_side_branch():
    _, out, report = integrate([edge(0, [[19, 20], [20, 19], [21, 19]]),
                                edge(1, [[19, 20], [20, 20]], recovered=True),
                                edge(2, [[20, 20], [20, 19]], recovered=True),
                                edge(3, [[20, 20], [21, 20]], recovered=True)])
    assert report["inserted_pixel_occurrences"] == 1
    assert len(out) == 2
    assert out[0]["pixels"] == [[19, 20], [20, 20], [20, 19], [21, 19]]
    assert out[1]["pixels"] == [[20, 20], [21, 20]]


def test_insertion_repairs_overlapping_originals_with_complete_parent_ownership():
    old = [[x, 20] for x in range(10, 31) if x != 20]
    _, out, report = integrate([edge(0, old), edge(1, old[::-1]),
                                edge(4, [[19, 20], [20, 20], [21, 20]], recovered=True)])
    assert len(out) == 2 and report["inserted_pixel_occurrences"] == 2
    assert report["input_to_output_edge_ids"]["4"] == [0, 1]


def test_partial_absorption_preserves_the_uncovered_recovery_suffix():
    _, out, report = integrate([edge(0, [[18, 20], [19, 20], [21, 20], [22, 20]]),
                                edge(2, [[19, 20], [20, 20], [21, 20], [21, 21], [21, 22]], recovered=True)])
    assert len(out) == 2
    assert out[1]["pixels"] == [[21, 20], [21, 21], [21, 22]]
    assert report["input_to_output_edge_ids"]["2"] == [0, 2]


@pytest.mark.parametrize("blocker", ["branch", "hatch", "dashed", "gap", "interior"])
def test_ambiguous_or_unsupported_joins_remain_separate(blocker):
    edges = [edge(0, [[x, 20] for x in range(10, 21)]),
             edge(1, [[20, 20], [21, 20], [22, 20]], recovered=True)]
    hatches, extra = [], []
    if blocker == "branch":
        extra = [[20, 21], [20, 22]]
    elif blocker == "hatch":
        hatches = [edge(5, [[20, 20], [20, 21]])]
    elif blocker == "dashed":
        edges[0]["is_dashed"] = True
    elif blocker == "gap":
        edges[1]["pixels"] = [[20, 20], [24, 20], [25, 20]]
    else:
        edges[0]["pixels"].extend([[21, 20], [22, 20], [23, 20]])
        edges[1]["pixels"] = [[20, 20], [20, 21], [20, 22]]
    _, out, report = integrate(edges, hatches, extra)
    assert len(out) == 2 and report["endpoint_joins"] == 0


def test_two_possible_interior_routes_are_not_guessed():
    _, out, report = integrate([edge(0, [[20, 20], [21, 21]]),
                                edge(1, [[20, 20], [20, 21], [21, 21]], recovered=True),
                                edge(2, [[20, 20], [21, 20], [21, 21]], recovered=True)])
    assert out[0]["pixels"] == [[20, 20], [21, 21]]
    assert report["inserted_pixel_occurrences"] == 0


def test_corner_tail_is_not_straightened():
    _, out, _ = integrate([edge(0, [[x, 20] for x in range(10, 21)]),
                            edge(1, [[20, y] for y in range(20, 26)], recovered=True)])
    assert len(out) == 1
    assert out[0]["pixels"] == [[x, 20] for x in range(10, 21)]+[[20, y] for y in range(21, 26)]


def test_unrelated_strokes_smoothing_and_closed_loops_stay_unchanged():
    edges = [edge(0, [[10, 20], [11, 20]]), edge(1, [[11, 20], [12, 20]]),
             edge(3, [[40, 40], [41, 40], [41, 41], [40, 41], [40, 40]], recovered=True)]
    edges[-1]["is_closed"] = True
    _, out, _ = integrate(edges)
    assert out == edges


def test_closed_recovery_reconnection_retains_every_corner():
    ring = [[x, 20] for x in range(20, 31)]+[[30, y] for y in range(21, 31)]
    ring += [[x, 30] for x in range(29, 19, -1)]+[[20, y] for y in range(29, 19, -1)]
    _, out, _ = integrate([edge(0, ring[:20]), edge(1, ring[19:], recovered=True)])
    assert len(out) == 1 and out[0]["is_closed"] and out[0]["is_simple_cycle"]


def test_full_stage2_ledgers_record_integration_without_new_missing_pixels(tmp_path):
    import cv2
    import json
    skeleton = np.zeros((100, 100), np.uint8)
    cv2.rectangle(skeleton, (20, 20), (60, 50), 255, 1)
    path = tmp_path / "skeleton.png"
    cv2.imwrite(str(path), skeleton)
    result = s2.run(path, tmp_path, "loop", {})
    g = json.loads(result.graph_path.read_text())
    assert g["coverage"]["represented_fraction"] == 1.0
    assert g["coverage"]["stages"][-1]["operation"] == "recovered_connection_integration"
    assert g["coverage"]["stages"][-1]["newly_missing_pixels"] == 0
    assert g["coverage"]["recovery"]["integration"]["main_pixel_set_preserved"]


@pytest.mark.parametrize("endpoint_shift,detail", [(0.5, False), (0, True)])
def test_integrated_fitting_cannot_reopen_contacts_or_erase_recovered_detail(endpoint_shift, detail):
    points = [[x, 20 + (2 if detail and 18 <= x <= 22 else 0)] for x in range(10, 31)]
    e = edge(0, points)
    e["topology_origin"] = "coverage_integration"
    candidate = {"type": "line", "edge_id": 0, "p1": [10+endpoint_shift, 20],
                 "p2": [30+endpoint_shift, 20], "confidence": 0.9}
    assert geometry.check_primitive(candidate, e)["status"] == "pass"
    result = s3._guard_source_primitive(e, candidate)
    check = geometry.check_primitive(result, e)
    assert check["status"] == "pass"
    assert check["metrics"]["endpoint_error"] < 1e-6
    assert check["metrics"]["source_to_model"]["max"] <= 1.0
    assert result != candidate
