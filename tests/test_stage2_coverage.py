from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from tools.batch_run import stage2_stroke_extract as s2, stage3_primitive_fit as s3
from tools import geometry_validation as geometry


def mask(points, shape=(80, 100)):
    image = np.zeros(shape, np.uint8)
    for x, y in points:
        image[y, x] = 255
    return image


def graph(points):
    nodes = [{"id": i, "x": p[0], "y": p[1], "type": "endpoint"}
             for i, p in enumerate((points[0], points[-1]))]
    return nodes, [{"id": 4, "source": 0, "target": 1, "pixels": points,
                    "smooth_pts": [], "is_closed": False}]


def pixels(edges):
    return {tuple(p) for edge in edges for p in edge["pixels"]}


def covered_doc(image, nodes, edges, hatches=()):
    ledger = s2._CoverageLedger(image)
    ledger.record("input_graph", edges, hatches)
    nodes, edges, recovery = s2.recover_source_coverage(image, nodes, edges, hatches)
    ledger.record("final_source_recovery", edges, hatches)
    return {"image_shape": list(image.shape), "stage2_scale": 1.0,
            "nodes": nodes, "edges": edges, "removed_hachures": list(hatches),
            "coverage": ledger.report(recovery)}


def test_ledger_identifies_the_operation_and_exact_lost_coordinates():
    points = [[x, 20] for x in range(10, 31)]
    image = mask(points)
    nodes, edges = graph(points)
    ledger = s2._CoverageLedger(image)
    ledger.record("tracing", edges)
    shortened = deepcopy(edges)
    shortened[0]["pixels"] = shortened[0]["pixels"][:-5]
    ledger.record("pruning", shortened)
    assert ledger.stages[0]["newly_missing_pixels"] == 0
    assert ledger.stages[1]["newly_missing_spans"] == [[20, 26, 31]]
    nodes, repaired, recovery = s2.recover_source_coverage(image, nodes, shortened)
    ledger.record("recovery", repaired)
    report = ledger.report(recovery)
    assert report["represented_fraction"] == report["accounted_fraction"] == 1.0
    assert pixels(repaired) == set(map(tuple, points))


@pytest.mark.parametrize("points", [
    [[10, 10], [11, 10]],
    [[x, 20] for x in range(10, 31)],
    [[x, 10] for x in range(10, 31)]+[[30, y] for y in range(11, 26)]
    + [[x, 25] for x in range(29, 9, -1)]+[[10, y] for y in range(24, 10, -1)],
    [[x, 20] for x in range(10, 31)]+[[20, y] for y in range(10, 31)],
])
def test_source_recovery_preserves_shapes_without_off_source_links(points):
    image = mask(points)
    before = image.copy()
    nodes, edges, report = s2.recover_source_coverage(image, [], [])
    assert np.array_equal(image, before)
    assert pixels(edges) == set(map(tuple, points))
    assert report["unresolved_pixels"] == 0
    for edge in edges:
        assert np.linalg.norm(np.diff(edge["pixels"], axis=0), axis=1).max() <= 2**0.5
        assert geometry.check_primitive(s3.fit_edge_ransac(edge), edge)["status"] == "pass"
    nodes2, edges2, second = s2.recover_source_coverage(image, nodes, edges)
    assert nodes2 == nodes and edges2 == edges
    assert second["new_edges"] == 0


def test_two_disconnected_strokes_are_never_bridged():
    points = [[x, 10] for x in list(range(10, 21))+list(range(50, 61))]
    _, edges, _ = s2.recover_source_coverage(mask(points), [], [])
    assert len(edges) == 2
    assert all(max(p[0] for p in e["pixels"])-min(p[0] for p in e["pixels"]) == 10 for e in edges)


def test_source_halo_recovers_one_missing_pixel_and_reuses_endpoint_nodes():
    points = [[x, 20] for x in range(10, 31)]
    nodes, left = graph(points[:10])
    n2, right = graph(points[11:])
    for n in n2:
        n["id"] += 2
    right[0].update(id=5, source=2, target=3)
    original = deepcopy(nodes+ n2), deepcopy(left+right)
    _, edges, report = s2.recover_source_coverage(mask(points), nodes+n2, left+right)
    assert report["recovered_pixels"] == 1
    assert edges[-1]["pixels"] == [[19, 20], [20, 20], [21, 20]]
    assert (edges[-1]["source"], edges[-1]["target"]) == (1, 2)
    assert (nodes+n2, left+right) == original


def test_hatch_ownership_is_not_recovered_again_as_main_geometry():
    points = [[x, 20] for x in range(10, 31)]
    _, hatches = graph(points)
    original = deepcopy(hatches)
    _, edges, report = s2.recover_source_coverage(mask(points), [], [], hatches)
    assert edges == [] and report["new_edges"] == 0
    assert hatches == original


@pytest.mark.parametrize("limits", [{"max_component_pixels": 5}, {"max_new_edges": 1}])
def test_recovery_budgets_retain_unresolved_evidence(limits):
    points = [[x, 20] for x in range(10, 31)]+[[20, y] for y in range(10, 31)]
    _, edges, report = s2.recover_source_coverage(mask(points), [], [], **limits)
    assert edges == [] and report["unresolved_pixels"] == len(set(map(tuple, points)))
    assert report["unresolved_spans"] and report["components"][0]["status"] == "review"


def test_singleton_is_accounted_but_never_claimed_as_rendered_or_noise():
    points = [[x, 20] for x in range(10, 31)]
    nodes, edges = graph(points)
    image = mask(points+[[50, 50]])
    g = covered_doc(image, nodes, edges)
    assert g["coverage"]["accounted_fraction"] == 1.0
    assert g["coverage"]["represented_fraction"] < 1.0
    assert g["coverage"]["recovery"]["unresolved_spans"] == [[50, 50, 51]]
    doc = {"image_size": [100, 80], "primitives": [s3.fit_edge_ransac(e) for e in g["edges"]]}
    report = geometry.validate(g, doc, image > 0)
    assert "geometry_stage2_residual_pending" in report["reason_codes"]
    assert "geometry_skeleton_coverage_lost" in report["reason_codes"]


@pytest.mark.parametrize("damage", ["hide_residual", "wrong_hash", "counters", "duplicate_span"])
def test_coverage_accounting_cannot_override_independent_source_evidence(damage):
    points = [[x, 20] for x in range(10, 31)]
    image = mask(points+[[50, 50]])
    g = covered_doc(image, *graph(points))
    c = g["coverage"]
    if damage == "hide_residual":
        c["recovery"]["unresolved_spans"] = []
    elif damage == "wrong_hash":
        c["source_mask_sha256"] = "0"*64
    elif damage == "counters":
        c["residual_source_pixels"] = 0
    else:
        c["recovery"]["unresolved_spans"] *= 2
    doc = {"image_size": [100, 80], "primitives": [s3.fit_edge_ransac(e) for e in g["edges"]]}
    report = geometry.validate(g, doc, image > 0)
    assert report["checks"]["stage2_coverage"]["status"] == "error"


def test_full_stage2_preserves_small_non_circular_loops(tmp_path):
    image = np.zeros((80, 100), np.uint8)
    cv2.rectangle(image, (20, 20), (40, 30), 255, 1)
    path = tmp_path / "skeleton.png"
    cv2.imwrite(str(path), image)
    result = s2.run(path, tmp_path, "small_loop", {"stage2": {"min_closed_loop_pixels": 80}})
    g = json.loads(result.graph_path.read_text())
    assert g["coverage"]["small_closed_shapes_preserved"] > 0
    assert g["coverage"]["represented_fraction"] == 1.0
    assert len(g["edges"]) > 0


def test_small_unclaimed_open_component_is_not_skipped():
    image = mask([[x, 10] for x in range(10, 14)])
    _, edges = s2._extract_topology(image, [])
    assert pixels(edges) == {(x, 10) for x in range(10, 14)}


@pytest.mark.parametrize("reverse", [False, True])
def test_hatch_deduplication_never_discards_unique_source_ink(reverse):
    edges = [{"id": 1, "pixels": [[x, 10] for x in range(10, 30)]},
             {"id": 2, "pixels": [[x, 10] for x in range(12, 32)]}]
    if reverse:
        edges.reverse()
    assert pixels(s2._deduplicate_hachure_edges(edges, 0.75)) == pixels(edges)


def test_working_grid_accounting_does_not_claim_original_resolution_validation():
    points = [[x, 20] for x in range(10, 31)]
    image = mask(points)
    g = covered_doc(image, *graph(points))
    g["stage2_scale"] = g["coverage"]["stage2_scale"] = 0.5
    result = geometry._stage2_coverage_check(g, image > 0, 0.5)
    assert result["status"] == "review"


def test_node_proximity_alone_is_not_rendered_coverage():
    image = mask([[30, 30]])
    nodes = [{"id": 0, "x": 30, "y": 30, "type": "endpoint"}]
    doc = covered_doc(image, nodes, [])
    assert doc["coverage"]["residual_source_pixels"] == 1
    assert doc["coverage"]["represented_source_pixels"] == 0
