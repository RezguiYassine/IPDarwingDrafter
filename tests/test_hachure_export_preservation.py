from copy import deepcopy
import json

import pytest

from tools.batch_run import stage3_primitive_fit as s3


def _graph(*, legacy=False):
    region = {
        "boundary": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "angles": [0], "spacing": 2, "n_lines": 1,
    }
    if not legacy:
        region["source_hachure_indices"] = [0]
    return {
        "image_shape": [64, 64],
        "edges": [],
        "hachure_regions": [region],
        "removed_hachures": [
            {"id": 10, "pixels": [[2, 4], [4, 4], [8, 4]]},
            {"id": 10, "pixels": [[20, 20], [25, 20], [30, 20]]},
        ],
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_region_does_not_swallow_outside_hatch_edges(legacy):
    graph = _graph(legacy=legacy)
    original = deepcopy(graph)
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert graph == original
    assert [p["type"] for p in primitives] == ["hatch", "line"]
    assert [p["source_hachure_indices"] for p in primitives] == [[0], [1]]
    assert coverage["source_edges"] == coverage["represented_edges"] == 2
    assert coverage["region_edges"] == coverage["fallback_edges"] == 1
    assert coverage["unrepresented_indices"] == []


def test_membership_is_not_trusted_when_an_edge_extends_outside_region():
    graph = _graph()
    graph["hachure_regions"][0]["source_hachure_indices"] = [0, 1]
    graph["removed_hachures"][1]["pixels"] = [[x, 5] for x in range(5, 15)]
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert primitives[0]["source_hachure_indices"] == [0]
    assert coverage["fallback_edges"] == 1
    assert primitives[1]["p2"] == pytest.approx([14, 5])


def test_declared_region_does_not_claim_unlisted_edges():
    graph = _graph()
    graph["removed_hachures"][1]["pixels"] = [[2, 6], [4, 6], [8, 6]]
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert primitives[0]["source_hachure_indices"] == [0]
    assert coverage["fallback_edges"] == 1


def test_curved_uncovered_hatch_keeps_its_trace_instead_of_a_short_line():
    graph = _graph()
    pixels = [[20, 20], [25, 20], [30, 20], [30, 25], [30, 30]]
    graph["removed_hachures"][1]["pixels"] = pixels
    primitives, _ = s3.fit_hachure_layer(graph, coord_scale=2)
    assert primitives[1]["type"] == "polyline"
    assert primitives[1]["points"] == [[2*x, 2*y] for x, y in pixels]
    assert primitives[0]["spacing"] == 4


def test_invalid_region_cannot_claim_edges_or_prevent_fallback():
    graph = _graph()
    graph["hachure_regions"][0]["spacing"] = 0
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert all(p["type"] != "hatch" for p in primitives)
    assert coverage["fallback_edges"] == 2


def test_one_edge_is_owned_only_once_when_regions_overlap():
    graph = _graph(legacy=True)
    graph["hachure_regions"].append(deepcopy(graph["hachure_regions"][0]))
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert [p["source_hachure_indices"] for p in primitives] == [[0], [], [1]]
    assert coverage["represented_edges"] == 2


def test_unrepresentable_hatch_is_reported_and_flags_stage3(tmp_path):
    graph = _graph()
    graph["edges"] = [{"id": 1, "pixels": [[20, 40], [30, 40]], "is_closed": False}]
    graph["removed_hachures"][1]["pixels"] = []
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    result = s3.run(path, tmp_path, "coverage", config={})
    doc = json.loads(result.primitives_path.read_text())
    assert result.flagged
    assert doc["quality_metrics"]["hachure_coverage"]["unrepresented_indices"] == [1]
