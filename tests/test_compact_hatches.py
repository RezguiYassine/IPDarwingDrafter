from copy import deepcopy
import json
import xml.etree.ElementTree as ET

import ezdxf
import numpy as np
import pytest

from tools.batch_run import stage3_primitive_fit as s3, stage4_export as s4
from tools import geometry_validation as geometry


def example():
    edges = [{"id": 10, "pixels": [[x, y] for x in range(10, 81)], "is_closed": False}
             for y in (20, 30, 40, 50)]
    graph = {"image_shape": [100, 100], "edges": [], "removed_hachures": edges}
    primitives, _ = s3.fit_hachure_layer(graph)
    return graph, {"sketch_id": "hatches", "image_size": [100, 100], "primitives": primitives}


def test_hatch_bundle_preserves_each_stroke_and_budget_without_bridges():
    graph, doc = example()
    assert len(doc["primitives"]) == 1
    primitive = doc["primitives"][0]
    assert primitive["type"] == "hatch_strokes"
    assert s4.primitive_budget_cost(primitive) == 4
    points, gap = geometry.sample_primitive(primitive)
    assert gap == 0
    assert set(points[:, 1]) == {20, 30, 40, 50}
    report = geometry.validate(graph, doc)
    assert report["checks"]["ownership"]["status"] == "pass"
    assert report["primitives"][0]["status"] == "pass"
    assert len(report["primitives"][0]["stroke_checks"]) == 4


def test_hatch_bundle_cannot_hide_wrong_per_stroke_ownership_with_union_support():
    graph, doc = example()
    p = doc["primitives"][0]
    p["stroke_source_indices"][0], p["stroke_source_indices"][1] = p["stroke_source_indices"][1], p["stroke_source_indices"][0]
    assert geometry.validate(graph, doc)["primitives"][0]["status"] == "fail"


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nan", "zero_length"])
def test_bad_bundle_is_rejected(damage):
    graph, doc = example()
    p = doc["primitives"][0]
    if damage == "missing":
        p["stroke_source_indices"].pop()
    elif damage == "duplicate":
        p["stroke_source_indices"][1] = [0]
    elif damage == "nan":
        p["strokes"][0][0][0] = float("nan")
    else:
        p["strokes"][0] = [[10, 20], [10, 20]]
    with pytest.raises(ValueError):
        s4.validate_primitive(p)
    assert geometry.validate(graph, doc)["status"] == "error"


def test_bundle_scaling_and_no_input_mutation():
    graph, doc = example()
    before = deepcopy(graph)
    scaled, _ = s3.fit_hachure_layer(graph, 2)
    assert graph == before
    assert np.asarray(scaled[0]["strokes"]) == pytest.approx(np.asarray(doc["primitives"][0]["strokes"])*2)


@pytest.mark.parametrize("mode", ["basic", "patent"])
def test_svg_dxf_keep_disconnected_strokes_and_verify_serialized_geometry(tmp_path, mode):
    _, doc = example()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(doc))
    result = s4.run(path, tmp_path, "hatches", formats=("svg", "dxf"), dxf_mode=mode)
    assert not result.flagged
    assert result.n_primitives_in == result.n_primitives_out == 1
    root = ET.parse(result.svg_path).getroot()
    element = root.find(".//{http://www.w3.org/2000/svg}path")
    assert element.get("d").count("M ") == 4
    dxf = ezdxf.readfile(result.dxf_path)
    assert len(dxf.modelspace()) == 4
    assert all(e.dxftype() == "LWPOLYLINE" for e in dxf.modelspace())
    if mode == "patent":
        assert all(e.dxf.layer == "HACHURE" for e in dxf.modelspace())
    first = next(iter(dxf.modelspace()))
    points = first.get_points("xy")
    points[0] = (points[0][0] + 10, points[0][1])
    first.set_points(points, format="xy")
    dxf.saveas(result.dxf_path)
    with pytest.raises(ValueError, match="differs from source"):
        s4.verify_serialized("dxf", result.dxf_path, result.format_reports["dxf"])


def test_bundle_limits_and_ownership_remain_explicit():
    graph, _ = example()
    graph["removed_hachures"] *= 25
    primitives, coverage = s3.fit_hachure_layer(graph)
    assert coverage["represented_edges"] == 100
    assert sum(s4.primitive_budget_cost(p) for p in primitives) == 100
    assert all(len(p["strokes"]) <= 32 for p in primitives)
    assert sorted(i for p in primitives for i in p["source_hachure_indices"]) == list(range(100))


def test_serialized_svg_connector_insertion_is_detected(tmp_path):
    _, doc = example()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(doc))
    result = s4.run(path, tmp_path, "hatches", formats=("svg",))
    tree = ET.parse(result.svg_path)
    node = tree.getroot().find(".//{http://www.w3.org/2000/svg}path")
    node.set("d", node.get("d") + " L 80 20")
    tree.write(result.svg_path)
    with pytest.raises(ValueError, match="differs from source"):
        s4.verify_serialized("svg", result.svg_path, result.format_reports["svg"])


def test_large_bundle_collection_does_not_evade_existing_primitive_limit():
    graph, _ = example()
    graph["removed_hachures"] *= 250
    primitives, _ = s3.fit_hachure_layer(graph)
    assert len(primitives) < 900
    assert sum(s4.primitive_budget_cost(p) for p in primitives) == 1000


def test_missing_mapped_stroke_is_not_hidden_by_remaining_bundle(tmp_path):
    _, doc = example()
    path = tmp_path / "source.json"
    path.write_text(json.dumps(doc))
    result = s4.run(path, tmp_path, "hatches", formats=("dxf",), dxf_mode="patent")
    drawing = ezdxf.readfile(result.dxf_path)
    drawing.modelspace().delete_entity(next(iter(drawing.modelspace())))
    drawing.saveas(result.dxf_path)
    with pytest.raises(ValueError, match="missing from the file"):
        s4.verify_serialized("dxf", result.dxf_path, result.format_reports["dxf"])
