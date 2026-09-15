from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import geometry_validation as geometry


def drawing(primitive, points, *, closed=False, scale=1.0):
    source = np.asarray(points, dtype=float)
    primitive = {**deepcopy(primitive), "edge_id": 7, "confidence": 0.999}
    graph = {"image_shape": [round(512*scale), round(512*scale)],
             "original_image_shape": [512, 512], "stage2_scale": scale,
             "edges": [{"id": 7, "pixels": (source*scale).tolist(), "is_closed": closed}],
             "removed_hachures": []}
    document = {"image_size": [512, 512], "stage2_scale": scale, "primitives": [primitive]}
    raster = np.zeros((512, 512), dtype=bool)
    pixels = np.rint(source).astype(int)
    raster[pixels[:, 1], pixels[:, 0]] = True
    return graph, document, raster


def circle_points(start=0, end=360, radius=60, center=(200, 200)):
    angle = np.radians(np.linspace(start, end, 1501))
    return np.column_stack([np.cos(angle), np.sin(angle)]) * radius + center


def line_example():
    return drawing({"type": "line", "p1": [20, 40], "p2": [220, 40]},
                   np.column_stack([np.arange(20, 221), np.full(201, 40)]))


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("kind", ["line", "circle", "arc", "ellipse", "bezier", "polyline", "polygon", "path"])
def test_supported_geometry_passes_at_multiple_stage2_scales(kind, scale):
    if kind == "line":
        primitive = {"type": kind, "p1": [20, 40], "p2": [220, 40]}
        points = np.column_stack([np.arange(20, 221), np.full(201, 40)])
    elif kind in {"circle", "arc"}:
        primitive = {"type": kind, "center": [200, 200], "radius": 60}
        if kind == "arc":
            primitive.update(start_angle=350, end_angle=30)
        points = circle_points(350, 390) if kind == "arc" else circle_points()
    elif kind == "ellipse":
        primitive = {"type": kind, "center": [200, 200], "a": 80, "b": 35, "angle": 40}
        t, angle = np.linspace(0, 2*np.pi, 2001), np.radians(40)
        x, y = 80*np.cos(t), 35*np.sin(t)
        points = np.column_stack([200+x*np.cos(angle)-y*np.sin(angle),
                                  200+x*np.sin(angle)+y*np.cos(angle)])
    elif kind == "bezier":
        primitive = {"type": kind, "points": [[50, 100], [100, 20], [150, 180], [200, 100]]}
        t = np.linspace(0, 1, 1001)
        points = np.column_stack([50+150*t, 100-240*t+720*t**2-480*t**3])
    else:
        vertices = [[50, 50], [150, 50], [150, 150]]
        points = np.vstack([np.linspace(vertices[0], vertices[1], 201),
                            np.linspace(vertices[1], vertices[2], 201)])
        primitive = {"type": kind, "points": vertices}
        if kind == "polygon":
            points = np.vstack([points, np.linspace(vertices[2], vertices[0], 301)])
        elif kind == "path":
            primitive = {"type": kind, "segments": [
                {"type": "line", "p1": vertices[0], "p2": vertices[1]},
                {"type": "line", "p1": vertices[2], "p2": vertices[1]}]}
    report = geometry.validate(*drawing(primitive, points, closed=kind in {"circle", "ellipse", "polygon"}, scale=scale))
    assert report["status"] == "pass", report


def test_high_confidence_partial_arc_cannot_support_a_whole_circle():
    args = drawing({"type": "circle", "center": [200, 200], "radius": 60}, circle_points(15, 45), closed=True)
    args[0]["edges"][0].update(topology_origin="unclaimed_component", is_simple_cycle=False)
    report = geometry.validate(*args)
    assert report["status"] == "fail"
    item = report["primitives"][0]
    assert "geometry_incomplete_angular_coverage" in item["reason_codes"]
    assert item["metrics"]["angular"]["coverage_degrees"] == pytest.approx(30)
    assert item["metrics"]["model_to_source"]["fraction_within_tolerance"] < 0.15


def test_closed_flag_alone_does_not_certify_noncycle_topology():
    graph, doc, raster = drawing({"type": "circle", "center": [200, 200], "radius": 60}, circle_points(), closed=True)
    graph["edges"][0].update(topology_origin="unclaimed_component", is_simple_cycle=False)
    report = geometry.validate(graph, doc, raster)
    assert report["status"] == "review"
    assert report["reason_codes"] == ["geometry_noncycle_source"]


def test_other_edges_cannot_lend_support_to_an_unrelated_primitive():
    graph, doc, raster = line_example()
    graph["edges"].append({"id": 8, "is_closed": False,
                           "pixels": [[x, 100] for x in range(20, 221)]})
    doc["primitives"].append({"edge_id": 8, "type": "line", "p1": [20, 100], "p2": [220, 100]})
    doc["primitives"][0].update(p1=[20, 100], p2=[220, 100])
    raster[100, 20:221] = True
    report = geometry.validate(graph, doc, raster)
    assert "geometry_unsupported_primitive" in report["primitives"][0]["reason_codes"]


def test_short_line_cannot_hide_missing_source_trace():
    graph, doc, raster = line_example()
    doc["primitives"][0]["p2"] = [40, 40]
    report = geometry.validate(graph, doc, raster)
    assert "geometry_source_trace_lost" in report["reason_codes"]
    assert "geometry_endpoint_mismatch" in report["reason_codes"]


def test_reversed_source_trace_does_not_cause_an_endpoint_failure():
    graph, doc, raster = line_example()
    graph["edges"][0]["pixels"].reverse()
    assert geometry.validate(graph, doc, raster)["status"] == "pass"


def test_small_but_long_unsupported_interval_cannot_hide_below_p95():
    x = np.arange(10, 500, 0.25)
    source = np.column_stack([x[(x < 240) | (x > 260)], np.full(np.count_nonzero((x < 240) | (x > 260)), 40)])
    report = geometry.validate(*drawing({"type": "line", "p1": [10, 40], "p2": [499.75, 40]}, source))
    item = report["primitives"][0]
    assert item["metrics"]["model_to_source"]["fraction_within_tolerance"] > 0.95
    assert "geometry_unsupported_run" in item["reason_codes"]


def test_path_bridge_and_dxf_endpoint_gap_are_not_invisible_to_validation():
    source = np.vstack([np.column_stack([np.arange(20, 101), np.full(81, 40)]),
                        np.column_stack([np.arange(120, 221), np.full(101, 40)])])
    primitive = {"type": "path", "segments": [
        {"type": "line", "p1": [20, 40], "p2": [100, 40]},
        {"type": "line", "p1": [120, 40], "p2": [220, 40]}]}
    report = geometry.validate(*drawing(primitive, source))
    assert "geometry_path_discontinuity" in report["reason_codes"]
    assert "geometry_unsupported_run" in report["reason_codes"]


def test_stage2_cannot_lose_strokes_without_raster_coverage_failure():
    graph, doc, raster = line_example()
    raster[100, 20:221] = True
    report = geometry.validate(graph, doc, raster)
    assert report["primitives"][0]["status"] == "pass"
    assert "geometry_skeleton_coverage_lost" in report["reason_codes"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_id", "duplicate_graph_id"])
def test_main_source_ownership_is_exact_once(mutation):
    graph, doc, raster = line_example()
    if mutation == "missing":
        doc["primitives"] = []
    elif mutation == "duplicate":
        doc["primitives"] *= 2
    elif mutation == "duplicate_graph_id":
        graph["edges"] *= 2
    else:
        doc["primitives"][0]["edge_id"] = 999
    assert geometry.validate(graph, doc, raster)["status"] == "error"


def test_hatch_edges_use_indices_not_colliding_main_edge_ids():
    graph, doc, raster = line_example()
    graph["removed_hachures"] = [{"id": 7, "is_closed": False, "pixels": [[x, 100] for x in range(20, 101)]}]
    doc["primitives"].append({"edge_id": 7, "type": "line", "style": "hachure",
                              "source_hachure_indices": [0], "p1": [20, 100], "p2": [100, 100]})
    raster[100, 20:101] = True
    assert geometry.validate(graph, doc, raster)["status"] == "pass"
    doc["primitives"][-1]["p2"] = [30, 100]
    assert geometry.validate(graph, doc, raster)["status"] == "fail"


def test_hatch_region_accounting_is_not_a_pattern_fidelity_pass():
    graph, doc, raster = line_example()
    graph["removed_hachures"] = [{"id": 7, "pixels": [[x, 100] for x in range(20, 101)]}]
    doc["primitives"].append({"type": "hatch", "style": "hachure", "source_hachure_indices": [0],
                              "boundary": [[10, 90], [110, 90], [110, 110], [10, 110]],
                              "angles": [0], "spacing": 10})
    report = geometry.validate(graph, doc, raster)
    assert report["status"] == "review"
    assert "geometry_hatch_pattern_unverified" in report["reason_codes"]


@pytest.mark.parametrize("mutation", ["nan", "scale", "shape", "zero_length"])
def test_invalid_geometry_or_coordinate_frame_fails_closed(mutation):
    graph, doc, raster = line_example()
    if mutation == "nan":
        doc["primitives"][0]["p2"][0] = float("nan")
    elif mutation == "scale":
        graph["stage2_scale"] = 0.5
    elif mutation == "shape":
        doc["image_size"] = [511, 512]
    else:
        doc["primitives"][0]["p2"] = [20, 40]
    assert geometry.validate(graph, doc, raster)["status"] == "error"


def test_sampling_budget_does_not_silently_reduce_validation_resolution():
    report = geometry.validate(*line_example(), policy=replace(geometry.POLICY, max_samples_per_primitive=10))
    assert report["status"] == "review"
    assert "geometry_sampling_budget" in report["reason_codes"]


def test_raster_is_required_for_a_full_geometry_pass():
    graph, doc, _ = line_example()
    report = geometry.validate(graph, doc)
    assert report["primitives"][0]["status"] == "pass"
    assert report["status"] == "review"


def test_run_binds_exact_input_files_and_saves_report(tmp_path):
    graph, doc, raster = line_example()
    paths = [tmp_path/name for name in ("graph.json", "primitives.json", "skeleton.png", "geometry.json")]
    paths[0].write_text(json.dumps(graph))
    paths[1].write_text(json.dumps(doc))
    cv2.imwrite(str(paths[2]), raster.astype(np.uint8)*255)
    report = geometry.run(*paths)
    assert report["status"] == "pass"
    assert json.loads(paths[3].read_text()) == report
    assert report["inputs"]["graph"]["sha256"] == geometry.file_digest(paths[0])


def test_confirmed_patent_circles_are_rejected_using_their_original_pixels():
    fixture = json.loads((Path(__file__).parent / "fixtures/unsupported_patent_circles.json").read_text())
    report = geometry.validate(fixture["graph"], fixture["document"])
    assert report["status"] == "fail"
    for item, expected in zip(report["primitives"], [30.8565, 28.2549]):
        assert item["status"] == "fail"
        assert item["metrics"]["source_to_model"]["p95"] < 1
        assert item["metrics"]["model_to_source"]["p95"] > 400
        assert item["metrics"]["angular"]["coverage_degrees"] == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_translation_and_rotation_preserve_supported_line_decisions(degrees):
    graph, doc, _ = line_example()
    angle = np.radians(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = (np.asarray(graph["edges"][0]["pixels"]) - [120, 40]) @ matrix.T + [256, 256]
    primitive = {"type": "line", "p1": points[0].tolist(), "p2": points[-1].tolist()}
    assert geometry.validate(*drawing(primitive, points))["status"] == "pass"
