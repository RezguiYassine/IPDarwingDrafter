import json
from pathlib import Path

import numpy as np
import pytest

from tools.batch_run import stage3_primitive_fit as s3
from tools.geometry_validation import check_primitive


CASES = json.loads((Path(__file__).parent / "fixtures/unsupported_fits.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['patent']}_{c['edge']['id']}")
def test_real_unsupported_fits_are_repaired_against_unchanged_source(case):
    edge, before = case["edge"], case["before"]
    assert check_primitive(before, edge)["status"] == "fail"
    after = s3._guard_source_primitive(edge, before)
    report = check_primitive(after, edge)
    assert report["status"] == "pass"
    assert report["metrics"]["max_connector_gap"] <= 1e-6


def test_tolerated_path_gap_is_still_anchored_for_actual_continuity():
    points = [[x, 10] for x in range(10, 51)] + [[50, y] for y in range(11, 51)]
    edge = {"id": 1, "pixels": points, "is_closed": False}
    before = {"type": "path", "edge_id": 1, "confidence": 0.9,
              "segments": [{"type": "line", "p1": [10, 10], "p2": [49, 10]},
                           {"type": "line", "p1": [50, 10], "p2": [50, 50]}]}
    assert check_primitive(before, edge)["status"] == "pass"
    after = s3._guard_source_primitive(edge, before)
    assert after["type"] == "path"
    assert check_primitive(after, edge)["metrics"]["max_connector_gap"] <= 1e-6


def test_raw_fallback_cannot_certify_an_unsupported_gap():
    edge = {"id": 1, "pixels": [[x, 10] for x in list(range(10, 31))+list(range(70, 91))],
            "is_closed": False}
    result = s3.fit_edge_ransac(edge)
    assert check_primitive(result, edge)["status"] == "fail"


def test_already_supported_analytic_geometry_is_unchanged():
    edge = {"id": 1, "pixels": [[x, 10] for x in range(10, 91)], "is_closed": False}
    primitive = {"type": "line", "edge_id": 1, "p1": [10, 10], "p2": [90, 10], "confidence": 0.99}
    assert s3._guard_source_primitive(edge, primitive) is primitive


HATCH_CASES = json.loads((Path(__file__).parent / "fixtures/unsupported_hatches.json").read_text())["cases"]


@pytest.mark.parametrize("case", HATCH_CASES, ids=lambda c: f"{c['patent']}_{c['edge']['id']}")
def test_legacy_hatch_without_region_metadata_keeps_its_complete_source(case):
    edge = case["edge"]
    assert check_primitive(case["before"], edge)["status"] == "fail"
    result = s3._fit_removed_hachure(edge)
    assert result["type"] == "polyline"
    assert result["points"] == edge["pixels"]
    assert check_primitive(result, edge)["status"] == "pass"


def test_endpoint_anchoring_retains_exact_arc_when_already_constrained(monkeypatch):
    angles = np.linspace(0, np.pi/2, 100)
    points = np.column_stack([50+30*np.cos(angles), 50+30*np.sin(angles)])
    edge = {"id": 1, "pixels": points.tolist(), "is_closed": False}
    monkeypatch.setattr(s3, "_split_at_corners", lambda pts: [pts])
    result = s3._fit_compound_path(edge, 1, require_fidelity=True, anchor_endpoints=True)
    assert result["segments"][0]["type"] == "arc"
