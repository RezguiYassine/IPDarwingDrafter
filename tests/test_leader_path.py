"""Curved leaders: traced as a path, masked along it, reinjected as it.

Release 1 kept only a leader's two endpoints, so the removal mask and the
reinjected leader were both straight chords. A curved leader escaped the
mask, its ink was vectorized, and a straight leader was drawn beside it --
the ghost line the owner found in the audit. These tests hold each link of
the fix, including the one that would fail silently: a polyline leader must
still count as a written leader, or every figure with a curved leader loses
its export check.
"""
import json
import math

import cv2
import ezdxf
import numpy as np
import pytest

from tests.test_acceptance import IDENTITY, _export, _seal, example  # noqa: F401
from tools import batch_run

s0 = batch_run.stage0_handle_references


# ── path helpers ─────────────────────────────────────────────────────────────

def test_a_straight_trace_simplifies_to_its_endpoints():
    line = [(10 + i, 20 + i) for i in range(60)]
    assert s0._simplify_path(line) == [[10.0, 20.0], [69.0, 79.0]]


def test_a_curved_trace_keeps_its_bends():
    arc = [(100 + 80 * math.cos(t), 100 + 80 * math.sin(t))
           for t in np.linspace(0, math.pi / 2, 120)]
    simplified = s0._simplify_path(arc)
    assert len(simplified) > 3, "a quarter circle is not two points"


def test_trim_shortens_the_path_by_arc_length_from_the_tip():
    path = [[0, 0], [10, 0], [10, 10]]                  # 20 px long
    trimmed = s0._trim_path_end(path, 5)
    assert trimmed[-1] == pytest.approx([10, 5])
    np.testing.assert_allclose(s0._trim_path_end(path, 15), [[0, 0], [5, 0]])


def test_trim_of_zero_is_the_identity():
    path = [[0, 0], [10, 0]]
    assert s0._trim_path_end(path, 0) == path


# ── the mask follows the curve ───────────────────────────────────────────────

def _arc():
    t = np.linspace(0, math.pi / 2, 200)
    return np.column_stack([60 + 150 * np.sin(t), 240 - 150 * (1 - np.cos(t))])


def test_the_mask_covers_a_curved_leader_that_a_chord_would_miss():
    arc = _arc()
    ink = np.zeros((300, 300), np.uint8)
    cv2.polylines(ink, [arc.astype(np.int32).reshape(-1, 1, 2)], False, 255, 2)
    ink = ink > 0
    cfg = {"text_bbox_pad": 0, "leader_mask_thickness": 5, "leader_tip_trim": 0}
    label = {"bbox": [0, 0, 1, 1], "leader_lines": [{
        "p1": arc[0].tolist(), "p2": arc[-1].tolist(),
        "label_endpoint": arc[0].tolist(), "leader_to": arc[-1].tolist(),
        "path": s0._simplify_path(arc)}]}
    with_path = s0._build_removal_mask((300, 300), [label], cfg) > 0
    chord_only = dict(label, leader_lines=[{k: v for k, v in label["leader_lines"][0].items()
                                            if k != "path"}])
    with_chord = s0._build_removal_mask((300, 300), [chord_only], cfg) > 0
    left_by_path = (ink & ~with_path).sum() / ink.sum()
    left_by_chord = (ink & ~with_chord).sum() / ink.sum()
    assert left_by_chord > 0.6, "the defect being fixed"
    assert left_by_path < 0.05, "the fix"


# ── reinjection, and the audit that would fail silently ──────────────────────

PATH = [[20.0, 34.0], [30.0, 36.0], [36.0, 42.0], [40.0, 50.0]]


def _add_path_reference(root, row):
    crop = root / "references/crop.png"
    cv2.imwrite(str(crop), np.zeros((8, 8), np.uint8))
    ref = root / "references/F1_references.json"
    document = json.loads(ref.read_text())
    document.update(active_removal=True, n_labels=1, n_leaders=1, reference_labels=[{
        "id": "ref_1", "text": "12", "position": [20, 30], "bbox": [16, 26, 8, 8],
        "crop_bbox": [16, 26, 8, 8], "crop_path": str(crop),
        "ref_class": "numeral", "confidence": 0.99, "in_vocabulary": "unknown",
        "leader_lines": [{"p1": [20, 34], "p2": [40, 50], "leader_to": [40, 50],
                          "removed": True, "path": PATH}],
    }])
    ref.write_text(json.dumps(document))
    return _export(root, row)


def test_a_path_leader_is_written_as_a_polyline_and_counted(example):
    root, row = example
    result = _add_path_reference(root, row)
    dxf = result.format_reports["dxf"]["annotations"][0]
    assert dxf["leaders_written"] == dxf["leaders_expected"] == 1
    assert dxf["complete"], "a polyline leader must still count as written"
    doc = ezdxf.readfile(root / "vectors/F1.dxf")
    leaders = [e for e in doc.modelspace() if e.dxf.layer == "LEADER"]
    assert [e.dxftype() for e in leaders] == ["LWPOLYLINE"]
    assert len(list(leaders[0].get_points())) == len(PATH)
    svg = (root / "vectors/F1.svg").read_text()
    assert "<polyline" in svg


def test_a_path_leader_passes_acceptance(example):
    root, row = example
    _add_path_reference(root, row)
    report, _ = _seal(root, row)
    assert report["checks"]["references"]["status"] == "pass"
    assert report["checks"]["exports"]["status"] == "pass"


def test_a_chord_leader_is_still_a_line(example):
    """Records made before the fix have no path and must export as before."""
    from tests.test_acceptance import _add_reference
    root, row = example
    _add_reference(root, row)
    doc = ezdxf.readfile(root / "vectors/F1.dxf")
    assert [e.dxftype() for e in doc.modelspace() if e.dxf.layer == "LEADER"] == ["LINE"]
