"""A filled region end to end: Stage 1 evidence, fill export, acceptance.

The fixture runs the real Stage 1 on a drawing with a solid section (holding
a label box) and a thin line, so the evidence the validator checks is the
evidence Stage 1 actually writes. Every way the export could drift from that
evidence -- a moved fill, missing evidence, region ink thinned as well, a
swapped mask -- must keep the figure out of the training set.
"""
import json

import cv2
import ezdxf
import numpy as np
import pytest

from tests.test_acceptance import IDENTITY, _export, _seal  # noqa: F401
from tools import acceptance, batch_run

s0 = batch_run.stage0_handle_references
s1 = batch_run.stage1_preprocess
s3 = batch_run.stage3_primitive_fit
CONFIG = {"stage1": {"filled_regions": {"enabled": True}}}


def _drawing():
    gray = np.full((200, 240), 255, np.uint8)
    cv2.rectangle(gray, (40, 40), (139, 109), 0, -1)      # a solid section, 100 x 70
    cv2.rectangle(gray, (70, 60), (99, 79), 255, -1)      # the label box inside it
    gray[160, 20:221] = 0                                 # a thin line, well clear
    return gray


@pytest.fixture
def filled(tmp_path):
    root = tmp_path / "run" / "EP2"
    for folder in ("references", "graphs", "primitives"):
        (root / folder).mkdir(parents=True)
    gray = _drawing()
    source = tmp_path / "source.tif"
    norefs, mask = root / "references/F1_norefs.png", root / "references/F1_references_mask.png"
    for path, image in ((source, gray), (norefs, gray), (mask, np.zeros_like(gray))):
        cv2.imwrite(str(path), image)
    stage1 = s1.run(norefs, root, "F1", CONFIG)
    assert stage1.n_regions == 1
    skeleton = cv2.imread(str(stage1.skeleton_path), cv2.IMREAD_GRAYSCALE) > 0
    ys, xs = np.nonzero(skeleton)
    assert set(ys.tolist()) == {160}, "the section is not thinned; only the line is"
    (root / "graphs/F1_graph.json").write_text(json.dumps({
        "image_shape": [200, 240], "removed_hachures": [],
        "edges": [{"id": 0, "is_closed": False,
                   "pixels": [[int(x), 160] for x in sorted(xs.tolist())]}]}))
    primitives = {
        "sketch_id": "F1", "image_size": [240, 200],
        "primitives": s3.region_primitives(stage1.regions_path, [240, 200]) + [
            {"edge_id": 0, "type": "line", "p1": [int(xs.min()), 160],
             "p2": [int(xs.max()), 160], "confidence": 1}],
        "quality_metrics": {"hachure_coverage": {
            "source_edges": 0, "represented_edges": 0, "unrepresented_indices": []}},
    }
    (root / "primitives/F1_primitives.json").write_text(json.dumps(primitives))
    references = s0._build_reference_doc(
        sketch_id="F1", source_path=source, gray=gray, ink=gray < 128, labels=[],
        mask_path=mask, reference_free_path=norefs, crop_dir=root / "references/crops",
        cfg={"crop_pad": 2}, active_removal=False, removed_ink_ratio=0, repair_pixels=0,
        n_iterations=1, iteration_summaries=[], flagged=False, reason="ok",
    )
    (root / "references/F1_references.json").write_text(json.dumps(references))
    row = {"patent_id": "EP2", "sketch_id": "F1", "input_path": str(source),
           "status": "ok", "error": None, "completed_at": "2026-09-29",
           **{f"s{i}_flagged": 0 for i in range(5)}}
    _export(root, row)
    return root, row


def _geometry(root):
    return json.loads((root / "geometry/F1_geometry_report.json").read_text())


def test_a_filled_region_is_exported_as_a_fill_and_accepted(filled):
    root, row = filled
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "accepted", report["reason_codes"]
    assert acceptance.eligibility_reason(row, root.parent, IDENTITY) is None
    regions = _geometry(root)["checks"]["regions"]
    assert regions["status"] == "pass" and regions["regions"] == 1
    assert regions["thinned_region_pixels"] == 0 and regions["unrepresented_pixels"] == 0
    assert {"regions", "regions_mask"} <= set(report["artifacts"])

    svg = (root / "vectors/F1.svg").read_text()
    assert 'fill="black"' in svg and 'fill-rule="evenodd"' in svg
    doc = ezdxf.readfile(root / "vectors/F1.dxf")
    [fill] = [e for e in doc.modelspace() if e.dxftype() == "HATCH"]
    assert fill.dxf.layer == "FILL" and fill.dxf.solid_fill == 1
    assert len(fill.paths) == 2, "the outline and the label box"


def test_the_fill_renders_where_the_ink_was(filled):
    import cairosvg
    root, _ = filled
    png = cairosvg.svg2png(url=str(root / "vectors/F1.svg"), output_width=240,
                           output_height=200, background_color="white")
    render = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE) < 128
    ink = _drawing() < 128
    section = np.zeros_like(ink)
    section[40:110, 40:140] = True
    agreement = (render & ink & section).sum() / (ink & section).sum()
    assert agreement > 0.97
    assert not render[62:78, 72:98].any(), "the label box stays paper"


def test_a_fill_moved_after_stage1_is_refused(filled):
    root, row = filled
    path = root / "primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"][0]["rings"][0] = [[x + 6, y] for x, y in document["primitives"][0]["rings"][0]]
    path.write_text(json.dumps(document))
    _export(root, row)
    report, _ = _seal(root, row)
    assert report["acceptance_status"] != "accepted"
    assert "geometry_region_changed_from_evidence" in report["reason_codes"]


def test_fills_without_region_evidence_are_refused(filled):
    root, row = filled
    (root / "cleaned/F1_regions.json").unlink()
    report, _ = _seal(root, row)
    assert report["acceptance_status"] != "accepted"
    assert "geometry_region_evidence_missing" in report["reason_codes"]


def test_region_ink_thinned_as_well_is_caught(filled):
    root, row = filled
    path = root / "cleaned/F1_skeleton.png"
    skeleton = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    skeleton[75, 45:135] = 255                     # a medial axis through the section
    cv2.imwrite(str(path), skeleton)
    report, _ = _seal(root, row)
    assert report["acceptance_status"] != "accepted"
    assert "geometry_region_double_representation" in report["reason_codes"]


def test_a_swapped_region_mask_is_detected(filled):
    root, row = filled
    path = root / "cleaned/F1_regions_mask.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask[45:50, 45:50] = 0
    cv2.imwrite(str(path), mask)
    report, _ = _seal(root, row)
    assert report["acceptance_status"] != "accepted"
    assert "geometry_region_evidence_invalid" in report["reason_codes"]


def test_region_evidence_changed_after_sealing_blocks_training(filled):
    root, row = filled
    _seal(root, row)
    path = root / "cleaned/F1_regions.json"
    document = json.loads(path.read_text())
    document["regions"][0]["solidity"] = 0.5
    path.write_text(json.dumps(document))
    assert acceptance.eligibility_reason(row, root.parent, IDENTITY) == "acceptance_artifact_changed"


def test_disabled_regions_leave_stage1_as_it_was(tmp_path):
    source = tmp_path / "in.png"
    cv2.imwrite(str(source), _drawing())
    stage1 = s1.run(source, tmp_path, "F1", {})
    assert stage1.regions_path is None and stage1.n_regions == 0
    assert not (tmp_path / "cleaned/F1_regions.json").exists()
    skeleton = cv2.imread(str(stage1.skeleton_path), cv2.IMREAD_GRAYSCALE) > 0
    assert skeleton[40:110, 40:140].any(), "without regions the section is thinned, as in release 1"
