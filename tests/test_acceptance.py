from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import cv2
import ezdxf
import numpy as np
import pytest

from tools import acceptance, batch_run, build_training_manifest, results_db

s0 = batch_run.stage0_handle_references
s4 = batch_run.stage4_export
IDENTITY = {"effective_config_sha256": "fixture-config", "implementation_sha256": "fixture-code"}
VALIDATIONS = {name: {"status": "pass", "validator": "test_fixture", "version": "1"}
               for name in ("content", "geometry")}


@pytest.fixture
def example(tmp_path):
    root = tmp_path / "run" / "EP1"
    for folder in ("cleaned", "references", "graphs", "primitives"):
        (root / folder).mkdir(parents=True)
    gray = np.full((64, 64), 255, dtype=np.uint8)
    gray[20, 10:51] = 0
    source = tmp_path / "source.tif"
    for path in (source, root / "cleaned/F1_skeleton.png", root / "cleaned/F1_cleaned.png",
                 root / "references/F1_norefs.png", root / "references/F1_references_mask.png"):
        cv2.imwrite(str(path), gray)
    cv2.imwrite(str(root / "cleaned/F1_skeleton.png"), 255 - gray)
    references = s0._build_reference_doc(
        sketch_id="F1", source_path=source, gray=gray, ink=gray < 128, labels=[],
        mask_path=root / "references/F1_references_mask.png",
        reference_free_path=root / "references/F1_norefs.png",
        crop_dir=root / "references/crops", cfg={"crop_pad": 2},
        active_removal=False, removed_ink_ratio=0, repair_pixels=0,
        n_iterations=1, iteration_summaries=[], flagged=False, reason="ok",
    )
    (root / "references/F1_references.json").write_text(json.dumps(references))
    (root / "graphs/F1_graph.json").write_text(json.dumps({
        "image_shape": [64, 64], "edges": [{"id": 0, "is_closed": False,
        "pixels": [[x, 20] for x in range(10, 51)]}], "removed_hachures": []}))
    primitives = {
        "sketch_id": "F1", "image_size": [64, 64],
        "primitives": [{"edge_id": 0, "type": "line", "p1": [10, 20], "p2": [50, 20], "confidence": 1}],
        "quality_metrics": {"hachure_coverage": {
            "source_edges": 0, "represented_edges": 0, "unrepresented_indices": []}},
    }
    (root / "primitives/F1_primitives.json").write_text(json.dumps(primitives))
    row = {"patent_id": "EP1", "sketch_id": "F1", "input_path": str(source),
           "status": "ok", "error": None, "completed_at": "2026-09-10",
           **{f"s{i}_flagged": 0 for i in range(5)}}
    _export(root, row)
    return root, row


def _export(root, row):
    path = s0.attach_references_to_primitives(root / "primitives/F1_primitives.json",
                                              root / "references/F1_references.json")
    result = s4.run(path, root, "F1", formats=("svg", "dxf"), dxf_mode="patent")
    row.update(s4_n_in=result.n_primitives_in, s4_n_out=result.n_primitives_out,
               s4_flagged=int(result.flagged))
    return result


def _seal(root, row, validations=VALIDATIONS):
    report, path = acceptance.record(row, root, IDENTITY, validations=validations)
    row.update(acceptance_status=report["acceptance_status"],
               acceptance_policy_version=report["policy_version"],
               acceptance_sha256=acceptance.file_digest(path),
               training_eligible=int(report["training_eligible"]))
    return report, path


def _add_reference(root, row, *, text="12"):
    crop = root / "references/crop.png"
    cv2.imwrite(str(crop), np.zeros((8, 8), np.uint8))
    path = root / "references/F1_references.json"
    document = json.loads(path.read_text())
    document.update(active_removal=True, n_labels=1, n_leaders=1, reference_labels=[{
        "id": "ref_1", "text": text, "position": [20, 30], "bbox": [16, 26, 8, 8],
        "crop_bbox": [16, 26, 8, 8], "crop_path": str(crop),
        "ref_class": "numeral", "confidence": 0.99,
        "leader_lines": [{"p1": [20, 34], "p2": [40, 50], "leader_to": [40, 50], "removed": True}],
    }])
    path.write_text(json.dumps(document))
    return _export(root, row)


def test_complete_validated_record_is_eligible(example):
    root, row = example
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "accepted"
    assert report["reason_codes"] == []
    assert acceptance.eligibility_reason(row, root.parent, IDENTITY) is None


def test_successful_execution_requires_content_and_computes_geometry_validation(example):
    root, row = example
    report, _ = _seal(root, row, validations={})
    assert report["acceptance_status"] == "review"
    assert not report["training_eligible"]
    assert set(report["reason_codes"]) == {"content_validation_pending"}
    assert report["checks"]["geometry"]["status"] == "pass"


@pytest.mark.parametrize("stage", range(5))
def test_every_stage_flag_blocks_training_without_changing_execution_status(example, stage):
    root, row = example
    row[f"s{stage}_flagged"] = 1
    report, _ = _seal(root, row)
    assert row["status"] == "ok"
    assert report["acceptance_status"] == "review"
    assert not report["training_eligible"]


@pytest.mark.parametrize("status,verdict", [
    ("stage0", "error"), ("stage4", "error"),
    ("quality_gate_stage2", "rejected"), ("ok_stage2", "review"),
])
def test_noncompleted_rows_receive_nonaccepted_records(example, status, verdict):
    root, row = example
    row["status"] = status
    report, path = _seal(root, row)
    assert path.exists()
    assert report["acceptance_status"] == verdict


@pytest.mark.parametrize("relative", ["vectors/F1.dxf", "graphs/F1_graph.json", "references/F1_references.json"])
def test_missing_required_artifacts_fail_closed(example, relative):
    root, row = example
    (root / relative).unlink()
    assert _seal(root, row)[0]["acceptance_status"] == "error"


def test_valid_svg_cannot_hide_failed_dxf(example, monkeypatch):
    root, row = example
    def fail(*args, **kwargs):
        raise RuntimeError("injected DXF failure")
    monkeypatch.setattr(s4, "_export_dxf_patent", fail)
    result = _export(root, row)
    assert result.format_reports["svg"]["complete"]
    assert not result.format_reports["dxf"]["complete"]
    assert result.n_primitives_out == 0
    assert result.flagged
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "error"
    assert "dxf_invalid_export" in report["reason_codes"]


@pytest.mark.parametrize("primitive", [
    {"type": "path", "segments": []}, {"type": "bezier", "points": []},
    {"type": "hatch", "boundary": []}, {"type": "unknown"},
    {"type": "path", "segments": [{"type": "unknown"}]},
    {"type": "circle", "center": [0, 0], "radius": float("nan")},
])
def test_empty_or_invalid_primitives_are_not_counted_as_exported(example, primitive):
    root, row = example
    path = root / "primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"] = [primitive]
    path.write_text(json.dumps(document))
    result = _export(root, row)
    assert result.flagged
    assert result.n_primitives_out == 0
    assert all(not r["primitives"][0]["complete"] for r in result.format_reports.values())


def test_compound_path_can_emit_multiple_dxf_entities_for_one_source_item(example):
    root, row = example
    path = root / "primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"] = [{"edge_id": 0, "type": "path", "segments": [
        {"type": "line", "p1": [10, 20], "p2": [30, 20]},
        {"type": "line", "p1": [30, 20], "p2": [30, 40]},
    ]}]
    path.write_text(json.dumps(document))
    graph_path = root / "graphs/F1_graph.json"
    graph = json.loads(graph_path.read_text())
    points = [[x, 20] for x in range(10, 31)] + [[30, y] for y in range(21, 41)]
    graph["edges"][0]["pixels"] = points
    graph_path.write_text(json.dumps(graph))
    skeleton = np.zeros((64, 64), dtype=np.uint8)
    for x, y in points:
        skeleton[y, x] = 255
    cv2.imwrite(str(root / "cleaned/F1_skeleton.png"), skeleton)
    result = _export(root, row)
    assert result.n_primitives_out == 1
    assert len(result.format_reports["dxf"]["primitives"][0]["entity_ids"]) == 2
    assert _seal(root, row)[0]["training_eligible"]


def test_supplied_geometry_pass_cannot_override_measured_failure(example):
    root, row = example
    path = root / "primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"][0].update(p1=[10, 50], p2=[50, 50])
    path.write_text(json.dumps(document))
    _export(root, row)
    report, _ = _seal(root, row, validations=VALIDATIONS)
    assert report["checks"]["geometry"]["status"] == "fail"
    assert report["acceptance_status"] == "rejected"
    assert not report["training_eligible"]


def test_changed_geometry_report_blocks_training(example):
    root, row = example
    report, _ = _seal(root, row)
    path = Path(report["artifacts"]["geometry_report"]["path"])
    geometry = json.loads(path.read_text())
    geometry["summary"]["primitives"] = 100
    path.write_text(json.dumps(geometry))
    assert acceptance.eligibility_reason(row, root.parent, IDENTITY) == "acceptance_artifact_changed"


def test_malformed_serialized_svg_is_detected(example, monkeypatch):
    root, row = example
    original = s4._export_svg
    def corrupt(data, path, **kwargs):
        count = original(data, path, **kwargs)
        path.write_text("<svg>broken")
        return count
    monkeypatch.setattr(s4, "_export_svg", corrupt)
    result = _export(root, row)
    assert not result.format_reports["svg"]["serialized_valid"]
    assert _seal(root, row)[0]["acceptance_status"] == "error"


def test_hatch_coverage_cannot_be_claimed_without_ownership(example):
    root, row = example
    (root / "graphs/F1_graph.json").write_text(json.dumps({"removed_hachures": [{"pixels": [[1, 1], [2, 2]]}]}))
    report, _ = _seal(root, row)
    assert "hachure_coverage_incomplete" in report["reason_codes"]
    assert not report["training_eligible"]


def test_known_reference_and_leader_are_accounted_independently(example):
    root, row = example
    result = _add_reference(root, row)
    assert not result.flagged
    assert result.format_reports["svg"]["annotations"][0]["label"] == "crop"
    assert result.format_reports["dxf"]["annotations"][0]["label"] == "text"
    assert result.format_reports["dxf"]["annotations"][0]["leaders_written"] == 1
    assert _seal(root, row)[0]["training_eligible"]


def test_unknown_dxf_text_is_review_not_fabricated_acceptance(example):
    root, row = example
    result = _add_reference(root, row, text="")
    assert result.format_reports["svg"]["complete"]
    assert not result.format_reports["dxf"]["complete"]
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "review"
    assert "unknown_reference_text" in report["reason_codes"]


def test_dropped_dxf_annotation_cannot_pass_with_complete_primitives(example, monkeypatch):
    root, row = example
    monkeypatch.setattr(s4, "_add_bezugszeichen", lambda *args: None)
    result = _add_reference(root, row)
    assert result.n_primitives_out == result.n_primitives_in
    assert result.flagged
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "error"
    assert "dxf_incomplete_leaders" in report["reason_codes"]
    assert "dxf_missing_reference_label" in report["reason_codes"]


def test_missing_crop_cannot_be_hidden_by_known_text_fallback(example):
    root, row = example
    _add_reference(root, row)
    (root / "references/crop.png").unlink()
    result = _export(root, row)
    assert result.format_reports["svg"]["annotations"][0]["label"] == "text"
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "error"
    assert "missing_reference_crop_0" in report["reason_codes"]


def test_lost_reference_attachment_is_detected_even_when_exports_are_valid(example):
    root, row = example
    _add_reference(root, row)
    path = root / "primitives/F1_primitives_with_refs.json"
    document = json.loads(path.read_text())
    document["annotations"] = []
    path.write_text(json.dumps(document))
    s4.run(path, root, "F1", formats=("svg", "dxf"), dxf_mode="patent")
    report, _ = _seal(root, row)
    assert "reference_annotation_count_mismatch" in report["reason_codes"]
    assert not report["training_eligible"]


def test_missing_compound_segment_is_recorded_as_partial_failure(example, monkeypatch):
    root, row = example
    path = root / "primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"] = [{"type": "path", "segments": [
        {"type": "line", "p1": [10, 20], "p2": [30, 20]},
        {"type": "line", "p1": [30, 20], "p2": [30, 40]},
    ]}]
    path.write_text(json.dumps(document))
    original = s4._dxf_add_segment
    def omit_second(msp, segment, *args):
        if segment["p1"] != [30, 20]:
            original(msp, segment, *args)
    monkeypatch.setattr(s4, "_dxf_add_segment", omit_second)
    result = _export(root, row)
    item = result.format_reports["dxf"]["primitives"][0]
    assert not item["complete"]
    assert len(item["entity_ids"]) == 1
    assert _seal(root, row)[0]["acceptance_status"] == "error"


def test_blank_svg_cannot_pass_serialization_check(example):
    root, row = example
    path = root / "primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"] = [{"type": "line", "p1": [100, 100], "p2": [200, 100]}]
    path.write_text(json.dumps(document))
    result = _export(root, row)
    assert not result.format_reports["svg"]["serialized_valid"]
    assert result.flagged


def test_dxf_entity_lost_during_serialization_is_not_hidden_by_created_count(example, monkeypatch):
    root, row = example
    original = s4._export_dxf_patent
    def lose_entity(data, path, **kwargs):
        count = original(data, path, **kwargs)
        document = ezdxf.readfile(path)
        modelspace = document.modelspace()
        modelspace.delete_entity(next(iter(modelspace)))
        document.saveas(path)
        return count
    monkeypatch.setattr(s4, "_export_dxf_patent", lose_entity)
    result = _export(root, row)
    assert result.format_reports["dxf"]["primitives_created"] == 1
    assert result.format_reports["dxf"]["primitives_written"] == 0
    assert _seal(root, row)[0]["acceptance_status"] == "error"


def test_unknown_label_does_not_hide_missing_leader(example, monkeypatch):
    root, row = example
    monkeypatch.setattr(s4, "_add_bezugszeichen", lambda *args: None)
    _add_reference(root, row, text="")
    report, _ = _seal(root, row)
    assert report["acceptance_status"] == "error"
    assert "dxf_incomplete_leaders" in report["reason_codes"]


def test_unversioned_pass_is_not_a_validation_result(example):
    root, row = example
    report, _ = _seal(root, row, {"content": {"status": "pass"}, "geometry": {"status": "pass"}})
    assert report["acceptance_status"] == "review"
    assert not report["training_eligible"]


@pytest.mark.parametrize("mutation,reason", [
    ("artifact", "acceptance_artifact_changed"), ("row", "acceptance_row_mismatch"),
    ("report", "acceptance_record_hash_mismatch"), ("identity", "acceptance_deployment_mismatch"),
    ("policy", "acceptance_policy_mismatch"), ("missing", "acceptance_missing"),
])
def test_manifest_rechecks_acceptance_provenance(example, mutation, reason):
    root, row = example
    report, path = _seal(root, row)
    identity = deepcopy(IDENTITY)
    if mutation == "artifact":
        (root / "vectors/F1.svg").write_text("changed")
    elif mutation == "row":
        row["s0_flagged"] = 1
    elif mutation == "identity":
        identity["implementation_sha256"] = "changed"
    elif mutation == "report":
        path.write_text("{}")
    elif mutation == "policy":
        report["policy_version"] = "obsolete"
        path.write_text(json.dumps(report))
        row["acceptance_sha256"] = acceptance.file_digest(path)
    else:
        path.unlink()
    assert acceptance.eligibility_reason(row, root.parent, identity) == reason


def test_batch_wrapper_records_stage0_warning_and_pending_checks(example, monkeypatch):
    root, row = example
    row["s0_flagged"] = 1
    monkeypatch.setattr(batch_run, "_run_stages", lambda job: dict(row))
    monkeypatch.setattr(batch_run, "_WORKER_CFG", {"stage0": {"enabled": True}})
    monkeypatch.setattr(batch_run, "_WORKER_DEPLOYMENT_IDENTITY", IDENTITY)
    result = batch_run._process_one(("EP1", "F1", row["input_path"], str(root)))
    assert result["status"] == "ok"
    assert result["acceptance_status"] == "review"
    assert result["training_eligible"] == 0
    assert Path(result["acceptance_path"]).is_file()


def test_database_migration_does_not_grandfather_old_ok_rows(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE results (patent_id TEXT, sketch_id TEXT, status TEXT)")
        connection.execute("INSERT INTO results VALUES ('EP1', 'F1', 'ok')")
    results_db.init_db(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT acceptance_status, training_eligible FROM results").fetchone() == (None, None)


def test_reprocessing_invalidates_previous_acceptance_before_workers_start(example):
    root, row = example
    _seal(root, row)
    db = root.parent / "results.db"
    results_db.init_db(db)
    with results_db.connect(db) as connection:
        results_db.insert_row(connection, row)
        results_db.invalidate_acceptance(connection, [("EP1", "F1")])
        assert connection.execute("SELECT status, training_eligible, acceptance_sha256 FROM results").fetchone() == ("ok", 0, None)


@pytest.mark.parametrize("accepted", [True, False])
def test_training_manifest_cannot_bypass_acceptance_with_permissive_curation(example, monkeypatch, accepted):
    root, row = example
    _seal(root, row, VALIDATIONS if accepted else {})
    db = root.parent / "results.db"
    results_db.init_db(db)
    with results_db.connect(db) as connection:
        results_db.insert_row(connection, row)
    (root.parent / "deployment_run.json").write_text(json.dumps({"identity": IDENTITY, "database_path": str(db.resolve())}))
    filtering = root.parent / "filter.csv"
    filtering.write_text("patent,filename,label,reason\nEP1,source.tif,drawing,long_engineering_lines\n")
    args = build_training_manifest._build_parser().parse_args([
        "--db", str(db), "--run-output", str(root.parent), "--filter-manifest", str(filtering),
    ])
    monkeypatch.setattr(build_training_manifest.deployment, "preflight", lambda path: {"identity": IDENTITY})
    monkeypatch.setattr(build_training_manifest, "_curation_reason", lambda row, args: "keep")
    keep, reject, _ = build_training_manifest.build(args)
    assert len(keep) == int(accepted)
    assert len(reject) == int(not accepted)
