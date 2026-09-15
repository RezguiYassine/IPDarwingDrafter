import csv
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from tools import build_pipeline_review as review
from tools.render_review_diagnostics import render_one


SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="60"><path d="M5 10 L30 40" stroke="black"/></svg>'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if isinstance(data, (dict, list)) else data)
    return path


@pytest.fixture
def cohort(tmp_path):
    worklist = tmp_path / "samples.csv"
    with worklist.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("patent_id", "sketch_id", "input_path"))
        writer.writeheader()
        for patent in ("A", "B"):
            source = tmp_path / (patent+" original.tif")
            Image.new("L", (40, 60), 255).save(source)
            writer.writerow(dict(patent_id=patent, sketch_id="F0001", input_path=str(source)))
    current, previous = tmp_path / "current", tmp_path / "previous"
    current.mkdir()
    with sqlite3.connect(current / "results.db") as db:
        db.execute("CREATE TABLE results (patent_id TEXT, sketch_id TEXT, status TEXT, acceptance_status TEXT, error TEXT, acceptance_reason_codes TEXT)")
        db.execute("INSERT INTO results VALUES ('A','F0001','quality_gate_stage2','rejected','fragmented','[\"s2_flagged\"]')")
        db.execute("INSERT INTO results VALUES ('B','F0001','ok','review',NULL,'[\"content_validation_pending\"]')")
    write(previous / "summary.json", {"rows": [{"folder": "A", "sketch_id": "F0001", "execution": "ok", "after": "fail"}]})
    write(previous / "A/vectors/F0001.svg", SVG)
    write(current / "B/vectors/F0001.svg", SVG)
    diagnostic = write(current / "review_diagnostics/A/vectors/F0001.svg", SVG)
    write(current / "review_diagnostics/A/F0001_review.json", {
        "status": "rendered", "diagnostic_only": True, "training_eligible": False, "svg": str(diagnostic)})
    icons = tmp_path / "icons"
    for name in ("arrow-left", "arrow-right", "zoom-in", "zoom-out", "scan", "download", "external-link"):
        write(icons / (name+".svg"), SVG)
    return worklist, current, previous, tmp_path / "viewer", icons


def test_review_keeps_missing_previous_rows_and_distinguishes_diagnostics(cohort):
    doc = review.build(*cohort)
    assert len(doc["figures"]) == 2
    assert doc["summary"]["completed"] == 2
    assert doc["summary"]["current_exports"] == 1
    assert doc["summary"]["current_diagnostics"] == 1
    assert doc["summary"]["previous_outputs"] == 1
    a, b = doc["figures"]
    assert a["current"]["status"] == "quality_gate_stage2"
    assert a["current"]["views"]["diagnostic"]["label"] == "Diagnostic only"
    assert a["current"]["acceptance"] == "rejected"
    assert a["current"]["acceptance_reasons"] == ["s2_flagged"]
    assert b["previous"]["status"] == "not_saved"
    assert b["previous"]["views"] == {}
    assert a["original"]["size"] == [40, 60]
    assert (cohort[3] / a["original"]["src"]).exists()
    assert "__REVIEW_DATA__" not in (cohort[3] / "index.html").read_text()


def test_incomplete_cohort_cannot_be_published_as_complete(cohort):
    with sqlite3.connect(cohort[1] / "results.db") as db:
        db.execute("DELETE FROM results WHERE patent_id='B'")
    with pytest.raises(ValueError, match="incomplete"):
        review.build(*cohort)
    doc = review.build(*cohort, allow_partial=True)
    assert doc["summary"]["completed"] == 1
    assert doc["figures"][1]["current"]["status"] == "pending"
    assert not doc["figures"][1]["current"]["canonical_export"]


def test_partial_svg_from_failed_export_is_not_counted_as_canonical_success(cohort):
    with sqlite3.connect(cohort[1] / "results.db") as db:
        db.execute("UPDATE results SET status='stage4' WHERE patent_id='B'")
    doc = review.build(*cohort)
    assert doc["summary"]["current_exports"] == 0
    assert doc["figures"][1]["current"]["views"]["vector"]["label"] == "Partial SVG"


def test_live_review_retains_drawing_when_graph_write_is_incomplete(cohort):
    write(cohort[1] / "A/graphs/F0001_graph.json", '{"edges": [')
    doc = review.build(*cohort)
    assert len(doc["figures"]) == 2
    assert doc["summary"]["graph_preview_errors"] == 1
    assert "JSONDecodeError" in doc["figures"][0]["current"]["graph_preview_error"]
    assert "diagnostic" in doc["figures"][0]["current"]["views"]


def test_inline_manifest_cannot_break_out_of_script():
    value = {"error": '</script><script>alert("bad")</script>&'}
    encoded = review.inline_json(value)
    assert "<" not in encoded
    assert json.loads(encoded) == value


@pytest.mark.parametrize("damage", ["duplicate", "missing", "unsafe"])
def test_worklist_validation(cohort, damage):
    worklist = cohort[0]
    with worklist.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if damage == "duplicate":
        rows.append(rows[0])
    elif damage == "missing":
        rows[0]["input_path"] += "not_found"
    else:
        rows[0]["patent_id"] = "../escape"
    with worklist.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        review.load_worklist(worklist)


def test_cached_assets_are_rebuilt_when_source_changes(cohort):
    first = review.build(*cohort)
    old = first["figures"][0]["original"]["source_sha256"]
    image = Path(review.load_worklist(cohort[0])[0]["input_path"])
    Image.new("L", (40, 60), 0).save(image)
    second = review.build(*cohort)
    assert second["figures"][0]["original"]["source_sha256"] != old
    with Image.open(cohort[3] / second["figures"][0]["original"]["src"]) as saved:
        assert saved.getpixel((0, 0)) == (0, 0, 0)


def test_diagnostic_missing_graph_is_explicit_and_not_eligible(tmp_path):
    config = write(tmp_path / "config.yaml", "{}")
    row = dict(patent_id="A", sketch_id="F0001", status="quality_gate_stage1", error="bad skeleton")
    result = render_one((tmp_path, config, row))
    assert result["status"] == "unavailable"
    assert result["training_eligible"] is False
    assert not list(tmp_path.rglob("*.svg"))


def test_diagnostic_export_never_modifies_canonical_graph_or_primitives(tmp_path):
    config = write(tmp_path / "config.yaml", "{}")
    graph = write(tmp_path / "A/graphs/F0001_graph.json", {"edges": []})
    primitives = write(tmp_path / "A/primitives/F0001_primitives.json", {
        "sketch_id": "F0001", "image_size": [40, 60],
        "primitives": [{"type": "line", "p1": [5, 10], "p2": [30, 40], "confidence": 0.2}]})
    hashes = [review.digest(p) for p in (graph, primitives)]
    row = dict(patent_id="A", sketch_id="F0001", status="quality_gate_stage3", error="low confidence")
    result = render_one((tmp_path, config, row))
    assert result["status"] == "rendered"
    assert result["canonical_status"] == "quality_gate_stage3"
    assert result["diagnostic_only"] and not result["training_eligible"]
    assert [review.digest(p) for p in (graph, primitives)] == hashes
    assert not (tmp_path / "A/vectors").exists()
    assert Path(result["svg"]).is_relative_to(tmp_path / "review_diagnostics")


def test_larger_diagnostic_budget_does_not_change_canonical_status(tmp_path):
    config = write(tmp_path / "config.yaml", "{}")
    write(tmp_path / "A/graphs/F0001_graph.json", {"edges": [{"pixels": []}, {"pixels": []}]})
    write(tmp_path / "A/primitives/F0001_primitives.json", {
        "sketch_id": "F0001", "image_size": [40, 60],
        "primitives": [{"type": "line", "p1": [5, 10], "p2": [30, 40], "confidence": 0.2}]})
    row = dict(patent_id="A", sketch_id="F0001", status="quality_gate_stage2", error="fragmented")
    job = (tmp_path, config, row)
    limited = render_one(job, max_strokes=1)
    assert limited["status"] == "unavailable"
    assert limited["source_strokes"] == 2
    retried = render_one(job, max_strokes=2)
    assert retried["status"] == "rendered"
    assert retried["canonical_status"] == row["status"]
    assert retried["budget"]["max_strokes"] == 2
    assert retried["diagnostic_only"] and not retried["training_eligible"]
    assert not list((tmp_path / "review_diagnostics").rglob("*.tmp"))


def test_diagnostic_bounds_one_path_even_when_total_graph_is_small(tmp_path):
    config = write(tmp_path / "config.yaml", "{}")
    write(tmp_path / "A/graphs/F0001_graph.json", {"edges": [{"pixels": [[0, 0]]*21}]})
    row = dict(patent_id="A", sketch_id="F0001", status="quality_gate_stage2", error="fragmented")
    result = render_one((tmp_path, config, row), max_trace_points=20)
    assert result["status"] == "unavailable"
    assert result["reason"] == "Diagnostic trace budget exceeded"
    assert result["longest_trace_points"] == 21
    assert not list(tmp_path.rglob("*.svg"))
