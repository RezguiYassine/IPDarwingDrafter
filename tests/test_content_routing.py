"""Content routing must fail closed: nothing turns an absent decision into a pass."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import content_routing


AUTHORITY = "human_curation"


@pytest.fixture
def source(tmp_path):
    image = np.full((64, 64), 255, np.uint8)
    cv2.line(image, (8, 8), (56, 56), 0, 2)
    path = tmp_path / "F1.tif"
    cv2.imwrite(str(path), image)
    return path


def write_manifest(tmp_path, source, **overrides):
    entry = {"patent_id": "P1", "sketch_id": "F1",
             "source_sha256": content_routing.file_digest(source),
             "routing": "in_scope", "content_class": "drawing",
             "decided_by": "tester", "decided_at": "2026-09-17T00:00:00+00:00"}
    entry.update(overrides)
    path = tmp_path / "decisions.json"
    path.write_text(json.dumps({"schema": content_routing.MANIFEST_SCHEMA,
                                "version": "1", "authority": AUTHORITY,
                                "decisions": [entry]}))
    return content_routing.load_manifest(path)


def decide(tmp_path, source, manifest, channels=None):
    return content_routing.run("P1", "F1", source, tmp_path / "report.json",
                               manifest=manifest, channels=channels)


# ── figure class routing ─────────────────────────────────────────────────────

def test_absent_manifest_is_pending_not_pass(tmp_path, source):
    report = decide(tmp_path, source, None)
    assert report["status"] == "pending"
    assert report["reason_codes"] == ["content_ink_routing_pending",
                                      "content_manifest_absent"]


def test_figure_absent_from_manifest_is_pending(tmp_path, source):
    manifest = write_manifest(tmp_path, source, sketch_id="F9")
    report = decide(tmp_path, source, manifest)
    assert report["class_decision"]["status"] == "pending"
    assert "content_decision_missing" in report["reason_codes"]


def test_in_scope_decision_passes_the_class_check(tmp_path, source):
    report = decide(tmp_path, source, write_manifest(tmp_path, source))
    assert report["class_decision"]["status"] == "pass"


def test_out_of_scope_decision_fails(tmp_path, source):
    manifest = write_manifest(tmp_path, source, routing="out_of_scope",
                              reason="flowchart")
    report = decide(tmp_path, source, manifest)
    assert report["class_decision"]["status"] == "fail"
    assert report["status"] == "fail"          # outranks the pending routing
    assert "content_class_flowchart" in report["reason_codes"]


def test_decision_does_not_transfer_to_a_different_image(tmp_path, source):
    manifest = write_manifest(tmp_path, source)
    replacement = np.full((64, 64), 255, np.uint8)
    cv2.circle(replacement, (32, 32), 20, 0, 2)
    cv2.imwrite(str(source), replacement)
    report = decide(tmp_path, source, manifest)
    assert report["class_decision"]["status"] == "review"
    assert "content_source_changed" in report["reason_codes"]


@pytest.mark.parametrize("mutation, message", [
    ({"schema": "something-else"}, "schema"),
    ({"authority": ""}, "authority"),
    ({"decisions": []}, "no decisions"),
])
def test_malformed_manifest_raises_rather_than_degrading(tmp_path, mutation, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": content_routing.MANIFEST_SCHEMA,
                                "authority": AUTHORITY,
                                "decisions": [{"patent_id": "P1", "sketch_id": "F1",
                                               "source_sha256": "abc",
                                               "routing": "in_scope"}],
                                **mutation}))
    with pytest.raises(ValueError, match=message):
        content_routing.load_manifest(path)


def test_duplicate_and_unbound_decisions_raise(tmp_path, source):
    entry = {"patent_id": "P1", "sketch_id": "F1", "routing": "in_scope",
             "source_sha256": "abc"}
    path = tmp_path / "dup.json"
    path.write_text(json.dumps({"schema": content_routing.MANIFEST_SCHEMA,
                                "authority": AUTHORITY, "decisions": [entry, entry]}))
    with pytest.raises(ValueError, match="Duplicate"):
        content_routing.load_manifest(path)
    path.write_text(json.dumps({"schema": content_routing.MANIFEST_SCHEMA,
                                "authority": AUTHORITY,
                                "decisions": [{**entry, "source_sha256": ""}]}))
    with pytest.raises(ValueError, match="source hash"):
        content_routing.load_manifest(path)


# ── source ink routing ───────────────────────────────────────────────────────

def channels_for(tmp_path, source, *, drop_region=None):
    """Stage-1 output that retains the source ink, optionally erasing a region."""
    image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    cleaned = np.where(image < 128, 255, 0).astype(np.uint8)
    if drop_region:
        x, y, w, h = drop_region
        cleaned[y:y + h, x:x + w] = 0
    path = tmp_path / "cleaned.png"
    cv2.imwrite(str(path), cleaned)
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"removed_hachures": []}))
    return {"cleaned": path, "graph": graph}


def test_retained_ink_routes_completely(tmp_path, source):
    report = decide(tmp_path, source, write_manifest(tmp_path, source),
                    channels_for(tmp_path, source))
    assert report["ink_routing"]["status"] == "pass"
    assert report["ink_routing"]["unrouted_pixels"] == 0
    assert report["status"] == "pass"


def test_silently_dropped_content_is_caught(tmp_path, source):
    channels = channels_for(tmp_path, source, drop_region=(20, 20, 30, 30))
    report = decide(tmp_path, source, write_manifest(tmp_path, source), channels)
    assert report["ink_routing"]["status"] == "review"
    assert report["reason_codes"] == ["content_unrouted_ink"]
    assert report["ink_routing"]["largest_unrouted_component"] > 0
    assert report["status"] == "review"        # class passed, routing did not


def test_hachure_removal_counts_as_routed(tmp_path, source):
    channels = channels_for(tmp_path, source, drop_region=(20, 20, 30, 30))
    # Declare the same diagonal run as a removed hachure.
    Path(channels["graph"]).write_text(json.dumps({"removed_hachures": [
        {"pixels": [[x, x] for x in range(8, 57)]}]}))
    report = decide(tmp_path, source, write_manifest(tmp_path, source), channels)
    assert report["ink_routing"]["status"] == "pass"
    assert report["ink_routing"]["channels"]["stage2_hachures"] > 0


def test_missing_channels_keep_routing_pending(tmp_path, source):
    report = decide(tmp_path, source, write_manifest(tmp_path, source), channels=None)
    assert report["ink_routing"]["status"] == "pending"
    assert report["status"] == "pending"


def test_report_binds_every_input_by_hash(tmp_path, source):
    channels = channels_for(tmp_path, source)
    report = decide(tmp_path, source, write_manifest(tmp_path, source), channels)
    assert report["inputs"]["source"]["sha256"] == content_routing.file_digest(source)
    for name in ("cleaned", "graph"):
        assert report["inputs"][name]["sha256"] == content_routing.file_digest(channels[name])
    assert content_routing.acceptance_check(report) == {
        "status": "pass", "reason_codes": [],
        "validator": content_routing.VALIDATOR, "version": content_routing.VERSION}
