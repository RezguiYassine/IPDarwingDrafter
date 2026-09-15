from copy import deepcopy
import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from tools import deployment
from tools import batch_run
from tools.batch_run import stage0_handle_references as s0
from tools.batch_run import stage1_preprocess as s1
from tools.batch_run import stage2_stroke_extract as s2


STRICT = {"pipeline": {"deployment": {"strict_models": True}}}


@pytest.mark.parametrize("arguments, message", [
    (["--workers", "0"], "--workers must be positive"),
    (["--filter-manifest", "/nonexistent/patent-filter.csv"], "filter manifest does not exist"),
])
def test_batch_rejects_invalid_execution_arguments(monkeypatch, capsys, arguments, message):
    monkeypatch.setattr(sys, "argv", ["batch_run", *arguments])
    with pytest.raises(SystemExit) as exc:
        batch_run.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.fixture
def deployment_files(tmp_path, monkeypatch):
    config = deepcopy(STRICT)
    weights = []
    for section, key, required in [
        ("puhachov", "weights", True),
        ("stage2", "hachure_cnn_model", True),
        ("sketchcleannet", "weights", False),
    ]:
        path = tmp_path / f"{section}.pth"
        path.write_bytes(section.encode())
        config[section] = {key: path.name}
        weights.append({"config_key": [section, key], "required": required,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    canonical = tmp_path / "config_deploy.yaml"
    canonical.write_text(yaml.safe_dump(config))
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"weights": weights}))
    monkeypatch.setattr(deployment, "CANONICAL_CONFIG", canonical)
    monkeypatch.setattr(deployment, "MODEL_MANIFEST", manifest)
    return canonical, config


def test_preflight_validates_weights_and_records_resolved_configuration(deployment_files):
    canonical, _ = deployment_files
    report = deployment.preflight(canonical)
    assert report["identity"]["implementation_sha256"]
    assert all(row["available"] for row in report["identity"]["weights"].values())
    assert report["effective_config"]["puhachov"]["weights"] == str(canonical.parent / "puhachov.pth")


def test_preflight_rejects_changed_checkpoint_bytes(deployment_files):
    canonical, _ = deployment_files
    (canonical.parent / "puhachov.pth").write_bytes(b"unapproved replacement")
    with pytest.raises(deployment.DeploymentError, match="SHA-256 mismatch"):
        deployment.preflight(canonical)


def test_preflight_rejects_missing_required_weight(deployment_files):
    canonical, _ = deployment_files
    (canonical.parent / "stage2.pth").unlink()
    with pytest.raises(deployment.DeploymentError, match="Required checkpoint is missing"):
        deployment.preflight(canonical)


def test_binary_pipeline_can_preflight_without_optional_cleaner(deployment_files):
    canonical, _ = deployment_files
    (canonical.parent / "sketchcleannet.pth").unlink()
    report = deployment.preflight(canonical)
    assert not report["identity"]["weights"]["sketchcleannet.weights"]["available"]


def test_device_override_is_logged_but_behavior_override_is_rejected(deployment_files):
    canonical, config = deployment_files
    candidate = canonical.parent / "candidate.yaml"
    config["puhachov"]["device"] = "cuda:1"
    config["pipeline"]["workers"] = 2
    candidate.write_text(yaml.safe_dump(config))
    assert deployment.preflight(candidate)["effective_config"]["puhachov"]["device"] == "cuda:1"
    config["puhachov"]["fusion"] = True
    candidate.write_text(yaml.safe_dump(config))
    with pytest.raises(deployment.DeploymentError, match="not the canonical"):
        deployment.preflight(candidate)


def test_resume_requires_matching_code_and_model_identity(tmp_path):
    path, db = tmp_path / "run.json", tmp_path / "results.db"
    report = {"identity": {"config": "a", "implementation": "b"}}
    deployment.record_run(path, report, db)
    deployment.record_run(path, report, db)
    with pytest.raises(deployment.DeploymentError, match="identity changed"):
        deployment.record_run(path, {"identity": {"config": "a", "implementation": "c"}}, db)
    with pytest.raises(deployment.DeploymentError, match="identity changed"):
        deployment.record_run(path, report, tmp_path / "different.db")


def test_legacy_nonempty_database_cannot_be_adopted_as_validated(tmp_path):
    db = tmp_path / "results.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE results (id INTEGER)")
        connection.execute("INSERT INTO results VALUES (1)")
    with pytest.raises(deployment.DeploymentError, match="no deployment manifest"):
        deployment.record_run(tmp_path / "run.json", {"identity": {}}, db)


def _input(tmp_path, *, binary):
    if binary:
        image = np.full((64, 64), 255, dtype=np.uint8)
        image[20, 10:50] = 0
    else:
        image = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
    path = tmp_path / "input.png"
    cv2.imwrite(str(path), image)
    return path


def test_strict_stage1_allows_binary_passthrough_without_cleaner(tmp_path):
    result = s1.run(_input(tmp_path, binary=True), tmp_path, "binary", STRICT, model=None)
    assert result.model_used == "passthrough_binary"


def test_strict_stage0_refuses_missing_ocr_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(s0, "_ocr_available", lambda: False)
    config = {**STRICT, "stage0": {"use_ocr": True}}
    with pytest.raises(RuntimeError, match="requires EasyOCR"):
        s0.run(_input(tmp_path, binary=True), tmp_path, "refs", config)


def test_strict_stage1_refuses_missing_model_for_grayscale(tmp_path):
    with pytest.raises(RuntimeError, match="requires SketchCleanNet"):
        s1.run(_input(tmp_path, binary=False), tmp_path, "gray", STRICT, model=None)


def test_strict_stage1_refuses_inference_fallback(tmp_path):
    def fail(_image):
        raise ValueError("inference failure")
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        s1.run(_input(tmp_path, binary=False), tmp_path, "gray", STRICT,
               model=SimpleNamespace(clean=fail))


def test_strict_stage2_requires_models(tmp_path):
    path = _input(tmp_path, binary=True)
    with pytest.raises(RuntimeError, match="Puhachov model is required"):
        s2.run(path, tmp_path, "graph", STRICT)
    config = {**STRICT, "stage2": {"hachure_use_cnn": True}}
    with pytest.raises(RuntimeError, match="hatch CNN requires"):
        s2.run(path, tmp_path, "graph", config, model=object())


def test_strict_stage2_refuses_keypoint_inference_fallback(tmp_path):
    def fail(*_args):
        raise ValueError("inference failure")
    path = _input(tmp_path, binary=True)
    with pytest.raises(RuntimeError, match="Puhachov inference failed"):
        s2.run(path, tmp_path, "graph", STRICT, model=SimpleNamespace(detect=fail))


def test_strict_stage2_refuses_hatch_inference_fallback(tmp_path, monkeypatch):
    def fail(*_args):
        raise ValueError("inference failure")
    monkeypatch.setattr(s2, "_aligned_hatch_mask", fail)
    path = _input(tmp_path, binary=True)
    config = {**STRICT, "stage2": {"hachure_use_cnn": True, "remove_hachures": True}}
    model = SimpleNamespace(detect=lambda *_args: [])
    with pytest.raises(RuntimeError, match="hatch CNN inference failed"):
        s2.run(path, tmp_path, "graph", config, model=model,
               hatch_model=object(), source_image_path=path)
