"""Verify the canonical deployment and prevent incompatible batch resume."""

from __future__ import annotations

import argparse
from copy import deepcopy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

import yaml


ROOT = Path(__file__).resolve().parent.parent
CANONICAL_CONFIG = ROOT / "config_deploy.yaml"
MODEL_MANIFEST = ROOT / "models/deployment_manifest.json"
IMPLEMENTATION_FILES = (
    "stage0_handling_references/stage0_handle_references.py",
    "stage1_preprocessing/stage1_preprocess.py",
    "stage2_strokeextraction/stage2_stroke_extract.py",
    "stage3_primitivesfitting/stage3_primitive_fit.py",
    "stage4_export/stage4_export.py",
    "stage4_export/export_audit.py",
    "tools/batch_run.py", "tools/deployment.py",
    "tools/acceptance.py", "tools/results_db.py", "tools/build_training_manifest.py",
    "tools/geometry_validation.py",
)


class DeploymentError(RuntimeError):
    pass


def strict_models(config: dict) -> bool:
    return bool((config.get("pipeline", {}).get("deployment", {}) or {}).get("strict_models", False))


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolved_config(path: Path, weights: list[dict]) -> dict:
    config = yaml.safe_load(path.read_text()) or {}
    for entry in weights:
        section, key = entry["config_key"]
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = str((path.parent / value).resolve())
    return config


def _behavior(config: dict) -> dict:
    profile = deepcopy({key: config.get(key, {}) for key in (
        "stage0", "stage1", "stage2", "stage3", "stage4",
        "puhachov", "sketchcleannet", "pipeline",
    )})
    for section, keys in {
        "puhachov": ("device", "devices"), "sketchcleannet": ("device",),
        "stage2": ("hachure_cnn_device", "hachure_cnn_devices"),
        # Where OCR runs is a placement choice like the CNN devices above, not a
        # behavioural one. effective_config_sha256 still hashes the whole
        # resolved config, so a GPU run and a CPU run remain distinguishable in
        # every acceptance record; only the canonical-behaviour test ignores it.
        "stage0": ("ocr_gpu", "ocr_gpu_devices"),
        "pipeline": ("workers", "log_level", "resume", "gpu_devices"),
    }.items():
        for key in keys:
            profile[section].pop(key, None)
    return profile


def preflight(config_path: Path) -> dict:
    """Hash registered weights and require canonical behavior, allowing devices/workers to differ."""
    config_path = config_path.resolve()
    weights = json.loads(MODEL_MANIFEST.read_text())["weights"]
    config = _resolved_config(config_path, weights)
    canonical = _resolved_config(CANONICAL_CONFIG, weights)
    if not strict_models(config) or _behavior(config) != _behavior(canonical):
        raise DeploymentError(
            "Configuration is not the canonical deployment behavior. Use "
            "config_deploy.yaml, or an explicit research configuration without strict deployment."
        )
    verified = {}
    for entry in weights:
        section, key = entry["config_key"]
        value = config.get(section, {}).get(key)
        path = Path(value) if value else None
        exists = path is not None and path.is_file()
        if not exists and entry["required"]:
            raise DeploymentError(f"Required checkpoint is missing: {section}.{key}={value}")
        digest = _file_digest(path) if exists else None
        if digest is not None and digest != entry["sha256"]:
            raise DeploymentError(f"Checkpoint SHA-256 mismatch: {section}.{key}={path}")
        verified[f"{section}.{key}"] = {
            "path": str(path) if path else None,
            "required": entry["required"], "available": exists, "sha256": digest,
        }
    implementation = {name: _file_digest(ROOT / name) for name in IMPLEMENTATION_FILES}
    identity = {
        "effective_config_sha256": _json_digest(config),
        "implementation_sha256": _json_digest(implementation),
        "weight_manifest_sha256": _file_digest(MODEL_MANIFEST),
        "weights": verified,
    }
    return {
        "schema": "ap3-deployment-run-v1",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path), "identity": identity,
        "effective_config": config, "implementation_files": implementation,
        "note": "Preflight checks artifacts and configuration, not reconstruction accuracy.",
    }


def record_run(path: Path, report: dict, database_path: Path) -> None:
    """Keep one implementation/configuration identity per results database."""
    report = {**report, "database_path": str(database_path.resolve())}
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous.get("identity") != report["identity"]
                or previous.get("database_path") != report["database_path"]):
            raise DeploymentError("Deployment identity changed; use a new output directory and database.")
        return
    if database_path.exists():
        with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            table = connection.execute("SELECT name FROM sqlite_master WHERE name='results'").fetchone()
            if table and connection.execute("SELECT 1 FROM results LIMIT 1").fetchone():
                raise DeploymentError("Existing results have no deployment manifest; use a new output directory and database.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CANONICAL_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = preflight(args.config)
    except (DeploymentError, OSError, ValueError) as exc:
        parser.exit(2, f"Deployment preflight failed: {exc}\n")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
