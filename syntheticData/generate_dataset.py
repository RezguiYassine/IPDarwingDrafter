from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from syntheticData.patentvec.generator import (
    ComplexPilotGenerator,
    SourcePool,
    compact_sample_payload,
)
from syntheticData.patentvec.render import degradation_parameters
from syntheticData.patentvec.stage2_targets import ARCHIVE_LABEL_CONTRACT, REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_ROOT = Path("/media/safe/secondary disk/IPdrawings")
GENERATOR_VERSION = "patentvec-generator-2.3"
_WORKER_GENERATOR: ComplexPilotGenerator | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate restart-safe compact PatentVec training shards."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "PatentVecScaleProbe1000",
    )
    parser.add_argument("--mode", choices=("probe", "experiment", "full"), default="probe")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--canvas", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=280725)
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--audit-strategy", choices=("periodic", "stratified"), default="periodic")
    parser.add_argument("--retain-rasters", action="store_true")
    parser.add_argument("--degradation-config", type=Path)
    parser.add_argument("--acquisition", choices=("grayscale", "binary"), default="grayscale")
    parser.add_argument("--noise-budget", type=float, help="Expected dark speckles per 1000 reference-free ink pixels")
    parser.add_argument("--layout", choices=("single", "four_figure"), default="single")
    parser.add_argument("--stage2-label-contract", choices=(ARCHIVE_LABEL_CONTRACT, REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT),
                        default=ARCHIVE_LABEL_CONTRACT)
    parser.add_argument("--max-sample-attempts", type=int, default=128)
    parser.add_argument(
        "--curriculum",
        choices=("baseline", "enhanced", "very_hard", "balanced", "patent"),
        default="baseline",
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument(
        "--sketchgraphs",
        type=Path,
        default=PROJECT_ROOT / "data" / "SketchGraphs" / "raw" / "sg_t16_train.npy",
    )
    parser.add_argument(
        "--cadvg-root", type=Path, default=PROJECT_ROOT / "data" / "Drawing2CAD"
    )
    parser.add_argument(
        "--source-index",
        type=Path,
        default=PROJECT_ROOT / "output" / "PatentVecSourceIndex" / "train.json",
    )
    parser.add_argument("--confirm-full-generation", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    caps = {"probe": 1000, "experiment": 10000, "full": 50000}
    if not 1 <= args.count <= caps[args.mode]:
        raise SystemExit(f"{args.mode} mode count must be in [1, {caps[args.mode]}]")
    if args.mode == "full" and not args.confirm_full_generation:
        raise SystemExit("full mode requires --confirm-full-generation")
    if not 1 <= args.workers <= 24:
        raise SystemExit("workers must be in [1, 24]")
    if not 1 <= args.shard_size <= 1000:
        raise SystemExit("shard-size must be in [1, 1000]")
    if args.audit_every < 0:
        raise SystemExit("audit-every must be non-negative")
    if not 16 <= args.max_sample_attempts <= 512:
        raise SystemExit("max-sample-attempts must be in [16, 512]")
    if not args.source_index.exists():
        raise SystemExit(f"missing source index: {args.source_index}")
    if not 128 <= args.canvas <= 4096:
        raise SystemExit("canvas must be in [128, 4096]")
    overrides = yaml.safe_load(args.degradation_config.read_text()) if args.degradation_config else None
    args.degradation = degradation_parameters(overrides)
    if args.noise_budget is not None and not 0 <= args.noise_budget <= 10:
        raise SystemExit("noise-budget must be in [0, 10]")
    if args.layout == "four_figure" and args.canvas < 1536:
        raise SystemExit("four-figure sheets require canvas >=1536")


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _worker_init(
    sketchgraphs: str,
    cadvg_root: str,
    source_index: str,
    split: str,
    canvas: int,
    layout: str = "single",
) -> None:
    global _WORKER_GENERATOR
    pool = SourcePool(
        Path(sketchgraphs),
        Path(cadvg_root),
        split=split,
        index_path=Path(source_index),
    )
    if layout == "four_figure":
        from syntheticData.patentvec.sheets import SheetGenerator
        _WORKER_GENERATOR = SheetGenerator(pool, canvas)
    else:
        _WORKER_GENERATOR = ComplexPilotGenerator(pool, canvas=canvas)


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))


def _generate_shard(task: dict) -> dict:
    generator = _WORKER_GENERATOR
    if generator is None:
        raise RuntimeError("PatentVec worker was not initialized")
    output = Path(task["output"])
    shard_path = output / "shards" / task["shard_name"]
    marker_path = output / "markers" / f"{shard_path.stem}.json"
    temporary = shard_path.with_suffix(shard_path.suffix + f".{os.getpid()}.tmp")
    rows = []
    started = time.monotonic()
    try:
        with tarfile.open(temporary, mode="w") as archive:
            for spec in task["samples"]:
                prepared = generator.generate_prepared(
                    sample_id=spec["sample_id"],
                    seed=spec["seed"],
                    difficulty=spec["difficulty"],
                    max_attempts=task["max_sample_attempts"],
                )
                payload, row = compact_sample_payload(
                    prepared,
                    source_index=spec["source_index"],
                    audit=spec["audit"],
                    retain_rasters=task["retain_rasters"],
                    degradation=task["degradation"],
                    label_contract=task["stage2_label_contract"],
                    acquisition=task["acquisition"], noise_budget=task["noise_budget"],
                )
                prefix = f"samples/{spec['sample_id']}"
                for filename in sorted(payload):
                    _tar_add_bytes(
                        archive, f"{prefix}/{filename}", payload[filename]
                    )
                row["archive"] = f"shards/{task['shard_name']}"
                row["member_prefix"] = prefix
                rows.append(row)
        temporary.replace(shard_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    marker = {
        "schema_version": "patentvec-shard-marker-1.1",
        "task_fingerprint": task["task_fingerprint"],
        "shard_index": task["shard_index"],
        "archive": f"shards/{task['shard_name']}",
        "sha256": _sha256(shard_path),
        "bytes": shard_path.stat().st_size,
        "sample_count": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }
    _atomic_json(marker_path, marker)
    return marker


def _valid_marker(
    output: Path, shard_name: str, task_fingerprint: str
) -> dict | None:
    marker_path = output / "markers" / f"{Path(shard_name).stem}.json"
    shard_path = output / "shards" / shard_name
    if not marker_path.exists() or not shard_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("task_fingerprint") != task_fingerprint:
        return None
    if marker.get("bytes") != shard_path.stat().st_size:
        return None
    if marker.get("sha256") != _sha256(shard_path):
        return None
    return marker


def _sample_specs(args: argparse.Namespace) -> list[dict]:
    schedules = {
        "baseline": ("medium", "hard"),
        "enhanced": ("medium", "hard", "hard", "very_hard"),
        "very_hard": ("very_hard",),
        "balanced": ("medium", "hard", "very_hard"),
        "patent": ("medium", "hard", "very_hard", "very_hard", "very_hard"),
    }
    schedule = schedules[args.curriculum]
    specs = []
    for index in range(args.count):
        difficulty = schedule[index % len(schedule)]
        specs.append(
            {
                "source_index": index,
                "sample_id": f"pv_{args.split}_{index:07d}_{difficulty}",
                "seed": int(args.seed + index * 1009),
                "difficulty": difficulty,
                "audit": bool(args.audit_every and index % args.audit_every == 0),
            }
        )
    if args.audit_every and args.audit_strategy == "stratified":
        for spec in specs:
            spec["audit"] = False
        for difficulty in sorted(set(schedule)):
            candidates = [spec for spec in specs if spec["difficulty"] == difficulty]
            count = (len(candidates) + args.audit_every - 1) // args.audit_every
            positions = ([len(candidates) // 2] if count == 1 else
                         [round(i * (len(candidates)-1) / (count-1)) for i in range(count)])
            for position in positions:
                candidates[position]["audit"] = True
    return specs


def _record_generation(output: Path, plan: dict) -> None:
    path = output / "generation.json"
    previous = path if path.exists() else output / "manifest.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if saved.get("run_fingerprint") != plan["run_fingerprint"]:
            raise ValueError("Generation identity changed; use a new output directory")
    elif any((output / "shards").glob("*.tar")):
        raise ValueError("Existing shards have no generation identity; use a new output directory")
    if not path.exists():
        _atomic_json(path, plan)


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


def _summary(rows: list[dict]) -> dict:
    difficulties = Counter()
    interactions = Counter()
    semantics = Counter()
    keypoints = Counter()
    edges = Counter()
    polyline_origins = Counter()
    attempts = []
    placement_rejections = []
    match_residuals = []
    topology_projected = 0
    for row in rows:
        difficulties[row["difficulty"]] += 1
        interactions.update(row["quality"]["interaction_counts"])
        semantics.update(row["quality"]["semantic_counts"])
        keypoints.update(row["training_targets"]["puhachov_keypoints"])
        edge_targets = row["training_targets"]["free2cad_edges"]
        edges.update(
            {
                key: edge_targets[key]
                for key in ("line", "arc", "circle", "polyline", "bezier")
            }
        )
        polyline_origins.update(
            {
                "direct": edge_targets["polyline_direct"],
                "aggregated_lines": edge_targets["polyline_aggregated_lines"],
            }
        )
        topology_projected += int(edge_targets["topology_projected"])
        match_residuals.append(edge_targets["match_p90_px_max"])
        attempts.append(row["generation"]["attempt"])
        placement_rejections.append(
            row["generation"]["placement_rejection_count"]
        )
    return {
        "difficulty_counts": dict(difficulties),
        "interaction_counts": dict(interactions),
        "semantic_counts": dict(semantics),
        "puhachov_keypoint_counts": dict(keypoints),
        "free2cad_edge_counts": dict(edges),
        "free2cad_polyline_origins": dict(polyline_origins),
        "free2cad_topology_projected_samples": topology_projected,
        "free2cad_match_p90_px_max_per_sample": _distribution(match_residuals),
        "generation_attempts": _distribution(attempts),
        "placement_rejections": _distribution(placement_rejections),
        "payload_bytes": int(sum(row["payload_bytes"] for row in rows)),
        "audit_samples": int(sum(row["audit"] for row in rows)),
    }


def _run(args: argparse.Namespace) -> int:
    (args.output / "shards").mkdir(exist_ok=True)
    (args.output / "markers").mkdir(exist_ok=True)
    specs = _sample_specs(args)
    source_files = [Path(__file__).resolve(), *sorted((PROJECT_ROOT / "syntheticData/patentvec").glob("*.py")),
                    PROJECT_ROOT / "tools/d2c_stage3_dataset.py",
                    PROJECT_ROOT / "stage2_strokeextraction/stage2_stroke_extract.py"]
    settings = {
            "generator_version": GENERATOR_VERSION,
            "mode": args.mode,
            "count": args.count,
            "canvas": args.canvas,
            "seed": args.seed,
            "audit_every": args.audit_every,
            "max_sample_attempts": args.max_sample_attempts,
            "curriculum": args.curriculum,
            "split": args.split,
            "sketchgraphs": str(args.sketchgraphs.resolve()),
            "cadvg_root": str(args.cadvg_root.resolve()),
            "source_index_sha256": _sha256(args.source_index),
            "shard_size": args.shard_size,
            "retain_rasters": args.retain_rasters,
            "audit_strategy": args.audit_strategy,
            "degradation": args.degradation,
            "acquisition": args.acquisition, "noise_budget": args.noise_budget, "layout": args.layout,
            "stage2_label_contract": args.stage2_label_contract,
            "implementation": {str(p.relative_to(PROJECT_ROOT)): _sha256(p) for p in source_files},
            "runtime": {"python": platform.python_version(), **{
                name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-image", "CairoSVG", "shapely", "Pillow")}},
    }
    run_fingerprint = _fingerprint(settings)
    _record_generation(args.output, {"run_fingerprint": run_fingerprint, "settings": settings,
                                    "release_acceptance": "not_evaluated"})
    total_shards = (len(specs) + args.shard_size - 1) // args.shard_size
    tasks = []
    markers: dict[int, dict] = {}
    for shard_index, offset in enumerate(range(0, len(specs), args.shard_size)):
        shard_name = f"shard_{shard_index:06d}.tar"
        shard_specs = specs[offset : offset + args.shard_size]
        task_fingerprint = _fingerprint(
            {
                "run_fingerprint": run_fingerprint,
                "shard_index": shard_index,
                "samples": shard_specs,
                "max_sample_attempts": args.max_sample_attempts,
                "retain_rasters": args.retain_rasters,
                "degradation": args.degradation,
                "acquisition": args.acquisition, "noise_budget": args.noise_budget,
                "stage2_label_contract": args.stage2_label_contract,
            }
        )
        existing = _valid_marker(args.output, shard_name, task_fingerprint)
        if existing is not None:
            markers[shard_index] = existing
            print(
                f"[{len(markers)}/{(len(specs) + args.shard_size - 1) // args.shard_size}] "
                f"{shard_name} (verified resume skip)",
                flush=True,
            )
            continue
        tasks.append(
            {
                "output": str(args.output),
                "shard_index": shard_index,
                "shard_name": shard_name,
                "samples": shard_specs,
                "task_fingerprint": task_fingerprint,
                "max_sample_attempts": args.max_sample_attempts,
                "retain_rasters": args.retain_rasters,
                "degradation": args.degradation,
                "acquisition": args.acquisition, "noise_budget": args.noise_budget,
                "stage2_label_contract": args.stage2_label_contract,
            }
        )

    started = time.monotonic()
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks)),
            initializer=_worker_init,
            initargs=(
                str(args.sketchgraphs),
                str(args.cadvg_root),
                str(args.source_index),
                args.split,
                args.canvas,
                args.layout,
            ),
        ) as executor:
            futures = {executor.submit(_generate_shard, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                marker = future.result()
                markers[int(marker["shard_index"])] = marker
                print(
                    f"[{len(markers)}/{total_shards}] "
                    f"{task['shard_name']} samples={marker['sample_count']} "
                    f"elapsed={marker['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    ordered_markers = [markers[index] for index in sorted(markers)]
    rows = [row for marker in ordered_markers for row in marker["rows"]]
    elapsed = time.monotonic() - started
    summary = _summary(rows)
    generated_sample_count = sum(len(task["samples"]) for task in tasks)
    manifest = {
        "schema_version": "patentvec-compact-dataset-1.0",
        "mode": args.mode,
        "generator_version": GENERATOR_VERSION,
        "run_fingerprint": run_fingerprint,
        "curriculum": args.curriculum,
        "full_generation_started": args.mode == "full",
        "planned_full_data_root": str(EXTERNAL_DATA_ROOT),
        "split": args.split,
        "count": len(rows),
        "requested_count": args.count,
        "canvas": args.canvas,
        "seed": args.seed,
        "workers": args.workers,
        "shard_size": args.shard_size,
        "audit_every": args.audit_every,
        "max_sample_attempts": args.max_sample_attempts,
        "source_index": str(args.source_index),
        "settings": settings,
        "raster_retention": "all" if args.retain_rasters else "audit_only",
        "stage2_label_contract": args.stage2_label_contract,
        "release_acceptance": "not_evaluated",
        "elapsed_seconds_this_run": elapsed,
        "generated_samples_this_run": generated_sample_count,
        "samples_per_second_this_run": (
            generated_sample_count / max(elapsed, 1e-9)
            if generated_sample_count
            else 0.0
        ),
        "archive_bytes": int(sum(marker["bytes"] for marker in ordered_markers)),
        "all_quality_gates_passed": all(
            row["quality"]["accepted"] for row in rows
        ),
        "summary": summary,
        "shards": [
            {key: value for key, value in marker.items() if key != "rows"}
            for marker in ordered_markers
        ],
    }
    _atomic_json(args.output / "manifest.json", manifest)
    with (args.output / "manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {args.output / 'manifest.json'}")
    return 0


def main() -> int:
    args = parse_args()
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".generation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another generator owns this output directory") from exc
        return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
