from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .patentvec.stage2_targets import (
    ARCHIVE_LABEL_CONTRACT,
    STAGE2_LABEL_CONTRACTS,
    SUPPORTED_RASTER_TOPOLOGY_CONTRACT,
    relabel_puhachov_payload,
)


EXPORTER_VERSION = "patentvec-training-export-1.3"
FREE2CAD_NAMES = ("line", "arc", "circle", "polyline", "bezier")
REQUIRED_STAGE3_KEYS = ("points", "mask", "types", "params")
OPTIONAL_STAGE3_KEYS = (
    "centers",
    "scales",
    "centers_px",
    "scales_px",
    "source_index",
    "primitive_ids",
    "member_primitive_ids",
    "edge_origins",
    "topology_projected",
    "match_p90_px",
    "line_p90_px",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export compact PatentVec tar shards to the native Puhachov and "
            "Free2CAD training layouts."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-modulus", type=int, default=10)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument(
        "--require-topology-projected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject Stage-3 labels that were not projected through Stage 2.",
    )
    parser.add_argument(
        "--stage2-label-contract",
        choices=STAGE2_LABEL_CONTRACTS,
        default=ARCHIVE_LABEL_CONTRACT,
        help=(
            "Keep archived Stage-2 labels or derive a versioned raster-topology "
            "contract from the archived skeleton and semantic masks."
        ),
    )
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Export only Stage-2 NPZ files; skip unchanged Free2CAD shards.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def _load_rows(dataset_root: Path) -> list[dict]:
    path = dataset_root / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"missing compact manifest rows: {path}")
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}")
    return rows


def _split_for(source_index: int, modulus: int, fold: int) -> str:
    if modulus == 1:
        return "validation"
    encoded = int(source_index).to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.blake2b(
        encoded, digest_size=8, person=b"PatentVecSplit1"
    ).digest()
    bucket = int.from_bytes(digest, byteorder="little") % modulus
    return "validation" if bucket == fold else "train"


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    extracted = archive.extractfile(name)
    if extracted is None:
        raise ValueError(f"missing archive member: {name}")
    return extracted.read()


def _validate_puhachov(payload: bytes, sample_id: str) -> dict[str, int]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        missing = {"skeleton", "kps", "meta"} - set(data.files)
        if missing:
            raise ValueError(f"{sample_id}: Puhachov target missing {sorted(missing)}")
        skeleton = np.asarray(data["skeleton"])
        keypoints = np.asarray(data["kps"])
        if skeleton.ndim != 2 or not np.any(skeleton):
            raise ValueError(f"{sample_id}: invalid or empty Puhachov skeleton")
        if keypoints.ndim != 2 or keypoints.shape[1:] != (3,):
            raise ValueError(f"{sample_id}: invalid Puhachov keypoint shape")
        if len(keypoints) and (np.any(keypoints[:, 2] < 0) or np.any(keypoints[:, 2] > 2)):
            raise ValueError(f"{sample_id}: invalid Puhachov keypoint class")
        counts = np.bincount(keypoints[:, 2], minlength=3) if len(keypoints) else np.zeros(3)
        return {
            "endpoint": int(counts[0]),
            "junction": int(counts[1]),
            "corner": int(counts[2]),
            "total": int(len(keypoints)),
        }


def _load_free2cad(payload: bytes, sample_id: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        missing = set(REQUIRED_STAGE3_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{sample_id}: Free2CAD target missing {sorted(missing)}")
        arrays = {
            key: np.asarray(data[key])
            for key in (*REQUIRED_STAGE3_KEYS, *OPTIONAL_STAGE3_KEYS)
            if key in data.files
        }
    count = len(arrays["types"])
    if count == 0:
        raise ValueError(f"{sample_id}: empty Free2CAD edge target")
    for key in REQUIRED_STAGE3_KEYS:
        if len(arrays[key]) != count:
            raise ValueError(f"{sample_id}: inconsistent Free2CAD array {key}")
    if arrays["points"].ndim != 3 or arrays["points"].shape[-1] != 2:
        raise ValueError(f"{sample_id}: invalid Free2CAD point shape")
    if arrays["mask"].shape != arrays["points"].shape[:2]:
        raise ValueError(f"{sample_id}: invalid Free2CAD mask shape")
    if np.any(arrays["types"] > 4):
        raise ValueError(f"{sample_id}: invalid Free2CAD class id")
    return arrays


def _stage3_shard(
    parts: list[tuple[dict, dict[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    if not parts:
        return {}
    shared_keys = set(parts[0][1])
    for _row, arrays in parts[1:]:
        shared_keys.intersection_update(arrays)
    keys = [
        key
        for key in (*REQUIRED_STAGE3_KEYS, *OPTIONAL_STAGE3_KEYS)
        if key in shared_keys
    ]
    payload = {
        key: np.concatenate([arrays[key] for _row, arrays in parts], axis=0)
        for key in keys
    }
    payload["sample_ids"] = np.concatenate(
        [
            np.full(len(arrays["types"]), row["sample_id"])
            for row, arrays in parts
        ]
    )
    payload["sample_source_index"] = np.concatenate(
        [
            np.full(
                len(arrays["types"]), int(row["source_index"]), dtype=np.int64
            )
            for row, arrays in parts
        ]
    )
    payload["edge_index_in_sample"] = np.concatenate(
        [np.arange(len(arrays["types"]), dtype=np.int32) for _row, arrays in parts]
    )
    return payload


def _valid_marker(output: Path, marker_path: Path, fingerprint: str) -> dict | None:
    if not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if marker.get("task_fingerprint") != fingerprint:
        return None
    for item in marker.get("outputs", []):
        path = output / item["path"]
        if not path.exists() or path.stat().st_size != item["bytes"]:
            return None
        if _sha256(path) != item["sha256"]:
            return None
    return marker


def _export_archive(
    dataset_root: Path,
    output: Path,
    archive_name: str,
    rows: list[dict],
    task_fingerprint: str,
    validation_modulus: int,
    validation_fold: int,
    require_topology_projected: bool,
    stage2_label_contract: str = ARCHIVE_LABEL_CONTRACT,
    stage2_only: bool = False,
) -> dict:
    archive_path = dataset_root / archive_name
    archive_stem = archive_path.stem
    output_records = []
    stage3_parts: dict[str, list[tuple[dict, dict[str, np.ndarray]]]] = defaultdict(list)
    split_samples = Counter()
    split_edges = Counter()
    class_counts: dict[str, Counter] = defaultdict(Counter)
    difficulty_counts: dict[str, Counter] = defaultdict(Counter)
    stage2_keypoint_counts = Counter()

    with tarfile.open(archive_path, mode="r") as archive:
        for row in sorted(rows, key=lambda item: int(item["source_index"])):
            sample_id = str(row["sample_id"])
            source_index = int(row["source_index"])
            split = _split_for(source_index, validation_modulus, validation_fold)
            member_prefix = row.get("member_prefix", row["relative_path"])
            puhachov_bytes = _member_bytes(
                archive, f"{member_prefix}/puhachov.npz"
            )
            if stage2_label_contract != ARCHIVE_LABEL_CONTRACT:
                puhachov_bytes = relabel_puhachov_payload(
                    puhachov_bytes,
                    _member_bytes(archive, f"{member_prefix}/masks.npz"),
                    label_contract=stage2_label_contract,
                )
            stage2_keypoint_counts.update(
                _validate_puhachov(puhachov_bytes, sample_id)
            )
            stage2_path = output / "stage2" / split / f"{sample_id}.npz"
            _atomic_bytes(stage2_path, puhachov_bytes)
            output_records.append(
                {
                    "path": str(stage2_path.relative_to(output)),
                    "bytes": len(puhachov_bytes),
                    "sha256": _sha256_bytes(puhachov_bytes),
                }
            )

            split_samples[split] += 1
            difficulty_counts[split][str(row.get("difficulty", "unknown"))] += 1
            if not stage2_only:
                free2cad = _load_free2cad(
                    _member_bytes(archive, f"{member_prefix}/free2cad_edges.npz"),
                    sample_id,
                )
                projected = free2cad.get("topology_projected")
                if (
                    require_topology_projected
                    and (projected is None or not np.all(projected))
                ):
                    raise ValueError(f"{sample_id}: Free2CAD target bypassed Stage 2")
                stage3_parts[split].append((row, free2cad))
                split_edges[split] += len(free2cad["types"])
                class_counts[split].update(
                    int(value) for value in np.asarray(free2cad["types"])
                )

    for split, parts in sorted(stage3_parts.items()):
        trainer_split = "val" if split == "validation" else split
        stage3_path = output / "stage3" / trainer_split / f"{archive_stem}.npz"
        _atomic_npz(stage3_path, _stage3_shard(parts))
        output_records.append(
            {
                "path": str(stage3_path.relative_to(output)),
                "bytes": stage3_path.stat().st_size,
                "sha256": _sha256(stage3_path),
            }
        )

    marker = {
        "schema_version": "patentvec-training-export-marker-1.2",
        "task_fingerprint": task_fingerprint,
        "input_archive": archive_name,
        "stage2_label_contract": stage2_label_contract,
        "stage2_only": stage2_only,
        "stage2_keypoint_counts": dict(stage2_keypoint_counts),
        "sample_counts": dict(split_samples),
        "edge_counts": dict(split_edges),
        "class_counts": {
            split: {
                FREE2CAD_NAMES[class_id]: int(counts.get(class_id, 0))
                for class_id in range(len(FREE2CAD_NAMES))
            }
            for split, counts in class_counts.items()
        },
        "difficulty_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in difficulty_counts.items()
        },
        "outputs": output_records,
    }
    return marker


def _manifest_from_markers(
    input_manifest: dict,
    input_manifest_sha256: str,
    markers: list[dict],
    validation_modulus: int,
    validation_fold: int,
    require_topology_projected: bool,
    stage2_label_contract: str,
    stage2_only: bool,
) -> dict:
    sample_counts = Counter()
    edge_counts = Counter()
    class_counts: dict[str, Counter] = defaultdict(Counter)
    difficulty_counts: dict[str, Counter] = defaultdict(Counter)
    stage2_keypoint_counts = Counter()
    for marker in markers:
        sample_counts.update(marker["sample_counts"])
        edge_counts.update(marker["edge_counts"])
        for split, counts in marker["class_counts"].items():
            class_counts[split].update(counts)
        for split, counts in marker["difficulty_counts"].items():
            difficulty_counts[split].update(counts)
        stage2_keypoint_counts.update(marker.get("stage2_keypoint_counts", {}))
    source_pool_split = input_manifest["split"]
    all_validation = validation_modulus == 1
    source_disjoint = all_validation and source_pool_split in {"validation", "test"}
    split_semantics = (
        f"all samples come from the source-disjoint {source_pool_split} pool"
        if source_disjoint
        else (
            "deterministic composite holdout only; source components may "
            "repeat across train and validation"
        )
    )
    return {
        "schema_version": "patentvec-training-export-1.2",
        "exporter_version": EXPORTER_VERSION,
        "input_run_fingerprint": input_manifest["run_fingerprint"],
        "input_manifest_sha256": input_manifest_sha256,
        "input_count": int(input_manifest["count"]),
        "source_pool_split": source_pool_split,
        "validation_rule": {
            "field": "blake2b64(source_index)",
            "hash_person": "PatentVecSplit1",
            "modulus": validation_modulus,
            "fold": validation_fold,
            "source_disjoint": source_disjoint,
            "semantics": split_semantics,
        },
        "require_topology_projected": require_topology_projected,
        "stage2_label_contract": stage2_label_contract,
        "stage2_only": stage2_only,
        "stage2_keypoint_counts": dict(stage2_keypoint_counts),
        "sample_counts": dict(sample_counts),
        "difficulty_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(difficulty_counts.items())
        },
        "stage3_edge_counts": dict(edge_counts),
        "stage3_class_counts": {
            split: {
                name: int(class_counts[split].get(name, 0))
                for name in FREE2CAD_NAMES
            }
            for split in sorted(class_counts)
        },
        "shards": [
            {key: value for key, value in marker.items() if key != "outputs"}
            for marker in markers
        ],
    }


def main() -> int:
    args = parse_args()
    if args.validation_modulus < 1:
        raise SystemExit("--validation-modulus must be at least 1")
    if not 0 <= args.validation_fold < args.validation_modulus:
        raise SystemExit("--validation-fold must be within the modulus")
    input_manifest_path = args.input / "manifest.json"
    if not input_manifest_path.exists():
        raise SystemExit(f"missing compact dataset manifest: {input_manifest_path}")
    input_manifest = json.loads(input_manifest_path.read_text())
    rows = _load_rows(args.input)
    if len(rows) != int(input_manifest["count"]):
        raise SystemExit(
            f"manifest row count {len(rows)} != dataset count {input_manifest['count']}"
        )
    if len({int(row["source_index"]) for row in rows}) != len(rows):
        raise SystemExit("compact manifest contains duplicate source_index values")
    if not all(row.get("quality", {}).get("accepted") for row in rows):
        raise SystemExit("compact manifest contains samples that failed quality gates")

    args.output.mkdir(parents=True, exist_ok=True)
    paths = [
        args.output / "stage2" / "train",
        args.output / "stage2" / "validation",
        args.output / "markers",
    ]
    if not args.stage2_only:
        paths.extend(
            [args.output / "stage3" / "train", args.output / "stage3" / "val"]
        )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    input_manifest_sha256 = _sha256(input_manifest_path)
    rows_by_archive: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_archive[str(row["archive"])].append(row)
    expected_archives = {str(item["archive"]): item for item in input_manifest["shards"]}
    if set(rows_by_archive) != set(expected_archives):
        raise SystemExit("manifest rows and shard inventory disagree")

    markers = []
    for completed, archive_name in enumerate(sorted(rows_by_archive), start=1):
        archive_path = args.input / archive_name
        expected = expected_archives[archive_name]
        if not archive_path.exists():
            raise SystemExit(f"missing input archive: {archive_path}")
        if archive_path.stat().st_size != int(expected["bytes"]):
            raise SystemExit(f"input archive size mismatch: {archive_path}")
        if _sha256(archive_path) != expected["sha256"]:
            raise SystemExit(f"input archive hash mismatch: {archive_path}")
        task_fingerprint = _fingerprint(
            {
                "exporter_version": EXPORTER_VERSION,
                "input_run_fingerprint": input_manifest["run_fingerprint"],
                "archive": archive_name,
                "archive_sha256": expected["sha256"],
                "validation_modulus": args.validation_modulus,
                "validation_fold": args.validation_fold,
                "require_topology_projected": args.require_topology_projected,
                "stage2_label_contract": args.stage2_label_contract,
                "stage2_only": args.stage2_only,
            }
        )
        marker_path = args.output / "markers" / f"{archive_path.stem}.json"
        marker = _valid_marker(args.output, marker_path, task_fingerprint)
        resumed = marker is not None
        if marker is None:
            marker = _export_archive(
                args.input,
                args.output,
                archive_name,
                rows_by_archive[archive_name],
                task_fingerprint,
                args.validation_modulus,
                args.validation_fold,
                args.require_topology_projected,
                args.stage2_label_contract,
                args.stage2_only,
            )
            _atomic_json(marker_path, marker)
        markers.append(marker)
        suffix = " (verified resume skip)" if resumed else ""
        print(
            f"[{completed}/{len(rows_by_archive)}] {archive_name}"
            f" samples={sum(marker['sample_counts'].values())}{suffix}",
            flush=True,
        )

    manifest = _manifest_from_markers(
        input_manifest,
        input_manifest_sha256,
        markers,
        args.validation_modulus,
        args.validation_fold,
        args.require_topology_projected,
        args.stage2_label_contract,
        args.stage2_only,
    )
    if sum(manifest["sample_counts"].values()) != int(input_manifest["count"]):
        raise RuntimeError("exported sample count does not match compact input")
    _atomic_json(args.output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "difficulty_counts": manifest["difficulty_counts"],
                "sample_counts": manifest["sample_counts"],
                "stage3_class_counts": manifest["stage3_class_counts"],
                "stage3_edge_counts": manifest["stage3_edge_counts"],
                "validation_rule": manifest["validation_rule"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
