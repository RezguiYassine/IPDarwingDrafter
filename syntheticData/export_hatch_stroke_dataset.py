"""Export PatentVec semantic masks as multilabel Stage-2 stroke samples."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .export_training_dataset import _split_for
from .patentvec.hatch_stroke_targets import (
    HATCH_STROKE_LABEL_CONTRACT,
    STRUCTURAL_MASKS,
    TARGET_KEYS,
    build_hatch_stroke_targets,
    validate_hatch_stroke_targets,
)


EXPORTER_VERSION = "patentvec-hatch-stroke-export-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export lossless structural/hachure labels from PatentVec masks."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--support-radius", type=int, default=0)
    parser.add_argument("--validation-modulus", type=int, default=10)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument(
        "--verify-archives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify selected shard sizes and SHA-256 hashes before export.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}")
    return rows


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    extracted = archive.extractfile(name)
    if extracted is None:
        raise ValueError(f"missing archive member: {name}")
    return extracted.read()


def _load_masks(payload: bytes, sample_id: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        masks = {name: np.asarray(archive[name]) for name in archive.files}
    required = {*STRUCTURAL_MASKS, "hatch"}
    missing = sorted(required - set(masks))
    if missing:
        raise ValueError(f"{sample_id}: masks.npz missing {missing}")
    return masks


def _metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "meta" not in data.files:
        raise ValueError("exported sample has no metadata")
    return json.loads(str(np.asarray(data["meta"])))


def _load_existing(path: Path, expected_meta: dict) -> tuple[dict, dict] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            targets = {name: np.asarray(data[name]) for name in TARGET_KEYS}
            metadata = _metadata(data)
        for key, value in expected_meta.items():
            if metadata.get(key) != value:
                return None
        stats = validate_hatch_stroke_targets(targets)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return targets, stats


def _verify_archive(path: Path, inventory: dict) -> None:
    if not path.exists():
        raise ValueError(f"missing input archive: {path}")
    if path.stat().st_size != int(inventory["bytes"]):
        raise ValueError(f"input archive size mismatch: {path}")
    if _sha256(path) != inventory["sha256"]:
        raise ValueError(f"input archive hash mismatch: {path}")


def export_hatch_stroke_dataset(
    input_root: Path,
    output_root: Path,
    *,
    limit: int | None = None,
    support_radius: int = 0,
    validation_modulus: int = 10,
    validation_fold: int = 0,
    verify_archives: bool = True,
    verbose: bool = False,
) -> dict:
    """Export selected compact shards and return the aggregate manifest."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if support_radius < 0:
        raise ValueError("support_radius must be non-negative")
    if validation_modulus < 1:
        raise ValueError("validation_modulus must be at least one")
    if not 0 <= validation_fold < validation_modulus:
        raise ValueError("validation_fold must be within validation_modulus")

    manifest_path = input_root / "manifest.json"
    rows_path = input_root / "manifest.jsonl"
    if not manifest_path.exists() or not rows_path.exists():
        raise ValueError(f"missing compact manifests under {input_root}")
    input_manifest = json.loads(manifest_path.read_text())
    all_rows = sorted(_load_rows(rows_path), key=lambda row: int(row["source_index"]))
    if len(all_rows) != int(input_manifest["count"]):
        raise ValueError("compact manifest count disagrees with manifest rows")
    if len({int(row["source_index"]) for row in all_rows}) != len(all_rows):
        raise ValueError("compact manifest contains duplicate source indices")
    if len({str(row["sample_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("compact manifest contains duplicate sample ids")
    rows = all_rows[:limit]

    output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        (output_root / split).mkdir(parents=True, exist_ok=True)

    input_manifest_sha256 = _sha256(manifest_path)
    task_fingerprint = _fingerprint(
        {
            "exporter_version": EXPORTER_VERSION,
            "input_run_fingerprint": input_manifest["run_fingerprint"],
            "support_radius": int(support_radius),
            "validation_modulus": int(validation_modulus),
            "validation_fold": int(validation_fold),
        }
    )
    inventory = {
        str(item["archive"]): item for item in input_manifest.get("shards", [])
    }
    rows_by_archive: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_archive[str(row["archive"])].append(row)

    exported_rows: list[dict] = []
    sample_counts = Counter()
    pixel_counts = Counter()
    difficulty_counts: dict[str, Counter] = defaultdict(Counter)
    resumed_samples = 0
    generated_samples = 0

    for archive_number, archive_name in enumerate(sorted(rows_by_archive), start=1):
        archive_path = input_root / archive_name
        if archive_name not in inventory:
            raise ValueError(f"archive {archive_name!r} absent from inventory")
        if verify_archives:
            _verify_archive(archive_path, inventory[archive_name])
        elif not archive_path.exists():
            raise ValueError(f"missing input archive: {archive_path}")

        if verbose:
            print(
                f"[{archive_number}/{len(rows_by_archive)}] {archive_name} "
                f"({len(rows_by_archive[archive_name])} samples)"
            )
        with tarfile.open(archive_path, mode="r") as archive:
            for row in rows_by_archive[archive_name]:
                sample_id = str(row["sample_id"])
                source_index = int(row["source_index"])
                split = _split_for(
                    source_index, validation_modulus, validation_fold
                )
                output_path = output_root / split / f"{sample_id}.npz"
                expected_meta = {
                    "label_contract": HATCH_STROKE_LABEL_CONTRACT,
                    "task_fingerprint": task_fingerprint,
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "support_radius": int(support_radius),
                }
                existing = _load_existing(output_path, expected_meta)
                if existing is None:
                    prefix = row.get("member_prefix", row["relative_path"])
                    masks = _load_masks(
                        _member_bytes(archive, f"{prefix}/masks.npz"), sample_id
                    )
                    targets = build_hatch_stroke_targets(
                        masks, support_radius=support_radius
                    )
                    stats = validate_hatch_stroke_targets(targets)
                    metadata = {
                        **expected_meta,
                        "schema_version": "patentvec-hatch-stroke-sample-1.0",
                        "split": split,
                        "difficulty": str(row.get("difficulty", "unknown")),
                        "source_archive": archive_name,
                        "input_support_masks": [*STRUCTURAL_MASKS, "hatch"],
                        "structural_masks": list(STRUCTURAL_MASKS),
                        "hatch_mask": "hatch",
                        "multilabel_crossings": True,
                        "unassigned_policy": "structural",
                        "pixel_counts": stats,
                    }
                    _atomic_npz(
                        output_path,
                        {**targets, "meta": np.asarray(json.dumps(metadata))},
                    )
                    generated_samples += 1
                else:
                    _targets, stats = existing
                    resumed_samples += 1

                row_summary = {
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "split": split,
                    "difficulty": str(row.get("difficulty", "unknown")),
                    "source_archive": archive_name,
                    "path": str(output_path.relative_to(output_root)),
                    "pixel_counts": stats,
                }
                exported_rows.append(row_summary)
                sample_counts[split] += 1
                difficulty_counts[split][row_summary["difficulty"]] += 1
                pixel_counts.update(stats)

    exported_rows.sort(key=lambda row: int(row["source_index"]))
    manifest = {
        "schema_version": "patentvec-hatch-stroke-export-1.0",
        "exporter_version": EXPORTER_VERSION,
        "label_contract": HATCH_STROKE_LABEL_CONTRACT,
        "task_fingerprint": task_fingerprint,
        "input_run_fingerprint": input_manifest["run_fingerprint"],
        "input_manifest_sha256": input_manifest_sha256,
        "input_count": int(input_manifest["count"]),
        "selected_count": len(rows),
        "limit": limit,
        "support_radius": int(support_radius),
        "validation_rule": {
            "field": "blake2b64(source_index)",
            "hash_person": "PatentVecSplit1",
            "modulus": int(validation_modulus),
            "fold": int(validation_fold),
        },
        "sample_counts": dict(sample_counts),
        "difficulty_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(difficulty_counts.items())
        },
        "pixel_counts": dict(pixel_counts),
        "hatch_positive_samples": int(
            sum(row["pixel_counts"]["hatch_pixels"] > 0 for row in exported_rows)
        ),
        "overlap_positive_samples": int(
            sum(row["pixel_counts"]["overlap_pixels"] > 0 for row in exported_rows)
        ),
        "generated_samples_this_run": generated_samples,
        "resumed_samples_this_run": resumed_samples,
        "selected_archives": sorted(rows_by_archive),
    }
    _atomic_jsonl(output_root / "manifest.jsonl", exported_rows)
    _atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = export_hatch_stroke_dataset(
            args.input,
            args.output,
            limit=args.limit,
            support_radius=args.support_radius,
            validation_modulus=args.validation_modulus,
            validation_fold=args.validation_fold,
            verify_archives=args.verify_archives,
            verbose=True,
        )
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
