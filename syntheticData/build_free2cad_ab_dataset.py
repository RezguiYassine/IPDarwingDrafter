from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.build_free2cad_mixed_dataset import (
    CMD_TYPES,
    _proportional_allocation,
    _training_visible_arrays,
    scan_supply,
    total_supply,
)


DOMAIN_BASE = 0
DOMAIN_PATENTVEC = 1
ARRAY_KEYS = (
    "points",
    "mask",
    "types",
    "params",
    "source_index",
    "source_domain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired, equal-budget Free2CAD datasets that differ only "
            "in their PatentVec baseline/enhanced synthetic curriculum."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--synthetic-a", type=Path, required=True)
    parser.add_argument("--synthetic-b", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--synthetic-fraction", type=float, default=0.20)
    parser.add_argument("--max-synthetic-repeats", type=float, default=2.0)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=0,
        help="0 uses the minimum trainer-visible class supply in the base corpus.",
    )
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=850725)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def paired_allocation(
    base_supply: np.ndarray,
    synthetic_a_supply: np.ndarray,
    synthetic_b_supply: np.ndarray,
    target_per_class: int,
    synthetic_fraction: float,
    max_synthetic_repeats: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    if target_per_class <= 0:
        target_per_class = int(np.min(base_supply))
    if target_per_class <= 0:
        raise ValueError("base corpus must supply every Free2CAD class")
    desired = int(round(target_per_class * synthetic_fraction))
    paired_supply = np.minimum(synthetic_a_supply, synthetic_b_supply)
    synthetic = np.minimum(
        desired,
        np.floor(paired_supply * max_synthetic_repeats).astype(np.int64),
    )
    base = target_per_class - synthetic
    if np.any(base > base_supply):
        classes = [
            name
            for name, class_id in CMD_TYPES.items()
            if base[class_id] > base_supply[class_id]
        ]
        raise ValueError(f"base corpus cannot fill classes: {classes}")
    return target_per_class, base, synthetic


def _select_arrays(shards, targets: np.ndarray, seed: int, domain: int):
    allocations = {
        class_id: _proportional_allocation(
            np.asarray([shard.counts[class_id] for shard in shards]),
            int(targets[class_id]),
        )
        for class_id in range(len(CMD_TYPES))
        if targets[class_id] > 0
    }
    rngs = {
        class_id: np.random.default_rng(
            np.random.SeedSequence([seed, domain, class_id])
        )
        for class_id in allocations
    }
    parts = {key: [] for key in ARRAY_KEYS}
    for shard_index, shard in enumerate(shards):
        requested = {
            class_id: int(allocation[shard_index])
            for class_id, allocation in allocations.items()
            if allocation[shard_index] > 0
        }
        if not requested:
            continue
        with np.load(shard.path, allow_pickle=False) as data:
            keep, types, params, _removed, _relabeled = _training_visible_arrays(data)
            points = np.asarray(data["points"])
            mask = np.asarray(data["mask"])
            source_index = (
                np.asarray(data["source_index"], dtype=np.int64)
                if "source_index" in data.files
                else np.arange(len(types), dtype=np.int64)
            )
            for class_id, count in requested.items():
                rows = np.flatnonzero(keep & (types == class_id))
                if not len(rows):
                    raise RuntimeError(
                        f"allocation selected empty class {class_id} in {shard.path}"
                    )
                selected = rngs[class_id].choice(
                    rows, size=count, replace=count > len(rows)
                )
                parts["points"].append(points[selected])
                parts["mask"].append(mask[selected])
                parts["types"].append(types[selected].astype(np.uint8, copy=False))
                parts["params"].append(params[selected])
                parts["source_index"].append(source_index[selected])
                parts["source_domain"].append(
                    np.full(count, domain, dtype=np.uint8)
                )
    selected = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    actual = np.bincount(selected["types"], minlength=len(CMD_TYPES))
    if not np.array_equal(actual, targets):
        raise RuntimeError(f"selected class supply {actual} != requested {targets}")
    return selected


def _canonicalize_class_order(
    arrays: dict[str, np.ndarray], seed: int
) -> dict[str, np.ndarray]:
    """Give equal-supply variants the same class schedule before shuffling."""
    indices = []
    for class_id in range(len(CMD_TYPES)):
        rows = np.flatnonzero(arrays["types"] == class_id)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 1701, class_id])
        )
        indices.append(rows[rng.permutation(len(rows))])
    order = np.concatenate(indices)
    return {key: value[order] for key, value in arrays.items()}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _prepare_output(root: Path, overwrite: bool) -> None:
    train = root / "train"
    validation = root / "val"
    train.mkdir(parents=True, exist_ok=True)
    validation.mkdir(parents=True, exist_ok=True)
    existing = list(train.glob("shard_*.npz")) + list(validation.glob("shard_*.npz"))
    if existing and not overwrite:
        raise SystemExit(f"{root} already has training shards; pass --overwrite")
    for path in existing:
        path.unlink()
    (root / "manifest.json").unlink(missing_ok=True)


def _link_fixed_validation(base: Path, output: Path) -> list[dict]:
    records = []
    for source in sorted((base / "val").glob("shard_*.npz")):
        target = output / "val" / source.name
        try:
            os.link(source, target)
            method = "hardlink"
        except OSError:
            import shutil

            shutil.copy2(source, target)
            method = "copy"
        records.append(
            {"path": str(target.relative_to(output)), "bytes": target.stat().st_size,
             "method": method}
        )
    if not records:
        raise FileNotFoundError(f"no fixed validation shards under {base / 'val'}")
    return records


def _write_variant(
    output: Path,
    base_arrays: dict[str, np.ndarray],
    synthetic_arrays: dict[str, np.ndarray],
    order: np.ndarray,
    shard_size: int,
) -> list[dict]:
    combined = {
        key: np.concatenate([base_arrays[key], synthetic_arrays[key]], axis=0)
        for key in ARRAY_KEYS
    }
    records = []
    for shard_index, start in enumerate(range(0, len(order), shard_size)):
        selected = order[start : start + shard_size]
        arrays = {key: value[selected] for key, value in combined.items()}
        path = output / "train" / f"shard_{shard_index:05d}.npz"
        _atomic_npz(path, arrays)
        records.append(
            {
                "path": str(path.relative_to(output)),
                "samples": len(selected),
                "classes": np.bincount(
                    arrays["types"], minlength=len(CMD_TYPES)
                ).astype(int).tolist(),
                "domains": np.bincount(
                    arrays["source_domain"], minlength=2
                ).astype(int).tolist(),
            }
        )
    return records


def _named_counts(counts: np.ndarray) -> dict[str, int]:
    return {name.lower(): int(counts[class_id]) for name, class_id in CMD_TYPES.items()}


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.synthetic_fraction <= 1.0:
        raise SystemExit("--synthetic-fraction must be in [0, 1]")
    if args.max_synthetic_repeats < 1.0:
        raise SystemExit("--max-synthetic-repeats must be at least 1")
    if args.shard_size < 1:
        raise SystemExit("--shard-size must be positive")
    if args.output_a.resolve() == args.output_b.resolve():
        raise SystemExit("A and B outputs must be different")

    base_shards = scan_supply(args.base, "train")
    synthetic_a_shards = scan_supply(args.synthetic_a, "train")
    synthetic_b_shards = scan_supply(args.synthetic_b, "train")
    base_supply = total_supply(base_shards)
    synthetic_a_supply = total_supply(synthetic_a_shards)
    synthetic_b_supply = total_supply(synthetic_b_shards)
    target, base_targets, synthetic_targets = paired_allocation(
        base_supply,
        synthetic_a_supply,
        synthetic_b_supply,
        args.target_per_class,
        args.synthetic_fraction,
        args.max_synthetic_repeats,
    )

    _prepare_output(args.output_a, args.overwrite)
    _prepare_output(args.output_b, args.overwrite)
    base_arrays = _select_arrays(base_shards, base_targets, args.seed, DOMAIN_BASE)
    synthetic_a_arrays = _select_arrays(
        synthetic_a_shards, synthetic_targets, args.seed + 1, DOMAIN_PATENTVEC
    )
    synthetic_b_arrays = _select_arrays(
        synthetic_b_shards, synthetic_targets, args.seed + 1, DOMAIN_PATENTVEC
    )
    synthetic_a_arrays = _canonicalize_class_order(
        synthetic_a_arrays, args.seed
    )
    synthetic_b_arrays = _canonicalize_class_order(
        synthetic_b_arrays, args.seed
    )
    total = target * len(CMD_TYPES)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 991]))
    order = rng.permutation(total)
    shards_a = _write_variant(
        args.output_a, base_arrays, synthetic_a_arrays, order, args.shard_size
    )
    shards_b = _write_variant(
        args.output_b, base_arrays, synthetic_b_arrays, order, args.shard_size
    )
    validation_a = _link_fixed_validation(args.base, args.output_a)
    validation_b = _link_fixed_validation(args.base, args.output_b)

    common = {
        "schema_version": "patentvec-free2cad-paired-ab-1.0",
        "strategy": "equal_budget_class_supply_replacement",
        "base": str(args.base),
        "synthetic_fraction_requested": args.synthetic_fraction,
        "max_synthetic_repeats": args.max_synthetic_repeats,
        "target_per_class": target,
        "base_source_supply": _named_counts(base_supply),
        "synthetic_a_source_supply": _named_counts(synthetic_a_supply),
        "synthetic_b_source_supply": _named_counts(synthetic_b_supply),
        "materialized_base_supply": _named_counts(base_targets),
        "materialized_synthetic_supply": _named_counts(synthetic_targets),
        "materialized_total_supply": {
            name.lower(): target for name in CMD_TYPES
        },
        "total_train_edges": total,
        "seed": args.seed,
        "fixed_validation_source": str(args.base / "val"),
    }
    for variant, synthetic, output, shards, validation in (
        ("A", args.synthetic_a, args.output_a, shards_a, validation_a),
        ("B", args.synthetic_b, args.output_b, shards_b, validation_b),
    ):
        manifest = {
            **common,
            "variant": variant,
            "synthetic_source": str(synthetic),
            "train_shards": shards,
            "validation_shards": validation,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(common, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
