"""Materialize a class-supply-balanced SketchGraphs + Drawing2CAD dataset.

Balancing is performed on Stage 3 edge labels, never on source sketch count.
The automatic epoch target is the larger of Drawing2CAD's POLYLINE and BEZIER
supplies. This keeps every sample from the scarcer geometric classes (with at
most modest replacement) while aggressively downsampling SketchGraphs LINE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CMD_TYPES = {"LINE": 0, "ARC": 1, "CIRCLE": 2, "POLYLINE": 3, "BEZIER": 4}
ANALYTIC_CLASSES = (0, 1, 2)
D2C_ONLY_CLASSES = (3, 4)


@dataclass(frozen=True)
class ShardSupply:
    path: Path
    counts: tuple[int, ...]
    raw_counts: tuple[int, ...]
    removed_circles: int
    relabeled_arcs: int


def _training_visible_arrays(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Apply the trainer's label cleaning while retaining source row indices."""
    points = np.asarray(data["points"])
    mask = np.asarray(data["mask"], dtype=bool)
    types = np.asarray(data["types"]).copy()
    params = np.asarray(data["params"]).copy()
    keep = np.ones(len(types), dtype=bool)

    circles = types == CMD_TYPES["CIRCLE"]
    invalid_circles = circles & (
        (np.abs(params[:, :2] - 0.5) > 0.30).any(axis=1)
        | (params[:, 2] > 0.65)
    )
    keep[invalid_circles] = False

    arc_rows = np.flatnonzero(types == CMD_TYPES["ARC"])
    lengths = mask.sum(axis=1).astype(int)
    relabel_parts = []
    for length in np.unique(lengths[arc_rows]):
        selected = arc_rows[lengths[arc_rows] == length]
        if length < 3:
            relabel_parts.append(selected)
            continue
        chord = points[selected, length - 1] - points[selected, 0]
        chord_length = np.linalg.norm(chord, axis=1)
        degenerate = chord_length < 1e-9
        safe_length = np.maximum(chord_length, 1e-9)
        normal = np.stack([-chord[:, 1], chord[:, 0]], axis=1) / safe_length[:, None]
        offsets = points[selected, :length] - points[selected, None, 0]
        sagitta = np.abs(np.einsum("npi,ni->np", offsets, normal)).max(axis=1)
        degenerate |= sagitta / safe_length < 0.01
        if degenerate.any():
            relabel_parts.append(selected[degenerate])

    relabeled = (
        np.concatenate(relabel_parts).astype(np.int64, copy=False)
        if relabel_parts else np.empty(0, dtype=np.int64)
    )
    if len(relabeled):
        types[relabeled] = CMD_TYPES["LINE"]
        params[relabeled] = 0.0
        params[relabeled, 0:2] = points[relabeled, 0]
        params[relabeled, 2:4] = points[relabeled, lengths[relabeled] - 1]

    return keep, types, params, int(invalid_circles.sum()), len(relabeled)


def scan_supply(data_dir: Path, split: str) -> list[ShardSupply]:
    shards = sorted((data_dir / split).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no Stage 3 shards in {data_dir / split}")
    result = []
    for path in shards:
        with np.load(path) as data:
            raw_types = np.asarray(data["types"])
            raw_counts = np.bincount(
                raw_types, minlength=len(CMD_TYPES)
            )[:len(CMD_TYPES)]
            keep, types, _params, removed, relabeled = _training_visible_arrays(data)
            counts = np.bincount(
                types[keep], minlength=len(CMD_TYPES)
            )[:len(CMD_TYPES)]
        result.append(ShardSupply(
            path=path,
            counts=tuple(map(int, counts)),
            raw_counts=tuple(map(int, raw_counts)),
            removed_circles=removed,
            relabeled_arcs=relabeled,
        ))
    return result


def total_supply(shards: list[ShardSupply]) -> np.ndarray:
    return np.asarray([shard.counts for shard in shards], dtype=np.int64).sum(axis=0)


def total_raw_supply(shards: list[ShardSupply]) -> np.ndarray:
    return np.asarray(
        [shard.raw_counts for shard in shards], dtype=np.int64
    ).sum(axis=0)


def choose_epoch_target(d2c_counts: np.ndarray, requested: int) -> int:
    """Choose equal class supply while limiting rare-class duplication."""
    if requested > 0:
        return requested
    polyline = int(d2c_counts[CMD_TYPES["POLYLINE"]])
    bezier = int(d2c_counts[CMD_TYPES["BEZIER"]])
    if not polyline or not bezier:
        raise ValueError(
            "automatic balancing requires non-zero Drawing2CAD POLYLINE and BEZIER labels"
        )
    return max(polyline, bezier)


def allocate_domains(
    target: int,
    class_id: int,
    sg_supply: np.ndarray,
    d2c_supply: np.ndarray,
    sg_fraction: float,
    max_repeats: float,
) -> dict[str, int]:
    """Allocate one class target to domains under explicit supply caps."""
    if class_id in D2C_ONLY_CLASSES:
        available = int(d2c_supply[class_id])
        return {"sketchgraphs": 0,
                "drawing2cad": min(target, int(math.floor(available * max_repeats)))}

    capacities = {
        "sketchgraphs": int(math.floor(sg_supply[class_id] * max_repeats)),
        "drawing2cad": int(math.floor(d2c_supply[class_id] * max_repeats)),
    }
    allocation = {
        "sketchgraphs": min(int(round(target * sg_fraction)), capacities["sketchgraphs"]),
        "drawing2cad": min(target - int(round(target * sg_fraction)),
                            capacities["drawing2cad"]),
    }
    remaining = target - sum(allocation.values())
    for domain in ("sketchgraphs", "drawing2cad"):
        room = capacities[domain] - allocation[domain]
        take = min(remaining, room)
        allocation[domain] += take
        remaining -= take
    return allocation


def _proportional_allocation(counts: np.ndarray, target: int) -> np.ndarray:
    if target <= 0 or counts.sum() <= 0:
        return np.zeros_like(counts)
    exact = counts.astype(np.float64) * (target / counts.sum())
    allocation = np.floor(exact).astype(np.int64)
    remaining = target - int(allocation.sum())
    if remaining:
        order = np.argsort(-(exact - allocation), kind="stable")
        allocation[order[:remaining]] += 1
    return allocation


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _select_domain_arrays(
    source_name: str,
    shards: list[ShardSupply],
    targets: dict[int, int],
    seed: int,
) -> dict[str, np.ndarray]:
    domain_id = 0 if source_name == "sketchgraphs" else 1
    allocations = {
        class_id: _proportional_allocation(
            np.asarray([shard.counts[class_id] for shard in shards], dtype=np.int64),
            target,
        )
        for class_id, target in targets.items() if target > 0
    }
    rngs = {
        class_id: np.random.default_rng(
            np.random.SeedSequence([seed, class_id, domain_id])
        )
        for class_id in allocations
    }
    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "points", "mask", "types", "params", "source_index", "source_domain"
        )
    }

    for shard_index, shard in enumerate(shards):
        shard_takes = {
            class_id: int(per_shard[shard_index])
            for class_id, per_shard in allocations.items()
            if per_shard[shard_index] > 0
        }
        if not shard_takes:
            continue
        with np.load(shard.path) as data:
            keep, types, params, _removed, _relabeled = _training_visible_arrays(data)
            points = np.asarray(data["points"])
            mask = np.asarray(data["mask"])
            source_indices = (
                np.asarray(data["source_index"])
                if "source_index" in data else np.arange(len(types), dtype=np.int64)
            )
            for class_id, take in shard_takes.items():
                rows = np.flatnonzero(keep & (types == class_id))
                selected = rngs[class_id].choice(
                    rows, size=take, replace=take > len(rows)
                )
                selected_arrays = {
                    "points": points[selected],
                    "mask": mask[selected],
                    "types": types[selected].astype(np.uint8, copy=False),
                    "params": params[selected],
                    "source_index": source_indices[selected],
                    "source_domain": np.full(len(selected), domain_id, dtype=np.uint8),
                }
                for key, value in selected_arrays.items():
                    parts[key].append(value)
    return {
        key: np.concatenate(value, axis=0) for key, value in parts.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Balance mixed Free2CAD data by per-class edge supply",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sketchgraphs", type=Path,
                        default=Path("output/SketchGraphsStage3Full/stage3"))
    parser.add_argument("--drawing2cad", type=Path,
                        default=Path("output/Drawing2CAD/stage3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--target-per-class", type=int, default=0,
                        help="0 derives target from D2C POLYLINE/BEZIER supply")
    parser.add_argument("--sg-analytic-fraction", type=float, default=0.70)
    parser.add_argument("--max-repeats", type=float, default=1.5)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace mixed shards for this output split")
    args = parser.parse_args()
    if not 0.0 <= args.sg_analytic_fraction <= 1.0:
        raise SystemExit("--sg-analytic-fraction must be between 0 and 1")
    if args.max_repeats < 1.0:
        raise SystemExit("--max-repeats must be at least 1")

    output_dir = args.output / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("shard_*.npz"))
    if existing:
        if not args.overwrite:
            raise SystemExit(
                f"{output_dir} already contains {len(existing)} shards; "
                "pass --overwrite or use a new output directory"
            )
        for path in existing:
            path.unlink()
        (output_dir / "manifest.json").unlink(missing_ok=True)

    sg_shards = scan_supply(args.sketchgraphs, args.split)
    d2c_shards = scan_supply(args.drawing2cad, args.split)
    sg_counts = total_supply(sg_shards)
    d2c_counts = total_supply(d2c_shards)
    sg_raw_counts = total_raw_supply(sg_shards)
    d2c_raw_counts = total_raw_supply(d2c_shards)
    target = choose_epoch_target(d2c_counts, args.target_per_class)

    allocations = {}
    selected_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "points", "mask", "types", "params", "source_index", "source_domain"
        )
    }
    domain_targets: dict[str, dict[int, int]] = {
        "sketchgraphs": {}, "drawing2cad": {},
    }
    for name, class_id in CMD_TYPES.items():
        domains = allocate_domains(
            target, class_id, sg_counts, d2c_counts,
            args.sg_analytic_fraction, args.max_repeats,
        )
        allocations[name] = domains
        for domain, count in domains.items():
            domain_targets[domain][class_id] = count

    for domain, targets in domain_targets.items():
        selected = _select_domain_arrays(
            domain,
            sg_shards if domain == "sketchgraphs" else d2c_shards,
            targets, args.seed,
        )
        for key, value in selected.items():
            selected_parts[key].append(value)

    combined = {
        key: np.concatenate(parts, axis=0) for key, parts in selected_parts.items()
    }
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 991]))
    order = rng.permutation(len(combined["types"]))
    records = []
    for shard_index, start in enumerate(range(0, len(order), args.shard_size)):
        selected = order[start:start + args.shard_size]
        arrays = {key: value[selected] for key, value in combined.items()}
        path = output_dir / f"shard_{shard_index:05d}.npz"
        _atomic_npz(path, arrays)
        records.append({
            "path": path.name,
            "samples": len(selected),
            "classes": np.bincount(
                arrays["types"], minlength=len(CMD_TYPES)
            ).astype(int).tolist(),
            "domains": np.bincount(
                arrays["source_domain"], minlength=2
            ).astype(int).tolist(),
        })

    supplied = {
        name: sum(allocations[name].values()) for name in CMD_TYPES
    }
    report = {
        "strategy": "class_supply_balanced",
        "split": args.split,
        "target_per_class": target,
        "max_repeats": args.max_repeats,
        "sg_analytic_fraction": args.sg_analytic_fraction,
        "source_supply": {
            "sketchgraphs": dict(zip(CMD_TYPES, map(int, sg_counts))),
            "drawing2cad": dict(zip(CMD_TYPES, map(int, d2c_counts))),
        },
        "source_raw_supply": {
            "sketchgraphs": dict(zip(CMD_TYPES, map(int, sg_raw_counts))),
            "drawing2cad": dict(zip(CMD_TYPES, map(int, d2c_raw_counts))),
        },
        "training_visibility": {
            "sketchgraphs": {
                "removed_circles": sum(s.removed_circles for s in sg_shards),
                "relabeled_arcs": sum(s.relabeled_arcs for s in sg_shards),
            },
            "drawing2cad": {
                "removed_circles": sum(s.removed_circles for s in d2c_shards),
                "relabeled_arcs": sum(s.relabeled_arcs for s in d2c_shards),
            },
        },
        "allocation": allocations,
        "materialized_supply": supplied,
        "total_samples": sum(supplied.values()),
        "shards": records,
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "shards"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
