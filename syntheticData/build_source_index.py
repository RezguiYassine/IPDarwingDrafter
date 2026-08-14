from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from syntheticData.patentvec.generator import SourcePool


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic capability index for PatentVec sources."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "PatentVecSourceIndex" / "train.json",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument(
        "--sketchgraphs",
        type=Path,
        default=PROJECT_ROOT / "data" / "SketchGraphs" / "raw" / "sg_t16_train.npy",
    )
    parser.add_argument(
        "--cadvg-root", type=Path, default=PROJECT_ROOT / "data" / "Drawing2CAD"
    )
    parser.add_argument("--scan-sketchgraphs", type=int, default=5000)
    parser.add_argument("--scan-cadvg", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=270725)
    return parser.parse_args()


def _entry(dataset: str, source_id, component, view: str | None = None) -> dict:
    return {
        "dataset": dataset,
        "source_id": str(source_id),
        "view": view,
        "component_key": component.component_key,
        "profiles": SourcePool.component_profiles(component),
        "primitive_count": len(component.primitives),
    }


def _sample_indices(rng: np.random.Generator, total: int, count: int) -> np.ndarray:
    count = min(max(0, int(count)), total)
    if count == total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=count, replace=False))


def main() -> int:
    args = parse_args()
    if args.scan_sketchgraphs <= 0 or args.scan_cadvg <= 0:
        raise SystemExit("source-index scan counts must be positive")
    rng = np.random.default_rng(args.seed)
    pool = SourcePool(args.sketchgraphs, args.cadvg_root, split=args.split)
    entries: dict[tuple[str, str, str | None, str], dict] = {}
    errors = Counter()

    sketchgraph_indices = _sample_indices(
        rng, len(pool.sketchgraphs), args.scan_sketchgraphs
    )
    for position, source_index in enumerate(sketchgraph_indices, start=1):
        try:
            components = pool.sketchgraphs.load_source_sample(int(source_index))
            for component in components:
                item = _entry("SketchGraphs", int(source_index), component)
                if item["profiles"]:
                    key = (
                        item["dataset"],
                        item["source_id"],
                        item["view"],
                        item["component_key"],
                    )
                    entries[key] = item
        except Exception as exc:
            errors[f"SketchGraphs:{type(exc).__name__}"] += 1
        if position % 500 == 0:
            print(
                f"SketchGraphs {position}/{len(sketchgraph_indices)} "
                f"indexed_components={len(entries)}",
                flush=True,
            )

    cad_total = len(pool.cadvg.sample_ids) * len(pool.cadvg.VIEWS)
    cad_indices = _sample_indices(rng, cad_total, args.scan_cadvg)
    for position, flat_index in enumerate(cad_indices, start=1):
        source_position, view_position = divmod(
            int(flat_index), len(pool.cadvg.VIEWS)
        )
        source_id = pool.cadvg.sample_ids[source_position]
        view = pool.cadvg.VIEWS[view_position]
        try:
            components = pool.cadvg.load_source_sample(source_id, view=view)
            for component in components:
                item = _entry("CAD-VGDrawing", source_id, component, view=view)
                if item["profiles"]:
                    key = (
                        item["dataset"],
                        item["source_id"],
                        item["view"],
                        item["component_key"],
                    )
                    entries[key] = item
        except Exception as exc:
            errors[f"CAD-VGDrawing:{type(exc).__name__}"] += 1
        if position % 500 == 0:
            print(
                f"CAD-VGDrawing {position}/{len(cad_indices)} "
                f"indexed_components={len(entries)}",
                flush=True,
            )

    ordered = [entries[key] for key in sorted(entries)]
    counts = Counter()
    for item in ordered:
        for profile in item["profiles"]:
            counts[f"{item['dataset']}:{profile}"] += 1
    required_buckets = {
        ("SketchGraphs", "rich_balanced_region_line"),
        ("CAD-VGDrawing", "rich_balanced_region_line"),
        ("SketchGraphs", "endpoint_rich"),
        ("CAD-VGDrawing", "endpoint_rich"),
        ("SketchGraphs", "circle"),
        ("CAD-VGDrawing", "endpoint_bezier"),
        ("SketchGraphs", "single_line"),
        ("CAD-VGDrawing", "single_line"),
    }
    missing = [
        f"{dataset}:{profile}"
        for dataset, profile in sorted(required_buckets)
        if not counts[f"{dataset}:{profile}"]
    ]
    payload = {
        "schema_version": "patentvec-source-index-1.0",
        "split": args.split,
        "seed": args.seed,
        "sources": {
            "sketchgraphs": str(args.sketchgraphs),
            "cadvg_root": str(args.cadvg_root),
        },
        "scan": {
            "sketchgraphs": len(sketchgraph_indices),
            "cadvg_views": len(cad_indices),
        },
        "entry_count": len(ordered),
        "profile_counts": dict(sorted(counts.items())),
        "missing_required_buckets": missing,
        "errors": dict(sorted(errors.items())),
        "entries": ordered,
    }
    if missing:
        raise RuntimeError(f"source index is missing required buckets: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in payload.items() if key != "entries"}, indent=2))
    print(f"index: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
