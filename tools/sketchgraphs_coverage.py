"""Inspect the compact source-status map written by full SketchGraphs training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stage2_strokeextraction.research.train_puhachov import STATUS_NAMES


def summarize(path: Path) -> dict:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing coverage metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    total = int(metadata["source_total"])
    if path.stat().st_size != total:
        raise ValueError(
            f"coverage size mismatch: {path.stat().st_size:,} bytes != {total:,} records"
        )
    status = np.memmap(path, dtype=np.int8, mode="r", shape=(total,))
    values, counts = np.unique(status, return_counts=True)
    raw = {int(value): int(count) for value, count in zip(values, counts)}
    attempted = total - raw.get(-1, 0)
    return {
        **metadata,
        "attempted": attempted,
        "unattempted": raw.get(-1, 0),
        "accepted": raw.get(0, 0),
        "rejected": sum(raw.get(code, 0) for code in STATUS_NAMES if code),
        "complete": attempted == total,
        "status_counts": {
            STATUS_NAMES[code]: raw.get(code, 0) for code in STATUS_NAMES
        },
    }


def export_rejections(path: Path, target: Path) -> int:
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    total = int(metadata["source_total"])
    status = np.memmap(path, dtype=np.int8, mode="r", shape=(total,))
    rejected = np.flatnonzero(status > 0)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_index", "status_code", "status"))
        for index in rejected:
            code = int(status[index])
            writer.writerow((int(index), code, STATUS_NAMES.get(code, "unknown")))
    return len(rejected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage", type=Path)
    parser.add_argument("--rejections-csv", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    report = summarize(args.coverage)
    print(json.dumps(report, indent=2))
    if args.rejections_csv:
        count = export_rejections(args.coverage, args.rejections_csv)
        print(f"wrote {count:,} rejected source records to {args.rejections_csv}")
    return 0 if report["complete"] or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
