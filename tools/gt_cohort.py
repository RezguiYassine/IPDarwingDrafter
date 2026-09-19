#!/usr/bin/env python3
"""Unpack synthetic sheets into a cohort the pipeline can be run over.

The generator writes sharded tars carrying, for every sheet, both the raster
the pipeline would see and the exact vector geometry it was drawn from. The
real corpus has no such ground truth, which is why every fidelity number this
project has quoted is self-consistency against its own Stage-1 skeleton. These
sheets are the only place the pipeline can be scored against what is actually
there.

    python -m tools.gt_cohort --shards output/PhaseC_gt_500 \
        --output data/SyntheticGT --limit 500

Writes a patent-shaped tree so no pipeline code needs to change:

    data/SyntheticGT/<sample_id>/<sample_id>_F0001.tif      the degraded raster
    data/SyntheticGT/_ground_truth/<sample_id>.json         primitives + canvas
    data/SyntheticGT/worklist.csv                           for --worklist
"""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np


def _samples(shard_dir: Path):
    for tar_path in sorted((shard_dir / "shards").glob("*.tar")):
        with tarfile.open(tar_path) as tar:
            names = tar.getnames()
            stems = sorted({n.rsplit("/", 1)[0] for n in names if "/" in n
                            and n.startswith("samples/")})
            for stem in stems:
                members = {n.rsplit("/", 1)[1]: n for n in names if n.startswith(stem + "/")}
                if "sample.json" not in members or "degraded.png" not in members:
                    continue
                meta = json.loads(tar.extractfile(members["sample.json"]).read())
                raster = cv2.imdecode(
                    np.frombuffer(tar.extractfile(members["degraded.png"]).read(), np.uint8),
                    cv2.IMREAD_GRAYSCALE)
                if raster is None:
                    continue
                yield meta, raster


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    truth_dir = args.output / "_ground_truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    rows, kept = [], 0
    for meta, raster in _samples(args.shards):
        if kept >= args.limit:
            break
        sample_id = meta["sample_id"]
        # The generator renders ink dark on light like a scan, which is what
        # Stage 0 expects; written as TIF only so the corpus walker finds it.
        folder = args.output / sample_id
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{sample_id}_F0001.tif"), raster)
        canvas = meta.get("canvas") or [raster.shape[1], raster.shape[0]]
        (truth_dir / f"{sample_id}.json").write_text(json.dumps({
            "sample_id": sample_id,
            "canvas": canvas,
            "image_shape": list(raster.shape),
            "difficulty": meta.get("difficulty"),
            "primitives_visible": meta.get("primitives_visible") or [],
            "junctions_visible": meta.get("junctions_visible") or [],
            "semantic_layers": meta.get("semantic_layers") or {},
        }, indent=2) + "\n")
        rows.append([sample_id, "F0001",
                     str((folder / f"{sample_id}_F0001.tif").resolve())])
        kept += 1

    worklist = args.output / "worklist.csv"
    with open(worklist, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patent_id", "sketch_id", "input_path"])
        writer.writerows(rows)
    print(f"sheets unpacked : {kept}")
    print(f"ground truth    : {truth_dir}")
    print(f"worklist        : {worklist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
