#!/usr/bin/env python3
"""Sample candidate TIF figures from the full PatentData pool for hatch selection.

Scans data/PatentData/ReorganisedData, excludes patents already processed
(hatch_pilot, hatch_gt, latest_run), and samples N figures for visual review
with tools/hatch_select.py.

    python -m tools.hatch_sample_candidates --n 200 \
        --out output/PatentData/hatch_candidates.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random


_DEFAULT_TIF_ROOT = "data/PatentData/ReorganisedData"
_DEFAULT_EXCLUDE = [
    "output/PatentData/hatch_pilot",
    "output/PatentData/hatch_gt",
]
_DEFAULT_EXCLUDE_RUNS = ["latest_run", "latest_run_puhachovft"]


def _already_processed_patents(exclude_dirs, exclude_run_dirs):
    """Return set of patent IDs already in any processed set."""
    patents: set[str] = set()
    for d in exclude_dirs:
        for f in glob.glob(os.path.join(d, "*_regions.json")):
            try:
                patents.add(json.load(open(f))["patent"])
            except Exception:
                pass
        for f in glob.glob(os.path.join(d, "*_mask.json")):
            try:
                patents.add(json.load(open(f))["patent"])
            except Exception:
                pass
    for d in exclude_run_dirs:
        if os.path.isdir(d):
            for entry in os.listdir(d):
                if os.path.isdir(os.path.join(d, entry)) and not entry.startswith("."):
                    patents.add(entry)
    return patents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif-root", default=_DEFAULT_TIF_ROOT)
    ap.add_argument("--n", type=int, default=200,
                     help="number of candidate figures to sample (default: 200)")
    ap.add_argument("--per-patent", type=int, default=1,
                     help="max figures sampled per patent (default: 1, keeps diversity)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output/PatentData/hatch_candidates.json")
    ap.add_argument("--exclude-dir", nargs="*", default=_DEFAULT_EXCLUDE,
                     help="dirs containing *_regions.json or *_mask.json to exclude")
    ap.add_argument("--exclude-run", nargs="*", default=_DEFAULT_EXCLUDE_RUNS,
                     help="run dirs (subfolders = patent IDs) to exclude")
    a = ap.parse_args()

    rng = random.Random(a.seed)

    processed = _already_processed_patents(a.exclude_dir, a.exclude_run)
    print(f"excluding {len(processed)} already-processed patent(s)")

    # Enumerate all patent dirs
    patent_dirs = sorted(
        p for p in glob.glob(os.path.join(a.tif_root, "*"))
        if os.path.isdir(p) and os.path.basename(p) not in processed
    )
    rng.shuffle(patent_dirs)
    print(f"{len(patent_dirs)} untouched patent dirs available")

    candidates = []
    for pdir in patent_dirs:
        if len(candidates) >= a.n:
            break
        patent = os.path.basename(pdir)
        tifs = sorted(glob.glob(os.path.join(pdir, "*.tif")))
        if not tifs:
            continue
        chosen = rng.sample(tifs, min(a.per_patent, len(tifs)))
        for tif_path in chosen:
            fname = os.path.basename(tif_path)          # e.g. EP1234567B1_F0003.tif
            sketch = fname[len(patent) + 1:].replace(".tif", "")  # e.g. F0003
            candidates.append({
                "patent": patent,
                "sketch": sketch,
                "tif": tif_path,
                "graph": None,   # no processed graph — selection is visual only
            })
            if len(candidates) >= a.n:
                break

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(candidates, open(a.out, "w"), indent=2)
    print(f"sampled {len(candidates)} candidate figures -> {a.out}")
    print(f"\nReview them with:\n"
          f"  python -m tools.hatch_select --candidate-list {a.out}")


if __name__ == "__main__":
    main()
