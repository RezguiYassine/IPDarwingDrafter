"""
Batch driver for evaluating the AP3 vectorization pipeline on a corpus of
patent sketches (Phase 0 + Phase 1 of the evaluation roadmap).

Reads sketches from a directory of patent subfolders, runs all four stages
per sketch, and writes one row of intrinsic metrics per sketch to a SQLite
results DB. Resumable: sketches whose row already exists in the DB are
skipped (unless --no-resume is given).

Usage from project root:

    # Phase 0 pilot — 100 sketches, one per random patent
    python -m tools.batch_run --limit 100 --stratified

    # Phase 1 full corpus
    python -m tools.batch_run --workers 8

    # Custom paths
    python -m tools.batch_run \\
        --patent-root data/PatentData/ReorganisedData \\
        --output      output/PatentData \\
        --db          output/PatentData/results.db \\
        --workers     8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional

import yaml
from tqdm import tqdm

# ─── Make each stage module importable as a plain Python module ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("stage1_preprocessing", "stage2_strokeextraction",
            "stage3_primitivesfitting", "stage4_export"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

import stage1_preprocess           # noqa: E402
import stage2_stroke_extract       # noqa: E402
import stage3_primitive_fit        # noqa: E402
import stage4_export               # noqa: E402

from tools import results_db       # noqa: E402


logger = logging.getLogger("batch_run")


# ─── Worker globals (one set per process, lazy-loaded at first task) ─────────

_WORKER_CFG = None
_WORKER_S1_MODEL = None
_WORKER_S2_MODEL = None


def _worker_init(config_path: str) -> None:
    """Initialise per-process state: load config, load Stage-1/2 ML models."""
    global _WORKER_CFG, _WORKER_S1_MODEL, _WORKER_S2_MODEL

    with open(config_path) as f:
        _WORKER_CFG = yaml.safe_load(f) or {}

    # Resolve relative weight paths against the config file's directory so
    # the path works regardless of the worker's cwd (mirrors stage1's CLI).
    config_dir = Path(config_path).resolve().parent
    for section in ("sketchcleannet", "puhachov"):
        block = _WORKER_CFG.get(section)
        if isinstance(block, dict):
            w = block.get("weights", "")
            if w and not Path(w).is_absolute():
                block["weights"] = str(config_dir / w)

    # Silence per-stage chatter inside workers; only the driver logs.
    logging.basicConfig(level=logging.ERROR, force=True)

    _WORKER_S1_MODEL = stage1_preprocess.load_model(_WORKER_CFG)
    _WORKER_S2_MODEL = stage2_stroke_extract.load_model(_WORKER_CFG)


def _process_one(job: tuple[str, str, str, str]) -> dict:
    """
    Run all four stages on a single sketch. Returns a dict suitable for
    `results_db.insert_row`. Never raises: errors are captured into the dict.
    """
    patent_id, sketch_id, input_path_str, output_dir_str = job
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)

    row: dict = {
        "patent_id":    patent_id,
        "sketch_id":    sketch_id,
        "input_path":   input_path_str,
        "status":       "ok",
        "error":        None,
        "completed_at": _dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    t0 = time.perf_counter()

    try:
        s1 = stage1_preprocess.run(
            input_path=input_path, output_dir=output_dir,
            sketch_id=sketch_id, config=_WORKER_CFG, model=_WORKER_S1_MODEL,
        )
        row.update({
            "s1_time":       s1.processing_time_s,
            "s1_quality":    s1.skeleton_quality,
            "s1_model_used": s1.model_used,
            "s1_flagged":    int(s1.flagged),
        })
    except Exception as exc:
        row["status"] = "stage1"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s2 = stage2_stroke_extract.run(
            skeleton_path=s1.skeleton_path, output_dir=output_dir,
            sketch_id=sketch_id, config=_WORKER_CFG, model=_WORKER_S2_MODEL,
        )
        row.update({
            "s2_time":         s2.processing_time_s,
            "s2_keypoint_src": s2.keypoint_source,
            "s2_n_nodes":      s2.n_nodes,
            "s2_n_edges":      s2.n_edges,
            "s2_isolation":    s2.isolation_ratio,
            "s2_flagged":      int(s2.flagged),
        })
    except Exception as exc:
        row["status"] = "stage2"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s3 = stage3_primitive_fit.run(
            graph_path=s2.graph_path, output_dir=output_dir,
            sketch_id=sketch_id, config=_WORKER_CFG,
        )
        row.update({
            "s3_time":         s3.processing_time_s,
            "s3_n_primitives": s3.n_primitives,
            "s3_mean_conf":    s3.mean_confidence,
            "s3_flagged":      int(s3.flagged),
        })
    except Exception as exc:
        row["status"] = "stage3"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s4 = stage4_export.run(
            input_json=s3.primitives_path, output_dir=output_dir,
            sketch_id=sketch_id, formats=("svg", "dxf"), dxf_mode="patent",
        )
        row.update({
            "s4_time":    s4.processing_time_s,
            "s4_n_in":    s4.n_primitives_in,
            "s4_n_out":   s4.n_primitives_out,
            "s4_flagged": int(s4.flagged),
        })
    except Exception as exc:
        row["status"] = "stage4"
        row["error"] = f"{type(exc).__name__}: {exc}"

    row["total_time"] = time.perf_counter() - t0
    return row


# ─── Dataset walker ──────────────────────────────────────────────────────────

def _iter_sketches(patent_root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield (patent_id, sketch_id, tif_path) over the whole corpus."""
    for patent_dir in sorted(patent_root.iterdir()):
        if not patent_dir.is_dir():
            continue
        patent_id = patent_dir.name
        for f in sorted(patent_dir.iterdir()):
            if f.suffix.lower() in (".tif", ".tiff"):
                # sketch_id = filename stem with patent prefix stripped
                stem = f.stem
                if stem.startswith(patent_id + "_"):
                    sketch_id = stem[len(patent_id) + 1:]
                else:
                    sketch_id = stem
                yield patent_id, sketch_id, f


def _stratified_sample(patent_root: Path, n: int,
                       seed: int = 42) -> list[tuple[str, str, Path]]:
    """One sketch from each of `n` random patents (those with >=1 sketch)."""
    rng = random.Random(seed)
    candidates = [d for d in patent_root.iterdir() if d.is_dir()]
    rng.shuffle(candidates)

    picks: list[tuple[str, str, Path]] = []
    for patent_dir in candidates:
        tifs = [f for f in sorted(patent_dir.iterdir())
                if f.suffix.lower() in (".tif", ".tiff")]
        if not tifs:
            continue
        f = tifs[0]
        stem = f.stem
        patent_id = patent_dir.name
        sketch_id = (stem[len(patent_id) + 1:]
                     if stem.startswith(patent_id + "_") else stem)
        picks.append((patent_id, sketch_id, f))
        if len(picks) >= n:
            break
    return picks


# ─── Driver ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate the AP3 pipeline over a patent corpus.",
    )
    parser.add_argument("--patent-root", type=Path,
                        default=PROJECT_ROOT / "data" / "PatentData"
                                              / "ReorganisedData",
                        help="Directory containing per-patent subfolders.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output" / "PatentData",
                        help="Root directory for stage outputs (cleaned/, "
                             "graphs/, primitives/, vectors/ are created "
                             "under <output>/<patent_id>/).")
    parser.add_argument("--db", type=Path, default=None,
                        help="SQLite results DB. Default: "
                             "<output>/results.db")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config.yaml",
                        help="Pipeline config file.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers. Default: "
                             "config.pipeline.workers or os.cpu_count().")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N sketches (Phase 0 pilot).")
    parser.add_argument("--stratified", action="store_true",
                        help="With --limit: one sketch per random patent. "
                             "Without: take the first N sketches of the corpus.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess sketches even if they already have "
                             "a row in the DB.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for stratified sampling.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.patent_root.exists():
        logger.error("Patent root does not exist: %s", args.patent_root)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    db_path: Path = args.db or (args.output / "results.db")
    results_db.init_db(db_path)

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    workers = args.workers or cfg.get("pipeline", {}).get("workers") \
              or max(1, (os.cpu_count() or 2) - 1)

    # ── Build the job list ────────────────────────────────────────────────
    if args.limit and args.stratified:
        sketches = _stratified_sample(args.patent_root, args.limit, args.seed)
    else:
        it = _iter_sketches(args.patent_root)
        if args.limit:
            sketches = []
            for s in it:
                sketches.append(s)
                if len(sketches) >= args.limit:
                    break
        else:
            sketches = list(it)

    if not sketches:
        logger.error("No sketches found under %s", args.patent_root)
        return 2

    # ── Skip already-processed (resume) ──────────────────────────────────
    if not args.no_resume:
        with results_db.connect(db_path) as conn:
            before = len(sketches)
            sketches = [
                s for s in sketches
                if not results_db.already_processed(conn, s[0], s[1])
            ]
            skipped = before - len(sketches)
            if skipped:
                logger.info("Resume: skipping %d sketches already in DB.", skipped)

    if not sketches:
        logger.info("Nothing to do.")
    else:
        logger.info(
            "Processing %d sketches with %d workers. Output → %s, DB → %s",
            len(sketches), workers, args.output, db_path,
        )

        # Build the (patent_id, sketch_id, input_path_str, output_dir_str) tuples.
        # Per-patent output dir keeps the cleaned/graphs/... convention scoped.
        jobs = [
            (patent_id, sketch_id, str(tif_path),
             str(args.output / patent_id))
            for (patent_id, sketch_id, tif_path) in sketches
        ]

        # ── Execute ──────────────────────────────────────────────────────
        n_ok = 0
        n_err = 0
        with results_db.connect(db_path) as conn, \
             ProcessPoolExecutor(
                 max_workers=workers,
                 initializer=_worker_init,
                 initargs=(str(args.config),),
             ) as pool:

            futures = [pool.submit(_process_one, j) for j in jobs]
            bar = tqdm(as_completed(futures), total=len(futures),
                       smoothing=0.1, mininterval=0.5)
            for fut in bar:
                try:
                    row = fut.result()
                except Exception as exc:                # process died mid-task
                    bar.write(f"Worker crashed: {exc!r}")
                    bar.write(traceback.format_exc())
                    n_err += 1
                    continue
                results_db.insert_row(conn, row)
                if row["status"] == "ok":
                    n_ok += 1
                else:
                    n_err += 1
                bar.set_postfix(ok=n_ok, err=n_err)

        logger.info("Done. ok=%d err=%d", n_ok, n_err)

    # ── Summary ──────────────────────────────────────────────────────────
    s = results_db.summarise(db_path)
    print()
    print("── DB summary ──────────────────────────────────────────────")
    print(f"  Total rows         : {s['total']}")
    print(f"  Pipeline 'ok'      : {s['ok']}  "
          f"({100.0 * s['ok'] / s['total']:.1f}%)" if s["total"] else "")
    print(f"  Status breakdown   : {s['by_status']}")
    if s["mean_total_s"]:
        print(f"  Mean total time/ok : {s['mean_total_s']:.2f} s")
        print(f"    Stage 1          : {s['mean_s1_s']:.2f} s")
        print(f"    Stage 2          : {s['mean_s2_s']:.2f} s")
        print(f"    Stage 3          : {s['mean_s3_s']:.2f} s")
        print(f"    Stage 4          : {s['mean_s4_s']:.2f} s")
    for k in ("flag_rate_s1", "flag_rate_s2", "flag_rate_s3", "flag_rate_s4"):
        v = s[k]
        if v is not None:
            print(f"  {k:18s}: {100.0 * v:.1f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
