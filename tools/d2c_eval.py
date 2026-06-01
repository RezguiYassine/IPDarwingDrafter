"""
Drawing2CAD evaluation driver.

Loops over a sample of Drawing2CAD test-set SVGs, rasterizes each to a
binary PNG (mimicking the patent-TIF input the pipeline is tuned for),
runs the full four-stage vectorization pipeline, then compares the
resulting SVG to the ground-truth SVG.

Metrics written to a separate SQLite DB (`d2c_results.db`):
  * iou_pixel    — pixel IoU after re-rasterizing the output SVG at the
                   same resolution as the GT raster.
  * iou_skeleton — IoU on the 1-pixel skeletonization of both rasters
                   (forgives stroke-width mismatches).
  * n_strokes_gt — count of <path> elements in the GT SVG.
  * n_prims_out  — count of primitives our pipeline emitted.
  * per-stage runtimes (mirrors batch_run.py's schema).

Resumable: a (sample_id, view) row already in the DB is skipped unless
--no-resume is given.

Usage (from project root):

    # 100-sample pilot, Front view only
    python -m tools.d2c_eval --limit 100 --views Front --workers 8

    # All four views, 25 samples
    python -m tools.d2c_eval --limit 25
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import logging
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("stage1_preprocessing", "stage2_strokeextraction",
            "stage3_primitivesfitting", "stage4_export"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))
import stage1_preprocess           # noqa: E402
import stage2_stroke_extract       # noqa: E402
import stage3_primitive_fit        # noqa: E402
import stage4_export               # noqa: E402

logger = logging.getLogger("d2c_eval")

D2C_ROOT_DEFAULT  = PROJECT_ROOT / "data" / "Drawing2CAD"
OUTPUT_DEFAULT    = PROJECT_ROOT / "output" / "Drawing2CAD"
RENDER_RES        = 1024     # rasterize SVG → 1024×1024 PNG

VIEWS = ("Front", "Top", "Right", "FrontTopRight")


# ─── Result DB ───────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS d2c_results (
    sample_id        TEXT NOT NULL,    -- "0078/00780135"
    view             TEXT NOT NULL,    -- "Front" | "Top" | …
    status           TEXT NOT NULL,    -- 'ok' or stage that errored
    error            TEXT,
    completed_at     TEXT NOT NULL,
    total_time       REAL,

    -- Inputs
    input_svg_path   TEXT,
    raster_path      TEXT,
    output_svg_path  TEXT,

    -- Pipeline timings (subset of batch_run schema)
    s1_time          REAL,
    s2_time          REAL,
    s3_time          REAL,
    s4_time          REAL,

    -- GT vs output counts
    n_strokes_gt     INTEGER,
    n_prims_out      INTEGER,

    -- Metrics
    iou_pixel        REAL,
    iou_skeleton     REAL,
    precision_pixel  REAL,
    recall_pixel     REAL,
    -- Chamfer distance on skeletons (lower is better, units: pixels).
    -- chamfer_sym is the symmetric mean; chamfer_gt2out / chamfer_out2gt are
    -- the directional means; chamfer_p95_sym is the symmetric 95th-percentile
    -- (catches outliers — most pixels right, a few far-off).
    chamfer_sym      REAL,
    chamfer_gt2out   REAL,
    chamfer_out2gt   REAL,
    chamfer_p95_sym  REAL,

    PRIMARY KEY (sample_id, view)
);

CREATE INDEX IF NOT EXISTS idx_d2c_status ON d2c_results(status);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("PRAGMA journal_mode = WAL;")
        conn.executescript(SCHEMA)


def already_processed(conn, sample_id: str, view: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM d2c_results WHERE sample_id=? AND view=? LIMIT 1",
        (sample_id, view),
    )
    return cur.fetchone() is not None


def insert_row(conn, row: dict) -> None:
    cols = list(row.keys())
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(
        f"INSERT OR REPLACE INTO d2c_results ({','.join(cols)}) "
        f"VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()


# ─── Rasterization & SVG parsing helpers ─────────────────────────────────────

def _rasterize_svg_to_binary(svg_path: Path, out_path: Path,
                             resolution: int = RENDER_RES) -> np.ndarray:
    """Render `svg_path` to `out_path` as a binary PNG (ink=0, bg=255).
    Returns the grayscale array."""
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=resolution,
        output_height=resolution,
        background_color="white",
    )
    img = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
    # Threshold any anti-aliased gray to strict binary.
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(out_path), binary)
    return binary


def _count_paths_in_svg(svg_path: Path) -> int:
    """Cheap count of <path> elements — GT "stroke" count proxy."""
    try:
        return Path(svg_path).read_text().count("<path")
    except Exception:
        return -1


# ─── Pipeline runner (worker-local) ──────────────────────────────────────────

_WORKER_CFG = None
_WORKER_S1_MODEL = None
_WORKER_S2_MODEL = None


def _worker_init(config_path: str) -> None:
    global _WORKER_CFG, _WORKER_S1_MODEL, _WORKER_S2_MODEL
    with open(config_path) as f:
        _WORKER_CFG = yaml.safe_load(f) or {}
    config_dir = Path(config_path).resolve().parent
    for section in ("sketchcleannet", "puhachov"):
        block = _WORKER_CFG.get(section)
        if isinstance(block, dict):
            w = block.get("weights", "")
            if w and not Path(w).is_absolute():
                block["weights"] = str(config_dir / w)
    logging.basicConfig(level=logging.ERROR, force=True)
    _WORKER_S1_MODEL = stage1_preprocess.load_model(_WORKER_CFG)
    _WORKER_S2_MODEL = stage2_stroke_extract.load_model(_WORKER_CFG)


def _process_one(job: tuple) -> dict:
    """Run rasterize → pipeline → compare on one (sample_id, view)."""
    sample_id, view, svg_path_str, work_dir_str = job
    svg_path = Path(svg_path_str)
    work_dir = Path(work_dir_str)
    work_dir.mkdir(parents=True, exist_ok=True)
    sketch_id = f"{sample_id.replace('/', '_')}_{view}"

    row: dict = {
        "sample_id":      sample_id,
        "view":           view,
        "input_svg_path": svg_path_str,
        "status":         "ok",
        "error":          None,
        "completed_at":   _dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    t0 = time.perf_counter()

    raster_path     = work_dir / f"{sketch_id}_input.png"
    output_svg_path = None

    try:
        gt_binary = _rasterize_svg_to_binary(svg_path, raster_path)
        row["raster_path"]   = str(raster_path)
        row["n_strokes_gt"]  = _count_paths_in_svg(svg_path)
    except Exception as exc:
        row["status"] = "rasterize"
        row["error"]  = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s1 = stage1_preprocess.run(
            input_path=raster_path, output_dir=work_dir,
            sketch_id=sketch_id, config=_WORKER_CFG, model=_WORKER_S1_MODEL,
        )
        row["s1_time"] = s1.processing_time_s
    except Exception as exc:
        row["status"] = "stage1"; row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s2 = stage2_stroke_extract.run(
            skeleton_path=s1.skeleton_path, output_dir=work_dir,
            sketch_id=sketch_id, config=_WORKER_CFG, model=_WORKER_S2_MODEL,
        )
        row["s2_time"] = s2.processing_time_s
    except Exception as exc:
        row["status"] = "stage2"; row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s3 = stage3_primitive_fit.run(
            graph_path=s2.graph_path, output_dir=work_dir,
            sketch_id=sketch_id, config=_WORKER_CFG,
            stroke_width=s1.mean_stroke_width,
        )
        row["s3_time"]      = s3.processing_time_s
        row["n_prims_out"]  = s3.n_primitives
    except Exception as exc:
        row["status"] = "stage3"; row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    try:
        s4 = stage4_export.run(
            input_json=s3.primitives_path, output_dir=work_dir,
            sketch_id=sketch_id, formats=("svg",), dxf_mode="basic",
        )
        row["s4_time"]         = s4.processing_time_s
        output_svg_path        = s4.svg_path
        row["output_svg_path"] = str(output_svg_path)
    except Exception as exc:
        row["status"] = "stage4"; row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = time.perf_counter() - t0
        return row

    # ── Metrics: re-rasterize our SVG, compare to GT raster ──────────
    try:
        out_raster = work_dir / f"{sketch_id}_output.png"
        out_binary = _rasterize_svg_to_binary(output_svg_path, out_raster)

        # Match shapes (re-rasterize should already match since both use
        # output_width=RENDER_RES; resize as a safety net).
        if out_binary.shape != gt_binary.shape:
            out_binary = cv2.resize(out_binary, (gt_binary.shape[1], gt_binary.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        gt_ink  = (gt_binary  < 128)
        out_ink = (out_binary < 128)

        intersection = np.logical_and(gt_ink, out_ink).sum()
        union        = np.logical_or (gt_ink, out_ink).sum()
        row["iou_pixel"]       = float(intersection / union) if union > 0 else 0.0
        row["precision_pixel"] = float(intersection / out_ink.sum()) if out_ink.sum() > 0 else 0.0
        row["recall_pixel"]    = float(intersection /  gt_ink.sum()) if  gt_ink.sum() > 0 else 0.0

        gt_skel  = skeletonize(gt_ink)
        out_skel = skeletonize(out_ink)
        # Skeleton IoU is harsh because 1-px lines rarely overlap exactly;
        # use a small dilation tolerance.
        kernel   = np.ones((3, 3), np.uint8)
        gt_d     = cv2.dilate(gt_skel.astype(np.uint8),  kernel).astype(bool)
        out_d    = cv2.dilate(out_skel.astype(np.uint8), kernel).astype(bool)
        inter    = np.logical_and(gt_d, out_d).sum()
        uni      = np.logical_or (gt_d, out_d).sum()
        row["iou_skeleton"] = float(inter / uni) if uni > 0 else 0.0

        # ── Chamfer distance on skeletons ────────────────────────────────
        # Width-invariant: measures only "did the strokes land in the right
        # place", insensitive to stroke thickness mismatch. The IoU floor
        # of ~0.5 we saw on Drawing2CAD is dominated by thickness; chamfer
        # bypasses that and is the standard metric in the line-drawing-
        # vectorization literature.
        if gt_skel.any() and out_skel.any():
            # distance_transform_edt on ~mask gives per-pixel distance to
            # nearest True pixel in `mask` (in pixel units).
            dt_to_gt  = distance_transform_edt(~gt_skel)
            dt_to_out = distance_transform_edt(~out_skel)
            d_out2gt  = dt_to_gt[out_skel]          # for each output skel px,
                                                    #   dist to nearest gt skel
            d_gt2out  = dt_to_out[gt_skel]          # and the reverse
            row["chamfer_out2gt"]  = float(d_out2gt.mean())
            row["chamfer_gt2out"]  = float(d_gt2out.mean())
            row["chamfer_sym"]     = 0.5 * (row["chamfer_out2gt"] + row["chamfer_gt2out"])
            # 95th-percentile symmetric: catches outliers (mostly aligned,
            # but a few stray strokes far from any GT).
            all_d = np.concatenate([d_out2gt, d_gt2out])
            row["chamfer_p95_sym"] = float(np.percentile(all_d, 95))
        else:
            row["chamfer_out2gt"]  = None
            row["chamfer_gt2out"]  = None
            row["chamfer_sym"]     = None
            row["chamfer_p95_sym"] = None
    except Exception as exc:
        row["status"] = "metric"; row["error"] = f"{type(exc).__name__}: {exc}"

    row["total_time"] = time.perf_counter() - t0
    return row


# ─── Sample walker ───────────────────────────────────────────────────────────

def _load_split(d2c_root: Path) -> dict[str, list[str]]:
    with open(d2c_root / "train_val_test_split.json") as f:
        return json.load(f)


def _stratified_samples(d2c_root: Path, n: int, split: str,
                        views: tuple[str, ...], seed: int) -> list[tuple]:
    """
    Pick N random samples from the chosen split. Return (sample_id, view,
    svg_path) tuples, one per requested view per sample.
    """
    rng = random.Random(seed)
    sample_ids = list(_load_split(d2c_root)[split])
    rng.shuffle(sample_ids)
    picks = []
    svg_root = d2c_root / "svg_raw"
    for sid in sample_ids:
        outer, inner = sid.split("/")
        sample_dir = svg_root / outer / inner
        if not sample_dir.is_dir():
            continue
        for v in views:
            svg = sample_dir / f"{inner}_{v}.svg"
            if svg.exists():
                picks.append((sid, v, svg))
        if len({p[0] for p in picks}) >= n:
            break
    return picks


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drawing2CAD ground-truth pipeline evaluator."
    )
    parser.add_argument("--d2c-root", type=Path, default=D2C_ROOT_DEFAULT)
    parser.add_argument("--output",   type=Path, default=OUTPUT_DEFAULT,
                        help="Working dir for rasters + pipeline outputs.")
    parser.add_argument("--db",       type=Path, default=None,
                        help="SQLite DB. Default: <output>/d2c_results.db")
    parser.add_argument("--config",   type=Path,
                        default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--workers",  type=int, default=8)
    parser.add_argument("--limit",    type=int, default=100,
                        help="Number of samples (each may have multiple views).")
    parser.add_argument("--split",    choices=("train", "validation", "test"),
                        default="test")
    parser.add_argument("--views",    nargs="+", default=["Front"],
                        choices=list(VIEWS) + ["all"])
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    views = tuple(VIEWS) if args.views == ["all"] else tuple(args.views)
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.db or (args.output / "d2c_results.db")
    init_db(db_path)

    sketches = _stratified_samples(args.d2c_root, args.limit,
                                   args.split, views, args.seed)
    logger.info("Selected %d (sample, view) entries from %d distinct samples "
                "(split=%s).",
                len(sketches), len({s[0] for s in sketches}), args.split)

    if not args.no_resume:
        with sqlite3.connect(db_path) as conn:
            before = len(sketches)
            sketches = [s for s in sketches
                        if not already_processed(conn, s[0], s[1])]
            skipped = before - len(sketches)
            if skipped:
                logger.info("Resume: skipping %d already-processed entries.",
                            skipped)

    if not sketches:
        logger.info("Nothing to do.")
        return 0

    jobs = [
        (sid, view, str(svg), str(args.output / sid.replace("/", "_")))
        for (sid, view, svg) in sketches
    ]

    n_ok = n_err = 0
    with sqlite3.connect(db_path) as conn, \
         ProcessPoolExecutor(
             max_workers=args.workers,
             initializer=_worker_init,
             initargs=(str(args.config),),
         ) as pool:
        futures = [pool.submit(_process_one, j) for j in jobs]
        bar = tqdm(as_completed(futures), total=len(futures),
                   smoothing=0.1, mininterval=0.5)
        for fut in bar:
            try:
                row = fut.result()
            except Exception as exc:
                bar.write(f"Worker crashed: {exc!r}")
                n_err += 1
                continue
            insert_row(conn, row)
            if row["status"] == "ok":
                n_ok += 1
                ch = row.get("chamfer_sym")
                bar.set_postfix(ok=n_ok, err=n_err,
                                ch=f"{ch:.1f}" if ch is not None else "—")
            else:
                n_err += 1
                bar.set_postfix(ok=n_ok, err=n_err)

    _print_summary(db_path, n_ok, n_err)
    return 0


def _print_summary(db_path: Path, n_ok: int, n_err: int) -> None:
    """Print a results table with Chamfer distance as the headline metric."""
    W = 54
    sep = "─" * W

    with sqlite3.connect(db_path) as conn:
        def col(sql: str):
            r = conn.execute(sql).fetchone()
            return r[0] if r else None

        ch_vals = [r[0] for r in conn.execute(
            "SELECT chamfer_sym FROM d2c_results "
            "WHERE status='ok' AND chamfer_sym IS NOT NULL "
            "ORDER BY chamfer_sym"
        ).fetchall()]
        iou_px  = col("SELECT avg(iou_pixel)       FROM d2c_results WHERE status='ok'")
        iou_sk  = col("SELECT avg(iou_skeleton)    FROM d2c_results WHERE status='ok'")
        recall  = col("SELECT avg(recall_pixel)    FROM d2c_results WHERE status='ok'")
        prec    = col("SELECT avg(precision_pixel) FROM d2c_results WHERE status='ok'")
        n_total = col("SELECT count(*) FROM d2c_results") or 0

    def pct(vals, p):
        if not vals:
            return None
        return vals[int(len(vals) * p / 100)]

    ch_mean = sum(ch_vals) / len(ch_vals) if ch_vals else None
    ch_p50  = pct(ch_vals, 50)
    ch_p75  = pct(ch_vals, 75)
    ch_p95  = pct(ch_vals, 95)

    def fmt(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None else "—"

    print(f"\n{sep}")
    print(f"  Drawing2CAD eval — {n_ok}/{n_total} ok  {n_err} errors")
    print(sep)
    print(f"  Chamfer distance on skeletons (px, lower = better)")
    print(f"    mean   {fmt(ch_mean):>8}    p75  {fmt(ch_p75):>8}")
    print(f"    median {fmt(ch_p50):>8}    p95  {fmt(ch_p95):>8}")
    print(sep)
    print(f"  Secondary (pixel IoU)")
    print(f"    iou_pixel  {fmt(iou_px):>6}    iou_skel  {fmt(iou_sk):>6}")
    print(f"    recall     {fmt(recall):>6}    precision {fmt(prec):>6}")
    print(f"{sep}\n")


if __name__ == "__main__":
    raise SystemExit(main())
