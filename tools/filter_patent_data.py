"""
PatentData content filter — keeps only mechanical / electrical drawings.

Patent TIF corpora contain a mixture of:
  - mechanical assembly drawings  (KEEP)
  - electrical / circuit schematics (KEEP)
  - chemical structural formulas   (DISCARD)
  - mathematical equations         (DISCARD)
  - dense text paragraphs          (DISCARD)

Classification uses four features extracted from a 512-px downsampled
binary image (no ML model, no GPU needed):

  1. LETTER_CODE  — EPO patent filename convention:
                    _F / _A → almost always a drawing figure (96 %+ empirically)
                    _C      → often chemistry formula
                    _D      → mixed; needs feature analysis
  2. LONG_LINE_SCORE — count of Hough segments ≥ 60 px in the downsampled image.
                    Drawings have many long straight strokes; chemistry bonds and
                    text glyphs are too short to register.
  3. LARGE_CC_FLAG   — whether any connected component has area ≥ 150 px².
                    Dense text has no large components; drawings do.
  4. TEXT_DENSITY    — foreground pixel fraction.
                    Dense (> 0.15) + no large CCs → text paragraph.

Decision tree (evaluated top-to-bottom; first match wins):
  a. letter_code ∈ {F, A}         → drawing   (high-confidence from naming)
  b. long_lines  ≥ LINE_THRESH     → drawing   (long strokes present)
  c. n_cc        > CC_NOISE_MAX    → discard   (hundreds of tiny glyphs = text)
  d. n_large==0 AND dens > DENS_HI → discard   (dense text, no large strokes)
  e. dens        < DENS_LO         → discard   (near-blank, nothing useful)
  f. default                       → drawing   (conservative — keep ambiguous cases)

Results are written to a CSV manifest (one row per TIF). An optional
``--move`` flag physically moves discarded files to a quarantine folder
so batch_run.py naturally ignores them.

Usage (from project root):

    # Dry run — produce manifest only, no files moved
    python -m tools.filter_patent_data

    # Move discarded files to data/PatentData/quarantine/
    python -m tools.filter_patent_data --move

    # Pilot on first 1000 TIFs, 4 workers
    python -m tools.filter_patent_data --limit 1000 --workers 4

    # Custom paths / thresholds
    python -m tools.filter_patent_data \\
        --patent-root data/PatentData/ReorganisedData \\
        --output      output/PatentData/filter_manifest.csv \\
        --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from skimage.transform import probabilistic_hough_line

logger = logging.getLogger("filter_patent_data")

# ─── Default thresholds (validated on 500-sample pilot) ──────────────────────
DEFAULT_SCALE_PX           = 512   # downsample long edge to this size
DEFAULT_MIN_LINE_PX        = 60    # minimum Hough segment length (downsampled px)
DEFAULT_LINE_THRESH        = 2     # ≥ this many long lines → drawing
DEFAULT_LARGE_CC_PX        = 150   # CC area threshold (downsampled px²)
DEFAULT_CC_NOISE_MAX       = 500   # more CCs than this → text page
DEFAULT_LARGE_CC_FRAC_THRESH = 0.40  # ≥ this fraction of fg in large CCs → chemistry
DEFAULT_DENS_LO            = 0.008 # fg fraction below this → near-blank


# ─── Feature extraction ───────────────────────────────────────────────────────

def _load_binary(path: Path, scale: int) -> np.ndarray | None:
    """Return boolean ndarray (True = ink) downsampled to ~scale px long edge."""
    try:
        img = Image.open(path).convert("L")
        w, h = img.size
        factor = max(w, h) / scale
        if factor > 1.0:
            nw = max(1, round(w / factor))
            nh = max(1, round(h / factor))
            img = img.resize((nw, nh), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        return arr < 128          # ink (dark pixels) = True
    except Exception:
        return None


def _letter_code(filename: str) -> str:
    """Return the EPO figure-type letter (F / A / C / D / …) or '' if absent."""
    m = re.search(r"_([A-Z])\d+\.tif$", filename, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _extract_features(binary: np.ndarray, min_line_px: int,
                      large_cc_px: int) -> dict:
    total_px = binary.size
    fg_px    = int(binary.sum())
    dens     = fg_px / total_px if total_px else 0.0

    labeled  = label(binary, connectivity=2)
    props    = regionprops(labeled)
    n_cc     = len(props)
    large_area = sum(r.area for r in props if r.area >= large_cc_px)
    # Fraction of foreground pixels in large CCs.
    # Chemistry ring structures: a few large blobs → high fraction.
    # Mechanical drawings: many separate strokes → lower fraction.
    large_cc_frac = large_area / fg_px if fg_px else 0.0

    uint8    = binary.astype(np.uint8) * 255
    lines    = probabilistic_hough_line(
        uint8,
        threshold=10,
        line_length=min_line_px,
        line_gap=4,
    )
    long_lines = len(lines)

    return {
        "text_density":   dens,
        "n_cc":           n_cc,
        "large_cc_frac":  large_cc_frac,
        "long_lines":     long_lines,
    }


def _classify(letter: str, feats: dict, cfg: dict) -> str:
    """Return 'drawing' or 'discard'."""
    dens          = feats["text_density"]
    n_cc          = feats["n_cc"]
    large_cc_frac = feats["large_cc_frac"]
    lines         = feats["long_lines"]

    # (a) High-confidence drawing by filename convention (_F / _A figures)
    if letter in ("F", "A"):
        return "drawing"

    # (b) Long engineering lines detected (≥ 60 px in 512-px downsampled image)
    if lines >= cfg["line_thresh"]:
        return "drawing"

    # (c) Hundreds of tiny glyphs → dense text paragraph
    if n_cc > cfg["cc_noise_max"]:
        return "discard"

    # (d) Compact blob structures with no long lines → chemistry / ring diagrams.
    # Most foreground ink is in a few large connected components (ring systems,
    # molecular graphs) rather than spread across many thin stroke segments.
    if large_cc_frac > cfg["large_cc_frac_thresh"] and lines == 0:
        return "discard"

    # (e) Near-blank page
    if dens < cfg["dens_lo"]:
        return "discard"

    # (f) Conservative default: keep ambiguous cases rather than lose drawings
    return "drawing"


# ─── Worker ───────────────────────────────────────────────────────────────────

def _process_tif(args: tuple) -> dict:
    tif_path_str, cfg = args
    path   = Path(tif_path_str)
    letter = _letter_code(path.name)
    result = {
        "path":         str(path),
        "patent":       path.parent.name,
        "filename":     path.name,
        "letter_code":  letter,
        "label":        "error",
        "text_density":  None,
        "n_cc":          None,
        "large_cc_frac": None,
        "long_lines":    None,
    }
    binary = _load_binary(path, cfg["scale_px"])
    if binary is None:
        return result
    # Fast-path: F/A files need no expensive feature extraction
    if letter in ("F", "A"):
        feats = {"text_density": None, "n_cc": None,
                 "large_cc_frac": None, "long_lines": None}
        result.update(feats)
        result["label"] = "drawing"
        return result

    feats = _extract_features(binary, cfg["min_line_px"], cfg["large_cc_px"])
    result.update(feats)
    result["label"] = _classify(letter, feats, cfg)
    return result


# ─── Discovery ────────────────────────────────────────────────────────────────

def _iter_tifs(root: Path):
    for patent_dir in sorted(root.iterdir()):
        if not patent_dir.is_dir():
            continue
        for tif in sorted(patent_dir.glob("*.tif")):
            yield tif


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter PatentData TIFs — keep only mechanical/electrical drawings."
    )
    p.add_argument("--patent-root",
                   default="data/PatentData/ReorganisedData",
                   help="Root of patent subdirectory tree.")
    p.add_argument("--output",
                   default="output/PatentData/filter_manifest.csv",
                   help="Path for the output CSV manifest.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes.")
    p.add_argument("--move", action="store_true",
                   help="Move discarded TIFs to <patent-root>/../quarantine/.")
    p.add_argument("--quarantine", default=None,
                   help="Quarantine directory (default: patent-root/../quarantine).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process at most this many TIFs (0 = all). For pilots.")
    p.add_argument("--scale-px",          type=int,   default=DEFAULT_SCALE_PX)
    p.add_argument("--min-line-px",       type=int,   default=DEFAULT_MIN_LINE_PX)
    p.add_argument("--line-thresh",       type=int,   default=DEFAULT_LINE_THRESH)
    p.add_argument("--large-cc-px",       type=int,   default=DEFAULT_LARGE_CC_PX)
    p.add_argument("--cc-noise-max",      type=int,   default=DEFAULT_CC_NOISE_MAX)
    p.add_argument("--large-cc-frac-thresh", type=float, default=DEFAULT_LARGE_CC_FRAC_THRESH)
    p.add_argument("--dens-lo",           type=float, default=DEFAULT_DENS_LO)
    p.add_argument("--log-level",         default="INFO")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    project_root = Path(__file__).resolve().parent.parent
    patent_root  = (project_root / args.patent_root).resolve()
    out_path     = (project_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quarantine = None
    if args.move:
        qdir = args.quarantine or str(patent_root.parent / "quarantine")
        quarantine = Path(qdir).resolve()
        quarantine.mkdir(parents=True, exist_ok=True)
        logger.info("Quarantine dir: %s", quarantine)

    cfg = {
        "scale_px":           args.scale_px,
        "min_line_px":        args.min_line_px,
        "line_thresh":        args.line_thresh,
        "large_cc_px":        args.large_cc_px,
        "cc_noise_max":       args.cc_noise_max,
        "large_cc_frac_thresh": args.large_cc_frac_thresh,
        "dens_lo":            args.dens_lo,
    }

    all_tifs = list(_iter_tifs(patent_root))
    if args.limit > 0:
        all_tifs = all_tifs[:args.limit]
    total = len(all_tifs)
    logger.info("Found %d TIFs under %s", total, patent_root)

    tasks = [(str(t), cfg) for t in all_tifs]

    counts = {"drawing": 0, "discard": 0, "error": 0}
    fieldnames = ["path", "patent", "filename", "letter_code", "label",
                  "text_density", "n_cc", "large_cc_frac", "long_lines"]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        workers = max(1, args.workers)
        done    = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_process_tif, t): t for t in tasks}
            try:
                from tqdm import tqdm as _tqdm
                pbar = _tqdm(total=total, unit="tif")
            except ImportError:
                pbar = None

            for fut in as_completed(futs):
                row = fut.result()
                writer.writerow(row)
                fh.flush()
                lbl = row["label"]
                counts[lbl] = counts.get(lbl, 0) + 1
                done += 1

                if quarantine and lbl == "discard":
                    src = Path(row["path"])
                    dst = quarantine / src.parent.name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(draw=counts["drawing"],
                                     disc=counts["discard"],
                                     err=counts["error"])
                elif done % 2000 == 0:
                    logger.info(
                        "%d / %d  drawing=%d  discard=%d  error=%d",
                        done, total,
                        counts["drawing"], counts["discard"], counts["error"],
                    )

            if pbar:
                pbar.close()

    total_classified = counts["drawing"] + counts["discard"]
    keep_pct = 100 * counts["drawing"] / total_classified if total_classified else 0
    logger.info(
        "Done. drawing=%d  discard=%d  error=%d  keep=%.1f%%",
        counts["drawing"], counts["discard"], counts["error"], keep_pct,
    )
    logger.info("Manifest written to %s", out_path)

    # Summary by letter code when running over a sample
    if args.limit > 0:
        logger.info("(pilot run — re-run without --limit to process full corpus)")


if __name__ == "__main__":
    main()
