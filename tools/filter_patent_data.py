"""
PatentData content filter — keeps only clean mechanical / electrical drawings.

Patent TIF corpora contain a mixture of:
  - mechanical assembly drawings  (KEEP)
  - electrical / circuit schematics (KEEP)
  - chemical structural formulas   (DISCARD)
  - mathematical equations         (DISCARD)
  - dense text paragraphs          (DISCARD)

Classification uses shape and skeleton features extracted from a downsampled
binary image (no ML model, no GPU needed):

  1. LETTER_CODE  — EPO patent filename convention. _C is often chemistry,
                    while _F / _A still need visual screening because the corpus
                    contains halftones, tables, dense labels, and hatch fields.
  2. LONG_LINE_SCORE — count of Hough segments in the downsampled image.
                    Drawings have many long straight strokes; chemistry bonds and
                    text glyphs are too short to register.
  3. LARGE_CC_FLAG   — whether any connected component has area ≥ 150 px².
                    Dense text has no large components; drawings do.
  4. TEXT_DENSITY    — foreground pixel fraction.
                    Dense (> 0.15) + no large CCs → text paragraph.

Decision tree is deliberately precision-oriented: it discards pages that look
like text/chemistry/halftone screens or that are too dense/hatched to become
good CAD training targets. The manifest records a reason code for audit.

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
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

logger = logging.getLogger("filter_patent_data")

# ─── Default thresholds (validated on 500-sample pilot) ──────────────────────
DEFAULT_SCALE_PX           = 1024  # keep halftone dots visible for screening
DEFAULT_MIN_LINE_PX        = 90    # minimum Hough segment length (downsampled px)
DEFAULT_LINE_THRESH        = 3     # ≥ this many long lines → drawing
DEFAULT_LARGE_CC_PX        = 150   # CC area threshold (downsampled px²)
DEFAULT_CC_NOISE_MAX       = 500   # more CCs than this → text page
DEFAULT_LARGE_CC_FRAC_THRESH = 0.40  # ≥ this fraction of fg in large CCs → chemistry
DEFAULT_DENS_LO            = 0.008 # fg fraction below this → near-blank
DEFAULT_HALFTONE_SKEL_CC_MAX = 4000
DEFAULT_HALFTONE_TINY_CC_FRAC = 0.90
DEFAULT_TEXT_DENS_MIN      = 0.030
DEFAULT_DENSE_HATCH_DENS_MIN = 0.18
DEFAULT_DENSE_HATCH_LARGE_FRAC_MIN = 0.75
DEFAULT_DENSE_TEXT_DENS_MIN = 0.10
DEFAULT_DENSE_TEXT_CC_MAX = 1500
DEFAULT_TINY_SKEL_CC_MAX = 1000
DEFAULT_TINY_SKEL_TINY_FRAC = 0.70
DEFAULT_MAX_CLEAN_DENSITY = 0.22
DEFAULT_ORTHO_LINE_MIN = 10
DEFAULT_ORTHO_HV_RATIO = 0.95
DEFAULT_ORTHO_DIAG_MAX = 0.03
DEFAULT_TEXT_HEAVY_CC_MIN = 180
DEFAULT_TEXT_HEAVY_SKEL_MEDIAN_MIN = 12.0
DEFAULT_TEXT_HEAVY_LARGE_FRAC_MAX = 0.75
DEFAULT_BLOCK_DIAGRAM_CC_MIN = 120
DEFAULT_BLOCK_DIAGRAM_LARGE_FRAC_MIN = 0.85
DEFAULT_BLOCK_DIAGRAM_DIAG_MIN = 0.12
DEFAULT_BLOCK_DIAGRAM_DIAG_MAX = 0.45
DEFAULT_CHART_CC_MIN = 350
DEFAULT_CHART_HV_RATIO = 0.85
DEFAULT_CHART_DIAG_MAX = 0.08
DEFAULT_CHART_TINY_FRAC_MIN = 0.55
DEFAULT_CHART_LINE_MIN = 150
DEFAULT_PLOT_FLOW_CC_MIN = 120
DEFAULT_PLOT_FLOW_LINE_MIN = 70
DEFAULT_PLOT_FLOW_LARGE_FRAC_MAX = 0.85
DEFAULT_PLOT_FLOW_DIAG_MAX = 0.22
DEFAULT_PLOT_FLOW_SKEL_MEDIAN_MIN = 15.0
DEFAULT_PLOT_FLOW_TINY_FRAC_MAX = 0.40
DEFAULT_SPARSE_PLOT_CC_MIN = 70
DEFAULT_SPARSE_PLOT_CC_MAX = 120
DEFAULT_SPARSE_PLOT_LINE_MAX = 35
DEFAULT_SPARSE_PLOT_LARGE_FRAC_MAX = 0.75
DEFAULT_SPARSE_PLOT_DIAG_MAX = 0.45
DEFAULT_SPARSE_PLOT_SKEL_MEDIAN_MIN = 20.0
DEFAULT_SPARSE_PLOT_TINY_FRAC_MAX = 0.25
DEFAULT_SINGLE_CURVE_CC_MAX = 80
DEFAULT_SINGLE_CURVE_LINE_MIN = 20
DEFAULT_SINGLE_CURVE_LARGE_FRAC_MIN = 0.90
DEFAULT_SINGLE_CURVE_DIAG_MAX = 0.10
DEFAULT_SINGLE_CURVE_SKEL_MEDIAN_MIN = 35.0
DEFAULT_SINGLE_CURVE_TINY_FRAC_MAX = 0.10
DEFAULT_MULTI_PANEL_PLOT_CC_MIN = 300
DEFAULT_MULTI_PANEL_PLOT_LARGE_FRAC_MAX = 0.65
DEFAULT_MULTI_PANEL_PLOT_DIAG_MIN = 0.20
DEFAULT_MULTI_PANEL_PLOT_DIAG_MAX = 0.45
DEFAULT_MULTI_PANEL_PLOT_SKEL_MEDIAN_MIN = 6.0
DEFAULT_MULTI_PANEL_PLOT_DENS_MAX = 0.05
DEFAULT_MULTI_PANEL_PLOT_TINY_FRAC_MAX = 0.60
DEFAULT_LINE_PLOT_CC_MIN = 90
DEFAULT_LINE_PLOT_CC_MAX = 220
DEFAULT_LINE_PLOT_LARGE_FRAC_MIN = 0.70
DEFAULT_LINE_PLOT_LARGE_FRAC_MAX = 0.88
DEFAULT_LINE_PLOT_HV_RATIO = 0.80
DEFAULT_LINE_PLOT_DIAG_MAX = 0.16
DEFAULT_LINE_PLOT_SKEL_MEDIAN_MIN = 12.0
DEFAULT_LINE_PLOT_TINY_FRAC_MAX = 0.25
DEFAULT_LINE_PLOT_DENS_MAX = 0.05
DEFAULT_TEXT_BOX_CC_MIN = 220
DEFAULT_TEXT_BOX_LARGE_FRAC_MAX = 0.75
DEFAULT_TEXT_BOX_HV_RATIO = 0.80
DEFAULT_TEXT_BOX_DIAG_MAX = 0.18
DEFAULT_TEXT_BOX_TINY_FRAC_MIN = 0.45


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
        rng=0,
    )
    long_lines = len(lines)
    line_angles = []
    line_lengths = []
    for p0, p1 in lines:
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        length = float((dx * dx + dy * dy) ** 0.5)
        angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        line_angles.append(angle)
        line_lengths.append(length)
    line_hv_ratio = (
        sum(1 for a in line_angles if a < 8.0 or a > 82.0) / long_lines
        if long_lines else 0.0
    )
    line_diag_ratio = (
        sum(1 for a in line_angles if 15.0 < a < 75.0) / long_lines
        if long_lines else 0.0
    )
    line_median_length = float(np.median(line_lengths)) if line_lengths else 0.0

    skel = skeletonize(binary)
    skel_labeled = label(skel, connectivity=2)
    skel_props = regionprops(skel_labeled)
    skel_areas = [r.area for r in skel_props]
    skel_n_cc = len(skel_areas)
    skel_tiny = sum(1 for area in skel_areas if area < 8)
    skel_tiny_frac = skel_tiny / skel_n_cc if skel_n_cc else 0.0
    skel_median_area = float(np.median(skel_areas)) if skel_areas else 0.0

    return {
        "text_density":   dens,
        "n_cc":           n_cc,
        "large_cc_frac":  large_cc_frac,
        "long_lines":     long_lines,
        "line_hv_ratio":  float(line_hv_ratio),
        "line_diag_ratio": float(line_diag_ratio),
        "line_median_length": line_median_length,
        "skel_density":   float(skel.mean()),
        "skel_n_cc":      skel_n_cc,
        "skel_tiny_cc_frac": skel_tiny_frac,
        "skel_median_cc_area": skel_median_area,
    }


def _classify(letter: str, feats: dict, cfg: dict) -> tuple[str, str]:
    """Return ('drawing'|'discard', reason_code)."""
    dens          = feats["text_density"]
    n_cc          = feats["n_cc"]
    large_cc_frac = feats["large_cc_frac"]
    lines         = feats["long_lines"]
    skel_n_cc     = feats.get("skel_n_cc") or 0
    skel_tiny_frac = feats.get("skel_tiny_cc_frac") or 0.0
    skel_median_area = feats.get("skel_median_cc_area") or 0.0
    line_diag_ratio = feats.get("line_diag_ratio", 0.0)

    # (a) Chemistry/formula figure buckets are too noisy for CAD vectorization.
    if letter == "C":
        return "discard", "letter_c_chemistry"
    if cfg["discard_d_bucket"] and letter == "D":
        return "discard", "letter_d_text_or_formula"

    # (b) Halftone/screenshots/dithered backgrounds explode into thousands of
    # tiny skeleton components.  These can still contain long lines and F/A
    # names, so this must run before the drawing-positive rules.
    if (skel_n_cc >= cfg["halftone_skel_cc_max"]
            and skel_tiny_frac >= cfg["halftone_tiny_cc_frac"]):
        return "discard", "halftone_many_tiny_components"

    # (c) High-precision training subset: heavily hatched/filled views and dense
    # text flowcharts may be real patent figures, but the current Stage 2/3 stack
    # turns them into thousands of low-value micro-primitives. Keep them out of
    # the automatic CAD target set until OCR/hatch/text-aware stages exist.
    if dens >= cfg["max_clean_density"]:
        return "discard", "too_dense_for_clean_cad"
    if (dens >= cfg["dense_hatch_dens_min"]
            and large_cc_frac >= cfg["dense_hatch_large_frac_min"]):
        return "discard", "dense_hatch_or_filled_region"
    if dens >= cfg["dense_text_dens_min"] and n_cc >= cfg["dense_text_cc_max"]:
        return "discard", "dense_text_or_flowchart"
    if (skel_n_cc >= cfg["tiny_skel_cc_max"]
            and skel_tiny_frac >= cfg["tiny_skel_tiny_frac"]):
        return "discard", "fragmented_tiny_skeleton_components"
    if (lines >= cfg["orthogonal_line_min"]
            and feats.get("line_hv_ratio", 0.0) >= cfg["orthogonal_hv_ratio"]
            and feats.get("line_diag_ratio", 0.0) <= cfg["orthogonal_diag_max"]):
        return "discard", "orthogonal_text_table_or_flowchart"
    if (n_cc >= cfg["text_heavy_cc_min"]
            and skel_median_area >= cfg["text_heavy_skel_median_min"]
            and large_cc_frac < cfg["text_heavy_large_frac_max"]):
        return "discard", "text_heavy_plot_table_or_block_diagram"
    if (n_cc >= cfg["block_diagram_cc_min"]
            and large_cc_frac >= cfg["block_diagram_large_frac_min"]
            and cfg["block_diagram_diag_min"]
            <= line_diag_ratio <= cfg["block_diagram_diag_max"]):
        return "discard", "block_diagram_text_boxes"
    if (n_cc >= cfg["chart_cc_min"]
            and line_diag_ratio <= cfg["chart_diag_max"]
            and skel_tiny_frac >= cfg["chart_tiny_frac_min"]
            and (feats.get("line_hv_ratio", 0.0) >= cfg["chart_hv_ratio"]
                 or lines >= cfg["chart_line_min"])):
        return "discard", "chart_or_axis_plot"
    if (n_cc >= cfg["plot_flow_cc_min"]
            and lines >= cfg["plot_flow_line_min"]
            and large_cc_frac <= cfg["plot_flow_large_frac_max"]
            and line_diag_ratio <= cfg["plot_flow_diag_max"]
            and skel_median_area >= cfg["plot_flow_skel_median_min"]
            and skel_tiny_frac <= cfg["plot_flow_tiny_frac_max"]):
        return "discard", "plot_or_flowchart"
    if (cfg["sparse_plot_cc_min"] <= n_cc < cfg["sparse_plot_cc_max"]
            and lines <= cfg["sparse_plot_line_max"]
            and large_cc_frac <= cfg["sparse_plot_large_frac_max"]
            and line_diag_ratio <= cfg["sparse_plot_diag_max"]
            and skel_median_area >= cfg["sparse_plot_skel_median_min"]
            and skel_tiny_frac <= cfg["sparse_plot_tiny_frac_max"]):
        return "discard", "sparse_scientific_plot"
    if (n_cc <= cfg["single_curve_cc_max"]
            and lines >= cfg["single_curve_line_min"]
            and large_cc_frac >= cfg["single_curve_large_frac_min"]
            and line_diag_ratio <= cfg["single_curve_diag_max"]
            and skel_median_area >= cfg["single_curve_skel_median_min"]
            and skel_tiny_frac <= cfg["single_curve_tiny_frac_max"]):
        return "discard", "single_axis_curve_plot"
    if (n_cc >= cfg["multi_panel_plot_cc_min"]
            and large_cc_frac <= cfg["multi_panel_plot_large_frac_max"]
            and cfg["multi_panel_plot_diag_min"]
            <= line_diag_ratio <= cfg["multi_panel_plot_diag_max"]
            and skel_median_area >= cfg["multi_panel_plot_skel_median_min"]
            and dens <= cfg["multi_panel_plot_dens_max"]
            and skel_tiny_frac <= cfg["multi_panel_plot_tiny_frac_max"]):
        return "discard", "multi_panel_scientific_plot"
    if (cfg["line_plot_cc_min"] <= n_cc <= cfg["line_plot_cc_max"]
            and cfg["line_plot_large_frac_min"]
            <= large_cc_frac <= cfg["line_plot_large_frac_max"]
            and feats.get("line_hv_ratio", 0.0) >= cfg["line_plot_hv_ratio"]
            and line_diag_ratio <= cfg["line_plot_diag_max"]
            and skel_median_area >= cfg["line_plot_skel_median_min"]
            and skel_tiny_frac <= cfg["line_plot_tiny_frac_max"]
            and dens <= cfg["line_plot_dens_max"]):
        return "discard", "scientific_line_plot"
    if (n_cc >= cfg["text_box_cc_min"]
            and large_cc_frac <= cfg["text_box_large_frac_max"]
            and feats.get("line_hv_ratio", 0.0) >= cfg["text_box_hv_ratio"]
            and line_diag_ratio <= cfg["text_box_diag_max"]
            and skel_tiny_frac >= cfg["text_box_tiny_frac_min"]):
        return "discard", "text_box_circuit_or_block_diagram"

    # (d) Text/formula snippets: no long engineering lines, no large drawing
    # components, but enough ink to be non-blank.
    if (lines < cfg["line_thresh"]
            and large_cc_frac < 0.05
            and dens >= cfg["text_dens_min"]):
        return "discard", "text_or_formula_snippet"

    # (e) Long engineering lines detected.
    if lines >= cfg["line_thresh"]:
        return "drawing", "long_engineering_lines"

    # (f) Hundreds of tiny glyphs → dense text paragraph
    if n_cc > cfg["cc_noise_max"]:
        return "discard", "many_connected_components"

    # (g) Compact blob structures with no long lines → chemistry / ring diagrams.
    # Most foreground ink is in a few large connected components (ring systems,
    # molecular graphs) rather than spread across many thin stroke segments.
    if large_cc_frac > cfg["large_cc_frac_thresh"] and lines == 0:
        return "discard", "compact_chemistry_blob"

    # (h) Near-blank page
    if dens < cfg["dens_lo"]:
        return "discard", "near_blank"

    # (i) Ambiguous but sparse enough to be worth trying.
    return "drawing", "sparse_ambiguous_drawing"


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
        "reason":       "load_error",
        "text_density":  None,
        "n_cc":          None,
        "large_cc_frac": None,
        "long_lines":    None,
        "line_hv_ratio": None,
        "line_diag_ratio": None,
        "line_median_length": None,
        "skel_density":  None,
        "skel_n_cc":     None,
        "skel_tiny_cc_frac": None,
        "skel_median_cc_area": None,
    }
    binary = _load_binary(path, cfg["scale_px"])
    if binary is None:
        return result
    feats = _extract_features(binary, cfg["min_line_px"], cfg["large_cc_px"])
    result.update(feats)
    result["label"], result["reason"] = _classify(letter, feats, cfg)
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
    p.add_argument("--halftone-skel-cc-max", type=int,
                   default=DEFAULT_HALFTONE_SKEL_CC_MAX)
    p.add_argument("--halftone-tiny-cc-frac", type=float,
                   default=DEFAULT_HALFTONE_TINY_CC_FRAC)
    p.add_argument("--text-dens-min", type=float,
                   default=DEFAULT_TEXT_DENS_MIN)
    p.add_argument("--dense-hatch-dens-min", type=float,
                   default=DEFAULT_DENSE_HATCH_DENS_MIN)
    p.add_argument("--dense-hatch-large-frac-min", type=float,
                   default=DEFAULT_DENSE_HATCH_LARGE_FRAC_MIN)
    p.add_argument("--dense-text-dens-min", type=float,
                   default=DEFAULT_DENSE_TEXT_DENS_MIN)
    p.add_argument("--dense-text-cc-max", type=int,
                   default=DEFAULT_DENSE_TEXT_CC_MAX)
    p.add_argument("--tiny-skel-cc-max", type=int,
                   default=DEFAULT_TINY_SKEL_CC_MAX)
    p.add_argument("--tiny-skel-tiny-frac", type=float,
                   default=DEFAULT_TINY_SKEL_TINY_FRAC)
    p.add_argument("--max-clean-density", type=float,
                   default=DEFAULT_MAX_CLEAN_DENSITY)
    p.add_argument("--orthogonal-line-min", type=int,
                   default=DEFAULT_ORTHO_LINE_MIN)
    p.add_argument("--orthogonal-hv-ratio", type=float,
                   default=DEFAULT_ORTHO_HV_RATIO)
    p.add_argument("--orthogonal-diag-max", type=float,
                   default=DEFAULT_ORTHO_DIAG_MAX)
    p.add_argument("--keep-d-bucket", action="store_true",
                   help="Keep _D patent buckets when visual features pass. "
                        "Default discards them for high-precision CAD targets.")
    p.add_argument("--text-heavy-cc-min", type=int,
                   default=DEFAULT_TEXT_HEAVY_CC_MIN)
    p.add_argument("--text-heavy-skel-median-min", type=float,
                   default=DEFAULT_TEXT_HEAVY_SKEL_MEDIAN_MIN)
    p.add_argument("--text-heavy-large-frac-max", type=float,
                   default=DEFAULT_TEXT_HEAVY_LARGE_FRAC_MAX)
    p.add_argument("--block-diagram-cc-min", type=int,
                   default=DEFAULT_BLOCK_DIAGRAM_CC_MIN)
    p.add_argument("--block-diagram-large-frac-min", type=float,
                   default=DEFAULT_BLOCK_DIAGRAM_LARGE_FRAC_MIN)
    p.add_argument("--block-diagram-diag-min", type=float,
                   default=DEFAULT_BLOCK_DIAGRAM_DIAG_MIN)
    p.add_argument("--block-diagram-diag-max", type=float,
                   default=DEFAULT_BLOCK_DIAGRAM_DIAG_MAX)
    p.add_argument("--chart-cc-min", type=int,
                   default=DEFAULT_CHART_CC_MIN)
    p.add_argument("--chart-hv-ratio", type=float,
                   default=DEFAULT_CHART_HV_RATIO)
    p.add_argument("--chart-diag-max", type=float,
                   default=DEFAULT_CHART_DIAG_MAX)
    p.add_argument("--chart-tiny-frac-min", type=float,
                   default=DEFAULT_CHART_TINY_FRAC_MIN)
    p.add_argument("--chart-line-min", type=int,
                   default=DEFAULT_CHART_LINE_MIN)
    p.add_argument("--plot-flow-cc-min", type=int,
                   default=DEFAULT_PLOT_FLOW_CC_MIN)
    p.add_argument("--plot-flow-line-min", type=int,
                   default=DEFAULT_PLOT_FLOW_LINE_MIN)
    p.add_argument("--plot-flow-large-frac-max", type=float,
                   default=DEFAULT_PLOT_FLOW_LARGE_FRAC_MAX)
    p.add_argument("--plot-flow-diag-max", type=float,
                   default=DEFAULT_PLOT_FLOW_DIAG_MAX)
    p.add_argument("--plot-flow-skel-median-min", type=float,
                   default=DEFAULT_PLOT_FLOW_SKEL_MEDIAN_MIN)
    p.add_argument("--plot-flow-tiny-frac-max", type=float,
                   default=DEFAULT_PLOT_FLOW_TINY_FRAC_MAX)
    p.add_argument("--sparse-plot-cc-min", type=int,
                   default=DEFAULT_SPARSE_PLOT_CC_MIN)
    p.add_argument("--sparse-plot-cc-max", type=int,
                   default=DEFAULT_SPARSE_PLOT_CC_MAX)
    p.add_argument("--sparse-plot-line-max", type=int,
                   default=DEFAULT_SPARSE_PLOT_LINE_MAX)
    p.add_argument("--sparse-plot-large-frac-max", type=float,
                   default=DEFAULT_SPARSE_PLOT_LARGE_FRAC_MAX)
    p.add_argument("--sparse-plot-diag-max", type=float,
                   default=DEFAULT_SPARSE_PLOT_DIAG_MAX)
    p.add_argument("--sparse-plot-skel-median-min", type=float,
                   default=DEFAULT_SPARSE_PLOT_SKEL_MEDIAN_MIN)
    p.add_argument("--sparse-plot-tiny-frac-max", type=float,
                   default=DEFAULT_SPARSE_PLOT_TINY_FRAC_MAX)
    p.add_argument("--single-curve-cc-max", type=int,
                   default=DEFAULT_SINGLE_CURVE_CC_MAX)
    p.add_argument("--single-curve-line-min", type=int,
                   default=DEFAULT_SINGLE_CURVE_LINE_MIN)
    p.add_argument("--single-curve-large-frac-min", type=float,
                   default=DEFAULT_SINGLE_CURVE_LARGE_FRAC_MIN)
    p.add_argument("--single-curve-diag-max", type=float,
                   default=DEFAULT_SINGLE_CURVE_DIAG_MAX)
    p.add_argument("--single-curve-skel-median-min", type=float,
                   default=DEFAULT_SINGLE_CURVE_SKEL_MEDIAN_MIN)
    p.add_argument("--single-curve-tiny-frac-max", type=float,
                   default=DEFAULT_SINGLE_CURVE_TINY_FRAC_MAX)
    p.add_argument("--multi-panel-plot-cc-min", type=int,
                   default=DEFAULT_MULTI_PANEL_PLOT_CC_MIN)
    p.add_argument("--multi-panel-plot-large-frac-max", type=float,
                   default=DEFAULT_MULTI_PANEL_PLOT_LARGE_FRAC_MAX)
    p.add_argument("--multi-panel-plot-diag-min", type=float,
                   default=DEFAULT_MULTI_PANEL_PLOT_DIAG_MIN)
    p.add_argument("--multi-panel-plot-diag-max", type=float,
                   default=DEFAULT_MULTI_PANEL_PLOT_DIAG_MAX)
    p.add_argument("--multi-panel-plot-skel-median-min", type=float,
                   default=DEFAULT_MULTI_PANEL_PLOT_SKEL_MEDIAN_MIN)
    p.add_argument("--multi-panel-plot-dens-max", type=float,
                   default=DEFAULT_MULTI_PANEL_PLOT_DENS_MAX)
    p.add_argument("--multi-panel-plot-tiny-frac-max", type=float,
                   default=DEFAULT_MULTI_PANEL_PLOT_TINY_FRAC_MAX)
    p.add_argument("--line-plot-cc-min", type=int,
                   default=DEFAULT_LINE_PLOT_CC_MIN)
    p.add_argument("--line-plot-cc-max", type=int,
                   default=DEFAULT_LINE_PLOT_CC_MAX)
    p.add_argument("--line-plot-large-frac-min", type=float,
                   default=DEFAULT_LINE_PLOT_LARGE_FRAC_MIN)
    p.add_argument("--line-plot-large-frac-max", type=float,
                   default=DEFAULT_LINE_PLOT_LARGE_FRAC_MAX)
    p.add_argument("--line-plot-hv-ratio", type=float,
                   default=DEFAULT_LINE_PLOT_HV_RATIO)
    p.add_argument("--line-plot-diag-max", type=float,
                   default=DEFAULT_LINE_PLOT_DIAG_MAX)
    p.add_argument("--line-plot-skel-median-min", type=float,
                   default=DEFAULT_LINE_PLOT_SKEL_MEDIAN_MIN)
    p.add_argument("--line-plot-tiny-frac-max", type=float,
                   default=DEFAULT_LINE_PLOT_TINY_FRAC_MAX)
    p.add_argument("--line-plot-dens-max", type=float,
                   default=DEFAULT_LINE_PLOT_DENS_MAX)
    p.add_argument("--text-box-cc-min", type=int,
                   default=DEFAULT_TEXT_BOX_CC_MIN)
    p.add_argument("--text-box-large-frac-max", type=float,
                   default=DEFAULT_TEXT_BOX_LARGE_FRAC_MAX)
    p.add_argument("--text-box-hv-ratio", type=float,
                   default=DEFAULT_TEXT_BOX_HV_RATIO)
    p.add_argument("--text-box-diag-max", type=float,
                   default=DEFAULT_TEXT_BOX_DIAG_MAX)
    p.add_argument("--text-box-tiny-frac-min", type=float,
                   default=DEFAULT_TEXT_BOX_TINY_FRAC_MIN)
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
        "halftone_skel_cc_max": args.halftone_skel_cc_max,
        "halftone_tiny_cc_frac": args.halftone_tiny_cc_frac,
        "text_dens_min":      args.text_dens_min,
        "dense_hatch_dens_min": args.dense_hatch_dens_min,
        "dense_hatch_large_frac_min": args.dense_hatch_large_frac_min,
        "dense_text_dens_min": args.dense_text_dens_min,
        "dense_text_cc_max": args.dense_text_cc_max,
        "tiny_skel_cc_max": args.tiny_skel_cc_max,
        "tiny_skel_tiny_frac": args.tiny_skel_tiny_frac,
        "max_clean_density": args.max_clean_density,
        "orthogonal_line_min": args.orthogonal_line_min,
        "orthogonal_hv_ratio": args.orthogonal_hv_ratio,
        "orthogonal_diag_max": args.orthogonal_diag_max,
        "discard_d_bucket": not args.keep_d_bucket,
        "text_heavy_cc_min": args.text_heavy_cc_min,
        "text_heavy_skel_median_min": args.text_heavy_skel_median_min,
        "text_heavy_large_frac_max": args.text_heavy_large_frac_max,
        "block_diagram_cc_min": args.block_diagram_cc_min,
        "block_diagram_large_frac_min": args.block_diagram_large_frac_min,
        "block_diagram_diag_min": args.block_diagram_diag_min,
        "block_diagram_diag_max": args.block_diagram_diag_max,
        "chart_cc_min": args.chart_cc_min,
        "chart_hv_ratio": args.chart_hv_ratio,
        "chart_diag_max": args.chart_diag_max,
        "chart_tiny_frac_min": args.chart_tiny_frac_min,
        "chart_line_min": args.chart_line_min,
        "plot_flow_cc_min": args.plot_flow_cc_min,
        "plot_flow_line_min": args.plot_flow_line_min,
        "plot_flow_large_frac_max": args.plot_flow_large_frac_max,
        "plot_flow_diag_max": args.plot_flow_diag_max,
        "plot_flow_skel_median_min": args.plot_flow_skel_median_min,
        "plot_flow_tiny_frac_max": args.plot_flow_tiny_frac_max,
        "sparse_plot_cc_min": args.sparse_plot_cc_min,
        "sparse_plot_cc_max": args.sparse_plot_cc_max,
        "sparse_plot_line_max": args.sparse_plot_line_max,
        "sparse_plot_large_frac_max": args.sparse_plot_large_frac_max,
        "sparse_plot_diag_max": args.sparse_plot_diag_max,
        "sparse_plot_skel_median_min": args.sparse_plot_skel_median_min,
        "sparse_plot_tiny_frac_max": args.sparse_plot_tiny_frac_max,
        "single_curve_cc_max": args.single_curve_cc_max,
        "single_curve_line_min": args.single_curve_line_min,
        "single_curve_large_frac_min": args.single_curve_large_frac_min,
        "single_curve_diag_max": args.single_curve_diag_max,
        "single_curve_skel_median_min": args.single_curve_skel_median_min,
        "single_curve_tiny_frac_max": args.single_curve_tiny_frac_max,
        "multi_panel_plot_cc_min": args.multi_panel_plot_cc_min,
        "multi_panel_plot_large_frac_max": args.multi_panel_plot_large_frac_max,
        "multi_panel_plot_diag_min": args.multi_panel_plot_diag_min,
        "multi_panel_plot_diag_max": args.multi_panel_plot_diag_max,
        "multi_panel_plot_skel_median_min": args.multi_panel_plot_skel_median_min,
        "multi_panel_plot_dens_max": args.multi_panel_plot_dens_max,
        "multi_panel_plot_tiny_frac_max": args.multi_panel_plot_tiny_frac_max,
        "line_plot_cc_min": args.line_plot_cc_min,
        "line_plot_cc_max": args.line_plot_cc_max,
        "line_plot_large_frac_min": args.line_plot_large_frac_min,
        "line_plot_large_frac_max": args.line_plot_large_frac_max,
        "line_plot_hv_ratio": args.line_plot_hv_ratio,
        "line_plot_diag_max": args.line_plot_diag_max,
        "line_plot_skel_median_min": args.line_plot_skel_median_min,
        "line_plot_tiny_frac_max": args.line_plot_tiny_frac_max,
        "line_plot_dens_max": args.line_plot_dens_max,
        "text_box_cc_min": args.text_box_cc_min,
        "text_box_large_frac_max": args.text_box_large_frac_max,
        "text_box_hv_ratio": args.text_box_hv_ratio,
        "text_box_diag_max": args.text_box_diag_max,
        "text_box_tiny_frac_min": args.text_box_tiny_frac_min,
    }

    all_tifs = list(_iter_tifs(patent_root))
    if args.limit > 0:
        all_tifs = all_tifs[:args.limit]
    total = len(all_tifs)
    logger.info("Found %d TIFs under %s", total, patent_root)

    tasks = [(str(t), cfg) for t in all_tifs]

    counts = {"drawing": 0, "discard": 0, "error": 0}
    fieldnames = ["path", "patent", "filename", "letter_code", "label", "reason",
                  "text_density", "n_cc", "large_cc_frac", "long_lines",
                  "line_hv_ratio", "line_diag_ratio", "line_median_length",
                  "skel_density", "skel_n_cc", "skel_tiny_cc_frac",
                  "skel_median_cc_area"]

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
