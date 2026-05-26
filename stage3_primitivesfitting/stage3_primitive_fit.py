"""
stage3_primitive_fit_final.py
=============================
AP3 Vectorization Pipeline — Stage 3: Primitive Fitting (RANSAC-only, final)

Converts each stroke edge from Stage 2's graph JSON into a geometric
primitive: line, arc, circle, ellipse, or polyline. Uses a single
deterministic RANSAC pipeline — no learned model.

Decision history:
  Earlier iterations explored a Free2CAD Transformer (v1 → v2 → v3 + a
  hybrid fast path). The investigation is documented in
  FREE2CAD_HANDOFF.md and FREE2CAD_RESULTS.md. Final-call evaluation on
  Picture1_skeleton_graph.json:

    fitter        time      type-agreement    LINE param accuracy
    --------      ------    --------------    -------------------
    free2cad-v3   1.55 s    86.9 % vs RANSAC  median 0.97 of edge span
    ransac        0.26 s    reference         exact by construction

  RANSAC remains the production choice because it is geometrically exact
  on closed loops, ~6× faster, and has no synthetic-vs-real distribution
  gap. The Free2CAD path is parked, not deleted — see the handoff doc if
  the model is to be revisited.

Priority order in fit_edge_ransac:
  circle (closed) → line → arc → ellipse → polyline (final fallback)

Confidence:
  inlier_ratio × max(0, 1 − rms / MAX_RMS)

Output per sketch:
  output/primitives/<sketch_id>_primitives.json

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ─── Output contract ─────────────────────────────────────────────────────────

@dataclass
class Stage3Result:
    sketch_id:         str
    primitives_path:   Path
    mean_confidence:   float
    flagged:           bool
    processing_time_s: float
    n_primitives:      int


# ═══════════════════════════════════════════════════════════════════════════
# RANSAC CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_INLIER_DIST_LINE    = 1.5   # px — perpendicular distance threshold
_INLIER_DIST_CIRCLE  = 1.5   # px — radial distance threshold
_INLIER_DIST_ELLIPSE = 2.0   # px — approximate algebraic distance
_MAX_RMS             = 3.0   # px — RMS above this → confidence = 0

_MIN_PTS_LINE    = 2
_MIN_PTS_CIRCLE  = 4
_MIN_PTS_ARC     = 5
_MIN_PTS_ELLIPSE = 6

_CONF_THRESH_LINE    = 0.75
_CONF_THRESH_CIRCLE  = 0.65
_CONF_THRESH_ARC     = 0.65
_CONF_THRESH_ELLIPSE = 0.55


# ═══════════════════════════════════════════════════════════════════════════
# RANSAC FALLBACK — pure NumPy / SciPy
# ═══════════════════════════════════════════════════════════════════════════

def _confidence(residuals: np.ndarray, inlier_thresh: float) -> float:
    """confidence = inlier_ratio × max(0, 1 − rms / MAX_RMS)"""
    inlier_ratio = float((residuals <= inlier_thresh).mean())
    rms          = float(np.sqrt((residuals ** 2).mean()))
    return inlier_ratio * max(0.0, 1.0 - rms / _MAX_RMS)


# ── Line ──────────────────────────────────────────────────────────────────────

def _fit_line_ransac(pts: np.ndarray) -> dict:
    """
    SVD-based total-least-squares line fit.
    Returns {'type','start','end','confidence'}.
    """
    if len(pts) < _MIN_PTS_LINE:
        raise ValueError(f"Need ≥{_MIN_PTS_LINE} pts for line")

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    if len(pts) == 2:
        diff = centered[1] - centered[0]
        n    = np.linalg.norm(diff)
        direction = diff / n if n > 1e-10 else np.array([1.0, 0.0])
    else:
        _, _, Vt  = np.linalg.svd(centered, full_matrices=False)
        direction = Vt[0]   # principal component

    t         = centered @ direction
    normal    = np.array([-direction[1], direction[0]])
    residuals = np.abs(centered @ normal)
    conf      = _confidence(residuals, _INLIER_DIST_LINE)

    # Endpoints from inlier projections
    mask  = residuals <= _INLIER_DIST_LINE
    t_sel = t[mask] if mask.sum() >= 2 else t
    start = centroid + float(t_sel.min()) * direction
    end   = centroid + float(t_sel.max()) * direction

    return {
        "type":       "line",
        "p1":         [float(start[0]), float(start[1])],
        "p2":         [float(end[0]),   float(end[1])],
        "confidence": conf,
    }


# ── Circle ────────────────────────────────────────────────────────────────────

def _fit_circle_algebraic(pts: np.ndarray) -> tuple[float, float, float]:
    """
    Algebraic circle fit (least squares on x²+y²+ax+by+c=0).
    Returns (cx, cy, r).
    """
    x, y = pts[:, 0], pts[:, 1]
    A    = np.column_stack([x, y, np.ones(len(x))])
    b    = -(x ** 2 + y ** 2)
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_c, c = coeffs
    cx, cy    = -a / 2.0, -b_c / 2.0
    r_sq      = cx ** 2 + cy ** 2 - c
    if r_sq <= 1e-6:
        raise ValueError("Circle fit: radius² ≤ 0")
    return float(cx), float(cy), float(np.sqrt(r_sq))


def _fit_circle_ransac(pts: np.ndarray) -> dict:
    """Returns {'type','center','radius','confidence'}."""
    if len(pts) < _MIN_PTS_CIRCLE:
        raise ValueError(f"Need ≥{_MIN_PTS_CIRCLE} pts for circle")
    cx, cy, r = _fit_circle_algebraic(pts)
    residuals = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    return {
        "type":       "circle",
        "center":     [cx, cy],
        "radius":     float(r),
        "confidence": _confidence(residuals, _INLIER_DIST_CIRCLE),
    }


# ── Arc ───────────────────────────────────────────────────────────────────────

def _arc_angles(pts: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    """
    Extract (start_angle, end_angle) in degrees [0, 360) from points on an arc.
    Finds the largest angular gap — the complement of that gap is the arc span.
    """
    angles       = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360.0
    angles_sorted = np.sort(angles)
    diffs         = np.diff(np.append(angles_sorted, angles_sorted[0] + 360.0))
    gap_idx       = int(np.argmax(diffs))
    start = float(angles_sorted[(gap_idx + 1) % len(angles_sorted)])
    end   = float(angles_sorted[gap_idx])
    return start, end


def _fit_arc_ransac(pts: np.ndarray) -> dict:
    """Returns {'type','center','radius','start_angle','end_angle','confidence'}."""
    if len(pts) < _MIN_PTS_ARC:
        raise ValueError(f"Need ≥{_MIN_PTS_ARC} pts for arc")
    cx, cy, r = _fit_circle_algebraic(pts)
    residuals  = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    conf       = _confidence(residuals, _INLIER_DIST_CIRCLE)
    # Use inliers for angle extraction to avoid endpoint noise
    mask  = residuals <= _INLIER_DIST_CIRCLE
    p_sel = pts[mask] if mask.sum() >= _MIN_PTS_ARC else pts
    start_angle, end_angle = _arc_angles(p_sel, cx, cy)
    return {
        "type":        "arc",
        "center":      [cx, cy],
        "radius":      float(r),
        "start_angle": start_angle,
        "end_angle":   end_angle,
        "confidence":  conf,
    }


# ── Ellipse ───────────────────────────────────────────────────────────────────

def _fit_ellipse_algebraic(pts: np.ndarray) -> dict:
    """
    Fitzgibbon (1996) constrained algebraic ellipse fit.
    Constraint 4ac − b² = 1 guarantees an ellipse (not hyperbola/parabola).
    Returns {'cx','cy','a','b','angle'} in pixel coordinates.
    """
    if len(pts) < _MIN_PTS_ELLIPSE:
        raise ValueError(f"Need ≥{_MIN_PTS_ELLIPSE} pts for ellipse")

    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)

    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones(len(x))])
    S = D.T @ D

    # Constraint matrix: 4ac − b² = 1
    C         = np.zeros((6, 6))
    C[0, 2]   = C[2, 0] = 2.0
    C[1, 1]   = -1.0

    try:
        from scipy.linalg import eig as scipy_eig
        evals, evecs = scipy_eig(C, S)
    except Exception as exc:
        raise ValueError(f"Ellipse eigendecomp failed: {exc}")

    evals = evals.real
    evecs = evecs.real
    pos   = np.isfinite(evals) & (evals > 1e-10)
    if not pos.any():
        raise ValueError("No positive eigenvalue — not an ellipse")

    coeffs = evecs[:, np.where(pos)[0][np.argmin(evals[pos])]]
    a, b, c, d, e, f = coeffs

    denom = b ** 2 - 4 * a * c
    if denom >= -1e-10:
        raise ValueError(f"Discriminant {denom:.4g} ≥ 0 — not an ellipse")

    cx = (2 * c * d - b * e) / denom
    cy = (2 * a * e - b * d) / denom

    # Semi-axes from eigenvalues of the shape matrix [[a, b/2],[b/2, c]]
    # and the conic value F₀ = F(cx, cy)
    F0 = a*cx**2 + b*cx*cy + c*cy**2 + d*cx + e*cy + f
    M  = np.array([[a, b / 2.0], [b / 2.0, c]])
    lam, vecs = np.linalg.eigh(M)   # lam[0] ≤ lam[1] for symmetric M

    if np.any(lam == 0):
        raise ValueError("Degenerate shape matrix (zero eigenvalue)")

    ax_sq = -F0 / lam
    if np.any(ax_sq <= 0):
        # Try flipping sign convention
        ax_sq = F0 / lam
    if np.any(ax_sq <= 0) or not np.all(np.isfinite(ax_sq)):
        raise ValueError("Invalid ellipse semi-axes")

    axes  = np.sqrt(ax_sq)          # [minor_or_major, major_or_minor]
    idx_major = int(np.argmax(axes))
    semi_major = float(axes[idx_major])
    semi_minor = float(axes[1 - idx_major])
    major_vec  = vecs[:, idx_major]
    angle      = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))

    return {"cx": float(cx), "cy": float(cy),
            "a": semi_major, "b": semi_minor, "angle": angle}


def _fit_ellipse_ransac(pts: np.ndarray) -> dict:
    """Returns {'type','center','a','b','angle','confidence'}."""
    params    = _fit_ellipse_algebraic(pts)
    cx, cy    = params["cx"], params["cy"]
    sa, sb    = params["a"],  params["b"]
    angle_rad = np.radians(params["angle"])

    # Rotate points into ellipse frame and compute approximate distance to rim
    ca, sa_ = np.cos(-angle_rad), np.sin(-angle_rad)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    xr =  ca * dx + sa_ * dy
    yr = -sa_ * dx + ca * dy
    # Distance proxy: deviation of normalised radius from 1, scaled by minor axis
    norm_r    = np.sqrt((xr / sa) ** 2 + (yr / sb) ** 2)
    residuals = np.abs(norm_r - 1.0) * min(sa, sb)
    conf      = _confidence(residuals, _INLIER_DIST_ELLIPSE)

    return {
        "type":       "ellipse",
        "center":     [cx, cy],
        "a":          sa,
        "b":          sb,
        "angle":      params["angle"],
        "confidence": conf,
    }


# ── Closed-loop pixel reordering ──────────────────────────────────────────────

def _reorder_loop_pixels(pixels) -> np.ndarray:
    """
    Reorder a closed-loop's pixel list into topological traversal order.

    Stage 2 stores closed-loop pixels in scanline-like order (each column
    sweep contains both the top and bottom edges of the loop at that X).
    The circle/ellipse fits are insensitive to order, but the polyline
    fallback emits a wildly zigzagging shape if fed scanline-ordered points
    (confirmed on Drawing2CAD samples — a clean hexagon outline rendered
    as a single 397-point polyline criss-crossing the interior).

    Greedy nearest-neighbour walk from the first pixel. O(N²) — fine for
    loops with up to a few thousand pixels.
    """
    n = len(pixels)
    if n < 3:
        return np.asarray(pixels, dtype=np.float64)
    pts    = np.asarray(pixels, dtype=np.float64)
    used   = np.zeros(n, dtype=bool)
    order  = [0]
    used[0] = True
    last   = pts[0]
    for _ in range(n - 1):
        diff      = pts - last
        d2        = np.einsum("ij,ij->i", diff, diff)
        d2[used]  = np.inf
        nxt       = int(d2.argmin())
        order.append(nxt)
        used[nxt] = True
        last      = pts[nxt]
    return pts[order]


# ── Priority selector ─────────────────────────────────────────────────────────

def fit_edge_ransac(edge: dict) -> dict:
    """
    Fit one edge with the priority order:
      circle (closed) → line → arc → ellipse → polyline

    Uses edge["pixels"] for closed loops — smooth_pts ordering for closed
    contours follows coordinate sort, not arc angle, causing unreliable splines.
    """
    edge_id   = edge["id"]
    is_closed = edge.get("is_closed", False)

    if is_closed:
        # Reorder pixels into topological loop traversal (Stage 2 emits
        # them scanline-ordered for closed loops; the polyline fallback
        # would otherwise zigzag through the interior).
        pts = _reorder_loop_pixels(edge["pixels"])
    else:
        raw = edge.get("smooth_pts") or []
        pts = (np.array(raw, dtype=np.float64) if raw
               else np.array(edge["pixels"], dtype=np.float64))

    # ── Degenerate guard ─────────────────────────────────────────────────────
    if len(pts) < 2:
        return {"edge_id": edge_id, "type": "polyline",
                "points": [[float(p[0]), float(p[1])] for p in pts],
                "confidence": 0.0}

    # ── Closed-loop branch: circle → ellipse → polyline ──────────────────────
    # Open-edge cascade (line / arc / …) doesn't apply: a forced line or arc
    # fit through a closed contour is structurally nonsense. Before the fix
    # any closed edge — including irregular outlines — was emitted as a
    # circle regardless of fit quality, producing visible phantom circles in
    # the SVG (66% of all circles in the pilot had confidence < 0.60).
    if is_closed:
        if len(pts) >= _MIN_PTS_CIRCLE:
            try:
                r = _fit_circle_ransac(pts)
                r["edge_id"] = edge_id
                if r["confidence"] >= _CONF_THRESH_CIRCLE:
                    return r
            except ValueError:
                pass
        if len(pts) >= _MIN_PTS_ELLIPSE:
            try:
                r = _fit_ellipse_ransac(pts)
                r["edge_id"] = edge_id
                if r["confidence"] >= _CONF_THRESH_ELLIPSE:
                    return r
            except ValueError:
                pass
        # Use topologically-ordered pts (smooth_pts is unreliable here
        # because it is a non-periodic cubic spline through scanline-ordered
        # raw pixels). Append the first point so the polyline visually
        # closes the loop.
        poly_points = [[float(p[0]), float(p[1])] for p in pts]
        if poly_points and poly_points[0] != poly_points[-1]:
            poly_points.append(poly_points[0])
        return {
            "edge_id":    edge_id,
            "type":       "polyline",
            "points":     poly_points,
            "confidence": 0.3,
        }

    # ── Open-edge cascade: line → arc → ellipse → best-candidate → polyline ─
    line_result = None
    if len(pts) >= _MIN_PTS_LINE:
        try:
            r = _fit_line_ransac(pts)
            r["edge_id"] = edge_id
            if r["confidence"] >= _CONF_THRESH_LINE:
                return r
            line_result = r
        except ValueError:
            pass

    # ── Priority 3: Arc ───────────────────────────────────────────────────────
    arc_result = None
    if len(pts) >= _MIN_PTS_ARC:
        try:
            r = _fit_arc_ransac(pts)
            r["edge_id"] = edge_id
            if r["confidence"] >= _CONF_THRESH_ARC:
                return r
            arc_result = r
        except ValueError:
            pass

    # ── Priority 4: Ellipse ───────────────────────────────────────────────────
    if len(pts) >= _MIN_PTS_ELLIPSE:
        try:
            r = _fit_ellipse_ransac(pts)
            r["edge_id"] = edge_id
            if r["confidence"] >= _CONF_THRESH_ELLIPSE:
                return r
        except ValueError:
            pass

    # ── Return best candidate so far, or polyline ─────────────────────────────
    best = None
    for candidate in (arc_result, line_result):
        if candidate is not None:
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
    if best is not None and best["confidence"] > 0.2:
        return best

    raw_poly = edge.get("smooth_pts") or edge["pixels"]
    return {
        "edge_id":    edge_id,
        "type":       "polyline",
        "points":     [[float(p[0]), float(p[1])] for p in raw_poly],
        "confidence": 0.3,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC STAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run(graph_path: Path, output_dir: Path, sketch_id: str,
        config: dict) -> Stage3Result:
    """
    Fit primitives to all edges in one stroke graph.

    Parameters
    ----------
    graph_path : Path
        Stage 2 output (output/graphs/<id>_graph.json).
    output_dir : Path
        Root output dir; writes to output_dir/primitives/.
    sketch_id : str
        Unique identifier for this sketch.
    config : dict
        Parsed config.yaml — only `stage3.confidence_threshold` is read.
    """
    t_start = time.perf_counter()

    prims_dir  = output_dir / "primitives"
    prims_dir.mkdir(parents=True, exist_ok=True)
    prims_path = prims_dir / f"{sketch_id}_primitives.json"

    with open(graph_path) as f:
        graph = json.load(f)
    edges = graph.get("edges", [])

    # Stage 2 writes image_shape as numpy convention [H, W]; Stage 4 expects
    # image_size as [W, H]. Swap once here so downstream consumers don't have
    # to remember the convention.
    img_shape = graph.get("image_shape")
    if img_shape and len(img_shape) == 2:
        image_size = [int(img_shape[1]), int(img_shape[0])]
    else:
        logger.warning(
            f"[{sketch_id}] graph has no 'image_shape' — Stage 4 export will "
            f"need image_size supplied another way"
        )
        image_size = None

    logger.info(f"[{sketch_id}] Stage 3 — {len(edges)} edge(s)")

    conf_thresh = config.get("stage3", {}).get("confidence_threshold", 0.60)

    primitives = [fit_edge_ransac(edge) for edge in edges]

    confidences = [p.get("confidence", 0.0) for p in primitives]
    mean_conf   = float(np.mean(confidences)) if confidences else 0.0
    flagged     = mean_conf < conf_thresh

    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    doc = {"sketch_id": sketch_id}
    if image_size is not None:
        doc["image_size"] = image_size
    doc["primitives"]  = primitives
    doc["annotations"] = []   # filled by AP6.4 (Bezugszeichen) when available
    doc = _to_python(doc)
    with open(prims_path, "w") as f:
        json.dump(doc, f, indent=2)

    elapsed = time.perf_counter() - t_start

    if flagged:
        logger.warning(
            f"[{sketch_id}] FLAGGED — mean conf {mean_conf:.3f} "
            f"< threshold {conf_thresh:.2f}"
        )
    else:
        logger.info(
            f"[{sketch_id}] Stage 3 done in {elapsed:.2f}s — "
            f"conf={mean_conf:.3f}"
        )

    return Stage3Result(
        sketch_id         = sketch_id,
        primitives_path   = prims_path,
        mean_confidence   = mean_conf,
        flagged           = flagged,
        processing_time_s = elapsed,
        n_primitives      = len(primitives),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 — Primitive Fitting (RANSAC, final).\n\n"
            "Single graph :  python stage3_primitive_fit_final.py path/to/graph.json\n"
            "Batch folder :  python stage3_primitive_fit_final.py --input-dir path/to/graphs/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "input", nargs="?", type=Path, default=None,
        help="Single *_graph.json file.",
    )
    input_group.add_argument(
        "--input-dir", type=Path, default=None, metavar="DIR",
        help="Process all *_graph.json files in DIR.",
    )
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output",
                        help="Output root directory (default: <project>/output)")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "config.yaml",
                        help="Pipeline config file (default: <project>/config.yaml)")
    parser.add_argument("--id", type=str, default=None,
                        help="Sketch ID (single-file mode only)")
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    if args.input is not None:
        graphs = [args.input]
    else:
        graphs = sorted(args.input_dir.glob("*_graph.json"))
        if not graphs:
            logger.error(f"No *_graph.json files found in {args.input_dir}")
            raise SystemExit(1)
        logger.info(f"Batch mode: {len(graphs)} graph(s) found")

    results   = []
    n_flagged = 0
    for graph_path in graphs:
        sid = (args.id if (args.id and len(graphs) == 1)
               else graph_path.stem.replace("_graph", ""))
        result = run(
            graph_path = graph_path,
            output_dir = args.output,
            sketch_id  = sid,
            config     = cfg,
        )
        results.append(result)
        if result.flagged:
            n_flagged += 1

    if len(results) == 1:
        r = results[0]
        print(f"\n{'─'*56}")
        print(f"  Sketch ID        : {r.sketch_id}")
        print(f"  Primitives       : {r.n_primitives}")
        print(f"  Mean confidence  : {r.mean_confidence:.3f}")
        print(f"  Flagged          : {'YES' if r.flagged else 'no'}")
        print(f"  Primitives JSON  : {r.primitives_path}")
        print(f"  Processing time  : {r.processing_time_s:.2f}s")
        print(f"{'─'*56}")
    else:
        total_time = sum(r.processing_time_s for r in results)
        print(f"\n{'─'*56}")
        print(f"  Batch complete")
        print(f"  Processed        : {len(results)} graph(s)")
        print(f"  Flagged          : {n_flagged}")
        print(f"  Output dir       : {args.output}")
        print(f"  Total time       : {total_time:.2f}s")
        print(f"{'─'*56}")
        if n_flagged:
            print("  Flagged sketches:")
            for r in results:
                if r.flagged:
                    print(f"    {r.sketch_id}  (conf={r.mean_confidence:.3f})")
