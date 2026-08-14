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

  Final call (2026-07-19, investigation CLOSED): three retrained
  generations (synthetic v3, SketchGraphs, SketchGraphs+Drawing2CAD mixed)
  were evaluated with primitive-level Chamfer-vs-GT on Drawing2CAD stage-2
  graphs. The mixed model repairs the class-coverage collapse (POLYLINE
  f1 0.0 -> 0.957) and beats the *research* RANSAC baseline (16.65 vs
  19.39 mean) — but THIS production cascade scores 0.43 on the same
  graphs, ~40x better. The gap is structural (exact solver vs
  approximator), not data-limited. Decision: RANSAC stays; no further
  Free2CAD corpora for production. See README "Active next steps" #14.

Priority order in fit_edge_ransac:
  closed: circle → ellipse → polygon → guarded simplified trace → raw trace
  open:   line → arc → ellipse → guarded compound path → raw fallback

Confidence:
  inlier_ratio × max(0, 1 − rms / MAX_RMS)

Output per sketch:
  output/primitives/<sketch_id>_primitives.json

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdp import rdp as _rdp

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
    n_hachure_primitives: int = 0


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
_CONF_THRESH_ELLIPSE  = 0.55
_CONF_THRESH_POLYGON  = 0.40  # lenient: skeleton corners are naturally rounded

_INLIER_DIST_POLYGON  = 3.0   # px — slightly wider than line/circle tolerance
_MAX_RMS_POLYGON      = 4.0   # px — more lenient denominator for polygon conf
_MAX_POLYGON_SIDES    = 12    # RDP won't be accepted above this vertex count

# Geometric guard: when an arc wins the cascade, fall back to the line fit
# if the actual skeleton bulges by less than 5% of its chord length AND the
# line fit at least somewhat matched (conf >= 0.50). RANSAC can always
# fit a high-radius arc through a slightly-noisy straight skeleton; the
# result visually looks like a line, which is what the user reported as
# "straight lines becoming curves in the SVG".
_ARC_MIN_SAGITTA_RATIO = 0.05
_ARC_MIN_SAGITTA_PX    = 2.5   # absolute floor: arc must bow ≥ 2.5 px to
                               # count, regardless of chord length (catches
                               # short noisy "straight" skeletons whose
                               # sagitta/chord exceeds the ratio simply
                               # because the chord is small)
_LINE_FALLBACK_CONF    = 0.50


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


# ── Closed polygon fitter ────────────────────────────────────────────────────

def _fit_polygon_closed(pts: np.ndarray) -> dict | None:
    """
    Fit a closed polygon (RDP-simplified line-segment sequence) to the
    topologically-ordered pixel loop. Intended for rectangular or otherwise
    angular closed shapes whose skeleton has naturally rounded corners that
    prevent a clean circle/ellipse fit.

    Tries RDP with progressively larger epsilon until the vertex count is
    ≤ _MAX_POLYGON_SIDES, then accepts the fit if the per-pixel distance to
    the nearest polygon edge is good enough.

    Returns a 'polygon' dict or None if no acceptable fit was found.
    """
    if len(pts) < 6:
        return None

    best_conf   = -1.0
    best_verts  = None

    for eps in (3.0, 5.0, 8.0, 12.0, 20.0):
        verts = np.asarray(_rdp(pts, epsilon=eps), dtype=np.float64)
        n_v   = len(verts)
        if n_v < 3 or n_v > _MAX_POLYGON_SIDES:
            continue

        # Vectorised: for every raw pixel compute the min distance to any edge
        # of the closed polygon (edges are verts[i] → verts[(i+1) % n_v]).
        min_dists = np.full(len(pts), np.inf)
        for i in range(n_v):
            a  = verts[i]
            b  = verts[(i + 1) % n_v]
            ab = b - a
            len_sq = float(np.dot(ab, ab))
            if len_sq < 1e-12:
                d = np.linalg.norm(pts - a, axis=1)
            else:
                t  = np.clip(((pts - a) @ ab) / len_sq, 0.0, 1.0)
                proj = a + t[:, None] * ab
                d  = np.linalg.norm(pts - proj, axis=1)
            np.minimum(min_dists, d, out=min_dists)

        inlier_ratio = float((min_dists <= _INLIER_DIST_POLYGON).mean())
        rms          = float(np.sqrt((min_dists ** 2).mean()))
        conf         = inlier_ratio * max(0.0, 1.0 - rms / _MAX_RMS_POLYGON)

        if conf > best_conf:
            best_conf  = conf
            best_verts = verts

        if n_v <= 4:
            break  # can't simplify a quadrilateral further

    if best_conf < _CONF_THRESH_POLYGON or best_verts is None:
        return None

    return {
        "type":       "polygon",
        "points":     [[float(p[0]), float(p[1])] for p in best_verts],
        "confidence": best_conf,
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

    A valid one-pixel digital cycle is traversed exactly through its local
    8-neighbour adjacency in O(N). Malformed residual networks use a bounded
    depth-first adjacency walk, also O(N), so a missed keypoint can never turn
    Stage 3 into the former quadratic nearest-neighbour search.
    """
    point_set = {
        (int(round(float(point[0]))), int(round(float(point[1]))))
        for point in pixels
    }
    if len(point_set) < 3:
        return np.asarray(sorted(point_set), dtype=np.float64)

    offsets = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),            (1, 0),
        (-1, 1),  (0, 1),   (1, 1),
    )

    def neighbours(point, *, suppress_corner_chords):
        x, y = point
        result = []
        for dx, dy in offsets:
            candidate = (x + dx, y + dy)
            if candidate not in point_set:
                continue
            if suppress_corner_chords and dx and dy and (
                (x + dx, y) in point_set or (x, y + dy) in point_set
            ):
                continue
            result.append(candidate)
        return sorted(result, key=lambda item: (item[1], item[0]))

    start = min(point_set, key=lambda item: (item[1], item[0]))
    cycle_neighbours = {
        point: neighbours(point, suppress_corner_chords=True)
        for point in point_set
    }
    if all(len(items) == 2 for items in cycle_neighbours.values()):
        ordered = [start]
        cycle_visited = {start}
        previous = None
        current = start
        while len(ordered) < len(point_set):
            candidates = [
                point for point in cycle_neighbours[current]
                if point != previous and point not in cycle_visited
            ]
            if not candidates:
                break
            following = candidates[0]
            ordered.append(following)
            cycle_visited.add(following)
            previous, current = current, following
        if (
            len(ordered) == len(point_set)
            and start in cycle_neighbours[current]
        ):
            return np.asarray(ordered, dtype=np.float64)

    # The component is not a simple loop. Follow real local adjacencies with
    # an iterative DFS and repeat parent pixels while backtracking. The trace
    # remains on the source skeleton and is bounded by 2*N-1 points.
    visited = {start}
    ordered = [start]
    stack = [(start, iter(neighbours(start, suppress_corner_chords=False)))]
    while stack:
        current, candidates = stack[-1]
        for following in candidates:
            if following in visited:
                continue
            visited.add(following)
            ordered.append(following)
            stack.append((
                following,
                iter(neighbours(following, suppress_corner_chords=False)),
            ))
            break
        else:
            stack.pop()
            if stack:
                ordered.append(stack[-1][0])

    if visited != point_set:
        # Sparse sampled loops (notably Drawing2CAD supervision) can have
        # two-pixel gaps, so exact 8-neighbour traversal sees many components.
        # A capped KD-tree walk restores local order without reintroducing the
        # old quadratic all-pairs search on giant malformed patent residuals.
        spatial_limit = 20_000
        if len(point_set) <= spatial_limit:
            from scipy.spatial import cKDTree

            points = np.asarray(
                sorted(point_set, key=lambda item: (item[1], item[0])),
                dtype=np.float64,
            )
            tree = cKDTree(points)
            used = np.zeros(len(points), dtype=bool)
            current = 0
            spatial_order = []
            for step in range(len(points)):
                spatial_order.append(points[current])
                used[current] = True
                if step + 1 == len(points):
                    break
                k = min(8, len(points))
                following = None
                while following is None:
                    distances, indices = tree.query(points[current], k=k)
                    distances = np.atleast_1d(distances)
                    indices = np.atleast_1d(indices)
                    candidates = [
                        (float(distance), int(index))
                        for distance, index in zip(distances, indices)
                        if np.isfinite(distance) and not used[int(index)]
                    ]
                    if candidates:
                        following = min(
                            candidates,
                            key=lambda item: (
                                item[0], points[item[1], 1],
                                points[item[1], 0], item[1],
                            ),
                        )[1]
                    elif k == len(points):
                        following = int(np.flatnonzero(~used)[0])
                    else:
                        k = min(len(points), k * 2)
                current = following
            return np.asarray(spatial_order, dtype=np.float64)

        # Defensive large-component fallback: retain all points with bounded
        # work even when an unexpected disconnected residual reaches Stage 3.
        ordered.extend(sorted(
            point_set - visited, key=lambda item: (item[1], item[0])
        ))
    return np.asarray(ordered, dtype=np.float64)


def _fit_closed_simplified_trace(
    ordered_pts: np.ndarray,
    edge_id,
    *,
    vertex_caps: tuple[int, ...] = (24, 32, 48, 64, 96, 128),
    target_confidence: float = 0.65,
    target_p95: float = 2.0,
) -> dict | None:
    """Fit a bounded closed polygon and score its bidirectional residual.

    Unlike the old raw-polyline fallback, this representation has a hard
    complexity ceiling. Confidence measures both source-to-vector coverage and
    vector-to-source precision, so a shortcut across a concavity cannot earn a
    high score merely because most source pixels lie near some segment.
    """
    pts = np.asarray(ordered_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.diff(pts, axis=0) != 0, axis=1)
        pts = pts[keep]
    if len(pts) < 3:
        return None

    caps = tuple(sorted({int(value) for value in vertex_caps if int(value) >= 3}))
    if not caps:
        return None

    import cv2
    from scipy.spatial import cKDTree

    contour = pts.astype(np.float32).reshape(-1, 1, 2)
    source_points = np.unique(pts, axis=0)

    def approximate(max_vertices: int) -> np.ndarray | None:
        def at_epsilon(epsilon: float) -> np.ndarray:
            return cv2.approxPolyDP(
                contour, float(epsilon), True,
            ).reshape(-1, 2).astype(np.float64)

        vertices = at_epsilon(0.0)
        if len(vertices) <= max_vertices:
            return vertices if len(vertices) >= 3 else None

        low = 0.0
        high = max(
            float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])), 1.0,
        )
        best = at_epsilon(high)
        for _ in range(22):
            middle = 0.5 * (low + high)
            candidate = at_epsilon(middle)
            if len(candidate) > max_vertices:
                low = middle
            else:
                high = middle
                best = candidate
        return best if 3 <= len(best) <= max_vertices else None

    def score(vertices: np.ndarray) -> tuple[float, float, float, float, float] | None:
        samples = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            length = float(np.linalg.norm(end - start))
            n_samples = max(1, int(math.ceil(length)))
            t = np.arange(n_samples, dtype=np.float64) / n_samples
            samples.append(start + (end - start) * t[:, None])
        if not samples:
            return None
        model_points = np.vstack(samples)
        source_to_model = cKDTree(model_points).query(source_points)[0]
        model_to_source = cKDTree(source_points).query(model_points)[0]
        source_rms = float(np.sqrt(np.mean(source_to_model ** 2)))
        model_rms = float(np.sqrt(np.mean(model_to_source ** 2)))
        symmetric_rms = math.sqrt(0.5 * (source_rms ** 2 + model_rms ** 2))
        source_p95 = float(np.percentile(source_to_model, 95))
        model_p95 = float(np.percentile(model_to_source, 95))
        symmetric_p95 = max(source_p95, model_p95)
        inlier_ratio = 0.5 * (
            float((source_to_model <= _INLIER_DIST_POLYGON).mean())
            + float((model_to_source <= _INLIER_DIST_POLYGON).mean())
        )
        confidence = inlier_ratio * max(
            0.0, 1.0 - symmetric_rms / _MAX_RMS_POLYGON,
        )
        return confidence, symmetric_rms, symmetric_p95, source_rms, model_rms

    selected = None
    selected_cap = None
    selected_score = None
    target_met = False
    for cap in caps:
        vertices = approximate(cap)
        if vertices is None:
            continue
        metrics = score(vertices)
        if metrics is None:
            continue
        selected = vertices
        selected_cap = cap
        selected_score = metrics
        if metrics[0] >= target_confidence and metrics[2] <= target_p95:
            target_met = True
            break

    # A bounded approximation that misses the fidelity contract is worse than
    # the verbose but exact raw trace. Keep the caller on its original fallback
    # instead of allowing a few giant malformed components to pass a
    # primitive-count-based confidence gate while visibly losing geometry.
    if not target_met or selected is None or selected_score is None:
        return None
    confidence, residual_rms, residual_p95, source_rms, model_rms = selected_score
    return {
        "edge_id": edge_id,
        "type": "polygon",
        "points": [[float(point[0]), float(point[1])] for point in selected],
        "confidence": float(confidence),
        "fit_metadata": {
            "strategy": "closed_simplified_trace",
            "source_points": int(len(source_points)),
            "output_vertices": int(len(selected)),
            "vertex_cap": int(selected_cap),
            "residual_rms": float(residual_rms),
            "residual_p95": float(residual_p95),
            "source_to_model_rms": float(source_rms),
            "model_to_source_rms": float(model_rms),
        },
    }


# ── Geometric guard helper ────────────────────────────────────────────────────

def _refit_arc_as_line(edge: dict, edge_id) -> dict | None:
    """
    If the raw-pixel skeleton of `edge` is essentially straight (small bulge
    relative to its chord length AND small absolute bulge), refit it as a
    line on raw pixels and return the line primitive. Otherwise return None.

    Called whenever the cascade is about to emit an arc, to catch cases where
    RANSAC threaded a high-radius arc through a noisy straight skeleton.
    """
    raw_pts = np.array(edge["pixels"], dtype=np.float64)
    if len(raw_pts) < 2:
        return None
    chord_vec = raw_pts[-1] - raw_pts[0]
    chord_len = float(np.linalg.norm(chord_vec))
    if chord_len <= 5:
        return None
    normal  = np.array([-chord_vec[1], chord_vec[0]]) / chord_len
    sagitta = float(np.abs(((raw_pts - raw_pts[0]) @ normal)).max())
    # Edge is essentially straight if its bulge is small relative to chord
    # length OR small in absolute terms.
    if (sagitta / chord_len) >= _ARC_MIN_SAGITTA_RATIO and sagitta >= _ARC_MIN_SAGITTA_PX:
        return None
    try:
        raw_line = _fit_line_ransac(raw_pts)
    except ValueError:
        return None
    if raw_line["confidence"] < _LINE_FALLBACK_CONF:
        return None
    raw_line["edge_id"] = edge_id
    return raw_line


# ═══════════════════════════════════════════════════════════════════════════
# COMPOUND-PATH FITTER  (corner-split → per-segment line / arc / cubic Bézier)
# ═══════════════════════════════════════════════════════════════════════════
#
# Why this exists: ~21% of patent primitives were raw polylines — and 100% of
# the *curved* ones had single-arc fit confidence < 0.45 (median 0.00). A single
# circular arc cannot represent a compound curve (fillets, isometric silhouettes,
# S-curves), and a single global Bézier would round the *sharp* corners that
# also produce low chord ratios (L-shapes). The fix splits the dense skeleton at
# tangent-discontinuity corners (keeping them sharp), then fits each piece with
# line → arc → cubic Bézier, emitting one compound "path" primitive. Measured on
# PatentData: 56% of curve length is line/arc-recoverable after the split, the
# remaining 44% genuinely needs Béziers.

_CORNER_TURN_DEG    = 35.0   # turn above this, *concentrated* over CORNER_WIN px,
                             # = a sharp corner → split (catches 45° chamfers)
_CORNER_WIN_PX      = 3.0    # arc-length window for the corner turn measurement.
                             # Short on purpose: it measures whether the turn is
                             # *concentrated* (corner) vs spread over many px
                             # (smooth curvature, which must NOT be split).
_SEG_MIN_PTS        = 4      # don't make sub-segments shorter than this
_PATH_LINE_CONF     = 0.80   # per-segment line accept (stricter than top-level)
_PATH_ARC_CONF      = 0.70   # per-segment arc accept
_BEZIER_MAX_ERR     = 1.0    # px — max allowed deviation of a cubic from pixels.
                             # Tight on purpose: the old polyline fallback traced
                             # the skeleton within ~1px, so the Bézier must hug it
                             # at least as closely or D2C Chamfer regresses.
_BEZIER_MAX_DEPTH   = 6      # recursion cap (≤2^6 cubics/sub-seg, anti blow-up)
_BEZIER_MAX_HANDLE_ARCLEN_RATIO = 2.0
_PATH_SEGMENT_MAX_P95 = 2.0  # px, bidirectional source/model fidelity contract
# Corner-split line fits can hand a 2-3 px transition to the adjacent segment;
# larger misses are endpoint features that must not be silently discarded.
_PATH_SEGMENT_MAX_ENDPOINT_ERROR = 3.0

# A/B kill-switch: STAGE3_NO_COMPOUND=1 reverts to the raw-polyline fallback.
import os as _os
_COMPOUND_PATH_ENABLED = _os.environ.get("STAGE3_NO_COMPOUND", "") != "1"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0])


def _split_at_corners(pts: np.ndarray) -> list[np.ndarray]:
    """Split an ordered dense pixel chain at sharp tangent discontinuities only.

    A *corner* is a large direction change concentrated within a short arc-length
    window (_CORNER_WIN_PX); a *smooth curve* turns the same total angle but
    spread over a long arc, so its windowed turn stays small. Measuring the turn
    over a fixed arc length — not a fixed index window — is what separates the two
    (an index window conflates curvature with corners and shatters smooth curves
    into line-soup). Local-maximum suppression keeps one cut per corner.
    """
    n = len(pts)
    if n < 2 * _SEG_MIN_PTS + 1:
        return [pts]
    seg_d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arclen = np.concatenate([[0.0], np.cumsum(seg_d)])
    turns = np.zeros(n)
    for i in range(1, n - 1):
        a = i
        while a > 0 and (arclen[i] - arclen[a]) < _CORNER_WIN_PX:
            a -= 1
        b = i
        while b < n - 1 and (arclen[b] - arclen[i]) < _CORNER_WIN_PX:
            b += 1
        d1 = _unit(pts[i] - pts[a])
        d2 = _unit(pts[b] - pts[i])
        if (d1[0] or d1[1]) and (d2[0] or d2[1]):
            turns[i] = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(d1, d2))))))
    # candidate corners = turns over threshold, kept only at local maxima
    cuts = [0]
    last_cut = 0
    for i in range(1, n - 1):
        if turns[i] <= _CORNER_TURN_DEG:
            continue
        lo = max(1, i - 2); hi = min(n - 1, i + 3)
        if turns[i] < turns[lo:hi].max():
            continue                       # not the local peak of this corner
        if (i - last_cut) >= _SEG_MIN_PTS:
            cuts.append(i)
            last_cut = i
    cuts.append(n - 1)
    segs = []
    for k in range(len(cuts) - 1):
        seg = pts[cuts[k]: cuts[k + 1] + 1]
        if len(seg) >= 2:
            segs.append(seg)
    return segs or [pts]


# ── Cubic Bézier fitting (Schneider, Graphics Gems) ───────────────────────────

def _bezier_eval(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate a cubic Bézier (4×2 control) at parameters t (vectorised)."""
    mt = 1.0 - t
    b0 = mt ** 3
    b1 = 3 * mt ** 2 * t
    b2 = 3 * mt * t ** 2
    b3 = t ** 3
    return (np.outer(b0, ctrl[0]) + np.outer(b1, ctrl[1])
            + np.outer(b2, ctrl[2]) + np.outer(b3, ctrl[3]))


def _chord_param(pts: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > 1e-12 else np.linspace(0, 1, len(pts))


def _generate_bezier(pts, u, that1, that2):
    """Least-squares cubic with fixed endpoints and given end tangents."""
    p0, p3 = pts[0], pts[-1]
    mt = 1.0 - u
    b0 = mt ** 3
    b1 = 3 * mt ** 2 * u
    b2 = 3 * mt * u ** 2
    b3 = u ** 3
    a1 = that1[None, :] * b1[:, None]
    a2 = that2[None, :] * b2[:, None]
    tmp = pts - ((p0[None, :] * (b0 + b1)[:, None]) + (p3[None, :] * (b2 + b3)[:, None]))
    c00 = float(np.sum(a1 * a1)); c01 = float(np.sum(a1 * a2)); c11 = float(np.sum(a2 * a2))
    x0  = float(np.sum(a1 * tmp)); x1 = float(np.sum(a2 * tmp))
    det = c00 * c11 - c01 * c01
    seg_len = float(np.linalg.norm(p3 - p0))
    if abs(det) < 1e-12:
        alpha1 = alpha2 = seg_len / 3.0
    else:
        alpha1 = (x0 * c11 - c01 * x1) / det
        alpha2 = (c00 * x1 - c01 * x0) / det

    # A matrix can be technically invertible while still being ill-conditioned.
    # In that case the least-squares solution may place a handle thousands of
    # pixels away from a short source chain. The sampled residual then catches
    # the error, but at the recursion limit the malformed cubic used to escape.
    # A valid local handle has no reason to exceed twice the complete observed
    # chain length; fall back to the stable chord construction when it does.
    polyline_length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    max_handle = _BEZIER_MAX_HANDLE_ARCLEN_RATIO * max(
        polyline_length, seg_len, 1e-6,
    )
    if (
        not np.isfinite(alpha1)
        or not np.isfinite(alpha2)
        or alpha1 < 1e-6
        or alpha2 < 1e-6
        or alpha1 > max_handle
        or alpha2 > max_handle
    ):
        alpha1 = alpha2 = seg_len / 3.0
    return np.array([p0, p0 + that1 * alpha1, p3 + that2 * alpha2, p3])


def _max_error(pts, ctrl, u):
    q = _bezier_eval(ctrl, u)
    d = np.linalg.norm(q - pts, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), i


def _fit_cubic_chain(pts, that1, that2, max_err, depth=0):
    """Recursively fit a chain of cubic Béziers; returns list of 4×2 controls."""
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        dist = float(np.linalg.norm(pts[1] - pts[0])) / 3.0
        return [np.array([pts[0], pts[0] + that1 * dist, pts[1] + that2 * dist, pts[1]])]
    u = _chord_param(pts)
    ctrl = _generate_bezier(pts, u, that1, that2)
    err, split = _max_error(pts, ctrl, u)
    if err < max_err:
        return [ctrl]
    if depth >= _BEZIER_MAX_DEPTH or split <= 0 or split >= len(pts) - 1:
        return [ctrl]
    that_c = _unit(pts[split - 1] - pts[split + 1])   # centre tangent (into left seg)
    left  = _fit_cubic_chain(pts[:split + 1], that1, that_c, max_err, depth + 1)
    right = _fit_cubic_chain(pts[split:], -that_c, that2, max_err, depth + 1)
    return left + right


def _fit_bezier_segment(sub: np.ndarray):
    """Fit a sub-chain as a poly-cubic-Bézier. Returns (segment_dict, confidence)."""
    k = min(4, len(sub) - 1)
    that1 = _unit(sub[k] - sub[0])
    that2 = _unit(sub[-1 - k] - sub[-1])
    chain = _fit_cubic_chain(sub, that1, that2, _BEZIER_MAX_ERR)
    if not chain:
        return None, 0.0
    # flat control-point list: P0,C1,C2,P1,C1',C2',P2,...  (SVG cubic path)
    flat = [chain[0][0].tolist()]
    for c in chain:
        flat.extend([c[1].tolist(), c[2].tolist(), c[3].tolist()])
    # confidence from densely-sampled deviation vs the pixels
    samp = []
    for c in chain:
        samp.append(_bezier_eval(c, np.linspace(0, 1, 24)))
    s = np.vstack(samp)
    from scipy.spatial import cKDTree
    dev = cKDTree(sub).query(s)[0]
    rms = float(np.sqrt((dev ** 2).mean()))
    conf = max(0.0, 1.0 - rms / _MAX_RMS)
    return {"type": "bezier", "points": flat}, conf


def _fit_subsegment(sub: np.ndarray):
    """Fit one corner-split sub-chain: line → arc → cubic Bézier.

    Returns (segment_dict, confidence). Segment dicts are the path-local form:
    line {p1,p2}, arc {center,radius,start_angle,end_angle}, bezier {points}.
    """
    # line
    try:
        r = _fit_line_ransac(sub)
        if r["confidence"] >= _PATH_LINE_CONF:
            return {"type": "line", "p1": r["p1"], "p2": r["p2"]}, r["confidence"]
    except ValueError:
        pass
    # arc — but only if the chain actually bows (else a straight noisy chain)
    if len(sub) >= _MIN_PTS_ARC:
        try:
            r = _fit_arc_ransac(sub)
            chord = float(np.linalg.norm(sub[-1] - sub[0]))
            if r["confidence"] >= _PATH_ARC_CONF and chord > 5:
                nrm = np.array([-(sub[-1] - sub[0])[1], (sub[-1] - sub[0])[0]]) / chord
                sag = float(np.abs(((sub - sub[0]) @ nrm)).max())
                if (sag / chord) >= _ARC_MIN_SAGITTA_RATIO or sag >= _ARC_MIN_SAGITTA_PX:
                    return ({"type": "arc", "center": r["center"], "radius": r["radius"],
                             "start_angle": r["start_angle"], "end_angle": r["end_angle"]},
                            r["confidence"])
        except ValueError:
            pass
    # cubic Bézier fallback (smooth, handles freeform + near-straight gracefully)
    return _fit_bezier_segment(sub)


def _sample_path_segment(segment: dict) -> np.ndarray:
    """Sample one path-local primitive at approximately one-pixel spacing."""
    segment_type = segment.get("type")
    if segment_type == "line":
        start = np.asarray(segment["p1"], dtype=np.float64)
        end = np.asarray(segment["p2"], dtype=np.float64)
        n = max(2, int(math.ceil(float(np.linalg.norm(end - start)))) + 1)
        return np.linspace(start, end, n)

    if segment_type == "arc":
        center = np.asarray(segment["center"], dtype=np.float64)
        radius = float(segment["radius"])
        start = float(segment["start_angle"])
        sweep = (float(segment["end_angle"]) - start) % 360.0
        arc_length = abs(radius) * math.radians(sweep)
        n = max(2, min(8192, int(math.ceil(arc_length)) + 1))
        angles = np.radians(start + np.linspace(0.0, sweep, n))
        return center + radius * np.column_stack([
            np.cos(angles), np.sin(angles),
        ])

    if segment_type == "bezier":
        points = np.asarray(segment.get("points") or [], dtype=np.float64)
        if len(points) < 4 or (len(points) - 1) % 3:
            return np.empty((0, 2), dtype=np.float64)
        samples = []
        for index in range(0, len(points) - 1, 3):
            control = points[index:index + 4]
            control_length = float(
                np.linalg.norm(np.diff(control, axis=0), axis=1).sum()
            )
            n = max(8, min(2048, int(math.ceil(control_length)) + 1))
            curve = _bezier_eval(control, np.linspace(0.0, 1.0, n))
            samples.append(curve if not samples else curve[1:])
        return np.vstack(samples) if samples else np.empty((0, 2), dtype=np.float64)

    return np.empty((0, 2), dtype=np.float64)


def _path_segment_fidelity(segment: dict, source: np.ndarray) -> dict | None:
    """Return bidirectional residuals, or None for malformed geometry."""
    model = _sample_path_segment(segment)
    source = np.asarray(source, dtype=np.float64)
    if (
        len(model) == 0
        or len(source) == 0
        or not np.isfinite(model).all()
        or not np.isfinite(source).all()
    ):
        return None

    from scipy.spatial import cKDTree

    source_to_model = cKDTree(model).query(source)[0]
    model_to_source = cKDTree(source).query(model)[0]
    direct_endpoint_error = max(
        float(np.linalg.norm(model[0] - source[0])),
        float(np.linalg.norm(model[-1] - source[-1])),
    )
    reverse_endpoint_error = max(
        float(np.linalg.norm(model[-1] - source[0])),
        float(np.linalg.norm(model[0] - source[-1])),
    )
    return {
        "source_p95": float(np.percentile(source_to_model, 95)),
        "model_p95": float(np.percentile(model_to_source, 95)),
        "symmetric_p95": max(
            float(np.percentile(source_to_model, 95)),
            float(np.percentile(model_to_source, 95)),
        ),
        "endpoint_error": min(
            direct_endpoint_error,
            reverse_endpoint_error,
        ),
    }


def _fit_compound_path(
    edge: dict,
    edge_id,
    *,
    require_fidelity: bool = False,
    max_segment_p95: float = _PATH_SEGMENT_MAX_P95,
    max_endpoint_error: float = _PATH_SEGMENT_MAX_ENDPOINT_ERROR,
) -> dict | None:
    """Corner-split the dense skeleton and fit each piece line/arc/Bézier.

    Returns a 'path' primitive {segments:[...]} or None to defer to a raw
    polyline. Operates on raw pixels (not smooth_pts) to avoid the B-spline
    oscillation Stage 2 already guards against.
    """
    pts = np.array(edge.get("pixels") or edge.get("smooth_pts") or [], dtype=np.float64)
    if len(pts) < _SEG_MIN_PTS:
        return None
    segments, confs, weights = [], [], []
    segment_p95, endpoint_errors = [], []
    for sub in _split_at_corners(pts):
        if len(sub) < 2:
            continue
        seg, conf = _fit_subsegment(sub)
        if seg is None:
            return None
        fidelity = _path_segment_fidelity(seg, sub)
        if fidelity is None:
            if require_fidelity:
                return None
        elif require_fidelity and (
            fidelity["symmetric_p95"] > max_segment_p95
            or fidelity["endpoint_error"] > max_endpoint_error
        ):
            return None
        segments.append(seg)
        confs.append(conf)
        weights.append(len(sub))
        if fidelity is not None:
            segment_p95.append(fidelity["symmetric_p95"])
            endpoint_errors.append(fidelity["endpoint_error"])
    if not segments:
        return None
    w = np.array(weights, dtype=np.float64)
    conf = float(np.average(confs, weights=w)) if confs else 0.0
    path = {
        "edge_id": edge_id,
        "type": "path",
        "segments": segments,
        "confidence": conf,
    }
    if segment_p95:
        path["fit_fidelity"] = {
            "max_segment_p95": float(max(segment_p95)),
            "target_p95": max_segment_p95,
            "max_endpoint_error": float(max(endpoint_errors)),
            "target_endpoint_error": max_endpoint_error,
        }
    return path


def _compound_path_atom_count(path: dict) -> int:
    """Count native line/arc/cubic pieces represented by a compound path."""
    count = 0
    for segment in path.get("segments", []):
        if segment.get("type") == "bezier":
            # Cubics are stored P0,C1,C2,P1,C1,C2,P2,...
            count += max(1, (len(segment.get("points", [])) - 1) // 3)
        else:
            count += 1
    return count


# ── Priority selector ─────────────────────────────────────────────────────────

def fit_edge_ransac(
    edge: dict,
    *,
    prefer_compound_over_weak: bool = False,
    weak_compound_confidence_threshold: float = 0.60,
    weak_compound_min_gain: float = 0.05,
    weak_compound_max_points: int = 20_000,
    weak_compound_max_path_atoms: int = 64,
    weak_compound_max_segment_p95: float = _PATH_SEGMENT_MAX_P95,
    weak_compound_max_endpoint_error: float = _PATH_SEGMENT_MAX_ENDPOINT_ERROR,
    simplify_closed_fallback: bool = False,
    closed_trace_vertex_caps: tuple[int, ...] = (24, 32, 48, 64, 96, 128),
    closed_trace_target_confidence: float = 0.65,
    closed_trace_target_p95: float = 2.0,
    closed_trace_compete_below_confidence: float = 0.60,
    closed_trace_min_gain: float = 0.05,
) -> dict:
    """
    Fit one edge with the priority order:
      closed: circle → ellipse → polygon → simplified trace → raw trace
      open: line → arc → ellipse → compound path → raw fallback

    For open edges the cascade fits on smooth_pts (Stage 2's spline-
    interpolated coords), which generally improves fit confidence on noisy
    skeletons. Two refinements compensate for smoothing's downsides:
      * closed loops use raw pixels (spline ordering of closed contours is
        unreliable);
      * the arc path applies a geometric guard against straight edges that
        only "look" curved because the spline introduced fake curvature
        (see _ARC_MIN_SAGITTA_RATIO).
    """
    edge_id   = edge["id"]
    is_closed = edge.get("is_closed", False)

    if is_closed:
        # Circle and ellipse fits are order-independent. Keep raw unique
        # samples here and pay the ordering cost only for polygon/polyline
        # fallbacks.
        pts = np.asarray(edge["pixels"], dtype=np.float64)
    else:
        raw            = edge.get("smooth_pts") or []
        raw_pixels_list = edge.get("pixels", [])
        pts = (np.array(raw, dtype=np.float64) if raw
               else np.array(raw_pixels_list, dtype=np.float64))

        # Detect Stage 2 RDP fallback: when the B-spline overshoots,
        # Stage 2 substitutes RDP corner points as smooth_pts (~4–8 pts for
        # angular edges).  On rectangular/polygonal open chains these sparse
        # corner points all lie near the circumscribed circle, so arc RANSAC
        # achieves artificially high confidence.  For those edges emit a
        # polyline from the RDP corner points instead.
        _spline_sparse = (bool(raw)
                          and len(raw_pixels_list) >= 100
                          and len(raw) < max(10, int(0.05 * len(raw_pixels_list))))

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
        ordered_pts = _reorder_loop_pixels(edge["pixels"])

        # Try closed polygon (handles rectangles, hexagons, etc. whose
        # skeleton corners are rounded and fool circle/ellipse fitters).
        poly = _fit_polygon_closed(ordered_pts)
        if poly is not None:
            poly["edge_id"] = edge_id
            if (
                simplify_closed_fallback
                and poly["confidence"] < closed_trace_compete_below_confidence
            ):
                trace = _fit_closed_simplified_trace(
                    ordered_pts,
                    edge_id,
                    vertex_caps=closed_trace_vertex_caps,
                    target_confidence=closed_trace_target_confidence,
                    target_p95=closed_trace_target_p95,
                )
                if (
                    trace is not None
                    and trace["confidence"]
                    >= poly["confidence"] + closed_trace_min_gain
                ):
                    trace["fit_metadata"]["replaced_type"] = "polygon"
                    trace["fit_metadata"]["replaced_confidence"] = float(
                        poly["confidence"]
                    )
                    return trace
            return poly

        if simplify_closed_fallback:
            trace = _fit_closed_simplified_trace(
                ordered_pts,
                edge_id,
                vertex_caps=closed_trace_vertex_caps,
                target_confidence=closed_trace_target_confidence,
                target_p95=closed_trace_target_p95,
            )
            if trace is not None:
                trace["fit_metadata"]["replaced_type"] = "polyline"
                trace["fit_metadata"]["replaced_confidence"] = 0.3
                return trace

        # Final fallback: raw ordered pixel trace.
        poly_points = [[float(p[0]), float(p[1])] for p in ordered_pts]
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
                line_fallback = _refit_arc_as_line(edge, edge_id)
                if line_fallback is not None:
                    return line_fallback
                # If smooth_pts are a sparse RDP fallback (angular corners),
                # arc confidence is artificially inflated — skip and use
                # the corner-polyline fallback instead.
                if not _spline_sparse:
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
    weak_candidate = None
    if best is not None and best["confidence"] > 0.2:
        # Same geometric guard as the arc-passes-threshold branch above —
        # the best-of fallback can also wrongly emit a "best-effort" arc
        # over a straight skeleton (arc confidence between 0.20 and 0.65).
        if best.get("type") == "arc":
            line_fallback = _refit_arc_as_line(edge, edge_id)
            if line_fallback is not None:
                weak_candidate = line_fallback
            # Sparse RDP fallback edges → prefer the polyline below
            elif not _spline_sparse:
                weak_candidate = best
        else:
            weak_candidate = best

    # A weak global line/arc can be a poor explanation of a continuous bent
    # stroke. Historically it returned here and made the compound fitter below
    # unreachable. The opt-in policy lets a bounded compound path compete, but
    # only when it clears the quality threshold by a meaningful margin. The
    # atom cap prevents confidence from being bought with an impractically
    # large chain of tiny cubics.
    if prefer_compound_over_weak and weak_candidate is not None:
        raw_points = edge.get("pixels") or edge.get("smooth_pts") or []
        within_point_budget = (
            weak_compound_max_points <= 0
            or len(raw_points) <= weak_compound_max_points
        )
        if (
            within_point_budget
            and weak_candidate["confidence"] < weak_compound_confidence_threshold
        ):
            path = _fit_compound_path(
                edge,
                edge_id,
                require_fidelity=True,
                max_segment_p95=weak_compound_max_segment_p95,
                max_endpoint_error=weak_compound_max_endpoint_error,
            )
            if path is not None:
                atom_count = _compound_path_atom_count(path)
                within_atom_budget = (
                    weak_compound_max_path_atoms <= 0
                    or atom_count <= weak_compound_max_path_atoms
                )
                if (
                    within_atom_budget
                    and path["confidence"] >= weak_compound_confidence_threshold
                    and path["confidence"]
                    >= weak_candidate["confidence"] + weak_compound_min_gain
                ):
                    path["fit_metadata"] = {
                        "strategy": "compound_over_weak",
                        "weak_type": weak_candidate.get("type"),
                        "weak_confidence": float(weak_candidate["confidence"]),
                        "path_atoms": atom_count,
                    }
                    return path

    if weak_candidate is not None:
        return weak_candidate

    # ── Compound-path fallback ────────────────────────────────────────────────
    # No single line/arc/ellipse fit it. Rather than dump a raw jagged polyline,
    # corner-split the dense skeleton (keeping sharp corners sharp) and fit each
    # piece as line / arc / cubic Bézier. This recovers compound curves (fillets,
    # isometric silhouettes, S-curves) as smooth segments and angular chains
    # (L-shapes, rectangles) as exact lines — the single biggest quality lever on
    # PatentData, where 21% of primitives were raw polylines.
    if _COMPOUND_PATH_ENABLED:
        path = _fit_compound_path(edge, edge_id)
        if path is not None:
            return path

    raw_poly = edge.get("smooth_pts") or edge["pixels"]
    poly_conf = 0.65 if _spline_sparse else 0.3
    return {
        "edge_id":    edge_id,
        "type":       "polyline",
        "points":     [[float(p[0]), float(p[1])] for p in raw_poly],
        "confidence": poly_conf,
    }


def _scale_point(pt: list, scale: float) -> list:
    return [float(pt[0]) * scale, float(pt[1]) * scale]


def _scale_primitive(prim: dict, scale: float) -> dict:
    """
    Map a primitive from Stage-2 working coordinates back to source-image
    coordinates.  Stage 2 may cap large patent TIFs to 1000 px before graph
    extraction; Stage 4 should still export in the original image frame.
    """
    if abs(scale - 1.0) < 1e-9:
        return prim

    p = dict(prim)
    ptype = p.get("type")
    if ptype == "line":
        p["p1"] = _scale_point(p["p1"], scale)
        p["p2"] = _scale_point(p["p2"], scale)
    elif ptype in ("circle", "arc"):
        p["center"] = _scale_point(p["center"], scale)
        p["radius"] = float(p["radius"]) * scale
    elif ptype == "ellipse":
        p["center"] = _scale_point(p["center"], scale)
        p["a"] = float(p["a"]) * scale
        p["b"] = float(p["b"]) * scale
    elif ptype in ("polyline", "polygon"):
        p["points"] = [_scale_point(pt, scale) for pt in p.get("points", [])]
    elif ptype == "bezier":
        p["points"] = [_scale_point(pt, scale) for pt in p.get("points", [])]
    elif ptype == "path":
        p["segments"] = [_scale_primitive(seg, scale) for seg in p.get("segments", [])]
    elif ptype == "hatch":
        p["boundary"] = [_scale_point(pt, scale) for pt in p.get("boundary", [])]
        p["spacing"] = float(p.get("spacing", 0.0)) * scale   # angles are scale-invariant
    return p


def _fit_removed_hachure(edge: dict) -> dict | None:
    """
    Convert a Stage 2 side-layer hatch edge into an exportable primitive.

    Hachures are fitted separately from the main graph: they are useful visual
    content, but they should not participate in the quality gate that evaluates
    the long outline primitives.
    """
    pix = edge.get("pixels") or []
    if len(pix) < 2:
        return None

    pts = np.array(pix, dtype=np.float64)
    try:
        prim = _fit_line_ransac(pts)
    except ValueError:
        prim = {
            "type": "polyline",
            "points": [[float(p[0]), float(p[1])] for p in pts],
            "confidence": 0.3,
        }

    prim["edge_id"] = edge.get("id")
    prim["style"] = "hachure"
    prim["source"] = "removed_hachure"
    if "hachure" in edge:
        prim["hachure"] = edge["hachure"]
    return prim


def _hatch_region_to_primitive(region: dict) -> dict:
    """Stage 2 hachure region → a 'hatch' primitive (boundary + parametric fill)."""
    return {
        "type":       "hatch",
        "boundary":   [[float(p[0]), float(p[1])] for p in region.get("boundary", [])],
        "angles":     [float(a) for a in region.get("angles", [])],
        "spacing":    float(region.get("spacing", 0.0)),
        "double":     bool(region.get("double", False)),
        "n_lines":    int(region.get("n_lines", 0)),
        "style":      "hachure",
        "source":     "hachure_region",
        "confidence": 0.6,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC STAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run(graph_path: Path, output_dir: Path, sketch_id: str,
        config: dict, stroke_width: float = None) -> Stage3Result:
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
        Parsed config.yaml. `stage3.confidence_threshold` controls main-geometry
        confidence; hachure-heavy graphs can use
        `stage3.confidence_threshold_after_hachure`. Guarded weak-open and
        closed-trace policies are controlled by
        `stage3.prefer_compound_over_weak` and
        `stage3.simplify_closed_fallback`.
    stroke_width : float | None
        Estimated original stroke width in pixels (from Stage 1). When
        provided it is embedded in the JSON so Stage 4 can produce an SVG
        whose stroke-width matches the source sketch.
    """
    t_start = time.perf_counter()

    prims_dir  = output_dir / "primitives"
    prims_dir.mkdir(parents=True, exist_ok=True)
    prims_path = prims_dir / f"{sketch_id}_primitives.json"

    with open(graph_path) as f:
        graph = json.load(f)
    edges = graph.get("edges", [])
    removed_hachures = graph.get("removed_hachures", [])

    # Stage 2 writes image_shape as numpy convention [H, W]. If Stage 2 capped
    # the image resolution, it also writes original_image_shape + stage2_scale;
    # export back in the original coordinate frame.
    img_shape = graph.get("image_shape")
    orig_shape = graph.get("original_image_shape")
    stage2_scale = float(graph.get("stage2_scale", 1.0) or 1.0)
    coord_scale = 1.0 / stage2_scale if stage2_scale > 0 else 1.0
    if orig_shape and len(orig_shape) == 2:
        image_size = [int(orig_shape[1]), int(orig_shape[0])]
    elif img_shape and len(img_shape) == 2:
        image_size = [int(img_shape[1]), int(img_shape[0])]
    else:
        logger.warning(
            f"[{sketch_id}] graph has no 'image_shape' — Stage 4 export will "
            f"need image_size supplied another way"
        )
        image_size = None

    logger.info(
        f"[{sketch_id}] Stage 3 — {len(edges)} main edge(s), "
        f"{len(removed_hachures)} hachure edge(s)"
    )

    stage3_cfg = config.get("stage3", {})
    conf_thresh = float(stage3_cfg.get("confidence_threshold", 0.60))
    prefer_compound_over_weak = bool(
        stage3_cfg.get("prefer_compound_over_weak", False)
    )
    weak_compound_confidence_threshold = float(
        stage3_cfg.get("weak_compound_confidence_threshold", conf_thresh)
    )
    weak_compound_min_gain = float(
        stage3_cfg.get("weak_compound_min_gain", 0.05)
    )
    weak_compound_max_points = int(
        stage3_cfg.get("weak_compound_max_points", 20_000) or 0
    )
    weak_compound_max_path_atoms = int(
        stage3_cfg.get("weak_compound_max_path_atoms", 64) or 0
    )
    weak_compound_max_segment_p95 = float(
        stage3_cfg.get(
            "weak_compound_max_segment_p95", _PATH_SEGMENT_MAX_P95,
        )
    )
    weak_compound_max_endpoint_error = float(
        stage3_cfg.get(
            "weak_compound_max_endpoint_error",
            _PATH_SEGMENT_MAX_ENDPOINT_ERROR,
        )
    )
    simplify_closed_fallback = bool(
        stage3_cfg.get("simplify_closed_fallback", False)
    )
    closed_trace_caps_value = stage3_cfg.get(
        "closed_trace_vertex_caps", [24, 32, 48, 64, 96, 128],
    )
    if isinstance(closed_trace_caps_value, (int, float)):
        closed_trace_caps_value = [closed_trace_caps_value]
    closed_trace_vertex_caps = tuple(
        int(value) for value in closed_trace_caps_value
    )
    closed_trace_target_confidence = float(
        stage3_cfg.get("closed_trace_target_confidence", 0.65)
    )
    closed_trace_target_p95 = float(
        stage3_cfg.get("closed_trace_target_p95", 2.0)
    )
    closed_trace_compete_below_confidence = float(
        stage3_cfg.get("closed_trace_compete_below_confidence", conf_thresh)
    )
    closed_trace_min_gain = float(
        stage3_cfg.get("closed_trace_min_gain", 0.05)
    )
    effective_conf_thresh = conf_thresh
    hachure_relax_min_edges = int(
        stage3_cfg.get("min_hachure_edges_for_relaxed_confidence", 0) or 0
    )
    hachure_conf_thresh = float(
        stage3_cfg.get("confidence_threshold_after_hachure", 0.0) or 0.0
    )
    if (
        hachure_conf_thresh
        and hachure_relax_min_edges
        and len(removed_hachures) >= hachure_relax_min_edges
    ):
        effective_conf_thresh = min(conf_thresh, hachure_conf_thresh)

    main_primitives = [
        _scale_primitive(
            fit_edge_ransac(
                edge,
                prefer_compound_over_weak=prefer_compound_over_weak,
                weak_compound_confidence_threshold=(
                    weak_compound_confidence_threshold
                ),
                weak_compound_min_gain=weak_compound_min_gain,
                weak_compound_max_points=weak_compound_max_points,
                weak_compound_max_path_atoms=weak_compound_max_path_atoms,
                weak_compound_max_segment_p95=weak_compound_max_segment_p95,
                weak_compound_max_endpoint_error=(
                    weak_compound_max_endpoint_error
                ),
                simplify_closed_fallback=simplify_closed_fallback,
                closed_trace_vertex_caps=closed_trace_vertex_caps,
                closed_trace_target_confidence=closed_trace_target_confidence,
                closed_trace_target_p95=closed_trace_target_p95,
                closed_trace_compete_below_confidence=(
                    closed_trace_compete_below_confidence
                ),
                closed_trace_min_gain=closed_trace_min_gain,
            ),
            coord_scale,
        )
        for edge in edges
    ]
    # Hachures: prefer parametric regions (one HATCH per filled area) over
    # per-line fitting, which fragments at cross-hatch intersections and bloats
    # the output (~45% of all primitives). Fall back to per-line if Stage 2
    # produced no regions (e.g. hachure_mode="line").
    hachure_regions = graph.get("hachure_regions") or []
    if hachure_regions:
        hachure_primitives = [
            _scale_primitive(_hatch_region_to_primitive(r), coord_scale)
            for r in hachure_regions
        ]
    else:
        hachure_primitives = [
            _scale_primitive(prim, coord_scale)
            for edge in removed_hachures
            for prim in [_fit_removed_hachure(edge)]
            if prim is not None
        ]
    primitives = main_primitives + hachure_primitives

    confidences = [p.get("confidence", 0.0) for p in main_primitives]
    mean_conf   = float(np.mean(confidences)) if confidences else 0.0
    flagged     = mean_conf < effective_conf_thresh
    weak_compound_promotions = [
        p for p in main_primitives
        if p.get("fit_metadata", {}).get("strategy") == "compound_over_weak"
    ]
    closed_trace_simplifications = [
        p for p in main_primitives
        if p.get("fit_metadata", {}).get("strategy")
        == "closed_simplified_trace"
    ]

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
    if stroke_width is not None:
        doc["stroke_width"] = round(float(stroke_width), 2)
    if abs(stage2_scale - 1.0) >= 1e-9:
        doc["stage2_scale"] = stage2_scale
    doc["primitives"]  = primitives
    doc["quality_metrics"] = {
        "n_main_primitives": len(main_primitives),
        "n_hachure_primitives": len(hachure_primitives),
        "main_mean_confidence": mean_conf,
        "confidence_threshold": conf_thresh,
        "effective_confidence_threshold": effective_conf_thresh,
        "n_weak_compound_promotions": len(weak_compound_promotions),
        "n_weak_compound_path_atoms": sum(
            int(p.get("fit_metadata", {}).get("path_atoms", 0))
            for p in weak_compound_promotions
        ),
        "n_closed_trace_simplifications": len(closed_trace_simplifications),
        "n_closed_trace_source_points": sum(
            int(p.get("fit_metadata", {}).get("source_points", 0))
            for p in closed_trace_simplifications
        ),
        "n_closed_trace_output_vertices": sum(
            int(p.get("fit_metadata", {}).get("output_vertices", 0))
            for p in closed_trace_simplifications
        ),
    }
    doc["annotations"] = []   # filled by AP6.4 (Bezugszeichen) when available
    doc = _to_python(doc)
    with open(prims_path, "w") as f:
        json.dump(doc, f, indent=2)

    elapsed = time.perf_counter() - t_start

    if flagged:
        logger.warning(
            f"[{sketch_id}] FLAGGED — mean conf {mean_conf:.3f} "
            f"< threshold {effective_conf_thresh:.2f}"
        )
    else:
        logger.info(
            f"[{sketch_id}] Stage 3 done in {elapsed:.2f}s — "
            f"main_conf={mean_conf:.3f}, hachures={len(hachure_primitives)}"
        )

    return Stage3Result(
        sketch_id         = sketch_id,
        primitives_path   = prims_path,
        mean_confidence   = mean_conf,
        flagged           = flagged,
        processing_time_s = elapsed,
        n_primitives      = len(primitives),
        n_hachure_primitives = len(hachure_primitives),
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
