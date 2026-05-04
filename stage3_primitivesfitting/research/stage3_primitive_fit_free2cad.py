"""
stage3_primitive_fit.py
========================
AP3 Vectorization Pipeline — Stage 3: Primitive Fitting

Converts each stroke edge (from Stage 2's graph JSON) into a geometric
primitive: line, arc, circle, ellipse, or polyline.

Two fitting layers:
  Layer 1 — Free2CAD DL fitter (Li et al., SIGGRAPH 2022)
      Autoregressive CNN+Transformer that classifies and fits each
      rasterised stroke edge as line / arc / circle.
      Requires: cloned Free2CAD repo + pretrained weights in config.yaml.

  Layer 2 — RANSAC geometric fallback (always available, pure NumPy)
      Priority: circle (closed) → line → arc → ellipse → polyline.
      Confidence = inlier_ratio × max(0, 1 − rms / MAX_RMS).

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
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Output contract ─────────────────────────────────────────────────────────

@dataclass
class Stage3Result:
    sketch_id:         str
    primitives_path:   Path
    fitter_used:       str    # "ransac" | "free2cad" | "mixed"
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
        "start":      [float(start[0]), float(start[1])],
        "end":        [float(end[0]),   float(end[1])],
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
        pts = np.array(edge["pixels"], dtype=np.float64)
    else:
        raw = edge.get("smooth_pts") or []
        pts = (np.array(raw, dtype=np.float64) if raw
               else np.array(edge["pixels"], dtype=np.float64))

    # ── Degenerate guard ─────────────────────────────────────────────────────
    if len(pts) < 2:
        return {"edge_id": edge_id, "type": "polyline",
                "points": [[float(p[0]), float(p[1])] for p in pts],
                "confidence": 0.0}

    # ── Priority 1: Circle (closed only) ────────────────────────────────────
    if is_closed and len(pts) >= _MIN_PTS_CIRCLE:
        try:
            r = _fit_circle_ransac(pts)
            r["edge_id"] = edge_id
            return r
        except ValueError:
            pass

    # ── Priority 2: Line ─────────────────────────────────────────────────────
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
# FREE2CAD DL FITTER  (Li et al., SIGGRAPH 2022)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# FREE2CAD DL FITTER  (PATCHED: v3 + hybrid fast path)
# ═══════════════════════════════════════════════════════════════════════════

class ModelNotAvailableError(Exception):
    pass


class Free2CADFitter:
    """
    Wrapper around the trained Free2CAD model. Supports three checkpoint
    versions:

      v1  : no BOS token in cmd_types       → seed decoder with LINE
      v2  : BOS token present in cmd_types  → seed decoder with BOS
      v3  : "version": 3 in checkpoint      → encoder-only, no decoder/seed

    PATCH (v3 + hybrid path):
      - Detects v3 checkpoints by the presence of `ckpt["version"] == 3` and
        builds the encoder-only model from train_free2cad_v3.build_model.
      - The fit_edge() method gains a hybrid fast path that handles the
        deterministic cases (closed-loop circles, 2-point lines) without
        invoking the model, and falls back to RANSAC if the model's
        confidence is below a threshold.
    """

    # Type ids — must match training
    _TYPE_LINE     = 0
    _TYPE_ARC      = 1
    _TYPE_CIRCLE   = 2
    _TYPE_POLYLINE = 3
    _TYPE_END      = 4   # only used by v1/v2 checkpoints

    _INFERENCE_FRACTION = 1.0

    # PATCH: thresholds for the hybrid fast path (Priority 2)
    _HYBRID_CIRCLE_MIN_PTS    = 12      # closed loop must have ≥ this many pts
    _HYBRID_CIRCLE_MAX_RES    = 0.15    # algebraic-circle residual / radius
    _HYBRID_FALLBACK_CONF     = 0.55    # below this confidence → RANSAC fallback

    def __init__(self, weights_path: str, device: str = "cuda"):
        self._model    = None
        self._device   = device
        self._ready    = False
        self._max_pts  = None

        # Version state — set during _load
        self._version  = 1            # 1, 2, or 3
        self._seed_id  = self._TYPE_LINE
        self._bos_id   = None         # None for v1 and v3, set for v2

        if not weights_path:
            raise ModelNotAvailableError("free2cad.weights not set in config.")
        weights = Path(weights_path)
        if not weights.exists():
            raise ModelNotAvailableError(
                f"Free2CAD weights not found: {weights}")
        self._load(weights, device)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, weights: Path, device: str) -> None:
        """
        PATCH: branches on `ckpt["version"]` to build the right architecture.
        v3 → train_free2cad_v3.build_model (encoder-only).
        v1/v2 → train_free2cad.build_model (seq2seq), as before.
        """
        import sys as _sys
        train_dir = str(Path(__file__).resolve().parent)
        if train_dir not in _sys.path:
            _sys.path.insert(0, train_dir)

        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but unavailable — loading Free2CAD on CPU.")
                device = "cpu"

            ckpt = torch.load(weights, map_location=device, weights_only=False)
            if "model_state_dict" not in ckpt or "config" not in ckpt:
                raise ModelNotAvailableError(
                    f"Checkpoint {weights} missing 'model_state_dict'/'config'.")

            version = int(ckpt.get("version", 1))
            cfg     = ckpt["config"]

            if version == 3:
                # PATCH: v3 — encoder-only architecture
                from train_free2cad_v3 import build_model as build_v3   # type: ignore
                model = build_v3(
                    max_pts      = cfg["max_pts"],
                    d_model      = cfg["d_model"],
                    n_heads      = cfg["n_heads"],
                    n_enc_layers = cfg["n_enc_layers"],
                    dropout      = cfg.get("dropout", 0.1),
                )
                seed_label = "encoder-only (v3, no decoder seed)"

            else:
                # v1 or v2 — original seq2seq architecture
                from train_free2cad import build_model as build_seq2seq # type: ignore
                model = build_seq2seq(
                    max_pts      = cfg["max_pts"],
                    max_strokes  = cfg["max_strokes"],
                    n_cmd_types  = cfg["n_cmd_types"],
                    d_model      = cfg["d_model"],
                    n_heads      = cfg["n_heads"],
                    n_enc_layers = cfg.get("n_enc_layers", 4),
                    n_dec_layers = cfg.get("n_dec_layers", 4),
                    dropout      = cfg.get("dropout",      0.1),
                )

                ckpt_vocab = ckpt.get("cmd_types")
                if isinstance(ckpt_vocab, dict) and "BOS" in ckpt_vocab:
                    self._bos_id  = int(ckpt_vocab["BOS"])
                    self._seed_id = self._bos_id
                    seed_label    = "BOS (v2)"
                else:
                    self._bos_id  = None
                    self._seed_id = self._TYPE_LINE
                    seed_label    = "LINE (v1, no BOS)"

            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval()
            model.to(device)

            self._model   = model
            self._device  = device
            self._max_pts = int(cfg["max_pts"])
            self._version = version
            self._ready   = True

            logger.info(
                f"Free2CAD v{version} loaded on {device} "
                f"(epoch {ckpt.get('epoch','?')}, "
                f"val_loss {ckpt.get('val_loss', float('nan')):.4f}, "
                f"seed={seed_label})")

        except ModelNotAvailableError:
            raise
        except Exception as exc:
            raise ModelNotAvailableError(f"Failed to load Free2CAD: {exc}")

    # ── Stroke encoding (unchanged for v1/v2; v3 uses point+mask form) ────────

    def _encode_edge_flat(self, edge: dict) -> tuple[np.ndarray, dict]:
        """
        Original v1/v2 encoding: returns a flat (max_pts*2,) feature vector.
        Used by the seq2seq forward path.
        """
        pts_norm, transform = self._encode_edge_points(edge)
        return pts_norm.flatten(), transform

    def _encode_edge_points(self, edge: dict) -> tuple[np.ndarray, dict]:
        """
        PATCH: shared point encoding used by both v1/v2 and v3.

        Returns (pts_norm, transform) where pts_norm has shape (max_pts, 2)
        and transform encodes the inverse for un-normalising parameters.

        Padding: -1 sentinel in the trailing rows. The boolean mask for v3
        is derived from this in fit_edge().
        """
        is_closed = edge.get("is_closed", False)
        raw = edge["pixels"] if is_closed else (edge.get("smooth_pts") or edge["pixels"])
        pts = np.array(raw, dtype=np.float64)

        frac = self._INFERENCE_FRACTION
        if len(pts) == 0:
            return (np.full((self._max_pts, 2), -1.0, dtype=np.float32),
                    {"center": [0.0, 0.0], "scale": 1.0, "frac": frac,
                     "n_real": 0})

        mn     = pts.min(axis=0)
        mx     = pts.max(axis=0)
        center = (mn + mx) / 2.0
        scale  = float((mx - mn).max())
        if scale < 1e-9:
            scale = 1.0

        pts_norm = ((pts - center) * (frac / scale) + 0.5).astype(np.float32)

        n_real = len(pts_norm)
        if n_real > self._max_pts:
            idx      = np.round(np.linspace(0, n_real - 1,
                                            self._max_pts)).astype(int)
            pts_norm = pts_norm[idx]
            n_real   = self._max_pts
        elif n_real < self._max_pts:
            pad      = np.full((self._max_pts - n_real, 2), -1.0,
                               dtype=np.float32)
            pts_norm = np.vstack([pts_norm, pad])

        return pts_norm, {
            "center": center.tolist(),
            "scale":  scale,
            "frac":   frac,
            "n_real": int(n_real),
        }

    # ── Inference (PATCHED with hybrid fast path) ─────────────────────────────

    def fit_edge(self, edge: dict) -> dict:
        """
        PATCH (Priority 2 hybrid path): handle deterministic cases without
        calling the model, and fall back to RANSAC when the model is uncertain.

        Order of attempts:
          1.  Fast path: closed loop with enough points → algebraic circle fit
          2.  Fast path: 2-point edge → exact line
          3.  Model inference (v1/v2 seq2seq OR v3 encoder-only)
          4.  Fallback: model confidence below threshold → RANSAC
        """
        # ── Fast path 1: closed loop → try circle fit ────────────────────────
        is_closed = edge.get("is_closed", False)
        raw       = edge.get("pixels", []) or []
        pts_arr   = np.array(raw, dtype=np.float64) if raw else np.zeros((0, 2))

        if is_closed and len(pts_arr) >= self._HYBRID_CIRCLE_MIN_PTS:
            try:
                circle = self._fast_circle_fit(pts_arr, edge)
                if circle:
                    return circle
            except Exception as exc:
                logger.debug(
                    f"Free2CAD hybrid circle-fit failed for edge "
                    f"{edge.get('id','?')}: {exc}")

        # ── Fast path 2: 2-point edge → exact line ────────────────────────────
        if len(pts_arr) == 2:
            return {
                "edge_id":    edge.get("id", -1),
                "type":       "line",
                "start":      [float(pts_arr[0, 0]), float(pts_arr[0, 1])],
                "end":        [float(pts_arr[1, 0]), float(pts_arr[1, 1])],
                "confidence": 1.0,
                "fitter":     "geometric_2pt",
            }

        # ── Slow path: model inference ────────────────────────────────────────
        try:
            if self._version == 3:
                result = self._fit_edge_v3(edge)
            else:
                result = self._fit_edge_v1v2(edge)
        except Exception as exc:
            logger.debug(
                f"Free2CAD inference failed for edge {edge.get('id','?')}: {exc}")
            result = {}

        # ── Fallback: low confidence → RANSAC ─────────────────────────────────
        if (not result
                or result.get("confidence", 0.0) < self._HYBRID_FALLBACK_CONF):
            try:
                ransac_result = fit_edge_ransac(edge)
                ransac_result["fitter"] = "ransac_fallback"
                return ransac_result
            except Exception:
                pass

        return result

    # ── Fast circle fit helper (Priority 2) ───────────────────────────────────

    def _fast_circle_fit(self, pts: np.ndarray, edge: dict) -> Optional[dict]:
        """
        PATCH: algebraic circle fit for closed loops.

        Returns the fit only if the residual (mean radial deviation / radius)
        is below _HYBRID_CIRCLE_MAX_RES. Otherwise returns None and the caller
        proceeds to model inference.
        """
        cx, cy, r = _fit_circle_algebraic(pts)   # from RANSAC section
        if r <= 1e-6:
            return None
        radii    = np.linalg.norm(pts - np.array([cx, cy]), axis=1)
        residual = float(np.mean(np.abs(radii - r)) / r)

        if residual > self._HYBRID_CIRCLE_MAX_RES:
            return None

        # Confidence shaped from residual: tight fit → high confidence
        conf = float(np.clip(1.0 - residual / self._HYBRID_CIRCLE_MAX_RES,
                             0.0, 1.0))
        return {
            "edge_id":    edge.get("id", -1),
            "type":       "circle",
            "center":     [float(cx), float(cy)],
            "radius":     float(r),
            "confidence": conf,
            "fitter":     "geometric_circle",
        }

    # ── v3 inference path (NEW) ───────────────────────────────────────────────

    def _fit_edge_v3(self, edge: dict) -> dict:
        """
        PATCH: encoder-only inference for v3 checkpoints.

        Forward signature: model(pts, mask) → (type_logits, param_pred)
        No BOS, no decoder, no autoregression.
        """
        import torch
        import torch.nn.functional as F

        pts_norm, transform = self._encode_edge_points(edge)
        n_real              = transform["n_real"]
        if n_real == 0:
            return {}

        # Build (1, max_pts, 2) tensor and (1, max_pts) bool mask
        pts_tensor = torch.from_numpy(pts_norm).unsqueeze(0).to(self._device)
        mask = torch.zeros(1, self._max_pts, dtype=torch.bool,
                           device=self._device)
        mask[0, :n_real] = True

        with torch.no_grad():
            type_logits, param_pred = self._model(pts_tensor, mask)
        # type_logits: (1, n_classes)   param_pred: (1, 6)

        probs      = F.softmax(type_logits[0], dim=-1).cpu().numpy()
        prim_type  = int(np.argmax(probs))
        confidence = float(probs[prim_type])
        params     = param_pred[0].cpu().numpy()

        result = self._decode(prim_type, params, confidence, edge, transform)
        if result:
            result["fitter"] = "free2cad_v3"
        return result

    # ── v1/v2 inference path (UNCHANGED) ─────────────────────────────────────

    def _fit_edge_v1v2(self, edge: dict) -> dict:
        """
        Original seq2seq inference for v1/v2 checkpoints.
        Forward: model(strokes, mask, seed_types, seed_params)
        """
        import torch
        import torch.nn.functional as F

        feature, transform = self._encode_edge_flat(edge)

        strokes = torch.from_numpy(feature).view(1, 1, -1).to(self._device)
        mask    = torch.ones(1, 1, dtype=torch.bool, device=self._device)
        seed_types  = torch.full((1, 1), self._seed_id,
                                 dtype=torch.long, device=self._device)
        seed_params = torch.zeros(1, 1, 6, dtype=torch.float32,
                                  device=self._device)

        with torch.no_grad():
            type_logits, param_preds = self._model(
                strokes, mask, seed_types, seed_params)

        logits = type_logits[0, 0].clone()
        if self._bos_id is not None and self._bos_id < logits.shape[0]:
            logits[self._bos_id] = float("-inf")

        probs      = F.softmax(logits, dim=-1).cpu().numpy()
        prim_type  = int(np.argmax(probs))
        confidence = float(probs[prim_type])
        params     = param_preds[0, 0].cpu().numpy()

        if prim_type == self._TYPE_END:
            return {}

        result = self._decode(prim_type, params, confidence, edge, transform)
        if result:
            result["fitter"] = f"free2cad_v{self._version}"
        return result

    # ── Output decoding (UNCHANGED) ───────────────────────────────────────────

    def _decode(self, prim_type: int, params: np.ndarray,
                confidence: float, edge: dict, transform: dict) -> dict:
        center = np.asarray(transform["center"], dtype=np.float64)
        scale  = float(transform["scale"])
        frac   = float(transform["frac"])
        p      = np.asarray(params, dtype=np.float64).flatten()
        eid    = edge.get("id", -1)
        ratio  = scale / frac

        def _pt(xn: float, yn: float) -> list:
            return [float(center[0] + (xn - 0.5) * ratio),
                    float(center[1] + (yn - 0.5) * ratio)]

        def _len(rn: float) -> float:
            return float(rn * ratio)

        if prim_type == self._TYPE_LINE:
            return {"edge_id": eid, "type": "line",
                    "start":      _pt(p[0], p[1]),
                    "end":        _pt(p[2], p[3]),
                    "confidence": confidence}

        if prim_type == self._TYPE_ARC:
            return {"edge_id": eid, "type": "arc",
                    "center":      _pt(p[0], p[1]),
                    "radius":      _len(p[2]),
                    "start_angle": float(p[3] * 360.0),
                    "end_angle":   float(p[4] * 360.0),
                    "confidence":  confidence}

        if prim_type == self._TYPE_CIRCLE:
            return {"edge_id": eid, "type": "circle",
                    "center":     _pt(p[0], p[1]),
                    "radius":     _len(p[2]),
                    "confidence": confidence}

        if prim_type == self._TYPE_POLYLINE:
            raw = edge.get("smooth_pts") or edge["pixels"]
            return {"edge_id": eid, "type": "polyline",
                    "points":     [[float(q[0]), float(q[1])] for q in raw],
                    "confidence": confidence}

        return {}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_model(config: dict, mode: str = "auto") -> Optional[Free2CADFitter]:
    """
    Load the Free2CAD fitter according to `mode`.

    mode = "ransac"   → always return None (pure RANSAC run).
    mode = "free2cad" → must succeed; raises if the model is unavailable.
    mode = "auto"     → try to load; on failure, warn and return None so
                        the caller falls back to RANSAC for every edge.
    """
    if mode == "ransac":
        logger.info("Fitter mode: ransac — skipping Free2CAD model load.")
        return None

    cfg     = config.get("free2cad", {})
    weights = cfg.get("weights", "")
    device  = cfg.get("device",  "cuda")

    try:
        model = Free2CADFitter(weights_path=weights, device=device)
        logger.info("Free2CAD fitter ready.")
        return model
    except ModelNotAvailableError as exc:
        if mode == "free2cad":
            raise SystemExit(
                f"Free2CAD not available: {exc}\n"
                "  → Train the model (see train_free2cad.py) or rerun with "
                "--fitter ransac."
            )
        logger.warning(
            f"Free2CAD not available: {exc}\n"
            "  → RANSAC fallback will be used for all edges."
        )
        return None


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC STAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run(
    graph_path:         Path,
    output_dir:         Path,
    sketch_id:          str,
    config:             dict,
    model:              Optional[Free2CADFitter] = None,
    cleaned_image_path: Optional[Path] = None,
) -> Stage3Result:
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
        Parsed config.yaml content.
    model : Free2CADFitter | None
        Pre-loaded DL fitter; None → RANSAC for every edge.
    cleaned_image_path : Path | None
        Reserved for the Egiazarian full-image fitter (not yet built).
    """
    t_start = time.perf_counter()

    prims_dir  = output_dir / "primitives"
    prims_dir.mkdir(parents=True, exist_ok=True)
    prims_path = prims_dir / f"{sketch_id}_primitives.json"

    with open(graph_path) as f:
        graph = json.load(f)
    edges = graph.get("edges", [])

    logger.info(f"[{sketch_id}] Stage 3 — {len(edges)} edge(s)")

    cfg_s3      = config.get("stage3", {})
    conf_thresh = cfg_s3.get("confidence_threshold", 0.60)

    primitives   = []
    fitters_used = set()

    for edge in edges:
        primitive = None

        # ── Try Free2CAD first ────────────────────────────────────────────
        if model is not None:
            try:
                result = model.fit_edge(edge)
                if result:
                    primitive = result
                    fitters_used.add("free2cad")
            except Exception as exc:
                logger.debug(
                    f"[{sketch_id}] Free2CAD error edge {edge['id']}: {exc}"
                )

        # ── RANSAC fallback ───────────────────────────────────────────────
        if primitive is None:
            primitive = fit_edge_ransac(edge)
            fitters_used.add("ransac")

        primitives.append(primitive)

    # ── Metadata ─────────────────────────────────────────────────────────────
    if fitters_used == {"free2cad"}:
        fitter_used = "free2cad"
    elif fitters_used == {"ransac"}:
        fitter_used = "ransac"
    else:
        fitter_used = "mixed"

    confidences = [p.get("confidence", 0.0) for p in primitives]
    mean_conf   = float(np.mean(confidences)) if confidences else 0.0
    flagged     = mean_conf < conf_thresh

    # ── Serialise ─────────────────────────────────────────────────────────────
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

    doc = _to_python({"sketch_id": sketch_id, "primitives": primitives})
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
            f"fitter={fitter_used}  conf={mean_conf:.3f}"
        )

    return Stage3Result(
        sketch_id         = sketch_id,
        primitives_path   = prims_path,
        fitter_used       = fitter_used,
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
            "Stage 3 — Primitive Fitting: fit geometric primitives to stroke graph(s).\n\n"
            "Single graph :  python stage3_primitive_fit.py path/to/graph.json\n"
            "Batch folder :  python stage3_primitive_fit.py --input-dir path/to/graphs/"
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
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output root directory (default: ./output)")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent.parent / "config.yaml",
                        help="Pipeline config file "
                             "(default: <project_root>/config.yaml)")
    parser.add_argument("--id", type=str, default=None,
                        help="Sketch ID (single-file mode only)")
    parser.add_argument("--fitter", choices=["auto", "free2cad", "ransac"],
                        default="auto",
                        help="Which fitter to use:\n"
                             "  auto     — try Free2CAD, fall back to RANSAC "
                             "if the model cannot be loaded (default).\n"
                             "  free2cad — require the trained Free2CAD model; "
                             "abort if unavailable. Per-edge parse errors still "
                             "fall back to RANSAC.\n"
                             "  ransac   — skip Free2CAD entirely; use RANSAC "
                             "for every edge.")
    args = parser.parse_args()

    # ── Load and resolve config ───────────────────────────────────────────────
    cfg = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        config_dir = args.config.resolve().parent
        if "free2cad" in cfg:
            w = cfg["free2cad"].get("weights", "")
            if w and not Path(w).is_absolute():
                cfg["free2cad"]["weights"] = str(config_dir / w)

    mdl = load_model(cfg, mode=args.fitter)

    # ── Collect input files ───────────────────────────────────────────────────
    if args.input is not None:
        graphs = [args.input]
    else:
        graphs = sorted(args.input_dir.glob("*_graph.json"))
        if not graphs:
            logger.error(f"No *_graph.json files found in {args.input_dir}")
            raise SystemExit(1)
        logger.info(f"Batch mode: {len(graphs)} graph(s) found")

    # ── Process ───────────────────────────────────────────────────────────────
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
            model      = mdl,
        )
        results.append(result)
        if result.flagged:
            n_flagged += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(results) == 1:
        r = results[0]
        print(f"\n{'─'*56}")
        print(f"  Sketch ID        : {r.sketch_id}")
        print(f"  Fitter used      : {r.fitter_used}")
        print(f"  Primitives       : {r.n_primitives}")
        print(f"  Mean confidence  : {r.mean_confidence:.3f}")
        print(f"  Flagged          : {'YES ⚠' if r.flagged else 'no'}")
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
                    print(f"    ⚠  {r.sketch_id}  (conf={r.mean_confidence:.3f})")
