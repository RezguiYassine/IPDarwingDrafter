"""Per-region feature vector for the learned hatch-region classifier.

Shared by tools/hatch_label_sample.py (build the labelling pack) and
tools/hatch_train.py (de-risk ROC-AUC). Each candidate hatch region (from
stage2._aggregate_hachure_regions) is turned into a fixed feature vector. The
single-feature discriminators all failed to separate genuine section hatching
from dense mechanical line-work; the bet is that a classifier over the JOINT
vector can. Features cover four families:

  structural : line count, cross-hatch flag, spacing, area, aspect, compactness
  angle      : spacing regularity (CV), orientation entropy, dominant-mode mass
  texture    : 2D fill-uniformity, ink density, FFT dominant-frequency power
  enclosure  : how much of the region boundary sits on main (non-hatch) ink
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FEATURE_NAMES = [
    "n_lines", "double", "n_angles", "spacing", "area", "aspect", "compactness",
    "spacing_cv", "angle_entropy", "dominant_mode_mass",
    "fill_uniformity", "ink_density", "fft_peak", "fft_sharpness",
    "boundary_on_main",
]


def _angle_delta(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _line_angle(pix) -> float | None:
    p = np.asarray(pix, dtype=np.float64)
    if len(p) < 2:
        return None
    c = p - p.mean(axis=0)
    d = (p[-1] - p[0]) if len(p) == 2 else np.linalg.svd(c, full_matrices=False)[2][0]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)


def _fft_peak(patch: np.ndarray) -> tuple[float, float]:
    """(peak/mean, peak-sharpness) of the dominant off-DC spatial frequency."""
    if patch.size < 64 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return 0.0, 0.0
    p = patch.astype(np.float32)
    p = p - p.mean()
    p = p * (np.hanning(p.shape[0])[:, None] * np.hanning(p.shape[1])[None, :])
    F = np.abs(np.fft.fftshift(np.fft.fft2(p)))
    cy, cx = np.array(F.shape) // 2
    F[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0
    m = float(F.mean()) + 1e-6
    peak = float(F.max())
    # sharpness: peak vs the 99th percentile (a single sharp peak >> broadband)
    p99 = float(np.percentile(F, 99)) + 1e-6
    return peak / m, peak / p99


def _fill_uniformity(mask: np.ndarray, cell: int = 14) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return 0.0
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    H, W = sub.shape
    gh, gw = max(1, H // cell), max(1, W // cell)
    occ = tot = 0
    for i in range(gh):
        for j in range(gw):
            tot += 1
            if sub[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].any():
                occ += 1
    return occ / max(1, tot)


def region_feature_vector(region: dict, member_pix: list, ink_all: np.ndarray,
                          ink_main: np.ndarray) -> dict:
    """Compute the feature dict for one candidate region.

    region     : {boundary, angles, spacing, n_lines, double}
    member_pix : list of Nx2 pixel arrays for the region's hatch lines
    ink_all    : full binary ink (all skeleton pixels)
    ink_main   : binary ink of MAIN (non-hachure) edges only — for enclosure
    """
    H, W = ink_all.shape
    boundary = np.asarray(region["boundary"], dtype=np.int32)
    angles = region.get("angles") or []
    spacing = float(region.get("spacing") or 0.0)
    n_lines = int(region.get("n_lines", len(member_pix)))

    # region mask + geometry
    rmask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(rmask, [boundary], 1)
    area = float(cv2.contourArea(boundary)) + 1e-6
    perim = float(cv2.arcLength(boundary, True)) + 1e-6
    x, y, bw, bh = cv2.boundingRect(boundary)
    aspect = max(bw, bh) / max(1.0, min(bw, bh))
    compactness = perim / math.sqrt(area)             # 2D fill ~3.5; thin run high

    # angle stats from members
    mangs = np.array([a for a in (_line_angle(p) for p in member_pix) if a is not None])
    if len(mangs):
        hist, _ = np.histogram(mangs % 180.0, bins=np.arange(0, 181, 12.0))
        ph = hist / max(1, hist.sum())
        angle_entropy = float(-np.sum(ph[ph > 0] * np.log(ph[ph > 0])))
        dom = (angles[0] if angles else float(mangs[np.argmax(np.bincount(
            (mangs // 12).astype(int)))] * 12))
        dominant_mode_mass = float(np.mean([_angle_delta(a, dom) <= 12 for a in mangs]))
    else:
        angle_entropy, dominant_mode_mass = 0.0, 0.0

    # spacing CV (members projected onto perpendicular of dominant angle)
    spacing_cv = 0.0
    if angles and len(member_pix) >= 3:
        th = math.radians(angles[0]); perp = np.array([-math.sin(th), math.cos(th)])
        cen = np.array([np.asarray(p, float).mean(axis=0) for p in member_pix])
        proj = np.sort(cen @ perp)
        gaps = np.diff(proj); gaps = gaps[gaps > 0.5]
        if len(gaps) > 1 and gaps.mean() > 0:
            spacing_cv = float(gaps.std() / gaps.mean())

    # texture within the region
    region_ink = (ink_all > 0) & (rmask > 0)
    ink_density = float(region_ink.sum()) / area
    fill_uniformity = _fill_uniformity(region_ink.astype(np.uint8))
    patch = (ink_all[y:y + bh, x:x + bw] > 0).astype(np.float32)
    fft_peak, fft_sharpness = _fft_peak(patch)

    # enclosure: fraction of boundary perimeter pixels near MAIN ink
    bmask = np.zeros((H, W), np.uint8)
    cv2.polylines(bmask, [boundary], True, 1, 1)
    main_d = cv2.dilate((ink_main > 0).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    bpix = bmask > 0
    boundary_on_main = float((main_d[bpix] > 0).mean()) if bpix.any() else 0.0

    return {
        "n_lines": n_lines, "double": int(bool(region.get("double"))),
        "n_angles": len(angles), "spacing": spacing, "area": area, "aspect": aspect,
        "compactness": compactness, "spacing_cv": spacing_cv,
        "angle_entropy": angle_entropy, "dominant_mode_mass": dominant_mode_mass,
        "fill_uniformity": fill_uniformity, "ink_density": ink_density,
        "fft_peak": fft_peak, "fft_sharpness": fft_sharpness,
        "boundary_on_main": boundary_on_main,
    }


def feature_row(feats: dict) -> list:
    return [float(feats.get(k, 0.0)) for k in FEATURE_NAMES]
