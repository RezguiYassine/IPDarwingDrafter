"""Ink that means an area: find it, and keep it out of the skeleton.

A skeleton reduces ink to its centreline. That is right for a stroke and wrong
for a filled region -- a solid section, a blackened face, a black slot --
whose medial axis is a branching scribble. The release-1 audit found figures
rendered that way ("no clean render"). Stage 1 now finds such regions, hands
each on as a filled polygon, and thins only the ink that is left.

A region is ink that passes four tests. Each was measured against a labelled
negative class before it shipped (docs/audits/2026-09-29/m1_region_detector.json):

  wide     a core of pixels at least `core_px` from paper, `min_core_px` large
  whole    the core grown back to the ink's edge -- a morphological opening,
           so a thin line touching the region stays a line
  solid    the ink fills at least `min_solidity` of the region's outline; a
           fill does, dense hatching merged at scan resolution does not
  compact  at most `max_aspect` widths long (outline area over the widest
           inscribed diameter squared). A bold stroke is wide and solid too,
           and is still a line: it is recorded, and left to the skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "ap3-stage1-regions-v1"

DEFAULTS = {
    "enabled": False,
    "core_px": 8,
    "min_core_px": 2500,
    "min_solidity": 0.85,
    "max_aspect": 12.0,
    # White specks inside a scanned fill are noise, not holes in the part.
    "min_hole_px": 30,
    # Ink the opening rounds off a sharp corner, when it touches nothing else.
    "tip_absorb_px": 128,
    "simplify_px": 1.0,
}


def settings(config: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(((config or {}).get("stage1") or {}).get("filled_regions") or {})
    return cfg


@dataclass
class Region:
    kind: str                  # "area": exported as a fill | "bold_stroke": left to the skeleton
    bbox: list                 # x, y, w, h in figure pixels
    outline_px: int            # pixels inside the outer contour
    ink_px: int                # ink pixels the region owns
    solidity: float
    diameter_px: float         # widest inscribed diameter
    aspect: float              # outline_px / diameter_px ** 2
    rings: list = field(default_factory=list)      # areas only: outer ring, then holes
    mask: np.ndarray | None = field(default=None, repr=False)   # bbox frame, bool

    def to_json(self) -> dict:
        return {"kind": self.kind, "bbox": [int(v) for v in self.bbox],
                "outline_px": int(self.outline_px), "ink_px": int(self.ink_px),
                "solidity": round(float(self.solidity), 4),
                "diameter_px": round(float(self.diameter_px), 2),
                "aspect": round(float(self.aspect), 2), "rings": self.rings}


def _filled_outline(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled > 0


def _fill_small_holes(mask: np.ndarray, min_hole_px: int) -> np.ndarray:
    holes = _filled_outline(mask) & ~mask
    if not holes.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 4)
    small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_hole_px) + 1
    return mask | np.isin(labels, small)


def _rings(mask: np.ndarray, origin: tuple[int, int], simplify_px: float) -> list:
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    x0, y0 = origin
    outer, holes = [], []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        simplified = cv2.approxPolyDP(contour, simplify_px, True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        ring = [[float(x + x0), float(y + y0)] for x, y in simplified]
        (holes if parent >= 0 else outer).append((cv2.contourArea(contour), ring))
    if not outer:
        return []
    # One connected mask has one outer boundary; keep the largest if a
    # degenerate contour ever splits it.
    outer.sort(key=lambda item: -item[0])
    return [outer[0][1]] + [ring for _, ring in holes]


def detect(ink: np.ndarray, cfg: dict) -> list[Region]:
    """Every wide, solid ink region, classified as an area or a bold stroke."""
    ink = np.asarray(ink, bool)
    core_px = float(cfg["core_px"])
    depth = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((depth >= core_px).astype(np.uint8), 8)
    big = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= cfg["min_core_px"]) + 1
    if not len(big):
        return []
    seeds = np.isin(labels, big)
    # Every pixel within core_px of a seed is ink by construction; the slack
    # absorbs the edge's quantisation so no one-pixel rim is left to thin.
    reach = cv2.distanceTransform((~seeds).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    grown = ink & (reach <= core_px + 1.5)

    m, grown_labels, grown_stats, _ = cv2.connectedComponentsWithStats(grown.astype(np.uint8), 8)
    # Ink within reach of a core but not joined to it -- a line passing close
    # by -- is not part of the region.
    anchored = set(np.unique(grown_labels[seeds]).tolist()) - {0}
    candidates, area_mask = [], np.zeros(ink.shape, bool)
    for i in sorted(anchored):
        x, y, w, h, count = (int(v) for v in grown_stats[i])
        mask = grown_labels[y:y + h, x:x + w] == i
        outline = int(_filled_outline(mask).sum())
        solidity = count / max(1, outline)
        if solidity < cfg["min_solidity"]:
            continue
        diameter = 2.0 * float(depth[y:y + h, x:x + w][mask].max())
        aspect = outline / max(1.0, diameter) ** 2
        kind = "area" if aspect <= cfg["max_aspect"] else "bold_stroke"
        candidates.append(Region(kind, [x, y, w, h], outline, count, solidity, diameter, aspect))
        if kind == "area":
            area_mask[y:y + h, x:x + w] |= mask

    regions = [r for r in candidates if r.kind == "bold_stroke"]
    if not area_mask.any():
        return regions

    # The opening rounds a sharp corner off; the tip it leaves is a speck of
    # ink touching nothing but the region, and belongs to it.
    residual = ink & ~area_mask
    r, residual_labels, residual_stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8)
    border = (cv2.dilate(area_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & residual
    touching = np.unique(residual_labels[border])
    tips = [t for t in touching
            if t and residual_stats[t, cv2.CC_STAT_AREA] <= cfg["tip_absorb_px"]]
    if tips:
        area_mask |= np.isin(residual_labels, tips)

    # Re-split so each absorbed tip joins the region it touches.
    k, final_labels, final_stats, _ = cv2.connectedComponentsWithStats(area_mask.astype(np.uint8), 8)
    detected = [c for c in candidates if c.kind == "area"]
    for j in range(1, k):
        x, y, w, h, count = (int(v) for v in final_stats[j])
        mask = final_labels[y:y + h, x:x + w] == j
        rings = _rings(_fill_small_holes(mask, int(cfg["min_hole_px"])), (x, y),
                       float(cfg["simplify_px"]))
        if not rings:
            continue
        # Measurements come from the detection-time region the tips joined.
        inside = [c for c in detected
                  if x <= c.bbox[0] and y <= c.bbox[1]
                  and c.bbox[0] + c.bbox[2] <= x + w and c.bbox[1] + c.bbox[3] <= y + h]
        source = max(inside, key=lambda c: c.ink_px) if inside else None
        regions.append(Region(
            "area", [x, y, w, h], int(_filled_outline(mask).sum()), count,
            source.solidity if source else 1.0,
            source.diameter_px if source else 0.0,
            source.aspect if source else 0.0,
            rings=rings, mask=mask))
    return regions


def routed_mask(regions: list[Region], shape: tuple[int, int]) -> np.ndarray:
    """The ink taken out of the skeleton: the union of every area's mask."""
    routed = np.zeros(shape, bool)
    for region in regions:
        if region.kind == "area" and region.mask is not None:
            x, y, w, h = region.bbox
            routed[y:y + h, x:x + w] |= region.mask
    return routed


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, bool)).tobytes()).hexdigest()


def write(regions: list[Region], routed: np.ndarray, cfg: dict,
          json_path: Path, mask_path: Path) -> dict:
    """The evidence Stage 3 exports from and the validator checks against."""
    cv2.imwrite(str(mask_path), routed.astype(np.uint8) * 255)
    areas = [r for r in regions if r.kind == "area"]
    document = {
        "schema": SCHEMA,
        "image_shape": [int(v) for v in routed.shape],
        "settings": {key: cfg[key] for key in DEFAULTS if key != "enabled"},
        "regions": [r.to_json() for r in areas],
        "bold_strokes": [r.to_json() for r in regions if r.kind == "bold_stroke"],
        "routed_mask": {"path": str(Path(mask_path).resolve()),
                        "pixels": int(routed.sum()), "sha256": mask_digest(routed)},
    }
    temporary = Path(json_path).with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    temporary.replace(json_path)
    return document
