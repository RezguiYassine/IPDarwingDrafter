"""
stage0_handle_references.py
===========================
AP3 Vectorization Pipeline - Stage 0: patent reference handling.

Patent drawings often contain reference numerals and leader/help lines. If
these are left in the raster before skeletonisation, Stage 2 sees every leader
intersection as a real topology event and splits long strokes into fragments.

This stage detects likely reference labels, links them to nearby leader-line
segments, records them in JSON, removes them from a reference-free raster, and
can later attach the extracted annotations to the Stage 3 primitive JSON for
Stage 4 reinjection.

The detector is deliberately bounded: leader-linked labels are preferred, but
small unleadered text clusters can also be removed when they pass strict shape
filters. OCR is not assumed. SVG export can preserve numeral appearance through
transparent crop overlays; DXF export reinjects leader geometry and will add
MTEXT only when a future OCR step fills annotation["text"].
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_iterations": 10,
    "min_component_area": 8,
    "max_component_area": 1800,
    "min_component_width": 2,
    "max_component_width": 90,
    "min_component_height": 5,
    "max_component_height": 95,
    "max_candidate_components": 2500,
    "text_group_max_gap": 22,
    "text_group_max_y_delta": 18,
    "cluster_min_width": 4,
    "cluster_max_width": 180,
    "cluster_min_height": 7,
    "cluster_max_height": 105,
    "cluster_max_components": 6,
    "cluster_min_density": 0.06,
    "cluster_max_density": 0.80,
    "require_leader": True,
    "remove_unleadered_labels": True,
    "unleadered_min_components": 1,
    "unleadered_max_components": 6,
    "unleadered_min_width": 4,
    "unleadered_max_width": 150,
    "unleadered_min_height": 7,
    "unleadered_max_height": 95,
    "unleadered_max_bbox_area": 6500,
    "unleadered_max_ink_area": 2200,
    "unleadered_min_density": 0.05,
    "unleadered_max_density": 0.78,
    "unleadered_allow_single_component": True,
    "unleadered_single_margin_ratio": 0.24,
    "unleadered_single_min_density": 0.18,
    "unleadered_single_max_bbox_area": 3000,
    "unleadered_single_max_ink_area": 1400,
    "remove_figure_labels": True,
    "figure_min_component_area": 40,
    "figure_max_component_area": 6500,
    "figure_min_component_width": 4,
    "figure_max_component_width": 150,
    "figure_min_component_height": 8,
    "figure_max_component_height": 150,
    "figure_component_max_aspect": 4.0,
    "figure_group_max_gap": 36,
    "figure_group_max_y_delta": 34,
    "figure_cluster_min_components": 2,
    "figure_cluster_max_components": 12,
    "figure_cluster_min_width": 18,
    "figure_cluster_max_width": 340,
    "figure_cluster_min_height": 10,
    "figure_cluster_max_height": 190,
    "figure_cluster_max_bbox_area": 32000,
    "figure_cluster_max_ink_area": 18000,
    "figure_cluster_min_density": 0.04,
    "figure_cluster_max_density": 0.86,
    "figure_require_margin": False,
    "figure_margin_ratio": 0.22,
    "hough_threshold": 24,
    "hough_min_line_length": 35,
    "hough_max_line_gap": 6,
    "leader_endpoint_margin": 18,
    "leader_segment_margin": 12,
    "leader_min_support_ratio": 0.50,
    "leader_support_radius": 1,
    "leader_max_length": 450,
    "leader_max_length_ratio": 0.28,
    "max_leaders_per_label": 2,
    "leader_dedup_distance": 8,
    "leader_dedup_angle": 8,
    "text_bbox_pad": 3,
    "crop_pad": 2,
    "leader_mask_thickness": 5,
    "leader_tip_trim": 8,
    "mask_dilate": 1,
    "repair_after_removal": True,
    "repair_close_kernel": 7,
    "max_reference_labels": 220,
    "max_removed_ink_ratio": 0.70,
    "max_total_removed_ink_ratio": 0.82,
    "text_only_on_guard": True,
    "max_guard_text_only_removed_ink_ratio": 0.20,
    "max_guard_text_only_labels": 240,
}


@dataclass
class Stage0Result:
    sketch_id: str
    reference_free_path: Path
    references_json_path: Path
    mask_path: Path
    n_labels: int = 0
    n_leaders: int = 0
    n_iterations: int = 0
    removed_ink_ratio: float = 0.0
    repair_pixels: int = 0
    active_removal: bool = False
    flagged: bool = False
    processing_time_s: float = 0.0


def _stage0_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config.get("stage0", {}) or {})
    return cfg


def _read_gray(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read raster image: {path}")
    return gray


def _ink_mask(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Return a bool mask where True means ink, plus the background gray value.
    Works for normal black-on-white scans and inverted scans.
    """
    bg_is_light = float(np.median(gray)) >= 127.0
    mode = cv2.THRESH_BINARY_INV if bg_is_light else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
    return binary > 0, (255 if bg_is_light else 0)


def _json_num(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _bbox_union(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    xe: list[int] = []
    ye: list[int] = []
    for x, y, w, h in boxes:
        xs.append(int(x))
        ys.append(int(y))
        xe.append(int(x + w))
        ye.append(int(y + h))
    x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
    return x0, y0, x1 - x0, y1 - y0


def _pad_bbox(
    bbox: tuple[int, int, int, int],
    pad: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(width, int(x + w) + pad)
    y1 = min(height, int(y + h) + pad)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _point_bbox_distance(
    point: tuple[float, float],
    bbox: tuple[int, int, int, int],
) -> float:
    px, py = point
    x, y, w, h = bbox
    dx = max(float(x) - px, 0.0, px - float(x + w))
    dy = max(float(y) - py, 0.0, py - float(y + h))
    return math.hypot(dx, dy)


def _point_segment_distance(
    point: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _segment_bbox_distance(
    p1: tuple[float, float],
    p2: tuple[float, float],
    bbox: tuple[int, int, int, int],
) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return min(_point_bbox_distance(p1, bbox), _point_bbox_distance(p2, bbox))

    rect = (int(x), int(y), int(w), int(h))
    pt1 = (int(round(p1[0])), int(round(p1[1])))
    pt2 = (int(round(p2[0])), int(round(p2[1])))
    if cv2.clipLine(rect, pt1, pt2)[0]:
        return 0.0

    corners = (
        (float(x), float(y)),
        (float(x + w), float(y)),
        (float(x), float(y + h)),
        (float(x + w), float(y + h)),
    )
    distances = [
        _point_bbox_distance(p1, bbox),
        _point_bbox_distance(p2, bbox),
        *(_point_segment_distance(corner, p1, p2) for corner in corners),
    ]
    return min(distances)


def _cluster_key(cluster: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in cluster["bbox"])


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def _bbox_intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _bbox_overlap_fraction(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    inter = _bbox_intersection_area(a, b)
    return inter / max(1.0, float(min(_bbox_area(a), _bbox_area(b))))


def _angle_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0


def _angle_delta(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _segment_similar(a: dict[str, Any], b: dict[str, Any], cfg: dict[str, Any]) -> bool:
    max_d = float(cfg["leader_dedup_distance"])
    max_a = float(cfg["leader_dedup_angle"])
    if _angle_delta(float(a["angle_deg"]), float(b["angle_deg"])) > max_a:
        return False
    ap1 = np.asarray(a["p1"], dtype=float)
    ap2 = np.asarray(a["p2"], dtype=float)
    bp1 = np.asarray(b["p1"], dtype=float)
    bp2 = np.asarray(b["p2"], dtype=float)
    same = max(np.linalg.norm(ap1 - bp1), np.linalg.norm(ap2 - bp2))
    swapped = max(np.linalg.norm(ap1 - bp2), np.linalg.norm(ap2 - bp1))
    return min(same, swapped) <= max_d


def _segment_support_ratio(
    mask: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    radius: int,
) -> float:
    length = max(1.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    n = max(8, int(round(length)))
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)
    h, w = mask.shape
    hits = 0
    for xf, yf in zip(xs, ys):
        x = int(round(xf))
        y = int(round(yf))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        if np.any(mask[y0:y1, x0:x1]):
            hits += 1
    return hits / float(n)


def _component_candidates(
    ink: np.ndarray,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    comps: list[dict[str, Any]] = []
    for idx in range(1, n):
        x, y, w, h, area = stats[idx]
        if not (cfg["min_component_area"] <= area <= cfg["max_component_area"]):
            continue
        if not (cfg["min_component_width"] <= w <= cfg["max_component_width"]):
            continue
        if not (cfg["min_component_height"] <= h <= cfg["max_component_height"]):
            continue
        aspect = w / max(float(h), 1.0)
        if aspect > 4.0:
            continue
        cx, cy = centroids[idx]
        comps.append({
            "idx": int(idx),
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "centroid": (float(cx), float(cy)),
        })
    return comps


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _vertical_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    _, ay, _, ah = a
    _, by, _, bh = b
    top = max(ay, by)
    bot = min(ay + ah, by + bh)
    overlap = max(0, bot - top)
    return overlap / max(1.0, float(min(ah, bh)))


def _group_text_clusters(
    comps: list[dict[str, Any]],
    ink: np.ndarray,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not comps:
        return []
    if len(comps) > int(cfg["max_candidate_components"]):
        logger.warning(
            "Stage 0: too many small components (%d); skipping text grouping",
            len(comps),
        )
        return []

    uf = _UnionFind(len(comps))
    ordered = sorted(range(len(comps)), key=lambda i: comps[i]["bbox"][0])
    max_gap = float(cfg["text_group_max_gap"])
    max_y = float(cfg["text_group_max_y_delta"])

    for pos, i in enumerate(ordered):
        xi, yi, wi, hi = comps[i]["bbox"]
        cxi, cyi = comps[i]["centroid"]
        right_i = xi + wi
        for j in ordered[pos + 1:]:
            xj, yj, wj, hj = comps[j]["bbox"]
            if xj - right_i > max_gap:
                break
            cxj, cyj = comps[j]["centroid"]
            x_gap = max(0, xj - right_i, xi - (xj + wj))
            same_line = (
                abs(cyi - cyj) <= max(max_y, 0.75 * max(hi, hj))
                or _vertical_overlap_ratio(comps[i]["bbox"], comps[j]["bbox"]) >= 0.35
            )
            if x_gap <= max_gap and same_line and abs(cxi - cxj) < 220:
                uf.union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for i, comp in enumerate(comps):
        groups.setdefault(uf.find(i), []).append(comp)

    clusters: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) > int(cfg["cluster_max_components"]):
            continue
        bbox = _bbox_union(m["bbox"] for m in members)
        x, y, w, h = bbox
        if not (cfg["cluster_min_width"] <= w <= cfg["cluster_max_width"]):
            continue
        if not (cfg["cluster_min_height"] <= h <= cfg["cluster_max_height"]):
            continue
        area = int(sum(int(m["area"]) for m in members))
        density = area / max(1.0, float(w * h))
        if not (cfg["cluster_min_density"] <= density <= cfg["cluster_max_density"]):
            continue
        ys, xs = np.where(ink[y:y + h, x:x + w])
        if len(xs) == 0:
            continue
        centroid = (float(x + xs.mean()), float(y + ys.mean()))
        clusters.append({
            "bbox": bbox,
            "centroid": centroid,
            "components": [m["bbox"] for m in members],
            "ink_area": area,
            "density": float(density),
        })

    clusters.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return clusters


def _hough_segments(
    ink: np.ndarray,
    clusters: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    line_source = (ink.astype(np.uint8) * 255)
    pad = int(cfg["text_bbox_pad"]) + 1
    h, w = line_source.shape
    for cluster in clusters:
        x, y, bw, bh = _pad_bbox(cluster["bbox"], pad, w, h)
        line_source[y:y + bh, x:x + bw] = 0
    line_source_bool = line_source > 0

    lines = cv2.HoughLinesP(
        line_source,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(cfg["hough_threshold"]),
        minLineLength=int(cfg["hough_min_line_length"]),
        maxLineGap=int(cfg["hough_max_line_gap"]),
    )
    if lines is None:
        return []

    long_edge = max(h, w)
    max_len_cfg = float(cfg.get("leader_max_length", 0) or 0)
    max_len_ratio = float(cfg.get("leader_max_length_ratio", 0) or 0)
    max_len = max_len_cfg
    if max_len_ratio > 0:
        ratio_len = max_len_ratio * long_edge
        max_len = min(max_len, ratio_len) if max_len > 0 else ratio_len

    segments: list[dict[str, Any]] = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = (float(v) for v in raw)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < float(cfg["hough_min_line_length"]):
            continue
        if max_len > 0 and length > max_len:
            continue
        p1 = (x1, y1)
        p2 = (x2, y2)
        support = _segment_support_ratio(
            line_source_bool,
            p1,
            p2,
            int(cfg["leader_support_radius"]),
        )
        if support < float(cfg["leader_min_support_ratio"]):
            continue
        segments.append({
            "p1": [x1, y1],
            "p2": [x2, y2],
            "length": float(length),
            "angle_deg": float(_angle_deg(p1, p2)),
            "support_ratio": float(support),
        })
    return segments


def _assign_leaders(
    clusters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    endpoint_margin = float(cfg["leader_endpoint_margin"])
    segment_margin = float(cfg.get("leader_segment_margin", 0) or 0)
    max_per_label = int(cfg["max_leaders_per_label"])
    labels: list[dict[str, Any]] = []

    for cluster in clusters:
        bbox = cluster["bbox"]
        candidates: list[tuple[float, dict[str, Any]]] = []
        for seg in segments:
            p1 = (float(seg["p1"][0]), float(seg["p1"][1]))
            p2 = (float(seg["p2"][0]), float(seg["p2"][1]))
            d1 = _point_bbox_distance(p1, bbox)
            d2 = _point_bbox_distance(p2, bbox)
            endpoint_dist = min(d1, d2)
            segment_dist = _segment_bbox_distance(p1, p2, bbox)
            mask_full_segment = False
            if endpoint_dist <= endpoint_margin:
                near_dist = endpoint_dist
                match_type = "endpoint"
                if d1 <= d2:
                    start, tip = p1, p2
                else:
                    start, tip = p2, p1
            elif segment_margin > 0 and segment_dist <= segment_margin:
                near_dist = segment_dist
                match_type = "segment"
                start, tip = p1, p2
                mask_full_segment = True
            else:
                continue
            leader = dict(seg)
            leader["label_endpoint"] = [float(start[0]), float(start[1])]
            leader["leader_to"] = [float(tip[0]), float(tip[1])]
            leader["endpoint_distance"] = float(near_dist)
            leader["segment_distance"] = float(segment_dist)
            leader["match_type"] = match_type
            leader["mask_full_segment"] = bool(mask_full_segment)
            score = near_dist + 0.015 * float(seg["length"])
            candidates.append((score, leader))

        selected: list[dict[str, Any]] = []
        for _, leader in sorted(candidates, key=lambda item: item[0]):
            if any(_segment_similar(leader, prev, cfg) for prev in selected):
                continue
            selected.append(leader)
            if len(selected) >= max_per_label:
                break

        if selected or not bool(cfg["require_leader"]):
            item = dict(cluster)
            item["leader_lines"] = selected
            item["kind"] = "leadered_text" if selected else "text_cluster"
            item["removal_mode"] = "full"
            labels.append(item)

    return labels


def _looks_like_unleadered_reference(
    cluster: dict[str, Any],
    cfg: dict[str, Any],
    image_shape: tuple[int, int],
) -> bool:
    x, y, w, h = cluster["bbox"]
    n_components = len(cluster.get("components", []))
    if not (
        int(cfg["unleadered_min_components"])
        <= n_components
        <= int(cfg["unleadered_max_components"])
    ):
        return False
    density = float(cluster.get("density", 0.0))
    if n_components == 1:
        if not bool(cfg.get("unleadered_allow_single_component", False)):
            return False
        if not _in_margin_band(
            cluster["bbox"],
            image_shape,
            float(cfg["unleadered_single_margin_ratio"]),
        ):
            return False
        if density < float(cfg["unleadered_single_min_density"]):
            return False
        if w * h > int(cfg["unleadered_single_max_bbox_area"]):
            return False
        if int(cluster.get("ink_area", 0)) > int(cfg["unleadered_single_max_ink_area"]):
            return False
    if not (int(cfg["unleadered_min_width"]) <= w <= int(cfg["unleadered_max_width"])):
        return False
    if not (int(cfg["unleadered_min_height"]) <= h <= int(cfg["unleadered_max_height"])):
        return False
    if w * h > int(cfg["unleadered_max_bbox_area"]):
        return False
    if int(cluster.get("ink_area", 0)) > int(cfg["unleadered_max_ink_area"]):
        return False
    if not (
        float(cfg["unleadered_min_density"])
        <= density
        <= float(cfg["unleadered_max_density"])
    ):
        return False
    return True


def _select_unleadered_labels(
    clusters: list[dict[str, Any]],
    leadered_labels: list[dict[str, Any]],
    cfg: dict[str, Any],
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    if not bool(cfg.get("remove_unleadered_labels", False)):
        return []
    leadered = {_cluster_key(label) for label in leadered_labels}
    labels: list[dict[str, Any]] = []
    for cluster in clusters:
        if _cluster_key(cluster) in leadered:
            continue
        if not _looks_like_unleadered_reference(cluster, cfg, image_shape):
            continue
        item = dict(cluster)
        item["leader_lines"] = []
        item["kind"] = "unleadered_text"
        item["removal_mode"] = "text"
        labels.append(item)
    return labels


def _figure_component_candidates(
    ink: np.ndarray,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    comps: list[dict[str, Any]] = []
    max_aspect = float(cfg["figure_component_max_aspect"])
    for idx in range(1, n):
        x, y, w, h, area = stats[idx]
        if not (cfg["figure_min_component_area"] <= area <= cfg["figure_max_component_area"]):
            continue
        if not (cfg["figure_min_component_width"] <= w <= cfg["figure_max_component_width"]):
            continue
        if not (cfg["figure_min_component_height"] <= h <= cfg["figure_max_component_height"]):
            continue
        aspect = max(w / max(float(h), 1.0), h / max(float(w), 1.0))
        if aspect > max_aspect:
            continue
        cx, cy = centroids[idx]
        comps.append({
            "idx": int(idx),
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "centroid": (float(cx), float(cy)),
        })
    return comps


def _figure_group_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    group_cfg = dict(cfg)
    group_cfg.update({
        "max_candidate_components": max(5000, int(cfg.get("max_candidate_components", 0))),
        "text_group_max_gap": int(cfg["figure_group_max_gap"]),
        "text_group_max_y_delta": int(cfg["figure_group_max_y_delta"]),
        "cluster_min_width": int(cfg["figure_cluster_min_width"]),
        "cluster_max_width": int(cfg["figure_cluster_max_width"]),
        "cluster_min_height": int(cfg["figure_cluster_min_height"]),
        "cluster_max_height": int(cfg["figure_cluster_max_height"]),
        "cluster_max_components": int(cfg["figure_cluster_max_components"]),
        "cluster_min_density": float(cfg["figure_cluster_min_density"]),
        "cluster_max_density": float(cfg["figure_cluster_max_density"]),
    })
    return group_cfg


def _transpose_components(comps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transposed: list[dict[str, Any]] = []
    for comp in comps:
        x, y, w, h = comp["bbox"]
        cx, cy = comp["centroid"]
        item = dict(comp)
        item["bbox"] = (int(y), int(x), int(h), int(w))
        item["centroid"] = (float(cy), float(cx))
        transposed.append(item)
    return transposed


def _untranspose_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    x, y, w, h = cluster["bbox"]
    cx, cy = cluster["centroid"]
    item = dict(cluster)
    item["bbox"] = (int(y), int(x), int(h), int(w))
    item["centroid"] = (float(cy), float(cx))
    item["components"] = [
        (int(cy0), int(cx0), int(ch), int(cw))
        for cx0, cy0, cw, ch in cluster.get("components", [])
    ]
    return item


def _in_margin_band(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    margin_ratio: float,
) -> bool:
    h, w = image_shape
    x, y, bw, bh = bbox
    cx = x + 0.5 * bw
    cy = y + 0.5 * bh
    mx = margin_ratio * w
    my = margin_ratio * h
    return cx <= mx or cx >= w - mx or cy <= my or cy >= h - my


def _figure_label_clusters(
    ink: np.ndarray,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(cfg.get("remove_figure_labels", False)):
        return []
    comps = _figure_component_candidates(ink, cfg)
    if not comps:
        return []

    group_cfg = _figure_group_cfg(cfg)
    horizontal = _group_text_clusters(comps, ink, group_cfg)
    vertical = [
        _untranspose_cluster(cluster)
        for cluster in _group_text_clusters(_transpose_components(comps), ink.T, group_cfg)
    ]

    clusters: list[dict[str, Any]] = []
    require_margin = bool(cfg.get("figure_require_margin", True))
    margin_ratio = float(cfg["figure_margin_ratio"])
    for cluster in [*horizontal, *vertical]:
        bbox = cluster["bbox"]
        if len(cluster.get("components", [])) < int(cfg["figure_cluster_min_components"]):
            continue
        if _bbox_area(bbox) > int(cfg["figure_cluster_max_bbox_area"]):
            continue
        if int(cluster.get("ink_area", 0)) > int(cfg["figure_cluster_max_ink_area"]):
            continue
        if require_margin and not _in_margin_band(bbox, ink.shape, margin_ratio):
            continue
        if any(_bbox_overlap_fraction(bbox, prev["bbox"]) > 0.65 for prev in clusters):
            continue
        item = dict(cluster)
        item["leader_lines"] = []
        item["kind"] = "figure_or_border_text"
        item["removal_mode"] = "text"
        clusters.append(item)

    clusters.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return clusters


def _transparent_label_crop(
    gray: np.ndarray,
    ink: np.ndarray,
    bbox: tuple[int, int, int, int],
    out_path: Path,
    crop_pad: int,
) -> tuple[int, int, int, int]:
    h, w = gray.shape
    x, y, bw, bh = _pad_bbox(bbox, crop_pad, w, h)
    crop_gray = gray[y:y + bh, x:x + bw]
    crop_ink = ink[y:y + bh, x:x + bw]
    rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    rgba[..., 0:3] = crop_gray[..., None]
    rgba[..., 3] = (crop_ink.astype(np.uint8) * 255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgba)
    return x, y, bw, bh


def _build_reference_doc(
    *,
    sketch_id: str,
    source_path: Path,
    gray: np.ndarray,
    ink: np.ndarray,
    labels: list[dict[str, Any]],
    mask_path: Path,
    reference_free_path: Path,
    crop_dir: Path,
    cfg: dict[str, Any],
    active_removal: bool,
    removed_ink_ratio: float,
    repair_pixels: int,
    n_iterations: int,
    iteration_summaries: list[dict[str, Any]],
    flagged: bool,
    reason: str,
) -> dict[str, Any]:
    h, w = gray.shape
    references = []
    for i, label in enumerate(labels, start=1):
        ref_id = f"ref_{i:03d}"
        crop_path = crop_dir / f"{sketch_id}_{ref_id}.png"
        crop_bbox = _transparent_label_crop(
            gray, ink, label["bbox"], crop_path, int(cfg["crop_pad"])
        )
        cx, cy = label["centroid"]
        leaders = []
        for leader in label.get("leader_lines", []):
            leaders.append({
                "p1": [_json_num(v) for v in leader["p1"]],
                "p2": [_json_num(v) for v in leader["p2"]],
                "label_endpoint": [_json_num(v) for v in leader["label_endpoint"]],
                "leader_to": [_json_num(v) for v in leader["leader_to"]],
                "length": _json_num(leader["length"]),
                "angle_deg": _json_num(leader["angle_deg"]),
                "endpoint_distance": _json_num(leader["endpoint_distance"]),
                "segment_distance": _json_num(leader.get("segment_distance", 0.0)),
                "match_type": leader.get("match_type", "endpoint"),
                "removed": bool(leader.get("removed", True)),
            })
        references.append({
            "id": ref_id,
            "iteration": int(label.get("iteration", 1)),
            "kind": label.get("kind", "leadered_text"),
            "removal_mode": label.get("removal_mode", "full"),
            "text": "",
            "bbox": [_json_num(v) for v in label["bbox"]],
            "crop_bbox": [_json_num(v) for v in crop_bbox],
            "position": [_json_num(cx), _json_num(cy)],
            "components": [
                [_json_num(v) for v in comp]
                for comp in label.get("components", [])
            ],
            "ink_area": _json_num(label.get("ink_area", 0)),
            "density": _json_num(label.get("density", 0.0)),
            "crop_path": str(crop_path),
            "leader_lines": leaders,
        })

    return {
        "schema": "stage0_references_v1",
        "sketch_id": sketch_id,
        "source_path": str(source_path),
        "image_size": [int(w), int(h)],
        "active_removal": bool(active_removal),
        "flagged": bool(flagged),
        "reason": reason,
        "n_iterations": int(n_iterations),
        "iterations": iteration_summaries,
        "removed_ink_ratio": float(removed_ink_ratio),
        "repair_pixels": int(repair_pixels),
        "n_labels": len(references),
        "n_leaders": sum(len(r.get("leader_lines", [])) for r in references),
        "mask_path": str(mask_path),
        "reference_free_path": str(reference_free_path),
        "reference_labels": references,
    }


def _build_removal_mask(
    image_shape: tuple[int, int],
    labels: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    include_leaders: bool = True,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pad = int(cfg["text_bbox_pad"])
    for label in labels:
        x, y, bw, bh = _pad_bbox(label["bbox"], pad, w, h)
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, thickness=-1)
        if not include_leaders:
            continue
        for leader in label.get("leader_lines", []):
            if leader.get("mask_full_segment", False):
                start_raw = leader["p1"]
                tip_raw = leader["p2"]
            else:
                start_raw = leader.get("label_endpoint", leader["p1"])
                tip_raw = leader.get("leader_to", leader["p2"])
            sx, sy = (float(start_raw[0]), float(start_raw[1]))
            tx, ty = (float(tip_raw[0]), float(tip_raw[1]))
            trim = 0.0 if leader.get("mask_full_segment", False) else float(
                cfg.get("leader_tip_trim", 0) or 0
            )
            length = math.hypot(tx - sx, ty - sy)
            if trim > 0 and length > trim:
                tx -= (tx - sx) / length * trim
                ty -= (ty - sy) / length * trim
            p1 = (int(round(sx)), int(round(sy)))
            p2 = (int(round(tx)), int(round(ty)))
            cv2.line(
                mask,
                p1,
                p2,
                color=255,
                thickness=int(cfg["leader_mask_thickness"]),
                lineType=cv2.LINE_AA,
            )
    dilate = int(cfg.get("mask_dilate", 0) or 0)
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def _detect_reference_pass(gray: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    ink, background = _ink_mask(gray)
    h, w = gray.shape
    comps = _component_candidates(ink, cfg)
    clusters = _group_text_clusters(comps, ink, cfg)
    segments = _hough_segments(ink, clusters, cfg)
    leadered_labels = _assign_leaders(clusters, segments, cfg)
    unleadered_labels = _select_unleadered_labels(
        clusters,
        leadered_labels,
        cfg,
        ink.shape,
    )
    existing_labels = [*leadered_labels, *unleadered_labels]
    figure_labels = [
        label
        for label in _figure_label_clusters(ink, cfg)
        if not any(
            _bbox_overlap_fraction(label["bbox"], existing["bbox"]) > 0.35
            for existing in existing_labels
        )
    ]
    labels = [*existing_labels, *figure_labels]
    mask = _build_removal_mask((h, w), labels, cfg) if labels else np.zeros((h, w), np.uint8)
    removed_ink = int(np.count_nonzero((mask > 0) & ink))
    total_ink = int(np.count_nonzero(ink))
    removed_ratio = removed_ink / max(1, total_ink)

    flagged = False
    reason = "ok"
    if len(labels) > int(cfg["max_reference_labels"]):
        flagged = True
        reason = f"too_many_reference_labels:{len(labels)}"
    if removed_ratio > float(cfg["max_removed_ink_ratio"]):
        flagged = True
        reason = f"removed_ink_ratio_too_high:{removed_ratio:.4f}"

    return {
        "ink": ink,
        "background": background,
        "labels": labels,
        "n_clusters": len(clusters),
        "n_leadered_labels": len(leadered_labels),
        "n_unleadered_labels": len(unleadered_labels),
        "n_figure_labels": len(figure_labels),
        "mask": mask,
        "removed_ink": removed_ink,
        "total_ink": total_ink,
        "removed_ratio": removed_ratio,
        "flagged": flagged,
        "reason": reason,
    }


def _text_only_labels(
    labels: list[dict[str, Any]],
    removal_mode: str,
) -> list[dict[str, Any]]:
    text_labels: list[dict[str, Any]] = []
    for label in labels:
        item = dict(label)
        item["leader_lines"] = []
        item["removal_mode"] = removal_mode
        item["kind"] = item.get("kind", "text_cluster")
        text_labels.append(item)
    return text_labels


def _apply_mask_to_raster(
    out: np.ndarray,
    det: dict[str, Any],
    mask: np.ndarray,
    cfg: dict[str, Any],
) -> int:
    background = int(det["background"])
    out[mask > 0] = background
    repair_kernel_size = int(cfg.get("repair_close_kernel", 0) or 0)
    if not bool(cfg.get("repair_after_removal", False)) or repair_kernel_size < 3:
        return 0
    if repair_kernel_size % 2 == 0:
        repair_kernel_size += 1

    out_ink = det["ink"].copy()
    out_ink[mask > 0] = False
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (repair_kernel_size, repair_kernel_size)
    )
    closed = cv2.morphologyEx(
        out_ink.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    ) > 0
    local = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1) > 0
    repaired = closed & (~out_ink) & local
    n_repaired = int(np.count_nonzero(repaired))
    if n_repaired:
        ink_value = 0 if background == 255 else 255
        out[repaired] = ink_value
    return n_repaired


def run(
    input_path: Path,
    output_dir: Path,
    sketch_id: str,
    config: Optional[dict[str, Any]] = None,
) -> Stage0Result:
    """
    Detect and remove patent reference numerals/leader lines.

    The reference-free raster has the same dimensions as the source image and is
    safe to feed into Stage 1. If the full leader mask is too aggressive, a
    bounded text-only fallback can still remove labels while flagging the JSON.
    """
    t_start = time.perf_counter()
    cfg = _stage0_cfg(config)
    ref_dir = output_dir / "references"
    crop_dir = ref_dir / "crops"
    ref_dir.mkdir(parents=True, exist_ok=True)

    mask_path = ref_dir / f"{sketch_id}_references_mask.png"
    no_refs_path = ref_dir / f"{sketch_id}_norefs.png"
    json_path = ref_dir / f"{sketch_id}_references.json"

    gray = _read_gray(input_path)
    original_ink, _ = _ink_mask(gray)
    h, w = gray.shape
    total_original_ink = int(np.count_nonzero(original_ink))
    max_iterations = max(1, int(cfg.get("max_iterations", 1) or 1))
    max_total_removed = float(cfg.get("max_total_removed_ink_ratio", 0) or 0)

    out = gray.copy()
    cumulative_mask = np.zeros((h, w), dtype=np.uint8)
    labels: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []
    flagged = False
    reason = "ok"
    active_removal = False
    repair_pixels = 0
    n_iterations_applied = 0

    for iteration in range(1, max_iterations + 1):
        det = _detect_reference_pass(out, cfg)
        pass_labels = det["labels"]
        pass_mask = det["mask"]
        pass_flagged = bool(det["flagged"])
        pass_reason = str(det["reason"])
        projected_mask = cv2.bitwise_or(cumulative_mask, pass_mask)
        projected_ratio = (
            int(np.count_nonzero((projected_mask > 0) & original_ink))
            / max(1, total_original_ink)
        )
        if max_total_removed > 0 and projected_ratio > max_total_removed:
            pass_flagged = True
            pass_reason = f"total_removed_ink_ratio_too_high:{projected_ratio:.4f}"

        iteration_summaries.append({
            "iteration": iteration,
            "n_labels": len(pass_labels),
            "n_leadered_labels": int(det.get("n_leadered_labels", 0)),
            "n_unleadered_labels": int(det.get("n_unleadered_labels", 0)),
            "n_figure_labels": int(det.get("n_figure_labels", 0)),
            "n_leaders": sum(len(label.get("leader_lines", [])) for label in pass_labels),
            "removed_ink_ratio": float(det["removed_ratio"]),
            "projected_total_removed_ink_ratio": float(projected_ratio),
            "active": bool(pass_labels) and not pass_flagged,
            "flagged": pass_flagged,
            "reason": pass_reason,
        })

        if not pass_labels:
            reason = "converged" if active_removal else "no_references_detected"
            break
        if pass_flagged:
            fallback_applied = False
            if bool(cfg.get("text_only_on_guard", False)):
                text_labels = _text_only_labels(
                    pass_labels,
                    "text_only_guard_fallback",
                )
                text_mask = _build_removal_mask(
                    (h, w),
                    text_labels,
                    cfg,
                    include_leaders=False,
                )
                projected_text_mask = cv2.bitwise_or(cumulative_mask, text_mask)
                projected_text_ratio = (
                    int(np.count_nonzero((projected_text_mask > 0) & original_ink))
                    / max(1, total_original_ink)
                )
                max_text_ratio = float(
                    cfg.get("max_guard_text_only_removed_ink_ratio", 0) or 0
                )
                max_text_labels = int(cfg.get("max_guard_text_only_labels", 0) or 0)
                text_ratio_ok = max_text_ratio <= 0 or projected_text_ratio <= max_text_ratio
                text_count_ok = max_text_labels <= 0 or len(text_labels) <= max_text_labels
                if text_ratio_ok and text_count_ok:
                    flagged = True
                    active_removal = True
                    n_iterations_applied += 1
                    reason = f"{pass_reason};text_only_guard_fallback"
                    for label in text_labels:
                        label["iteration"] = iteration
                    labels.extend(text_labels)
                    cumulative_mask = projected_text_mask
                    repair_pixels += _apply_mask_to_raster(out, det, text_mask, cfg)
                    iteration_summaries[-1]["active"] = True
                    iteration_summaries[-1]["fallback"] = "text_only"
                    iteration_summaries[-1]["fallback_total_removed_ink_ratio"] = float(
                        projected_text_ratio
                    )
                    fallback_applied = True

            if fallback_applied:
                break
            flagged = True
            reason = pass_reason
            if not active_removal:
                labels = pass_labels
                cumulative_mask = pass_mask
            break

        active_removal = True
        n_iterations_applied += 1
        reason = "ok"
        for label in pass_labels:
            label["iteration"] = iteration
        labels.extend(pass_labels)
        cumulative_mask = projected_mask
        repair_pixels += _apply_mask_to_raster(out, det, pass_mask, cfg)

    removed_ratio = (
        int(np.count_nonzero((cumulative_mask > 0) & original_ink))
        / max(1, total_original_ink)
    )

    cv2.imwrite(str(mask_path), cumulative_mask)
    cv2.imwrite(str(no_refs_path), out)

    doc = _build_reference_doc(
        sketch_id=sketch_id,
        source_path=input_path,
        gray=gray,
        ink=original_ink,
        labels=labels,
        mask_path=mask_path,
        reference_free_path=no_refs_path,
        crop_dir=crop_dir,
        cfg=cfg,
        active_removal=active_removal,
        removed_ink_ratio=removed_ratio,
        repair_pixels=repair_pixels,
        n_iterations=n_iterations_applied,
        iteration_summaries=iteration_summaries,
        flagged=flagged,
        reason=reason,
    )
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)

    return Stage0Result(
        sketch_id=sketch_id,
        reference_free_path=no_refs_path,
        references_json_path=json_path,
        mask_path=mask_path,
        n_labels=int(doc["n_labels"]),
        n_leaders=int(doc["n_leaders"]),
        n_iterations=n_iterations_applied,
        removed_ink_ratio=float(removed_ratio),
        repair_pixels=repair_pixels,
        active_removal=active_removal,
        flagged=flagged,
        processing_time_s=time.perf_counter() - t_start,
    )


def annotations_from_reference_json(references_json_path: Path) -> list[dict[str, Any]]:
    """Convert Stage 0 JSON references to Stage 4 annotation records."""
    with open(references_json_path) as fh:
        doc = json.load(fh)
    if not doc.get("active_removal", False):
        return []

    annotations: list[dict[str, Any]] = []
    for ref in doc.get("reference_labels", []):
        leaders = [
            leader
            for leader in ref.get("leader_lines", [])
            if leader.get("removed", True)
        ]
        ann = {
            "id": ref.get("id", ""),
            "text": ref.get("text", ""),
            "position": ref.get("position", [0, 0]),
            "bbox": ref.get("bbox"),
            "crop_bbox": ref.get("crop_bbox"),
            "image_path": ref.get("crop_path"),
            "kind": ref.get("kind", ""),
            "removal_mode": ref.get("removal_mode", ""),
            "leader_lines": [
                {
                    "p1": leader.get("p1"),
                    "p2": leader.get("p2"),
                    "leader_to": leader.get("leader_to"),
                }
                for leader in leaders
            ],
            "source": "stage0_references",
        }
        if leaders:
            ann["leader_to"] = leaders[0].get("leader_to")
        annotations.append(ann)
    return annotations


def attach_references_to_primitives(
    primitives_path: Path,
    references_json_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Copy Stage 3 primitives JSON and append Stage 0 annotations for Stage 4.
    Returns the JSON path to pass into Stage 4.
    """
    with open(primitives_path) as fh:
        prim_doc = json.load(fh)
    with open(references_json_path) as fh:
        ref_doc = json.load(fh)

    annotations = annotations_from_reference_json(references_json_path)
    existing = prim_doc.get("annotations") or []
    prim_doc["annotations"] = [*existing, *annotations]
    prim_doc["reference_extraction"] = {
        "schema": ref_doc.get("schema"),
        "json_path": str(references_json_path),
        "active_removal": bool(ref_doc.get("active_removal", False)),
        "n_labels": int(ref_doc.get("n_labels", 0)),
        "n_leaders": int(ref_doc.get("n_leaders", 0)),
        "removed_ink_ratio": float(ref_doc.get("removed_ink_ratio", 0.0)),
        "repair_pixels": int(ref_doc.get("repair_pixels", 0)),
        "flagged": bool(ref_doc.get("flagged", False)),
        "reason": ref_doc.get("reason", ""),
    }

    out_path = output_path or (
        primitives_path.parent / f"{primitives_path.stem}_with_refs.json"
    )
    with open(out_path, "w") as fh:
        json.dump(prim_doc, fh, indent=2)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 0 - detect/remove patent reference numerals and leaders."
    )
    parser.add_argument("input", type=Path, help="Input TIF/PNG drawing")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    if args.config.exists():
        import yaml
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}

    sid = args.id or args.input.stem
    result = run(args.input, args.output, sid, cfg)
    print(f"Reference-free raster : {result.reference_free_path}")
    print(f"Reference JSON        : {result.references_json_path}")
    print(f"Labels/leaders        : {result.n_labels}/{result.n_leaders}")
    print(f"Iterations applied    : {result.n_iterations}")
    print(f"Active removal        : {result.active_removal}")
    print(f"Removed ink ratio     : {result.removed_ink_ratio:.4f}")
    print(f"Repair pixels         : {result.repair_pixels}")
    print(f"Flagged               : {result.flagged}")
    print(f"Processing time       : {result.processing_time_s:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
