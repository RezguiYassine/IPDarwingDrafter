"""Stage-2 keypoint targets aligned with the production topology tracer."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence

import cv2
import numpy as np
from skimage.morphology import skeletonize


ARCHIVE_LABEL_CONTRACT = "archive"
SUPPORTED_RASTER_TOPOLOGY_CONTRACT = "supported-raster-topology-v1"
REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT = "reference-free-raster-topology-v1"
HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT = "hatch-suppressed-raster-topology-v1"
STAGE2_LABEL_CONTRACTS = (
    ARCHIVE_LABEL_CONTRACT,
    SUPPORTED_RASTER_TOPOLOGY_CONTRACT,
    REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT,
    HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
)
DEFAULT_TOPOLOGY_MASKS = ("object", "hidden_center", "hatch")
STRUCTURAL_TOPOLOGY_MASKS = ("object", "hidden_center")


def crossing_number_map(binary: np.ndarray) -> np.ndarray:
    """Return the same 8-neighbour crossing number used by production Stage 2."""
    if binary.ndim != 2:
        raise ValueError(f"expected a 2-D skeleton, got shape {binary.shape}")
    foreground = (binary > 0).astype(np.int16)
    height, width = foreground.shape
    crossing_number = np.zeros((height, width), dtype=np.int16)
    if height < 3 or width < 3:
        return crossing_number
    ring = [
        foreground[:-2, 1:-1],
        foreground[:-2, 2:],
        foreground[1:-1, 2:],
        foreground[2:, 2:],
        foreground[2:, 1:-1],
        foreground[2:, :-2],
        foreground[1:-1, :-2],
        foreground[:-2, :-2],
    ]
    crossing_number[1:-1, 1:-1] = sum(
        np.abs(ring[index] - ring[(index + 1) % 8])
        for index in range(8)
    ) // 2
    crossing_number[foreground == 0] = 0
    return crossing_number


def _support_mask(
    masks: Mapping[str, np.ndarray],
    names: Sequence[str],
    shape: tuple[int, int],
    radius: int,
) -> np.ndarray:
    missing = sorted(set(names) - set(masks))
    if missing:
        raise ValueError(f"missing Stage-2 semantic masks: {missing}")
    support = np.zeros(shape, dtype=np.uint8)
    for name in names:
        mask = np.asarray(masks[name])
        if mask.shape != shape:
            raise ValueError(
                f"Stage-2 mask {name!r} shape {mask.shape} != skeleton {shape}"
            )
        support |= (mask > 0).astype(np.uint8)
    if radius > 0:
        size = radius * 2 + 1
        support = cv2.dilate(support, np.ones((size, size), dtype=np.uint8))
    return support > 0


def _supported_cluster_centres(
    candidate_mask: np.ndarray,
    support: np.ndarray,
    kind: int,
) -> list[tuple[int, int, int]]:
    count, labels, _stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return []
    ys, xs = np.where(labels > 0)
    if not len(xs):
        return []
    supported_labels = np.unique(labels[ys[support[ys, xs]], xs[support[ys, xs]]])
    points = []
    for label in supported_labels.tolist():
        if label == 0:
            continue
        x, y = centroids[label]
        # Production _cn_keypoint_clusters uses int(mean), not rounding.
        points.append((int(x), int(y), kind))
    return points


def reference_free_skeleton(
    masks: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    support_masks: Sequence[str] = DEFAULT_TOPOLOGY_MASKS,
) -> np.ndarray:
    """Rebuild the Stage-2 input from semantics that survive Stage 0."""
    foreground = _support_mask(masks, support_masks, shape, radius=0)
    return skeletonize(foreground).astype(np.uint8) * 255


def _snap_corners(
    corners: np.ndarray,
    binary: np.ndarray,
    radius: float,
) -> tuple[list[tuple[int, int, int]], int]:
    ys, xs = np.where(binary > 0)
    if not len(xs):
        return [], int(len(corners))
    snapped = []
    dropped = 0
    maximum_squared = float(radius) ** 2
    for x, y, _kind in corners.tolist():
        distances = (xs - int(x)) ** 2 + (ys - int(y)) ** 2
        index = int(np.argmin(distances))
        if float(distances[index]) > maximum_squared:
            dropped += 1
            continue
        snapped.append((int(xs[index]), int(ys[index]), 2))
    return sorted(set(snapped)), dropped


def supported_raster_topology_keypoints(
    skeleton: np.ndarray,
    source_keypoints: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    support_masks: Sequence[str] = DEFAULT_TOPOLOGY_MASKS,
    support_radius: int = 2,
    corner_min_distance: float = 5.0,
    corner_snap_radius: float = 16.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build CN endpoint/junction labels on useful semantics and retain corners.

    Hatches and dashed object lines need their raster endpoints and crossings to
    be supervised because those pixels seed the pure-CNN topology walk. Text,
    numerals, leaders, and dimensions remain visible negative distractors unless
    they overlap one of the explicitly supported semantic masks.
    """
    skeleton = np.asarray(skeleton)
    source_keypoints = np.asarray(source_keypoints, dtype=np.int32)
    if source_keypoints.ndim != 2 or source_keypoints.shape[1:] != (3,):
        raise ValueError(f"invalid source keypoint shape {source_keypoints.shape}")
    if support_radius < 0:
        raise ValueError("support_radius must be non-negative")
    if corner_min_distance < 0:
        raise ValueError("corner_min_distance must be non-negative")
    if corner_snap_radius < 0:
        raise ValueError("corner_snap_radius must be non-negative")

    binary = (skeleton > 0).astype(np.uint8)
    support = _support_mask(masks, support_masks, binary.shape, support_radius)
    crossing_number = crossing_number_map(binary)
    topology = _supported_cluster_centres(
        (crossing_number == 1) & (binary > 0), support, 0
    )
    topology.extend(
        _supported_cluster_centres(
            (crossing_number >= 3) & (binary > 0), support, 1
        )
    )

    corners = source_keypoints[source_keypoints[:, 2] == 2]
    snapped_corners, dropped_corners = _snap_corners(
        corners, binary, corner_snap_radius
    )
    retained_corners: list[tuple[int, int, int]] = []
    base_xy = np.asarray([point[:2] for point in topology], dtype=np.float64)
    minimum_squared = float(corner_min_distance) ** 2
    for x, y, _kind in snapped_corners:
        if len(base_xy):
            distances = np.sum((base_xy - np.asarray([x, y])) ** 2, axis=1)
            if float(np.min(distances)) <= minimum_squared:
                continue
        retained_corners.append((int(x), int(y), 2))

    points = np.asarray(
        sorted(set(topology + retained_corners)), dtype=np.int32
    ).reshape(-1, 3)
    counts = np.bincount(points[:, 2], minlength=3) if len(points) else np.zeros(3)
    stats = {
        "endpoint": int(counts[0]),
        "junction": int(counts[1]),
        "corner": int(counts[2]),
        "total": int(len(points)),
        "source_corner": int(len(corners)),
        "dropped_corner": int(dropped_corners),
        "suppressed_corner": int(len(snapped_corners) - len(retained_corners)),
    }
    return points, stats


def relabel_puhachov_payload(
    puhachov_payload: bytes,
    masks_payload: bytes,
    *,
    support_masks: Sequence[str] = DEFAULT_TOPOLOGY_MASKS,
    support_radius: int = 2,
    corner_min_distance: float = 5.0,
    corner_snap_radius: float = 16.0,
    label_contract: str = SUPPORTED_RASTER_TOPOLOGY_CONTRACT,
) -> bytes:
    """Relabel one archived Puhachov NPZ without regenerating its drawing."""
    with np.load(io.BytesIO(puhachov_payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(io.BytesIO(masks_payload), allow_pickle=False) as archive:
        masks = {name: np.asarray(archive[name]) for name in archive.files}
    missing = {"skeleton", "kps", "meta"} - set(arrays)
    if missing:
        raise ValueError(f"Puhachov payload missing {sorted(missing)}")

    if label_contract not in {
        SUPPORTED_RASTER_TOPOLOGY_CONTRACT,
        REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT,
        HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
    }:
        raise ValueError(f"cannot relabel with contract {label_contract!r}")
    source_skeleton = np.asarray(arrays["skeleton"])
    effective_support_radius = int(support_radius)
    reference_free_contract = label_contract in {
        REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT,
        HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
    }
    if reference_free_contract:
        arrays["skeleton"] = reference_free_skeleton(
            masks, source_skeleton.shape, support_masks
        )
        effective_support_radius = 0

    topology_skeleton = arrays["skeleton"]
    topology_support_masks = tuple(support_masks)
    if label_contract == HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT:
        topology_support_masks = tuple(
            name for name in STRUCTURAL_TOPOLOGY_MASKS if name in support_masks
        )
        if not topology_support_masks:
            raise ValueError("hatch-suppressed topology needs structural masks")
        topology_skeleton = reference_free_skeleton(
            masks, source_skeleton.shape, topology_support_masks
        )
        arrays["topology_skeleton"] = topology_skeleton

    keypoints, stats = supported_raster_topology_keypoints(
        topology_skeleton,
        arrays["kps"],
        masks,
        support_masks=topology_support_masks,
        support_radius=effective_support_radius,
        corner_min_distance=corner_min_distance,
        corner_snap_radius=corner_snap_radius,
    )
    metadata = json.loads(str(np.asarray(arrays["meta"]).item()))
    original_contract = metadata.get(
        "source_label_contract", metadata.get("label_contract", "exact-semantic-v1")
    )
    metadata.update(
        {
            "label_contract": label_contract,
            "source_label_contract": original_contract,
            "input_support_masks": list(support_masks),
            "topology_support_masks": list(topology_support_masks),
            "topology_support_radius_px": effective_support_radius,
            "corner_min_distance_px": float(corner_min_distance),
            "corner_snap_radius_px": float(corner_snap_radius),
            "reference_free_input": reference_free_contract,
            "source_skeleton_foreground_px": int(np.count_nonzero(source_skeleton)),
            "skeleton_foreground_px": int(np.count_nonzero(arrays["skeleton"])),
            "topology_skeleton_foreground_px": int(
                np.count_nonzero(topology_skeleton)
            ),
            "keypoint_counts": stats,
        }
    )
    if reference_free_contract:
        metadata["removed_distractor_semantics"] = metadata.get(
            "distractor_semantics_in_skeleton", []
        )
        metadata["distractor_semantics_in_skeleton"] = []
    if label_contract == HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT:
        metadata["negative_topology_semantics"] = ["hatch"]
        metadata["hachure_side_layer_only"] = True
    arrays["kps"] = keypoints
    arrays["meta"] = np.asarray(json.dumps(metadata, sort_keys=True))
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()
