"""Multilabel stroke targets for safe structural/hachure decomposition."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np
from skimage.morphology import skeletonize


HATCH_STROKE_LABEL_CONTRACT = "reference-free-hatch-stroke-multilabel-v1"
REAL_HATCH_STROKE_LABEL_CONTRACT = "reviewed-patent-hatch-stroke-multilabel-v1"
SUPPORTED_HATCH_STROKE_LABEL_CONTRACTS = (
    HATCH_STROKE_LABEL_CONTRACT,
    REAL_HATCH_STROKE_LABEL_CONTRACT,
)
SUPERVISION_KEY = "supervision_mask"
STRUCTURAL_MASKS = ("object", "hidden_center")
HATCH_MASK = "hatch"
TARGET_KEYS = (
    "input_skeleton",
    "structural_target",
    "hatch_target",
    "overlap_target",
)


def _semantic_mask(
    masks: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    missing = sorted(set(names) - set(masks))
    if missing:
        raise ValueError(f"missing hatch-stroke semantic masks: {missing}")

    combined: np.ndarray | None = None
    for name in names:
        mask = np.asarray(masks[name])
        if mask.ndim != 2:
            raise ValueError(f"mask {name!r} must be 2-D, got shape {mask.shape}")
        if shape is not None and mask.shape != shape:
            raise ValueError(
                f"hatch-stroke mask {name!r} shape {mask.shape} != {shape}"
            )
        if combined is None:
            combined = np.zeros(mask.shape, dtype=bool)
            shape = mask.shape
        elif mask.shape != combined.shape:
            raise ValueError(
                f"hatch-stroke mask {name!r} shape {mask.shape} "
                f"!= {combined.shape}"
            )
        combined |= mask > 0
    assert combined is not None
    return combined


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask
    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def build_hatch_stroke_targets(
    masks: Mapping[str, np.ndarray],
    *,
    support_radius: int = 0,
) -> dict[str, np.ndarray]:
    """Build lossless structural/hachure labels on a reference-free skeleton.

    The labels are intentionally multilabel. A skeleton pixel supported by both
    structural ink and hachure ink remains positive in both channels, which is
    the key behavior needed at crossings. Any numerically unassigned skeleton
    pixel falls back to the structural channel so training data never teaches
    the model to delete unexplained geometry.
    """
    if not isinstance(support_radius, (int, np.integer)) or support_radius < 0:
        raise ValueError("support_radius must be a non-negative integer")
    support_radius = int(support_radius)

    structural_raster = _semantic_mask(masks, STRUCTURAL_MASKS)
    hatch_raster = _semantic_mask(masks, (HATCH_MASK,), structural_raster.shape)
    input_skeleton = skeletonize(structural_raster | hatch_raster)

    structural_support = _dilate(structural_raster, support_radius)
    hatch_support = _dilate(hatch_raster, support_radius)
    structural_target = input_skeleton & structural_support
    hatch_target = input_skeleton & hatch_support

    # Skeletonization can move a centreline by a pixel in thick, merged regions.
    # Preserve those rare pixels rather than creating an implicit deletion label.
    unassigned = input_skeleton & ~(structural_target | hatch_target)
    structural_target |= unassigned
    overlap_target = structural_target & hatch_target

    targets = {
        "input_skeleton": input_skeleton.astype(np.uint8) * 255,
        "structural_target": structural_target.astype(np.uint8) * 255,
        "hatch_target": hatch_target.astype(np.uint8) * 255,
        "overlap_target": overlap_target.astype(np.uint8) * 255,
    }
    validate_hatch_stroke_targets(targets)
    return targets


def validate_hatch_stroke_targets(
    targets: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """Validate the lossless multilabel contract and return pixel statistics."""
    missing = sorted(set(TARGET_KEYS) - set(targets))
    if missing:
        raise ValueError(f"hatch-stroke targets missing arrays: {missing}")

    arrays: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for name in TARGET_KEYS:
        array = np.asarray(targets[name])
        if array.ndim != 2:
            raise ValueError(f"target {name!r} must be 2-D, got shape {array.shape}")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"target {name!r} shape {array.shape} != {shape}")
        arrays[name] = array > 0

    skeleton = arrays["input_skeleton"]
    structural = arrays["structural_target"]
    hatch = arrays["hatch_target"]
    overlap = arrays["overlap_target"]
    if np.any(structural & ~skeleton):
        raise ValueError("structural target contains pixels outside input skeleton")
    if np.any(hatch & ~skeleton):
        raise ValueError("hatch target contains pixels outside input skeleton")
    if not np.array_equal(structural | hatch, skeleton):
        raise ValueError("structural and hatch targets do not cover input skeleton")
    if not np.array_equal(structural & hatch, overlap):
        raise ValueError("overlap target is inconsistent with channel intersection")

    return {
        "skeleton_pixels": int(np.count_nonzero(skeleton)),
        "structural_pixels": int(np.count_nonzero(structural)),
        "hatch_pixels": int(np.count_nonzero(hatch)),
        "overlap_pixels": int(np.count_nonzero(overlap)),
        "structural_only_pixels": int(np.count_nonzero(structural & ~hatch)),
        "hatch_only_pixels": int(np.count_nonzero(hatch & ~structural)),
        "unassigned_pixels": int(
            np.count_nonzero(skeleton & ~(structural | hatch))
        ),
    }


def validate_hatch_stroke_supervision(
    targets: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """Validate optional per-channel supervision for partially labeled ink."""
    stats = validate_hatch_stroke_targets(targets)
    skeleton = np.asarray(targets["input_skeleton"]) > 0
    if SUPERVISION_KEY not in targets:
        supervision = np.broadcast_to(skeleton, (2, *skeleton.shape))
    else:
        supervision = np.asarray(targets[SUPERVISION_KEY]) > 0
        if supervision.shape != (2, *skeleton.shape):
            raise ValueError(
                "supervision mask must have shape "
                f"(2, {skeleton.shape[0]}, {skeleton.shape[1]})"
            )
        if np.any(supervision & ~skeleton[None]):
            raise ValueError("supervision mask contains pixels outside input skeleton")
    stats.update({
        "structural_supervised_pixels": int(np.count_nonzero(supervision[0])),
        "hatch_supervised_pixels": int(np.count_nonzero(supervision[1])),
        "jointly_supervised_pixels": int(
            np.count_nonzero(supervision[0] & supervision[1])
        ),
        "unsupervised_skeleton_pixels": int(
            np.count_nonzero(skeleton & ~(supervision[0] | supervision[1]))
        ),
    })
    return stats
