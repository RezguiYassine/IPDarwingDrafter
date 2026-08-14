"""Train lossless structural/hachure decomposition on Stage-1 skeletons."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from syntheticData.patentvec.hatch_stroke_targets import (
    HATCH_STROKE_LABEL_CONTRACT,
    SUPPORTED_HATCH_STROKE_LABEL_CONTRACTS,
    SUPERVISION_KEY,
    TARGET_KEYS,
    validate_hatch_stroke_supervision,
)
from tools.hatch_model import HatchUNet


CHANNEL_NAMES = ("structural", "hatch")
DIFFICULTY_NAMES = ("medium", "hard", "very_hard")


def parse_difficulty_weights(value: str) -> dict[str, float]:
    """Parse ``medium=0.2,hard=0.3,very_hard=0.5`` CLI weights."""
    if not value.strip():
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        try:
            name, raw_weight = item.split("=", 1)
            name = name.strip()
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "difficulty weights must use name=value pairs"
            ) from exc
        if name not in DIFFICULTY_NAMES:
            raise argparse.ArgumentTypeError(f"unknown difficulty: {name}")
        if weight <= 0:
            raise argparse.ArgumentTypeError("difficulty weights must be positive")
        if name in weights:
            raise argparse.ArgumentTypeError(f"duplicate difficulty: {name}")
        weights[name] = weight
    return weights


def allocate_difficulty_counts(
    total: int, weights: dict[str, float]
) -> dict[str, int]:
    if total < 1 or not weights:
        raise ValueError("total and difficulty weights must be non-empty")
    weight_sum = sum(weights.values())
    exact = {name: total * weight / weight_sum for name, weight in weights.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    priority = sorted(
        weights,
        key=lambda name: (exact[name] - counts[name], name),
        reverse=True,
    )
    for name in priority[:remaining]:
        counts[name] += 1
    return counts


def _load_sample(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in TARGET_KEYS}
        if SUPERVISION_KEY in data.files:
            arrays[SUPERVISION_KEY] = np.asarray(data[SUPERVISION_KEY])
        metadata = json.loads(str(np.asarray(data["meta"])))
    validate_hatch_stroke_supervision(arrays)
    if metadata.get("label_contract") not in SUPPORTED_HATCH_STROKE_LABEL_CONTRACTS:
        raise ValueError(f"{path}: incompatible hatch-stroke label contract")
    skeleton = (arrays["input_skeleton"] > 0).astype(np.float32)
    targets = np.stack(
        [arrays["structural_target"] > 0, arrays["hatch_target"] > 0]
    ).astype(np.float32)
    supervision = arrays.get(SUPERVISION_KEY)
    if supervision is None:
        supervision = np.broadcast_to(
            skeleton > 0, (2, *skeleton.shape)
        ).copy()
    supervision = (np.asarray(supervision) > 0).astype(np.float32)
    return skeleton, targets, supervision, metadata


def _crop_with_padding(
    array: np.ndarray,
    top: int,
    left: int,
    size: int,
) -> np.ndarray:
    source = array[..., top : top + size, left : left + size]
    output = np.zeros((*array.shape[:-2], size, size), dtype=array.dtype)
    output[..., : source.shape[-2], : source.shape[-1]] = source
    return output


class HatchStrokePatchDataset(Dataset):
    """Deterministic epoch-wise random crops with hatch-biased sampling."""

    def __init__(
        self,
        paths: list[Path],
        *,
        real_paths: list[Path] | None = None,
        real_fraction: float = 0.0,
        patch_size: int = 512,
        samples_per_epoch: int | None = None,
        hatch_bias: float = 0.75,
        difficulty_weights: dict[str, float] | None = None,
        augment: bool = True,
        seed: int = 42,
    ):
        if not paths:
            raise ValueError("hatch-stroke dataset is empty")
        if patch_size < 32 or patch_size % 32:
            raise ValueError("patch_size must be a multiple of 32")
        if not 0.0 <= hatch_bias <= 1.0:
            raise ValueError("hatch_bias must be between zero and one")
        if not 0.0 <= real_fraction < 1.0:
            raise ValueError("real_fraction must be in [0, 1)")
        if real_fraction > 0 and not real_paths:
            raise ValueError("real_paths are required when real_fraction is positive")
        self.synthetic_paths = sorted(paths)
        self.real_paths = sorted(real_paths or [])
        self.paths = self.synthetic_paths + self.real_paths
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch or len(self.synthetic_paths))
        self.hatch_bias = float(hatch_bias)
        self.real_fraction = float(real_fraction)
        self.difficulty_weights = dict(difficulty_weights or {})
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.difficulty_groups: dict[str, list[Path]] = {}
        self.difficulty_schedule: tuple[str, ...] = ()
        self.source_schedule: tuple[str, ...] = ()
        self.source_ordinals: tuple[int, ...] = ()
        real_count = int(round(self.samples_per_epoch * self.real_fraction))
        synthetic_count = self.samples_per_epoch - real_count
        if synthetic_count < 1:
            raise ValueError("real_fraction leaves no synthetic samples in the epoch")
        if self.real_paths and real_count:
            source_schedule = ["real"] * real_count + ["synthetic"] * synthetic_count
            source_rng = random.Random(self.seed)
            source_rng.shuffle(source_schedule)
            ordinals = []
            seen = Counter()
            for source in source_schedule:
                ordinals.append(seen[source])
                seen[source] += 1
            self.source_schedule = tuple(source_schedule)
            self.source_ordinals = tuple(ordinals)
        self.source_counts = {
            "synthetic": synthetic_count,
            "real": real_count,
        }
        if self.difficulty_weights:
            for name in self.difficulty_weights:
                suffix = f"_{name}"
                group = [
                    path for path in self.synthetic_paths
                    if path.stem.endswith(suffix)
                ]
                if not group:
                    raise ValueError(f"no training paths for difficulty {name!r}")
                self.difficulty_groups[name] = group
            counts = allocate_difficulty_counts(
                synthetic_count, self.difficulty_weights
            )
            self.difficulty_schedule = tuple(
                name
                for name in sorted(counts)
                for _ in range(counts[name])
            )
            if len(self.difficulty_schedule) != synthetic_count:
                raise AssertionError("difficulty allocation does not cover the epoch")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )
        source = self.source_schedule[int(index)] if self.source_schedule else "synthetic"
        source_index = (
            self.source_ordinals[int(index)] if self.source_ordinals else int(index)
        )
        if source == "real":
            path = self.real_paths[int(rng.integers(len(self.real_paths)))]
        elif self.difficulty_schedule:
            difficulty = self.difficulty_schedule[source_index]
            group = self.difficulty_groups[difficulty]
            path_index = int(rng.integers(len(group)))
            path = group[path_index]
        elif self.source_counts["synthetic"] <= len(self.synthetic_paths):
            path_index = source_index % len(self.synthetic_paths)
            path = self.synthetic_paths[path_index]
        else:
            path_index = int(rng.integers(len(self.synthetic_paths)))
            path = self.synthetic_paths[path_index]
        skeleton, targets, supervision, metadata = _load_sample(path)
        height, width = skeleton.shape
        maximum_top = max(0, height - self.patch_size)
        maximum_left = max(0, width - self.patch_size)

        hatch_points = np.argwhere((targets[1] > 0) & (supervision[1] > 0))
        use_hatch = len(hatch_points) and rng.random() < self.hatch_bias
        if use_hatch:
            y, x = hatch_points[int(rng.integers(len(hatch_points)))]
            anchor_low = self.patch_size // 4
            anchor_high = max(anchor_low + 1, 3 * self.patch_size // 4)
            top = int(np.clip(y - rng.integers(anchor_low, anchor_high), 0, maximum_top))
            left = int(np.clip(x - rng.integers(anchor_low, anchor_high), 0, maximum_left))
        elif metadata.get("domain") == "real_patent_reviewed":
            supervised_points = np.argwhere(np.any(supervision > 0, axis=0))
            if len(supervised_points):
                y, x = supervised_points[int(rng.integers(len(supervised_points)))]
                anchor_low = self.patch_size // 4
                anchor_high = max(anchor_low + 1, 3 * self.patch_size // 4)
                top = int(np.clip(
                    y - rng.integers(anchor_low, anchor_high), 0, maximum_top
                ))
                left = int(np.clip(
                    x - rng.integers(anchor_low, anchor_high), 0, maximum_left
                ))
            else:
                top = int(rng.integers(maximum_top + 1))
                left = int(rng.integers(maximum_left + 1))
        else:
            top = int(rng.integers(maximum_top + 1))
            left = int(rng.integers(maximum_left + 1))

        skeleton = _crop_with_padding(skeleton, top, left, self.patch_size)
        targets = _crop_with_padding(targets, top, left, self.patch_size)
        supervision = _crop_with_padding(
            supervision, top, left, self.patch_size
        )
        stacked = np.concatenate(
            [skeleton[None], targets, supervision], axis=0
        )
        if self.augment:
            stacked = np.rot90(stacked, k=int(rng.integers(4)), axes=(-2, -1))
            if rng.random() < 0.5:
                stacked = stacked[..., ::-1]
            if rng.random() < 0.5:
                stacked = stacked[..., ::-1, :]
        stacked = np.ascontiguousarray(stacked, dtype=np.float32)
        return (
            torch.from_numpy(stacked[:1]),
            torch.from_numpy(stacked[1:3]),
            torch.from_numpy(stacked[3:5]),
        )


def masked_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    input_mask: torch.Tensor,
    *,
    supervision_mask: torch.Tensor | None = None,
    structural_weight: float = 2.0,
    structural_negative_weight: float = 1.0,
    hatch_weight: float = 1.0,
    crossing_weight: float = 4.0,
    bce_fraction: float = 0.6,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """BCE plus Dice on input ink, with extra protection at crossings."""
    if logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits and targets must have shape (B, 2, H, W)")
    if input_mask.shape != logits[:, :1].shape:
        raise ValueError("input_mask must have shape (B, 1, H, W)")
    if (
        structural_weight <= 0
        or structural_negative_weight <= 0
        or hatch_weight <= 0
        or crossing_weight < 1
    ):
        raise ValueError("loss weights must be positive and crossing_weight >= 1")
    if not 0.0 <= bce_fraction <= 1.0:
        raise ValueError("bce_fraction must be between zero and one")

    mask = (input_mask > 0).to(logits.dtype).expand_as(logits)
    if supervision_mask is not None:
        if supervision_mask.shape != logits.shape:
            raise ValueError("supervision_mask must match logits shape")
        mask = mask * (supervision_mask > 0).to(logits.dtype)
    crossing = (targets[:, :1] * targets[:, 1:2] > 0).to(logits.dtype)
    pixel_weights = mask * (1.0 + (crossing_weight - 1.0) * crossing)
    channel_weights = logits.new_tensor(
        [structural_weight, hatch_weight]
    ).view(1, 2, 1, 1)
    label_weights = torch.ones_like(targets)
    label_weights[:, :1] = torch.where(
        targets[:, :1] > 0,
        label_weights[:, :1],
        logits.new_tensor(structural_negative_weight),
    )

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weighted = bce * pixel_weights * channel_weights * label_weights
    bce_denominator = (
        pixel_weights * channel_weights * label_weights
    ).sum().clamp_min(1.0)
    bce_loss = weighted.sum() / bce_denominator

    probs = torch.sigmoid(logits) * mask
    masked_targets = targets * mask
    intersection = (probs * masked_targets).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + masked_targets.sum(dim=(0, 2, 3))
    dice_per_channel = 1.0 - (2.0 * intersection + epsilon) / (
        denominator + epsilon
    )
    flat_weights = channel_weights.flatten()
    dice_loss = (dice_per_channel * flat_weights).sum() / flat_weights.sum()
    return bce_fraction * bce_loss + (1.0 - bce_fraction) * dice_loss


def _safe_divide(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty)


class StrokeMetricAccumulator:
    def __init__(
        self,
        *,
        channel_threshold: float = 0.5,
        hatch_high_threshold: float = 0.8,
        structural_low_threshold: float = 0.2,
    ):
        self.channel_threshold = float(channel_threshold)
        self.hatch_high_threshold = float(hatch_high_threshold)
        self.structural_low_threshold = float(structural_low_threshold)
        self.counts = Counter()

    def update(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        input_skeleton: np.ndarray,
        supervision_mask: np.ndarray | None = None,
    ) -> None:
        probabilities = np.asarray(probabilities)
        targets = np.asarray(targets) > 0
        skeleton = np.asarray(input_skeleton) > 0
        if probabilities.shape != targets.shape or probabilities.shape[0] != 2:
            raise ValueError("probabilities and targets must have shape (2, H, W)")
        if skeleton.shape != probabilities.shape[1:]:
            raise ValueError("input skeleton shape disagrees with probabilities")
        if supervision_mask is None:
            supervision = np.broadcast_to(skeleton, probabilities.shape)
        else:
            supervision = np.asarray(supervision_mask) > 0
            if supervision.shape != probabilities.shape:
                raise ValueError("supervision_mask shape disagrees with probabilities")
            supervision &= skeleton[None]

        predictions = probabilities >= self.channel_threshold
        for channel, name in enumerate(CHANNEL_NAMES):
            prediction = predictions[channel] & supervision[channel]
            truth = targets[channel] & supervision[channel]
            self.counts[f"{name}_tp"] += int(np.count_nonzero(prediction & truth))
            self.counts[f"{name}_pred"] += int(np.count_nonzero(prediction))
            self.counts[f"{name}_true"] += int(np.count_nonzero(truth))

        jointly_supervised = supervision[0] & supervision[1] & skeleton
        overlap_truth = targets[0] & targets[1] & jointly_supervised
        overlap_prediction = (
            predictions[0] & predictions[1] & jointly_supervised
        )
        self.counts["overlap_tp"] += int(
            np.count_nonzero(overlap_truth & overlap_prediction)
        )
        self.counts["overlap_true"] += int(np.count_nonzero(overlap_truth))

        safe_prediction = (
            (probabilities[1] >= self.hatch_high_threshold)
            & (probabilities[0] < self.structural_low_threshold)
            & jointly_supervised
        )
        safe_truth = targets[1] & ~targets[0] & jointly_supervised
        self.counts["safe_tp"] += int(np.count_nonzero(safe_prediction & safe_truth))
        self.counts["safe_pred"] += int(np.count_nonzero(safe_prediction))
        self.counts["safe_true"] += int(np.count_nonzero(safe_truth))
        self.counts["structural_removed"] += int(
            np.count_nonzero(safe_prediction & targets[0] & jointly_supervised)
        )
        self.counts["skeleton"] += int(np.count_nonzero(jointly_supervised))
        self.counts["structural_supervised"] += int(
            np.count_nonzero(supervision[0])
        )
        self.counts["hatch_supervised"] += int(
            np.count_nonzero(supervision[1])
        )

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name in CHANNEL_NAMES:
            precision = _safe_divide(
                self.counts[f"{name}_tp"], self.counts[f"{name}_pred"]
            )
            recall = _safe_divide(
                self.counts[f"{name}_tp"], self.counts[f"{name}_true"]
            )
            f1 = _safe_divide(2.0 * precision * recall, precision + recall)
            metrics[f"{name}_precision"] = precision
            metrics[f"{name}_recall"] = recall
            metrics[f"{name}_f1"] = f1

        safe_precision = _safe_divide(
            self.counts["safe_tp"], self.counts["safe_pred"]
        )
        safe_recall = _safe_divide(
            self.counts["safe_tp"], self.counts["safe_true"]
        )
        metrics.update(
            {
                "overlap_recall": _safe_divide(
                    self.counts["overlap_tp"], self.counts["overlap_true"]
                ),
                "safe_removal_precision": safe_precision,
                "safe_removal_recall": safe_recall,
                "safe_removal_f1": _safe_divide(
                    2.0 * safe_precision * safe_recall,
                    safe_precision + safe_recall,
                ),
                "safe_removal_structural_error_rate": _safe_divide(
                    self.counts["structural_removed"],
                    self.counts["structural_true"],
                ),
                "skeleton_pixels": float(self.counts["skeleton"]),
                "structural_supervised_pixels": float(
                    self.counts["structural_supervised"]
                ),
                "hatch_supervised_pixels": float(
                    self.counts["hatch_supervised"]
                ),
            }
        )
        return metrics


def checkpoint_rank(
    metrics: dict[str, float],
    minimum_structural_recall: float,
    real_metrics: dict[str, float] | None = None,
    maximum_structural_error_rate: float = float("inf"),
) -> tuple[float, ...]:
    """Rank useful models only after preservation and removal-safety gates."""
    structural_recall = float(metrics["structural_recall"])
    real_structural_recall = (
        float(real_metrics["structural_recall"])
        if real_metrics is not None
        else structural_recall
    )
    preservation_recall = min(structural_recall, real_structural_recall)
    synthetic_error = float(
        metrics.get("safe_removal_structural_error_rate", 0.0)
    )
    real_error = (
        float(real_metrics.get("safe_removal_structural_error_rate", 0.0))
        if real_metrics is not None
        else synthetic_error
    )
    maximum_error = max(synthetic_error, real_error)
    recall_eligible = preservation_recall >= minimum_structural_recall
    error_eligible = maximum_error <= maximum_structural_error_rate
    if recall_eligible and error_eligible:
        if real_metrics is not None:
            return (
                2.0,
                float(real_metrics["safe_removal_f1"]),
                float(metrics["safe_removal_f1"]),
                float(real_metrics["overlap_recall"]),
                float(metrics["overlap_recall"]),
                float(real_metrics["hatch_f1"]),
                preservation_recall,
            )
        return (
            2.0,
            float(metrics["safe_removal_f1"]),
            float(metrics["overlap_recall"]),
            float(metrics["hatch_f1"]),
            structural_recall,
        )
    if recall_eligible:
        return (
            1.0,
            -maximum_error,
            float(real_metrics["safe_removal_f1"])
            if real_metrics is not None
            else float(metrics["safe_removal_f1"]),
            float(metrics["safe_removal_f1"]),
            preservation_recall,
        )
    base = (
        0.0,
        preservation_recall,
        -maximum_error,
        float(metrics["safe_removal_f1"]),
        float(metrics["overlap_recall"]),
        float(metrics["hatch_f1"]),
    )
    if real_metrics is None:
        return base
    return base + (
        float(real_metrics["safe_removal_f1"]),
        float(real_metrics["overlap_recall"]),
        float(real_metrics["hatch_f1"]),
    )


def _window_origins(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    origins = list(range(0, length - patch_size + 1, stride))
    if origins[-1] != length - patch_size:
        origins.append(length - patch_size)
    return origins


@torch.no_grad()
def sliding_window_probabilities(
    model: HatchUNet,
    skeleton: np.ndarray,
    device: torch.device,
    *,
    patch_size: int = 512,
    stride: int = 256,
) -> np.ndarray:
    height, width = skeleton.shape
    accumulated = np.zeros((2, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    model.eval()
    for top in _window_origins(height, patch_size, stride):
        for left in _window_origins(width, patch_size, stride):
            patch = _crop_with_padding(skeleton, top, left, patch_size)
            patch_height = min(patch_size, height - top)
            patch_width = min(patch_size, width - left)
            tensor = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
            probabilities = torch.sigmoid(model(tensor))[0].cpu().numpy()
            accumulated[
                :, top : top + patch_height, left : left + patch_width
            ] += probabilities[:, :patch_height, :patch_width]
            counts[top : top + patch_height, left : left + patch_width] += 1.0
    probabilities = accumulated / np.maximum(counts[None], 1.0)
    probabilities *= (skeleton > 0)[None]
    return probabilities


@torch.no_grad()
def validate(
    model: HatchUNet,
    paths: list[Path],
    device: torch.device,
    *,
    patch_size: int,
    stride: int,
    maximum_samples: int | None,
    channel_threshold: float,
    hatch_high_threshold: float,
    structural_low_threshold: float,
) -> dict[str, float]:
    selected = paths[:maximum_samples] if maximum_samples else paths
    if not selected:
        raise ValueError("validation split is empty")
    accumulator = StrokeMetricAccumulator(
        channel_threshold=channel_threshold,
        hatch_high_threshold=hatch_high_threshold,
        structural_low_threshold=structural_low_threshold,
    )
    for path in selected:
        skeleton, targets, supervision, _metadata = _load_sample(path)
        probabilities = sliding_window_probabilities(
            model,
            skeleton,
            device,
            patch_size=patch_size,
            stride=stride,
        )
        accumulator.update(probabilities, targets, skeleton, supervision)
    metrics = accumulator.compute()
    metrics["validation_samples"] = float(len(selected))
    return metrics


def train_one_epoch(
    model: HatchUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    structural_weight: float,
    structural_negative_weight: float,
    hatch_weight: float,
    crossing_weight: float,
    bce_fraction: float,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    losses = []
    for skeleton, targets, supervision in loader:
        skeleton = skeleton.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        supervision = supervision.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(skeleton)
            loss = masked_multilabel_loss(
                logits,
                targets,
                skeleton,
                supervision_mask=supervision,
                structural_weight=structural_weight,
                structural_negative_weight=structural_negative_weight,
                hatch_weight=hatch_weight,
                crossing_weight=crossing_weight,
                bce_fraction=bce_fraction,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("training loader produced no batches")
    return float(np.mean(losses))


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two-channel Stage-2 structural/hachure decomposition."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--real-dataset", type=Path, default=None,
        help="Optional reviewed PatentData stroke dataset with supervision masks.",
    )
    parser.add_argument(
        "--real-fraction", type=float, default=0.0,
        help="Exact real-domain share of each training epoch.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=None,
        help="Load model weights only, then start a fresh optimizer/schedule.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=200)
    parser.add_argument("--max-real-validation-samples", type=int, default=None)
    parser.add_argument("--hatch-bias", type=float, default=0.75)
    parser.add_argument(
        "--difficulty-weights",
        type=parse_difficulty_weights,
        default={},
        help="Optional exact epoch supply, e.g. medium=0.2,hard=0.3,very_hard=0.5.",
    )
    parser.add_argument("--structural-weight", type=float, default=2.0)
    parser.add_argument(
        "--structural-negative-weight", type=float, default=1.0,
        help=("Extra BCE weight for supervised hatch-only pixels in the "
              "structural channel; 1.0 preserves the historical loss."),
    )
    parser.add_argument("--hatch-weight", type=float, default=1.0)
    parser.add_argument("--crossing-weight", type=float, default=4.0)
    parser.add_argument("--bce-fraction", type=float, default=0.6)
    parser.add_argument("--channel-threshold", type=float, default=0.5)
    parser.add_argument("--hatch-high-threshold", type=float, default=0.8)
    parser.add_argument("--structural-low-threshold", type=float, default=0.2)
    parser.add_argument("--minimum-structural-recall", type=float, default=0.995)
    parser.add_argument(
        "--maximum-structural-error-rate", type=float, default=0.001
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--freeze-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise SystemExit("epochs/batch must be positive and workers non-negative")
    if args.structural_negative_weight <= 0:
        raise SystemExit("structural-negative-weight must be positive")
    if args.patch < 32 or args.patch % 32 or args.stride < 1:
        raise SystemExit("patch must be a multiple of 32 and stride positive")
    if not 0.0 <= args.real_fraction < 1.0:
        raise SystemExit("real-fraction must be in [0, 1)")
    if not 0.0 <= args.maximum_structural_error_rate <= 1.0:
        raise SystemExit("maximum-structural-error-rate must be in [0, 1]")
    if args.real_fraction > 0 and args.real_dataset is None:
        raise SystemExit("positive real-fraction requires --real-dataset")
    if args.resume is not None and args.initial_checkpoint is not None:
        raise SystemExit("--resume and --initial-checkpoint are mutually exclusive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_paths = sorted((args.dataset / "train").glob("*.npz"))
    validation_paths = sorted((args.dataset / "validation").glob("*.npz"))
    if not train_paths or not validation_paths:
        raise SystemExit("dataset must contain non-empty train and validation splits")
    real_train_paths: list[Path] = []
    real_validation_paths: list[Path] = []
    if args.real_dataset is not None:
        real_train_paths = sorted((args.real_dataset / "train").glob("*.npz"))
        real_validation_paths = sorted(
            (args.real_dataset / "validation").glob("*.npz")
        )
        if not real_train_paths or not real_validation_paths:
            raise SystemExit(
                "real dataset must contain non-empty train and validation splits"
            )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested but unavailable: {args.device}")

    dataset = HatchStrokePatchDataset(
        train_paths,
        real_paths=real_train_paths,
        real_fraction=args.real_fraction,
        patch_size=args.patch,
        samples_per_epoch=args.samples_per_epoch,
        hatch_bias=args.hatch_bias,
        difficulty_weights=args.difficulty_weights,
        augment=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model = HatchUNet(
        freeze_encoder=args.freeze_encoder,
        out_channels=2,
        pretrained=args.pretrained,
    ).to(device)
    if args.initial_checkpoint is not None:
        initial = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=False
        )
        if initial.get("label_contract") != HATCH_STROKE_LABEL_CONTRACT:
            raise SystemExit("initial checkpoint uses an incompatible label contract")
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    configuration = {
        "dataset": str(args.dataset),
        "real_dataset": str(args.real_dataset) if args.real_dataset else None,
        "real_fraction": args.real_fraction,
        "source_counts_per_epoch": dataset.source_counts,
        "initial_checkpoint": (
            str(args.initial_checkpoint) if args.initial_checkpoint else None
        ),
        "label_contract": HATCH_STROKE_LABEL_CONTRACT,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patch": args.patch,
        "stride": args.stride,
        "samples_per_epoch": args.samples_per_epoch,
        "max_validation_samples": args.max_validation_samples,
        "max_real_validation_samples": args.max_real_validation_samples,
        "hatch_bias": args.hatch_bias,
        "difficulty_weights": args.difficulty_weights,
        "difficulty_counts_per_epoch": (
            allocate_difficulty_counts(
                dataset.source_counts["synthetic"], args.difficulty_weights
            )
            if args.difficulty_weights
            else None
        ),
        "structural_weight": args.structural_weight,
        "structural_negative_weight": args.structural_negative_weight,
        "hatch_weight": args.hatch_weight,
        "crossing_weight": args.crossing_weight,
        "bce_fraction": args.bce_fraction,
        "channel_threshold": args.channel_threshold,
        "hatch_high_threshold": args.hatch_high_threshold,
        "structural_low_threshold": args.structural_low_threshold,
        "minimum_structural_recall": args.minimum_structural_recall,
        "maximum_structural_error_rate": args.maximum_structural_error_rate,
        "pretrained": args.pretrained,
        "freeze_encoder": args.freeze_encoder,
        "seed": args.seed,
    }
    history: list[dict] = []
    best_rank: tuple[float, ...] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("label_contract") != HATCH_STROKE_LABEL_CONTRACT:
            raise SystemExit("resume checkpoint uses an incompatible label contract")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = list(checkpoint.get("history", []))
        best_rank_value = checkpoint.get("best_rank")
        best_rank = tuple(best_rank_value) if best_rank_value is not None else None
        best_metrics = checkpoint.get("best_metrics")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint["epoch"]) + 1

    last_path = args.out.with_name(args.out.stem + "_last" + args.out.suffix)
    history_path = args.history_out or args.out.with_name(
        args.out.stem + "_history.json"
    )
    print(
        f"train={len(train_paths)} validation={len(validation_paths)} "
        f"real_train={len(real_train_paths)} "
        f"real_validation={len(real_validation_paths)} "
        f"device={device} amp={amp_enabled}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        dataset.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            structural_weight=args.structural_weight,
            structural_negative_weight=args.structural_negative_weight,
            hatch_weight=args.hatch_weight,
            crossing_weight=args.crossing_weight,
            bce_fraction=args.bce_fraction,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        metrics = validate(
            model,
            validation_paths,
            device,
            patch_size=args.patch,
            stride=args.stride,
            maximum_samples=args.max_validation_samples,
            channel_threshold=args.channel_threshold,
            hatch_high_threshold=args.hatch_high_threshold,
            structural_low_threshold=args.structural_low_threshold,
        )
        real_metrics = None
        if real_validation_paths:
            real_metrics = validate(
                model,
                real_validation_paths,
                device,
                patch_size=args.patch,
                stride=args.stride,
                maximum_samples=args.max_real_validation_samples,
                channel_threshold=args.channel_threshold,
                hatch_high_threshold=args.hatch_high_threshold,
                structural_low_threshold=args.structural_low_threshold,
            )
        scheduler.step()
        rank = checkpoint_rank(
            metrics,
            args.minimum_structural_recall,
            real_metrics,
            args.maximum_structural_error_rate,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "time_seconds": time.time() - started,
            **metrics,
        }
        if real_metrics is not None:
            row.update({
                f"real_{name}": value for name, value in real_metrics.items()
            })
        history.append(row)
        improved = best_rank is None or rank > best_rank
        if improved:
            best_rank = rank
            best_metrics = {
                **metrics,
                "real_validation": real_metrics,
            }
            best_epoch = epoch

        checkpoint_payload = {
            "schema_version": "hatch-stroke-checkpoint-1.0",
            "label_contract": HATCH_STROKE_LABEL_CONTRACT,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "history": history,
            "best_rank": list(best_rank),
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "config": configuration,
            "channel_names": list(CHANNEL_NAMES),
        }
        _atomic_torch_save(checkpoint_payload, last_path)
        if improved:
            _atomic_torch_save(checkpoint_payload, args.out)
        _atomic_json(
            history_path,
            {
                "schema_version": "hatch-stroke-training-history-1.0",
                "label_contract": HATCH_STROKE_LABEL_CONTRACT,
                "config": configuration,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "history": history,
                "checkpoint": str(args.out),
                "last_checkpoint": str(last_path),
            },
        )
        marker = " best" if improved else ""
        print(
            f"epoch={epoch:03d} loss={train_loss:.5f} "
            f"struct_recall={metrics['structural_recall']:.5f} "
            f"safe_f1={metrics['safe_removal_f1']:.5f} "
            f"overlap_recall={metrics['overlap_recall']:.5f}"
            + (
                f" real_struct_recall={real_metrics['structural_recall']:.5f} "
                f"real_safe_f1={real_metrics['safe_removal_f1']:.5f}"
                if real_metrics is not None else ""
            )
            + marker
        )

    print(f"best_epoch={best_epoch} checkpoint={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
