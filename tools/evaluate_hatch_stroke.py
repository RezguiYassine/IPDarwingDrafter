"""Evaluate and calibrate a two-channel hatch-stroke checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.hatch_model import HatchUNet
from tools.hatch_stroke_train import (
    CHANNEL_NAMES,
    StrokeMetricAccumulator,
    _load_sample,
    sliding_window_probabilities,
)


class SafeRemovalHistogram:
    """Compact joint probability histogram for post-hoc policy sweeps."""

    def __init__(self, resolution: int = 100):
        if resolution < 20:
            raise ValueError("histogram resolution must be at least 20")
        self.resolution = int(resolution)
        shape = (self.resolution, self.resolution)
        self.all_pixels = np.zeros(shape, dtype=np.int64)
        self.safe_truth = np.zeros(shape, dtype=np.int64)
        self.structural_truth = np.zeros(shape, dtype=np.int64)
        self.safe_true_count = 0
        self.structural_true_count = 0

    def _add(self, destination: np.ndarray, flat_bins: np.ndarray) -> None:
        counts = np.bincount(
            flat_bins, minlength=self.resolution * self.resolution
        )
        destination += counts.reshape(destination.shape)

    def update(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        skeleton: np.ndarray,
        supervision_mask: np.ndarray | None = None,
    ) -> None:
        probabilities = np.asarray(probabilities)
        targets = np.asarray(targets) > 0
        skeleton = np.asarray(skeleton) > 0
        if probabilities.shape != targets.shape or probabilities.shape[0] != 2:
            raise ValueError("probabilities and targets must have shape (2, H, W)")
        if skeleton.shape != probabilities.shape[1:]:
            raise ValueError("skeleton shape disagrees with probabilities")
        if supervision_mask is None:
            supervision = np.broadcast_to(skeleton, probabilities.shape)
        else:
            supervision = np.asarray(supervision_mask) > 0
            if supervision.shape != probabilities.shape:
                raise ValueError("supervision_mask shape disagrees with probabilities")
            supervision &= skeleton[None]
        jointly_supervised = supervision[0] & supervision[1] & skeleton

        structural_probability = np.clip(
            probabilities[0][jointly_supervised], 0.0, 1.0
        )
        hatch_probability = np.clip(
            probabilities[1][jointly_supervised], 0.0, 1.0
        )
        structural_bins = np.minimum(
            (structural_probability * self.resolution).astype(np.int64),
            self.resolution - 1,
        )
        hatch_bins = np.minimum(
            (hatch_probability * self.resolution).astype(np.int64),
            self.resolution - 1,
        )
        flat_bins = structural_bins * self.resolution + hatch_bins
        safe_truth = (
            targets[1] & ~targets[0] & jointly_supervised
        )[jointly_supervised]
        structural_truth = (targets[0] & jointly_supervised)[jointly_supervised]

        self._add(self.all_pixels, flat_bins)
        self._add(self.safe_truth, flat_bins[safe_truth])
        self._add(self.structural_truth, flat_bins[structural_truth])
        self.safe_true_count += int(np.count_nonzero(safe_truth))
        self.structural_true_count += int(np.count_nonzero(structural_truth))

    def metrics(
        self, hatch_high_threshold: float, structural_low_threshold: float
    ) -> dict[str, float]:
        if not 0.0 <= hatch_high_threshold <= 1.0:
            raise ValueError("hatch threshold must be in [0, 1]")
        if not 0.0 <= structural_low_threshold <= 1.0:
            raise ValueError("structural threshold must be in [0, 1]")
        hatch_index = min(
            self.resolution,
            max(0, int(round(hatch_high_threshold * self.resolution))),
        )
        structural_index = min(
            self.resolution,
            max(0, int(round(structural_low_threshold * self.resolution))),
        )
        selection = np.s_[:structural_index, hatch_index:]
        predicted = int(self.all_pixels[selection].sum())
        true_positive = int(self.safe_truth[selection].sum())
        structural_removed = int(self.structural_truth[selection].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = (
            true_positive / self.safe_true_count if self.safe_true_count else 0.0
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        structural_error = (
            structural_removed / self.structural_true_count
            if self.structural_true_count
            else 0.0
        )
        return {
            "hatch_high_threshold": float(hatch_high_threshold),
            "structural_low_threshold": float(structural_low_threshold),
            "safe_removal_precision": precision,
            "safe_removal_recall": recall,
            "safe_removal_f1": f1,
            "safe_removal_structural_error_rate": structural_error,
            "safe_predicted_pixels": predicted,
            "safe_true_positive_pixels": true_positive,
            "structural_removed_pixels": structural_removed,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thresholds(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item < 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in [0, 1]")
    return values


def _difficulty_recall_floors(value: str) -> dict[str, float]:
    floors: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "difficulty floors must use name=value entries"
            )
        name, raw_floor = (part.strip() for part in item.split("=", 1))
        if not name or name in floors:
            raise argparse.ArgumentTypeError(
                "difficulty floor names must be non-empty and unique"
            )
        try:
            floor = float(raw_floor)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid difficulty floor for {name!r}: {raw_floor!r}"
            ) from exc
        if not 0.0 <= floor <= 1.0:
            raise argparse.ArgumentTypeError(
                "difficulty recall floors must be in [0, 1]"
            )
        floors[name] = floor
    return floors


def _difficulty_recall_gate(
    metrics: dict[str, dict[str, float]],
    default_floor: float,
    floors: dict[str, float],
) -> tuple[bool, dict[str, float]]:
    unknown = sorted(set(floors) - set(metrics))
    if unknown:
        raise ValueError(
            "difficulty recall floors name absent groups: " + ", ".join(unknown)
        )
    effective = {
        name: float(floors.get(name, default_floor)) for name in metrics
    }
    return (
        all(
            float(group_metrics["structural_recall"]) >= effective[name]
            for name, group_metrics in metrics.items()
        ),
        effective,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_root: Path,
    *,
    split: str,
    device: torch.device,
    patch_size: int,
    stride: int,
    maximum_samples: int | None,
    channel_threshold: float,
    fixed_hatch_high_threshold: float,
    fixed_structural_low_threshold: float,
    hatch_thresholds: list[float],
    structural_thresholds: list[float],
    minimum_structural_recall: float,
    difficulty_structural_recall_floors: dict[str, float] | None,
    maximum_structural_error_rate: float,
    histogram_resolution: int = 100,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = HatchUNet(
        freeze_encoder=False, out_channels=2, pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    paths = sorted((dataset_root / split).glob("*.npz"))
    if maximum_samples:
        paths = paths[:maximum_samples]
    if not paths:
        raise ValueError(f"no samples found for split {split!r}")

    fixed = StrokeMetricAccumulator(
        channel_threshold=channel_threshold,
        hatch_high_threshold=fixed_hatch_high_threshold,
        structural_low_threshold=fixed_structural_low_threshold,
    )
    by_difficulty: dict[str, StrokeMetricAccumulator] = defaultdict(
        lambda: StrokeMetricAccumulator(
            channel_threshold=channel_threshold,
            hatch_high_threshold=fixed_hatch_high_threshold,
            structural_low_threshold=fixed_structural_low_threshold,
        )
    )
    histogram = SafeRemovalHistogram(histogram_resolution)
    histograms_by_difficulty: dict[str, SafeRemovalHistogram] = {}
    started = time.time()
    for index, path in enumerate(paths, start=1):
        skeleton, targets, supervision, metadata = _load_sample(path)
        probabilities = sliding_window_probabilities(
            model,
            skeleton,
            device,
            patch_size=patch_size,
            stride=stride,
        )
        fixed.update(probabilities, targets, skeleton, supervision)
        by_difficulty[str(metadata.get("difficulty", "unknown"))].update(
            probabilities, targets, skeleton, supervision
        )
        histogram.update(probabilities, targets, skeleton, supervision)
        difficulty = str(metadata.get("difficulty", "unknown"))
        if difficulty not in histograms_by_difficulty:
            histograms_by_difficulty[difficulty] = SafeRemovalHistogram(
                histogram_resolution
            )
        histograms_by_difficulty[difficulty].update(
            probabilities, targets, skeleton, supervision
        )
        if index % 50 == 0 or index == len(paths):
            print(f"evaluated={index}/{len(paths)}", flush=True)

    fixed_metrics = fixed.compute()
    difficulty_metrics = {
        name: accumulator.compute()
        for name, accumulator in sorted(by_difficulty.items())
    }
    policies = [
        histogram.metrics(hatch_threshold, structural_threshold)
        for hatch_threshold in hatch_thresholds
        for structural_threshold in structural_thresholds
    ]
    structural_gate = (
        fixed_metrics["structural_recall"] >= minimum_structural_recall
    )
    difficulty_structural_gate, effective_difficulty_floors = (
        _difficulty_recall_gate(
            difficulty_metrics,
            minimum_structural_recall,
            difficulty_structural_recall_floors or {},
        )
    )
    for policy in policies:
        policy["difficulty_metrics"] = {
            name: difficulty_histogram.metrics(
                policy["hatch_high_threshold"],
                policy["structural_low_threshold"],
            )
            for name, difficulty_histogram in sorted(
                histograms_by_difficulty.items()
            )
        }
        policy["meets_structural_recall"] = structural_gate
        policy["meets_difficulty_structural_recall"] = (
            difficulty_structural_gate
        )
        policy["meets_structural_error"] = (
            policy["safe_removal_structural_error_rate"]
            <= maximum_structural_error_rate
        )
        policy["meets_difficulty_structural_error"] = all(
            metrics["safe_removal_structural_error_rate"]
            <= maximum_structural_error_rate
            for metrics in policy["difficulty_metrics"].values()
        )
        policy["eligible"] = bool(
            policy["meets_structural_recall"]
            and policy["meets_difficulty_structural_recall"]
            and policy["meets_structural_error"]
            and policy["meets_difficulty_structural_error"]
        )
    policies.sort(
        key=lambda item: (
            item["eligible"],
            item["safe_removal_f1"],
            item["safe_removal_recall"],
            -item["safe_removal_structural_error_rate"],
        ),
        reverse=True,
    )
    selected_policy = next(
        (policy for policy in policies if policy["eligible"]), None
    )
    return {
        "schema_version": "hatch-stroke-evaluation-1.1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_best_epoch": int(checkpoint.get("best_epoch", 0)),
        "label_contract": checkpoint.get("label_contract"),
        "dataset": str(dataset_root),
        "split": split,
        "samples": len(paths),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "channel_names": list(CHANNEL_NAMES),
        "fixed_thresholds": {
            "channel": channel_threshold,
            "hatch_high": fixed_hatch_high_threshold,
            "structural_low": fixed_structural_low_threshold,
        },
        "fixed_metrics": fixed_metrics,
        "difficulty_metrics": difficulty_metrics,
        "selection_constraints": {
            "minimum_structural_recall": minimum_structural_recall,
            "difficulty_structural_recall_floors": effective_difficulty_floors,
            "maximum_structural_error_rate": maximum_structural_error_rate,
        },
        "selected_policy": selected_policy,
        "policy_sweep": policies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate and threshold-calibrate a hatch-stroke checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--channel-threshold", type=float, default=0.5)
    parser.add_argument("--fixed-hatch-high-threshold", type=float, default=0.8)
    parser.add_argument("--fixed-structural-low-threshold", type=float, default=0.2)
    parser.add_argument(
        "--hatch-thresholds",
        type=_thresholds,
        default=_thresholds("0.50,0.60,0.70,0.80,0.90,0.95"),
    )
    parser.add_argument(
        "--structural-thresholds",
        type=_thresholds,
        default=_thresholds("0.05,0.10,0.15,0.20,0.30,0.40,0.50"),
    )
    parser.add_argument("--minimum-structural-recall", type=float, default=0.995)
    parser.add_argument(
        "--difficulty-structural-recall-floors",
        type=_difficulty_recall_floors,
        default={},
        help=("Optional baseline-relative subgroup floors, for example "
              "hard=0.9968,medium=0.9986,very_hard=0.9934."),
    )
    parser.add_argument("--maximum-structural-error-rate", type=float, default=0.001)
    parser.add_argument("--histogram-resolution", type=int, default=100)
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        args.dataset,
        split=args.split,
        device=torch.device(args.device),
        patch_size=args.patch,
        stride=args.stride,
        maximum_samples=args.max_samples,
        channel_threshold=args.channel_threshold,
        fixed_hatch_high_threshold=args.fixed_hatch_high_threshold,
        fixed_structural_low_threshold=args.fixed_structural_low_threshold,
        hatch_thresholds=args.hatch_thresholds,
        structural_thresholds=args.structural_thresholds,
        minimum_structural_recall=args.minimum_structural_recall,
        difficulty_structural_recall_floors=(
            args.difficulty_structural_recall_floors
        ),
        maximum_structural_error_rate=args.maximum_structural_error_rate,
        histogram_resolution=args.histogram_resolution,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "fixed_metrics": result["fixed_metrics"],
        "selected_policy": result["selected_policy"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
