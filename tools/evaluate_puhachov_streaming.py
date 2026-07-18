"""Distributed Stage-2 evaluation directly from official SketchGraphs sequences."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage2_strokeextraction"))

import stage2_stroke_extract as s2  # noqa: E402
from stage2_strokeextraction.research.train_puhachov import (  # noqa: E402
    N_CLASSES,
    STATUS_ACCEPTED,
    STATUS_EMPTY_SKELETON,
    STATUS_NAMES,
    STATUS_OTHER_ERROR,
    SketchGraphsKPDataset,
    _sketchgraphs_status,
)


class SketchGraphsEvalDataset(Dataset):
    def __init__(self, raw_path: Path, canvas: int = 512, margin: int = 24,
                 stroke_width: int = 2):
        self.source = SketchGraphsKPDataset(
            raw_path, canvas=canvas, margin=margin, stroke_width=stroke_width,
            crop=canvas, sigma=3.0, augment=False, seed=0,
        )
        self.canvas = canvas

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        try:
            skeleton, keypoints = self.source._render(index)
            image = torch.from_numpy(
                (skeleton > 0).astype(np.float32)[None]
            )
            status = STATUS_ACCEPTED
        except Exception as exc:
            status = _sketchgraphs_status(exc)
            if "empty skeleton" in str(exc).lower():
                status = STATUS_EMPTY_SKELETON
            image = torch.zeros((1, self.canvas, self.canvas), dtype=torch.float32)
            keypoints = np.zeros((0, 3), dtype=np.int32)
        return image, keypoints, index, status


class CachedEvalDataset(Dataset):
    def __init__(self, labels: Path, split: str):
        self.paths = sorted(Path(path) for path in glob.glob(
            str(labels / split / "**" / "*.npz"), recursive=True,
        ))
        if not self.paths:
            raise FileNotFoundError(f"no labels under {labels / split}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        try:
            with np.load(self.paths[index], allow_pickle=True) as data:
                skeleton = data["skeleton"]
                keypoints = np.asarray(data["kps"], dtype=np.int32)
            image = torch.from_numpy(
                (skeleton > 0).astype(np.float32)[None]
            )
            status = STATUS_ACCEPTED
        except Exception:
            shape = (1, 512, 512)
            image = torch.zeros(shape, dtype=torch.float32)
            keypoints = np.zeros((0, 3), dtype=np.int32)
            status = STATUS_OTHER_ERROR
        return image, keypoints, index, status


class ExactEvalSampler(Sampler):
    """Assign every source index to exactly one rank without padding."""

    def __init__(self, size: int, rank: int, world_size: int):
        self.size = size
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.size:
            return 0
        return (self.size - 1 - self.rank) // self.world_size + 1


def _collate(batch):
    images, keypoints, indices, statuses = zip(*batch)
    shapes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
    max_height = max((height + 63) // 64 * 64 for height, _width in shapes)
    max_width = max((width + 63) // 64 * 64 for _height, width in shapes)
    padded = [
        torch.nn.functional.pad(
            image, (0, max_width - image.shape[-1], 0, max_height - image.shape[-2])
        )
        for image in images
    ]
    return (
        torch.stack(padded), list(keypoints), shapes,
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(statuses, dtype=torch.int64),
    )


def _match_counts(heatmaps: np.ndarray, keypoints: np.ndarray, conf: float,
                  nms_radius: int, match_radius: float):
    tp = np.zeros(N_CLASSES, dtype=np.int64)
    fp = np.zeros(N_CLASSES, dtype=np.int64)
    fn = np.zeros(N_CLASSES, dtype=np.int64)
    for class_id in range(N_CLASSES):
        peaks = s2._extract_peaks(heatmaps[class_id], conf, nms_radius)
        ground_truth = [
            (int(x), int(y)) for x, y, kind in keypoints
            if int(kind) == class_id
        ]
        used = np.zeros(len(ground_truth), dtype=bool)
        for peak_x, peak_y, _confidence in peaks:
            best_distance = match_radius + 1e-6
            best_index = -1
            for index, (gt_x, gt_y) in enumerate(ground_truth):
                if used[index]:
                    continue
                distance = float(np.hypot(peak_x - gt_x, peak_y - gt_y))
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index >= 0:
                used[best_index] = True
                tp[class_id] += 1
            else:
                fp[class_id] += 1
        fn[class_id] += int((~used).sum())
    return tp, fp, fn


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _distributed_context(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, f"cuda:{local_rank}"
    return rank, world_size, device_arg


def evaluate(args) -> dict | None:
    rank, world_size, device = _distributed_context(args.device)
    is_main = rank == 0
    if args.raw:
        dataset = SketchGraphsEvalDataset(
            args.raw, canvas=args.canvas, margin=args.margin,
            stroke_width=args.stroke_width,
        )
        source = {"raw": str(args.raw)}
    else:
        dataset = CachedEvalDataset(args.labels, args.split)
        source = {"labels": str(args.labels), "split": args.split}
    source_total = len(dataset)
    evaluation_total = source_total if args.limit <= 0 else min(args.limit, source_total)
    sampler = ExactEvalSampler(evaluation_total, rank, world_size)
    loader_options = dict(
        dataset=dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        persistent_workers=args.workers > 0, collate_fn=_collate,
        worker_init_fn=_worker_init,
    )
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = s2._build_stacked_hourglass().to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    counters = np.zeros(3 * N_CLASSES + len(STATUS_NAMES) + 1, dtype=np.int64)
    tp_slice = slice(0, N_CLASSES)
    fp_slice = slice(N_CLASSES, 2 * N_CLASSES)
    fn_slice = slice(2 * N_CLASSES, 3 * N_CLASSES)
    status_offset = 3 * N_CLASSES
    seen_offset = status_offset + len(STATUS_NAMES)
    started = time.time()

    with torch.inference_mode():
        for images, keypoints, shapes, _indices, statuses in loader:
            counters[seen_offset] += len(statuses)
            status_values, status_counts = np.unique(statuses.numpy(), return_counts=True)
            for status, count in zip(status_values, status_counts):
                counters[status_offset + int(status)] += int(count)
            valid_positions = torch.nonzero(statuses == STATUS_ACCEPTED).flatten().tolist()
            if valid_positions:
                valid_images = images[valid_positions].to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype,
                    enabled=str(device).startswith("cuda"),
                ):
                    heatmaps = torch.sigmoid(model(valid_images))
                heatmaps_np = heatmaps.float().cpu().numpy()
                for output_index, batch_index in enumerate(valid_positions):
                    height, width = shapes[batch_index]
                    tp, fp, fn = _match_counts(
                        heatmaps_np[output_index, :, :height, :width],
                        keypoints[batch_index],
                        args.conf, args.nms_radius, args.match_radius,
                    )
                    counters[tp_slice] += tp
                    counters[fp_slice] += fp
                    counters[fn_slice] += fn

            if is_main and counters[seen_offset] % args.log_every < len(statuses):
                approximate_seen = min(
                    evaluation_total, int(counters[seen_offset]) * world_size
                )
                rate = approximate_seen / max(time.time() - started, 1e-9)
                print(
                    f"evaluated {approximate_seen:,}/{evaluation_total:,} "
                    f"({100*approximate_seen/evaluation_total:.1f}%) "
                    f"at {rate:.1f} sketches/s",
                    flush=True,
                )

    reduced = torch.from_numpy(counters).to(device)
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    counters = reduced.cpu().numpy()
    elapsed = time.time() - started

    report = None
    if is_main:
        tp = counters[tp_slice].astype(np.float64)
        fp = counters[fp_slice].astype(np.float64)
        fn = counters[fn_slice].astype(np.float64)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / np.maximum(tp + fn, 1)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        support = tp + fn
        supported = support > 0
        names = ("endpoint", "junction", "corner")
        status_counts = {
            STATUS_NAMES[code]: int(counters[status_offset + code])
            for code in STATUS_NAMES
        }
        report = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": checkpoint.get("step"),
            **source,
            "source_total": source_total,
            "evaluated": int(counters[seen_offset]),
            "accepted": status_counts["accepted"],
            "rejected": int(counters[seen_offset]) - status_counts["accepted"],
            "status_counts": status_counts,
            "world_size": world_size,
            "batch_per_rank": args.batch,
            "threshold": args.conf,
            "nms_radius": args.nms_radius,
            "match_radius": args.match_radius,
            "elapsed_seconds": elapsed,
            "throughput_sketches_s": int(counters[seen_offset]) / max(elapsed, 1e-9),
            "macro_f1": float(f1[supported].mean()),
            "classes": {
                name: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                    "tp": int(tp[index]),
                    "fp": int(fp[index]),
                    "fn": int(fn[index]),
                }
                for index, name in enumerate(names)
            },
        }
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
    if world_size > 1:
        dist.destroy_process_group()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Puhachov directly on an official SketchGraphs split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path)
    source.add_argument("--labels", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--stroke-width", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=3)
    parser.add_argument("--match-radius", type=float, default=6.0)
    parser.add_argument("--log-every", type=int, default=10_000)
    args = parser.parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
