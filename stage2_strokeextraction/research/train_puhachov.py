"""
Train the Stage 2 keypoint CNN on Drawing2CAD  (Puhachov roadmap — Phase 2)
==========================================================================

Trains the in-repo stacked-hourglass keypoint detector
(`_build_stacked_hourglass` from stage2_stroke_extract) on the labels produced
by `tools/d2c_keypoint_labels.py`, and exports a checkpoint that loads back
through the existing guarded `PuhachovKeypointDetector` loader unchanged.

Design
------
* **Inputs**  cached npz `{skeleton uint8 (H,W), kps int32 (N,3)=(x,y,type)}`.
* **Targets** 3-channel Gaussian-splat heatmaps (endpoint/junction/corner),
  built on the fly (channel order matches PuhachovKeypointDetector.detect).
* **Crops**   native-scale `crop`×`crop` windows, biased to contain a keypoint.
  No resampling → 1-px skeletons stay intact and the pixel scale matches the
  full-resolution skeletons seen at inference.
* **Loss**    CenterNet penalty-reduced focal loss (robust to sparse peaks).
* **Aug**     lossless 90° rotations + flips applied to skeleton and heatmap
  together.
* **Select**  checkpoint the best per-class peak-F1 on a validation subset
  (full-image inference + greedy peak matching).

Usage (from project root, after labels exist):

    python -m stage2_strokeextraction.research.train_puhachov \
        --labels output/Drawing2CAD/kp_labels \
        --out models/puhachov_d2c.pth \
        --steps 40000 --batch 12 --device cuda:0

    # quick smoke test on whatever labels exist so far
    python -m stage2_strokeextraction.research.train_puhachov \
        --labels output/Drawing2CAD/kp_labels --steps 3 --batch 2 \
        --val-subset 8 --out /tmp/puhachov_smoke.pth
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "stage2_strokeextraction"))
import stage2_stroke_extract as s2  # noqa: E402

N_CLASSES = 3   # endpoint=0, junction=1, corner=2  (matches detect())
STRIDE = 64     # hourglass total downsample; inputs padded to a multiple


# ─── Gaussian heatmap target ─────────────────────────────────────────────────

def _gaussian2d(sigma: float) -> np.ndarray:
    r = int(round(3 * sigma))
    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return g.astype(np.float32)


def _splat(hm: np.ndarray, cx: int, cy: int, g: np.ndarray) -> None:
    """Max-combine a Gaussian patch centered at (cx, cy) into a heatmap plane."""
    H, W = hm.shape
    r = g.shape[0] // 2
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    gx0, gy0 = x0 - (cx - r), y0 - (cy - r)
    patch = g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
    np.maximum(hm[y0:y1, x0:x1], patch, out=hm[y0:y1, x0:x1])


def make_heatmap(kps: np.ndarray, H: int, W: int, g: np.ndarray) -> np.ndarray:
    hm = np.zeros((N_CLASSES, H, W), dtype=np.float32)
    for x, y, t in kps:
        if 0 <= t < N_CLASSES and 0 <= x < W and 0 <= y < H:
            _splat(hm[int(t)], int(x), int(y), g)
    return hm


# ─── Patent-style degradation (domain randomization for fine-tuning) ──────────

def patent_degrade(sk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make a clean D2C skeleton look patent-scan-like, WITHOUT moving keypoints.

    The measured OOD gap is *clutter*, not staircase (the CNN already trained on
    Zhang-Suen-staircased D2C). So the degradations are **additive clutter** that
    leaves the labelled structure intact — speckle, short spurs, hachure clusters
    — plus a few mild gaps. No thicken+re-skeletonize: it can merge close parallel
    rails and clean rather than degrade, moving keypoints off the skeleton.
    """
    out = (sk > 0).astype(np.uint8) * 255
    H, W = out.shape
    ys, xs = np.where(out > 0)
    if len(xs) == 0:
        return out

    # 1. speckle: isolated foreground pixels + tiny blobs (scan noise)
    if rng.random() < 0.7:
        for _ in range(int(rng.integers(20, 100))):
            y, x = int(rng.integers(0, H)), int(rng.integers(0, W))
            out[y, x] = 255
            if rng.random() < 0.3:  # occasional 2px blob
                out[min(H - 1, y + 1), x] = 255
    # 2. short spurs/barbs off existing strokes
    if rng.random() < 0.6:
        for _ in range(int(rng.integers(5, 25))):
            i = int(rng.integers(0, len(xs)))
            a = rng.uniform(0, 2 * np.pi); L = int(rng.integers(2, 8))
            x2 = int(np.clip(xs[i] + L * np.cos(a), 0, W - 1))
            y2 = int(np.clip(ys[i] + L * np.sin(a), 0, H - 1))
            cv2.line(out, (int(xs[i]), int(ys[i])), (x2, y2), 255, 1)
    # 3. hachure clusters: short parallel segments in random regions
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(2, 5))):
            cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
            a = rng.uniform(0, np.pi); L = int(rng.integers(8, 22))
            gap = int(rng.integers(3, 7)); n = int(rng.integers(3, 9))
            px, py = np.cos(a + np.pi / 2), np.sin(a + np.pi / 2)
            for j in range(n):
                x1 = int(np.clip(cx + j * gap * px, 0, W - 1))
                y1 = int(np.clip(cy + j * gap * py, 0, H - 1))
                x2 = int(np.clip(x1 + L * np.cos(a), 0, W - 1))
                y2 = int(np.clip(y1 + L * np.sin(a), 0, H - 1))
                cv2.line(out, (x1, y1), (x2, y2), 255, 1)
    # 4. mild gaps: erase a few tiny patches (broken strokes)
    if rng.random() < 0.3:
        for _ in range(int(rng.integers(2, 8))):
            i = int(rng.integers(0, len(xs)))
            cv2.circle(out, (int(xs[i]), int(ys[i])), int(rng.integers(1, 3)), 0, -1)
    return out


# ─── Dataset ─────────────────────────────────────────────────────────────────

SOURCE_CACHED = 0
SOURCE_SKETCHGRAPHS = 1

STATUS_ACCEPTED = 0
STATUS_NO_SUPPORTED_GEOMETRY = 1
STATUS_NONFINITE_GEOMETRY = 2
STATUS_DEGENERATE_GEOMETRY = 3
STATUS_EMPTY_SKELETON = 4
STATUS_DECODE_ERROR = 5
STATUS_OTHER_ERROR = 6
STATUS_NAMES = {
    STATUS_ACCEPTED: "accepted",
    STATUS_NO_SUPPORTED_GEOMETRY: "no_supported_geometry",
    STATUS_NONFINITE_GEOMETRY: "nonfinite_geometry",
    STATUS_DEGENERATE_GEOMETRY: "degenerate_geometry",
    STATUS_EMPTY_SKELETON: "empty_skeleton",
    STATUS_DECODE_ERROR: "decode_error",
    STATUS_OTHER_ERROR: "other_error",
}


def _dataset_key(key) -> tuple[int, int]:
    if isinstance(key, tuple):
        return int(key[0]), int(key[1])
    return 0, int(key)


def _sample_rng(seed: int, epoch: int, index: int, source: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([seed, epoch, index, source])
    return np.random.default_rng(sequence)


class KPDataset(Dataset):
    def __init__(self, npz_paths, crop=512, sigma=3.0, augment=True,
                 pos_crop_prob=0.8, patent_aug=0.0, seed=42):
        self.paths = list(npz_paths)
        self.crop = crop
        self.sigma = sigma
        self.augment = augment
        self.pos_crop_prob = pos_crop_prob
        self.patent_aug = patent_aug      # P(apply patent degradation per sample)
        self.g = _gaussian2d(sigma)
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def _pick_window(self, H, W, kps, rng):
        cs = self.crop
        if H <= cs and W <= cs:
            return 0, 0
        if len(kps) and rng.random() < self.pos_crop_prob:
            kx, ky = kps[int(rng.integers(len(kps)))][:2]
            jit = cs // 4
            cx = kx + int(rng.integers(-jit, jit + 1))
            cy = ky + int(rng.integers(-jit, jit + 1))
        else:
            cx = int(rng.integers(0, W + 1))
            cy = int(rng.integers(0, H + 1))
        x0 = int(np.clip(cx - cs // 2, 0, max(0, W - cs)))
        y0 = int(np.clip(cy - cs // 2, 0, max(0, H - cs)))
        return x0, y0

    def _transform(self, sk, kps, rng):
        H, W = sk.shape
        cs = self.crop

        x0, y0 = self._pick_window(H, W, kps, rng)
        sk_c = sk[y0:y0 + cs, x0:x0 + cs]
        # pad to crop size if the image is smaller than the window
        ph, pw = cs - sk_c.shape[0], cs - sk_c.shape[1]
        if ph or pw:
            sk_c = np.pad(sk_c, ((0, ph), (0, pw)), mode="constant")
        kc = []
        for x, y, t in kps:
            cx, cy = x - x0, y - y0
            if 0 <= cx < cs and 0 <= cy < cs:
                kc.append((cx, cy, t))
        kc = np.array(kc, dtype=np.int32) if kc else np.zeros((0, 3), np.int32)

        # Patent-style degradation on the input only (labels stay valid).
        if self.patent_aug and rng.random() < self.patent_aug:
            sk_c = patent_degrade(sk_c, rng)

        hm = make_heatmap(kc, cs, cs, self.g)
        img = (sk_c > 0).astype(np.float32)[None]   # (1, cs, cs)

        if self.augment:
            k = int(rng.integers(0, 4))
            if k:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                hm = np.rot90(hm, k, axes=(1, 2)).copy()
            if rng.random() < 0.5:
                img = img[:, :, ::-1].copy(); hm = hm[:, :, ::-1].copy()
            if rng.random() < 0.5:
                img = img[:, ::-1, :].copy(); hm = hm[:, ::-1, :].copy()

        return torch.from_numpy(img), torch.from_numpy(hm)

    def __getitem__(self, key):
        epoch, i = _dataset_key(key)
        d = np.load(self.paths[i], allow_pickle=True)
        rng = _sample_rng(self.seed, epoch, i, SOURCE_CACHED)
        img, hm = self._transform(d["skeleton"], d["kps"], rng)
        return img, hm, True, SOURCE_CACHED, i, STATUS_ACCEPTED


def _sketchgraphs_status(exc: Exception) -> int:
    message = str(exc).lower()
    if "no supported" in message:
        return STATUS_NO_SUPPORTED_GEOMETRY
    if "non-finite" in message:
        return STATUS_NONFINITE_GEOMETRY
    if "degenerate geometry" in message:
        return STATUS_DEGENERATE_GEOMETRY
    if isinstance(exc, (IndexError, KeyError, TypeError)):
        return STATUS_DECODE_ERROR
    return STATUS_OTHER_ERROR


class SketchGraphsKPDataset(KPDataset):
    """Render Stage-2 labels lazily from the official flat-array sequence file."""

    def __init__(self, raw_path, canvas=512, margin=24, stroke_width=2, **kwargs):
        super().__init__([], **kwargs)
        self.raw_path = str(raw_path)
        self.canvas = canvas
        self.margin = margin
        self.stroke_width = stroke_width
        self._data = None
        self._helpers = None
        from sketchgraphs.data import flat_array
        data = flat_array.load_dictionary_flat(self.raw_path)
        self.length = len(data["sequences"])
        del data

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_data"] = None
        state["_helpers"] = None
        return state

    def _ensure_open(self):
        if self._data is not None:
            return
        from sketchgraphs.data import flat_array, sketch_from_sequence
        from tools.sketchgraphs_dataset import (
            PrepareConfig, _build_primitives, _keypoints, _render_skeleton,
        )
        self._data = flat_array.load_dictionary_flat(self.raw_path)
        self._helpers = (
            sketch_from_sequence, PrepareConfig,
            _build_primitives, _keypoints, _render_skeleton,
        )

    def _render(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_open()
        sketch_from_sequence, config_cls, build, keypoints, render = self._helpers
        cfg = config_cls(
            canvas=self.canvas, margin=self.margin,
            stroke_width=self.stroke_width, write_stage2=False,
        )
        sketch = sketch_from_sequence(self._data["sequences"][index])
        primitives, _transform = build(sketch, cfg)
        skeleton = render(primitives, cfg)
        if not np.any(skeleton):
            raise ValueError("empty skeleton")
        kps, _clusters = keypoints(primitives, skeleton)
        return skeleton, kps

    def __getitem__(self, key):
        epoch, i = _dataset_key(key)
        try:
            skeleton, kps = self._render(i)
            rng = _sample_rng(self.seed, epoch, i, SOURCE_SKETCHGRAPHS)
            img, hm = self._transform(skeleton, kps, rng)
            valid, status = True, STATUS_ACCEPTED
        except Exception as exc:
            status = _sketchgraphs_status(exc)
            if "empty skeleton" in str(exc).lower():
                status = STATUS_EMPTY_SKELETON
            img = torch.zeros((1, self.crop, self.crop), dtype=torch.float32)
            hm = torch.zeros((N_CLASSES, self.crop, self.crop), dtype=torch.float32)
            valid = False
        return img, hm, valid, SOURCE_SKETCHGRAPHS, i, status


def _affine_parameters(size: int, seed: int) -> tuple[int, int]:
    if size <= 1:
        return 0, 0
    rng = random.Random(seed)
    multiplier = rng.randrange(1, size)
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size
        if multiplier == 0:
            multiplier = 1
    offset = rng.randrange(size)
    return multiplier, offset


def _affine_permutation(index: int, size: int, seed: int) -> int:
    multiplier, offset = _affine_parameters(size, seed)
    return (multiplier * index + offset) % size if size > 1 else 0


class ExactMixedDataset(Dataset):
    """Mix cached labels with exactly one slot per SketchGraphs sequence."""

    def __init__(self, primary: Dataset, sketchgraphs: Dataset,
                 primary_fraction: float, seed: int):
        if not 0.0 <= primary_fraction < 1.0:
            raise ValueError("streaming mix requires 0 <= --mix < 1")
        if primary_fraction > 0.0 and len(primary) == 0:
            raise ValueError("primary label pool is empty")
        self.primary = primary
        self.sketchgraphs = sketchgraphs
        self.primary_fraction = primary_fraction
        self.seed = seed
        self.n_sketchgraphs = len(sketchgraphs)
        self.n_primary = int(round(
            self.n_sketchgraphs * primary_fraction / (1.0 - primary_fraction)
        ))
        self.length = self.n_sketchgraphs + self.n_primary
        self._permutation_cache = {}

    def __len__(self):
        return self.length

    def _primary_index(self, ordinal: int, epoch: int) -> int:
        pool = len(self.primary)
        cycle, position = divmod(ordinal, pool)
        cache_key = epoch, cycle
        if cache_key not in self._permutation_cache:
            self._permutation_cache[cache_key] = _affine_parameters(
                pool, self.seed + epoch * 1009 + cycle
            )
        multiplier, offset = self._permutation_cache[cache_key]
        return (multiplier * position + offset) % pool

    def source_for_index(self, index: int, epoch: int = 0) -> tuple[int, int]:
        before = (index * self.n_sketchgraphs) // self.length
        after = ((index + 1) * self.n_sketchgraphs) // self.length
        if after > before:
            return SOURCE_SKETCHGRAPHS, after - 1
        primary_ordinal = index - before
        return SOURCE_CACHED, self._primary_index(primary_ordinal, epoch)

    def __getitem__(self, key):
        epoch, i = _dataset_key(key)
        source, source_index = self.source_for_index(i, epoch)
        if source == SOURCE_SKETCHGRAPHS:
            return self.sketchgraphs[(epoch, source_index)]
        return self.primary[(epoch, source_index)]


class ConstantMemoryDistributedSampler(Sampler):
    """Distributed full-coverage shuffle without materializing randperm(N)."""

    def __init__(self, dataset, num_replicas=1, rank=0, seed=42, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start_index = 0
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError("sampler resume offset is outside this rank")
        self.start_index = start_index

    def __iter__(self):
        size = len(self.dataset)
        permutation_seed = self.seed + self.epoch * 1_000_003
        multiplier, offset = _affine_parameters(size, permutation_seed)
        for local_position in range(self.start_index, self.num_samples):
            global_position = local_position * self.num_replicas + self.rank
            base_index = global_position if global_position < size else global_position - size
            index = ((multiplier * base_index + offset) % size
                     if self.shuffle and size > 1 else base_index)
            yield self.epoch, index

    def __len__(self):
        return self.num_samples - self.start_index


# ─── CenterNet penalty-reduced focal loss ────────────────────────────────────

def focal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # Sparse focal loss is numerically fragile in FP16. Keep this reduction in
    # FP32 even when convolution layers run under autocast.
    pred = torch.clamp(torch.sigmoid(logits.float()), 1e-6, 1 - 1e-6)
    gt = gt.float()
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_w = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    n_pos = pos.sum()
    pos_loss, neg_loss = pos_loss.sum(), neg_loss.sum()
    if n_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / n_pos


# ─── Validation: per-class peak F1 ───────────────────────────────────────────

@torch.no_grad()
def evaluate_f1(model, val_paths, device, conf=0.3, nms_radius=3,
                match_radius=6):
    model.eval()
    tp = np.zeros(N_CLASSES); fp = np.zeros(N_CLASSES); fn = np.zeros(N_CLASSES)
    for p in val_paths:
        d = np.load(p, allow_pickle=True)
        sk = (d["skeleton"] > 0).astype(np.float32)
        H, W = sk.shape
        ph = (STRIDE - H % STRIDE) % STRIDE
        pw = (STRIDE - W % STRIDE) % STRIDE
        inp = np.pad(sk, ((0, ph), (0, pw)))
        t = torch.from_numpy(inp)[None, None].to(device)
        use_amp = str(device).startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            hm = torch.sigmoid(model(t))[0, :, :H, :W].float().cpu().numpy()

        gt = d["kps"]
        for c in range(N_CLASSES):
            peaks = s2._extract_peaks(hm[c], conf, nms_radius)   # (x,y,conf)
            gt_c = [(x, y) for x, y, tt in gt if tt == c]
            used = [False] * len(gt_c)
            for px, py, _ in peaks:
                best, bj = match_radius + 1e-6, -1
                for j, (gx, gy) in enumerate(gt_c):
                    if used[j]:
                        continue
                    dd = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                    if dd < best:
                        best, bj = dd, j
                if bj >= 0:
                    used[bj] = True; tp[c] += 1
                else:
                    fp[c] += 1
            fn[c] += used.count(False)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    support = tp + fn
    supported = support > 0
    macro_f1 = float(f1[supported].mean()) if supported.any() else 0.0
    model.train()
    return {
        "f1": f1, "prec": prec, "rec": rec, "support": support,
        "macro_f1": macro_f1,
    }


# ─── Training loop ───────────────────────────────────────────────────────────

def _collect(labels_root: Path, split: str) -> list[Path]:
    return sorted(Path(p) for p in
                  glob.glob(str(labels_root / split / "**" / "*.npz"),
                            recursive=True))


def _dist_info(args) -> tuple[int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available():
            raise SystemExit("torchrun multi-process training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, f"cuda:{local_rank}"
    return rank, world_size, args.device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _raw_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _atomic_torch_save(payload: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


class CoverageTracker:
    def __init__(self, path: Path, size: int, rank: int, world_size: int,
                 reset: bool, source_path: str):
        self.path = path
        self.meta_path = path.with_suffix(path.suffix + ".json")
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.source_path = source_path
        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            if reset or not path.exists() or path.stat().st_size != size:
                status = np.memmap(path, dtype=np.int8, mode="w+", shape=(size,))
                status[:] = -1
                status.flush()
                del status
        _barrier(world_size)
        self.status = np.memmap(path, dtype=np.int8, mode="r+", shape=(size,))

    def update(self, source, indices, statuses) -> None:
        source = np.asarray(source)
        mask = source == SOURCE_SKETCHGRAPHS
        if np.any(mask):
            idx = np.asarray(indices, dtype=np.int64)[mask]
            values = np.asarray(statuses, dtype=np.int8)[mask]
            self.status[idx] = values

    def flush(self) -> None:
        self.status.flush()

    def summarize(self, epoch: int, step: int) -> dict | None:
        self.flush()
        _barrier(self.world_size)
        report = None
        if self.rank == 0:
            values, counts = np.unique(self.status, return_counts=True)
            raw_counts = {int(k): int(v) for k, v in zip(values, counts)}
            report = {
                "source_path": self.source_path,
                "source_total": self.size,
                "epoch": epoch,
                "step": step,
                "attempted": self.size - raw_counts.get(-1, 0),
                "unattempted": raw_counts.get(-1, 0),
                "accepted": raw_counts.get(STATUS_ACCEPTED, 0),
                "rejected": sum(raw_counts.get(code, 0) for code in STATUS_NAMES if code),
                "status_counts": {
                    STATUS_NAMES[code]: raw_counts.get(code, 0)
                    for code in STATUS_NAMES
                },
            }
            report["complete"] = report["unattempted"] == 0
            self.meta_path.write_text(json.dumps(report, indent=2) + "\n")
        _barrier(self.world_size)
        return report


def _checkpoint_payload(model, optimizer, scaler, args, step, epoch,
                        samples_in_epoch, best_f1, world_size, dataset_size,
                        include_optimizer=True) -> dict:
    payload = {
        "model_state_dict": _raw_model(model).state_dict(),
        "step": step,
        "macro_f1": best_f1,
        "n_classes": N_CLASSES,
        "training_config": vars(args),
        "training_state": {
            "epoch": epoch,
            "samples_in_epoch_per_rank": samples_in_epoch,
            "world_size": world_size,
            "batch_per_rank": args.batch,
            "dataset_size": dataset_size,
        },
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def _build_training_dataset(args, train_paths: list[Path]):
    common = dict(
        crop=args.crop, sigma=args.sigma, augment=True,
        patent_aug=args.patent_aug, seed=args.seed,
    )
    primary = KPDataset(train_paths, **common)
    if args.sketchgraphs_raw:
        stream = SketchGraphsKPDataset(
            args.sketchgraphs_raw, canvas=args.sketchgraphs_canvas,
            margin=args.sketchgraphs_margin,
            stroke_width=args.sketchgraphs_stroke_width, **common,
        )
        return ExactMixedDataset(primary, stream, args.mix, args.seed), stream

    if args.labels_archcad:
        secondary_paths = _collect(Path(args.labels_archcad), "train")
        if not secondary_paths:
            raise SystemExit(f"No train npz under {args.labels_archcad}/train")
        rng_mix = random.Random(args.seed)
        n_primary = int(round(args.mix * args.mix_size))
        n_secondary = args.mix_size - n_primary
        mixed = (rng_mix.choices(train_paths, k=n_primary)
                 + rng_mix.choices(secondary_paths, k=n_secondary))
        rng_mix.shuffle(mixed)
        return KPDataset(mixed, **common), None
    return primary, None


def train(args):
    rank, world_size, device = _dist_info(args)
    is_main = rank == 0
    labels = Path(args.labels)
    train_paths = _collect(labels, "train") if args.labels else []
    val_paths = _collect(labels, "validation") if args.labels else []
    if not train_paths and not (args.sketchgraphs_raw and args.mix == 0.0):
        raise SystemExit(f"No train npz under {labels/'train'}")

    secondary_val_root = args.sketchgraphs_val_labels or args.labels_archcad
    secondary_val_paths = (
        _collect(Path(secondary_val_root), "validation")
        if secondary_val_root else []
    )
    rng = random.Random(args.seed)
    rng.shuffle(val_paths)
    rng.shuffle(secondary_val_paths)
    val_subset = val_paths[:args.val_subset]
    secondary_val_subset = secondary_val_paths[:args.secondary_val_subset]

    dataset, stream = _build_training_dataset(args, train_paths)
    sampler = ConstantMemoryDistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        seed=args.seed, shuffle=not args.no_shuffle,
    )
    loader_options = dict(
        dataset=dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, drop_last=False, pin_memory=True,
        persistent_workers=args.workers > 0, worker_init_fn=_worker_init,
    )
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_options)

    if is_main:
        print(f"dataset={len(dataset):,} samples  rank samples={sampler.num_samples:,} "
              f"world={world_size}  device={device}")
        print(f"primary train npz={len(train_paths):,}  primary val={len(val_paths):,} "
              f"(eval on {len(val_subset):,})")
        if stream is not None:
            print(f"SketchGraphs streaming={len(stream):,} sequences from "
                  f"{args.sketchgraphs_raw}; exact mix={args.mix:.0%} cached / "
                  f"{1-args.mix:.0%} SketchGraphs")
        if secondary_val_subset:
            print(f"secondary validation={len(secondary_val_paths):,} "
                  f"(eval on {len(secondary_val_subset):,})")

    model = s2._build_stacked_hourglass().to(device)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload.get("model_state_dict", resume_payload))
    elif args.init_weights:
        state = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        if is_main:
            print(f"fine-tuning from {args.init_weights}")
    else:
        prior_bias = -math.log((1 - args.prior) / args.prior)
        with torch.no_grad():
            for head in (model.out1, model.out2):
                torch.nn.init.normal_(head.weight, std=1e-3)
                torch.nn.init.constant_(head.bias, prior_bias)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and str(device).startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_scaler, init_scale=args.amp_init_scale,
    )
    step = epoch = samples_in_epoch = 0
    best_f1 = -1.0
    if resume_payload is not None:
        state = resume_payload.get("training_state", {})
        if state.get("world_size", world_size) != world_size:
            raise SystemExit("resume requires the same torchrun world size")
        if state.get("batch_per_rank", args.batch) != args.batch:
            raise SystemExit("resume requires the same per-rank batch size")
        if state.get("dataset_size", len(dataset)) != len(dataset):
            raise SystemExit("resume dataset size does not match the current run")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scaler.load_state_dict(resume_payload.get("scaler_state_dict", {}))
        step = int(resume_payload.get("step", 0))
        epoch = int(state.get("epoch", 0))
        samples_in_epoch = int(state.get("samples_in_epoch_per_rank", 0))
        best_f1 = float(resume_payload.get("macro_f1", -1.0))
        if "torch_rng_state" in resume_payload:
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if is_main:
            print(f"resuming {args.resume}: step={step:,}, epoch={epoch}, "
                  f"rank_offset={samples_in_epoch:,}")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[int(device.split(":")[-1])])
    model.train()

    out_path = Path(args.out)
    state_path = (Path(args.state_out) if args.state_out else
                  out_path.with_name(out_path.stem + "_last" + out_path.suffix))
    coverage = None
    if stream is not None:
        coverage_path = (Path(args.coverage_file) if args.coverage_file else
                         out_path.with_suffix(".coverage.i8"))
        coverage = CoverageTracker(
            coverage_path, len(stream), rank, world_size,
            reset=resume_payload is None, source_path=args.sketchgraphs_raw,
        )

    if use_amp and is_main:
        mode = "BF16 autocast" if amp_dtype == torch.bfloat16 else "FP16 autocast + GradScaler"
        print(f"mixed precision: {mode}")
    if args.steps <= 0 and args.epochs <= 0:
        raise SystemExit("set --steps or --epochs")

    t0 = time.time()
    running_loss = 0.0
    running_batches = running_valid = running_seen = 0
    stop = False
    while not stop and (args.epochs <= 0 or epoch < args.epochs):
        sampler.set_epoch(epoch)
        sampler.set_start_index(samples_in_epoch)
        for img, hm, valid, source, source_index, status in loader:
            if coverage is not None:
                coverage.update(source.numpy(), source_index.numpy(), status.numpy())
            img = img.to(device, non_blocking=True)
            hm = hm.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True).bool()
            for amp_attempt in range(args.amp_max_retries + 1):
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp,
                ):
                    logits = model(img)
                    loss = (focal_loss(logits[valid], hm[valid]) if valid.any()
                            else logits.float().sum() * 0.0)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                overflow = use_scaler and scaler.get_scale() < scale_before
                if not overflow:
                    break
                if amp_attempt == args.amp_max_retries:
                    raise RuntimeError(
                        "AMP overflow persisted after retry limit; lower "
                        "--amp-init-scale or disable --amp"
                    )
                if is_main:
                    print(f"  AMP overflow: retrying batch at scale "
                          f"{scaler.get_scale():.0f}")

            batch_size = int(img.shape[0])
            samples_in_epoch += batch_size
            step += 1
            running_loss += float(loss.item())
            running_batches += 1
            running_valid += int(valid.sum().item())
            running_seen += batch_size

            if step % args.log_every == 0:
                totals = torch.tensor(
                    [running_loss, running_batches, running_valid, running_seen],
                    dtype=torch.float64, device=device,
                )
                if world_size > 1:
                    dist.all_reduce(totals)
                if is_main:
                    rate = step / max(time.time() - t0, 1e-9)
                    target = f"/{args.steps}" if args.steps > 0 else ""
                    print(f"step {step:>7}{target} epoch={epoch} "
                          f"loss={totals[0].item()/max(totals[1].item(), 1):.4f} "
                          f"valid={totals[2].item()/max(totals[3].item(), 1):.1%} "
                          f"{rate:.2f} it/s")
                running_loss = 0.0
                running_batches = running_valid = running_seen = 0

            should_validate = bool(val_subset and args.val_every > 0
                                   and step % args.val_every == 0)
            if should_validate:
                _barrier(world_size)
                if is_main:
                    raw = _raw_model(model)
                    metrics = evaluate_f1(
                        raw, val_subset, device, match_radius=args.match_radius,
                    )
                    selection_f1 = metrics["macro_f1"]
                    secondary_f1 = None
                    print(f"  [val@{step}] macro_f1={selection_f1:.3f} "
                          f"end={metrics['f1'][0]:.3f} "
                          f"junc={metrics['f1'][1]:.3f} "
                          f"corner={metrics['f1'][2]:.3f}")
                    if secondary_val_subset:
                        secondary = evaluate_f1(
                            raw, secondary_val_subset, device,
                            match_radius=args.match_radius,
                        )
                        secondary_f1 = secondary["macro_f1"]
                        selection_f1 = 0.5 * (selection_f1 + secondary_f1)
                        print(f"  [secondary@{step}] macro_f1={secondary_f1:.3f}; "
                              f"dual-domain={selection_f1:.3f}")
                    if selection_f1 > best_f1:
                        best_f1 = selection_f1
                        payload = _checkpoint_payload(
                            model, optimizer, scaler, args, step, epoch,
                            samples_in_epoch, best_f1, world_size, len(dataset),
                            include_optimizer=False,
                        )
                        payload["primary_macro_f1"] = metrics["macro_f1"]
                        payload["secondary_macro_f1"] = secondary_f1
                        _atomic_torch_save(payload, out_path)
                        print(f"  saved best ({best_f1:.3f}) -> {out_path}")
                _barrier(world_size)
                model.train()

            if args.save_every > 0 and step % args.save_every == 0:
                _barrier(world_size)
                if coverage is not None:
                    coverage.flush()
                if is_main:
                    _atomic_torch_save(_checkpoint_payload(
                        model, optimizer, scaler, args, step, epoch,
                        samples_in_epoch, best_f1, world_size, len(dataset),
                    ), state_path)
                    print(f"  saved resumable state -> {state_path}")
                _barrier(world_size)

            if args.steps > 0 and step >= args.steps:
                stop = True
                break

        if samples_in_epoch >= sampler.num_samples:
            epoch += 1
            samples_in_epoch = 0
            if coverage is not None:
                report = coverage.summarize(epoch, step)
                if is_main:
                    print("coverage: " + json.dumps(report, sort_keys=True))

    _barrier(world_size)
    if coverage is not None:
        report = coverage.summarize(epoch, step)
    else:
        report = None
    if is_main:
        _atomic_torch_save(_checkpoint_payload(
            model, optimizer, scaler, args, step, epoch, samples_in_epoch,
            best_f1, world_size, len(dataset),
        ), state_path)
        if best_f1 < 0:
            _atomic_torch_save(_checkpoint_payload(
                model, optimizer, scaler, args, step, epoch, samples_in_epoch,
                best_f1, world_size, len(dataset), include_optimizer=False,
            ), out_path)
        print(f"done. best macro_f1={best_f1:.3f} ({step:,} steps, "
              f"{(time.time()-t0)/60:.1f} min)")
    _barrier(world_size)

    incomplete = bool(report and not report["complete"])
    if args.require_full_coverage and incomplete:
        if world_size > 1:
            dist.destroy_process_group()
        raise SystemExit("SketchGraphs coverage is incomplete; see the coverage JSON")
    if world_size > 1:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="output/Drawing2CAD/kp_labels")
    ap.add_argument("--out", default="models/puhachov_d2c.pth")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--epochs", type=int, default=0,
                    help="full dataset passes; use --steps 0 for coverage-driven training")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--prior", type=float, default=0.01,
                    help="focal-loss output prior; sets initial output bias")
    ap.add_argument("--labels-archcad", "--secondary-labels",
                    dest="labels_archcad", default="",
                    help="second label root to blend with --labels (e.g. "
                         "output/ArchCAD/kp_labels); validation stays on --labels")
    ap.add_argument("--mix", type=float, default=0.5,
                    help="fraction of D2C (--labels) samples per mixed epoch")
    ap.add_argument("--mix-size", type=int, default=40000,
                    help="mixed-epoch size (paths sampled with replacement)")
    ap.add_argument("--sketchgraphs-raw", default="",
                    help="official sg_t16_train.npy; enables on-the-fly rendering")
    ap.add_argument("--sketchgraphs-val-labels", default="",
                    help="cached SketchGraphs pilot labels used only for validation")
    ap.add_argument("--sketchgraphs-canvas", type=int, default=512)
    ap.add_argument("--sketchgraphs-margin", type=int, default=24)
    ap.add_argument("--sketchgraphs-stroke-width", type=int, default=2)
    ap.add_argument("--init-weights", default="",
                    help="checkpoint to fine-tune from (skips random init)")
    ap.add_argument("--resume", default="",
                    help="resumable *_last.pth checkpoint")
    ap.add_argument("--state-out", default="",
                    help="periodic full-state checkpoint (default: <out>_last.pth)")
    ap.add_argument("--save-every", type=int, default=2000,
                    help="steps between atomic resumable checkpoints; 0 disables")
    ap.add_argument("--coverage-file", default="",
                    help="int8 SketchGraphs source-status map")
    ap.add_argument("--require-full-coverage", action="store_true",
                    help="fail unless every SketchGraphs source record was attempted")
    ap.add_argument("--patent-aug", type=float, default=0.0,
                    help="P(apply patent-style degradation per sample); 0 = off")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true",
                    help="use CUDA mixed precision")
    ap.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16",
                    help="BF16 is the stable default on RTX 40-series GPUs")
    ap.add_argument("--amp-init-scale", type=float, default=4096.0,
                    help="initial FP16 GradScaler value")
    ap.add_argument("--amp-max-retries", type=int, default=4,
                    help="retry an overflowed batch after scaler backoff")
    ap.add_argument("--val-subset", type=int, default=200)
    ap.add_argument("--secondary-val-subset", type=int, default=0,
                    help="validation samples from --labels-archcad; when >0, "
                         "select checkpoints by mean primary/secondary F1")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--match-radius", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true",
                    help="disable the constant-memory affine epoch permutation")
    args = ap.parse_args()
    if args.resume and args.init_weights:
        ap.error("--resume and --init-weights are mutually exclusive")
    if not 0.0 <= args.mix <= 1.0:
        ap.error("--mix must be between 0 and 1")
    if args.sketchgraphs_raw and args.mix >= 1.0:
        ap.error("streaming SketchGraphs requires --mix < 1")
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
