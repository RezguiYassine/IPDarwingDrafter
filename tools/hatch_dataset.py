"""Data loader for Phase 2 hatch-region CNN training.

Reads hatch_gt polygon masks, rasterizes them to pixel masks at full-TIF
resolution, and yields (patch_img, patch_mask) pairs for training.

Key design decisions:
  • Split by figure (patent+sketch), never by sub-drawing — no figure leaks
    across train/val/test.
  • Biased patch sampling: for positive figures, 70 % of patches are centred
    on a hatch pixel (hard-positive mining); 30 % are random within the valid
    labeled region.
  • Negative figures contribute random patches from their whole image.
  • Valid region: for sub-drawing masks (crop_box ≠ full image) only the
    crop_box area is treated as reliably labeled; patches are constrained
    to stay inside it.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_DEFAULT_GT      = "output/PatentData/hatch_gt"
_DEFAULT_TIF_ROOT = "data/PatentData/ReorganisedData"
_PATCH           = 512
_HATCH_BIAS      = 0.7   # probability of centring a patch on a hatch pixel


# ─── TIF lookup ──────────────────────────────────────────────────────────────

def _find_tif(patent: str, sketch: str, tif_root: str) -> Optional[str]:
    candidates = [
        os.path.join(tif_root, patent, f"{patent}_{sketch}.tif"),
        os.path.join("latest_run", patent, f"{sketch}_original.tif"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ─── Figure record ───────────────────────────────────────────────────────────

@dataclass
class FigureRecord:
    patent: str
    sketch: str
    tif_path: str
    # Rasterized at full-TIF resolution (loaded lazily)
    _gray:        Optional[np.ndarray] = field(default=None, repr=False)
    _mask:        Optional[np.ndarray] = field(default=None, repr=False)
    _valid:       Optional[np.ndarray] = field(default=None, repr=False)
    _hatch_pts:   Optional[np.ndarray] = field(default=None, repr=False)  # (N, 2) yx
    is_positive:  bool = False

    # ── Lazy load ────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._gray is not None:
            return
        gray = cv2.imread(self.tif_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(self.tif_path)
        H, W = gray.shape
        self._gray  = gray
        self._mask  = np.zeros((H, W), np.uint8)
        self._valid = np.zeros((H, W), np.uint8)

    def add_subdrawing(self, crop_box, polygons) -> None:
        """Rasterize one sub-drawing's polygons into the full-TIF mask."""
        self._ensure_loaded()
        H, W = self._gray.shape
        x0, y0, x1, y1 = (int(v) for v in crop_box)
        x1, y1 = min(x1, W), min(y1, H)
        self._valid[y0:y1, x0:x1] = 1
        for poly in polygons:
            if len(poly) < 3:
                continue
            pts = np.array([[p[0], p[1]] for p in poly],
                           dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(self._mask, [pts], 1)
        if polygons:
            self.is_positive = True

    def finalise(self) -> None:
        """Cache hatch pixel coordinates for biased sampling."""
        self._ensure_loaded()
        ys, xs = np.nonzero(self._mask)
        self._hatch_pts = np.column_stack([ys, xs]) if len(ys) else np.empty((0, 2), int)

    # ── Patch extraction ─────────────────────────────────────────────────────

    def sample_patch(self, patch_size: int = _PATCH,
                     rng: random.Random = random) -> tuple[np.ndarray, np.ndarray]:
        """Return (img_patch, mask_patch) arrays of shape (patch_size, patch_size)."""
        self._ensure_loaded()
        H, W = self._gray.shape

        if (self.is_positive
                and len(self._hatch_pts) > 0
                and rng.random() < _HATCH_BIAS):
            # Centre on a random hatch pixel
            pt = self._hatch_pts[rng.randrange(len(self._hatch_pts))]
            cy, cx = int(pt[0]), int(pt[1])
        else:
            # Random position inside the valid region, with a small margin
            ys_v, xs_v = np.nonzero(self._valid)
            if len(ys_v) == 0:
                cy = rng.randint(patch_size // 2, max(H - patch_size // 2, patch_size // 2 + 1))
                cx = rng.randint(patch_size // 2, max(W - patch_size // 2, patch_size // 2 + 1))
            else:
                idx = rng.randrange(len(ys_v))
                cy, cx = int(ys_v[idx]), int(xs_v[idx])

        half = patch_size // 2
        y0 = max(0, min(cy - half, H - patch_size))
        x0 = max(0, min(cx - half, W - patch_size))
        y1, x1 = y0 + patch_size, x0 + patch_size

        img_p  = self._gray[y0:y1, x0:x1].astype(np.float32) / 255.0
        mask_p = self._mask[y0:y1, x0:x1].astype(np.float32)
        return img_p, mask_p


# ─── Figure list builder ─────────────────────────────────────────────────────

def load_figures(gt_dir: str = _DEFAULT_GT,
                 tif_root: str = _DEFAULT_TIF_ROOT) -> list[FigureRecord]:
    """Load and group all reviewed mask files into FigureRecord objects."""
    by_fig: dict[tuple[str, str], FigureRecord] = {}

    for path in sorted(glob.glob(os.path.join(gt_dir, "*_mask.json"))):
        d = json.load(open(path))
        if not d.get("reviewed"):
            continue
        pat, sk = d["patent"], d["sketch"]
        cb = d.get("crop_box")
        if cb is None:
            continue   # legacy pre-split file without crop_box
        key = (pat, sk)
        if key not in by_fig:
            tif = _find_tif(pat, sk, tif_root)
            if tif is None:
                print(f"warning: TIF not found for {pat}__{sk}, skipping")
                continue
            by_fig[key] = FigureRecord(patent=pat, sketch=sk, tif_path=tif)
        by_fig[key].add_subdrawing(cb, d.get("polygons", []))

    records = list(by_fig.values())
    for r in records:
        r.finalise()
    return records


# ─── Train / val / test split ────────────────────────────────────────────────

def split_figures(
    records: list[FigureRecord],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[FigureRecord], list[FigureRecord], list[FigureRecord]]:
    """Stratified by-figure split (keeps positive/negative ratio in each fold)."""
    rng = random.Random(seed)
    pos = [r for r in records if r.is_positive]
    neg = [r for r in records if not r.is_positive]

    def _split(lst):
        lst = lst[:]
        rng.shuffle(lst)
        n_test = max(1, round(len(lst) * test_frac))
        n_val  = max(1, round(len(lst) * val_frac))
        return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]

    tr_p, va_p, te_p = _split(pos)
    tr_n, va_n, te_n = _split(neg)

    train = tr_p + tr_n
    val   = va_p + va_n
    test  = te_p + te_n
    rng.shuffle(train)

    print(f"split — train: {len(train)} ({sum(r.is_positive for r in train)} pos)  "
          f"val: {len(val)} ({sum(r.is_positive for r in val)} pos)  "
          f"test: {len(test)} ({sum(r.is_positive for r in test)} pos)")
    return train, val, test


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class HatchPatchDataset(Dataset):
    """Yields (img_tensor, mask_tensor) patches for training.

    img_tensor : (1, patch_size, patch_size)  float32 in [0, 1]
    mask_tensor: (1, patch_size, patch_size)  float32 in {0, 1}
    """

    def __init__(
        self,
        figures: list[FigureRecord],
        patch_size: int = _PATCH,
        samples_per_epoch: int = 1000,
        augment: bool = True,
        seed: int = 0,
    ):
        self.figures   = figures
        self.patch_size = patch_size
        self.n         = samples_per_epoch
        self.augment   = augment
        self._rng      = random.Random(seed)
        self._pos      = [r for r in figures if r.is_positive]
        self._neg      = [r for r in figures if not r.is_positive]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 50 / 50 balance between positive and negative figures
        if self._pos and self._neg:
            pool = self._pos if self._rng.random() < 0.5 else self._neg
        elif self._pos:
            pool = self._pos
        else:
            pool = self._neg

        rec = self._rng.choice(pool)
        img, mask = rec.sample_patch(self.patch_size, self._rng)

        if self.augment:
            img, mask = _augment(img, mask, self._rng)

        img_t  = torch.from_numpy(img[None])   # (1, H, W)
        mask_t = torch.from_numpy(mask[None])  # (1, H, W)
        return img_t, mask_t


# ─── Augmentation ────────────────────────────────────────────────────────────

def _augment(
    img: np.ndarray, mask: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Flip, 90° rotate, and brightness jitter (lossless for line art)."""
    # Horizontal flip
    if rng.random() < 0.5:
        img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
    # Vertical flip
    if rng.random() < 0.5:
        img, mask = img[::-1].copy(), mask[::-1].copy()
    # 90° rotations (k=0,1,2,3)
    k = rng.randint(0, 3)
    if k:
        img  = np.rot90(img, k).copy()
        mask = np.rot90(mask, k).copy()
    # Brightness / contrast jitter (skip for binary line art if mostly 0/1)
    if rng.random() < 0.4:
        alpha = rng.uniform(0.85, 1.15)   # contrast
        beta  = rng.uniform(-0.05, 0.05)  # brightness
        img = (img * alpha + beta).clip(0, 1).astype(np.float32)
    return img, mask
