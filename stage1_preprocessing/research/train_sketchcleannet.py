"""
train_sketchcleannet.py
=======================
Training script for SketchCleanNet (U-Net sketch cleaning model).

Uses the _UNet architecture defined in stage1_preprocess.py.

Dataset layout (discovered):
  SketchCleanNet_Data/Train Data/      632 clean PNGs (512×512, RGBA)
  SketchCleanNet_Data/Validation Data/ 169 clean PNGs (512×512, RGBA)

Training strategy:
  Clean images are ground-truth targets.
  Noisy inputs are synthesized on-the-fly by a sketch-degradation pipeline
  that simulates scan artefacts: Gaussian noise, stroke broadening, smudges,
  blur, brightness drift, and salt-and-pepper noise.

Loss:  0.7 × L1  +  0.3 × (1 − SSIM)
Optim: Adam, lr=1e-4, cosine annealing to lr=1e-6
Saves: Models/sketchcleannet.pth  (best val loss)
       Models/sketchcleannet_last.pth  (last epoch, for resume)

Usage:
  python train_sketchcleannet.py                          # defaults
  python train_sketchcleannet.py --epochs 150 --batch 8
  python train_sketchcleannet.py --resume               # continue from last checkpoint

Author: Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ─── Import the U-Net architecture from stage1_preprocess ────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from stage1_preprocess import _UNet, ModelNotAvailableError


# ─────────────────────────────────────────────────────────────────────────────
# SSIM loss (structural similarity, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    1 − SSIM averaged over the batch.
    pred, target: (B, 1, H, W) float32 in [0, 1]
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)           # (1, 1, W, W)
    pad = window_size // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=1)

    mu1   = conv(pred)
    mu2   = conv(target)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1  = conv(pred   * pred)   - mu1_sq
    s2  = conv(target * target) - mu2_sq
    s12 = conv(pred   * target) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return 1.0 - ssim_map.mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.7 * F.l1_loss(pred, target) + 0.3 * ssim_loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Sketch degradation — synthesizes noisy inputs from clean targets
# ─────────────────────────────────────────────────────────────────────────────

class SketchDegrader:
    """
    Applies a randomised degradation pipeline to a clean sketch.
    Each call returns a different random degradation, so the model
    sees a different noisy version of every image each epoch.

    All operations work on float32 arrays in [0, 1] with white background.
    """

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    # ── individual degradations ───────────────────────────────────────────────

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        sigma = self.rng.uniform(0.02, 0.12)
        noise = self.rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)

    def _salt_pepper(self, img: np.ndarray) -> np.ndarray:
        density = self.rng.uniform(0.005, 0.03)
        out = img.copy()
        n = int(density * img.size)
        coords = (self.rng.integers(0, img.shape[0], n),
                  self.rng.integers(0, img.shape[1], n))
        values = self.rng.choice([0.0, 1.0], size=n).astype(np.float32)
        out[coords] = values
        return out

    def _blur(self, img: np.ndarray) -> np.ndarray:
        sigma = self.rng.uniform(0.3, 1.8)
        k = max(3, int(sigma * 3) | 1)   # odd kernel
        blurred = cv2.GaussianBlur(img, (k, k), sigma)
        return blurred.astype(np.float32)

    def _stroke_broaden(self, img: np.ndarray) -> np.ndarray:
        """Dilate dark strokes (lower pixel value = darker)."""
        inv = 1.0 - img          # invert: strokes become bright blobs
        k = self.rng.integers(1, 4) * 2 + 1   # 3, 5, or 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        inv_dilated = cv2.dilate((inv * 255).astype(np.uint8), kernel)
        return 1.0 - inv_dilated.astype(np.float32) / 255.0

    def _background_smudge(self, img: np.ndarray) -> np.ndarray:
        """Add random gray blobs that simulate paper texture / scan smudges."""
        out = img.copy()
        n_blobs = self.rng.integers(3, 12)
        H, W = img.shape
        for _ in range(n_blobs):
            cy = self.rng.integers(0, H)
            cx = self.rng.integers(0, W)
            r  = self.rng.integers(5, 40)
            intensity = self.rng.uniform(0.0, 0.25)   # dark smudge
            # create small blob mask
            ys = np.arange(max(0, cy - r), min(H, cy + r))
            xs = np.arange(max(0, cx - r), min(W, cx + r))
            if len(ys) == 0 or len(xs) == 0:
                continue
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
            out[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1][mask] -= intensity
        return np.clip(out, 0.0, 1.0)

    def _brightness_shift(self, img: np.ndarray) -> np.ndarray:
        delta = self.rng.uniform(-0.08, 0.08)
        return np.clip(img + delta, 0.0, 1.0)

    def _speckle(self, img: np.ndarray) -> np.ndarray:
        """Multiplicative noise — more prominent on strokes."""
        noise = self.rng.normal(1.0, 0.05, img.shape).astype(np.float32)
        return np.clip(img * noise, 0.0, 1.0)

    # ── main entry point ──────────────────────────────────────────────────────

    def __call__(self, clean: np.ndarray) -> np.ndarray:
        """
        Apply a randomised combination of degradations.
        clean: float32 [0,1] HxW array (grayscale, white bg).
        Returns: noisy float32 [0,1] HxW array.
        """
        ops = [
            (self._gaussian_noise,   0.90),
            (self._salt_pepper,      0.50),
            (self._blur,             0.60),
            (self._stroke_broaden,   0.40),
            (self._background_smudge,0.35),
            (self._brightness_shift, 0.50),
            (self._speckle,          0.30),
        ]
        noisy = clean.copy()
        for fn, prob in ops:
            if self.rng.random() < prob:
                noisy = fn(noisy)
        return noisy


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SketchDataset(Dataset):
    """
    Loads clean PNG sketches and returns (noisy_tensor, clean_tensor) pairs.

    For training: random crop + random flip + on-the-fly degradation.
    For validation: centre crop, fixed mild degradation (seeded per image).
    """

    CROP = 256   # spatial crop size fed to the network during training

    def __init__(self, folder: Path, mode: str = "train"):
        assert mode in ("train", "val")
        self.mode    = mode
        self.files   = sorted(folder.glob("*.png"),
                              key=lambda p: int(p.stem))
        if not self.files:
            raise FileNotFoundError(f"No PNG files found in {folder}")
        self.degrader = SketchDegrader()

    def __len__(self) -> int:
        return len(self.files)

    def _load_gray(self, path: Path) -> np.ndarray:
        """Load PNG, discard alpha, return float32 HxW in [0, 1]."""
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape
        if H <= self.CROP and W <= self.CROP:
            return img
        y0 = np.random.randint(0, max(1, H - self.CROP))
        x0 = np.random.randint(0, max(1, W - self.CROP))
        return img[y0:y0 + self.CROP, x0:x0 + self.CROP]

    def _centre_crop(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape
        y0 = max(0, (H - self.CROP) // 2)
        x0 = max(0, (W - self.CROP) // 2)
        return img[y0:y0 + self.CROP, x0:x0 + self.CROP]

    def __getitem__(self, idx: int):
        clean = self._load_gray(self.files[idx])

        if self.mode == "train":
            # Spatial augmentation
            clean = self._random_crop(clean)
            if np.random.rand() > 0.5:
                clean = np.fliplr(clean).copy()
            if np.random.rand() > 0.5:
                clean = np.flipud(clean).copy()
            noisy = self.degrader(clean)
        else:
            # Deterministic: centre crop, reproducible degradation
            clean = self._centre_crop(clean)
            rng   = np.random.default_rng(seed=idx)
            noisy = SketchDegrader(rng=rng)(clean)

        # → (1, H, W) float32 tensors
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)
        return noisy_t, clean_t


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Paths ────────────────────────────────────────────────────────────────
    data_root  = Path(args.data)
    train_dir  = data_root / "Train Data"
    val_dir    = data_root / "Validation Data"
    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_path = models_dir / "sketchcleannet.pth"
    last_path = models_dir / "sketchcleannet_last.pth"

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = SketchDataset(train_dir, mode="train")
    val_ds   = SketchDataset(val_dir,   mode="val")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = _UNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Optimiser & schedule ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val    = float("inf")

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} "
              f"(best val loss so far: {best_val:.5f})")

    # ── Mixed precision (AMP) ────────────────────────────────────────────────
    use_amp = (device.type == "cuda") and args.amp
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs "
          f"(batch={args.batch}, lr={args.lr}, amp={use_amp})\n")

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        # ── train epoch ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dl, desc=f"Ep {epoch:03d}/{args.epochs} [train]",
                    leave=False, ncols=90)
        for noisy, clean in pbar:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(noisy)
                loss = combined_loss(pred, clean)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_dl)

        # ── val epoch ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(val_dl,
                                     desc=f"Ep {epoch:03d}/{args.epochs} [val] ",
                                     leave=False, ncols=90):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(noisy)
                    loss = combined_loss(pred, clean)
                val_loss += loss.item()
        val_loss /= len(val_dl)

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"Ep {epoch:03d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={lr_now:.2e}  [{elapsed:.1f}s]")

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        # ── save last checkpoint (for resume) ────────────────────────────────
        torch.save({
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scheduler_state_dict":scheduler.state_dict(),
            "best_val_loss":       best_val,
            "train_loss":          train_loss,
            "val_loss":            val_loss,
        }, last_path)

        # ── save best model ───────────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"  ✓ New best val loss {best_val:.5f} → saved to {best_path}")

    # ── done ─────────────────────────────────────────────────────────────────
    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val:.5f}")
    print(f"Best weights  : {best_path}")
    print(f"Last weights  : {last_path}")
    print(f"\nTo use the model, ensure config.yaml contains:")
    print(f"  sketchcleannet:")
    print(f"    weights: \"{best_path}\"")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SketchCleanNet on the SketchCleanNet_Data dataset."
    )
    parser.add_argument(
        "--data",    type=str,
        default="SketchCleanNet_Data",
        help="Root folder containing 'Train Data' and 'Validation Data' sub-folders "
             "(default: SketchCleanNet_Data)",
    )
    parser.add_argument(
        "--models",  type=str,
        default="Models",
        help="Output folder for .pth checkpoints (default: Models)",
    )
    parser.add_argument(
        "--epochs",  type=int,   default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",   type=int,   default=4,
        help="Batch size (default: 4; reduce if GPU OOM)",
    )
    parser.add_argument(
        "--lr",      type=float, default=1e-4,
        help="Initial learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--workers", type=int,   default=2,
        help="DataLoader worker processes (default: 2; use 0 on Windows if issues)",
    )
    parser.add_argument(
        "--device",  type=str,   default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto — cuda if available, else cpu)",
    )
    parser.add_argument(
        "--amp",     action="store_true", default=True,
        help="Use automatic mixed precision on CUDA (default: on)",
    )
    parser.add_argument(
        "--no-amp",  dest="amp", action="store_false",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--resume",  action="store_true",
        help="Resume training from Models/sketchcleannet_last.pth",
    )

    args = parser.parse_args()
    train(args)
