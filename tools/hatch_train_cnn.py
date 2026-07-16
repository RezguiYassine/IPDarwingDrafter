"""Phase 2 hatch-region CNN trainer.

Trains HatchUNet (frozen ResNet18 encoder + lightweight decoder) on the
polygon ground-truth masks in output/PatentData/hatch_gt.

Loss    : 0.5 × weighted BCE  +  0.5 × Dice
Optim   : Adam, lr=1e-4, cosine LR decay
Val     : full sliding-window IoU on held-out figures (no patch leakage)
Checkpoint: best val-IoU model saved to --out

Single run:
    python -m tools.hatch_train_cnn \
        --gt output/PatentData/hatch_gt \
        --tif-root data/PatentData/ReorganisedData \
        --out models/hatch_unet.pth \
        --epochs 40 --batch 32 --device cuda:0

Grid search (preferred):
    python -m tools.hatch_grid_search
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.hatch_dataset import (HatchPatchDataset, FigureRecord,
                                  load_figures, split_figures)
from tools.hatch_model import HatchUNet

_PATCH  = 512
_STRIDE = 256


# ─── Loss ────────────────────────────────────────────────────────────────────

def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  pos_weight: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    dice = dice_loss(logits, targets)
    return alpha * bce + (1 - alpha) * dice


# ─── Sliding-window validation ───────────────────────────────────────────────

@torch.no_grad()
def sliding_window_probs(model: HatchUNet, gray: np.ndarray,
                          device: torch.device,
                          patch_size: int = _PATCH,
                          stride: int = _STRIDE) -> np.ndarray:
    H, W = gray.shape
    acc   = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - patch_size), stride)) + [max(0, H - patch_size)]
    xs = list(range(0, max(1, W - patch_size), stride)) + [max(0, W - patch_size)]
    model.eval()
    for y0 in dict.fromkeys(ys):
        for x0 in dict.fromkeys(xs):
            patch = gray[y0:y0 + patch_size, x0:x0 + patch_size].astype(np.float32) / 255.0
            ph, pw = patch.shape
            if ph < patch_size or pw < patch_size:
                pad = np.zeros((patch_size, patch_size), np.float32)
                pad[:ph, :pw] = patch
                patch = pad
            t    = torch.from_numpy(patch[None, None]).to(device)
            prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            acc[y0:y0 + ph, x0:x0 + pw]   += prob[:ph, :pw]
            count[y0:y0 + ph, x0:x0 + pw] += 1.0
    return acc / np.maximum(count, 1e-6)


def figure_iou(probs: np.ndarray, mask: np.ndarray,
               threshold: float = 0.5) -> float:
    pred  = (probs >= threshold).astype(np.uint8)
    inter = int((pred & mask).sum())
    union = int((pred | mask).sum())
    return inter / union if union > 0 else 1.0


@torch.no_grad()
def validate(model: HatchUNet, figures: list[FigureRecord],
             device: torch.device,
             patch_size: int = _PATCH,
             stride: int = _STRIDE) -> dict[str, float]:
    model.eval()
    ious, pos_ious = [], []
    for rec in figures:
        rec._ensure_loaded()
        probs = sliding_window_probs(model, rec._gray, device, patch_size, stride)
        iou   = figure_iou(probs, rec._mask)
        ious.append(iou)
        if rec.is_positive:
            pos_ious.append(iou)
    return {
        "iou_all": float(np.mean(ious)),
        "iou_pos": float(np.mean(pos_ious)) if pos_ious else 0.0,
    }


# ─── One epoch ───────────────────────────────────────────────────────────────

def train_one_epoch(model: HatchUNet, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    pos_weight: torch.Tensor,
                    device: torch.device) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = combined_loss(model(imgs), masks, pos_weight)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


# ─── Core training function (called by main() and hatch_grid_search.py) ─────

def run_training(
    *,
    # paths
    gt: str       = "output/PatentData/hatch_gt",
    tif_root: str = "data/PatentData/ReorganisedData",
    out: str      = "models/hatch_unet.pth",
    # hyperparameters
    epochs: int       = 40,
    batch: int        = 32,
    lr: float         = 1e-4,
    patch: int        = _PATCH,
    stride: int       = _STRIDE,
    pos_weight: float = 5.0,
    samples: int      = 2000,
    # infrastructure
    workers: int      = 8,
    val_frac: float   = 0.1,
    test_frac: float  = 0.1,
    seed: int         = 42,
    device: str       = "cuda:0",
    unfreeze_after: Optional[int] = None,
    # pre-split figures (pass from grid search to reuse same split)
    train_figs: Optional[list] = None,
    val_figs:   Optional[list] = None,
    test_figs:  Optional[list] = None,
    verbose: bool = True,
) -> dict:
    """Train one HatchUNet configuration. Returns full history dict for paper.

    Return schema
    -------------
    {
      "config":         {lr, pos_weight, batch, epochs, patch, stride, samples, seed},
      "history":        [{epoch, train_loss, val_iou_all, val_iou_pos, lr, time_s}, ...],
      "best_epoch":     int,
      "best_val_iou_pos": float,
      "best_val_iou_all": float,
      "test_iou_pos":   float,
      "test_iou_all":   float,
      "checkpoint":     str,
    }
    """
    device_ = torch.device(device)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    if train_figs is None:
        if verbose:
            print("Loading ground-truth masks …")
        records = load_figures(gt, tif_root)
        train_figs, val_figs, test_figs = split_figures(
            records, val_frac, test_frac, seed)

    train_ds = HatchPatchDataset(train_figs, patch, samples, augment=True, seed=seed)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=(workers > 0))

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = HatchUNet(freeze_encoder=True).to(device_)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pos_w     = torch.tensor([pos_weight], device=device_)

    if verbose:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable params: {n_tr:,}  |  "
              f"lr={lr:.0e}  pos_weight={pos_weight}  batch={batch}")
        print(f"\n{'epoch':>6}  {'loss':>8}  {'val_iou_all':>12}  "
              f"{'val_iou_pos':>12}  {'lr':>10}  {'time':>6}")
        print("─" * 68)

    # ── Training loop ─────────────────────────────────────────────────────────
    history:    list[dict] = []
    best_iou:   float      = -1.0
    best_epoch: int        = 1
    best_state: dict       = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if unfreeze_after and epoch == unfreeze_after:
            if verbose:
                print(f"  → unfreezing last 2 encoder blocks at epoch {epoch}")
            model.unfreeze_encoder(blocks=2)
            optimizer.add_param_group(
                {"params": list(model.encoder_parameters()), "lr": lr * 0.1})

        loss    = train_one_epoch(model, train_dl, optimizer, pos_w, device_)
        scheduler.step()
        metrics = validate(model, val_figs, device_, patch, stride)
        lr_now  = float(scheduler.get_last_lr()[0])
        elapsed = time.time() - t0

        row = {
            "epoch":       epoch,
            "train_loss":  round(loss, 6),
            "val_iou_all": round(metrics["iou_all"], 6),
            "val_iou_pos": round(metrics["iou_pos"], 6),
            "lr":          lr_now,
            "time_s":      round(elapsed, 1),
        }
        history.append(row)

        improved = ""
        if metrics["iou_pos"] > best_iou:
            best_iou   = metrics["iou_pos"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved   = "  ✓"

        if verbose:
            print(f"{epoch:>6}  {loss:>8.4f}  {metrics['iou_all']:>12.4f}  "
                  f"{metrics['iou_pos']:>12.4f}  {lr_now:>10.2e}  "
                  f"{elapsed:>5.1f}s{improved}")

    # ── Save best checkpoint ──────────────────────────────────────────────────
    best_val_iou_all = history[best_epoch - 1]["val_iou_all"]
    ckpt = {
        "epoch":       best_epoch,
        "model_state": best_state,
        "val_iou_pos": best_iou,
        "val_iou_all": best_val_iou_all,
        "config": {
            "lr": lr, "pos_weight": pos_weight, "batch": batch,
            "epochs": epochs, "patch": patch, "stride": stride,
            "samples": samples, "seed": seed,
        },
    }
    torch.save(ckpt, out)

    # ── Test evaluation (best model) ─────────────────────────────────────────
    model.load_state_dict(best_state)
    test_m = validate(model, test_figs, device_, patch, stride)

    result = {
        "config": {
            "lr": lr, "pos_weight": pos_weight, "batch": batch,
            "epochs": epochs, "patch": patch, "stride": stride,
            "samples": samples, "seed": seed,
        },
        "history":          history,
        "best_epoch":       best_epoch,
        "best_val_iou_pos": round(best_iou, 6),
        "best_val_iou_all": round(best_val_iou_all, 6),
        "test_iou_pos":     round(test_m["iou_pos"], 6),
        "test_iou_all":     round(test_m["iou_all"], 6),
        "checkpoint":       out,
    }

    if verbose:
        print(f"\nBest epoch {best_epoch}  "
              f"val_iou_pos={best_iou:.4f}  val_iou_all={best_val_iou_all:.4f}")
        print(f"Test  iou_pos={test_m['iou_pos']:.4f}  "
              f"iou_all={test_m['iou_all']:.4f}")
        print(f"Checkpoint → {out}")

    return result


# ─── CLI (single run) ────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",           default="output/PatentData/hatch_gt")
    ap.add_argument("--tif-root",     default="data/PatentData/ReorganisedData")
    ap.add_argument("--out",          default="models/hatch_unet.pth")
    ap.add_argument("--history-out",  default=None,
                     help="where to save per-epoch history JSON "
                          "(default: <out>.json)")
    ap.add_argument("--epochs",       type=int,   default=40)
    ap.add_argument("--batch",        type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--patch",        type=int,   default=_PATCH)
    ap.add_argument("--stride",       type=int,   default=_STRIDE)
    ap.add_argument("--pos-weight",   type=float, default=5.0)
    ap.add_argument("--samples",      type=int,   default=2000)
    ap.add_argument("--workers",      type=int,   default=8)
    ap.add_argument("--val-frac",     type=float, default=0.1)
    ap.add_argument("--test-frac",    type=float, default=0.1)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--device",
                     default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--unfreeze-after", type=int, default=None)
    a = ap.parse_args()

    records = load_figures(a.gt, a.tif_root)
    train_figs, val_figs, test_figs = split_figures(
        records, a.val_frac, a.test_frac, a.seed)

    result = run_training(
        gt=a.gt, tif_root=a.tif_root, out=a.out,
        epochs=a.epochs, batch=a.batch, lr=a.lr,
        patch=a.patch, stride=a.stride,
        pos_weight=a.pos_weight, samples=a.samples,
        workers=a.workers, val_frac=a.val_frac, test_frac=a.test_frac,
        seed=a.seed, device=a.device, unfreeze_after=a.unfreeze_after,
        train_figs=train_figs, val_figs=val_figs, test_figs=test_figs,
    )

    hist_path = a.history_out or a.out.replace(".pth", "_history.json")
    json.dump(result, open(hist_path, "w"), indent=2)
    print(f"\nFull history → {hist_path}")


if __name__ == "__main__":
    main()
