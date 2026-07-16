"""Sliding-window hatch-region inference on a full patent TIF.

Loads a trained HatchUNet checkpoint, runs 512×512 patches with 50 % overlap
over the full image, averages overlapping predictions, and outputs a binary
mask (PNG) and optionally a probability heatmap.

    python -m tools.hatch_infer \
        --model models/hatch_unet.pth \
        --tif   data/PatentData/ReorganisedData/EP1234B1/EP1234B1_F0001.tif \
        --out   /tmp/EP1234B1_F0001_hatch_mask.png

    # batch mode over a directory
    python -m tools.hatch_infer \
        --model models/hatch_unet.pth \
        --tif-dir data/PatentData/ReorganisedData/EP1234B1 \
        --out-dir /tmp/hatch_masks
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from tools.hatch_model import HatchUNet

_PATCH  = 512
_STRIDE = 256


# ─── Core inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def infer_tif(
    model: HatchUNet,
    tif_path: str,
    device: torch.device,
    patch_size: int = _PATCH,
    stride: int = _STRIDE,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prob_map, binary_mask) both with same shape as the input TIF.

    prob_map   : float32 [0, 1] per pixel
    binary_mask: uint8   {0, 255} per pixel
    """
    gray = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(tif_path)
    H, W = gray.shape

    acc   = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)

    ys = list(range(0, max(1, H - patch_size), stride)) + [max(0, H - patch_size)]
    xs = list(range(0, max(1, W - patch_size), stride)) + [max(0, W - patch_size)]

    model.eval()
    for y0 in dict.fromkeys(ys):
        for x0 in dict.fromkeys(xs):
            y1, x1 = y0 + patch_size, x0 + patch_size
            patch = gray[y0:y1, x0:x1].astype(np.float32) / 255.0
            ph, pw = patch.shape
            if ph < patch_size or pw < patch_size:
                pad = np.zeros((patch_size, patch_size), np.float32)
                pad[:ph, :pw] = patch
                patch = pad
            t = torch.from_numpy(patch[None, None]).to(device)
            prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            acc[y0:y0 + ph, x0:x0 + pw]   += prob[:ph, :pw]
            count[y0:y0 + ph, x0:x0 + pw] += 1.0

    prob_map = acc / np.maximum(count, 1e-6)
    binary   = ((prob_map >= threshold) * 255).astype(np.uint8)
    return prob_map, binary


# ─── Visualisation helper ────────────────────────────────────────────────────

def overlay_mask(gray: np.ndarray, binary: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Return RGB image with hatch regions highlighted in orange."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    orange = np.zeros_like(rgb); orange[:, :] = (0, 120, 255)  # BGR
    mask_bool = binary > 0
    rgb[mask_bool] = (rgb[mask_bool] * (1 - alpha) + orange[mask_bool] * alpha).astype(np.uint8)
    return rgb


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str, device: torch.device) -> HatchUNet:
    ck = torch.load(ckpt_path, map_location=device)
    model = HatchUNet(freeze_encoder=False).to(device)  # load all weights
    model.load_state_dict(ck["model_state"])
    model.eval()
    val_iou = ck.get("val_iou_pos", "?")
    print(f"loaded {ckpt_path}  (val_iou_pos={val_iou})")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     required=True, help="path to .pth checkpoint")
    ap.add_argument("--tif",       default=None,  help="single TIF to process")
    ap.add_argument("--tif-dir",   default=None,  help="directory of TIFs (batch mode)")
    ap.add_argument("--out",       default=None,  help="output mask PNG (single mode)")
    ap.add_argument("--out-dir",   default=None,  help="output directory (batch mode)")
    ap.add_argument("--threshold", type=float, default=0.7,
                     help="operating point validated on the v2 held-out test "
                          "set (pos IoU 0.814, all negatives <0.5%% FP)")
    ap.add_argument("--patch",     type=int,   default=_PATCH)
    ap.add_argument("--stride",    type=int,   default=_STRIDE)
    ap.add_argument("--overlay",   action="store_true",
                     help="also write an RGB overlay image (orange = hatch)")
    ap.add_argument("--probs",     action="store_true",
                     help="also write a float32 probability PNG (scaled ×255)")
    ap.add_argument("--device",    default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    device = torch.device(a.device)
    model  = _load_model(a.model, device)

    # Build list of (tif_path, out_path) pairs
    jobs: list[tuple[str, str]] = []
    if a.tif:
        out = a.out or a.tif.replace(".tif", "_hatch_mask.png")
        jobs.append((a.tif, out))
    elif a.tif_dir:
        out_dir = a.out_dir or a.tif_dir
        os.makedirs(out_dir, exist_ok=True)
        for tif in sorted(glob.glob(os.path.join(a.tif_dir, "*.tif"))):
            name = os.path.basename(tif).replace(".tif", "_hatch_mask.png")
            jobs.append((tif, os.path.join(out_dir, name)))
    else:
        ap.error("provide --tif or --tif-dir")

    for tif_path, out_path in jobs:
        print(f"  {tif_path} …", end=" ", flush=True)
        prob_map, binary = infer_tif(
            model, tif_path, device, a.patch, a.stride, a.threshold)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, binary)

        if a.probs:
            prob_u8 = (prob_map * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(out_path.replace("_mask.png", "_probs.png"), prob_u8)

        if a.overlay:
            gray = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
            ov = overlay_mask(gray, binary)
            cv2.imwrite(out_path.replace("_mask.png", "_overlay.png"), ov)

        px = int(binary.sum() // 255)
        frac = px / binary.size
        print(f"done  hatch={frac:.1%}  → {out_path}")


if __name__ == "__main__":
    main()
