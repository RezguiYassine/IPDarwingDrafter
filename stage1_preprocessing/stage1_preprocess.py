"""
stage1_preprocess.py
====================
AP3 Vectorization Pipeline — Stage 1: Preprocessing

Transforms a normalized grayscale PNG (output of Stage 0) into:
  - <sketch_id>_cleaned.png  : cleaned raster (SketchCleanNet or classical fallback)
  - <sketch_id>_skeleton.png : 1px binary skeleton (Zhang-Suen thinning)

Outputs one confidence value per sketch:
  skeleton_quality : float in [0, 1]  (1.0 = clean, <threshold = flag for review)

SketchCleanNet weights are optional. If the weights path in config is empty or the
file does not exist, the stage falls back to classical cleaning (Otsu + morphological
opening). When weights become available, set `sketchcleannet.weights` in config.yaml
and the rest of the pipeline is unaffected.

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label

logger = logging.getLogger(__name__)


# ─── Output contract ────────────────────────────────────────────────────────

@dataclass
class Stage1Result:
    sketch_id: str
    cleaned_path: Path          # cleaned raster PNG
    skeleton_path: Path         # 1px binary skeleton PNG
    skeleton_quality: float     # confidence signal in [0, 1]
    processing_time_s: float
    model_used: str             # "sketchcleannet" | "classical"
    flagged: bool               # True if skeleton_quality < threshold


# ─── SketchCleanNet model wrapper ────────────────────────────────────────────

class SketchCleanNet:
    """
    Wrapper around SketchCleanNet (U-Net for sketch cleanup).

    If weights are provided and PyTorch is available, runs full DL inference.
    Otherwise raises ModelNotAvailableError so the caller can fall back gracefully.

    Tiling strategy for high-DPI images:
      - Split image into overlapping patches of `tile_size` with `overlap` margin
      - Run inference on each patch
      - Blend overlapping regions with a linear weight ramp (avoids seam artefacts)
    """

    TILE_SIZE = 512    # pixels — patch size fed to network
    OVERLAP   = 64     # pixels — overlap between adjacent patches

    def __init__(self, weights_path: Optional[str], device: str = "cuda"):
        self._model   = None
        self._device  = device
        self._ready   = False

        if not weights_path:
            raise ModelNotAvailableError("No weights path specified in config.")

        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise ModelNotAvailableError(
                f"SketchCleanNet weights not found at: {weights_file}\n"
                f"  → Download pretrained weights and set sketchcleannet.weights "
                f"in config.yaml\n"
                f"  → Repository: https://github.com/BardOfCodes/SketchCleanNet"
            )

        self._load_model(weights_file, device)

    def _load_model(self, weights_file: Path, device: str) -> None:
        try:
            import torch
            import torch.nn as nn

            model = _UNet()
            state = torch.load(weights_file, map_location=device)
            # Handle both raw state_dict and checkpoint dicts
            state_dict = state.get("model_state_dict", state.get("state_dict", state))
            model.load_state_dict(state_dict)
            model.eval()
            model.to(device)
            self._model  = model
            self._device = device
            self._ready  = True
            logger.info(f"SketchCleanNet loaded from {weights_file} on {device}")
        except ImportError:
            raise ModelNotAvailableError("PyTorch is not installed.")
        except Exception as exc:
            raise ModelNotAvailableError(f"Failed to load SketchCleanNet: {exc}")

    def clean(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Run SketchCleanNet on a grayscale uint8 image.
        Returns a cleaned grayscale uint8 image of the same shape.
        """
        import torch

        H, W = image_gray.shape
        # Pad image so it tiles evenly
        pad_h = (self.TILE_SIZE - H % self.TILE_SIZE) % self.TILE_SIZE
        pad_w = (self.TILE_SIZE - W % self.TILE_SIZE) % self.TILE_SIZE
        padded = np.pad(image_gray, ((0, pad_h), (0, pad_w)), mode="reflect")

        output  = np.zeros_like(padded, dtype=np.float32)
        weights = np.zeros_like(padded, dtype=np.float32)

        step = self.TILE_SIZE - self.OVERLAP
        pH, pW = padded.shape

        # Build a 2D weight ramp: edges blend smoothly, centre has full weight
        ramp  = _cosine_weight_ramp(self.TILE_SIZE)

        with torch.no_grad():
            for y in range(0, pH - self.TILE_SIZE + 1, step):
                for x in range(0, pW - self.TILE_SIZE + 1, step):
                    patch = padded[y:y + self.TILE_SIZE, x:x + self.TILE_SIZE]
                    tensor = ((torch.from_numpy(patch).float() / 255.0)
                              .unsqueeze(0).unsqueeze(0)
                              .to(self._device))
                    pred = self._model(tensor).squeeze().cpu().numpy()
                    output [y:y + self.TILE_SIZE, x:x + self.TILE_SIZE] += pred * ramp
                    weights[y:y + self.TILE_SIZE, x:x + self.TILE_SIZE] += ramp

        blended = np.divide(output, weights, where=weights > 0)
        result  = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        return result[:H, :W]


class ModelNotAvailableError(Exception):
    pass


# ─── Minimal U-Net definition (matches SketchCleanNet architecture) ──────────

def _UNet():
    """
    Reconstruct the SketchCleanNet U-Net architecture.
    Input : (B, 1, H, W) float32 in [0, 1]
    Output: (B, 1, H, W) float32 in [0, 1]

    Architecture follows the original paper:
      Manda et al., "SketchCleanNet", Computers & Graphics 107 (2022), pp. 73–83.
    """
    try:
        import torch.nn as nn

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )

        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = _block(1,   64);  self.pool1 = nn.MaxPool2d(2)
                self.enc2 = _block(64,  128); self.pool2 = nn.MaxPool2d(2)
                self.enc3 = _block(128, 256); self.pool3 = nn.MaxPool2d(2)
                self.enc4 = _block(256, 512); self.pool4 = nn.MaxPool2d(2)
                self.bottleneck = _block(512, 1024)
                self.up4   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
                self.dec4  = _block(1024, 512)
                self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3  = _block(512, 256)
                self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2  = _block(256, 128)
                self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1  = _block(128, 64)
                self.out   = nn.Conv2d(64, 1, 1)

            def forward(self, x):
                import torch
                e1 = self.enc1(x);    e2 = self.enc2(self.pool1(e1))
                e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
                b  = self.bottleneck(self.pool4(e4))
                d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                return torch.sigmoid(self.out(d1))

        return UNet()
    except ImportError:
        raise ModelNotAvailableError("PyTorch not installed — cannot build U-Net.")


def _cosine_weight_ramp(size: int) -> np.ndarray:
    """2D cosine window for smooth patch blending."""
    ramp_1d = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, size))
    return np.outer(ramp_1d, ramp_1d).astype(np.float32)


# ─── Classical fallback cleaning ─────────────────────────────────────────────

def _classical_clean(image_gray: np.ndarray, config: dict) -> np.ndarray:
    """
    Classical cleaning pipeline used when SketchCleanNet weights are unavailable.
    Mirrors the three stages of sketch_preprocessor.py:

      Stage A — Binarization   : Otsu + adaptive blending
      Stage B — Noise removal  : morphological opening + CC filter
      Stage C — Normalization  : ensure clean binary output

    Parameters are read from config["classical"] so they remain tunable.
    """
    cfg = config.get("classical", {})

    # ── Stage A: Binarization ────────────────────────────────────────────────
    blur_size = cfg.get("blur_kernel", 3)
    if blur_size > 1:
        blurred = cv2.GaussianBlur(image_gray, (blur_size, blur_size), 0)
    else:
        blurred = image_gray

    # Global Otsu
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Adaptive (for uneven illumination)
    block = cfg.get("adaptive_block", 35)
    C     = cfg.get("adaptive_C", 10)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, C
    )

    # Blend: weighted combination of Otsu and adaptive
    alpha = cfg.get("blend_alpha", 0.6)   # weight for Otsu
    blended = cv2.addWeighted(otsu, alpha, adaptive, 1 - alpha, 0)
    _, binary = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)

    # ── Stage B: Noise removal ───────────────────────────────────────────────
    # Morphological opening: remove isolated salt-and-pepper noise
    morph_k = cfg.get("morph_kernel", 2)
    kernel   = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_k, morph_k)
    )
    cleaned  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Connected component filter: remove tiny blobs
    min_cc_size = cfg.get("min_cc_size", 30)  # pixels
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    filtered = np.zeros_like(cleaned)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cc_size:
            filtered[labels == lbl] = 255

    return filtered


# ─── Skeleton quality metric ──────────────────────────────────────────────────

def _compute_skeleton_quality(skeleton: np.ndarray) -> float:
    """
    Estimate skeleton quality as a score in [0, 1].

    Strategy:
      1. Label every connected component of foreground pixels.
      2. For each component, compute the ratio:
             (component area) / (component bounding-box area)
         A perfect 1px skeleton has ratio ≈ component_length / bbox_area → low.
         Thick residual blobs have ratio → 1.0 (filled bbox).
      3. Quality = fraction of foreground pixels that belong to
         "thin" components (ratio < thickness_threshold).

    Score = 1.0 means perfectly thin skeleton.
    Score < threshold in config → flag for manual review.
    """
    if skeleton.sum() == 0:
        return 0.0   # empty skeleton — definitely flag

    binary = (skeleton > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    thin_pixels = 0
    total_pixels = int(binary.sum())
    THICKNESS_RATIO = 0.35   # components below this are considered "thin"

    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        w    = stats[lbl, cv2.CC_STAT_WIDTH]
        h    = stats[lbl, cv2.CC_STAT_HEIGHT]
        bbox_area = max(w * h, 1)
        ratio = area / bbox_area
        if ratio < THICKNESS_RATIO:
            thin_pixels += area

    return thin_pixels / total_pixels


# ─── Thinning ─────────────────────────────────────────────────────────────────

def _thin_to_skeleton(binary_image: np.ndarray) -> np.ndarray:
    """
    Apply Zhang-Suen thinning via scikit-image skeletonize.

    Input : binary uint8 (0 / 255) — cleaned raster
    Output: binary uint8 (0 / 255) — 1px-wide skeleton
    """
    # skeletonize expects a boolean array with True = foreground
    bool_img  = binary_image > 0
    skeleton  = skeletonize(bool_img)
    return (skeleton * 255).astype(np.uint8)


# ─── Public stage function ───────────────────────────────────────────────────

def run(
    input_path: Path,
    output_dir: Path,
    sketch_id: str,
    config: dict,
    model: Optional[SketchCleanNet] = None,
) -> Stage1Result:
    """
    Run Stage 1 preprocessing on a single sketch.

    Parameters
    ----------
    input_path : Path
        Normalized grayscale PNG from Stage 0.
    output_dir : Path
        Root output directory. Stage 1 writes to output_dir/cleaned/.
    sketch_id : str
        Unique identifier for this sketch (used in filenames and logs).
    config : dict
        Parsed config.yaml content.
    model : SketchCleanNet | None
        Pre-loaded model instance (shared across all sketches in the batch).
        Pass None to always use the classical fallback.

    Returns
    -------
    Stage1Result
        Paths to outputs and confidence signal.
    """
    t_start = time.perf_counter()

    cleaned_dir = output_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path  = cleaned_dir / f"{sketch_id}_cleaned.png"
    skeleton_path = cleaned_dir / f"{sketch_id}_skeleton.png"

    # ── Load input image ─────────────────────────────────────────────────────
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read input image: {input_path}")

    logger.info(f"[{sketch_id}] Stage 1 — input {image.shape[1]}×{image.shape[0]}px")

    # ── Clean ─────────────────────────────────────────────────────────────────
    if model is not None:
        try:
            cleaned_img = model.clean(image)
            model_used  = "sketchcleannet"
            logger.info(f"[{sketch_id}] SketchCleanNet inference complete")
        except Exception as exc:
            logger.warning(
                f"[{sketch_id}] SketchCleanNet failed ({exc}), "
                f"falling back to classical cleaning"
            )
            cleaned_img = _classical_clean(image, config)
            model_used  = "classical_fallback"
    else:
        cleaned_img = _classical_clean(image, config)
        model_used  = "classical"
        logger.info(f"[{sketch_id}] Classical cleaning applied (no model loaded)")

    # ── Save cleaned raster ───────────────────────────────────────────────────
    cv2.imwrite(str(cleaned_path), cleaned_img)

    # ── Thin to 1px skeleton ──────────────────────────────────────────────────
    skeleton = _thin_to_skeleton(cleaned_img)
    cv2.imwrite(str(skeleton_path), skeleton)

    # ── Compute quality signal ────────────────────────────────────────────────
    quality   = _compute_skeleton_quality(skeleton)
    threshold = config.get("stage1", {}).get("quality_threshold", 0.70)
    flagged   = quality < threshold

    elapsed = time.perf_counter() - t_start

    if flagged:
        logger.warning(
            f"[{sketch_id}] FLAGGED — skeleton quality {quality:.3f} < "
            f"threshold {threshold:.2f}"
        )
    else:
        logger.info(
            f"[{sketch_id}] Stage 1 done in {elapsed:.2f}s — "
            f"quality={quality:.3f} model={model_used}"
        )

    return Stage1Result(
        sketch_id        = sketch_id,
        cleaned_path     = cleaned_path,
        skeleton_path    = skeleton_path,
        skeleton_quality = quality,
        processing_time_s= elapsed,
        model_used       = model_used,
        flagged          = flagged,
    )


# ─── Model loader (called once at batch start) ────────────────────────────────

def load_model(config: dict) -> Optional[SketchCleanNet]:
    """
    Attempt to load SketchCleanNet. Returns None if weights are not available.
    Called once per batch run — the returned instance is reused across all sketches.
    """
    weights_path = config.get("sketchcleannet", {}).get("weights", "")
    device       = config.get("sketchcleannet", {}).get("device", "cuda")

    try:
        model = SketchCleanNet(weights_path=weights_path, device=device)
        logger.info(f"SketchCleanNet ready on device={device}")
        return model
    except ModelNotAvailableError as exc:
        logger.warning(
            f"SketchCleanNet not available: {exc}\n"
            f"  → Classical cleaning will be used until weights are added."
        )
        return None


# ─── CLI for standalone testing ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 1 — Preprocessing: clean a single sketch image."
    )
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser.add_argument("input",   type=Path, help="Input grayscale PNG")
    parser.add_argument("--output",type=Path, default=PROJECT_ROOT / "output",
                        help="Output root directory (default: <project>/output)")
    parser.add_argument("--config",type=Path, default=PROJECT_ROOT / "config.yaml",
                        help="Pipeline config file (default: <project>/config.yaml)")
    parser.add_argument("--id",    type=str,  default=None,
                        help="Sketch ID (default: input filename stem)")
    args = parser.parse_args()

    # Load config
    cfg = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    sketch_id = args.id or args.input.stem

    # Attempt to load model
    model = load_model(cfg)

    # Run stage
    result = run(
        input_path = args.input,
        output_dir = args.output,
        sketch_id  = sketch_id,
        config     = cfg,
        model      = model,
    )

    print(f"\n{'─'*50}")
    print(f"  Sketch ID       : {result.sketch_id}")
    print(f"  Model used      : {result.model_used}")
    print(f"  Cleaned PNG     : {result.cleaned_path}")
    print(f"  Skeleton PNG    : {result.skeleton_path}")
    print(f"  Quality score   : {result.skeleton_quality:.3f}")
    print(f"  Flagged         : {'YES ⚠' if result.flagged else 'no'}")
    print(f"  Processing time : {result.processing_time_s:.2f}s")
    print(f"{'─'*50}")
