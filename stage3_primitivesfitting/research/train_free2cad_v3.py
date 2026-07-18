"""
train_free2cad_v3.py
====================
AP3 Vectorization Pipeline — Free2CAD Training (v3, per-edge architecture)

This is a ground-up rewrite of the trainer following Priority 1 of the
Free2CAD investigation roadmap: drop the seq2seq Transformer (which never
matched the per-edge inference distribution) and replace it with an
encoder-only classifier+regressor that operates directly on a single edge.

ARCHITECTURE
------------
  Input  : (B, max_pts, 2)   single stroke, padded with -1 sentinels
           + (B, max_pts)    boolean mask, True = real point
                ↓
  PerPointProjection : Linear(2 → d_model) + sinusoidal positional embedding
                ↓
  StrokeEncoder      : Transformer encoder, n_enc_layers, d_model, n_heads
                       (self-attention with padding mask)
                ↓
  Masked mean-pool over real points → (B, d_model)
                ↓
  TypeHead   : Linear(d_model → 4)    softmax over LINE/ARC/CIRCLE/POLYLINE
  ParamHead  : Linear(d_model → 6)    L1-regression of geometric parameters

~1.5M parameters (vs 7.4M for the v2 seq2seq model).

INPUT DATA
----------
Reads per-edge JSON files produced by generate_sketches_v3.py:

    {
      "stroke":  [[x0,y0], [x1,y1], ...],
      "command": {"type": "LINE",   "start": [x,y], "end": [x,y]}
               | {"type": "ARC",    "center": [x,y], "radius": r,
                                    "start_angle": deg, "end_angle": deg}
               | {"type": "CIRCLE", "center": [x,y], "radius": r}
               | {"type": "POLYLINE"}
    }

LOSS
----
  type_loss  : class-weighted CrossEntropy over 4 classes
  param_loss : L1, gated on {LINE, ARC, CIRCLE} only
               (POLYLINE has no canonical params, so its param target is zero
                and contributes nothing to the param-loss numerator)
  total      : type_loss + 0.5 * param_loss

CHECKPOINT FORMAT
-----------------
  {
    "model_state_dict": ...,
    "epoch":            int,
    "val_loss":         float,
    "version":          3,                   # v3 marker for the inference wrapper
    "architecture":     "encoder_only",
    "config": {
        "max_pts":       int,
        "n_cmd_types":   4,
        "d_model":       int,
        "n_heads":       int,
        "n_enc_layers":  int,
        "dropout":       float,
    },
    "cmd_types":     {"LINE":0, "ARC":1, "CIRCLE":2, "POLYLINE":3},
    "class_weights": [w_LINE, w_ARC, w_CIRCLE, w_POLYLINE],
    "metrics":       {... per-class P/R/F1, param_L1, pred_entropy ...},
  }

The inference wrapper (`stage3_primitive_fit.py`) needs a small patch to
recognise `version: 3` and build the encoder-only model — see the note
at the bottom of this file.

USAGE
-----
  # Train from scratch
  python train_free2cad_v3.py \\
      --data_dir   data/free2cad_training_v3 \\
      --output_dir weights/free2cad \\
      --epochs     150 \\
      --batch_size 64 \\
      --lr         3e-4 \\
      --device     cuda

  # Resume
  python train_free2cad_v3.py \\
      --data_dir   data/free2cad_training_v3 \\
      --output_dir weights/free2cad \\
      --resume     weights/free2cad/free2cad_v3_latest.pth

  # Monitor
  tensorboard --logdir weights/free2cad/tb_logs_v3

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import json
import math
import os
import time
import argparse
from pathlib import Path

import numpy as np


# ─── Vocabulary (v3: no END, no BOS) ─────────────────────────────────────────

CMD_TYPES   = {"LINE": 0, "ARC": 1, "CIRCLE": 2, "POLYLINE": 3}
N_CMD_TYPES = len(CMD_TYPES)
N_PARAMS    = 6

# Classes that contribute to the parameter regression loss.
# POLYLINE is excluded because it has no canonical parameter form.
PARAM_CLASSES = {CMD_TYPES["LINE"], CMD_TYPES["ARC"], CMD_TYPES["CIRCLE"]}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Free2CAD v3 — per-edge classifier+regressor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="weights/free2cad")
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device",       type=str,   default="cuda")
    p.add_argument("--max_pts",      type=int,   default=32)
    p.add_argument("--arc_encoding", choices=("three_point", "center_radius_angles"),
                   default="three_point",
                   help="bounded three-point targets avoid shallow-arc outliers")
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--n_heads",      type=int,   default=8)
    p.add_argument("--n_enc_layers", type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--param_weight", type=float, default=0.5,
                   help="Weight of the param-regression loss (vs type CE)")
    p.add_argument("--label_smoothing", type=float, default=0.05,
                   help="Label smoothing for type CE; 0 disables")
    p.add_argument("--max_class_weight", type=float, default=10.0,
                   help="Cap on inverse-frequency class weights")
    p.add_argument("--class_weight_power", type=float, default=1.0,
                   help="Exponent applied to inverse-frequency weights; "
                        "0.5 often improves minority precision")
    p.add_argument("--resume",       type=str,   default="")
    p.add_argument("--init_weights", type=str,   default="",
                   help="Warm-start model weights but reset training progress")
    p.add_argument("--stream_shards", action="store_true",
                   help="Load one training NPZ shard at a time")
    p.add_argument("--save_every_shards", type=int, default=10,
                   help="Save resumable state every N training shards")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log_every",    type=int,   default=1)
    return p.parse_args()


# ─── Data loading ─────────────────────────────────────────────────────────────

def _encode_stroke(stroke_pts: list, max_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a stroke into a fixed-size point matrix + boolean mask.

    Strokes shorter than max_pts are padded with -1 sentinels.
    Strokes longer than max_pts are uniformly subsampled.

    Returns:
      pts  : (max_pts, 2)  float32
      mask : (max_pts,)    bool — True where pts contains a real point
    """
    arr = np.array(stroke_pts, dtype=np.float32)
    n   = len(arr)

    if n > max_pts:
        idx = np.round(np.linspace(0, n - 1, max_pts)).astype(int)
        arr = arr[idx]
        mask = np.ones(max_pts, dtype=bool)
    elif n < max_pts:
        pad  = np.full((max_pts - n, 2), -1.0, dtype=np.float32)
        arr  = np.vstack([arr, pad])
        mask = np.zeros(max_pts, dtype=bool)
        mask[:n] = True
    else:
        mask = np.ones(max_pts, dtype=bool)

    return arr, mask


def _encode_command(cmd: dict) -> tuple[int, np.ndarray]:
    """Encode a command into (type_id, 6-d parameter vector). Same as v1/v2."""
    type_id = CMD_TYPES[cmd["type"]]
    p = np.zeros(N_PARAMS, dtype=np.float32)

    if cmd["type"] == "LINE":
        p[0:2] = cmd["start"]
        p[2:4] = cmd["end"]
    elif cmd["type"] == "ARC":
        p[0:2] = cmd["center"]
        p[2]   = cmd["radius"]
        p[3]   = cmd["start_angle"] / 360.0
        p[4]   = cmd["end_angle"]   / 360.0
    elif cmd["type"] == "CIRCLE":
        p[0:2] = cmd["center"]
        p[2]   = cmd["radius"]
    # POLYLINE: zeros
    return type_id, p


class EncodedDataset:
    """Compact in-memory edge tensors loaded from JSON or prepared NPZ shards."""

    def __init__(self, pts, mask, types, params):
        self.pts = np.asarray(pts, dtype=np.float32)
        self.mask = np.asarray(mask, dtype=bool)
        self.types = np.asarray(types, dtype=np.int64)
        self.params = np.asarray(params, dtype=np.float32)
        n = len(self.types)
        if not (len(self.pts) == len(self.mask) == len(self.params) == n):
            raise ValueError("inconsistent encoded dataset lengths")

    def __len__(self):
        return len(self.types)


def _resize_encoded_points(points: np.ndarray, mask: np.ndarray, max_pts: int):
    """Resize an encoded point axis while preserving uniform stroke coverage."""
    source_pts = points.shape[1]
    if source_pts == max_pts:
        return points, mask
    out = np.full((len(points), max_pts, 2), -1.0, dtype=np.float32)
    out_mask = np.zeros((len(points), max_pts), dtype=bool)
    lengths = mask.sum(axis=1).astype(int)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        if length <= 0:
            continue
        if length > max_pts:
            idx = np.round(np.linspace(0, length - 1, max_pts)).astype(int)
            out[rows] = points[rows][:, idx]
            out_mask[rows] = True
        else:
            out[rows, :length] = points[rows, :length]
            out_mask[rows, :length] = True
    return out, out_mask


def _apply_arc_encoding(dataset: EncodedDataset, arc_encoding: str) -> EncodedDataset:
    if arc_encoding == "center_radius_angles":
        return dataset
    if arc_encoding != "three_point":
        raise ValueError(f"unknown arc encoding: {arc_encoding}")

    rows = np.flatnonzero(dataset.types == CMD_TYPES["ARC"])
    if not len(rows):
        return dataset
    params = dataset.params.copy()
    lengths = dataset.mask[rows].sum(axis=1).astype(int)
    for length in np.unique(lengths):
        selected = rows[lengths == length]
        if length <= 0:
            continue
        middle = (length - 1) // 2
        params[selected, 0:2] = dataset.pts[selected, 0]
        params[selected, 2:4] = dataset.pts[selected, middle]
        params[selected, 4:6] = dataset.pts[selected, length - 1]
    dataset.params = params
    return dataset


def _filter_inconsistent_circles(dataset: EncodedDataset) -> tuple[EncodedDataset, int]:
    """Remove incomplete loops carrying non-identifiable full-circle targets."""
    circles = dataset.types == CMD_TYPES["CIRCLE"]
    invalid = circles & (
        (np.abs(dataset.params[:, :2] - 0.5) > 0.30).any(axis=1)
        | (dataset.params[:, 2] > 0.65)
    )
    removed = int(invalid.sum())
    if not removed:
        return dataset, 0
    keep = ~invalid
    return EncodedDataset(
        dataset.pts[keep], dataset.mask[keep],
        dataset.types[keep], dataset.params[keep],
    ), removed


def _relabel_degenerate_arcs(
    dataset: EncodedDataset, min_sagitta_ratio: float = 0.01,
) -> tuple[EncodedDataset, int]:
    """Relabel raster-straight source arcs to match the inference contract."""
    relabel = []
    for row in np.flatnonzero(dataset.types == CMD_TYPES["ARC"]):
        points = dataset.pts[row, dataset.mask[row]]
        if len(points) < 3:
            relabel.append(row)
            continue
        chord = points[-1] - points[0]
        length = float(np.linalg.norm(chord))
        if length < 1e-9:
            relabel.append(row)
            continue
        normal = np.array([-chord[1], chord[0]], dtype=np.float32) / length
        ratio = float(np.abs((points - points[0]) @ normal).max()) / length
        if ratio < min_sagitta_ratio:
            relabel.append(row)

    if not relabel:
        return dataset, 0
    relabel = np.asarray(relabel, dtype=np.int64)
    lengths = dataset.mask[relabel].sum(axis=1).astype(int)
    dataset.types[relabel] = CMD_TYPES["LINE"]
    dataset.params[relabel] = 0.0
    dataset.params[relabel, 0:2] = dataset.pts[relabel, 0]
    dataset.params[relabel, 2:4] = dataset.pts[relabel, lengths - 1]
    return dataset, len(relabel)


def load_dataset(
    data_dir: str, split: str, max_pts: int,
    arc_encoding: str = "center_radius_angles",
) -> EncodedDataset:
    """Load prepared NPZ shards, falling back to legacy per-edge JSON."""
    split_dir = Path(data_dir) / split
    shards = sorted(split_dir.glob("shard_*.npz"))
    if shards:
        parts = []
        removed_total = 0
        relabeled_total = 0
        for path in shards:
            part, removed, relabeled = load_npz_shard(
                path, max_pts, arc_encoding
            )
            parts.append((part.pts, part.mask, part.types, part.params))
            removed_total += removed
            relabeled_total += relabeled
        dataset = EncodedDataset(*(
            np.concatenate([part[i] for part in parts], axis=0) for i in range(4)
        ))
        suffix = (f"; filtered {removed_total} incomplete circles; relabeled "
                  f"{relabeled_total} raster-straight arcs")
        print(f"  Loaded {len(dataset)} '{split}' samples from "
              f"{len(shards)} NPZ shards{suffix}")
        return dataset

    files = sorted(split_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No shard_*.npz or JSON files in {split_dir}. Run a Stage-3 "
            "dataset generator first.")

    points, masks, types, all_params = [], [], [], []
    for path in files:
        with open(path) as f:
            s = json.load(f)
        if "stroke" not in s or "command" not in s:
            raise ValueError(
                f"{path} is not a v3 sample (missing 'stroke'/'command'). "
                f"Did you point this trainer at v2 data?")

        pts, mask  = _encode_stroke(s["stroke"], max_pts)
        type_id, params = _encode_command(s["command"])
        points.append(pts)
        masks.append(mask)
        types.append(type_id)
        all_params.append(params)

    dataset = EncodedDataset(points, masks, types, all_params)
    dataset, removed = _filter_inconsistent_circles(dataset)
    dataset, relabeled = _relabel_degenerate_arcs(dataset)
    print(f"  Loaded {len(dataset)} '{split}' JSON samples from {split_dir}")
    return _apply_arc_encoding(dataset, arc_encoding)


def load_npz_shard(
    path: str | Path,
    max_pts: int,
    arc_encoding: str = "center_radius_angles",
) -> tuple[EncodedDataset, int, int]:
    """Load and normalize one prepared shard without retaining the NPZ handle."""
    with np.load(path) as data:
        points, mask = _resize_encoded_points(
            np.asarray(data["points"]), np.asarray(data["mask"]), max_pts
        )
        dataset = EncodedDataset(
            points,
            mask,
            np.asarray(data["types"]),
            np.asarray(data["params"]),
        )
    dataset, removed = _filter_inconsistent_circles(dataset)
    dataset, relabeled = _relabel_degenerate_arcs(dataset)
    return _apply_arc_encoding(dataset, arc_encoding), removed, relabeled


def list_npz_shards(data_dir: str, split: str) -> list[Path]:
    shards = sorted((Path(data_dir) / split).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(
            f"No shard_*.npz files in {Path(data_dir) / split}"
        )
    return shards


def scan_shard_class_counts(
    shards: list[Path], max_pts: int, arc_encoding: str,
) -> tuple[np.ndarray, int, int, int]:
    """Compute post-cleaning class counts without holding all shards in RAM."""
    counts = np.zeros(N_CMD_TYPES, dtype=np.int64)
    samples = removed_total = relabeled_total = 0
    for path in shards:
        dataset, removed, relabeled = load_npz_shard(
            path, max_pts, arc_encoding
        )
        counts += np.bincount(dataset.types, minlength=N_CMD_TYPES)
        samples += len(dataset)
        removed_total += removed
        relabeled_total += relabeled
    return counts, samples, removed_total, relabeled_total


def make_batch(samples: EncodedDataset, indices: list[int], device):
    """Collate sample indices into batch tensors."""
    import torch
    pts = samples.pts[indices]
    mask = samples.mask[indices]
    types = samples.types[indices]
    params = samples.params[indices]
    return (
        torch.from_numpy(pts).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(types).to(device),
        torch.from_numpy(params).to(device),
    )


def compute_class_weights(
    samples: EncodedDataset,
    max_weight: float,
    power: float = 1.0,
) -> np.ndarray:
    """Powered inverse-frequency class weights, capped at max_weight."""
    if power < 0.0:
        raise ValueError("class_weight_power must be non-negative")
    counts = np.bincount(samples.types, minlength=N_CMD_TYPES)
    return compute_class_weights_from_counts(counts, max_weight, power)


def compute_class_weights_from_counts(
    counts: np.ndarray, max_weight: float, power: float = 1.0,
) -> np.ndarray:
    """Compute powered inverse-frequency weights from exact class counts."""
    if power < 0.0:
        raise ValueError("class_weight_power must be non-negative")
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    weights = np.ones(N_CMD_TYPES, dtype=np.float32)
    for cls in range(N_CMD_TYPES):
        c = int(counts[cls])
        if c == 0:
            # SketchGraphs t16 has no spline/polyline entities. A zero weight
            # also prevents label smoothing from inventing POLYLINE targets.
            weights[cls] = 0.0
        else:
            w = (total / (N_CMD_TYPES * c)) ** power
            weights[cls] = min(w, max_weight)
    return weights


def parameter_loss(param_pred, params, types, arc_encoding="center_radius_angles"):
    """Class-aware parameter loss with periodic arc-angle residuals."""
    import torch

    class_losses = []

    line = types == CMD_TYPES["LINE"]
    if line.any():
        class_losses.append(torch.abs(
            param_pred[line, :4] - params[line, :4]
        ).sum(dim=-1).mean())

    arc = types == CMD_TYPES["ARC"]
    if arc.any():
        if arc_encoding == "three_point":
            class_losses.append(torch.abs(
                param_pred[arc, :6] - params[arc, :6]
            ).sum(dim=-1).mean())
        else:
            geom = torch.abs(param_pred[arc, :3] - params[arc, :3]).sum(dim=-1)
            delta = torch.remainder(
                param_pred[arc, 3:5] - params[arc, 3:5] + 0.5, 1.0
            ) - 0.5
            class_losses.append((geom + torch.abs(delta).sum(dim=-1)).mean())

    circle = types == CMD_TYPES["CIRCLE"]
    if circle.any():
        class_losses.append(torch.abs(
            param_pred[circle, :3] - params[circle, :3]
        ).sum(dim=-1).mean())

    return torch.stack(class_losses).mean() if class_losses else param_pred.sum() * 0.0


# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(max_pts: int, d_model: int, n_heads: int,
                n_enc_layers: int, dropout: float):
    """
    Encoder-only per-edge model.

    Input:
      pts  : (B, max_pts, 2)
      mask : (B, max_pts)  bool — True where real point
    Output:
      type_logits : (B, n_classes)
      params      : (B, 6)
    """
    import torch
    import torch.nn as nn

    class SinusoidalPositionalEmbedding(nn.Module):
        def __init__(self, max_len: int, d_model: int):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class EdgeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.point_proj = nn.Linear(2, d_model)
            self.pos_emb    = SinusoidalPositionalEmbedding(max_pts, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder    = nn.TransformerEncoder(enc_layer, n_enc_layers)
            self.type_head  = nn.Linear(d_model, N_CMD_TYPES)
            self.param_head = nn.Linear(d_model, N_PARAMS)

        def forward(self, pts, mask):
            # pts:  (B, P, 2)    mask: (B, P) — True where real
            x = self.point_proj(pts)                       # (B, P, D)
            x = self.pos_emb(x)
            # src_key_padding_mask: True where IGNORE (inverse of `mask`)
            x = self.encoder(x, src_key_padding_mask=~mask)

            # Masked mean-pool over real points
            mask_f = mask.float().unsqueeze(-1)            # (B, P, 1)
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            return self.type_head(pooled), self.param_head(pooled)

    return EdgeClassifier()


# ─── Validation metrics ───────────────────────────────────────────────────────

def _stroke_residuals(
    params: np.ndarray,
    points: np.ndarray,
    masks: np.ndarray,
    types: np.ndarray,
    arc_encoding: str,
) -> np.ndarray:
    """Mean point-to-predicted-primitive error in normalized edge units."""
    residuals = np.full(len(types), np.nan, dtype=np.float64)

    def line_residual(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        chord = b - a
        length = float(np.linalg.norm(chord))
        if length < 1e-8:
            return float(np.linalg.norm(q - a, axis=1).mean())
        return float(np.abs((q - a) @ np.array([-chord[1], chord[0]])).mean()
                     / length)

    for i, cls in enumerate(types):
        q = points[i, masks[i]]
        if not len(q):
            continue
        p = params[i]
        if cls == CMD_TYPES["LINE"]:
            residuals[i] = line_residual(q, p[0:2], p[2:4])
        elif cls == CMD_TYPES["CIRCLE"]:
            residuals[i] = float(np.abs(
                np.linalg.norm(q - p[0:2], axis=1) - abs(float(p[2]))
            ).mean())
        elif cls == CMD_TYPES["ARC"] and arc_encoding == "three_point":
            a, middle, b = p[0:2], p[2:4], p[4:6]
            determinant = 2.0 * (
                a[0] * (middle[1] - b[1])
                + middle[0] * (b[1] - a[1])
                + b[0] * (a[1] - middle[1])
            )
            if abs(float(determinant)) < 1e-8:
                residuals[i] = line_residual(q, a, b)
                continue
            a2, m2, b2 = float(a @ a), float(middle @ middle), float(b @ b)
            center = np.array([
                (a2 * (middle[1] - b[1])
                 + m2 * (b[1] - a[1])
                 + b2 * (a[1] - middle[1])) / determinant,
                (a2 * (b[0] - middle[0])
                 + m2 * (a[0] - b[0])
                 + b2 * (middle[0] - a[0])) / determinant,
            ])
            radius = float(np.linalg.norm(a - center))
            residuals[i] = float(np.abs(
                np.linalg.norm(q - center, axis=1) - radius
            ).mean())
        elif cls == CMD_TYPES["ARC"]:
            residuals[i] = float(np.abs(
                np.linalg.norm(q - p[0:2], axis=1) - abs(float(p[2]))
            ).mean())
    return residuals

def evaluate(model, val_data: EncodedDataset, device, batch_size: int,
             type_loss_fn, param_weight: float,
             arc_encoding: str = "center_radius_angles") -> dict:
    """Run model on val_data; return loss + per-class P/R/F1 + extras."""
    import torch

    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    confusion    = np.zeros((N_CMD_TYPES, N_CMD_TYPES), dtype=np.int64)
    param_l1_sum = np.zeros(N_CMD_TYPES, dtype=np.float64)
    param_n      = np.zeros(N_CMD_TYPES, dtype=np.int64)
    residual_sum = np.zeros(N_CMD_TYPES, dtype=np.float64)
    residual_n   = np.zeros(N_CMD_TYPES, dtype=np.int64)
    pred_counts  = np.zeros(N_CMD_TYPES, dtype=np.int64)

    with torch.no_grad():
        for s in range(0, len(val_data), batch_size):
            idx = list(range(s, min(s + batch_size, len(val_data))))
            pts, mask, types, params = make_batch(val_data, idx, device)
            type_logits, param_pred = model(pts, mask)

            t_loss = type_loss_fn(type_logits, types)

            p_loss = parameter_loss(param_pred, params, types, arc_encoding)

            total_loss += (t_loss + param_weight * p_loss).item()
            n_batches  += 1

            # Confusion matrix + per-class param L1
            preds = type_logits.argmax(dim=-1).cpu().numpy()
            true  = types.cpu().numpy()
            for t, p in zip(true, preds):
                confusion[t, p] += 1
                pred_counts[p]  += 1

            residuals = _stroke_residuals(
                param_pred.cpu().numpy(), pts.cpu().numpy(), mask.cpu().numpy(),
                true, arc_encoding)
            for cls in PARAM_CLASSES:
                cls_residuals = residuals[true == cls]
                cls_residuals = cls_residuals[np.isfinite(cls_residuals)]
                residual_sum[cls] += cls_residuals.sum()
                residual_n[cls] += len(cls_residuals)

            errors = torch.abs(param_pred - params)
            arc_mask = types == CMD_TYPES["ARC"]
            if arc_mask.any() and arc_encoding == "center_radius_angles":
                errors[arc_mask, 3:5] = torch.abs(torch.remainder(
                    param_pred[arc_mask, 3:5] - params[arc_mask, 3:5] + 0.5,
                    1.0,
                ) - 0.5)
            for cls in PARAM_CLASSES:
                m = (true == cls)
                if m.any():
                    if cls == CMD_TYPES["LINE"]:
                        dims = slice(0, 4)
                    elif cls == CMD_TYPES["ARC"]:
                        dims = slice(0, 6) if arc_encoding == "three_point" else slice(0, 5)
                    else:
                        dims = slice(0, 3)
                    class_l1 = errors[:, dims].mean(dim=-1).cpu().numpy()
                    param_l1_sum[cls] += class_l1[m].sum()
                    param_n[cls]      += int(m.sum())

    # Per-class precision / recall / F1
    metrics: dict = {"per_class": {}}
    inv = {v: k for k, v in CMD_TYPES.items()}
    for cls in range(N_CMD_TYPES):
        tp = int(confusion[cls, cls])
        fn = int(confusion[cls, :].sum() - tp)
        fp = int(confusion[:, cls].sum() - tp)
        prec   = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1     = (2 * prec * recall / (prec + recall)
                  if (prec + recall) else 0.0)
        param_l1 = (param_l1_sum[cls] / param_n[cls]
                    if param_n[cls] else 0.0)
        stroke_residual = (residual_sum[cls] / residual_n[cls]
                           if residual_n[cls] else 0.0)
        metrics["per_class"][inv[cls]] = {
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "param_l1":  round(float(param_l1), 4),
            "stroke_residual": round(float(stroke_residual), 4),
            "support":   int(confusion[cls, :].sum()),
        }

    # Prediction-distribution entropy (collapse detector)
    supported_classes = np.flatnonzero(confusion.sum(axis=1) > 0)
    active_pred_counts = pred_counts[supported_classes]
    dist = active_pred_counts / active_pred_counts.sum().clip(min=1)
    H_max = math.log(max(len(supported_classes), 2))
    H     = -(dist * np.where(dist > 0, np.log(dist + 1e-12), 0)).sum()
    metrics["pred_entropy"]      = round(float(H / H_max), 4)
    metrics["pred_distribution"] = {inv[c]: int(pred_counts[c])
                                    for c in range(N_CMD_TYPES)}
    metrics["confusion"]  = confusion.tolist()
    metrics["val_loss"]   = total_loss / max(n_batches, 1)
    metrics["accuracy"] = round(float(np.trace(confusion) / max(confusion.sum(), 1)), 4)
    supported = [m["f1"] for m in metrics["per_class"].values() if m["support"] > 0]
    metrics["supported_macro_f1"] = round(float(np.mean(supported)) if supported else 0.0, 4)
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"  val_loss     : {metrics['val_loss']:.4f}")
    print(f"  accuracy     : {metrics['accuracy']:.3f}")
    print(f"  supported F1 : {metrics['supported_macro_f1']:.3f}")
    print(f"  pred_entropy : {metrics['pred_entropy']:.3f} "
          "(normalized over supported classes)")
    print(f"  {'class':<10} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'paramL1':>9} {'strokeErr':>9} {'support':>8}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<10} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['param_l1']:>9.4f} "
              f"{m['stroke_residual']:>9.4f} {m['support']:>8d}")


# ─── Training loop ────────────────────────────────────────────────────────────

def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    if args.resume and args.init_weights:
        raise ValueError("--resume and --init_weights are mutually exclusive")
    if args.save_every_shards < 0:
        raise ValueError("--save_every_shards must be non-negative")

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(Path(args.output_dir) / "tb_logs_v3"))
        print("TensorBoard: tensorboard --logdir",
              Path(args.output_dir) / "tb_logs_v3")
    except ImportError:
        tb = None

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable — falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading datasets ...")
    train_data = None
    train_shards = None
    if args.stream_shards:
        train_shards = list_npz_shards(args.data_dir, "train")
        print(f"  Scanning {len(train_shards)} training shards ...")
        class_counts, n_train, removed, relabeled = scan_shard_class_counts(
            train_shards, args.max_pts, args.arc_encoding
        )
        print(
            f"  Streaming {n_train:,} train samples; filtered {removed:,} "
            f"incomplete circles; relabeled {relabeled:,} straight arcs"
        )
    else:
        train_data = load_dataset(
            args.data_dir, "train", args.max_pts, args.arc_encoding)
        class_counts = np.bincount(train_data.types, minlength=N_CMD_TYPES)
    val_data = load_dataset(
        args.data_dir, "val", args.max_pts, args.arc_encoding)

    # Class weights
    class_weights = compute_class_weights_from_counts(
        class_counts, args.max_class_weight, args.class_weight_power)
    print("\nClass weights (inverse-frequency power "
          f"{args.class_weight_power}, capped at {args.max_class_weight}):")
    inv = {v: k for k, v in CMD_TYPES.items()}
    for cls in range(N_CMD_TYPES):
        print(f"  {inv[cls]:<10} weight={class_weights[cls]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        max_pts      = args.max_pts,
        d_model      = args.d_model,
        n_heads      = args.n_heads,
        n_enc_layers = args.n_enc_layers,
        dropout      = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: encoder-only per-edge classifier — "
          f"{n_params:,} trainable params")

    if args.init_weights:
        init_path = Path(args.init_weights)
        if not init_path.exists():
            raise FileNotFoundError(f"missing initial checkpoint: {init_path}")
        checkpoint = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Warm-started model weights from {init_path}")

    # ── Loss & optimiser ──────────────────────────────────────────────────────
    cw_tensor = torch.from_numpy(class_weights).to(device)
    type_loss_fn = nn.CrossEntropyLoss(
        weight          = cw_tensor,
        label_smoothing = args.label_smoothing,
    )
    optim     = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    best_macro_f1 = -1.0
    resume_shard_position = 0
    partial_train_loss = 0.0
    partial_n_batches = 0
    last_metrics = None
    last_val_loss = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        if ckpt.get("version") != 3:
            print(f"WARNING: resuming from a non-v3 checkpoint "
                  f"(version={ckpt.get('version')}). State-dict load may fail.")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optim.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        completed = ckpt.get("epoch_complete", True)
        start_epoch = ckpt.get("epoch", 0) + (1 if completed else 0)
        resume_shard_position = (
            0 if completed else int(ckpt.get("next_shard_position", 0))
        )
        partial_train_loss = (
            0.0 if completed else float(ckpt.get("partial_train_loss", 0.0))
        )
        partial_n_batches = (
            0 if completed else int(ckpt.get("partial_n_batches", 0))
        )
        best_val_loss = float(ckpt.get(
            "best_val_loss", ckpt.get("val_loss", float("inf"))
        ))
        best_macro_f1 = float(ckpt.get(
            "best_macro_f1",
            ckpt.get("metrics", {}).get("supported_macro_f1", -1.0),
        ))
        last_metrics = ckpt.get("metrics")
        last_val_loss = float(ckpt.get("val_loss", float("inf")))
        print(
            f"Resumed epoch {start_epoch + 1}, shard position "
            f"{resume_shard_position}; best val_loss={best_val_loss:.4f}"
        )

    # ── Loop ──────────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, param_weight={args.param_weight}")
    print(f"{'─'*70}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()

        continuing = epoch == start_epoch and resume_shard_position > 0
        train_loss = partial_train_loss if continuing else 0.0
        n_batches = partial_n_batches if continuing else 0

        if train_shards is not None:
            shard_order = np.random.default_rng(
                np.random.SeedSequence([args.seed, epoch, 17])
            ).permutation(len(train_shards))
            first_position = resume_shard_position if continuing else 0
            shard_stream = enumerate(
                shard_order[first_position:], start=first_position
            )
        else:
            shard_stream = [(0, -1)]

        for shard_position, shard_index in shard_stream:
            if train_shards is not None:
                current_data, _removed, _relabeled = load_npz_shard(
                    train_shards[int(shard_index)], args.max_pts,
                    args.arc_encoding,
                )
                sample_indices = np.arange(len(current_data), dtype=np.int64)
                np.random.default_rng(np.random.SeedSequence([
                    args.seed, epoch, int(shard_index), 31,
                ])).shuffle(sample_indices)
            else:
                current_data = train_data
                sample_indices = np.arange(len(current_data), dtype=np.int64)
                np.random.default_rng(np.random.SeedSequence([
                    args.seed, epoch, 31,
                ])).shuffle(sample_indices)

            for s in range(0, len(sample_indices), args.batch_size):
                idx = sample_indices[s:s + args.batch_size]
                pts, mask, types, params = make_batch(current_data, idx, device)

                optim.zero_grad(set_to_none=True)
                type_logits, param_pred = model(pts, mask)
                t_loss = type_loss_fn(type_logits, types)
                p_loss = parameter_loss(
                    param_pred, params, types, args.arc_encoding
                )
                loss = t_loss + args.param_weight * p_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

                train_loss += loss.item()
                n_batches += 1

            if train_shards is not None:
                completed_shards = shard_position + 1
                if (
                    args.save_every_shards
                    and completed_shards % args.save_every_shards == 0
                    and completed_shards < len(train_shards)
                ):
                    partial = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "epoch_complete": False,
                        "next_shard_position": completed_shards,
                        "partial_train_loss": train_loss,
                        "partial_n_batches": n_batches,
                        "val_loss": last_val_loss,
                        "best_val_loss": best_val_loss,
                        "best_macro_f1": best_macro_f1,
                        "version": 3,
                        "architecture": "encoder_only",
                        "config": {
                            "max_pts": args.max_pts,
                            "n_cmd_types": N_CMD_TYPES,
                            "d_model": args.d_model,
                            "n_heads": args.n_heads,
                            "n_enc_layers": args.n_enc_layers,
                            "dropout": args.dropout,
                            "arc_encoding": args.arc_encoding,
                            "class_weight_power": args.class_weight_power,
                            "label_smoothing": args.label_smoothing,
                            "param_weight": args.param_weight,
                        },
                        "cmd_types": CMD_TYPES,
                        "class_weights": class_weights.tolist(),
                        "metrics": last_metrics or {},
                    }
                    _atomic_torch_save(
                        partial, out_dir / "free2cad_v3_latest.pth"
                    )
                    print(
                        f"  epoch {epoch + 1}: saved shard "
                        f"{completed_shards}/{len(train_shards)}"
                    )
                del current_data

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        metrics = evaluate(model, val_data, device, args.batch_size,
                           type_loss_fn, args.param_weight, args.arc_encoding)
        avg_val = metrics["val_loss"]

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1:3d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"lr={lr_now:.2e}  t={elapsed:.1f}s")
        print_metrics(metrics)

        # TensorBoard
        if tb and (epoch + 1) % args.log_every == 0:
            tb.add_scalar("Loss/train",       avg_train,                epoch)
            tb.add_scalar("Loss/val",         avg_val,                  epoch)
            tb.add_scalar("LR",               lr_now,                   epoch)
            tb.add_scalar("PredEntropy",      metrics["pred_entropy"],  epoch)
            for cls, m in metrics["per_class"].items():
                tb.add_scalar(f"F1/{cls}",       m["f1"],       epoch)
                tb.add_scalar(f"Recall/{cls}",   m["recall"],   epoch)
                tb.add_scalar(f"ParamL1/{cls}",  m["param_l1"], epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        macro_f1 = metrics["supported_macro_f1"]
        improved_loss = avg_val < best_val_loss
        improved_f1 = macro_f1 > best_macro_f1
        if improved_loss:
            best_val_loss = avg_val
        if improved_f1:
            best_macro_f1 = macro_f1
        last_metrics = metrics
        last_val_loss = avg_val
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch":            epoch,
            "epoch_complete":   True,
            "next_shard_position": 0,
            "val_loss":         avg_val,
            "best_val_loss":    best_val_loss,
            "best_macro_f1":    best_macro_f1,
            "version":          3,
            "architecture":     "encoder_only",
            "config": {
                "max_pts":      args.max_pts,
                "n_cmd_types":  N_CMD_TYPES,
                "d_model":      args.d_model,
                "n_heads":      args.n_heads,
                "n_enc_layers": args.n_enc_layers,
                "dropout":      args.dropout,
                "arc_encoding": args.arc_encoding,
                "class_weight_power": args.class_weight_power,
                "label_smoothing": args.label_smoothing,
                "param_weight": args.param_weight,
            },
            "cmd_types":     CMD_TYPES,
            "class_weights": class_weights.tolist(),
            "metrics":       metrics,
        }
        _atomic_torch_save(ckpt, out_dir / "free2cad_v3_latest.pth")

        if improved_loss:
            _atomic_torch_save(ckpt, out_dir / "free2cad_v3_best.pth")
            print(f"  → New best (val_loss={best_val_loss:.4f})")

        if improved_f1:
            _atomic_torch_save(ckpt, out_dir / "free2cad_v3_best_f1.pth")
            print(f"  → New best supported macro-F1 ({best_macro_f1:.4f})")

        if (epoch + 1) % 25 == 0:
            _atomic_torch_save(
                ckpt, out_dir / f"free2cad_v3_ep{epoch+1:04d}.pth"
            )

        resume_shard_position = 0
        partial_train_loss = 0.0
        partial_n_batches = 0

    if tb:
        tb.close()

    print(f"\n{'─'*70}")
    print(f"Training complete.")
    print(f"  Best checkpoint : {out_dir / 'free2cad_v3_best.pth'}")
    print(f"  Best val_loss   : {best_val_loss:.4f}")
    print(f"  Best macro-F1   : {best_macro_f1:.4f}")
    print(f"\nUpdate config.yaml:")
    print(f"  free2cad:")
    print(f"    weights: \"{out_dir / 'free2cad_v3_best.pth'}\"")
    print(f"    device:  \"{args.device}\"")
    print(f"\nIMPORTANT: stage3_primitive_fit.py must be patched to handle")
    print(f"checkpoints with version=3 (encoder-only architecture).")
    print(f"See the patch note at the bottom of train_free2cad_v3.py.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)


# =============================================================================
# PATCH NOTE FOR stage3_primitive_fit.py
# =============================================================================
#
# The current Free2CADFitter in stage3_primitive_fit.py builds a seq2seq model
# regardless of checkpoint contents. v3 checkpoints have a different
# architecture (encoder-only) and a "version": 3 marker.
#
# Required changes in Free2CADFitter._load:
#
#   1. After `ckpt = torch.load(weights, map_location=device)`, check:
#          version = ckpt.get("version", 1)
#
#   2. If version == 3, build the encoder-only model (copy the build_model
#      function from this file) and skip BOS-related logic in fit_edge.
#
#   3. The forward pass changes from
#          type_logits, param_preds = model(strokes, mask, tgt_in, ...)
#      to
#          type_logits, param_pred = model(pts, mask)
#      where pts is (1, max_pts, 2) and mask is (1, max_pts).
#
#   4. The output is per-edge directly — no decoding loop, no END token,
#      no autoregression. Just argmax(type_logits) and param_pred.
#
# Single-stroke encoding for inference:
#   - Use the same canvas-normalisation as v3 training (centroid-anchor,
#     10% margin, isotropic scale to fill the canvas). The current
#     _encode_edge with _INFERENCE_FRACTION=1.0 already does this.
#   - Pad/subsample to max_pts; build the boolean mask.
#
# =============================================================================
