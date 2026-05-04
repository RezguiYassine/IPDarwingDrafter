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
import time
import argparse
from pathlib import Path
from collections import Counter

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
    p.add_argument("--resume",       type=str,   default="")
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


def load_dataset(data_dir: str, split: str, max_pts: int) -> list[dict]:
    """Load all v3 per-edge JSON samples from data_dir/<split>/*.json."""
    split_dir = Path(data_dir) / split
    files = sorted(split_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No JSON files in {split_dir}. "
            f"Run generate_sketches_v3.py first.")

    samples = []
    for path in files:
        with open(path) as f:
            s = json.load(f)
        if "stroke" not in s or "command" not in s:
            raise ValueError(
                f"{path} is not a v3 sample (missing 'stroke'/'command'). "
                f"Did you point this trainer at v2 data?")

        pts, mask  = _encode_stroke(s["stroke"], max_pts)
        type_id, params = _encode_command(s["command"])
        samples.append({
            "pts":      pts,
            "mask":     mask,
            "type_id":  type_id,
            "params":   params,
        })

    print(f"  Loaded {len(samples)} '{split}' samples from {split_dir}")
    return samples


def make_batch(samples: list[dict], indices: list[int], device):
    """Collate sample indices into batch tensors."""
    import torch
    pts    = np.stack([samples[i]["pts"]    for i in indices])
    mask   = np.stack([samples[i]["mask"]   for i in indices])
    types  = np.array([samples[i]["type_id"] for i in indices], dtype=np.int64)
    params = np.stack([samples[i]["params"] for i in indices])
    return (
        torch.from_numpy(pts).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(types).to(device),
        torch.from_numpy(params).to(device),
    )


def compute_class_weights(samples: list[dict], max_weight: float) -> np.ndarray:
    """Inverse-frequency class weights, capped at max_weight."""
    counts = Counter(s["type_id"] for s in samples)
    total  = sum(counts.values())
    weights = np.ones(N_CMD_TYPES, dtype=np.float32)
    for cls in range(N_CMD_TYPES):
        c = counts.get(cls, 0)
        if c == 0:
            weights[cls] = max_weight
        else:
            w = total / (N_CMD_TYPES * c)
            weights[cls] = min(w, max_weight)
    return weights


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

def evaluate(model, val_data: list[dict], device, batch_size: int,
             type_loss_fn, param_weight: float) -> dict:
    """Run model on val_data; return loss + per-class P/R/F1 + extras."""
    import torch

    model.eval()
    total_loss   = 0.0
    n_batches    = 0
    confusion    = np.zeros((N_CMD_TYPES, N_CMD_TYPES), dtype=np.int64)
    param_l1_sum = np.zeros(N_CMD_TYPES, dtype=np.float64)
    param_n      = np.zeros(N_CMD_TYPES, dtype=np.int64)
    pred_counts  = np.zeros(N_CMD_TYPES, dtype=np.int64)

    with torch.no_grad():
        for s in range(0, len(val_data), batch_size):
            idx = list(range(s, min(s + batch_size, len(val_data))))
            pts, mask, types, params = make_batch(val_data, idx, device)
            type_logits, param_pred = model(pts, mask)

            t_loss = type_loss_fn(type_logits, types)

            valid = torch.zeros_like(types, dtype=torch.float)
            for cls in PARAM_CLASSES:
                valid = valid + (types == cls).float()
            n_valid = valid.sum().clamp(min=1)
            p_loss  = (
                torch.abs(param_pred - params).sum(dim=-1) * valid
            ).sum() / n_valid

            total_loss += (t_loss + param_weight * p_loss).item()
            n_batches  += 1

            # Confusion matrix + per-class param L1
            preds = type_logits.argmax(dim=-1).cpu().numpy()
            true  = types.cpu().numpy()
            for t, p in zip(true, preds):
                confusion[t, p] += 1
                pred_counts[p]  += 1

            l1_per = torch.abs(param_pred - params).mean(dim=-1).cpu().numpy()
            for cls in PARAM_CLASSES:
                m = (true == cls)
                if m.any():
                    param_l1_sum[cls] += l1_per[m].sum()
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
        metrics["per_class"][inv[cls]] = {
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "param_l1":  round(float(param_l1), 4),
            "support":   int(confusion[cls, :].sum()),
        }

    # Prediction-distribution entropy (collapse detector)
    dist = pred_counts / pred_counts.sum().clip(min=1)
    H_max = math.log(N_CMD_TYPES)
    H     = -(dist * np.where(dist > 0, np.log(dist + 1e-12), 0)).sum()
    metrics["pred_entropy"]      = round(float(H / H_max), 4)
    metrics["pred_distribution"] = {inv[c]: int(pred_counts[c])
                                    for c in range(N_CMD_TYPES)}
    metrics["confusion"]  = confusion.tolist()
    metrics["val_loss"]   = total_loss / max(n_batches, 1)
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"  val_loss     : {metrics['val_loss']:.4f}")
    print(f"  pred_entropy : {metrics['pred_entropy']:.3f} "
          f"(1.0 = perfectly balanced; <0.5 = collapse risk)")
    print(f"  {'class':<10} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'paramL1':>9} {'support':>8}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<10} {m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['param_l1']:>9.4f} {m['support']:>8d}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

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
    train_data = load_dataset(args.data_dir, "train", args.max_pts)
    val_data   = load_dataset(args.data_dir, "val",   args.max_pts)

    # Class weights
    class_weights = compute_class_weights(train_data, args.max_class_weight)
    print("\nClass weights (capped at "
          f"{args.max_class_weight}):")
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
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("version") != 3:
            print(f"WARNING: resuming from a non-v3 checkpoint "
                  f"(version={ckpt.get('version')}). State-dict load may fail.")
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, "
              f"best val_loss={best_val_loss:.4f}")

    # ── Loop ──────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    print(f"\n{'─'*70}")
    print(f"Training: {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, param_weight={args.param_weight}")
    print(f"{'─'*70}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()

        indices = list(range(len(train_data)))
        rng.shuffle(indices)

        train_loss = 0.0
        n_batches  = 0

        for s in range(0, len(indices), args.batch_size):
            idx = indices[s : s + args.batch_size]
            pts, mask, types, params = make_batch(train_data, idx, device)

            optim.zero_grad()
            type_logits, param_pred = model(pts, mask)

            t_loss = type_loss_fn(type_logits, types)

            # Param L1, gated on classes that have meaningful params
            valid = torch.zeros_like(types, dtype=torch.float)
            for cls in PARAM_CLASSES:
                valid = valid + (types == cls).float()
            n_valid = valid.sum().clamp(min=1)
            p_loss = (
                torch.abs(param_pred - params).sum(dim=-1) * valid
            ).sum() / n_valid

            loss = t_loss + args.param_weight * p_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            train_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validation
        metrics = evaluate(model, val_data, device, args.batch_size,
                           type_loss_fn, args.param_weight)
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
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch":            epoch,
            "val_loss":         avg_val,
            "version":          3,
            "architecture":     "encoder_only",
            "config": {
                "max_pts":      args.max_pts,
                "n_cmd_types":  N_CMD_TYPES,
                "d_model":      args.d_model,
                "n_heads":      args.n_heads,
                "n_enc_layers": args.n_enc_layers,
                "dropout":      args.dropout,
            },
            "cmd_types":     CMD_TYPES,
            "class_weights": class_weights.tolist(),
            "metrics":       metrics,
        }
        torch.save(ckpt, out_dir / "free2cad_v3_latest.pth")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, out_dir / "free2cad_v3_best.pth")
            print(f"  → New best (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % 25 == 0:
            torch.save(ckpt, out_dir / f"free2cad_v3_ep{epoch+1:04d}.pth")

    if tb:
        tb.close()

    print(f"\n{'─'*70}")
    print(f"Training complete.")
    print(f"  Best checkpoint : {out_dir / 'free2cad_v3_best.pth'}")
    print(f"  Best val_loss   : {best_val_loss:.4f}")
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