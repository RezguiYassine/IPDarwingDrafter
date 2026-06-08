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
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


# ─── Dataset ─────────────────────────────────────────────────────────────────

class KPDataset(Dataset):
    def __init__(self, npz_paths, crop=512, sigma=3.0, augment=True,
                 pos_crop_prob=0.8):
        self.paths = list(npz_paths)
        self.crop = crop
        self.sigma = sigma
        self.augment = augment
        self.pos_crop_prob = pos_crop_prob
        self.g = _gaussian2d(sigma)

    def __len__(self):
        return len(self.paths)

    def _pick_window(self, H, W, kps):
        cs = self.crop
        if H <= cs and W <= cs:
            return 0, 0
        if len(kps) and random.random() < self.pos_crop_prob:
            kx, ky = kps[random.randrange(len(kps))][:2]
            jit = cs // 4
            cx = kx + random.randint(-jit, jit)
            cy = ky + random.randint(-jit, jit)
        else:
            cx, cy = random.randint(0, W), random.randint(0, H)
        x0 = int(np.clip(cx - cs // 2, 0, max(0, W - cs)))
        y0 = int(np.clip(cy - cs // 2, 0, max(0, H - cs)))
        return x0, y0

    def __getitem__(self, i):
        d = np.load(self.paths[i], allow_pickle=True)
        sk = d["skeleton"]
        kps = d["kps"]
        H, W = sk.shape
        cs = self.crop

        x0, y0 = self._pick_window(H, W, kps)
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

        hm = make_heatmap(kc, cs, cs, self.g)
        img = (sk_c > 0).astype(np.float32)[None]   # (1, cs, cs)

        if self.augment:
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                hm = np.rot90(hm, k, axes=(1, 2)).copy()
            if random.random() < 0.5:
                img = img[:, :, ::-1].copy(); hm = hm[:, :, ::-1].copy()
            if random.random() < 0.5:
                img = img[:, ::-1, :].copy(); hm = hm[:, ::-1, :].copy()

        return torch.from_numpy(img), torch.from_numpy(hm)


# ─── CenterNet penalty-reduced focal loss ────────────────────────────────────

def focal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = torch.clamp(torch.sigmoid(logits), 1e-6, 1 - 1e-6)
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
        hm = torch.sigmoid(model(t))[0, :, :H, :W].cpu().numpy()

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
    model.train()
    return {"f1": f1, "prec": prec, "rec": rec, "macro_f1": float(f1.mean())}


# ─── Training loop ───────────────────────────────────────────────────────────

def _collect(labels_root: Path, split: str) -> list[Path]:
    return sorted(Path(p) for p in
                  glob.glob(str(labels_root / split / "**" / "*.npz"),
                            recursive=True))


def train(args):
    device = args.device
    labels = Path(args.labels)
    train_paths = _collect(labels, "train")
    val_paths = _collect(labels, "validation")
    if not train_paths:
        raise SystemExit(f"No train npz under {labels/'train'} — run "
                         "tools.d2c_keypoint_labels first.")
    rng = random.Random(args.seed)
    rng.shuffle(val_paths)
    val_subset = val_paths[:args.val_subset]
    print(f"train npz={len(train_paths)}  val npz={len(val_paths)} "
          f"(eval on {len(val_subset)})  device={device}")

    ds = KPDataset(train_paths, crop=args.crop, sigma=args.sigma, augment=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=True, persistent_workers=args.workers > 0)

    model = s2._build_stacked_hourglass().to(device)
    # Focal-loss output-head init (RetinaNet/CenterNet): tiny weights + negative
    # prior bias so the *initial* output is ~uniformly `prior` everywhere. This
    # keeps the initial background loss O(1) instead of ~1e5; without it the
    # first updates diverge the logits and the model collapses to predicting
    # "background everywhere" (loss frozen at the -log(1e-6) clamp) and never
    # recovers. Both the bias AND the weights must be controlled — biasing alone
    # leaves the default-magnitude weights to dominate the initial logits.
    prior_bias = -math.log((1 - args.prior) / args.prior)
    with torch.no_grad():
        for head in (model.out1, model.out2):
            torch.nn.init.normal_(head.weight, std=1e-3)
            torch.nn.init.constant_(head.bias, prior_bias)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    best_f1, step, t0 = -1.0, 0, time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    running = 0.0

    while step < args.steps:
        for img, hm in dl:
            img, hm = img.to(device, non_blocking=True), hm.to(device, non_blocking=True)
            logits = model(img)
            loss = focal_loss(logits, hm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += loss.item()
            step += 1

            if step % args.log_every == 0:
                rate = step / (time.time() - t0)
                print(f"step {step:>6}/{args.steps}  loss={running/args.log_every:.4f}"
                      f"  {rate:.1f} it/s")
                running = 0.0

            if val_subset and step % args.val_every == 0:
                m = evaluate_f1(model, val_subset, device,
                                match_radius=args.match_radius)
                print(f"  [val@{step}] macro_f1={m['macro_f1']:.3f}  "
                      f"end={m['f1'][0]:.3f} junc={m['f1'][1]:.3f} "
                      f"corner={m['f1'][2]:.3f}")
                if m["macro_f1"] > best_f1:
                    best_f1 = m["macro_f1"]
                    torch.save({"model_state_dict": model.state_dict(),
                                "step": step, "macro_f1": best_f1,
                                "n_classes": N_CLASSES}, out_path)
                    print(f"  ✓ saved best ({best_f1:.3f}) → {out_path}")
            if step >= args.steps:
                break

    # always keep a final checkpoint even if val was disabled
    if best_f1 < 0:
        torch.save({"model_state_dict": model.state_dict(), "step": step,
                    "n_classes": N_CLASSES}, out_path)
        print(f"saved final (no val) → {out_path}")
    print(f"done. best macro_f1={best_f1:.3f}  ({step} steps, "
          f"{(time.time()-t0)/60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="output/Drawing2CAD/kp_labels")
    ap.add_argument("--out", default="models/puhachov_d2c.pth")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--prior", type=float, default=0.01,
                    help="focal-loss output prior; sets initial output bias")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-subset", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--match-radius", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
