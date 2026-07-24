#!/usr/bin/env python3
"""Patent-sheet compositor — merge GT-labelled single-drawing panels from
different datasets (Drawing2CAD / SketchGraphs / ArchCAD / CAD-VG) into one
complex, near-patent-distribution SHEET with exact ground truth.

Why this exists
---------------
Every dataset we have is a *single* clean drawing per image; real patent sheets
are the opposite -- several figures per page, mixed drawing styles, reference
numerals with leaders, a border and "FIG. N" labels, and scan clutter. We
cannot hand-label real patent rasters (the year-long task). But a compositor
*knows the provenance of every pixel it places*, so it can synthesise
near-patent sheets and emit, for free, the one thing real patents can't give:
per-pixel semantic ground truth AND exact keypoints.

This reproduces the three hardness axes a single-drawing dataset never can:
  * multi-panel sheet layout (a patent page IS a composite of figures),
  * geometric density (many primitives / sheet),
  * intra-sheet style heterogeneity (panels drawn from different datasets).
The patent-style traits (numerals+leaders, FIG labels, border, scan noise) are
layered on top, reusing train_puhachov.patent_degrade for the scan realism.

I/O
---
Input pools : directories of Puhachov-format npz {skeleton uint8 (H,W),
              kps int32 (N,3)=(x,y,type)} -- the output of the existing
              per-dataset converters (tools/archcad_to_puhachov.py, the D2C /
              SketchGraphs kp-label caches).
Output      : per sheet, an npz {skeleton uint8, kps int32 (N,3), semantic
              uint8 (H,W)} + a preview PNG.
  kps       : ONLY object-geometry keypoints, offset into sheet coords. The
              reference numerals / FIG labels / border are deliberately EXCLUDED
              from kps -- so a Puhachov model trained here learns numeral !=
              geometry, exactly the supervision that is otherwise unobtainable.
  semantic  : 0 background, 1 object geometry, 2 annotation text (numerals +
              FIG labels), 3 sheet border.

Scope
-----
v1 = NON-overlapping grid layout, so GT is the exact union of the panels'
labels under a pure translation (no keypoint resampling, no intersection
recomputation). Overlapping panels with junction-GT recomputation, and
synthetic dashed/hatch/dimension injectors, are v1.5/v2.

    python -m tools.patent_sheet_compositor \
        --pools output/Drawing2CAD/kp_labels output/SketchGraphsTraining/stage2/train \
                output/ArchCAD/kp_labels \
        --n-sheets 500 --out output/PatentSheets/train --seed 0
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "stage2_strokeextraction", "research"))
try:
    from train_puhachov import patent_degrade  # noqa: E402
except Exception:                              # keep the tool usable without torch
    patent_degrade = None

SEM_BG, SEM_OBJECT, SEM_TEXT, SEM_BORDER = 0, 1, 2, 3

# grid layouts (rows, cols) keyed by panel count -- portrait, patent-like
_LAYOUTS = {1: [(1, 1)], 2: [(2, 1), (1, 2)], 3: [(3, 1), (2, 2)],
            4: [(2, 2)], 5: [(3, 2)], 6: [(3, 2)]}


def load_pool(dirs: list[str]) -> list[str]:
    paths: list[str] = []
    for d in dirs:
        paths += glob.glob(os.path.join(d, "*.npz"))
    return sorted(paths)


def _load_panel(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        d = np.load(path, allow_pickle=True)
        if "skeleton" not in d or "kps" not in d:
            return None
        sk = (d["skeleton"] > 0).astype(np.uint8) * 255
        kps = np.asarray(d["kps"], dtype=np.int32).reshape(-1, 3)
        if not sk.any():
            return None
        return sk, kps
    except Exception:
        return None


def _fit_to_cell(sk: np.ndarray, kps: np.ndarray, cell: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Center a panel in a cell x cell tile (crop if larger, pad if smaller).
    Pure translation -> keypoints stay exactly on the skeleton."""
    h, w = sk.shape
    out = np.zeros((cell, cell), np.uint8)
    # source crop (if panel bigger than cell) centered
    sx0 = max(0, (w - cell) // 2); sy0 = max(0, (h - cell) // 2)
    cw = min(w, cell); ch = min(h, cell)
    dx0 = (cell - cw) // 2; dy0 = (cell - ch) // 2
    out[dy0:dy0 + ch, dx0:dx0 + cw] = sk[sy0:sy0 + ch, sx0:sx0 + cw]
    dxs = dx0 - sx0; dys = dy0 - sy0
    k = kps.copy()
    if len(k):
        k[:, 0] += dxs; k[:, 1] += dys
        inside = (k[:, 0] >= 0) & (k[:, 0] < cell) & (k[:, 1] >= 0) & (k[:, 1] < cell)
        k = k[inside]
    return out, k


def _stamp(semantic, region_slice, code):
    """Write a semantic code only where the mask is still background, so object
    geometry (written first) always wins over annotation/border overlap."""
    sub = semantic[region_slice]
    sub[sub == SEM_BG] = code


def _put_text(canvas, semantic, text, org, scale=0.6, thick=1):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick,
                cv2.LINE_AA)
    # stamp the text footprint into the semantic layer (object pixels preserved)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    _stamp(semantic, (slice(max(0, y - th - 2), y + base + 2),
                      slice(max(0, x - 2), x + tw + 2)), SEM_TEXT)


def _add_reference_numerals(canvas, semantic, kps, rng, cell_bbox, k=3):
    """Draw k reference numerals with short leaders pointing at object kps."""
    if len(kps) == 0:
        return
    x0, y0, x1, y1 = cell_bbox
    picks = rng.choice(len(kps), size=min(k, len(kps)), replace=False)
    for i in picks:
        kx, ky = int(kps[i][0]), int(kps[i][1])
        num = str(int(rng.integers(2, 400)))
        if rng.random() < 0.15:
            num += chr(int(rng.integers(0, 4)) + ord("a"))
        ang = rng.uniform(0, 2 * np.pi)
        dist = int(rng.integers(18, 40))
        tx = int(np.clip(kx + dist * np.cos(ang), x0 + 2, x1 - 24))
        ty = int(np.clip(ky + dist * np.sin(ang), y0 + 12, y1 - 4))
        cv2.line(canvas, (kx, ky), (tx, ty), 255, 1, cv2.LINE_AA)  # leader
        _put_text(canvas, semantic, num, (tx, ty), scale=0.5, thick=1)


def compose_sheet(panels, rng, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = cfg["cell"]; M = cfg["margin"]; G = cfg["gap"]; LH = cfg["label_h"]
    n = len(panels)
    rows, cols = _LAYOUTS[n][int(rng.integers(len(_LAYOUTS[n])))]
    # pad panel list to rows*cols by repeating (rare; keeps grid full)
    cell_h = cell + LH
    W = M * 2 + cols * cell + (cols - 1) * G
    H = M * 2 + rows * cell_h + (rows - 1) * G
    canvas = np.zeros((H, W), np.uint8)
    semantic = np.zeros((H, W), np.uint8)
    all_kps: list[np.ndarray] = []

    for idx in range(min(n, rows * cols)):
        r, c = divmod(idx, cols)
        x0 = M + c * (cell + G)
        y0 = M + r * (cell_h + G)
        sk, kps = _fit_to_cell(*panels[idx], cell)
        # paste object skeleton
        sub = canvas[y0:y0 + cell, x0:x0 + cell]
        obj = sk > 0
        sub[obj] = 255
        semantic[y0:y0 + cell, x0:x0 + cell][obj] = SEM_OBJECT
        # offset kps into sheet coords
        if len(kps):
            k = kps.copy(); k[:, 0] += x0; k[:, 1] += y0
            all_kps.append(k)
        # FIG label under the panel
        fig_no = idx + 1
        _put_text(canvas, semantic, f"FIG. {fig_no}",
                  (x0 + cell // 2 - 34, y0 + cell + LH - 6), scale=0.6, thick=2)
        # reference numerals with leaders
        if cfg["numerals"] and len(kps):
            k_sheet = kps.copy(); k_sheet[:, 0] += x0; k_sheet[:, 1] += y0
            _add_reference_numerals(canvas, semantic, k_sheet, rng,
                                    (x0, y0, x0 + cell, y0 + cell),
                                    k=int(rng.integers(2, 5)))

    # sheet border
    if cfg["border"]:
        b = M // 2
        cv2.rectangle(canvas, (b, b), (W - b, H - b), 255, 2)
        cv2.rectangle(semantic, (b, b), (W - b, H - b), SEM_BORDER, 2)

    # scan degrade (adds unlabeled clutter -> stays semantic bg, kps unaffected)
    if cfg["degrade"] and patent_degrade is not None:
        canvas = patent_degrade(canvas, rng)

    kps = (np.concatenate(all_kps, axis=0) if all_kps
           else np.zeros((0, 3), np.int32)).astype(np.int32)
    return canvas, kps, semantic


def _preview(sk, kps, semantic) -> np.ndarray:
    """Color preview: object=black, text=blue, border=green, kps=red dots."""
    H, W = sk.shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[sk > 0] = (40, 40, 40)
    img[semantic == SEM_TEXT] = (200, 120, 0)
    img[semantic == SEM_BORDER] = (0, 150, 0)
    for x, y, t in kps:
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 220), -1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True,
                    help="dirs of Puhachov-format npz {skeleton, kps}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-sheets", type=int, default=100)
    ap.add_argument("--min-panels", type=int, default=2)
    ap.add_argument("--max-panels", type=int, default=6)
    ap.add_argument("--cell", type=int, default=512)
    ap.add_argument("--margin", type=int, default=40)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--label-h", type=int, default=34)
    ap.add_argument("--no-numerals", action="store_true")
    ap.add_argument("--no-border", action="store_true")
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--previews", type=int, default=8,
                    help="how many preview PNGs to also write")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = load_pool(a.pools)
    if not pool:
        raise SystemExit(f"no npz panels found in pools: {a.pools}")
    print(f"panel pool: {len(pool)} drawings from {len(a.pools)} source(s)")
    a.out.mkdir(parents=True, exist_ok=True)
    prev_dir = a.out / "previews"; prev_dir.mkdir(exist_ok=True)

    cfg = dict(cell=a.cell, margin=a.margin, gap=a.gap, label_h=a.label_h,
               numerals=not a.no_numerals, border=not a.no_border,
               degrade=not a.no_degrade)
    rng = np.random.default_rng(a.seed)
    written = 0
    for i in range(a.n_sheets):
        n = int(rng.integers(a.min_panels, a.max_panels + 1))
        panels = []
        tries = 0
        while len(panels) < n and tries < n * 5:
            p = _load_panel(pool[int(rng.integers(len(pool)))])
            tries += 1
            if p is not None:
                panels.append(p)
        if len(panels) < a.min_panels:
            continue
        sk, kps, sem = compose_sheet(panels, rng, cfg)
        np.savez_compressed(a.out / f"sheet_{i:05d}.npz",
                            skeleton=sk, kps=kps, semantic=sem)
        if written < a.previews:
            cv2.imwrite(str(prev_dir / f"sheet_{i:05d}.png"),
                        _preview(sk, kps, sem))
        written += 1

    print(f"wrote {written} sheets -> {a.out}  ({a.previews} previews in {prev_dir})")


if __name__ == "__main__":
    main()
