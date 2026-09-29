#!/usr/bin/env python3
"""Phase N2: how much of each hatch *style* does the shipped detector find?

Phase C's 98.2% hatch recall was measured on synthetic sheets drawn at one
thin stroke width, so it says nothing about bold bands, shading or solid
hatching. The owner's N1 masks tag every drawn hatch area with its style;
this tool runs HatchUNet over those figures and reports, per style, how much
of the area's *ink* the detector marks -- ink, because an area is mostly
paper and the detector's job is the strokes.

    python -m tools.hatch_style_recall \
        --gt-dir output/PatentData/hatch_gt_styles \
        --model models/hatch_unet.pth --output docs/audits/<date>/hatch_style_recall.json

GPU 0 runs the owner's simulator: the default device is cuda:1, and cuda:0
is refused.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "ap3-hatch-style-recall-v1"
TARGET_RECALL = 0.70          # R4, a hypothesis until this tool has run


def area_scores(prediction: np.ndarray, ink: np.ndarray, polygons: list, styles: list) -> list[dict]:
    """Per drawn area: its style, ink pixels, and the share the detector marks."""
    rows = []
    for index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        region = np.zeros(ink.shape, np.uint8)
        cv2.fillPoly(region, [np.asarray(polygon, np.int32).reshape(-1, 1, 2)], 1)
        region_ink = (region > 0) & ink
        n = int(region_ink.sum())
        if n == 0:
            continue
        rows.append({"index": index, "style": styles[index] if index < len(styles) else "thin",
                     "ink_px": n, "recall": float((prediction & region_ink).sum() / n)})
    return rows


def summarise(rows: list[dict]) -> dict:
    by_style = collections.defaultdict(list)
    for row in rows:
        by_style[row["style"]].append(row)
    summary = {}
    for style, items in sorted(by_style.items()):
        recall = np.array([r["recall"] for r in items])
        weights = np.array([r["ink_px"] for r in items], float)
        summary[style] = {
            "areas": len(items), "figures": len({r["figure"] for r in items}),
            "recall_median": float(np.median(recall)),
            "recall_ink_weighted": float((recall * weights).sum() / weights.sum()),
            "areas_at_target": int((recall >= TARGET_RECALL).sum()),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--model", default="models/hatch_unet.pth")
    ap.add_argument("--threshold", type=float, default=0.7, help="shipped operating point")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.device.strip() in {"cuda", "cuda:0"}:
        raise SystemExit("cuda:0 is the simulator's GPU; use --device cuda:1 or cpu")

    import torch
    from tools.hatch_dataset import _find_tif, _DEFAULT_TIF_ROOT
    from tools.hatch_infer import _load_model, infer_tif
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _load_model(args.model, device)

    rows, figures = [], 0
    for path in sorted(glob.glob(os.path.join(args.gt_dir, "*_mask.json"))):
        document = json.loads(Path(path).read_text())
        if not document.get("reviewed") or not document.get("polygons"):
            continue
        tif = _find_tif(document["patent"], document["sketch"], _DEFAULT_TIF_ROOT)
        if tif is None:
            print(f"warning: no TIF for {document['patent']}__{document['sketch']}")
            continue
        gray = cv2.imread(tif, cv2.IMREAD_GRAYSCALE)
        _, binary = infer_tif(model, tif, device, threshold=args.threshold)
        figure = f"{document['patent']}/{document['sketch']}"
        for row in area_scores(binary > 0, gray < 128, document["polygons"],
                               document.get("styles") or []):
            rows.append({**row, "figure": figure})
        figures += 1

    summary = summarise(rows)
    print(f"figures {figures}, areas {len(rows)}; threshold {args.threshold}")
    for style, s in summary.items():
        print(f"  {style:8s} areas {s['areas']:3d}  figures {s['figures']:3d}  "
              f"recall median {s['recall_median']:.2f}  ink-weighted {s['recall_ink_weighted']:.2f}  "
              f"at >= {TARGET_RECALL:.2f}: {s['areas_at_target']}/{s['areas']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema": SCHEMA, "model": args.model,
                                           "threshold": args.threshold, "summary": summary,
                                           "areas": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
