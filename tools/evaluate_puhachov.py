"""Evaluate a Stage-2 Puhachov checkpoint on cached keypoint labels."""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage2_strokeextraction"))

import stage2_stroke_extract as s2  # noqa: E402
from stage2_strokeextraction.research.train_puhachov import evaluate_f1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-2 endpoint/junction/corner peak F1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=3)
    parser.add_argument("--match-radius", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(Path(p) for p in glob.glob(
        str(args.labels / args.split / "**" / "*.npz"), recursive=True
    ))
    if not paths:
        raise SystemExit(f"no labels under {args.labels / args.split}")
    random.Random(args.seed).shuffle(paths)
    if args.limit > 0:
        paths = paths[:args.limit]

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = s2._build_stacked_hourglass().to(args.device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    metrics = evaluate_f1(
        model, paths, args.device, conf=args.conf,
        nms_radius=args.nms_radius, match_radius=args.match_radius,
    )
    names = ("endpoint", "junction", "corner")
    report = {
        "checkpoint": str(args.checkpoint), "labels": str(args.labels),
        "split": args.split, "n_samples": len(paths),
        "macro_f1": metrics["macro_f1"],
        "classes": {
            name: {
                "precision": float(metrics["prec"][i]),
                "recall": float(metrics["rec"][i]),
                "f1": float(metrics["f1"][i]),
                "support": int(metrics["support"][i]),
            }
            for i, name in enumerate(names)
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
