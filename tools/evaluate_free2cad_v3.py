"""Evaluate an encoder-only Free2CAD v3 checkpoint on a prepared split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from stage3_primitivesfitting.research.train_free2cad_v3 import (
    build_model,
    evaluate,
    load_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Free2CAD v3 type and normalized-parameter accuracy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--param-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if int(checkpoint.get("version", 0)) != 3:
        raise SystemExit("checkpoint is not an encoder-only Free2CAD v3 model")
    cfg = checkpoint["config"]
    model = build_model(
        max_pts=cfg["max_pts"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_enc_layers=cfg["n_enc_layers"],
        dropout=cfg.get("dropout", 0.1),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    arc_encoding = cfg.get("arc_encoding", "center_radius_angles")
    dataset = load_dataset(
        args.data_dir, args.split, cfg["max_pts"], arc_encoding)
    weights = np.asarray(checkpoint.get("class_weights", [1.0] * 4), dtype=np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(args.device))
    metrics = evaluate(
        model, dataset, torch.device(args.device), args.batch_size,
        loss_fn, args.param_weight, arc_encoding,
    )
    report = {
        "checkpoint": str(args.checkpoint), "data_dir": args.data_dir,
        "split": args.split, "n_samples": len(dataset), **metrics,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
