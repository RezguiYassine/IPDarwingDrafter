#!/usr/bin/env python3
"""Learn the drawing / not-drawing decision from the curated labels.

Content routing needs an authority that says whether a figure is an
engineering drawing. Today that authority is a person, one figure at a time,
which does not reach 15,600 patents; and the corpus filter cannot stand in for
one, since 17 of the 58 replacements it offered during curation were rejected,
a false-positive rate near 29%.

The curation session left 158 decisions with reasons. This trains a linear
probe on frozen ImageNet features from them -- a small model is the right size
for 158 examples -- and reports precision as a function of decision margin, so
the contract can accept only confident predictions and route the rest to a
person.

    python -m tools.content_classifier --train --session benchmarks/curatedv1
    python -m tools.content_classifier --predict data/PatentData/ReorganisedData \
        --model models/content_classifier_v1.pt --output output/PatentData/predictions.json

Reports cross-validated accuracy; a model that has only seen its own training
set is not evidence, so the margin threshold is chosen on held-out folds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SCHEMA = "ap3-content-classifier-v1"
VERSION = "1"
IMAGE_SIZE = 224


def _backbone(device: str):
    import torch
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()          # frozen features, 512-d
    return model.eval().to(device), weights.transforms()


def embed(paths: list[Path], device: str = "cuda:0", batch: int = 32) -> np.ndarray:
    """Frozen 512-d features. A drawing and a flowchart differ globally, which
    is what a generic backbone represents well; fine-tuning 11M parameters on
    158 examples would only memorise them."""
    import torch
    import cv2
    from PIL import Image
    model, transform = _backbone(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(paths), batch):
            tensors = []
            for path in paths[start:start + batch]:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    tensors.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                tensors.append(transform(Image.fromarray(rgb)))
            out.append(model(torch.stack(tensors).to(device)).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 512), np.float32)


def load_labels(session: Path, *, perspective_is_drawing: bool = False
                ) -> tuple[list[Path], np.ndarray, list[str], list[str]]:
    """Read a curate_pilot session: accept -> in scope, reject -> out of scope.

    Works from session.json, which carries every figure's path, so it reads
    both the original curated set and later sessions without a separate
    content manifest.

    `perspective_is_drawing` reclassifies the shaded_render_or_photo rejects
    as in scope. On the mechanical session those are not photographs but
    perspective and isometric line art -- motorcycles, aircraft, exploded
    assemblies -- with an ink fraction of 0.045 against 0.052 for accepted
    drawings. Whether they belong in the corpus is a policy choice, so both
    readings can be trained and compared rather than one being assumed.
    """
    state = json.loads((session / "session.json").read_text())
    by_key = {f"{item['patent']}/{Path(item['filename']).stem}": item for item in state["queue"]}
    paths, labels, keys, reasons = [], [], [], []
    for key, decision in state["decisions"].items():
        if decision["status"] not in {"accept", "reject"}:
            continue
        item = by_key.get(key)
        if item is None or not Path(item["path"]).is_file():
            continue
        reason = decision.get("reason") or ""
        in_scope = decision["status"] == "accept"
        if perspective_is_drawing and reason == "shaded_render_or_photo":
            in_scope = True
        paths.append(Path(item["path"]))
        labels.append(in_scope)
        keys.append(key)
        reasons.append(reason)
    return paths, np.array(labels, bool), keys, reasons


def _fit(features: np.ndarray, labels: np.ndarray, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                               random_state=seed)
    model.fit(scaler.transform(features), labels)
    return scaler, model


def cross_validate(features: np.ndarray, labels: np.ndarray, folds: int = 5) -> dict:
    """Out-of-fold predictions, so every score is on data the model never saw."""
    from sklearn.model_selection import StratifiedKFold
    scores = np.zeros(len(labels))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    for train, test in splitter.split(features, labels):
        scaler, model = _fit(features[train], labels[train])
        scores[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    predicted = scores >= 0.5
    accuracy = float((predicted == labels).mean())
    margins = np.abs(scores - 0.5) * 2
    by_margin = []
    for threshold in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
        keep = margins >= threshold
        if keep.sum() == 0:
            continue
        by_margin.append({
            "margin": threshold,
            "covered": float(keep.mean()),
            "accuracy": float((predicted[keep] == labels[keep]).mean()),
            "n": int(keep.sum()),
        })
    return {"accuracy": accuracy, "n": len(labels), "scores": scores.tolist(),
            "by_margin": by_margin}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=Path("benchmarks/curatedv1"))
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--model", type=Path, default=Path("models/content_classifier_v1.npz"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--evidence", type=Path)
    ap.add_argument("--extra-session", type=Path, action="append", default=[],
                    help="additional curate_pilot session directories to train on")
    ap.add_argument("--perspective-is-drawing", action="store_true",
                    help="treat shaded_render_or_photo rejects as in scope")
    args = ap.parse_args()

    paths, labels, keys, _ = load_labels(args.session,
                                         perspective_is_drawing=args.perspective_is_drawing)
    for extra in args.extra_session:
        p2, l2, k2, _ = load_labels(extra, perspective_is_drawing=args.perspective_is_drawing)
        paths += p2
        labels = np.concatenate([labels, l2])
        keys += k2
    print(f"labelled figures: {len(paths)}  in scope {int(labels.sum())}  "
          f"out of scope {int((~labels).sum())}")
    features = embed(paths, args.device)
    print(f"features: {features.shape}")

    report = cross_validate(features, labels)
    print(f"\ncross-validated accuracy: {report['accuracy']:.1%}  (n={report['n']})")
    print(f"{'margin':>8}{'covered':>10}{'accuracy':>10}{'n':>6}")
    for row in report["by_margin"]:
        print(f"{row['margin']:>8.1f}{row['covered']:>10.1%}{row['accuracy']:>10.1%}{row['n']:>6}")

    if args.train:
        scaler, model = _fit(features, labels)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.model, mean=scaler.mean_, scale=scaler.scale_,
                 coef=model.coef_, intercept=model.intercept_,
                 schema=SCHEMA, version=VERSION)
        print(f"\nmodel written: {args.model}")
    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(
            {"schema": SCHEMA, "version": VERSION, "session": str(args.session),
             "n": report["n"], "accuracy": report["accuracy"],
             "by_margin": report["by_margin"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
