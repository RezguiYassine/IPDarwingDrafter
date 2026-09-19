#!/usr/bin/env python3
"""Score fitted primitives against known vector ground truth.

Every fidelity number this project has quoted -- raster F1, Chamfer, the
geometry validator's coverage rules -- measures agreement between Stage 3 and
the Stage-1 skeleton that Stage 2 derived from. That is self-consistency. It
cannot distinguish a faithful reconstruction from a confident wrong one,
because both stages see the same evidence and share the same mistakes.

Synthetic sheets carry the geometry they were drawn from, so here a primitive
can be scored against what is actually there:

  recall     a ground-truth primitive is FOUND when some fitted primitive
             follows it within `tolerance` along at least `overlap` of its length
  precision  a fitted primitive is SUPPORTED when it lies within `tolerance` of
             some ground-truth primitive along at least `overlap` of its length

Both directions are needed. Recall alone rewards covering the drawing with
noise; precision alone rewards drawing nothing.

    python -m tools.gt_fidelity --run output/PhaseC_run --truth data/SyntheticGT/_ground_truth
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tools import geometry_validation as gv

SCHEMA = "ap3-gt-fidelity-v1"
VERSION = "1"


def _sample_truth(primitive: dict, scale: float, step: float = 1.0) -> np.ndarray | None:
    """Ground-truth geometry in pixels. Normalised coordinates, canvas-scaled."""
    geometry = primitive.get("geometry") or {}
    kind = (primitive.get("kind") or primitive.get("primitive_type")
            or primitive.get("type") or "").lower()
    if "p0" in geometry and "p1" in geometry and not kind.startswith("arc"):
        a = np.asarray(geometry["p0"], float) * scale
        b = np.asarray(geometry["p1"], float) * scale
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        return np.linspace(a, b, n)
    if kind in {"cubic_bezier", "bezier"} and "points" in geometry:
        control = np.asarray(geometry["points"], float) * scale
        if len(control) >= 4:
            t = np.linspace(0, 1, 64)[:, None]
            c = control[:4]
            return ((1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1]
                    + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3])
    if "points" in geometry:
        points = np.asarray(geometry["points"], float) * scale
        if len(points) < 2:
            return None
        lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if lengths[-1] < 1e-9:
            return None
        n = max(2, int(math.ceil(lengths[-1] / step)) + 1)
        t = np.linspace(0, lengths[-1], n)
        return np.column_stack([np.interp(t, lengths, points[:, i]) for i in (0, 1)])
    if "center" in geometry and ("radius" in geometry or "r" in geometry):
        centre = np.asarray(geometry["center"], float) * scale
        radius = float(geometry.get("radius", geometry.get("r"))) * scale
        start = math.radians(float(geometry.get("start_angle", 0.0)))
        end = math.radians(float(geometry.get("end_angle", 360.0)))
        sweep = (end - start) % (2 * math.pi) or 2 * math.pi
        n = max(8, int(math.ceil(radius * sweep / step)) + 1)
        angles = np.linspace(start, start + sweep, n)
        return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    return None


def _covered_fraction(points: np.ndarray, tree: cKDTree, tolerance: float) -> float:
    if tree is None or len(points) == 0:
        return 0.0
    return float((tree.query(points)[0] <= tolerance).mean())


def score(truth: dict, primitives: list[dict], *, tolerance: float = 3.0,
          overlap: float = 0.80, step: float = 1.0,
          drop_layers: tuple = ("reference_numeral", "text", "text_box")) -> dict:
    """Precision and recall over one sheet.

    Reference numerals and captions are excluded from the ground truth: Stage 0
    removes them deliberately, so counting them as missed geometry would score
    the pipeline down for doing its job.
    """
    canvas = truth.get("canvas") or [1024, 1024]
    scale = float(canvas[0])
    truth_samples, truth_types = [], []
    for primitive in truth.get("primitives_visible") or []:
        semantic = primitive.get("semantic") or primitive.get("semantic_layer")
        if semantic in drop_layers or (primitive.get("kind") or "") == "text":
            continue
        points = _sample_truth(primitive, scale, step)
        if points is not None and len(points) >= 2:
            truth_samples.append(points)
            truth_types.append(semantic or primitive.get("kind") or "unknown")
    fitted_samples = []
    for primitive in primitives:
        try:
            sampled, _ = gv.sample_primitive(primitive, 1.0, gv.POLICY)
        except Exception:
            continue
        if len(sampled) >= 2:
            fitted_samples.append(np.asarray(sampled, float))

    fitted_tree = cKDTree(np.vstack(fitted_samples)) if fitted_samples else None
    truth_tree = cKDTree(np.vstack(truth_samples)) if truth_samples else None

    found = [_covered_fraction(p, fitted_tree, tolerance) >= overlap for p in truth_samples]
    supported = [_covered_fraction(p, truth_tree, tolerance) >= overlap for p in fitted_samples]
    recall = float(np.mean(found)) if found else float("nan")
    precision = float(np.mean(supported)) if supported else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 and not math.isnan(precision + recall) else float("nan"))

    by_type = collections.defaultdict(lambda: [0, 0])
    for kind, hit in zip(truth_types, found):
        by_type[kind][0] += int(hit)
        by_type[kind][1] += 1
    return {
        "truth_primitives": len(truth_samples), "fitted_primitives": len(fitted_samples),
        "recall": recall, "precision": precision, "f1": f1,
        "missed": int(sum(1 for f in found if not f)),
        "unsupported": int(sum(1 for s in supported if not s)),
        "recall_by_type": {k: {"found": v[0], "total": v[1]} for k, v in sorted(by_type.items())},
        "policy": {"tolerance": tolerance, "overlap": overlap, "step": step},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--overlap", type=float, default=0.80)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    rows = []
    for primitives_path in sorted(args.run.glob("*/primitives/*_primitives.json")):
        sample_id = primitives_path.parent.parent.name
        truth_path = args.truth / f"{sample_id}.json"
        if not truth_path.is_file():
            continue
        truth = json.loads(truth_path.read_text())
        document = json.loads(primitives_path.read_text())
        result = score(truth, document.get("primitives") or [],
                       tolerance=args.tolerance, overlap=args.overlap)
        result["sample_id"] = sample_id
        result["difficulty"] = truth.get("difficulty")
        rows.append(result)

    if not rows:
        print("no scored sheets; check --run and --truth")
        return 1
    recall = np.array([r["recall"] for r in rows], float)
    precision = np.array([r["precision"] for r in rows], float)
    f1 = np.array([r["f1"] for r in rows], float)
    print(f"sheets scored: {len(rows)}")
    for name, values in (("recall", recall), ("precision", precision), ("f1", f1)):
        good = values[~np.isnan(values)]
        print(f"  {name:10s} median {np.median(good):.4f}  p10 {np.percentile(good, 10):.4f}"
              f"  mean {good.mean():.4f}")
    by_difficulty = collections.defaultdict(list)
    for r in rows:
        by_difficulty[r["difficulty"]].append(r["f1"])
    print("\n  by difficulty:")
    for level, values in sorted(by_difficulty.items()):
        clean = [v for v in values if not math.isnan(v)]
        if clean:
            print(f"    {str(level):12s} n={len(clean):4d}  median F1 {np.median(clean):.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"schema": SCHEMA, "version": VERSION, "run": str(args.run),
             "policy": {"tolerance": args.tolerance, "overlap": args.overlap},
             "sheets": rows}, indent=2) + "\n")
        print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
