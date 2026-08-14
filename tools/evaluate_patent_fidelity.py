"""Post-hoc raster fidelity evaluation for paired PatentData runs.

The shared Stage-1 skeleton is treated as the raster reference. Stage-2 graph
pixels (including the separated hachure layer) and Stage-3 primitives are
compared in the original image frame with symmetric Chamfer and tolerance-based
precision/recall. This is not vector ground truth, but it detects geometry loss
that topology-only metrics can accidentally reward.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import cairosvg
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage4_export"))
import stage4_export  # noqa: E402

from tools.compare_patent_runs import _bootstrap_mean_ci  # noqa: E402


METRIC_DIRECTIONS = {
    "chamfer_sym": "lower",
    "chamfer_p95": "lower",
    "precision_at_2px": "higher",
    "recall_at_2px": "higher",
    "f1_at_2px": "higher",
}


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=RUN_DIR")
    name, raw_path = value.split("=", 1)
    run_dir = Path(raw_path)
    if not name or not run_dir.joinpath("results.db").exists():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return name, run_dir


def _load_rows(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with sqlite3.connect(run_dir / "results.db") as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM results").fetchall()
    return {
        (str(row["patent_id"]), str(row["sketch_id"])): dict(row)
        for row in rows
    }


def _read_skeleton(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image > 0


def _graph_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    graph = json.loads(path.read_text())
    scale = float(graph.get("stage2_scale", 1.0) or 1.0)
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for collection in ("edges", "removed_hachures"):
        for edge in graph.get(collection, []):
            for x, y in edge.get("pixels", []):
                xi = int(round(float(x) / scale))
                yi = int(round(float(y) / scale))
                if 0 <= xi < width and 0 <= yi < height:
                    mask[yi, xi] = True
    return mask


def _primitive_mask(
    primitives_path: Path,
    output_dir: Path,
    sketch_id: str,
    shape: tuple[int, int],
) -> np.ndarray:
    result = stage4_export.run(
        input_json=primitives_path,
        output_dir=output_dir,
        sketch_id=sketch_id,
        formats=("svg",),
        dxf_mode="basic",
    )
    height, width = shape
    png = cairosvg.svg2png(
        url=str(result.svg_path),
        output_width=width,
        output_height=height,
        background_color="white",
    )
    grayscale = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    return skeletonize(grayscale < 200)


def fidelity(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float] | None:
    reference = reference.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)
    if not reference.any() or not prediction.any():
        return None
    distance_to_reference = distance_transform_edt(~reference)
    distance_to_prediction = distance_transform_edt(~prediction)
    reference_to_prediction = distance_to_prediction[reference]
    prediction_to_reference = distance_to_reference[prediction]
    precision = float((prediction_to_reference <= 2.0).mean())
    recall = float((reference_to_prediction <= 2.0).mean())
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    distances = np.concatenate((reference_to_prediction, prediction_to_reference))
    return {
        "chamfer_sym": float(
            0.5 * (reference_to_prediction.mean() + prediction_to_reference.mean())
        ),
        "chamfer_p95": float(np.percentile(distances, 95)),
        "precision_at_2px": precision,
        "recall_at_2px": recall,
        "f1_at_2px": float(f1),
    }


def evaluate(
    runs: list[tuple[str, Path]],
    *,
    seed: int = 850725,
    bootstrap_samples: int = 10000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = {name: _load_rows(path) for name, path in runs}
    run_dirs = dict(runs)
    run_names = [name for name, _path in runs]
    baseline_name = run_names[0]
    shared_keys = sorted(set.intersection(*(set(loaded[name]) for name in run_names)))
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="patent_fidelity_") as temp_value:
        temp_root = Path(temp_value)
        for patent_id, sketch_id in shared_keys:
            reference_path = (
                run_dirs[baseline_name] / patent_id / "cleaned"
                / f"{sketch_id}_skeleton.png"
            )
            if not reference_path.exists():
                continue
            reference = _read_skeleton(reference_path)
            for run_name in run_names:
                run_dir = run_dirs[run_name]
                record: dict[str, Any] = {
                    "run": run_name,
                    "patent_id": patent_id,
                    "sketch_id": sketch_id,
                    "status": loaded[run_name][(patent_id, sketch_id)]["status"],
                }
                graph_path = (
                    run_dir / patent_id / "graphs" / f"{sketch_id}_graph.json"
                )
                if graph_path.exists():
                    metrics = fidelity(reference, _graph_mask(graph_path, reference.shape))
                    if metrics:
                        record.update({f"stage2_{key}": value for key, value in metrics.items()})
                primitives_path = (
                    run_dir / patent_id / "primitives"
                    / f"{sketch_id}_primitives.json"
                )
                if primitives_path.exists():
                    render_dir = temp_root / run_name / patent_id
                    metrics = fidelity(
                        reference,
                        _primitive_mask(
                            primitives_path, render_dir, sketch_id, reference.shape,
                        ),
                    )
                    if metrics:
                        record.update({f"stage3_{key}": value for key, value in metrics.items()})
                records.append(record)

    report: dict[str, Any] = {
        "reference_run": baseline_name,
        "shared_db_rows": len(shared_keys),
        "runs": {},
        "comparisons": {},
    }
    fields = [
        f"{stage}_{metric}"
        for stage in ("stage2", "stage3")
        for metric in METRIC_DIRECTIONS
    ]
    by_run = {
        name: [record for record in records if record["run"] == name]
        for name in run_names
    }
    for name in run_names:
        report["runs"][name] = {
            "rows": len(by_run[name]),
            "metrics": {
                field: {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
                for field in fields
                if (values := [
                    float(record[field]) for record in by_run[name] if field in record
                ])
            },
        }

    baseline_records = {
        (record["patent_id"], record["sketch_id"]): record
        for record in by_run[baseline_name]
    }
    for candidate_index, candidate_name in enumerate(run_names[1:], start=1):
        candidate_records = {
            (record["patent_id"], record["sketch_id"]): record
            for record in by_run[candidate_name]
        }
        comparison: dict[str, Any] = {"metrics": {}}
        for field_index, field in enumerate(fields):
            keys = [
                key for key in sorted(set(baseline_records) & set(candidate_records))
                if field in baseline_records[key] and field in candidate_records[key]
            ]
            if not keys:
                continue
            base = np.asarray([baseline_records[key][field] for key in keys], float)
            cand = np.asarray([candidate_records[key][field] for key in keys], float)
            delta = cand - base
            metric_name = field.split("_", 1)[1]
            direction = METRIC_DIRECTIONS[metric_name]
            ties = np.isclose(base, cand, rtol=1e-9, atol=1e-12)
            better = cand > base if direction == "higher" else cand < base
            comparison["metrics"][field] = {
                "direction": direction,
                "n": len(keys),
                "baseline_mean": float(base.mean()),
                "candidate_mean": float(cand.mean()),
                "mean_delta": float(delta.mean()),
                "mean_delta_ci95": _bootstrap_mean_ci(
                    delta,
                    seed + candidate_index * 1000 + field_index,
                    bootstrap_samples,
                ),
                "better_count": int(np.logical_and(better, ~ties).sum()),
                "tie_count": int(ties.sum()),
                "worse_count": int((~better & ~ties).sum()),
            }
        report["comparisons"][candidate_name] = comparison
    return report, records


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = sorted(set().union(*(record.keys() for record in records))) if records else []
    preferred = ["run", "patent_id", "sketch_id", "status"]
    fields = preferred + [field for field in fields if field not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=850725)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("at least two --run arguments are required")
    if len(dict(args.run)) != len(args.run):
        parser.error("run names must be unique")
    report, records = evaluate(
        args.run,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    _write_csv(args.output_csv, records)
    print(f"shared DB rows: {report['shared_db_rows']}")
    for name, run_report in report["runs"].items():
        print(f"{name}: evaluated rows={run_report['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
