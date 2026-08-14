"""Paired subgroup and outlier analysis for a Drawing2CAD Stage 3 policy."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "chamfer_sym": "lower",
    "chamfer_p95_sym": "lower",
    "iou_pixel": "higher",
    "iou_skeleton": "higher",
    "precision_pixel": "higher",
    "recall_pixel": "higher",
}
PRIMARY_METRICS = ("chamfer_sym", "chamfer_p95_sym", "iou_pixel")
_METRIC_TOLERANCE = 1e-12


def _load_rows(database: Path) -> dict[tuple[str, str], dict]:
    columns = ",".join(METRICS)
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            f"SELECT sample_id,view,{columns} FROM d2c_results "
            "WHERE status='ok'"
        ).fetchall()
    return {
        (row[0], row[1]): dict(zip(METRICS, row[2:]))
        for row in rows
    }


def _primitive_path(run: Path, sample_id: str, view: str) -> Path:
    folder = sample_id.replace("/", "_")
    sketch_id = f"{folder}_{view}"
    return run / folder / "primitives" / f"{sketch_id}_primitives.json"


def _classify(doc: dict) -> tuple[str, list[dict], list[dict]]:
    primitives = doc.get("primitives", [])
    promoted = [
        primitive for primitive in primitives
        if primitive.get("fit_metadata", {}).get("strategy")
        == "compound_over_weak"
    ]
    closed = [
        primitive for primitive in primitives
        if primitive.get("fit_metadata", {}).get("strategy")
        == "closed_simplified_trace"
    ]
    if promoted and closed:
        group = "both"
    elif closed:
        group = "closed"
    elif promoted:
        atom_types = {
            segment.get("type")
            for primitive in promoted
            for segment in primitive.get("segments", [])
        }
        group = "open_all_line" if atom_types == {"line"} else "open_nonline"
    else:
        group = "inactive"
    return group, promoted, closed


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _cluster_bootstrap_ci(
    records: list[dict],
    metric: str,
    samples: int,
    seed: int,
) -> list[float] | None:
    by_sample: dict[str, list[float]] = defaultdict(list)
    for record in records:
        delta = record["metrics"].get(metric, {}).get("delta")
        if delta is not None and math.isfinite(delta):
            by_sample[record["sample_id"]].append(delta)
    cluster_means = np.asarray(
        [np.mean(values) for values in by_sample.values()],
        dtype=np.float64,
    )
    if not len(cluster_means):
        return None
    if samples <= 0:
        value = float(cluster_means.mean())
        return [value, value]

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    offset = 0
    while offset < samples:
        block = min(200, samples - offset)
        indices = rng.integers(
            0, len(cluster_means), size=(block, len(cluster_means))
        )
        bootstrap_means[offset:offset + block] = cluster_means[indices].mean(axis=1)
        offset += block
    return [
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    ]


def _paired_metric_summary(records: list[dict], metric: str) -> dict:
    direction = METRICS[metric]
    pairs = [
        record["metrics"][metric]
        for record in records
        if metric in record["metrics"]
    ]
    if not pairs:
        return {"direction": direction, "n": 0}
    baseline = np.asarray([pair["baseline"] for pair in pairs])
    candidate = np.asarray([pair["candidate"] for pair in pairs])
    delta = candidate - baseline
    favorable = -delta if direction == "lower" else delta
    return {
        "direction": direction,
        "n": int(len(delta)),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_delta": float(delta.mean()),
        "better": int((favorable > _METRIC_TOLERANCE).sum()),
        "tie": int((np.abs(favorable) <= _METRIC_TOLERANCE).sum()),
        "worse": int((favorable < -_METRIC_TOLERANCE).sum()),
    }


def _metric_outcome(metrics: dict) -> dict:
    """Classify a paired row without privileging one fidelity metric."""
    better = []
    worse = []
    tied = []
    for metric, pair in metrics.items():
        delta = pair["delta"]
        favorable = -delta if METRICS[metric] == "lower" else delta
        if favorable > _METRIC_TOLERANCE:
            better.append(metric)
        elif favorable < -_METRIC_TOLERANCE:
            worse.append(metric)
        else:
            tied.append(metric)

    if better and worse:
        classification = "mixed"
    elif better:
        classification = "pareto_improved"
    elif worse:
        classification = "pareto_regressed"
    else:
        classification = "all_tied"
    return {
        "classification": classification,
        "better": better,
        "worse": worse,
        "tied": tied,
    }


def _is_worse(record: dict, metric: str) -> bool:
    pair = record["metrics"].get(metric)
    if pair is None:
        return False
    delta = pair["delta"]
    return (
        delta > _METRIC_TOLERANCE
        if METRICS[metric] == "lower"
        else delta < -_METRIC_TOLERANCE
    )


def _is_better(record: dict, metric: str) -> bool:
    pair = record["metrics"].get(metric)
    if pair is None:
        return False
    delta = pair["delta"]
    return (
        delta < -_METRIC_TOLERANCE
        if METRICS[metric] == "lower"
        else delta > _METRIC_TOLERANCE
    )


def _json_size(run: Path | None) -> int | None:
    if run is None:
        return None
    return sum(
        path.stat().st_size
        for path in run.glob("*/primitives/*_primitives.json")
    )


def analyze(
    baseline_db: Path,
    candidate_db: Path,
    candidate_run: Path,
    *,
    baseline_run: Path | None,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    baseline_rows = _load_rows(baseline_db)
    candidate_rows = _load_rows(candidate_db)
    shared_keys = sorted(baseline_rows.keys() & candidate_rows.keys())

    records = []
    group_counts: Counter[str] = Counter()
    activation = Counter()
    path_p95 = []
    endpoint_errors = []
    missing_primitives = []
    for sample_id, view in shared_keys:
        primitive_path = _primitive_path(candidate_run, sample_id, view)
        if not primitive_path.exists():
            missing_primitives.append(str(primitive_path))
            group, promoted, closed = "unknown", [], []
            quality = {}
        else:
            doc = json.loads(primitive_path.read_text())
            group, promoted, closed = _classify(doc)
            quality = doc.get("quality_metrics", {})
        group_counts[group] += 1
        activation["open_promotions"] += len(promoted)
        activation["open_atoms"] += sum(
            int(item.get("fit_metadata", {}).get("path_atoms", 0))
            for item in promoted
        )
        activation["closed_simplifications"] += len(closed)
        activation["closed_source_points"] += int(
            quality.get("n_closed_trace_source_points", 0)
        )
        activation["closed_output_vertices"] += int(
            quality.get("n_closed_trace_output_vertices", 0)
        )
        for primitive in promoted:
            fidelity = primitive.get("fit_fidelity", {})
            if fidelity.get("max_segment_p95") is not None:
                path_p95.append(float(fidelity["max_segment_p95"]))
            if fidelity.get("max_endpoint_error") is not None:
                endpoint_errors.append(float(fidelity["max_endpoint_error"]))

        metric_pairs = {}
        for metric in METRICS:
            baseline = baseline_rows[(sample_id, view)].get(metric)
            candidate = candidate_rows[(sample_id, view)].get(metric)
            if baseline is None or candidate is None:
                continue
            metric_pairs[metric] = {
                "baseline": float(baseline),
                "candidate": float(candidate),
                "delta": float(candidate - baseline),
            }
        record = {
            "sample_id": sample_id,
            "view": view,
            "group": group,
            "metrics": metric_pairs,
        }
        record["metric_outcome"] = _metric_outcome(metric_pairs)
        records.append(record)

    overall = {}
    for index, metric in enumerate(METRICS):
        overall[metric] = _paired_metric_summary(records, metric)
        overall[metric]["cluster_bootstrap_mean_delta_ci95"] = (
            _cluster_bootstrap_ci(
                records, metric, bootstrap_samples, seed + index,
            )
        )

    subgroups = {}
    for group in sorted(group_counts):
        group_records = [record for record in records if record["group"] == group]
        subgroups[group] = {
            "views": len(group_records),
            "metrics": {
                metric: _paired_metric_summary(group_records, metric)
                for metric in METRICS
            },
        }

    outliers = sorted(
        [record for record in records if "chamfer_sym" in record["metrics"]],
        key=lambda record: record["metrics"]["chamfer_sym"]["delta"],
        reverse=True,
    )
    chamfer_deltas = [
        record["metrics"]["chamfer_sym"]["delta"] for record in outliers
    ]
    metric_outcomes = Counter(
        record["metric_outcome"]["classification"] for record in records
    )
    chamfer_tail = [
        record for record in outliers
        if record["metrics"]["chamfer_sym"]["delta"] > 0.1
    ]
    baseline_size = _json_size(baseline_run)
    candidate_size = _json_size(candidate_run)
    report = {
        "baseline_db": str(baseline_db),
        "candidate_db": str(candidate_db),
        "baseline_run": str(baseline_run) if baseline_run else None,
        "candidate_run": str(candidate_run),
        "baseline_ok": len(baseline_rows),
        "candidate_ok": len(candidate_rows),
        "paired": len(records),
        "missing_primitive_json": missing_primitives[:100],
        "overall": overall,
        "groups": dict(group_counts),
        "subgroups": subgroups,
        "activation": {
            **dict(activation),
            "closed_compression_ratio": (
                activation["closed_source_points"]
                / activation["closed_output_vertices"]
                if activation["closed_output_vertices"] else None
            ),
            "promoted_path_p95": _summary(path_p95),
            "promoted_endpoint_error": _summary(endpoint_errors),
        },
        "primitive_json_bytes": {
            "baseline": baseline_size,
            "candidate": candidate_size,
            "relative_reduction": (
                1.0 - candidate_size / baseline_size
                if baseline_size and candidate_size is not None else None
            ),
        },
        "outlier_counts": {
            "measurable": len(chamfer_deltas),
            "improved": sum(delta < -1e-12 for delta in chamfer_deltas),
            "tied": sum(abs(delta) <= 1e-12 for delta in chamfer_deltas),
            "regressed": sum(delta > 1e-12 for delta in chamfer_deltas),
            "regressed_gt_0_1": sum(delta > 0.1 for delta in chamfer_deltas),
            "regressed_gt_0_25": sum(delta > 0.25 for delta in chamfer_deltas),
            "regressed_gt_1": sum(delta > 1.0 for delta in chamfer_deltas),
        },
        "tradeoff_counts": {
            "all_metrics": dict(metric_outcomes),
            "mean_chamfer_gt_0_1": {
                "views": len(chamfer_tail),
                "p95_better": sum(
                    _is_better(record, "chamfer_p95_sym")
                    for record in chamfer_tail
                ),
                "pixel_iou_better": sum(
                    _is_better(record, "iou_pixel")
                    for record in chamfer_tail
                ),
                "p95_and_pixel_iou_better": sum(
                    _is_better(record, "chamfer_p95_sym")
                    and _is_better(record, "iou_pixel")
                    for record in chamfer_tail
                ),
                "all_three_primary_worse": sum(
                    all(_is_worse(record, metric) for metric in PRIMARY_METRICS)
                    for record in chamfer_tail
                ),
            },
        },
        "worst_chamfer": outliers[:50],
        "best_chamfer": list(reversed(outliers[-50:])),
        "bootstrap": {
            "samples": bootstrap_samples,
            "unit": "Drawing2CAD sample_id cluster",
            "seed": seed,
        },
    }
    return report, outliers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-db", type=Path, required=True)
    parser.add_argument("--candidate-db", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    report, outliers = analyze(
        args.baseline_db,
        args.candidate_db,
        args.candidate_run,
        baseline_run=args.baseline_run,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    csv_path = args.output.with_suffix(".outliers.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_id", "view", "group", "baseline_chamfer",
            "candidate_chamfer", "delta_chamfer",
        ])
        for record in outliers:
            metric = record["metrics"]["chamfer_sym"]
            writer.writerow([
                record["sample_id"], record["view"], record["group"],
                metric["baseline"], metric["candidate"], metric["delta"],
            ])
    print(json.dumps({
        "output": str(args.output),
        "outliers_csv": str(csv_path),
        "paired": report["paired"],
        "overall": report["overall"],
        "groups": report["groups"],
        "outlier_counts": report["outlier_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
