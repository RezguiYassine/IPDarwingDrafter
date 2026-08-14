"""Paired intrinsic comparison of PatentData vectorization runs."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "s2_n_nodes": None,
    "s2_n_edges": None,
    "s2_n_closed_edges": None,
    "s2_n_hachure_edges_removed": None,
    "s2_median_edge_len": "higher",
    "s2_micro_edge_ratio": "lower",
    "s2_short_edge_ratio": "lower",
    "s2_isolation": "lower",
    "s3_n_primitives": None,
    "s3_n_hachure_primitives": None,
    "s3_mean_conf": "higher",
    "s3_low_conf_ratio": "lower",
}

PREPROCESSING_FIELDS = (
    "s0_n_labels",
    "s0_n_leaders",
    "s0_n_iterations",
    "s0_removed_ink_ratio",
    "s0_active_removal",
    "s0_flagged",
    "s1_quality",
    "s1_model_used",
    "s1_flagged",
)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=RUN_DIR_OR_DB")
    name, raw_path = value.split("=", 1)
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(
            "run name must contain only letters, digits, '_' or '-'"
        )
    path = Path(raw_path)
    db_path = path if path.suffix == ".db" else path / "results.db"
    if not db_path.exists():
        raise argparse.ArgumentTypeError(f"results DB does not exist: {db_path}")
    return name, db_path


def _load(db_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM results").fetchall()
    return {
        (str(row["patent_id"]), str(row["sketch_id"])): dict(row)
        for row in rows
    }


def _bootstrap_mean_ci(
    delta: np.ndarray,
    seed: int,
    samples: int,
) -> list[float]:
    if delta.size == 1:
        value = float(delta[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 1000
    for start in range(0, samples, batch):
        size = min(batch, samples - start)
        indices = rng.integers(0, delta.size, size=(size, delta.size))
        means[start:start + size] = delta[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1e-9, atol=1e-12))
    return left == right


def compare(
    runs: list[tuple[str, Path]],
    *,
    seed: int = 850725,
    bootstrap_samples: int = 10000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    loaded = {name: _load(path) for name, path in runs}
    baseline_name = runs[0][0]
    baseline = loaded[baseline_name]
    all_shared = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    report: dict[str, Any] = {
        "baseline": baseline_name,
        "runs": {},
        "shared_all_runs": len(all_shared),
        "comparisons": {},
    }
    for name, path in runs:
        rows = loaded[name]
        report["runs"][name] = {
            "db": str(path),
            "rows": len(rows),
            "status_counts": dict(sorted(Counter(
                str(row["status"]) for row in rows.values()
            ).items())),
            "keypoint_source_counts": dict(sorted(Counter(
                str(row["s2_keypoint_src"])
                for row in rows.values()
                if row.get("s2_keypoint_src") is not None
            ).items())),
        }

    csv_rows: list[dict[str, Any]] = []
    for key in all_shared:
        patent_id, sketch_id = key
        record: dict[str, Any] = {
            "patent_id": patent_id,
            "sketch_id": sketch_id,
            "input_path": baseline[key].get("input_path"),
        }
        for name, _path in runs:
            row = loaded[name][key]
            record[f"{name}_status"] = row.get("status")
            for metric in METRICS:
                record[f"{name}_{metric}"] = row.get(metric)
        csv_rows.append(record)

    for candidate_index, (candidate_name, _path) in enumerate(runs[1:], start=1):
        candidate = loaded[candidate_name]
        keys = sorted(set(baseline) & set(candidate))
        transitions = Counter(
            f"{baseline[key]['status']} -> {candidate[key]['status']}"
            for key in keys
        )
        preprocessing_mismatches = []
        for key in keys:
            changed = [
                field for field in PREPROCESSING_FIELDS
                if not _same_value(baseline[key].get(field), candidate[key].get(field))
            ]
            if changed:
                preprocessing_mismatches.append({
                    "patent_id": key[0],
                    "sketch_id": key[1],
                    "fields": changed,
                })

        comparison: dict[str, Any] = {
            "paired_rows": len(keys),
            "status_transitions": dict(sorted(transitions.items())),
            "preprocessing_exact_match": not preprocessing_mismatches,
            "preprocessing_mismatch_count": len(preprocessing_mismatches),
            "preprocessing_mismatches": preprocessing_mismatches[:20],
            "metrics": {},
        }
        for metric_index, (metric, direction) in enumerate(METRICS.items()):
            metric_keys = [
                key for key in keys
                if baseline[key].get(metric) is not None
                and candidate[key].get(metric) is not None
            ]
            if not metric_keys:
                continue
            base = np.asarray(
                [baseline[key][metric] for key in metric_keys], dtype=np.float64,
            )
            cand = np.asarray(
                [candidate[key][metric] for key in metric_keys], dtype=np.float64,
            )
            delta = cand - base
            entry: dict[str, Any] = {
                "direction": direction or "neutral",
                "n": len(metric_keys),
                "baseline_mean": float(base.mean()),
                "candidate_mean": float(cand.mean()),
                "mean_delta": float(delta.mean()),
                "mean_delta_ci95": _bootstrap_mean_ci(
                    delta,
                    seed + candidate_index * 1000 + metric_index,
                    bootstrap_samples,
                ),
                "baseline_median": float(np.median(base)),
                "candidate_median": float(np.median(cand)),
                "median_delta": float(np.median(delta)),
            }
            ties = np.isclose(cand, base, rtol=1e-9, atol=1e-12)
            if direction is not None:
                better = cand > base if direction == "higher" else cand < base
                entry.update({
                    "better_count": int(np.logical_and(better, ~ties).sum()),
                    "tie_count": int(ties.sum()),
                    "worse_count": int((~better & ~ties).sum()),
                    "better_rate": float(np.logical_and(better, ~ties).mean()),
                })
            comparison["metrics"][metric] = entry

            for record in csv_rows:
                key = (record["patent_id"], record["sketch_id"])
                if key in metric_keys:
                    record[f"{candidate_name}_minus_{baseline_name}_{metric}"] = (
                        candidate[key][metric] - baseline[key][metric]
                    )
        report["comparisons"][candidate_name] = comparison

    return report, csv_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", type=_parse_run, required=True,
        help="NAME=RUN_DIR_OR_DB; first run is the baseline (repeat 2+ times)",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--seed", type=int, default=850725)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("at least two --run arguments are required")
    names = [name for name, _path in args.run]
    if len(names) != len(set(names)):
        parser.error("run names must be unique")

    report, rows = compare(
        args.run,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    rendered = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    else:
        print(rendered)
    if args.output_csv:
        _write_csv(args.output_csv, rows)

    print(f"baseline: {report['baseline']}")
    print(f"shared rows across all runs: {report['shared_all_runs']}")
    for name, comparison in report["comparisons"].items():
        print(
            f"{name}: paired={comparison['paired_rows']} "
            f"preprocessing_match={comparison['preprocessing_exact_match']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
