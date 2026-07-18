"""Paired comparison of two Drawing2CAD end-to-end evaluation databases."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np


METRICS = {
    "chamfer_sym": "lower",
    "chamfer_p95_sym": "lower",
    "iou_pixel": "higher",
    "iou_skeleton": "higher",
    "precision_pixel": "higher",
    "recall_pixel": "higher",
    "n_prims_out": "lower",
    "n_edges": "lower",
    "median_edge_length": "higher",
    "micro_edge_ratio": "lower",
    "short_edge_ratio": "lower",
    "s2_time": "lower",
    "total_time": "lower",
}


def _columns(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        return {
            row[1] for row in connection.execute("PRAGMA table_info(d2c_results)")
        }


def _load(db_path: Path, columns: list[str]) -> dict[tuple[str, str], dict]:
    selected = ",".join(["sample_id", "view", "n_strokes_gt", *columns])
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT {selected} FROM d2c_results WHERE status='ok'"
        ).fetchall()
    result = {}
    for row in rows:
        values = dict(zip(["sample_id", "view", "n_strokes_gt", *columns], row))
        result[(values.pop("sample_id"), values.pop("view"))] = values
    return result


def compare(baseline_db: Path, candidate_db: Path) -> dict:
    shared_columns = _columns(baseline_db) & _columns(candidate_db)
    metrics = [metric for metric in METRICS if metric in shared_columns]
    baseline = _load(baseline_db, metrics)
    candidate = _load(candidate_db, metrics)
    keys = sorted(baseline.keys() & candidate.keys())
    report = {
        "baseline_db": str(baseline_db),
        "candidate_db": str(candidate_db),
        "baseline_ok": len(baseline),
        "candidate_ok": len(candidate),
        "paired": len(keys),
        "metrics": {},
    }
    for metric in metrics:
        pairs = [
            (baseline[key][metric], candidate[key][metric]) for key in keys
            if baseline[key][metric] is not None and candidate[key][metric] is not None
        ]
        if not pairs:
            continue
        base = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        cand = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        delta = cand - base
        direction = METRICS[metric]
        wins = cand > base if direction == "higher" else cand < base
        ties = np.isclose(cand, base, rtol=1e-9, atol=1e-12)
        base_mean = float(base.mean())
        candidate_mean = float(cand.mean())
        report["metrics"][metric] = {
            "direction": direction,
            "n": len(pairs),
            "baseline_mean": base_mean,
            "candidate_mean": candidate_mean,
            "absolute_delta": candidate_mean - base_mean,
            "relative_delta_pct": (
                100.0 * (candidate_mean - base_mean) / abs(base_mean)
                if abs(base_mean) > 1e-12 else None
            ),
            "baseline_median": float(np.median(base)),
            "candidate_median": float(np.median(cand)),
            "paired_win_rate": float(np.logical_and(wins, ~ties).mean()),
            "tie_rate": float(ties.mean()),
        }

    for name, numerator in (("primitive_inflation", "n_prims_out"),
                            ("edge_inflation", "n_edges")):
        if numerator not in metrics:
            continue
        pairs = []
        for key in keys:
            gt = baseline[key]["n_strokes_gt"]
            b_value = baseline[key][numerator]
            c_value = candidate[key][numerator]
            if gt and b_value is not None and c_value is not None:
                pairs.append((b_value / gt, c_value / gt))
        if pairs:
            base = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
            cand = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
            report["metrics"][name] = {
                "direction": "lower",
                "n": len(pairs),
                "baseline_mean": float(base.mean()),
                "candidate_mean": float(cand.mean()),
                "absolute_delta": float((cand - base).mean()),
                "relative_delta_pct": float(
                    100.0 * (cand.mean() - base.mean()) / abs(base.mean())
                ) if abs(base.mean()) > 1e-12 else None,
                "baseline_median": float(np.median(base)),
                "candidate_median": float(np.median(cand)),
                "paired_win_rate": float((cand < base).mean()),
                "tie_rate": float(np.isclose(cand, base).mean()),
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
