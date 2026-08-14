from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from syntheticData.analyze_dataset import CLASS_NAMES, COMPLEXITY_METRICS, analyze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired PatentVec baseline/enhanced compact datasets."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("enhanced", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def _rows(root: Path) -> dict[int, dict]:
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    keyed = {int(row["source_index"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate source indices in {root}")
    return keyed


def _analysis(root: Path) -> dict:
    path = root / "analysis.json"
    return json.loads(path.read_text()) if path.exists() else analyze(
        root, scan_rejections=False
    )


def _paired_stats(before, after) -> dict[str, float]:
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    delta = after - before
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else 0.0
    return {
        "baseline_mean": float(before.mean()),
        "enhanced_mean": float(after.mean()),
        "mean_delta": mean,
        "mean_delta_ci95_low": mean - 1.96 * standard_error,
        "mean_delta_ci95_high": mean + 1.96 * standard_error,
        "relative_mean_change": mean / max(abs(float(before.mean())), 1e-12),
        "median_delta": float(np.median(delta)),
        "enhanced_greater_fraction": float(np.mean(delta > 0)),
        "equal_fraction": float(np.mean(delta == 0)),
    }


def _unpaired_stats(before, after) -> dict[str, float]:
    """Estimate a curriculum-level difference without assuming paired draws."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if not len(before) or not len(after):
        raise ValueError("unpaired statistics require two non-empty samples")
    baseline_mean = float(before.mean())
    enhanced_mean = float(after.mean())
    mean = enhanced_mean - baseline_mean
    before_variance = float(before.var(ddof=1)) if len(before) > 1 else 0.0
    after_variance = float(after.var(ddof=1)) if len(after) > 1 else 0.0
    standard_error = math.sqrt(
        before_variance / len(before) + after_variance / len(after)
    )
    return {
        "baseline_mean": baseline_mean,
        "enhanced_mean": enhanced_mean,
        "mean_delta": mean,
        "mean_delta_ci95_low": mean - 1.96 * standard_error,
        "mean_delta_ci95_high": mean + 1.96 * standard_error,
        "relative_mean_change": mean / max(abs(baseline_mean), 1e-12),
        "median_delta": float(np.median(after) - np.median(before)),
    }


def _metric_stats(rows, getter, *, paired: bool) -> dict[str, float]:
    before = [getter(row[0]) for row in rows]
    after = [getter(row[1]) for row in rows]
    return _paired_stats(before, after) if paired else _unpaired_stats(before, after)


def _source_pairs(rows: list[tuple[dict, dict]]) -> dict:
    by_dataset = defaultdict(lambda: [set(), set()])
    for baseline, enhanced in rows:
        for variant, row in enumerate((baseline, enhanced)):
            for source in row["sources"]:
                by_dataset[source["dataset"]][variant].add(str(source["sample_id"]))
    report = {}
    for dataset, (baseline_ids, enhanced_ids) in sorted(by_dataset.items()):
        union = baseline_ids | enhanced_ids
        report[dataset] = {
            "baseline_unique": len(baseline_ids),
            "enhanced_unique": len(enhanced_ids),
            "intersection": len(baseline_ids & enhanced_ids),
            "jaccard": len(baseline_ids & enhanced_ids) / max(len(union), 1),
        }
    return report


def compare(baseline_root: Path, enhanced_root: Path) -> dict:
    baseline_rows = _rows(baseline_root)
    enhanced_rows = _rows(enhanced_root)
    if set(baseline_rows) != set(enhanced_rows):
        raise ValueError("A/B source_index sets differ; comparison is not paired")
    source_indices = sorted(baseline_rows)
    pairs = [(baseline_rows[index], enhanced_rows[index]) for index in source_indices]
    seed_mismatches = [
        index
        for index, (baseline, enhanced) in zip(source_indices, pairs)
        if int(baseline["seed"]) != int(enhanced["seed"])
    ]
    exact_seed_pairs = [
        pair
        for pair in pairs
        if int(pair[0]["seed"]) == int(pair[1]["seed"])
    ]
    if not exact_seed_pairs:
        raise ValueError("A/B datasets have no exact-seed pairs")

    baseline_analysis = _analysis(baseline_root)
    enhanced_analysis = _analysis(enhanced_root)
    complexity = {
        metric: _metric_stats(
            pairs,
            lambda row, name=metric: row["quality"]["complexity"][name],
            paired=False,
        )
        for metric in COMPLEXITY_METRICS
    }
    complexity_exact_seed = {
        metric: _metric_stats(
            exact_seed_pairs,
            lambda row, name=metric: row["quality"]["complexity"][name],
            paired=True,
        )
        for metric in COMPLEXITY_METRICS
    }
    class_supply = {
        name: _metric_stats(
            pairs,
            lambda row, class_name=name: row["training_targets"][
                "free2cad_edges"
            ][class_name],
            paired=False,
        )
        for name in CLASS_NAMES
    }
    class_supply_exact_seed = {
        name: _metric_stats(
            exact_seed_pairs,
            lambda row, class_name=name: row["training_targets"][
                "free2cad_edges"
            ][class_name],
            paired=True,
        )
        for name in CLASS_NAMES
    }
    keypoint_supply = {
        kind: _metric_stats(
            pairs,
            lambda row, keypoint_kind=kind: row["training_targets"][
                "puhachov_keypoints"
            ][keypoint_kind],
            paired=False,
        )
        for kind in ("endpoint", "junction", "corner", "exact", "dropped")
    }
    generation = {
        "attempt_index": _metric_stats(
            pairs,
            lambda row: row["generation"]["attempt"],
            paired=False,
        ),
        "placement_rejections": _metric_stats(
            pairs,
            lambda row: row["generation"]["placement_rejection_count"],
            paired=False,
        ),
    }
    transitions = Counter(
        f"{baseline['difficulty']}->{enhanced['difficulty']}"
        for baseline, enhanced in pairs
    )
    source_overlap = _source_pairs(pairs)

    enhanced_contract = enhanced_analysis["free2cad"].get("stage2_contract") or {}
    minimum_exact_seed_pairs = 5_000
    gates = {
        "same_source_index_set": True,
        "at_least_5000_exact_seed_pairs": (
            len(exact_seed_pairs) >= minimum_exact_seed_pairs
        ),
        "enhanced_all_quality_gates_passed": bool(
            enhanced_analysis["quality_gates"]["all_quality_gates_passed"]
        ),
        "enhanced_all_targets_topology_projected": bool(
            enhanced_analysis["quality_gates"]["all_targets_topology_projected"]
        ),
        "enhanced_polyline_contract_at_least_95_percent": (
            float(enhanced_contract.get("target_acceptance", 0.0)) >= 0.95
        ),
        "enhanced_puhachov_drop_rate_below_0_1_percent": (
            enhanced_analysis["puhachov"]["drop_rate"] < 0.001
        ),
        "enhanced_polyline_supply_increased": (
            class_supply["polyline"]["mean_delta_ci95_low"] > 0.0
            and class_supply_exact_seed["polyline"]["mean_delta_ci95_low"] > 0.0
        ),
        "enhanced_components_increased": (
            complexity["source_component_count"]["mean_delta_ci95_low"] > 0.0
            and complexity_exact_seed["source_component_count"][
                "mean_delta_ci95_low"
            ] > 0.0
        ),
        "enhanced_junctions_increased": (
            complexity["structural_junction_count"]["mean_delta_ci95_low"] > 0.0
            and complexity_exact_seed["structural_junction_count"][
                "mean_delta_ci95_low"
            ] > 0.0
        ),
        "enhanced_source_diversity_not_collapsed": all(
            item["enhanced_unique"] >= 0.8 * item["baseline_unique"]
            for item in source_overlap.values()
            if item["baseline_unique"]
        ),
    }
    return {
        "schema_version": "patentvec-complexity-ab-comparison-1.0",
        "baseline": str(baseline_root),
        "enhanced": str(enhanced_root),
        "paired_sample_count": len(pairs),
        "comparison_design": {
            "full_sample_estimator": "unpaired_welch_normal_ci",
            "full_sample_count_per_variant": len(pairs),
            "exact_seed_estimator": "paired_normal_ci",
            "exact_seed_pair_count": len(exact_seed_pairs),
            "exact_seed_pair_fraction": len(exact_seed_pairs) / len(pairs),
            "retry_seed_mismatch_count": len(seed_mismatches),
            "interpretation": (
                "Curriculum-specific rejection changes accepted retry seeds. "
                "Release gates require positive full-sample and exact-seed-subset "
                "confidence intervals."
            ),
        },
        "difficulty_transitions": dict(transitions),
        "gates": gates,
        "ready_for_equal_budget_training": all(gates.values()),
        "complexity": complexity,
        "complexity_exact_seed_paired": complexity_exact_seed,
        "free2cad_class_supply": class_supply,
        "free2cad_class_supply_exact_seed_paired": class_supply_exact_seed,
        "puhachov_keypoint_supply": keypoint_supply,
        "generation_cost": generation,
        "signature_diversity": {
            "baseline_unique_coarse_signatures": baseline_analysis[
                "signature_risk"
            ]["unique_coarse_signatures"],
            "enhanced_unique_coarse_signatures": enhanced_analysis[
                "signature_risk"
            ]["unique_coarse_signatures"],
        },
        "source_overlap": source_overlap,
        "baseline_quality_gates": baseline_analysis["quality_gates"],
        "enhanced_quality_gates": enhanced_analysis["quality_gates"],
        "baseline_throughput_storage": baseline_analysis["throughput_storage"],
        "enhanced_throughput_storage": enhanced_analysis["throughput_storage"],
    }


def _markdown(report: dict) -> str:
    lines = [
        "# PatentVec Complexity A/B Comparison",
        "",
        f"Matched source indices: {report['paired_sample_count']:,}",
        f"Exact-seed pairs: {report['comparison_design']['exact_seed_pair_count']:,} "
        f"({report['comparison_design']['exact_seed_pair_fraction']:.1%})",
        "",
        report["comparison_design"]["interpretation"],
        "",
        "## Decision Gates",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'yes' if passed else 'no'} |"
        for name, passed in report["gates"].items()
    )
    lines.extend(
        [
            "",
            f"Ready for equal-budget training: **{'yes' if report['ready_for_equal_budget_training'] else 'no'}**",
            "",
            "## Complexity",
            "",
            "| metric | baseline mean | enhanced mean | relative change | full-sample 95% CI | exact-seed paired 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["complexity"].items():
        exact = report["complexity_exact_seed_paired"][name]
        lines.append(
            f"| {name} | {item['baseline_mean']:.4f} | {item['enhanced_mean']:.4f} "
            f"| {item['relative_mean_change']:+.1%} | "
            f"[{item['mean_delta_ci95_low']:.4f}, {item['mean_delta_ci95_high']:.4f}] | "
            f"[{exact['mean_delta_ci95_low']:.4f}, {exact['mean_delta_ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Stage 3 Supply",
            "",
            "| class | baseline/sample | enhanced/sample | relative change | full-sample 95% CI | exact-seed paired 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["free2cad_class_supply"].items():
        exact = report["free2cad_class_supply_exact_seed_paired"][name]
        lines.append(
            f"| {name} | {item['baseline_mean']:.4f} | {item['enhanced_mean']:.4f} "
            f"| {item['relative_mean_change']:+.1%} | "
            f"[{item['mean_delta_ci95_low']:.4f}, {item['mean_delta_ci95_high']:.4f}] | "
            f"[{exact['mean_delta_ci95_low']:.4f}, {exact['mean_delta_ci95_high']:.4f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = compare(args.baseline, args.enhanced)
    output = args.output or args.enhanced.parent / "PatentVecComplexityAB.json"
    markdown = args.markdown or output.with_suffix(".md")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown.write_text(_markdown(report))
    print(
        json.dumps(
            {
                "paired_sample_count": report["paired_sample_count"],
                "gates": report["gates"],
                "ready_for_equal_budget_training": report[
                    "ready_for_equal_budget_training"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"comparison: {output}")
    print(f"report: {markdown}")
    return 0 if report["ready_for_equal_budget_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
