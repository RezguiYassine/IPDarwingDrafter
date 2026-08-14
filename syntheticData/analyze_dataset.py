from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = ("line", "arc", "circle", "polyline", "bezier")
COMPLEXITY_METRICS = (
    "source_component_count",
    "object_primitive_count",
    "structural_junction_count",
    "interaction_count",
    "interaction_type_count",
    "cycle_rank",
    "occupied_tile_fraction",
    "dense_tile_fraction",
    "local_density_p95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a compact PatentVec generation run."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-rejection-scan",
        action="store_true",
        help="Do not inspect sample.json histories inside the tar shards.",
    )
    return parser.parse_args()


def _distribution(values: list[float]) -> dict[str, float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    def quantile(fraction: float) -> float:
        position = fraction * (len(finite) - 1)
        lower = int(position)
        upper = min(lower + 1, len(finite) - 1)
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "min": finite[0],
        "mean": sum(finite) / len(finite),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "max": finite[-1],
    }


def _normalise_failure(text: str) -> str:
    if "Free2CAD polyline supply" in text:
        return "quality:polyline_supply"
    if "complexity gate" in text:
        metric = text.split("complexity gate:", 1)[-1].strip().split()[0]
        return f"quality:complexity:{metric}"
    if "unintended cross-component intersections" in text:
        return "quality:unintended_intersection"
    if "coincident cross-component stroke" in text:
        return "quality:coincident_stroke"
    if "residual" in text:
        return "quality:interaction_residual"
    return "quality:other"


def _rejection_reason(record: dict) -> str:
    failures = record.get("failures", [])
    if failures:
        return _normalise_failure(str(failures[0]))
    error = str(record.get("error", "unknown"))
    placement = re.search(r"placement step ([a-z0-9_]+) exhausted", error)
    if placement:
        return f"placement:{placement.group(1)}"
    if error.startswith("IndexError"):
        return "bug:IndexError"
    if error.startswith("ValueError: no room for"):
        detail = error.split("ValueError: no room for", 1)[1].strip()
        return f"patent_layer:no_room:{detail}"
    return error.split(":", 1)[0] or "unknown"


def _scan_rejections(dataset: Path, rows: list[dict]) -> dict:
    by_archive: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_archive[row["archive"]].append(row)
    rejected_attempts = Counter()
    placement_steps = Counter()
    core_payload_bytes = []
    audit_payload_bytes = []
    member_bytes = 0
    archive_bytes = 0
    audit_filenames = {
        "clean.png",
        "degraded.png",
        "semantic.png",
        "preview.png",
        "visible.svg",
        "amodal.svg",
    }
    for archive_name in sorted(by_archive):
        archive_path = dataset / archive_name
        archive_bytes += archive_path.stat().st_size
        with tarfile.open(archive_path, mode="r") as archive:
            sizes = {member.name: member.size for member in archive.getmembers()}
            member_bytes += sum(sizes.values())
            for row in by_archive[archive_name]:
                member = archive.extractfile(f"{row['member_prefix']}/sample.json")
                if member is None:
                    raise ValueError(f"missing sample.json for {row['sample_id']}")
                drawing = json.load(member)
                processing = drawing.get("processing", {})
                if "rejected_attempt_reason_counts" in processing:
                    rejected_attempts.update(
                        processing["rejected_attempt_reason_counts"]
                    )
                else:
                    rejected_attempts.update(
                        _rejection_reason(item)
                        for item in processing.get("rejected_attempts", [])
                    )
                if "placement_rejection_counts" in processing:
                    placement_steps.update(processing["placement_rejection_counts"])
                else:
                    placement_steps.update(
                        str(item.get("step", "unknown"))
                        for item in processing.get("placement_rejections", [])
                    )
                prefix = f"{row['member_prefix']}/"
                core_size = 0
                audit_size = 0
                for name, size in sizes.items():
                    if not name.startswith(prefix):
                        continue
                    filename = name[len(prefix) :]
                    if filename in audit_filenames:
                        audit_size += size
                    else:
                        core_size += size
                core_payload_bytes.append(core_size)
                audit_payload_bytes.append(audit_size)
    return {
        "rejected_attempt_reasons": dict(rejected_attempts.most_common()),
        "accepted_attempt_placement_rejections": dict(
            placement_steps.most_common()
        ),
        "payload_storage": {
            "core_member_bytes_per_sample": _distribution(core_payload_bytes),
            "audit_member_bytes_per_audit_sample": _distribution(
                [value for value in audit_payload_bytes if value]
            ),
            "tar_overhead_bytes_per_sample": (
                (archive_bytes - member_bytes) / max(len(rows), 1)
            ),
        },
    }


def _complexity(rows: list[dict]) -> dict:
    output = {}
    for difficulty in ("all", "medium", "hard", "very_hard"):
        selected = rows if difficulty == "all" else [
            row for row in rows if row["difficulty"] == difficulty
        ]
        if not selected:
            continue
        output[difficulty] = {
            metric: _distribution(
                [row["quality"]["complexity"][metric] for row in selected]
            )
            for metric in COMPLEXITY_METRICS
        }
        output[difficulty]["foreground_ratio"] = _distribution(
            [row["quality"]["foreground_ratio"] for row in selected]
        )
    return output


def _fixed_signatures(rows: list[dict]) -> dict:
    interactions = sorted(
        {
            name
            for row in rows
            for name in row["quality"]["interaction_counts"]
        }
    )
    presence = {
        name: sum(
            row["quality"]["interaction_counts"].get(name, 0) > 0
            for row in rows
        )
        / len(rows)
        for name in interactions
    }
    signatures = Counter()
    for row in rows:
        complexity = row["quality"]["complexity"]
        signature = (
            row["difficulty"],
            complexity["source_component_count"],
            complexity["interaction_count"],
            complexity["interaction_type_count"],
            complexity["structural_junction_count"],
            complexity["cycle_rank"],
        )
        signatures[signature] += 1
    return {
        "interaction_presence_fraction": presence,
        "unique_coarse_signatures": len(signatures),
        "top_coarse_signatures": [
            {
                "difficulty": signature[0],
                "components": signature[1],
                "interactions": signature[2],
                "interaction_types": signature[3],
                "junctions": signature[4],
                "cycle_rank": signature[5],
                "samples": count,
            }
            for signature, count in signatures.most_common(10)
        ],
    }


def analyze(dataset: Path, scan_rejections: bool = True) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    if len(rows) != manifest["count"]:
        raise ValueError("manifest and manifest.jsonl sample counts disagree")

    class_counts = Counter()
    per_sample_classes: dict[str, list[float]] = defaultdict(list)
    polyline_origins = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    dropped_samples = []
    for row in rows:
        edges = row["training_targets"]["free2cad_edges"]
        for name in CLASS_NAMES:
            class_counts[name] += int(edges[name])
            per_sample_classes[name].append(float(edges[name]))
        polyline_origins.update(
            direct=int(edges["polyline_direct"]),
            aggregated_lines=int(edges["polyline_aggregated_lines"]),
        )
        for source in row["sources"]:
            sources[source["dataset"]].add(str(source["sample_id"]))
        keypoints = row["training_targets"]["puhachov_keypoints"]
        if keypoints["dropped"]:
            dropped_samples.append(
                {"sample_id": row["sample_id"], "dropped": keypoints["dropped"]}
            )

    total_edges = sum(class_counts.values())
    total_exact = sum(
        row["training_targets"]["puhachov_keypoints"]["exact"] for row in rows
    )
    total_dropped = sum(item["dropped"] for item in dropped_samples)
    attempts = [row["generation"]["attempt"] for row in rows]
    placement_rejections = [
        row["generation"]["placement_rejection_count"] for row in rows
    ]
    bytes_per_sample = manifest["archive_bytes"] / len(rows)
    samples_per_second = float(manifest["samples_per_second_this_run"])

    contract_path = dataset / "polyline_stage2_report.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    gates = {
        "all_quality_gates_passed": bool(manifest["all_quality_gates_passed"]),
        "all_targets_topology_projected": all(
            row["training_targets"]["free2cad_edges"]["topology_projected"]
            for row in rows
        ),
        "polyline_minimum_met": all(
            row["training_targets"]["free2cad_edges"]["polyline"]
            >= row["generation"]["free2cad_supply"]["minimum_polyline"]
            for row in rows
        ),
        "puhachov_drop_rate_below_0_1_percent": (
            total_dropped / max(total_exact, 1) < 0.001
        ),
        "polyline_contract_at_least_95_percent": bool(
            contract and contract["target_acceptance"] >= 0.95
        ),
    }

    scan = _scan_rejections(dataset, rows) if scan_rejections else {}
    payload_storage = scan.get("payload_storage")
    projected_bytes_per_sample_at_audit_100 = None
    if payload_storage:
        projected_bytes_per_sample_at_audit_100 = (
            payload_storage["core_member_bytes_per_sample"]["mean"]
            + payload_storage["tar_overhead_bytes_per_sample"]
            + payload_storage["audit_member_bytes_per_audit_sample"]["mean"] / 100.0
        )

    report = {
        "schema_version": "patentvec-scale-analysis-1.0",
        "dataset": str(dataset),
        "sample_count": len(rows),
        "quality_gates": gates,
        "ready_for_10k_experiment": all(gates.values()),
        "throughput_storage": {
            "elapsed_seconds": manifest["elapsed_seconds_this_run"],
            "samples_per_second": samples_per_second,
            "archive_bytes": manifest["archive_bytes"],
            "bytes_per_sample": bytes_per_sample,
            "projected_10k_gib": bytes_per_sample * 10_000 / 2**30,
            "projected_50k_gib": bytes_per_sample * 50_000 / 2**30,
            "projected_10k_hours": (
                10_000 / samples_per_second / 3600 if samples_per_second else None
            ),
            "projected_50k_hours": (
                50_000 / samples_per_second / 3600 if samples_per_second else None
            ),
            "payload_storage": payload_storage,
            "projected_at_audit_every_100": (
                {
                    "bytes_per_sample": projected_bytes_per_sample_at_audit_100,
                    "projected_10k_gib": (
                        projected_bytes_per_sample_at_audit_100 * 10_000 / 2**30
                    ),
                    "projected_50k_gib": (
                        projected_bytes_per_sample_at_audit_100 * 50_000 / 2**30
                    ),
                }
                if projected_bytes_per_sample_at_audit_100 is not None
                else None
            ),
        },
        "free2cad": {
            "total_edges": total_edges,
            "class_counts": dict(class_counts),
            "class_fraction": {
                name: class_counts[name] / max(total_edges, 1)
                for name in CLASS_NAMES
            },
            "per_sample": {
                name: _distribution(values)
                for name, values in per_sample_classes.items()
            },
            "polyline_origins": dict(polyline_origins),
            "stage2_contract": contract,
        },
        "puhachov": {
            "exact_keypoints": total_exact,
            "dropped_keypoints": total_dropped,
            "drop_rate": total_dropped / max(total_exact, 1),
            "samples_with_dropped_keypoints": dropped_samples,
        },
        "generation": {
            "attempt_index": _distribution(attempts),
            "samples_requiring_more_than_10_retries": sum(
                attempt > 10 for attempt in attempts
            ),
            "samples_requiring_more_than_20_retries": sum(
                attempt > 20 for attempt in attempts
            ),
            "placement_rejections": _distribution(placement_rejections),
            "highest_retry_samples": [
                {
                    "sample_id": row["sample_id"],
                    "difficulty": row["difficulty"],
                    "attempt": row["generation"]["attempt"],
                }
                for row in sorted(
                    rows, key=lambda item: item["generation"]["attempt"], reverse=True
                )[:20]
            ],
        },
        "complexity": _complexity(rows),
        "signature_risk": _fixed_signatures(rows),
        "source_diversity": {
            dataset_name: len(sample_ids)
            for dataset_name, sample_ids in sorted(sources.items())
        },
    }
    if scan_rejections:
        report["generation"].update(
            {key: value for key, value in scan.items() if key != "payload_storage"}
        )
    return report


def main() -> int:
    args = parse_args()
    report = analyze(args.dataset, scan_rejections=not args.skip_rejection_scan)
    output = args.output or args.dataset / "analysis.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    summary = {
        "sample_count": report["sample_count"],
        "ready_for_10k_experiment": report["ready_for_10k_experiment"],
        "throughput_storage": report["throughput_storage"],
        "free2cad_class_counts": report["free2cad"]["class_counts"],
        "puhachov_drop_rate": report["puhachov"]["drop_rate"],
        "attempt_index": report["generation"]["attempt_index"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"analysis: {output}")
    return 0 if report["ready_for_10k_experiment"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
