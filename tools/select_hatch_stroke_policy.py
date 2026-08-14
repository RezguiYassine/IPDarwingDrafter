"""Select one hatch-removal threshold policy across evaluation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _policy_key(policy: dict) -> tuple[float, float]:
    return (
        round(float(policy["hatch_high_threshold"]), 12),
        round(float(policy["structural_low_threshold"]), 12),
    )


def select_joint_policy(evaluations: list[dict]) -> dict:
    if len(evaluations) < 2:
        raise ValueError("joint policy selection requires at least two evaluations")

    checkpoint_hashes = {
        str(evaluation.get("checkpoint_sha256", "")) for evaluation in evaluations
    }
    if "" in checkpoint_hashes or len(checkpoint_hashes) != 1:
        raise ValueError("evaluations must describe the same checkpoint SHA-256")

    labels: list[str] = []
    policy_maps: list[dict[tuple[float, float], dict]] = []
    for index, evaluation in enumerate(evaluations):
        label = f"{evaluation.get('dataset', 'dataset')}:{evaluation.get('split', 'split')}"
        if label in labels:
            label = f"{label}#{index + 1}"
        labels.append(label)
        policies = evaluation.get("policy_sweep")
        if not isinstance(policies, list) or not policies:
            raise ValueError(f"evaluation {label!r} has no policy sweep")
        policy_map = {_policy_key(policy): policy for policy in policies}
        if len(policy_map) != len(policies):
            raise ValueError(f"evaluation {label!r} has duplicate threshold policies")
        policy_maps.append(policy_map)

    common_keys = set(policy_maps[0])
    for policy_map in policy_maps[1:]:
        common_keys.intersection_update(policy_map)
    if not common_keys:
        raise ValueError("evaluations have no common threshold policies")

    joint_policies = []
    for hatch_threshold, structural_threshold in sorted(common_keys):
        entries = [
            policy_map[(hatch_threshold, structural_threshold)]
            for policy_map in policy_maps
        ]
        f1_values = [float(entry["safe_removal_f1"]) for entry in entries]
        recall_values = [float(entry["safe_removal_recall"]) for entry in entries]
        precision_values = [
            float(entry["safe_removal_precision"]) for entry in entries
        ]
        error_values = [
            float(entry["safe_removal_structural_error_rate"])
            for entry in entries
        ]
        joint_policies.append({
            "hatch_high_threshold": hatch_threshold,
            "structural_low_threshold": structural_threshold,
            "eligible": all(bool(entry.get("eligible")) for entry in entries),
            "worst_case_safe_removal_f1": min(f1_values),
            "worst_case_safe_removal_recall": min(recall_values),
            "worst_case_safe_removal_precision": min(precision_values),
            "maximum_structural_error_rate": max(error_values),
            "mean_safe_removal_f1": sum(f1_values) / len(f1_values),
            "evaluations": {
                label: {
                    "eligible": bool(entry.get("eligible")),
                    "safe_removal_f1": float(entry["safe_removal_f1"]),
                    "safe_removal_precision": float(
                        entry["safe_removal_precision"]
                    ),
                    "safe_removal_recall": float(entry["safe_removal_recall"]),
                    "safe_removal_structural_error_rate": float(
                        entry["safe_removal_structural_error_rate"]
                    ),
                }
                for label, entry in zip(labels, entries)
            },
        })

    joint_policies.sort(
        key=lambda policy: (
            policy["eligible"],
            policy["worst_case_safe_removal_f1"],
            policy["worst_case_safe_removal_recall"],
            policy["worst_case_safe_removal_precision"],
            -policy["maximum_structural_error_rate"],
            -policy["hatch_high_threshold"],
            -policy["structural_low_threshold"],
        ),
        reverse=True,
    )
    selected = next(
        (policy for policy in joint_policies if policy["eligible"]), None
    )
    return {
        "schema_version": "hatch-stroke-joint-policy-1.0",
        "checkpoint_sha256": checkpoint_hashes.pop(),
        "evaluation_labels": labels,
        "selected_policy": selected,
        "joint_policy_sweep": joint_policies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select one eligible hatch-removal policy across datasets."
    )
    parser.add_argument(
        "--evaluation", type=Path, action="append", required=True,
        help="evaluation JSON; repeat for each required deployment dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluations = [json.loads(path.read_text()) for path in args.evaluation]
    result = select_joint_policy(evaluations)
    result["evaluation_files"] = [str(path) for path in args.evaluation]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "selected_policy": result["selected_policy"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
