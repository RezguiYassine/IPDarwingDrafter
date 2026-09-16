"""Verify acquisition artifacts and summarize paired effects by source drawing."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from syntheticData.measure_patent_like import digest


def estimate(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(150926).integers(len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return {"count": len(values), "mean_delta": float(values.mean()),
            "median_delta": float(np.median(values)),
            "mean_ci95": np.quantile(means, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    args = parser.parse_args()
    report = json.loads((args.probe / "report.json").read_text())
    groups = report["groups"]
    assert len(groups) == report["completed_samples"] * 3
    assert report["stage1_passes"] == len(groups) * 5
    cases = defaultdict(list)
    routes = Counter()
    targets = Counter()
    verified = 0
    for group in groups:
        cases[group["render_case"]].append(group)
        targets.update(group["target_audit"])
        for variant, row in group["variants"].items():
            path = args.probe / "samples" / group["sample_id"] / group["render_case"] / variant
            assert digest(path / "reference_free.png") == row["input_sha256"]
            assert digest(path / "cleaned/probe_skeleton.png") == row["skeleton_sha256"]
            routes[row["route"]] += 1
            verified += 1
    assert not any(targets[key] for key in ("missing_topology_labels", "extra_topology_labels",
                                           "duplicate_topology_labels", "unsupported_skeleton_pixels"))
    keys = ("c2_f1_at2px", "c2_precision_at2px", "c2_recall_at2px", "tiny_component_fraction",
            "cn_junction_clusters_per_1k", "tile_count_512_stride256")
    medians, effects = [], []
    comparisons = {"encoding_default": ("binary_default", "gray_default"),
                   "encoding_low_speckle": ("binary_low_speckle", "gray_low_speckle"),
                   "lower_noise_gray": ("gray_low_speckle", "gray_default"),
                   "lower_noise_binary": ("binary_low_speckle", "binary_default")}
    for case, rows in sorted(cases.items()):
        for variant in rows[0]["variants"]:
            medians.append({"render_case": case, "variant": variant, "count": len(rows),
                            **{key: float(np.median([r["variants"][variant]["metrics"][key] for r in rows])) for key in keys}})
        for comparison, (candidate, baseline) in comparisons.items():
            delta = [r["variants"][candidate]["metrics"]["c2_f1_at2px"] -
                     r["variants"][baseline]["metrics"]["c2_f1_at2px"] for r in rows]
            effects.append({"render_case": case, "comparison": comparison, **estimate(delta)})
    summary = {key: value for key, value in report.items() if key != "groups"}
    summary.update(report_sha256=digest(args.probe / "report.json"), verified_input_output_pairs=verified,
                   routes=dict(routes), clean_target_audit_totals=dict(targets), medians=medians,
                   paired_f1_effects=effects,
                   uncertainty="10000 deterministic paired-drawing bootstrap samples; exploratory, not a real-patent release test")
    output = Path(__file__).with_name("phase05_summary.json")
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "verified_pairs": verified, "routes": dict(routes),
                      "medians": medians, "paired_f1_effects": effects}, indent=2))


if __name__ == "__main__":
    main()
