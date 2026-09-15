"""Read-only project audit; writes only the requested JSON evidence snapshot."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def database(path):
    return sqlite3.connect((ROOT / path).resolve().as_uri() + "?mode=ro", uri=True)


def config_differences(left, right, prefix=""):
    result = {}
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            result.update(config_differences(a, b, prefix + key + "."))
        elif a != b:
            result[prefix + key] = {"deploy": a, "replay": b}
    return result


def collect():
    run = Path("output/PatentData100_Stage3GuardedP2FixedFrozen")
    cfg = yaml.safe_load((ROOT / "config_deploy.yaml").read_text())
    replay = yaml.safe_load((ROOT / run / "stage34_replay_config.yaml").read_text())
    result = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "Existing artifacts only; not a fresh model or pipeline evaluation",
        "deploy_replay_differences": config_differences(cfg, replay),
        "effective_implicit_defaults": {"hachure_mode": "region", "dashed_grouping": False},
    }
    result["weights"] = {}
    for name in (cfg["puhachov"]["weights"], cfg["stage2"]["hachure_cnn_model"],
                 cfg["sketchcleannet"]["weights"], cfg["free2cad"]["weights"]):
        path = ROOT / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        result["weights"][name] = {"bytes": path.stat().st_size, "sha256": digest}

    result["synthetic"] = {}
    for variant in ("A", "B"):
        path = Path(f"output/PatentVecComplexity{variant}10k")
        manifest = read_json(path / "manifest.json")
        result["synthetic"][variant] = {
            key: value for key, value in manifest.items() if key != "shards"
        }
        result["synthetic"][variant]["archive_files"] = len(list((ROOT / path / "shards").glob("*.tar")))
        with (ROOT / path / "manifest.jsonl").open() as stream:
            rows = [json.loads(line) for line in stream]
        result["synthetic"][variant]["actual_manifest_rows"] = len(rows)
        result["synthetic"][variant]["actual_difficulty_counts"] = dict(Counter(row["difficulty"] for row in rows))
    external = Path("/media/safe/secondary disk/IPdrawings")
    result["external_data_root"] = {"path": str(external), "exists": external.exists(),
                                    "entries": [p.name for p in external.iterdir()] if external.exists() else None}

    result["training_coverage"] = {
        name: read_json(f"output/{directory}/stage2_train.coverage.i8.json")
        for name, directory in (("phaseA", "SketchGraphsFull"), ("phaseB", "SketchGraphsCADVG40"))
    }
    result["stage2_evaluations"] = {}
    for variant in ("A", "B", "C2", "C3"):
        for domain in ("d2c_test", "sketchgraphs_test", "archcad_val"):
            if variant in ("A", "B"):
                path = f"guard4way/evaluation_{domain}_model{variant}.json"
            else:
                folder = "reference_free_topology" if variant == "C2" else "hatch_suppressed_topology"
                path = f"{folder}/evaluation_{domain}_{variant}.json"
            doc = read_json("output/PatentVecComplexityABTraining/stage2/" + path)
            result["stage2_evaluations"][f"{variant}:{domain}"] = {
                k: v for k, v in doc.items() if not isinstance(v, (dict, list))
            }
    for name, path in {
        "patent_replay": str(run / "stage34_replay_report.json"),
        "d2c_replay": "output/Drawing2CAD/stage3_guarded_p2_fixed_full/stage34_replay_report.json",
        "hatch_region_best": "models/hatch_grid_v2/best_config.json",
        "hatch_stroke_real": "output/PatentVecHatchStrokeReal/manifest.json",
    }.items():
        result[name] = read_json(path)

    with database(run / "results.db") as conn:
        result["patent_status"] = dict(conn.execute("SELECT status, COUNT(*) FROM results GROUP BY status"))
        result["stage1_methods"] = dict(conn.execute("SELECT s1_model_used, COUNT(*) FROM results GROUP BY s1_model_used"))
        result["stage2_methods"] = [list(row) for row in conn.execute("SELECT s2_keypoint_src, COUNT(*) FROM results GROUP BY s2_keypoint_src")]
        patents = {row[0] for row in conn.execute("SELECT DISTINCT patent_id FROM results")}
        times = [row[0] for row in conn.execute("SELECT s3_time FROM results WHERE s3_time IS NOT NULL")]
        result["stage3_runtime_seconds"] = {"n": len(times), "mean": float(np.mean(times)),
            "median": float(np.median(times)), "p95": float(np.percentile(times, 95)), "max": max(times)}
    counts = Counter()
    for path in (ROOT / run).glob("*/references/*_references.json"):
        counts["files"] += 1
        for ref in json.loads(path.read_text())["reference_labels"]:
            counts["labels"] += 1
            counts["nonempty_text"] += bool(ref.get("text"))
            counts[ref.get("kind", "unknown")] += 1
    result["serialized_references"] = dict(counts)

    # Check representation coverage, not ground-truth hatch classification.
    coverage = []
    for path in sorted((ROOT / run).glob("*/graphs/*.json")):
        graph = json.loads(path.read_text())
        regions = graph.get("hachure_regions", [])
        if not regions:
            continue
        mask = np.zeros(graph["image_shape"], np.uint8)
        for region in regions:
            cv2.fillPoly(mask, [np.asarray(region["boundary"], np.int32)], 1)
        pixels = set()
        partial = whole = 0
        for edge in graph.get("removed_hachures", []):
            coords = np.asarray(edge.get("pixels", []), dtype=np.int32)
            if not coords.size:
                continue
            outside = mask[coords[:, 1], coords[:, 0]] == 0
            partial += bool(outside.any())
            whole += bool(outside.all())
            pixels.update(map(tuple, coords[outside]))
        coverage.append({"graph": str(path.relative_to(ROOT)), "regions": len(regions),
            "side_layer_edges": len(graph.get("removed_hachures", [])),
            "region_member_count": sum(r["n_lines"] for r in regions),
            "edges_partly_or_wholly_outside_regions": partial,
            "edges_wholly_outside_regions": whole,
            "unique_side_layer_pixels_outside_regions": len(pixels)})
    result["hatch_representation_coverage"] = coverage
    result["hatch_coverage_caveat"] = "Not an absolute count of lost final-image pixels; some may overlap other geometry. Region fills are approximations."
    result["exports"] = dict(Counter(p.suffix for p in (ROOT / run).glob("*/vectors/*")))

    splits = read_json("models/hatch_grid_v2/splits.json")
    groups = {key: {r["patent"] for r in rows} for key, rows in splits.items()}
    result["hatch_region_splits"] = {key: {"figures": len(rows), "positives": sum(r["is_positive"] for r in rows)} for key, rows in splits.items()}
    result["hatch_region_patent_overlap"] = {f"{a}:{b}": sorted(groups[a] & groups[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    result["current100_overlap_with_region_label_patents"] = sorted(patents & set.union(*groups.values()))
    result["content_manifests"] = {}
    for name in ("filter_manifest_clean12.csv", "filter_manifest_v3.csv"):
        with (ROOT / "output/PatentData" / name).open() as stream:
            result["content_manifests"][name] = dict(Counter(r["label"] for r in csv.DictReader(stream)))
    with (ROOT / "benchmarks/pilotv3/pilotv3_labeled_audit.csv").open() as stream:
        result["pilotv3_content_buckets"] = dict(Counter(r["bucket"] for r in csv.DictReader(stream)))
    with database("output/PatentData_clean12_gated/results.db") as conn:
        result["historical_clean12_status"] = dict(conn.execute("SELECT status, COUNT(*) FROM results GROUP BY status"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = collect()
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"Evidence written to {args.output}")
