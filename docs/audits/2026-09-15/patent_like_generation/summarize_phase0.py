"""Keep compact Phase 0 evidence in git while retaining the corpus externally."""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from syntheticData.generate_dataset import _fingerprint, _sample_specs, _valid_marker


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--patent-run", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    measured = json.loads((root / "measurements/measurements.json").read_text())
    polyline = json.loads((root / "polyline_contract.json").read_text())
    settings = manifest["settings"]
    specs = _sample_specs(SimpleNamespace(**settings))
    assert manifest["run_fingerprint"] == _fingerprint(settings)
    assert measured["dataset_manifest_sha256"] == digest(root / "manifest.json")
    verified = []
    for index, offset in enumerate(range(0, len(specs), settings["shard_size"])):
        fingerprint = _fingerprint({
            "run_fingerprint": manifest["run_fingerprint"], "shard_index": index,
            "samples": specs[offset:offset + settings["shard_size"]],
            **{key: settings[key] for key in ("max_sample_attempts", "retain_rasters", "degradation", "stage2_label_contract")},
        })
        name = f"shard_{index:06d}.tar"
        marker = _valid_marker(root, name, fingerprint)
        assert marker is not None, f"Invalid resume marker: {name}"
        verified.append(name)
    database = args.patent_run / "results.db"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        patent_routes = dict(connection.execute("SELECT s1_model_used, count(*) FROM results GROUP BY s1_model_used"))
    artifacts = ("generation.json", "manifest.json", "manifest.jsonl",
                 "measurements/measurements.json", "polyline_contract.json", "audit/index.html")
    report = {
        "dataset": str(root.resolve()), "manifest": {key: value for key, value in manifest.items() if key != "shards"},
        "artifact_sha256": {name: digest(root / name) for name in artifacts},
        "verified_resume_shards": len(verified),
        "implementation_hashes_match": all(digest(Path(name)) == sha for name, sha in settings["implementation"].items()),
        "stage1": measured["stage1"], "synthetic_stage1_routes": measured["stage1_routes"],
        "patent_stage1_routes": patent_routes, "patent_results_db_sha256": digest(database),
        "contract_pass": measured["contract_pass"], "contract_totals": measured["contract_totals"],
        "metric_summaries": measured["summaries"], "metric_definitions": measured["definitions"],
        "polyline_contract": {key: value for key, value in polyline.items() if key != "reports"},
        "phase0_measurement_complete": True, "training_release_accepted": False,
    }
    output = Path(__file__).with_name("phase0_summary.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "verified_resume_shards": len(verified),
                      "implementation_hashes_match": report["implementation_hashes_match"],
                      "patent_routes": patent_routes}))


if __name__ == "__main__":
    main()
