"""Verify the completed review against its worklist, source files and deployment."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from PIL import Image, ImageChops

from tools.build_pipeline_review import digest, load_worklist
from tools.deployment import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gallery", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    gallery = args.gallery.resolve()
    document = json.loads((gallery / "review.json").read_text())
    summary, figures = document["summary"], document["figures"]
    worklist = load_worklist(gallery / "samples.csv")
    expected = [(r["patent_id"], r["sketch_id"]) for r in worklist]
    assert len(expected) == len(set(expected)) == 100
    assert [(f["patent"], f["sketch"]) for f in figures] == expected
    assert summary["completed"] == summary["samples"] == 100
    assert digest(gallery / "samples.csv") == summary["worklist_sha256"]
    root = Path(summary["current_root"])
    with sqlite3.connect(root / "results.db") as db:
        assert set(db.execute("SELECT patent_id, sketch_id FROM results")) == set(expected)
    deployment = json.loads((root / "deployment_run.json").read_text())
    assert preflight(Path(deployment["config_path"]))["identity"] == deployment["identity"]
    assert summary["deployment_identity"] == deployment["identity"]
    manifest = json.loads((gallery / "asset_manifest.json").read_text())
    originals = links = previews = 0
    diagnostic_statuses = Counter()
    for figure, row in zip(figures, worklist):
        original = Path(row["input_path"]).resolve()
        assert (gallery / figure["tif"]).resolve() == original
        assert digest(original) == figure["original"]["source_sha256"]
        with Image.open(original) as source, Image.open(gallery / figure["original"]["src"]) as preview:
            assert source.size == preview.size
            assert ImageChops.difference(source.convert("RGB"), preview.convert("RGB")).getbbox() is None
        originals += 1
        views = [figure["original"]]
        for side in ("current", "previous"):
            run = figure[side]
            assert not run.get("preview_error") and not run.get("graph_preview_error")
            for link in run["links"].values():
                assert link and (gallery / link).is_file(), (figure["patent"], link)
                links += 1
            views.extend(run["views"].values())
        diagnostic = figure["current"].get("diagnostic")
        if diagnostic:
            diagnostic_statuses[diagnostic["status"]] += 1
            assert diagnostic["diagnostic_only"] and not diagnostic["training_eligible"]
            if "source_graph_sha256" in diagnostic:
                graph = root / figure["patent"] / "graphs" / (figure["sketch"]+"_graph.json")
                assert digest(graph) == diagnostic["source_graph_sha256"]
            if diagnostic["status"] == "rendered":
                assert digest(Path(diagnostic["svg"])) == diagnostic["svg_sha256"]
        elif not figure["current"]["canonical_export"]:
            raise AssertionError(f"Missing diagnostic disposition: {figure['patent']}")
        for view in views:
            path = gallery / view["src"]
            signature = manifest[path.name]
            assert digest(Path(signature["source"])) == signature["sha256"] == view["source_sha256"]
            with Image.open(path) as image:
                image.verify()
            previews += 1
    report = {"status": "pass", "originals_pixel_identical": originals,
              "preview_files_verified": previews, "download_links_verified": links,
              "diagnostic_statuses": dict(diagnostic_statuses), "summary": summary}
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}))


if __name__ == "__main__":
    main()
