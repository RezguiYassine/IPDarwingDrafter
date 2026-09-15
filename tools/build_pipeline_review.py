"""Build an offline original/current/previous patent review without hiding gated rows."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import xml.etree.ElementTree as ET

import cairosvg
from PIL import Image

from tools.make_patent_comparison_sheet import _load, _load_graph, _load_raster


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_worklist(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = []
    for row in rows:
        key = (row["patent_id"], row["sketch_id"])
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in key):
            raise ValueError(f"Unsafe drawing identity: {key}")
        if not Path(row["input_path"]).is_file():
            raise ValueError(f"Missing original TIF: {row['input_path']}")
        keys.append(key)
    if not rows or len(set(keys)) != len(keys):
        raise ValueError("Worklist must be nonempty with unique identities")
    return rows


def historical_rows(root):
    if (root / "results.db").exists():
        return _load(root)
    path = root / "summary.json"
    rows = json.loads(path.read_text()).get("rows", []) if path.exists() else []
    return {(r["folder"], r["sketch_id"]): r for r in rows}


def inline_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")


class Assets:
    def __init__(self, output):
        self.output = output
        self.folder = output / "assets"
        self.folder.mkdir(parents=True, exist_ok=True)
        path = output / "asset_manifest.json"
        self.manifest = json.loads(path.read_text()) if path.exists() else {}

    def preview(self, source, name, kind):
        path = self.folder / (name+".png")
        signature = {"source": str(source.resolve()), "sha256": digest(source), "kind": kind}
        if self.manifest.get(path.name) != signature or not path.exists():
            if kind == "svg":
                cairosvg.svg2png(url=str(source), write_to=str(path), background_color="white")
            else:
                image = _load_graph(source) if kind == "graph" else _load_raster(source)
                image.save(path)
                image.close()
            self.manifest[path.name] = signature
        with Image.open(path) as image:
            size = list(image.size)
        return {"src": "assets/"+path.name, "size": size, "source_sha256": signature["sha256"]}

    def link(self, source):
        return Path(os.path.relpath(source.resolve(), self.output.resolve())).as_posix() if source.is_file() else None

    def finish(self):
        (self.output / "asset_manifest.json").write_text(json.dumps(self.manifest, indent=2)+"\n")


def describe_run(root, patent, sketch, row, assets, prefix, *, current):
    folder = root / patent
    svg = folder / "vectors" / (sketch+".svg")
    graph = folder / "graphs" / (sketch+"_graph.json")
    primitive = folder / "primitives" / (sketch+"_primitives.json")
    status = row.get("status", "pending" if current else "not_saved")
    info = {"status": status, "acceptance": row.get("acceptance_status", "not_evaluated"),
            "acceptance_reasons": json.loads(row.get("acceptance_reason_codes") or "[]"),
            "reason": row.get("error"), "views": {}, "links": {}, "metrics": [],
            "geometry": row.get("after"),
            "canonical_export": bool(current and svg.exists() and status == "ok")}
    if not current and svg.exists() and not row.get("status"):
        info["status"] = "saved_diagnostic"
    if svg.exists():
        try:
            label = "Canonical output" if info["canonical_export"] else "Partial SVG" if current else "Saved output"
            info["views"]["vector"] = dict(assets.preview(svg, prefix+"_vector", "svg"), label=label)
            tree = ET.parse(svg)
            info["embedded_images"] = sum(e.tag.rsplit("}", 1)[-1] == "image" for e in tree.iter())
        except Exception as exc:
            info["preview_error"] = f"SVG preview: {type(exc).__name__}: {exc}"
    if current and not svg.exists():
        report = root / "review_diagnostics" / patent / (sketch+"_review.json")
        if report.exists():
            diagnostic = json.loads(report.read_text())
            info["diagnostic"] = diagnostic
            if diagnostic.get("status") == "rendered":
                path = Path(diagnostic["svg"])
                try:
                    info["views"]["diagnostic"] = dict(assets.preview(path, prefix+"_diagnostic", "svg"),
                                                       label="Diagnostic only")
                    info["links"]["diagnostic_svg"] = assets.link(path)
                except Exception as exc:
                    info["preview_error"] = f"Diagnostic preview: {type(exc).__name__}: {exc}"
    if graph.exists():
        try:
            # A live gallery can encounter a worker's incomplete graph write.
            data = json.loads(graph.read_text())
            info["metrics"].append(f"{len(data.get('edges', [])):,} main strokes")
            coverage = data.get("coverage")
            if coverage:
                info["coverage"] = {k: coverage[k] for k in (
                    "source_pixels", "represented_source_pixels", "residual_source_pixels")}
                info["metrics"].append(f"{coverage['represented_fraction']*100:.3f}% source coverage")
            # Graphs are intermediate; they never stand in for a vector export.
            info["views"]["graph"] = dict(assets.preview(graph, prefix+"_graph", "graph"), label="Stage 2 graph")
        except Exception as exc:
            info["graph_preview_error"] = f"{type(exc).__name__}: {exc}"
    if row.get("s3_n_budget_primitives") is not None:
        info["metrics"].append(f"{row['s3_n_budget_primitives']:,} budget strokes")
    for name, path in (("svg", svg), ("dxf", svg.with_suffix(".dxf")), ("json", primitive), ("graph", graph)):
        if path.exists():
            info["links"][name] = assets.link(path)
    info["initial_view"] = next(iter(info["views"]), None)
    return info


def build(worklist, current, previous, output, icons, *, allow_partial=False):
    rows = load_worklist(worklist)
    now, old = _load(current), historical_rows(previous)
    completed = sum((r["patent_id"], r["sketch_id"]) in now for r in rows)
    if completed != len(rows) and not allow_partial:
        raise ValueError(f"Current batch is incomplete: {completed}/{len(rows)}")
    assets = Assets(output)
    figures = []
    for index, row in enumerate(rows):
        patent, sketch = row["patent_id"], row["sketch_id"]
        key, prefix = (patent, sketch), f"{index+1:03d}_{patent}_{sketch}"
        original = Path(row["input_path"])
        figure = {"number": index+1, "patent": patent, "sketch": sketch,
                  "original": assets.preview(original, prefix+"_original", "raster"),
                  "tif": assets.link(original)}
        figure["current"] = describe_run(current, patent, sketch, now.get(key, {}), assets,
                                          prefix+"_current", current=True)
        figure["previous"] = describe_run(previous, patent, sketch, old.get(key, {}), assets,
                                           prefix+"_previous", current=False)
        figure["original"]["label"] = "Original TIF"
        figures.append(figure)
        if (index+1) % 10 == 0:
            assets.finish()
            print(f"Gallery assets {index+1}/{len(rows)}", flush=True)
    icon_output = output / "icons"
    icon_output.mkdir(exist_ok=True)
    for name in ("arrow-left", "arrow-right", "zoom-in", "zoom-out", "scan", "download", "external-link"):
        shutil.copy2(icons / (name+".svg"), icon_output / (name+".svg"))
    license_path = icons.parent / "LICENSE"
    if license_path.exists():
        shutil.copy2(license_path, icon_output / "LICENSE")
    summary = {"samples": len(rows), "completed": completed,
               "current_exports": sum(f["current"]["canonical_export"] for f in figures),
               "current_diagnostics": sum("diagnostic" in f["current"]["views"] for f in figures),
               "previous_outputs": sum("vector" in f["previous"]["views"] for f in figures),
               "current_statuses": dict(Counter(f["current"]["status"] for f in figures)),
               "acceptance": dict(Counter(f["current"]["acceptance"] for f in figures)),
               "preview_errors": sum(bool(f[side].get("preview_error")) for f in figures for side in ("current", "previous")),
               "graph_preview_errors": sum(bool(f[side].get("graph_preview_error")) for f in figures for side in ("current", "previous")),
               "current_root": str(current.resolve()), "previous_root": str(previous.resolve()),
               "worklist_sha256": digest(worklist), "pipeline_modified_for_review": False}
    deployment = current / "deployment_run.json"
    if deployment.exists():
        report = json.loads(deployment.read_text())
        summary["deployment_identity"] = report["identity"]
        summary["devices"] = {"puhachov": report["effective_config"]["puhachov"]["device"],
                              "hatch": report["effective_config"]["stage2"].get("hachure_cnn_device"),
                              "ocr_gpu": report["effective_config"]["stage0"].get("ocr_gpu")}
    document = {"summary": summary, "figures": figures}
    template = Path(__file__).with_name("pipeline_review.html").read_text()
    output.joinpath("index.html").write_text(template.replace("__REVIEW_DATA__", inline_json(document)))
    output.joinpath("review.json").write_text(json.dumps(document, indent=2, allow_nan=False)+"\n")
    shutil.copy2(worklist, output / "samples.csv")
    assets.finish()
    print(json.dumps(summary, indent=2), flush=True)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--icons", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--watch-count", type=int, default=0,
                        help="Refresh the offline gallery every five completed rows until this count")
    args = parser.parse_args()
    if args.watch_count < 0:
        parser.error("Watch count must be nonnegative")
    last = -5
    while True:
        completed = len(_load(args.current))
        if not args.watch_count or completed-last >= 5 or completed >= args.watch_count:
            build(args.worklist, args.current, args.previous, args.output, args.icons,
                  allow_partial=args.allow_partial or bool(args.watch_count and completed < args.watch_count))
            last = completed
        if not args.watch_count or completed >= args.watch_count:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
