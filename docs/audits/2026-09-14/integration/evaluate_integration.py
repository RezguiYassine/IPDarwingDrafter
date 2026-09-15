"""Reproduce fresh CAD checks, paired integration evidence and width diagnostics."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from tools.batch_run import stage2_stroke_extract as s2, stage3_primitive_fit as s3, stage4_export as s4
from tools import geometry_validation as geometry
from tools.evaluate_patent_fidelity import _primitive_mask, fidelity

AUDIT = Path(__file__).resolve().parent
CAD = ROOT / "output/Drawing2CAD250_Stage2CoverageV2"
NEW = ROOT / "output/Drawing2CAD250_CoverageIntegrationV2"
PATENT = ROOT / "output/PatentData2_CoverageIntegrationPilotV3"
SMOKE = ROOT / "output/PatentData2_CoverageIntegrationSmokeV2"
CONFIG = ROOT / "output/Drawing2CAD/stage3_open_closed_full/evaluation_config.yaml"


def aggregate(path):
    data = json.loads(path.read_text())
    rows = data["rows"]
    assert all(r["execution"] == "ok" and r["source_graph_unchanged"]
               and r["source_pixels_unchanged"] and r["integration"]["main_pixel_set_preserved"]
               and r["integration"]["hatch_edges_unchanged"] for r in rows)
    result = {"path": str(path.relative_to(ROOT)), "sha256": geometry.file_digest(path),
              "implementation": data["implementation"], "before": data["before_counts"],
              "after": data["after_counts"], "raster": {},
              "before_edges": sum(r["integration"]["before_edges"] for r in rows),
              "after_edges": sum(r["integration"]["after_edges"] for r in rows),
              "residual_pixels": sum(r["coverage"]["residual_source_pixels"] for r in rows)}
    for key in rows[0]["raster"]["before"]:
        a, b = [r["raster"]["before"][key] for r in rows], [r["raster"]["after"][key] for r in rows]
        delta = [y-x for x, y in zip(a, b)]
        result["raster"][key] = {"before": float(np.mean(a)), "after": float(np.mean(b)),
                                 "increases": sum(d > 1e-10 for d in delta),
                                 "decreases": sum(d < -1e-10 for d in delta),
                                 "worst_delta": max(delta) if key.startswith("chamfer") else min(delta)}
    for side in ("before", "after"):
        totals, statuses = Counter(), Counter()
        for row in rows:
            totals.update(row[side+"_encoding"])
            statuses.update(row[side+"_summary"]["by_status"])
        result[side+"_encoding"] = dict(totals)
        result[side+"_primitive_status"] = dict(statuses)
    result["f1_regressions"] = sorted([
        {"folder": r["folder"], "sketch_id": r["sketch_id"],
         "delta": r["raster"]["after"]["f1_at_2px"]-r["raster"]["before"]["f1_at_2px"]}
        for r in rows if r["raster"]["after"]["f1_at_2px"] < r["raster"]["before"]["f1_at_2px"]-1e-10
    ], key=lambda r: r["delta"])
    return result


def fresh_cad(output):
    config = yaml.safe_load(CONFIG.read_text())
    config["puhachov"]["device"] = "cpu"
    model = s2.load_model(config)
    if model is None:
        raise RuntimeError("Fresh CAD checks require the trained D2C Puhachov model")
    rows = []
    for folder, sketch in (("0026_00267762", "0026_00267762_Right"),
                           ("0078_00787655", "0078_00787655_FrontTopRight"),
                           ("0079_00793147", "0079_00793147_Front")):
        source = ROOT / "output/Stage2CoverageFresh5" / folder
        target = output / folder
        skeleton_path = source / "cleaned" / (sketch+"_skeleton.png")
        skeleton = cv2.imread(str(skeleton_path), 0)
        (target / "cleaned").mkdir(parents=True)
        shutil.copy2(skeleton_path, target / "cleaned" / skeleton_path.name)
        extracted = s2.run(skeleton_path, target, sketch, config, model=model)
        graph = json.loads(extracted.graph_path.read_text())
        original = source / "primitives" / (sketch+"_primitives.json")
        before = json.loads(original.read_text())
        fitted = s3.run(extracted.graph_path, target, sketch, config, stroke_width=before.get("stroke_width"))
        report = geometry.run(extracted.graph_path, fitted.primitives_path, skeleton_path,
                              target / "geometry" / (sketch+"_geometry_report.json"))
        with tempfile.TemporaryDirectory(prefix="integration_fresh_") as temp:
            raster = {side: fidelity(skeleton > 0, _primitive_mask(path, Path(temp)/side, sketch, skeleton.shape))
                      for side, path in (("before", original), ("after", fitted.primitives_path))}
        exported = s4.run(fitted.primitives_path, target, sketch, formats=("svg", "dxf"))
        assert exported.n_primitives_in == exported.n_primitives_out
        old_graph = json.loads((source / "graphs" / (sketch+"_graph.json")).read_text())
        source_set = lambda g: {tuple(p) for e in g["edges"] for p in e["pixels"]}
        assert source_set(old_graph) == source_set(graph)
        assert old_graph["removed_hachures"] == graph["removed_hachures"]
        rows.append({"folder": folder, "sketch_id": sketch, "geometry": report["status"],
                     "stage2_flagged": extracted.flagged, "stage3_flagged": fitted.flagged,
                     "primitive_status": report["summary"]["by_status"], "raster": raster,
                     "coverage": graph["coverage"], "main_edges": len(graph["edges"]),
                     "source_skeleton_sha256": geometry.file_digest(skeleton_path),
                     "graph_sha256": geometry.file_digest(extracted.graph_path)})
        print("Fresh", sketch, report["status"], flush=True)
    return rows


def width_diagnostics(cases):
    rows = []
    canvas = Image.new("RGB", (1440, len(cases)*430), "white")
    draw = ImageDraw.Draw(canvas)
    for index, case in enumerate(cases):
        folder, sketch = case["folder"], case["sketch_id"]
        source = cv2.imread(str(CAD / folder / "cleaned" / (sketch+"_skeleton.png")), 0)
        y, x = np.nonzero(source)
        box = (max(0, int(x.min())-10), max(0, int(y.min())-10),
               min(source.shape[1], int(x.max())+11), min(source.shape[0], int(y.max())+11))
        images = [Image.fromarray(255-source).convert("RGB")]
        scores = {}
        with tempfile.TemporaryDirectory(prefix="integration_width_") as temp:
            for side, root in (("before", CAD), ("after", NEW)):
                path = root / folder / "primitives" / (sketch+"_primitives.json")
                document = json.loads(path.read_text())
                original_width = document["stroke_width"]
                unit = deepcopy(document)
                unit["stroke_width"] = 1.0
                unit_path = Path(temp) / (side+"_unit.json")
                unit_path.write_text(json.dumps(unit))
                scores[side] = fidelity(source > 0, _primitive_mask(unit_path, Path(temp)/side, sketch, source.shape))
                svg = root / folder / "vectors" / (sketch+".svg")
                images.append(Image.open(io.BytesIO(cairosvg.svg2png(url=str(svg), background_color="white"))).convert("RGB"))
        for col, (title, image) in enumerate(zip(("Stage 1 skeleton", "Before integration", "After integration"), images)):
            left, top = col*480, index*430
            draw.text((left+10, top+8), title+" | "+sketch, fill="black")
            tile = image.crop(box)
            tile.thumbnail((460, 390), Image.Resampling.NEAREST)
            canvas.paste(tile, (left+10, top+35))
        rows.append(dict(case, original_stroke_width=original_width, unit_width_raster=scores,
                         scope="Diagnostic width=1 only; actual exported stroke width unchanged"))
    canvas.save(AUDIT / "rendered_outliers.png")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"date": "2026-09-14", "fresh_cad": fresh_cad(args.output),
                "replays": {"cad250": aggregate(NEW / "summary.json"),
                            "patent2": aggregate(PATENT / "summary.json")},
                "cad_config_sha256": geometry.file_digest(CONFIG),
                "cad_weights_sha256": geometry.file_digest(ROOT / "models/puhachov_d2c.pth"),
                "implementation": {str(Path(m.__file__).relative_to(ROOT)): geometry.file_digest(Path(m.__file__))
                                   for m in (s2, s3, s4, geometry)}}
    with sqlite3.connect(SMOKE / "results.db") as db:
        db.row_factory = sqlite3.Row
        evidence["canonical"] = [dict(r) for r in db.execute(
            "SELECT patent_id,status,error,s2_n_edges,s2_micro_edge_ratio,s2_short_edge_ratio,"
            "s3_n_primitives,s3_n_budget_primitives,acceptance_status,training_eligible FROM results")]
    with (SMOKE / "training_manifest.csv").open(newline="") as stream:
        evidence["training_manifest_kept"] = len(list(csv.DictReader(stream)))
    evidence["width_diagnostics"] = width_diagnostics(evidence["replays"]["cad250"]["f1_regressions"][:3])
    for path in (args.output / "summary.json", AUDIT / "evidence.json"):
        path.write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
