"""Fresh targeted Stage 2 checks and paired coverage evidence. Run from repo root."""

import argparse
from collections import Counter
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.batch_run import stage0_handle_references as s0, stage2_stroke_extract as s2, stage3_primitive_fit as s3, stage4_export as s4
from tools import geometry_validation as geometry
from tools.evaluate_patent_fidelity import _primitive_mask, fidelity

AUDIT = Path(__file__).resolve().parent
CAD = ROOT / "output/Drawing2CAD250_CompactHatchFitV2"
PATENT = ROOT / "output/PatentData2_CompactHatchFitV4"
SMOKE = ROOT / "output/PatentData2_Stage2CoverageSmokeV2"


def aggregate(path):
    d = json.loads(path.read_text())
    rows = d["rows"]
    assert all(r["execution"] == "ok" and r["source_graph_unchanged"] and r["original_edges_unchanged"] for r in rows)
    result = {"path": str(path.relative_to(ROOT)), "sha256": geometry.file_digest(path),
              "implementation": d["implementation"], "before": d["before_counts"], "after": d["after_counts"],
              "recovered_pixels": sum(r["coverage"]["recovery"]["recovered_pixels"] for r in rows),
              "new_edges": sum(r["coverage"]["recovery"]["new_edges"] for r in rows),
              "residual_pixels": sum(r["coverage"]["residual_source_pixels"] for r in rows), "raster": {}}
    for key in rows[0]["raster"]["before"]:
        a, b = [r["raster"]["before"][key] for r in rows], [r["raster"]["after"][key] for r in rows]
        delta = [y-x for x, y in zip(a, b)]
        result["raster"][key] = {"before": float(np.mean(a)), "after": float(np.mean(b)),
                                 "increases": sum(d > 1e-10 for d in delta), "decreases": sum(d < -1e-10 for d in delta),
                                 "worst_delta": max(delta) if key.startswith("chamfer") else min(delta)}
    for side in ("before", "after"):
        totals, statuses = Counter(), Counter()
        for row in rows:
            totals.update(row[side+"_encoding"])
            statuses.update(row[side+"_summary"]["by_status"])
        result[side+"_encoding"] = dict(totals)
        result[side+"_primitive_status"] = dict(statuses)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "output/Drawing2CAD/stage3_open_closed_full/evaluation_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["puhachov"]["device"] = "cpu"
    model = s2.load_model(config)
    if model is None:
        raise RuntimeError("Fresh CAD checks require the recorded D2C Puhachov checkpoint")
    cases = [(CAD, "0026_00267762", "0026_00267762_Right"),
             (CAD, "0078_00787655", "0078_00787655_FrontTopRight"),
             (CAD, "0079_00793147", "0079_00793147_Front"),
             (PATENT, "EP2565056B1", "F0001"), (PATENT, "EP3082506B1", "F0001")]
    rows = []
    for baseline, folder, sketch in cases:
        target = args.output / folder
        source = (baseline if baseline == CAD else SMOKE) / folder
        skeleton_path = source / "cleaned" / (sketch+"_skeleton.png")
        skeleton = cv2.imread(str(skeleton_path), 0)
        original_hash = geometry.file_digest(skeleton_path)
        (target / "cleaned").mkdir(parents=True)
        shutil.copy2(skeleton_path, target / "cleaned" / skeleton_path.name)
        if baseline == CAD:
            extracted = s2.run(skeleton_path, target, sketch, config, model=model)
            gp, fit_config = extracted.graph_path, config
        else:
            gp = target / "graphs" / (sketch+"_graph.json")
            gp.parent.mkdir(parents=True)
            shutil.copy2(source / "graphs" / gp.name, gp)
            fit_config = yaml.safe_load((ROOT / "config_deploy.yaml").read_text())
        graph = json.loads(gp.read_text())
        before_path = baseline / folder / "primitives" / (sketch+"_primitives.json")
        before = json.loads(before_path.read_text())
        fitted = s3.run(gp, target, sketch, fit_config, stroke_width=before.get("stroke_width"))
        report = geometry.run(gp, fitted.primitives_path, skeleton_path, target / "geometry" / (sketch+"_geometry_report.json"))
        raster = {}
        with tempfile.TemporaryDirectory(prefix="coverage_audit_") as temp:
            for side, path in (("before", before_path), ("after", fitted.primitives_path)):
                raster[side] = fidelity(skeleton > 0, _primitive_mask(path, Path(temp)/side, sketch, skeleton.shape))
        export_path = fitted.primitives_path
        refs = source / "references" / (sketch+"_references.json")
        if refs.exists():
            export_path = s0.attach_references_to_primitives(export_path, refs)
        exported = s4.run(export_path, target, sketch, formats=("svg", "dxf"), dxf_mode="patent")
        assert exported.n_primitives_in == exported.n_primitives_out
        assert geometry.file_digest(skeleton_path) == original_hash
        row = {"folder": folder, "sketch_id": sketch, "geometry": report["status"],
               "reasons": report["reason_codes"], "primitive_status": report["summary"]["by_status"],
               "coverage": graph["coverage"], "graph_sha256": geometry.file_digest(gp),
               "source_skeleton_sha256": original_hash, "raster": raster,
               "main_edges": len(graph["edges"]), "hatch_edges": len(graph.get("removed_hachures", [])),
               "scope": "Fresh CAD Stage 2 or canonical patent graph, diagnostic ungated Stage 3/4"}
        rows.append(row)
        print(folder, report["status"], report["reason_codes"], flush=True)

    with sqlite3.connect(SMOKE / "results.db") as connection:
        connection.row_factory = sqlite3.Row
        canonical = [dict(r) for r in connection.execute(
            "SELECT patent_id,status,error,s2_micro_edge_ratio,s2_short_edge_ratio,acceptance_status,training_eligible FROM results")]
    with (SMOKE / "training_manifest.csv").open(newline="") as stream:
        manifest_rows = len(list(csv.DictReader(stream)))
    assert manifest_rows == 0
    evidence = {"date": "2026-09-14", "fresh": rows, "canonical": canonical,
                "training_manifest_kept": manifest_rows,
                "cad_config_sha256": geometry.file_digest(config_path),
                "cad_weights_sha256": geometry.file_digest(Path(config["puhachov"]["weights"])),
                "implementation": {str(Path(m.__file__).relative_to(ROOT)): geometry.file_digest(Path(m.__file__))
                                   for m in (s2, s3, s4, geometry)},
                "replays": {name: aggregate(ROOT / "output" / name / "summary.json") for name in (
                    "Drawing2CAD250_Stage2CoverageV2", "PatentData2_Stage2CoverageV2")}}
    (args.output / "summary.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    (AUDIT / "stage2_coverage_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")

    folder, sketch = "0078_00787655", "0078_00787655_FrontTopRight"
    box = (337, 375, 383, 411)
    canvas = Image.new("RGB", (1440, 420), "white")
    draw = ImageDraw.Draw(canvas)
    src = cv2.imread(str(CAD / folder / "cleaned" / (sketch+"_skeleton.png")), 0)
    images = [Image.fromarray(255-src)]
    for root in (CAD, args.output):
        png = cairosvg.svg2png(url=str(root / folder / "vectors" / (sketch+".svg")), background_color="white")
        images.append(Image.open(io.BytesIO(png)).convert("RGB"))
    for i, (title, image) in enumerate(zip(("Stage 1 source", "Before", "Fresh Stage 2 + guarded fit"), images)):
        draw.text((i*480+12, 10), title, fill="black")
        canvas.paste(image.crop(box).resize((460, 360), Image.Resampling.NEAREST), (i*480+10, 45))
    canvas.save(AUDIT / "stage2_coverage_preview.png")


if __name__ == "__main__":
    main()
