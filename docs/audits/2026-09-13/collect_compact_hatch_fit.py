"""Reproduce the compact-hatch/fitting audit after the paired replays finish.

Run from the repository root with .venv/bin/python and this file's path.
Only generated audit artifacts are written; source runs are never modified.
"""

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.batch_run import stage4_export as s4

OUT = Path(__file__).resolve().parent
RUNS = {
    "patent91": "PatentData91_CompactHatchFitV2",
    "cad250": "Drawing2CAD250_CompactHatchFitV2",
    "pilot2": "PatentData2_CompactHatchFitV4",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def aggregate(name):
    root = ROOT / "output" / name
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    rows = summary["rows"]
    assert all(r["execution"] == "ok" and r["source_pixels_unchanged"]
               and r["ownership"]["status"] == "pass" and r["export_geometry_complete"] for r in rows)
    assert all(digest(Path(p)) == sha for p, sha in summary["implementation"].items())
    result = {"path": str(root.relative_to(ROOT)), "summary_sha256": digest(path),
              "implementation": summary["implementation"], "config_sha256": summary["config_sha256"],
              "drawings": len(rows), "before": {}, "after": {}, "raster": {},
              "all_graphs_unchanged": True, "all_ownership_pass": True, "all_exports_complete": True}
    for side in ("before", "after"):
        counters = {k: Counter() for k in ("drawing_status", "primitive_status", "curves", "reasons", "encoding")}
        for row in rows:
            counters["drawing_status"].update([row[side]])
            counters["primitive_status"].update(row[side + "_summary"]["by_status"])
            counters["curves"].update(row[side + "_curves"])
            counters["reasons"].update(row[side + "_reasons"])
            counters["encoding"].update(row[side + "_encoding"])
        result[side] = {k: dict(v) for k, v in counters.items()}
        result[side]["over_900_budget"] = sum(r[side+"_encoding"]["budget_primitives"] > 900 for r in rows)
    for metric in rows[0]["raster"]["before"]:
        before = [r["raster"]["before"][metric] for r in rows]
        after = [r["raster"]["after"][metric] for r in rows]
        delta = [b-a for a, b in zip(before, after)]
        lower = metric.startswith("chamfer")
        result["raster"][metric] = {
            "before": sum(before)/len(rows), "after": sum(after)/len(rows),
            "improved": sum((d < -1e-10 if lower else d > 1e-10) for d in delta),
            "regressed": sum((d > 1e-10 if lower else d < -1e-10) for d in delta),
            "ties": sum(abs(d) <= 1e-10 for d in delta),
            "worst": [{"folder": rows[i]["folder"], "sketch_id": rows[i]["sketch_id"], "delta": delta[i]}
                      for i in sorted(range(len(rows)), key=lambda i: delta[i], reverse=lower)[:5]],
        }
    remaining, by_collection, missing_graph_ink = [], Counter(), []
    for row in rows:
        report = json.loads((root / row["folder"] / "geometry" / (row["sketch_id"] + "_geometry_report.json")).read_text())
        if len(rows) <= 250 and name != RUNS["patent91"] and report["status"] != "pass":
            base, sketch = root / row["folder"], row["sketch_id"]
            graph = json.loads((base / "graphs" / (sketch+"_graph.json")).read_text())
            points = np.vstack([e["pixels"] for key in ("edges", "removed_hachures")
                                for e in graph.get(key, []) if e.get("pixels")])
            y, x = np.nonzero(cv2.imread(str(base / "cleaned" / (sketch+"_skeleton.png")), 0))
            raster = np.column_stack([x, y]) * graph.get("stage2_scale", 1)
            distance = cKDTree(points).query(raster)[0]
            missing_graph_ink.append({"folder": row["folder"], "sketch_id": sketch,
                                      "graph_distance_max": float(distance.max()),
                                      "skeleton_pixels_over_8px_from_graph": int((distance > 8).sum()),
                                      "worst_point_stage2": raster[distance.argmax()].tolist()})
        for item in report["primitives"]:
            if item["status"] != "pass":
                by_collection.update([item["source_collection"] + ":" + item["status"]])
                remaining.append({"folder": row["folder"], "sketch_id": row["sketch_id"], **item})
    result["remaining_by_collection"] = dict(by_collection)
    result["remaining_primitives"] = remaining
    result["missing_graph_ink"] = missing_graph_ink
    if len(rows) <= 2:
        result["rows"] = rows
    return result


def canonical():
    root = ROOT / "output/PatentData2_CompactHatchFitSmokeV2"
    with sqlite3.connect(root / "results.db") as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(r) for r in connection.execute(
            "SELECT patent_id, sketch_id, status, error, s3_n_primitives, s3_n_budget_primitives, "
            "acceptance_status, acceptance_reason_codes, training_eligible FROM results ORDER BY patent_id")]
    baseline = ROOT / "output/PatentData2_ResidualRepairGuardedSmoke"
    pilot = ROOT / "output" / RUNS["pilot2"]
    for row in rows:
        folder, sketch = row["patent_id"], row["sketch_id"]
        row["pilot_parity"] = {}
        for sub, suffix in (("graphs", "_graph.json"), ("primitives", "_primitives.json"), ("cleaned", "_skeleton.png")):
            relative = Path(folder) / sub / (sketch + suffix)
            row["pilot_parity"][sub] = digest(root / relative) == digest(pilot / relative)
        row["preprocessing_parity"] = {}
        for sub, suffix in (("cleaned", "_skeleton.png"), ("references", "_norefs.png")):
            relative = Path(folder) / sub / (sketch + suffix)
            row["preprocessing_parity"][sub] = digest(root / relative) == digest(baseline / relative)
        assert all(row["pilot_parity"].values()) and all(row["preprocessing_parity"].values())
    with (root / "training_manifest.csv").open(newline="") as stream:
        kept = len(list(csv.DictReader(stream)))
    assert kept == 0 and all(not r["training_eligible"] for r in rows)
    return {"path": str(root.relative_to(ROOT)), "rows": rows, "training_manifest_kept": kept,
            "deployment_sha256": digest(root / "deployment_run.json")}


def preview():
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    canvas = Image.new("RGB", (1560, 1360), "white")
    draw = ImageDraw.Draw(canvas)
    cases = [("EP2565056B1", "F0001", "PatentData2_ResidualRepairHatchGuard", RUNS["pilot2"], None),
             ("EP3082506B1", "F0001", "PatentData2_ResidualRepairHatchGuard", RUNS["pilot2"], None),
             ("EP1839350B1", "F0002", "PatentData100_ResidualRepairV2", RUNS["patent91"], (375, 1655, 480, 1760))]
    bounds = []
    for row, (folder, sketch, before, after, box) in enumerate(cases):
        root = ROOT / "output" / after / folder
        graph = json.loads((root / "graphs" / (sketch+"_graph.json")).read_text())
        if box is None:
            groups = Counter()
            for edge in graph["removed_hachures"]:
                points = np.asarray(edge["pixels"], dtype=float) / graph.get("stage2_scale", 1)
                for x, y in points:
                    groups[(int(x//256), int(y//256))] += 1
            x, y = groups.most_common(1)[0][0]
            box = (max(0, x*256-12), max(0, y*256-12), (x+1)*256+12, (y+1)*256+12)
        bounds.append({"folder": folder, "sketch_id": sketch, "crop_original_pixels": box})
        skeleton = Image.open(root / "cleaned" / (sketch+"_skeleton.png")).convert("L")
        images = [Image.fromarray(255-np.asarray(skeleton))]
        with tempfile.TemporaryDirectory(prefix="compact_hatch_preview_") as temp:
            for label, run in (("before", before), ("after", after)):
                primitive = ROOT / "output" / run / folder / "primitives" / (sketch+"_primitives.json")
                exported = s4.run(primitive, Path(temp) / label, sketch, formats=("svg",), dxf_mode="basic")
                png = cairosvg.svg2png(url=str(exported.svg_path), background_color="white")
                images.append(Image.open(io.BytesIO(png)).convert("RGB"))
        for column, (label, image) in enumerate(zip(("Stage 1 source", "Before", "After"), images)):
            x, y = column*520+12, row*450+12
            draw.text((x, y), folder+" / "+label, fill="black", font=font)
            crop = image.crop(box).convert("RGB")
            scale = min(490/crop.width, 390/crop.height)
            crop = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.NEAREST)
            canvas.paste(crop, (x, y+35))
    canvas.save(OUT / "compact_hatch_fit_preview.png")
    return bounds


if __name__ == "__main__":
    suite_path = OUT / "compact_hatch_fit_tests.xml"
    suite = ET.parse(suite_path).getroot().find("testsuite")
    assert int(suite.get("failures")) == int(suite.get("errors")) == 0
    evidence = {"date": "2026-09-13", "scope": "Paired source-skeleton fidelity, not CAD ground truth or release acceptance",
                "tests": dict(suite.attrib), "test_report_sha256": digest(suite_path),
                "runs": {key: aggregate(name) for key, name in RUNS.items()},
                "canonical": canonical(), "preview_crops": preview()}
    (OUT / "compact_hatch_fit_evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: {"before": value["before"]["primitive_status"],
                           "after": value["after"]["primitive_status"], "raster": value["raster"]}
                      for key, value in evidence["runs"].items()}, indent=2))
