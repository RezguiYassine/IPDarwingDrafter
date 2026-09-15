"""Audit saved reference/hatch preservation without rerunning inference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

import ezdxf


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(run: Path, baseline: Path | None = None) -> dict:
    with sqlite3.connect((run / "results.db").resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM results ORDER BY patent_id, sketch_id")]
    report = {"run": str(run), "baseline": str(baseline) if baseline else None,
              "status_counts": dict(Counter(row["status"] for row in rows)),
              "rows": [], "errors": []}
    for row in rows:
        patent, sketch = row["patent_id"], row["sketch_id"]
        root = run / patent
        path = root / "primitives" / f"{sketch}_primitives.json"
        if not path.exists():
            if row["status"] == "ok":
                report["errors"].append(f"{patent}/{sketch}: missing primitives for ok row")
            continue
        graph = read_json(root / "graphs" / f"{sketch}_graph.json")
        document = read_json(path)
        primitives = document["primitives"]
        main = [p for p in primitives if p.get("style") != "hachure"]
        hatch = [p for p in primitives if p.get("style") == "hachure"]
        owners = Counter(index for p in hatch for index in p.get("source_hachure_indices", []))
        expected = set(range(len(graph.get("removed_hachures", []))))
        coverage = document["quality_metrics"]["hachure_coverage"]
        record = {"patent_id": patent, "sketch_id": sketch,
                  "coverage": coverage, "main_primitives": len(main),
                  "hatch_primitives": len(hatch), "keypoint_source": graph.get("keypoint_source"),
                  "hachure_source": graph.get("hachure_source"), "stage1_method": row["s1_model_used"]}
        errors = []
        if set(owners) != expected or any(count != 1 for count in owners.values()):
            errors.append("hatch source ownership is missing or duplicated")
        if coverage["unrepresented_indices"] or coverage["represented_edges"] != len(expected):
            errors.append("hatch coverage is incomplete")
        if baseline:
            previous_root = baseline / patent
            previous = read_json(previous_root / "primitives" / path.name)["primitives"]
            record["main_identical"] = main == [p for p in previous if p.get("style") != "hachure"]
            record["additional_hatch_primitives"] = len(hatch) - sum(p.get("style") == "hachure" for p in previous)
            record["frozen_inputs_identical"] = all(
                digest(root / relative) == digest(previous_root / relative)
                for relative in [f"graphs/{sketch}_graph.json", f"cleaned/{sketch}_skeleton.png",
                                 f"references/{sketch}_references.json"]
            )
            if not record["main_identical"] or not record["frozen_inputs_identical"]:
                errors.append("paired main geometry or frozen inputs changed")
        references = read_json(root / "references" / f"{sketch}_references.json")
        labels = references["reference_labels"]
        known = Counter(str(label["text"]) for label in labels if label.get("text"))
        reinjected = labels if references.get("active_removal", False) else []
        expected_text = known if reinjected else Counter()
        record["references"] = {"total": len(labels), "known_text": sum(known.values()),
                                "unknown_text": sum(not label.get("text") for label in labels),
                                "missing_crops": sum(not Path(label["crop_path"]).is_file() for label in labels)}
        if record["references"]["missing_crops"]:
            errors.append("reference crops are missing")
        try:
            svg = ET.parse(root / "vectors" / f"{sketch}.svg").getroot()
            dxf = ezdxf.readfile(root / "vectors" / f"{sketch}.dxf")
            text = Counter(entity.plain_text() for entity in dxf.modelspace() if entity.dxftype() == "MTEXT")
            record["exports"] = {"svg_images": len(svg.findall(".//{*}image")),
                                 "svg_texts": len(svg.findall(".//{*}text")),
                                 "dxf_texts": sum(text.values()),
                                 "dxf_audit_errors": len(dxf.audit().errors),
                                 "dxf_known_text_matches": expected_text == text}
            if record["exports"]["dxf_audit_errors"] or expected_text != text:
                errors.append("DXF audit or recognized-text round trip failed")
            if record["exports"]["svg_images"] != len(reinjected) or record["exports"]["svg_texts"]:
                errors.append("SVG default crop representation is missing or double-rendered")
        except (OSError, ValueError, ET.ParseError, ezdxf.DXFError) as exc:
            errors.append(f"export parse failed: {exc}")
        report["rows"].append(record)
        report["errors"].extend(f"{patent}/{sketch}: {error}" for error in errors)
    report["totals"] = {
        "exported_drawings": len(report["rows"]),
        "source_hatch_edges": sum(row["coverage"]["source_edges"] for row in report["rows"]),
        "represented_hatch_edges": sum(row["coverage"]["represented_edges"] for row in report["rows"]),
        "known_reference_texts": sum(row["references"]["known_text"] for row in report["rows"]),
        "unknown_reference_texts": sum(row["references"]["unknown_text"] for row in report["rows"]),
    }
    if baseline:
        report["totals"].update({
            "main_identical_drawings": sum(row["main_identical"] for row in report["rows"]),
            "additional_hatch_primitives": sum(row["additional_hatch_primitives"] for row in report["rows"]),
            "changed_hatch_drawings": sum(row["additional_hatch_primitives"] != 0 for row in report["rows"]),
        })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"totals": report["totals"], "errors": report["errors"]}, indent=2))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
