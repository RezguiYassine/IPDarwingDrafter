"""Re-export frozen patent primitives and check export integrity and SVG appearance."""

import argparse
import io
import json
import logging
from pathlib import Path
import sys

import cairosvg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.batch_run import stage4_export  # noqa: E402


def raster(path, width, height):
    scale = 512 / max(width, height)
    data = cairosvg.svg2png(url=str(path), output_width=max(1, round(width * scale)),
                           output_height=max(1, round(height * scale)), background_color="white")
    return np.asarray(Image.open(io.BytesIO(data)).convert("L"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; source and prior checks are preserved.")
    logging.basicConfig(level=logging.ERROR)
    rows = []
    for path in sorted(args.source.glob("*/primitives/*_primitives_with_refs.json")):
        document = json.loads(path.read_text())
        patent, sketch = path.parent.parent.name, document["sketch_id"]
        result = stage4_export.run(path, args.output / patent, sketch,
                                   formats=("svg", "dxf"), dxf_mode="patent")
        old_svg = args.source / patent / "vectors" / f"{sketch}.svg"
        width, height = document["image_size"]
        changed = int(np.count_nonzero(raster(old_svg, width, height)
                                      != raster(result.svg_path, width, height)))
        rows.append({
            "patent_id": patent, "sketch_id": sketch,
            "expected_primitives": result.n_primitives_in,
            "minimum_exported_primitives": result.n_primitives_out,
            "svg_changed_pixels_at_512": changed,
            "formats": {name: {
                "serialized_valid": data["serialized_valid"],
                "primitives_written": data["primitives_written"],
                "errors": data["errors"],
                "incomplete_primitives": sum(not p["complete"] for p in data["primitives"]),
                "unknown_labels": sum(a.get("label") == "unknown" for a in data["annotations"]),
                "annotation_errors": sum(bool(a["errors"]) for a in data["annotations"]),
            } for name, data in result.format_reports.items()},
        })
        if len(rows) % 10 == 0:
            print(f"Checked {len(rows)} figures", flush=True)
    summary = {
        "drawings": len(rows),
        "identical_svg_thumbnails": sum(r["svg_changed_pixels_at_512"] == 0 for r in rows),
        "fully_exported_geometry": sum(r["expected_primitives"] == r["minimum_exported_primitives"] for r in rows),
        "serialized_formats_valid": sum(d["serialized_valid"] for r in rows for d in r["formats"].values()),
        "unknown_dxf_labels": sum(r["formats"]["dxf"]["unknown_labels"] for r in rows),
    }
    report = {"source": str(args.source), "output": str(args.output), "summary": summary,
              "scope": "Stage 4 only; thumbnail appearance is not a full-resolution geometry accuracy metric.",
              "rows": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(not rows or summary["fully_exported_geometry"] != len(rows)
               or summary["serialized_formats_valid"] != 2 * len(rows)
               or summary["identical_svg_thumbnails"] != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
