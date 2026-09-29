#!/usr/bin/env python3
"""Contact sheets for auditing reference numerals against their own crops.

The vocabulary gate can tell that a reading is absent from the patent's
description. It cannot tell that a reading present in the description was put
on the wrong mark, or that a '6' was read where an '8' is drawn. Only a person
looking at the crop can, and this is the cheapest way to let them.

One sheet per figure: every identified label as the crop Stage 0 cut, with the
text the pipeline assigned printed beneath it. Scanning a sheet takes about a
minute and covers every numeral on the drawing, which is far faster than
clicking through a viewer one label at a time and gives a per-figure error
rate rather than a scatter.

    python -m tools.build_label_audit --run output/RELEASE_2026-09-28 \
        --worklist benchmarks/mechanical/k2_audit_full_100.csv \
        --figures 15 --output output/K2_label_audit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2                      # noqa: E402
import numpy as np              # noqa: E402


def sheet(labels: list[dict], title: str, cell: int = 132, cols: int = 10) -> np.ndarray:
    rows = max(1, (len(labels) + cols - 1) // cols)
    canvas = np.full((rows * cell + 34, cols * cell), 255, np.uint8)
    cv2.putText(canvas, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, 0, 1)
    for index, label in enumerate(labels):
        crop_path = label.get("crop_path")
        if not crop_path or not Path(crop_path).is_file():
            continue
        crop = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        if crop is None:
            continue
        if crop.ndim == 3:                       # crops carry an alpha channel
            alpha = crop[:, :, 3] if crop.shape[2] == 4 else None
            grey = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
            crop = np.where(alpha == 0, 255, grey).astype(np.uint8) if alpha is not None else grey
        box = cell - 40
        scale = min(box / max(1, crop.shape[0]), box / max(1, crop.shape[1]), 4.0)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        r, c = divmod(index, cols)
        oy, ox = r * cell + 34, c * cell
        y = oy + (box - crop.shape[0]) // 2 + 4
        x = ox + (cell - crop.shape[1]) // 2
        canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        # the number the pipeline will write into the DXF for this mark
        cv2.putText(canvas, str(label.get("text", "")), (ox + 6, oy + cell - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, 0, 2)
        cv2.putText(canvas, f"#{index + 1}", (ox + 6, oy + cell - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, 110, 1)
    for r in range(1, rows):
        canvas[r * cell + 34, :] = 205
    for c in range(1, cols):
        canvas[34:, c * cell] = 205
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--worklist", type=Path, required=True)
    ap.add_argument("--figures", type=int, default=15,
                    help="figures to sheet, taken in worklist order")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.worklist)))
    made, total = 0, 0
    index_rows = []
    for row in rows:
        if made >= args.figures:
            break
        path = args.run / row["patent_id"] / "references" / f"{row['sketch_id']}_references.json"
        if not path.is_file():
            continue
        labels = [l for l in json.loads(path.read_text()).get("reference_labels") or []
                  if l.get("text")]
        if not labels:
            continue
        made += 1
        total += len(labels)
        name = f"{made:02d}_{row['patent_id']}_{row['sketch_id']}.png"
        title = (f"{made:02d}  {row['patent_id']}/{row['sketch_id']}   "
                 f"{len(labels)} identified numerals   "
                 f"mark any whose printed text does not match the crop")
        cv2.imwrite(str(args.output / name), sheet(labels, title))
        index_rows.append({"sheet": name, "patent_id": row["patent_id"],
                           "sketch_id": row["sketch_id"], "labels": len(labels),
                           "wrong_numbers": "", "notes": ""})
    with open(args.output / "label_audit.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sheet", "patent_id", "sketch_id",
                                                "labels", "wrong_numbers", "notes"])
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"sheets   : {made}")
    print(f"numerals : {total}")
    print(f"output   : {args.output}")
    print(f"tally in : {args.output / 'label_audit.csv'}  (fill wrong_numbers with the #N you spot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
