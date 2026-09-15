"""Audit frozen geometry without changing its exports or acceptance records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

from tools import geometry_validation as geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--graph-source", type=Path, help="Frozen graph/raster run when the source is a Stage 3 replay")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Seeded sample of figures; default all")
    parser.add_argument("--seed", type=int, default=850725)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory; historical evidence is never overwritten.")
    if not args.source.is_dir() or args.graph_source is not None and not args.graph_source.is_dir():
        parser.error("Source and graph-source must be existing run directories")
    paths = sorted(args.source.glob("*/primitives/*_primitives.json"))
    if not paths or args.limit is not None and args.limit < 1:
        parser.error("No primitive documents found or invalid sample limit")
    available = len(paths)
    if args.limit is not None and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    graph_root = args.graph_source or args.source
    rows, reasons, by_type = [], Counter(), defaultdict(Counter)
    for index, path in enumerate(paths):
        started = time.perf_counter()
        folder = path.parent.parent.name
        sketch = path.name.removesuffix("_primitives.json")
        destination = args.output / folder / "geometry" / f"{sketch}_geometry_report.json"
        try:
            graph_path = graph_root / folder / "graphs" / f"{sketch}_graph.json"
            skeleton_path = graph_root / folder / "cleaned" / f"{sketch}_skeleton.png"
            if args.graph_source is not None:
                for selected in (graph_path, skeleton_path):
                    local = args.source / selected.relative_to(graph_root)
                    if local.is_file() and geometry.file_digest(local) != geometry.file_digest(selected):
                        raise ValueError(f"External evidence differs from the source run's own artifact: {local}")
            report = geometry.run(
                graph_path, path, skeleton_path, destination)
            for item in report["primitives"]:
                by_type[item["type"]][item["status"]] += 1
            reasons.update(report["reason_codes"])
            rows.append({"folder": folder, "sketch_id": sketch, "status": report["status"],
                         "reason_codes": report["reason_codes"], "summary": report["summary"],
                         "report": str(destination), "seconds": time.perf_counter() - started})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            rows.append({"folder": folder, "sketch_id": sketch, "status": "error", "error": str(exc)})
        if (index + 1) % 10 == 0:
            print(f"Audited {index + 1}/{len(paths)} figures", flush=True)
    result = {
        "validator": geometry.VALIDATOR, "version": geometry.VERSION,
        "implementation_sha256": geometry.file_digest(Path(geometry.__file__)),
        "exporter_sha256": geometry.file_digest(Path(geometry.stage4_export.__file__)),
        "source": str(args.source.resolve()), "graph_source": str(graph_root.resolve()),
        "available_figures": available, "audited_figures": len(rows), "seed": args.seed,
        "scope": "Frozen post-hoc source-support audit, not CAD ground-truth accuracy or canonical acceptance.",
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "drawing_reason_counts": dict(reasons), "primitive_type_status_counts": dict(by_type), "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("audited_figures", "status_counts", "primitive_type_status_counts")}, indent=2))
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
