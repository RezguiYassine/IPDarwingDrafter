"""Render gated graph/primitive outputs for visual review, never for acceptance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import logging
from pathlib import Path
import shutil
import time

import yaml

from tools.batch_run import stage0_handle_references as s0, stage3_primitive_fit as s3, stage4_export as s4
from tools.geometry_validation import file_digest
from tools.make_patent_comparison_sheet import _load


def render_one(job, *, max_strokes=25_000, max_trace_points=50_000):
    if max_strokes < 1 or max_trace_points < 1:
        raise ValueError("Diagnostic stroke budgets must be positive")
    root, config_path, row = job
    logging.getLogger().setLevel(logging.ERROR)
    patent, sketch = row["patent_id"], row["sketch_id"]
    source = root / patent
    output = root / "review_diagnostics" / patent
    report_path = output / (sketch+"_review.json")
    graph = source / "graphs" / (sketch+"_graph.json")
    primitive = source / "primitives" / (sketch+"_primitives.json")
    report = {"patent_id": patent, "sketch_id": sketch,
              "diagnostic_only": True, "training_eligible": False,
              "canonical_status": row["status"], "canonical_error": row.get("error"),
              "config_sha256": file_digest(config_path),
              "budget": {"max_strokes": max_strokes, "max_point_occurrences": 2_000_000,
                         "max_trace_points": max_trace_points}}
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if not graph.exists():
            report.update(status="unavailable", reason="No completed Stage 2 graph")
        else:
            digest = file_digest(graph)
            report["source_graph_sha256"] = digest
            data = json.loads(graph.read_text())
            strokes = data.get("edges", []) + data.get("removed_hachures", [])
            count = sum(len(e.get("pixels", [])) for e in strokes)
            longest = max((len(e.get("pixels", [])) for e in strokes), default=0)
            # Keep review-only work bounded; a missing preview never becomes a pass.
            if len(strokes) > max_strokes or count > 2_000_000:
                report.update(status="unavailable", reason="Diagnostic budget exceeded",
                              source_strokes=len(strokes), source_point_occurrences=count)
            elif longest > max_trace_points and not primitive.exists():
                report.update(status="unavailable", reason="Diagnostic trace budget exceeded",
                              source_strokes=len(strokes), source_point_occurrences=count,
                              longest_trace_points=longest)
            else:
                config = yaml.safe_load(config_path.read_text())
                if primitive.exists():
                    target = output / "primitives" / primitive.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(primitive, target)
                    report["source_primitives_sha256"] = file_digest(primitive)
                else:
                    target = s3.run(graph, output, sketch, config).primitives_path
                refs = source / "references" / (sketch+"_references.json")
                if refs.exists():
                    target = s0.attach_references_to_primitives(target, refs)
                result = s4.run(target, output, sketch, formats=("svg",), dxf_mode="patent")
                if result.n_primitives_in != result.n_primitives_out:
                    raise RuntimeError("Diagnostic export did not retain every primitive")
                if file_digest(graph) != digest:
                    raise RuntimeError("Source graph changed during diagnostic export")
                report.update(status="rendered", svg=str(result.svg_path.resolve()),
                              svg_sha256=file_digest(result.svg_path),
                              export_flagged=result.flagged,
                              primitives=result.n_primitives_out)
    except Exception as exc:
        report.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    report["seconds"] = time.monotonic()-started
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(report_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--watch-count", type=int, default=0)
    parser.add_argument("--max-strokes", type=int, default=25_000)
    parser.add_argument("--max-trace-points", type=int, default=50_000)
    parser.add_argument("--retry-budget-exceeded", action="store_true",
                        help="Retry cached budget-limited diagnostics within the new stroke budget")
    args = parser.parse_args()
    if args.workers < 1 or args.watch_count < 0 or args.max_strokes < 1 or args.max_trace_points < 1:
        parser.error("Worker/count limits must be positive/nonnegative")
    root = args.current.resolve()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            rows = _load(root)
            jobs = []
            for (patent, sketch), row in sorted(rows.items()):
                if (root / patent / "vectors" / (sketch+".svg")).exists():
                    continue
                report = root / "review_diagnostics" / patent / (sketch+"_review.json")
                if report.exists():
                    cached = json.loads(report.read_text())
                    retry = (args.retry_budget_exceeded
                             and cached.get("status") == "unavailable"
                             and cached.get("reason") == "Diagnostic budget exceeded"
                             and cached.get("source_strokes", float("inf")) <= args.max_strokes
                             and cached.get("source_point_occurrences", float("inf")) <= 2_000_000)
                    if not retry:
                        continue
                jobs.append((root, args.config.resolve(), row))
            for result in pool.map(partial(render_one, max_strokes=args.max_strokes,
                                           max_trace_points=args.max_trace_points), jobs):
                completed += 1
                print(f"Diagnostic {result['patent_id']}/{result['sketch_id']}: {result['status']}", flush=True)
            if not args.watch_count or len(rows) >= args.watch_count:
                break
            print(f"Canonical rows {len(rows)}/{args.watch_count}; new diagnostic reports {completed}", flush=True)
            time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
