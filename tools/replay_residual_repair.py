"""Paired frozen residual/curve repair; never changes historical graph artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import logging
from pathlib import Path
import random
import shutil
import tempfile
import time

import cv2
import yaml

from tools.batch_run import stage0_handle_references as s0, stage2_stroke_extract as s2
from tools.batch_run import stage3_primitive_fit as s3, stage4_export as s4
from tools import geometry_validation as geometry
from tools.evaluate_patent_fidelity import _primitive_mask, fidelity


def repair_graph(graph, config):
    """Repair residuals in isolation; other edges and the hatch layer stay exact."""
    candidate = deepcopy(graph)
    nodes, edges = s2.repair_noncycle_residuals(candidate["nodes"], candidate["edges"])
    recovered = [edge for edge in edges if edge.get("residual_parent_edge_ids")]
    unchanged = [edge for edge in edges if not edge.get("residual_parent_edge_ids")]
    if recovered:
        ids = {edge[key] for edge in recovered for key in ("source", "target")}
        cfg = config.get("stage2", {})
        local_nodes, recovered = s2._simplify_graph(
            [node for node in nodes if node["id"] in ids], recovered,
            spur_min_len=0, junction_merge_radius=0,
            collinear_max_angle=cfg.get("merge_collinear_max_angle", 28.0))
        recovered = s2._smooth_edges(
            recovered, cfg.get("rdp_epsilon", 1.5), cfg.get("spline_smoothing", 2.0),
            spline_overshoot_limit=cfg.get("spline_overshoot_limit", 5.0))
        next_id = max((edge["id"] for edge in graph["edges"]), default=-1) + 1
        for index, edge in enumerate(recovered):
            edge["id"] = next_id + index
        nodes = [node for node in nodes if node["id"] not in ids] + local_nodes
    candidate["nodes"], candidate["edges"] = nodes, unchanged + recovered
    old_pixels = {tuple(p) for edge in graph["edges"] for p in edge["pixels"]}
    new_pixels = {tuple(p) for edge in candidate["edges"] for p in edge["pixels"]}
    if old_pixels != new_pixels:
        raise RuntimeError("Residual repair changed the source pixel set")
    candidate["residual_repair"] = {"source_pixels": len(old_pixels), "preserved_pixels": len(new_pixels),
                                     "recovered_edges": len(recovered),
                                     "unrepaired_noncycles": sum(e.get("topology_origin") == "unclaimed_component"
                                          and e.get("is_simple_cycle") is False for e in candidate["edges"])}
    return candidate


def replay(job):
    path, output, config_path, raster_metrics = job
    logging.getLogger().setLevel(logging.ERROR)
    started = time.perf_counter()
    folder, sketch = path.parent.parent.name, path.name.removesuffix("_primitives.json")
    target = output / folder
    try:
        config = yaml.safe_load(config_path.read_text())
        original = json.loads(path.read_text())
        source_graph = path.parent.parent / "graphs" / f"{sketch}_graph.json"
        graph = json.loads(source_graph.read_text())
        candidate_graph = repair_graph(graph, config)
        graph_path = target / "graphs" / source_graph.name
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(candidate_graph, indent=2) + "\n")
        skeleton_path = path.parent.parent / "cleaned" / f"{sketch}_skeleton.png"
        skeleton = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise ValueError(f"Cannot read {skeleton_path}")
        baseline = geometry.validate(graph, original, skeleton > 0)
        result = s3.run(graph_path, target, sketch, config, stroke_width=original.get("stroke_width"))
        primitive_doc = json.loads(result.primitives_path.read_text())
        old_hatches = [p for p in original["primitives"] if p.get("style") == "hachure"]
        new_hatches = [p for p in primitive_doc["primitives"] if p.get("style") == "hachure"]
        if old_hatches != new_hatches:
            raise RuntimeError("Frozen hatch primitives changed")
        report_path = target / "geometry" / f"{sketch}_geometry_report.json"
        candidate = geometry.run(graph_path, result.primitives_path, skeleton_path, report_path)
        target.joinpath("cleaned").mkdir(exist_ok=True)
        shutil.copy2(skeleton_path, target / "cleaned" / skeleton_path.name)
        raster = {}
        if raster_metrics:
            with tempfile.TemporaryDirectory(prefix="residual_fidelity_") as temp:
                for name, primitive_path in (("before", path), ("after", result.primitives_path)):
                    mask = _primitive_mask(primitive_path, Path(temp)/name, sketch, skeleton.shape)
                    raster[name] = fidelity(skeleton > 0, mask)
        export_path = result.primitives_path
        refs = path.parent.parent / "references" / f"{sketch}_references.json"
        if refs.exists():
            export_path = s0.attach_references_to_primitives(export_path, refs)
        export = s4.run(export_path, target, sketch, formats=("svg", "dxf"), dxf_mode="patent")
        if export.n_primitives_out != export.n_primitives_in:
            raise RuntimeError("Incomplete primitive export")
        def curve_counts(report):
            return dict(Counter(p["status"] for p in report["primitives"] if p["type"] in {"circle", "ellipse"}))
        return {"folder": folder, "sketch_id": sketch, "execution": "ok",
                "source_graph_sha256": geometry.file_digest(source_graph),
                "source_primitives_sha256": geometry.file_digest(path),
                "source_pixels_preserved": True, "hatch_primitives_unchanged": True,
                "repair": candidate_graph["residual_repair"], "before": baseline["status"], "after": candidate["status"],
                "before_reasons": baseline["reason_codes"], "after_reasons": candidate["reason_codes"],
                "before_curves": curve_counts(baseline), "after_curves": curve_counts(candidate),
                "before_primitives": baseline["summary"], "after_primitives": candidate["summary"],
                "raster": raster, "report": str(report_path), "seconds": time.perf_counter()-started}
    except Exception as exc:
        return {"folder": folder, "sketch_id": sketch, "execution": "error", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=850725)
    parser.add_argument("--raster-metrics", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("Use a new output directory and positive workers/limit")
    paths = sorted(args.source.glob("*/primitives/*_primitives.json"))
    if not paths:
        parser.error("No source primitive documents")
    if args.limit and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    args.output.mkdir(parents=True)
    implementation = {str(Path(module.__file__).resolve()): geometry.file_digest(Path(module.__file__))
                      for module in (s2, s3, s4, geometry)}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(path, args.output, args.config, args.raster_metrics) for path in paths]
        futures = [pool.submit(replay, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 10 == 0 or len(rows) == len(paths):
                print(f"Replayed {len(rows)}/{len(paths)}; errors={sum(r['execution']=='error' for r in rows)}", flush=True)
    rows.sort(key=lambda r: (r["folder"], r["sketch_id"]))
    result = {"source": str(args.source.resolve()), "config": str(args.config.resolve()),
              "config_sha256": geometry.file_digest(args.config), "implementation": implementation,
              "seed": args.seed, "scope": "Frozen residual repair and Stage 3/4 replay, not a new canonical batch or acceptance claim.",
              "execution_counts": dict(Counter(r["execution"] for r in rows)),
              "before_counts": dict(Counter(r["before"] for r in rows if "before" in r)),
              "after_counts": dict(Counter(r["after"] for r in rows if "after" in r)), "rows": rows}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:result[key] for key in ("execution_counts", "before_counts", "after_counts")}, indent=2))
    return int(any(r["execution"] == "error" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
