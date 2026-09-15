"""Paired fitting diagnostics with optional coverage recovery/integration; not canonical acceptance."""

import argparse
from copy import deepcopy
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import random
import shutil
import tempfile
import time

import cv2
import yaml

from tools.batch_run import stage0_handle_references as s0, stage2_stroke_extract as s2, stage3_primitive_fit as s3, stage4_export as s4
from tools import geometry_validation as geometry
from tools.evaluate_patent_fidelity import _primitive_mask, fidelity


def replay(job):
    source, graph_root, output, config_path, raster_metrics, recover_coverage = job[:6]
    integrate_coverage = bool(job[6]) if len(job) > 6 else False
    folder, sketch = source.parent.parent.name, source.stem.removesuffix("_primitives")
    target = output / folder
    started = time.perf_counter()
    row = {"folder": folder, "sketch_id": sketch}
    logging.getLogger().setLevel(logging.ERROR)
    try:
        config = yaml.safe_load(config_path.read_text())
        old = json.loads(source.read_text())
        graph_path = graph_root / folder / "graphs" / f"{sketch}_graph.json"
        skeleton_path = graph_root / folder / "cleaned" / f"{sketch}_skeleton.png"
        graph = json.loads(graph_path.read_text())
        source_graph_digest = geometry.file_digest(graph_path)
        skeleton = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise ValueError("Missing source skeleton")
        for path, subdir in ((graph_path, "graphs"), (skeleton_path, "cleaned")):
            (target / subdir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / subdir / path.name)
        before = geometry.validate(graph, old, skeleton > 0)
        fitting_graph_path = graph_path
        coverage = None
        integration = None
        edges = graph["edges"]
        if recover_coverage or integrate_coverage:
            if graph.get("stage2_scale", 1.0) != 1.0:
                raise ValueError("Frozen coverage recovery requires the native working skeleton")
            ledger = s2._CoverageLedger(skeleton)
            hatches = graph.get("removed_hachures", [])
            ledger.record("frozen_input_graph", graph["edges"], hatches)
            nodes = graph["nodes"]
            if recover_coverage:
                nodes, edges, recovery = s2.recover_source_coverage(skeleton, nodes, edges, hatches)
                if edges[:len(graph["edges"])] != graph["edges"]:
                    raise RuntimeError("Coverage recovery changed an existing edge")
                ledger.record("final_source_recovery", edges, hatches)
            else:
                recovery = deepcopy(graph["coverage"]["recovery"])
                ledger.stages = deepcopy(graph["coverage"]["stages"]) + ledger.stages
            if integrate_coverage:
                nodes, edges, integration = s2.integrate_recovered_connections(skeleton, nodes, edges, hatches)
                recovery["integration"] = integration
                ledger.record("recovered_connection_integration", edges, hatches)
            coverage = ledger.report(recovery)
            updated = dict(graph, nodes=nodes, edges=edges, coverage=coverage)
            # Frozen Stage 2 gates are not rerun by this diagnostic replay.
            updated["diagnostic_replay"] = {"canonical_acceptance": False,
                                            "inherited_metrics_not_recomputed": True}
            fitting_graph_path = target / "graphs" / graph_path.name
            fitting_graph_path.write_text(json.dumps(updated, indent=2, allow_nan=False)+"\n")
        result = s3.run(fitting_graph_path, target, sketch, config, stroke_width=old.get("stroke_width"))
        new = json.loads(result.primitives_path.read_text())
        after = geometry.run(fitting_graph_path, result.primitives_path, skeleton_path,
                             target / "geometry" / f"{sketch}_geometry_report.json")
        raster = {}
        if raster_metrics:
            with tempfile.TemporaryDirectory(prefix="fitting_fidelity_") as temp:
                for name, path in (("before", source), ("after", result.primitives_path)):
                    mask = _primitive_mask(path, Path(temp) / name, sketch, skeleton.shape)
                    raster[name] = fidelity(skeleton > 0, mask)
        export_path = result.primitives_path
        refs = graph_root / folder / "references" / f"{sketch}_references.json"
        if refs.exists():
            export_path = s0.attach_references_to_primitives(export_path, refs)
        export = s4.run(export_path, target, sketch, formats=("svg", "dxf"), dxf_mode="patent")
        if export.n_primitives_in != export.n_primitives_out:
            raise RuntimeError("Incomplete primitive export")
        if (source_graph_digest != geometry.file_digest(graph_path)
                or not (recover_coverage or integrate_coverage) and source_graph_digest != geometry.file_digest(target / "graphs" / graph_path.name)):
            raise RuntimeError("Source graph changed during replay")
        def curve_counts(report):
            return dict(Counter(p["status"] for p in report["primitives"] if p["type"] in {"circle", "ellipse"}))
        def encoding(doc):
            hatch = [p for p in doc["primitives"] if p.get("style") == "hachure"]
            return {"json_bytes": len(json.dumps(doc, separators=(",", ":")).encode()),
                    "hatch_json_bytes": len(json.dumps(hatch, separators=(",", ":")).encode()),
                    "budget_primitives": sum(s4.primitive_budget_cost(p) for p in doc["primitives"]),
                    "hatch_documents": len(hatch), "hatch_strokes": sum(s4.primitive_budget_cost(p) for p in hatch)}
        row.update(execution="ok", before=before["status"], after=after["status"],
                   before_summary=before["summary"], after_summary=after["summary"],
                   before_reasons=before["reason_codes"], after_reasons=after["reason_codes"],
                   before_curves=curve_counts(before), after_curves=curve_counts(after),
                   before_encoding=encoding(old), after_encoding=encoding(new), raster=raster,
                   source_graph_sha256=source_graph_digest,
                   source_primitives_sha256=geometry.file_digest(source),
                   ownership=after["checks"]["ownership"],
                   source_pixels_unchanged=not recover_coverage, source_graph_unchanged=True,
                   original_edges_unchanged=edges[:len(graph["edges"])] == graph["edges"],
                   coverage=coverage, integration=integration,
                   before_edge_stats=dict(zip(("median_length", "micro_ratio", "short_ratio"),
                                              s2._open_edge_length_stats(graph["edges"])[1:])),
                   after_edge_stats=dict(zip(("median_length", "micro_ratio", "short_ratio"),
                                             s2._open_edge_length_stats(edges)[1:])),
                   export_geometry_complete=True, seconds=time.perf_counter()-started)
    except Exception as exc:
        row.update(execution="error", error=f"{type(exc).__name__}: {exc}")
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{sketch}_replay.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--graphs", type=Path)
    parser.add_argument("--folder", action="append", help="Evaluate only named source folders (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=850725)
    parser.add_argument("--raster-metrics", action="store_true")
    parser.add_argument("--recover-coverage", action="store_true",
                        help="Recover omitted native-resolution source ink before Stage 3; retain original edges")
    parser.add_argument("--integrate-coverage", action="store_true",
                        help="Splice recovery into source-supported strokes, preserving the main pixel union")
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1 or args.limit is not None and args.limit < 1:
        parser.error("Use a new output path and positive worker/sample counts")
    paths = sorted(args.source.glob("*/primitives/*_primitives.json"))
    if args.folder:
        paths = [p for p in paths if p.parent.parent.name in args.folder]
    if not paths:
        parser.error("No baseline primitives")
    if args.limit and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    args.output.mkdir(parents=True)
    metadata = {"source": str(args.source.resolve()), "graphs": str((args.graphs or args.source).resolve()),
                "config_sha256": geometry.file_digest(args.config), "seed": args.seed,
                "scope": __doc__, "recover_coverage": args.recover_coverage,
                "integrate_coverage": args.integrate_coverage,
                "folders": args.folder,
                "implementation": {str(Path(m.__file__).resolve()): geometry.file_digest(Path(m.__file__))
                                   for m in (s2, s3, s4, geometry)}}
    for path in (Path(__file__).resolve(), Path(s4.__file__).with_name("export_audit.py")):
        metadata["implementation"][str(path)] = geometry.file_digest(path)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(p, args.graphs or args.source, args.output, args.config, args.raster_metrics,
                 args.recover_coverage, args.integrate_coverage) for p in paths]
        for future in as_completed([pool.submit(replay, job) for job in jobs]):
            rows.append(future.result())
            if len(rows) % 10 == 0 or len(rows) == len(paths):
                print(f"Replayed {len(rows)}/{len(paths)}; errors={sum(r['execution']=='error' for r in rows)}", flush=True)
    rows.sort(key=lambda r: (r["folder"], r["sketch_id"]))
    metadata.update(rows=rows, execution_counts=dict(Counter(r["execution"] for r in rows)),
                    before_counts=dict(Counter(r["before"] for r in rows if "before" in r)),
                    after_counts=dict(Counter(r["after"] for r in rows if "after" in r)))
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: metadata[k] for k in ("execution_counts", "before_counts", "after_counts")}, indent=2))
    return int(any(r["execution"] == "error" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
