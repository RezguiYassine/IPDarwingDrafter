from __future__ import annotations

import argparse
import copy
import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from syntheticData.patentvec.schema import CanonicalDrawing
from syntheticData.patentvec.training import derive_puhachov_keypoints
from tools import d2c_stage3_dataset as d2c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate synthetic polyline targets against real Stage-2 edges."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--match-p90-px", type=float, default=4.0)
    parser.add_argument("--line-p90-px", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_member(archive: tarfile.TarFile, name: str) -> bytes:
    member = archive.getmember(name)
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"cannot read archive member {name}")
    return stream.read()


def _symmetric_p90(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or len(second) < 2:
        return math.inf
    first_to_second = cKDTree(second).query(first, k=1)[0]
    second_to_first = cKDTree(first).query(second, k=1)[0]
    return max(
        float(np.quantile(first_to_second, 0.90)),
        float(np.quantile(second_to_first, 0.90)),
    )


def _stage2_edges(skeleton: np.ndarray, keypoints: np.ndarray) -> list[dict]:
    candidates = [
        {
            "x": int(x),
            "y": int(y),
            "type": d2c.ID_TO_KP[int(kind)],
            "confidence": 1.0,
        }
        for x, y, kind in keypoints
        if int(kind) in d2c.ID_TO_KP
    ]
    clusters = d2c.s2._clusters_from_points(
        candidates, skeleton, snap_radius=4
    )
    nodes, edges = d2c.s2._extract_topology(skeleton, clusters)
    nodes, edges = d2c.s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=6.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )
    edges = d2c.s2._smooth_edges(edges)
    output = []
    for edge in edges:
        raw = d2c._edge_points(edge)
        if len(raw) < 2:
            continue
        model_input = raw if edge.get("is_closed") else np.asarray(
            edge.get("smooth_pts") or edge["pixels"], dtype=np.float64
        )
        output.append({"input": model_input, "raw": raw})
    return output


def _target_polylines(free2cad, canvas: list[int]) -> tuple[list[np.ndarray], list[int]]:
    output = []
    origins = []
    width, height = canvas
    pixel_scale = np.asarray([width - 1, height - 1], dtype=float)
    for index in np.flatnonzero(free2cad["types"] == 3):
        mask = np.asarray(free2cad["mask"][index], dtype=bool)
        normalized = np.asarray(free2cad["points"][index][mask], dtype=float)
        if "centers_px" in free2cad.files and "scales_px" in free2cad.files:
            center = np.asarray(free2cad["centers_px"][index], dtype=float)
            scale = float(free2cad["scales_px"][index])
            output.append((normalized - 0.5) * scale + center)
        else:
            center = np.asarray(free2cad["centers"][index], dtype=float)
            scale = float(free2cad["scales"][index])
            output.append(((normalized - 0.5) * scale + center) * pixel_scale)
        origins.append(int(free2cad["edge_origins"][index]))
    return output, origins


def _validate_sample(
    row: dict,
    archive: tarfile.TarFile,
    match_threshold: float,
    line_threshold: float,
) -> dict:
    prefix = row["member_prefix"]
    drawing = CanonicalDrawing.from_dict(
        json.loads(_read_member(archive, f"{prefix}/sample.json"))
    )
    object_drawing = copy.deepcopy(drawing)
    object_drawing.primitives_visible = [
        primitive
        for primitive in object_drawing.primitives_visible
        if primitive.semantic == "object_visible"
    ]
    object_ids = {
        primitive.primitive_id for primitive in object_drawing.primitives_visible
    }
    object_drawing.junctions_visible = [
        junction
        for junction in object_drawing.junctions_visible
        if set(junction.primitive_ids) <= object_ids
    ]
    with np.load(
        io.BytesIO(_read_member(archive, f"{prefix}/masks.npz")),
        allow_pickle=False,
    ) as masks:
        skeleton = skeletonize(masks["object"] > 0).astype(np.uint8) * 255
    keypoints = derive_puhachov_keypoints(object_drawing)
    extracted = _stage2_edges(skeleton, keypoints)
    with np.load(
        io.BytesIO(_read_member(archive, f"{prefix}/free2cad_edges.npz")),
        allow_pickle=False,
    ) as free2cad:
        targets, origins = _target_polylines(free2cad, drawing.canvas)

    if not targets or not extracted:
        return {
            "sample_id": drawing.sample_id,
            "accepted": False,
            "target_count": len(targets),
            "extracted_edge_count": len(extracted),
            "failures": ["missing polyline targets or extracted Stage-2 edges"],
        }
    costs = np.asarray(
        [
            [_symmetric_p90(target, edge["input"]) for edge in extracted]
            for target in targets
        ],
        dtype=float,
    )
    target_indices, edge_indices = linear_sum_assignment(costs)
    assignments = []
    failures = []
    assigned_targets = set(map(int, target_indices))
    for target_index, edge_index in zip(target_indices, edge_indices):
        target_index = int(target_index)
        edge_index = int(edge_index)
        match_p90 = float(costs[target_index, edge_index])
        line_p90 = float(d2c._line_residual(extracted[edge_index]["raw"]))
        accepted = match_p90 <= match_threshold and line_p90 > line_threshold
        assignments.append(
            {
                "target_index": target_index,
                "edge_index": edge_index,
                "origin": "aggregated_lines" if origins[target_index] else "direct_polyline",
                "match_p90_px": match_p90,
                "line_p90_px": line_p90,
                "accepted": accepted,
            }
        )
        if not accepted:
            failures.append(
                f"target {target_index}: match p90 {match_p90:.3f}, "
                f"line p90 {line_p90:.3f}"
            )
    for target_index in sorted(set(range(len(targets))) - assigned_targets):
        failures.append(f"target {target_index}: no unique Stage-2 edge")
    return {
        "sample_id": drawing.sample_id,
        "accepted": not failures,
        "target_count": len(targets),
        "extracted_edge_count": len(extracted),
        "assignments": assignments,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.dataset / "manifest.json"
    rows_path = args.dataset / "manifest.jsonl"
    if not manifest_path.exists() or not rows_path.exists():
        raise SystemExit(f"not a compact PatentVec dataset: {args.dataset}")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if args.sample_limit > 0 and len(rows) > args.sample_limit:
        indices = np.linspace(0, len(rows) - 1, args.sample_limit).round().astype(int)
        rows = [rows[int(index)] for index in indices]
    archives: dict[str, tarfile.TarFile] = {}
    reports = []
    try:
        for position, row in enumerate(rows, start=1):
            archive_name = row["archive"]
            archive = archives.get(archive_name)
            if archive is None:
                archive = tarfile.open(args.dataset / archive_name, mode="r")
                archives[archive_name] = archive
            report = _validate_sample(
                row,
                archive,
                match_threshold=args.match_p90_px,
                line_threshold=args.line_p90_px,
            )
            reports.append(report)
            print(
                f"[{position}/{len(rows)}] {row['sample_id']} "
                f"polylines={report['target_count']} accepted={report['accepted']}",
                flush=True,
            )
    finally:
        for archive in archives.values():
            archive.close()

    assignments = [
        assignment
        for report in reports
        for assignment in report.get("assignments", [])
    ]
    accepted_assignments = [item for item in assignments if item["accepted"]]
    target_count = sum(report["target_count"] for report in reports)
    payload = {
        "schema_version": "patentvec-polyline-stage2-audit-1.0",
        "dataset": str(args.dataset),
        "sample_count": len(reports),
        "accepted_samples": sum(report["accepted"] for report in reports),
        "target_count": target_count,
        "accepted_target_count": len(accepted_assignments),
        "target_acceptance": len(accepted_assignments) / max(target_count, 1),
        "match_p90_px": {
            "mean": float(np.mean([item["match_p90_px"] for item in assignments])),
            "p95": float(np.quantile([item["match_p90_px"] for item in assignments], 0.95)),
            "max": float(np.max([item["match_p90_px"] for item in assignments])),
        },
        "thresholds": {
            "match_p90_px": args.match_p90_px,
            "line_p90_px": args.line_p90_px,
        },
        "reports": reports,
    }
    output = args.output or args.dataset / "polyline_stage2_report.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({key: value for key, value in payload.items() if key != "reports"}, indent=2))
    print(f"report: {output}")
    return 0 if payload["target_acceptance"] >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
