from __future__ import annotations

import copy
import json
import math
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .geometry import primitive_endpoints, primitive_tangent, resample_polyline, sample_primitive
from tools import d2c_stage3_dataset as d2c

from .render import render_clean, render_masks
from .schema import CanonicalDrawing, Primitive


FREE2CAD_TYPES = {
    "line": 0,
    "arc": 1,
    "circle": 2,
    "polyline": 3,
    "spline": 3,
    "quadratic_bezier": 4,
    "cubic_bezier": 4,
}

PUHACHOV_TARGET_SEMANTICS = {
    "object_visible",
    "object_hidden",
    "object_center",
    "object_section_boundary",
}


def derive_puhachov_keypoints(
    drawing: CanonicalDrawing,
    corner_angle_degrees: float = 35.0,
    merge_tolerance: float = 1e-5,
) -> np.ndarray:
    records: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    coordinates: dict[tuple[int, int], np.ndarray] = {}
    for primitive in drawing.primitives_visible:
        if primitive.semantic not in PUHACHOV_TARGET_SEMANTICS:
            continue
        for endpoint_index, endpoint in enumerate(primitive_endpoints(primitive)):
            direction = primitive_tangent(primitive, endpoint_index)
            key = tuple(np.round(endpoint / merge_tolerance).astype(int))
            records.setdefault(key, []).append((endpoint, direction))
            coordinates[key] = endpoint

    keypoints: list[tuple[int, int, int]] = []
    cos_threshold = math.cos(math.radians(180.0 - corner_angle_degrees))
    for key, incident in records.items():
        point = coordinates[key]
        degree = len(incident)
        kind = None
        if degree == 1:
            kind = 0
        elif degree >= 3:
            kind = 1
        elif degree == 2:
            dot = float(np.dot(incident[0][1], incident[1][1]))
            if dot > cos_threshold:
                kind = 2
        if kind is not None:
            x = int(np.clip(round(point[0] * (drawing.canvas[0] - 1)), 0, drawing.canvas[0] - 1))
            y = int(np.clip(round(point[1] * (drawing.canvas[1] - 1)), 0, drawing.canvas[1] - 1))
            keypoints.append((x, y, kind))

    # Circular tangent contacts have no explicit curve endpoint, so retain the
    # exact topology annotation as a junction keypoint.
    for junction in drawing.junctions_visible:
        if junction.type not in {
            "t_junction",
            "x_junction",
            "y_junction",
            "tangent_contact",
            "line_arc_transition",
            "arc_arc_transition",
        }:
            continue
        x = int(np.clip(round(junction.position[0] * (drawing.canvas[0] - 1)), 0, drawing.canvas[0] - 1))
        y = int(np.clip(round(junction.position[1] * (drawing.canvas[1] - 1)), 0, drawing.canvas[1] - 1))
        keypoints.append((x, y, 1))
    unique = sorted(set(keypoints))
    return np.asarray(unique, dtype=np.int32).reshape(-1, 3)


def puhachov_arrays(
    drawing: CanonicalDrawing,
    clean: np.ndarray | None = None,
) -> dict[str, np.ndarray | str]:
    clean = render_clean(drawing) if clean is None else clean
    foreground = clean < 210
    skeleton = skeletonize(foreground).astype(np.uint8) * 255
    exact_keypoints = derive_puhachov_keypoints(drawing)
    ys, xs = np.where(skeleton > 0)
    snapped = []
    dropped = 0
    if len(xs):
        tree = cKDTree(np.column_stack([xs, ys]))
        snap_radius = max(8.0, drawing.canvas[0] / 64.0)
        for x, y, kind in exact_keypoints:
            distance, index = tree.query(
                [x, y], distance_upper_bound=snap_radius + 1e-6
            )
            if not np.isfinite(distance):
                dropped += 1
                continue
            snapped.append((int(xs[index]), int(ys[index]), int(kind)))
    else:
        dropped = len(exact_keypoints)
    keypoints = np.asarray(sorted(set(snapped)), dtype=np.int32).reshape(-1, 3)
    metadata = {
        "sample_id": drawing.sample_id,
        "split": drawing.split,
        "render_res": drawing.canvas[0],
        "target_semantics": sorted(PUHACHOV_TARGET_SEMANTICS),
        "distractor_semantics_in_skeleton": sorted(
            set(drawing.semantic_layers) - PUHACHOV_TARGET_SEMANTICS
        ),
        "channel_order": ["endpoint", "junction", "corner"],
        "exact_keypoint_count": int(len(exact_keypoints)),
        "snapped_keypoint_count": int(len(keypoints)),
        "dropped_keypoint_count": int(dropped),
    }
    return {
        "skeleton": skeleton,
        "kps": keypoints,
        "meta": json.dumps(metadata, sort_keys=True),
    }


def _normalise_edge(
    primitive: Primitive, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = sample_primitive(primitive, max(128, max_points * 2))
    if primitive.kind == "circle":
        points = np.vstack([points, points[0]])
    points = resample_polyline(points, max_points)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    scale = float(np.max(upper - lower))
    if scale <= 1e-9:
        raise ValueError(f"degenerate Stage-3 primitive {primitive.primitive_id}")
    normalized = ((points - center) / scale + 0.5).astype(np.float32)
    mask = np.ones(max_points, dtype=bool)
    return normalized, mask, center, scale


def _point_key(point: np.ndarray, tolerance: float = 1e-6) -> tuple[int, int]:
    return tuple(np.round(np.asarray(point, dtype=float) / tolerance).astype(int))


def _pixel_line_p90(primitive: Primitive, canvas: list[int]) -> float:
    points = sample_primitive(primitive, 128) * np.asarray(
        [canvas[0] - 1, canvas[1] - 1], dtype=float
    )
    centered = points - points.mean(axis=0)
    if len(points) < 3 or not np.any(centered):
        return 0.0
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = np.array([-vt[0, 1], vt[0, 0]])
    return float(np.quantile(np.abs(centered @ normal), 0.90))


def _line_chain_polyline_targets(
    drawing: CanonicalDrawing,
    minimum_turn_degrees: float = 4.0,
    maximum_turn_degrees: float = 28.0,
    maximum_segments: int = 6,
    minimum_relative_deviation: float = 0.01,
) -> tuple[list[Primitive], set[str], dict[str, list[str]]]:
    """Aggregate non-branching shallow line chains into Stage-3 polylines."""
    structural_contacts = {
        _point_key(np.asarray(junction.position, dtype=float))
        for junction in drawing.junctions_visible
        if junction.interaction_id is not None
    }
    by_component: dict[str, list[Primitive]] = defaultdict(list)
    for primitive in drawing.primitives_visible:
        if primitive.semantic == "object_visible" and primitive.kind == "line":
            by_component[primitive.component_id].append(primitive)

    aggregates: list[Primitive] = []
    consumed: set[str] = set()
    members: dict[str, list[str]] = {}
    serial = 0
    for component_id in sorted(by_component):
        lines = sorted(by_component[component_id], key=lambda item: item.primitive_id)
        by_id = {line.primitive_id: line for line in lines}
        incident: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = defaultdict(list)
        contact_position: dict[tuple[int, int], np.ndarray] = {}
        for line in lines:
            p0 = np.asarray(line.geometry["p0"], dtype=float)
            p1 = np.asarray(line.geometry["p1"], dtype=float)
            for point, other in ((p0, p1), (p1, p0)):
                key = _point_key(point)
                incident[key].append((line.primitive_id, other - point))
                contact_position[key] = point

        adjacency: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for key in sorted(incident):
            records = incident[key]
            if len(records) != 2 or key in structural_contacts:
                continue
            (first_id, first_vector), (second_id, second_vector) = records
            first_norm = float(np.linalg.norm(first_vector))
            second_norm = float(np.linalg.norm(second_vector))
            if first_norm <= 1e-12 or second_norm <= 1e-12:
                continue
            cosine = float(
                np.clip(
                    np.dot(first_vector / first_norm, second_vector / second_norm),
                    -1.0,
                    1.0,
                )
            )
            opening = math.degrees(math.acos(cosine))
            turn = 180.0 - opening
            if not minimum_turn_degrees <= turn <= maximum_turn_degrees:
                continue
            contact = contact_position[key]
            adjacency[first_id][second_id] = contact
            adjacency[second_id][first_id] = contact

        remaining = set(adjacency)
        ordered_chains: list[list[str]] = []
        while remaining:
            root = min(remaining)
            component = set()
            stack = [root]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(sorted(set(adjacency[current]) - component, reverse=True))
            remaining -= component
            starts = sorted(
                line_id
                for line_id in component
                if len(set(adjacency[line_id]) & component) == 1
            )
            if not starts:
                continue
            ordered = []
            previous = None
            current = starts[0]
            while current is not None:
                ordered.append(current)
                choices = sorted(
                    (set(adjacency[current]) & component) - set(ordered)
                )
                previous, current = current, choices[0] if choices else None
            if len(ordered) >= 2:
                ordered_chains.append(ordered)

        for ordered in ordered_chains:
            offset = 0
            while len(ordered) - offset >= 2:
                remaining_count = len(ordered) - offset
                chunk_size = min(maximum_segments, remaining_count)
                if remaining_count - chunk_size == 1:
                    chunk_size -= 1
                chunk = ordered[offset : offset + chunk_size]
                offset += chunk_size

                contacts = [
                    adjacency[first_id][second_id]
                    for first_id, second_id in zip(chunk[:-1], chunk[1:])
                ]
                first = by_id[chunk[0]]
                first_endpoints = [
                    np.asarray(first.geometry["p0"], dtype=float),
                    np.asarray(first.geometry["p1"], dtype=float),
                ]
                start = max(
                    first_endpoints,
                    key=lambda point: np.linalg.norm(point - contacts[0]),
                )
                last = by_id[chunk[-1]]
                last_endpoints = [
                    np.asarray(last.geometry["p0"], dtype=float),
                    np.asarray(last.geometry["p1"], dtype=float),
                ]
                end = max(
                    last_endpoints,
                    key=lambda point: np.linalg.norm(point - contacts[-1]),
                )
                points = np.asarray([start, *contacts, end], dtype=float)
                chord = end - start
                chord_length = float(np.linalg.norm(chord))
                if chord_length <= 1e-12:
                    continue
                offsets = points - start
                cross_magnitudes = np.abs(
                    chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]
                )
                relative_deviation = float(
                    np.max(cross_magnitudes) / (chord_length * chord_length)
                )
                if relative_deviation < minimum_relative_deviation:
                    continue

                source = first
                primitive_id = f"{drawing.sample_id}_polyagg_{serial:04d}"
                serial += 1
                member_ids = list(chunk)
                aggregate = Primitive(
                    primitive_id=primitive_id,
                    kind="polyline",
                    geometry={"points": points.tolist(), "closed": False},
                    semantic="object_visible",
                    component_id=component_id,
                    source_dataset=source.source_dataset,
                    source_sample_id=source.source_sample_id,
                    source_component_id=source.source_component_id,
                    source_primitive_id="|".join(
                        by_id[line_id].source_primitive_id for line_id in chunk
                    ),
                    generated_by="free2cad_shallow_chain_aggregation",
                    transform_id=source.transform_id,
                    parent_primitive_ids=member_ids,
                    interaction_ids=sorted(
                        {
                            interaction_id
                            for line_id in chunk
                            for interaction_id in by_id[line_id].interaction_ids
                        }
                    ),
                    style={"training_projection": True},
                )
                if _pixel_line_p90(aggregate, drawing.canvas) <= 1.5:
                    continue
                aggregates.append(aggregate)
                members[primitive_id] = member_ids
                consumed.update(member_ids)
    return aggregates, consumed, members


def _primitive_free2cad_arrays(
    drawing: CanonicalDrawing,
    source_index: int,
    max_points: int = 64,
) -> dict[str, np.ndarray]:
    aggregates, consumed_line_ids, aggregate_members = _line_chain_polyline_targets(
        drawing
    )
    candidates = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.primitive_id not in consumed_line_ids
    ]
    candidates.extend(aggregates)
    rows = []
    for primitive in candidates:
        if primitive.semantic != "object_visible" or primitive.kind not in FREE2CAD_TYPES:
            continue
        try:
            points, mask, center, scale = _normalise_edge(primitive, max_points)
        except ValueError:
            continue
        target_kind = primitive.kind
        if primitive.kind == "polyline" and _pixel_line_p90(
            primitive, drawing.canvas
        ) <= 1.5:
            target_kind = "line"
        params = np.zeros(6, dtype=np.float32)
        if target_kind == "line":
            endpoints = primitive_endpoints(primitive)
            p0 = np.asarray(endpoints[0], dtype=float)
            p1 = np.asarray(endpoints[-1], dtype=float)
            params[:2] = (p0 - center) / scale + 0.5
            params[2:4] = (p1 - center) / scale + 0.5
        elif primitive.kind in {"arc", "circle"}:
            circle_center = np.asarray(primitive.geometry["center"], dtype=float)
            params[:2] = (circle_center - center) / scale + 0.5
            params[2] = float(primitive.geometry["radius"]) / scale
            if primitive.kind == "arc":
                params[3] = (
                    math.degrees(float(primitive.geometry["start_angle"])) % 360.0
                ) / 360.0
                params[4] = (
                    math.degrees(float(primitive.geometry["end_angle"])) % 360.0
                ) / 360.0
        rows.append(
            (
                primitive,
                target_kind,
                points,
                mask,
                params,
                aggregate_members.get(primitive.primitive_id, [primitive.primitive_id]),
                center,
                scale,
            )
        )
    if not rows:
        raise ValueError(f"drawing {drawing.sample_id} has no supported Stage-3 edges")
    return {
        "points": np.stack([row[2] for row in rows]),
        "mask": np.stack([row[3] for row in rows]),
        "types": np.asarray([FREE2CAD_TYPES[row[1]] for row in rows], dtype=np.uint8),
        "params": np.stack([row[4] for row in rows]),
        "centers": np.stack([row[6] for row in rows]).astype(np.float32),
        "scales": np.asarray([row[7] for row in rows], dtype=np.float32),
        "source_index": np.full(len(rows), int(source_index), dtype=np.int64),
        "primitive_ids": np.asarray([row[0].primitive_id for row in rows]),
        "member_primitive_ids": np.asarray(["|".join(row[5]) for row in rows]),
        "edge_origins": np.asarray(
            [
                1
                if row[0].generated_by == "free2cad_shallow_chain_aggregation"
                else 0
                for row in rows
            ],
            dtype=np.uint8,
        ),
    }


def _free2cad_candidates(
    drawing: CanonicalDrawing,
) -> tuple[list[Primitive], dict[str, list[str]]]:
    aggregates, consumed_line_ids, aggregate_members = _line_chain_polyline_targets(
        drawing
    )
    candidates = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.primitive_id not in consumed_line_ids
        and primitive.semantic == "object_visible"
        and primitive.kind in FREE2CAD_TYPES
    ]
    candidates.extend(aggregates)
    return candidates, aggregate_members


def _dense_curve_pixels(primitive: Primitive, pixel_scale: np.ndarray) -> np.ndarray:
    pilot = sample_primitive(primitive, 128) * pixel_scale
    length = float(np.linalg.norm(np.diff(pilot, axis=0), axis=1).sum())
    count = int(np.clip(math.ceil(length / 1.25), 12, 384))
    return sample_primitive(primitive, count + 1) * pixel_scale


def _canonical_source_index(
    drawing: CanonicalDrawing,
) -> tuple[d2c.SourceIndex, dict[int, dict]]:
    """Represent canonical object geometry like Drawing2CAD SVG commands."""
    candidates, aggregate_members = _free2cad_candidates(drawing)
    pixel_scale = np.asarray(
        [drawing.canvas[0] - 1, drawing.canvas[1] - 1], dtype=float
    )
    segments: list[d2c.SourceSegment] = []
    path_info: dict[int, dict] = {}
    for path_id, primitive in enumerate(candidates):
        member_ids = aggregate_members.get(
            primitive.primitive_id, [primitive.primitive_id]
        )
        path_info[path_id] = {
            "member_ids": member_ids,
            "origin": (
                1
                if primitive.generated_by == "free2cad_shallow_chain_aggregation"
                else 0
            ),
        }
        if primitive.kind == "line":
            points = np.asarray(
                [primitive.geometry["p0"], primitive.geometry["p1"]], dtype=float
            )
            length = float(np.linalg.norm((points[1] - points[0]) * pixel_scale))
            count = int(np.clip(math.ceil(length), 2, 2048))
            segments.append(
                d2c.SourceSegment(
                    path_id, 0, "L", np.linspace(points[0], points[1], count + 1) * pixel_scale
                )
            )
            continue
        if primitive.kind == "polyline":
            points = np.asarray(primitive.geometry["points"], dtype=float)
            if primitive.geometry.get("closed", False) and not np.allclose(
                points[0], points[-1]
            ):
                points = np.vstack([points, points[0]])
            for order, (first, second) in enumerate(zip(points[:-1], points[1:])):
                length = float(np.linalg.norm((second - first) * pixel_scale))
                count = int(np.clip(math.ceil(length), 2, 2048))
                segments.append(
                    d2c.SourceSegment(
                        path_id,
                        order,
                        "L",
                        np.linspace(first, second, count + 1) * pixel_scale,
                    )
                )
            continue
        segments.append(
            d2c.SourceSegment(
                path_id, 0, "C", _dense_curve_pixels(primitive, pixel_scale)
            )
        )
    return d2c._source_index(segments), path_info


def _object_stage2_edges(
    drawing: CanonicalDrawing,
    object_mask: np.ndarray | None,
) -> list[dict]:
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
    if object_mask is None:
        object_mask = render_masks(drawing)["object"]
    skeleton = skeletonize(np.asarray(object_mask) > 0).astype(np.uint8) * 255
    keypoints = derive_puhachov_keypoints(object_drawing)
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
    clusters = d2c.s2._clusters_from_points(candidates, skeleton, snap_radius=4)
    nodes, edges = d2c.s2._extract_topology(skeleton, clusters)
    _, edges = d2c.s2._simplify_graph(
        nodes,
        edges,
        spur_min_len=6.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )
    return d2c.s2._smooth_edges(edges)


def _stage2_free2cad_arrays(
    drawing: CanonicalDrawing,
    source_index: int,
    max_points: int,
    object_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
    geometry_index, path_info = _canonical_source_index(drawing)
    edges = _object_stage2_edges(drawing, object_mask)
    config = d2c.PrepareConfig(max_pts=max_points)
    rows = []
    for edge_index, edge in enumerate(edges):
        raw = d2c._edge_points(edge)
        selected, _ = d2c._source_match(raw, geometry_index, config)
        command, fit = d2c.classify_edge(edge, geometry_index, config)
        if selected is None or command is None:
            continue
        try:
            sample = d2c._sample_from_edge(
                edge, command, fit, config, source_index
            )
        except ValueError:
            continue
        input_points = raw if edge.get("is_closed") else np.asarray(
            edge.get("smooth_pts") or edge["pixels"], dtype=np.float64
        )
        _, _, center_px, scale_px = d2c._normalise_edge(input_points, max_points)
        selected_paths = sorted({segment.path_id for segment in selected})
        member_ids = sorted(
            {
                member_id
                for path_id in selected_paths
                for member_id in path_info[path_id]["member_ids"]
            }
        )
        origin = max(path_info[path_id]["origin"] for path_id in selected_paths)
        rows.append(
            {
                **sample,
                "primitive_id": f"{drawing.sample_id}_s2e_{edge_index:04d}",
                "member_ids": member_ids,
                "origin": origin,
                "center_px": center_px,
                "scale_px": scale_px,
                "match_p90_px": float(fit.get("match_p90", math.nan)),
                "line_p90_px": float(fit.get("line_p90", math.nan)),
            }
        )
    if not rows:
        raise ValueError(
            f"drawing {drawing.sample_id} has no matched post-Stage-2 edges"
        )
    pixel_scale = np.asarray(
        [drawing.canvas[0] - 1, drawing.canvas[1] - 1], dtype=float
    )
    scale_denominator = float(np.max(pixel_scale))
    return {
        "points": np.stack([row["points"] for row in rows]),
        "mask": np.stack([row["mask"] for row in rows]),
        "types": np.asarray([row["type"] for row in rows], dtype=np.uint8),
        "params": np.stack([row["params"] for row in rows]),
        "centers": np.stack([row["center_px"] / pixel_scale for row in rows]).astype(
            np.float32
        ),
        "scales": np.asarray(
            [row["scale_px"] / scale_denominator for row in rows], dtype=np.float32
        ),
        "centers_px": np.stack([row["center_px"] for row in rows]).astype(np.float32),
        "scales_px": np.asarray([row["scale_px"] for row in rows], dtype=np.float32),
        "source_index": np.full(len(rows), int(source_index), dtype=np.int64),
        "primitive_ids": np.asarray([row["primitive_id"] for row in rows]),
        "member_primitive_ids": np.asarray(
            ["|".join(row["member_ids"]) for row in rows]
        ),
        "edge_origins": np.asarray([row["origin"] for row in rows], dtype=np.uint8),
        "topology_projected": np.ones(len(rows), dtype=np.uint8),
        "match_p90_px": np.asarray(
            [row["match_p90_px"] for row in rows], dtype=np.float32
        ),
        "line_p90_px": np.asarray(
            [row["line_p90_px"] for row in rows], dtype=np.float32
        ),
    }


def free2cad_arrays(
    drawing: CanonicalDrawing,
    source_index: int,
    max_points: int = 64,
    object_mask: np.ndarray | None = None,
    project_stage2: bool = True,
) -> dict[str, np.ndarray]:
    """Export Free2CAD supervision in the same shape produced by Stage 2."""
    if not project_stage2:
        return _primitive_free2cad_arrays(drawing, source_index, max_points)
    return _stage2_free2cad_arrays(
        drawing,
        source_index=source_index,
        max_points=max_points,
        object_mask=object_mask,
    )
