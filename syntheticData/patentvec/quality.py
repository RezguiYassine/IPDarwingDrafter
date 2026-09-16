from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from shapely.geometry import LineString

from .geometry import sample_primitive
from .schema import CanonicalDrawing, validate_drawing


def _intersection_points(geometry) -> tuple[list[np.ndarray], bool]:
    if geometry.is_empty:
        return [], False
    if geometry.geom_type == "Point":
        return [np.asarray(geometry.coords[0], dtype=float)], False
    if geometry.geom_type == "MultiPoint":
        return [np.asarray(item.coords[0], dtype=float) for item in geometry.geoms], False
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [], geometry.length > 1e-6
    if hasattr(geometry, "geoms"):
        points = []
        overlap = False
        for item in geometry.geoms:
            item_points, item_overlap = _intersection_points(item)
            points.extend(item_points)
            overlap = overlap or item_overlap
        return points, overlap
    return [], False


def _component_intersections(drawing: CanonicalDrawing, host_id: str, donor_id: str):
    host = [
        item for item in drawing.primitives_visible
        if item.component_id == host_id and item.semantic == "object_visible"
    ]
    donor = [
        item for item in drawing.primitives_visible
        if item.component_id == donor_id and item.semantic == "object_visible"
    ]
    points = []
    overlap = False
    for first in host:
        first_points = sample_primitive(first, 256)
        if first.kind == "circle":
            first_points = np.vstack([first_points, first_points[0]])
        if len(first_points) < 2:
            continue
        first_line = LineString(first_points)
        for second in donor:
            second_points = sample_primitive(second, 256)
            if second.kind == "circle":
                second_points = np.vstack([second_points, second_points[0]])
            if len(second_points) < 2:
                continue
            item_points, item_overlap = _intersection_points(
                first_line.intersection(LineString(second_points))
            )
            points.extend(item_points)
            overlap = overlap or item_overlap
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - other) < 0.002 for other in unique):
            unique.append(point)
    return unique, overlap


def _complexity_metrics(
    drawing: CanonicalDrawing, clean: np.ndarray | None = None
) -> dict:
    source_components = [
        component
        for component in drawing.components
        if component.source_dataset != "generated"
    ]
    source_ids = {component.component_id for component in source_components}
    interaction_types = {interaction.type for interaction in drawing.interactions}
    object_primitives = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
        and primitive.component_id in source_ids
    ]
    structural_junctions = [
        junction
        for junction in drawing.junctions_visible
        if junction.interaction_id is not None
    ]
    node_count = len(source_components)
    edge_count = len(drawing.interactions)
    cycle_rank = max(0, edge_count - node_count + 1) if node_count else 0
    metrics = {
        "source_component_count": node_count,
        "interaction_count": edge_count,
        "interaction_type_count": len(interaction_types),
        "interaction_types": sorted(interaction_types),
        "cycle_rank": cycle_rank,
        "object_primitive_count": len(object_primitives),
        "structural_junction_count": len(structural_junctions),
        "source_dataset_count": len({item.source_dataset for item in source_components}),
    }
    if clean is not None:
        foreground = clean < 220
        tile_ratios = []
        for y_indices in np.array_split(np.arange(foreground.shape[0]), 8):
            for x_indices in np.array_split(np.arange(foreground.shape[1]), 8):
                tile = foreground[np.ix_(y_indices, x_indices)]
                tile_ratios.append(float(tile.mean()))
        metrics.update(
            {
                "local_density_p95": float(np.quantile(tile_ratios, 0.95)),
                "dense_tile_fraction": float(np.mean(np.asarray(tile_ratios) > 0.04)),
                "occupied_tile_fraction": float(np.mean(np.asarray(tile_ratios) > 0.002)),
            }
        )
    return metrics


def evaluate_quality(
    drawing: CanonicalDrawing,
    clean: np.ndarray | None = None,
    masks: dict[str, np.ndarray] | None = None,
    require_vertical_slice: bool = False,
    difficulty_gate: str | None = None,
) -> dict:
    from .annotations import annotation_failures
    failures = list(validate_drawing(drawing)) + annotation_failures(drawing)
    warnings: list[str] = []

    all_points = []
    for primitive in drawing.primitives_visible:
        try:
            all_points.append(sample_primitive(primitive, 128))
        except ValueError:
            failures.append(f"{primitive.primitive_id}: cannot sample for bounds gate")
    points = np.concatenate(all_points) if all_points else np.zeros((0, 2))
    bounds = None
    if len(points):
        lower, upper = points.min(axis=0), points.max(axis=0)
        bounds = [*lower.tolist(), *upper.tolist()]
        if np.any(lower < -0.005) or np.any(upper > 1.005):
            failures.append(f"visible geometry leaves canvas: {bounds}")
        elif np.any(lower < 0.015) or np.any(upper > 0.985):
            warnings.append(f"visible geometry is close to canvas edge: {bounds}")

    max_residual = max((item.residual for item in drawing.interactions), default=0.0)
    intended_by_pair: dict[frozenset[str], list[np.ndarray]] = defaultdict(list)
    interaction_ids_by_pair: dict[frozenset[str], list[str]] = defaultdict(list)
    for interaction in drawing.interactions:
        tolerance = 1e-7 if interaction.type != "containment" else 0.02
        if interaction.residual > tolerance:
            failures.append(
                f"{interaction.interaction_id}: {interaction.type} residual "
                f"{interaction.residual:.6g} exceeds {tolerance:.6g}"
            )
        pair = frozenset([interaction.host_component_id, interaction.donor_component_id])
        intended_by_pair[pair].extend(
            np.asarray(position, dtype=float) for position in interaction.intended_contacts
        )
        interaction_ids_by_pair[pair].append(interaction.interaction_id)

    source_component_ids = [
        component.component_id
        for component in drawing.components
        if component.source_dataset != "generated"
    ]
    for first_id, second_id in combinations(source_component_ids, 2):
        pair = frozenset([first_id, second_id])
        intersection_points, overlap = _component_intersections(
            drawing, first_id, second_id
        )
        label = ",".join(interaction_ids_by_pair.get(pair, [])) or f"{first_id}/{second_id}"
        if overlap:
            failures.append(f"{label}: coincident cross-component stroke")
        intended = intended_by_pair.get(pair, [])
        unexpected = [
            point for point in intersection_points
            if not intended
            or min(np.linalg.norm(point - target) for target in intended) > 0.004
        ]
        if unexpected:
            failures.append(
                f"{label}: {len(unexpected)} unintended cross-component intersections"
            )

    amodal_by_component = defaultdict(set)
    for primitive in drawing.primitives_amodal:
        if primitive.source_dataset != "generated":
            amodal_by_component[primitive.component_id].add(primitive.source_primitive_id)
    for component in drawing.components:
        if component.source_dataset == "generated":
            continue
        expected = int(component.descriptor.get("primitive_count", 0))
        actual = len(amodal_by_component[component.component_id])
        if expected and actual != expected:
            failures.append(
                f"{component.component_id}: amodal source completeness {actual}/{expected}"
            )

    intervals = defaultdict(list)
    for primitive in drawing.primitives_visible:
        if primitive.source_dataset != "generated":
            intervals[(primitive.component_id, primitive.source_primitive_id)].append(
                primitive.source_interval
            )
    for key, source_intervals in intervals.items():
        ordered = sorted(source_intervals)
        if abs(ordered[0][0]) > 1e-7 or abs(ordered[-1][1] - 1.0) > 1e-7:
            failures.append(f"{key}: visible source intervals do not cover [0, 1]")
            continue
        for first, second in zip(ordered[:-1], ordered[1:]):
            if abs(first[1] - second[0]) > 1e-7:
                failures.append(f"{key}: visible source interval gap or overlap")

    semantic_counts = {
        semantic: len(primitive_ids)
        for semantic, primitive_ids in drawing.semantic_layers.items()
    }
    interaction_counts = defaultdict(int)
    for interaction in drawing.interactions:
        interaction_counts[interaction.type] += 1

    if require_vertical_slice:
        source_datasets = {source.dataset for source in drawing.sources}
        if not {"SketchGraphs", "CAD-VGDrawing"}.issubset(source_datasets):
            failures.append("vertical slice lacks SketchGraphs + CAD-VGDrawing sources")
        if not interaction_counts["t_junction"]:
            failures.append("vertical slice lacks a T-junction")
        for semantic in ("object_center", "hatch", "leader", "reference_numeral"):
            if not semantic_counts.get(semantic, 0):
                failures.append(f"vertical slice lacks {semantic} semantics")

    complexity = _complexity_metrics(drawing, clean=clean)
    if difficulty_gate in {"medium", "hard", "very_hard"}:
        requirements = {
            "medium": {
                "source_component_count": 4,
                "interaction_type_count": 2,
                "object_primitive_count": 16,
                "structural_junction_count": 2,
                "cycle_rank": 0,
            },
            "hard": {
                "source_component_count": 5,
                "interaction_type_count": 3,
                "object_primitive_count": 16,
                "structural_junction_count": 4,
                "cycle_rank": 1,
            },
            "very_hard": {
                "source_component_count": 9,
                "interaction_type_count": 5,
                "object_primitive_count": 24,
                "structural_junction_count": 8,
                "cycle_rank": 2,
            },
        }[difficulty_gate]
        for metric, minimum in requirements.items():
            if complexity[metric] < minimum:
                failures.append(
                    f"{difficulty_gate} complexity gate: {metric} "
                    f"{complexity[metric]} < {minimum}"
                )
        required_semantics = {"hatch", "leader", "reference_numeral", "dimension"}
        if difficulty_gate in {"hard", "very_hard"}:
            required_semantics.update({"object_hidden", "text_box"})
        for semantic in sorted(required_semantics):
            if not semantic_counts.get(semantic, 0):
                failures.append(
                    f"{difficulty_gate} complexity gate: missing semantic {semantic}"
                )

    foreground_ratio = None
    if clean is not None:
        foreground_ratio = float(np.mean(clean < 220))
        minimum_foreground = {
            "medium": 0.006,
            "hard": 0.009,
            "very_hard": 0.012,
        }.get(difficulty_gate, 0.003)
        if foreground_ratio < minimum_foreground:
            failures.append(f"foreground ratio too low: {foreground_ratio:.5f}")
        if foreground_ratio > 0.35:
            failures.append(f"foreground ratio too high: {foreground_ratio:.5f}")

    mask_foreground = {}
    if masks is not None:
        for name, mask in masks.items():
            mask_foreground[name] = int(np.count_nonzero(mask))
        expected_masks = {
            "object": "object_visible",
            "hidden_center": "object_center",
            "hatch": "hatch",
            "leader_dimension": "leader",
            "text_numeral": "reference_numeral",
        }
        for mask_name, semantic in expected_masks.items():
            if semantic_counts.get(semantic, 0) and not mask_foreground.get(mask_name, 0):
                failures.append(f"semantic mask {mask_name} is blank despite {semantic}")

    report = {
        "accepted": not failures,
        "failures": failures,
        "warnings": warnings,
        "sample_id": drawing.sample_id,
        "component_count": len(
            [item for item in drawing.components if item.source_dataset != "generated"]
        ),
        "visible_primitive_count": len(drawing.primitives_visible),
        "amodal_primitive_count": len(drawing.primitives_amodal),
        "interaction_counts": dict(interaction_counts),
        "semantic_counts": semantic_counts,
        "maximum_interaction_residual": max_residual,
        "bounds": bounds,
        "foreground_ratio": foreground_ratio,
        "mask_foreground_pixels": mask_foreground,
        "complexity": complexity,
    }
    drawing.quality = report
    return report
