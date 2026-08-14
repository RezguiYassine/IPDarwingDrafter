from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from shapely import affinity
from shapely.geometry import LineString

from .anchors import component_region
from .geometry import primitive_bounds, primitive_endpoints, primitive_length, sample_primitive
from .schema import CanonicalDrawing, Component, Junction, Primitive


ANNOTATION_COMPONENT_ID = "annotations"


def _annotation_component(drawing: CanonicalDrawing) -> Component:
    for component in drawing.components:
        if component.component_id == ANNOTATION_COMPONENT_ID:
            return component
    component = Component(
        component_id=ANNOTATION_COMPONENT_ID,
        source_dataset="generated",
        source_sample_id=drawing.sample_id,
        source_component_id="patent_semantic_layers",
        source_split=drawing.split,
        primitive_ids=[],
        descriptor={"semantic_annotation_component": True},
    )
    drawing.components.append(component)
    return component


def _add_generated(
    drawing: CanonicalDrawing,
    kind: str,
    geometry: dict,
    semantic: str,
    generated_by: str,
    style: dict | None = None,
) -> Primitive:
    component = _annotation_component(drawing)
    serial = len(component.primitive_ids)
    primitive_id = f"{drawing.sample_id}_ann_p{serial:04d}"
    primitive = Primitive(
        primitive_id=primitive_id,
        kind=kind,
        geometry=geometry,
        semantic=semantic,
        component_id=component.component_id,
        source_dataset="generated",
        source_sample_id=drawing.sample_id,
        source_component_id=component.source_component_id,
        source_primitive_id=f"{generated_by}_{serial:04d}",
        generated_by=generated_by,
        style=style or {},
    )
    drawing.primitives_visible.append(primitive)
    drawing.primitives_amodal.append(replace(primitive))
    component.primitive_ids.append(primitive_id)
    drawing.semantic_layers.setdefault(semantic, []).append(primitive_id)
    return primitive


def _amodal_primitive_id(
    drawing: CanonicalDrawing, visible_primitive: Primitive
) -> str:
    matches = [
        primitive
        for primitive in drawing.primitives_amodal
        if primitive.component_id == visible_primitive.component_id
        and primitive.source_primitive_id == visible_primitive.source_primitive_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cannot resolve amodal parent for {visible_primitive.primitive_id}"
        )
    return matches[0].primitive_id


def _object_components(drawing: CanonicalDrawing) -> list[tuple[Component, list[Primitive]]]:
    visible_by_component: dict[str, list[Primitive]] = {}
    for primitive in drawing.primitives_visible:
        if primitive.semantic == "object_visible":
            visible_by_component.setdefault(primitive.component_id, []).append(primitive)
    return [
        (component, visible_by_component[component.component_id])
        for component in drawing.components
        if component.source_dataset != "generated"
        and component.component_id in visible_by_component
    ]


def _layout_boxes(drawing: CanonicalDrawing) -> list[list[float]]:
    return drawing.processing.setdefault("patent_layout", {}).setdefault(
        "reserved_boxes", []
    )


def _box_available(
    drawing: CanonicalDrawing,
    box: list[float],
    padding: float = 0.008,
) -> bool:
    x0, y0, x1, y1 = box
    if x0 < 0.018 or y0 < 0.018 or x1 > 0.982 or y1 > 0.982:
        return False
    for existing in _layout_boxes(drawing):
        ex0, ey0, ex1, ey1 = existing
        if not (
            x1 + padding < ex0
            or ex1 + padding < x0
            or y1 + padding < ey0
            or ey1 + padding < y0
        ):
            return False
    return True


def _reserve_box(drawing: CanonicalDrawing, box: list[float]) -> None:
    _layout_boxes(drawing).append([float(value) for value in box])


def add_hatching(
    drawing: CanonicalDrawing,
    angle_degrees: float = 45.0,
    spacing: float = 0.018,
    cross_hatch: bool = False,
    host_component_id: str | None = None,
) -> list[Primitive]:
    candidates = []
    for component, primitives in _object_components(drawing):
        region = component_region(primitives)
        if (
            region is not None
            and region.area > 1e-4
            and (host_component_id is None or component.component_id == host_component_id)
        ):
            candidates.append((region.area, component, region))
    if not candidates:
        return []
    _, host_component, region = max(candidates, key=lambda item: item[0])
    output: list[Primitive] = []
    angles = [angle_degrees]
    if cross_hatch:
        angles.append(angle_degrees + 90.0)
    for angle in angles:
        # Rotate the clipping region so hatch construction is axis-aligned,
        # then rotate each exact clipped segment back into the drawing frame.
        rotated = affinity.rotate(region, -angle, origin="centroid", use_radians=False)
        min_x, min_y, max_x, max_y = rotated.bounds
        y_values = np.arange(min_y - spacing, max_y + spacing, spacing)
        for y_value in y_values:
            line = LineString([(min_x - spacing, y_value), (max_x + spacing, y_value)])
            clipped = rotated.intersection(line)
            segments = []
            if clipped.geom_type == "LineString":
                segments = [clipped]
            elif clipped.geom_type in {"MultiLineString", "GeometryCollection"}:
                segments = [item for item in clipped.geoms if item.geom_type == "LineString"]
            for segment in segments:
                restored = affinity.rotate(
                    segment, angle, origin=region.centroid, use_radians=False
                )
                coordinates = np.asarray(restored.coords, dtype=float)
                if len(coordinates) < 2 or np.linalg.norm(coordinates[-1] - coordinates[0]) < 1e-5:
                    continue
                output.append(
                    _add_generated(
                        drawing,
                        "line",
                        {"p0": coordinates[0].tolist(), "p1": coordinates[-1].tolist()},
                        "hatch",
                        "patent_hatching",
                        {
                            "stroke_width": 0.00075,
                            "host_component_id": host_component.component_id,
                            "hatch_angle_degrees": angle,
                        },
                    )
                )
    drawing.processing.setdefault("patent_layers", {}).setdefault(
        "hatching", []
    ).append(
        {
            "host_component_id": host_component.component_id,
            "angle_degrees": angles,
            "spacing": spacing,
            "segment_count": len(output),
        }
    )
    return output


def add_dashed_centerline(
    drawing: CanonicalDrawing,
    orientation: str = "horizontal",
    offset: float = 0.0,
) -> Primitive:
    object_primitives = [
        primitive for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
    ]
    lower, upper = primitive_bounds(object_primitives)
    center = 0.5 * (lower + upper)
    if orientation == "vertical":
        center[0] += offset
    else:
        center[1] += offset
    extension = 0.035
    if orientation == "vertical":
        p0 = [float(center[0]), float(max(0.02, lower[1] - extension))]
        p1 = [float(center[0]), float(min(0.98, upper[1] + extension))]
    else:
        p0 = [float(max(0.02, lower[0] - extension)), float(center[1])]
        p1 = [float(min(0.98, upper[0] + extension)), float(center[1])]
    primitive = _add_generated(
        drawing,
        "line",
        {"p0": p0, "p1": p1},
        "object_center",
        "patent_centerline",
        {
            "stroke_width": 0.0010,
            "dasharray": [0.028, 0.009, 0.004, 0.009],
            "logical_primitive": True,
        },
    )
    drawing.processing.setdefault("patent_layers", {}).setdefault(
        "centerlines", []
    ).append(
        {
            "orientation": orientation,
            "offset": offset,
            "primitive_id": primitive.primitive_id,
        }
    )
    return primitive


def add_leader_and_numeral(
    drawing: CanonicalDrawing,
    numeral: str,
    rng: np.random.Generator,
    target_component_id: str | None = None,
) -> list[Primitive]:
    object_primitives = [
        primitive for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
        and (
            target_component_id is None
            or primitive.component_id == target_component_id
        )
    ]
    if not object_primitives:
        return []
    eligible = [item for item in object_primitives if primitive_length(item) > 0.025]
    target_primitive = (
        eligible[int(rng.integers(len(eligible)))]
        if eligible
        else max(object_primitives, key=primitive_length)
    )
    target_samples = sample_primitive(target_primitive, 64)
    target = target_samples[int(rng.integers(12, max(13, len(target_samples) - 12)))]
    object_primitives_all = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
    ]
    lower, upper = primitive_bounds(object_primitives_all)
    object_center = 0.5 * (lower + upper)
    preferred_x = 1.0 if target[0] >= object_center[0] else -1.0
    preferred_y = 1.0 if target[1] >= object_center[1] else -1.0
    directions = [
        (preferred_x, preferred_y),
        (preferred_x, -preferred_y),
        (-preferred_x, preferred_y),
        (-preferred_x, -preferred_y),
    ]
    if rng.random() < 0.5:
        directions[1], directions[2] = directions[2], directions[1]
    text_size = float(rng.uniform(0.020, 0.027))
    selected = None
    for direction_x, direction_y in directions:
        for distance in (0.085, 0.115, 0.145):
            direction = np.asarray([direction_x, direction_y], dtype=float)
            direction /= np.linalg.norm(direction)
            elbow = target + direction * distance
            end = elbow + np.array([direction_x * 0.075, 0.0])
            all_points = np.clip(np.vstack([target, elbow, end]), 0.025, 0.975)
            anchor = "start" if direction_x > 0 else "end"
            text_position = all_points[-1] + np.array([direction_x * 0.007, -0.005])
            text_width = text_size * 0.62 * max(1, len(str(numeral)))
            text_box = [
                float(text_position[0] if anchor == "start" else text_position[0] - text_width),
                float(text_position[1] - text_size),
                float(text_position[0] + text_width if anchor == "start" else text_position[0]),
                float(text_position[1] + text_size * 0.18),
            ]
            if _box_available(drawing, text_box):
                selected = (all_points, text_position, anchor, text_box)
                break
        if selected is not None:
            break
    if selected is None:
        all_points = np.clip(
            np.vstack([target, target + [0.07, -0.07], target + [0.14, -0.07]]),
            0.025,
            0.975,
        )
        text_position = all_points[-1] + [0.007, -0.005]
        anchor = "start"
        text_box = [
            float(text_position[0]),
            float(text_position[1] - text_size),
            float(min(0.98, text_position[0] + text_size * 0.62 * len(str(numeral)))),
            float(text_position[1] + text_size * 0.18),
        ]
    else:
        all_points, text_position, anchor, text_box = selected
    _reserve_box(drawing, text_box)

    leader = _add_generated(
        drawing,
        "polyline",
        {"points": all_points.tolist(), "closed": False},
        "leader",
        "patent_leader",
        {"stroke_width": 0.00115},
    )
    segment_direction = all_points[1] - all_points[0]
    segment_direction /= max(np.linalg.norm(segment_direction), 1e-12)
    normal = np.array([-segment_direction[1], segment_direction[0]])
    arrow_length = 0.014
    arrow_width = 0.006
    wing_a = all_points[0] + arrow_length * segment_direction + arrow_width * normal
    wing_b = all_points[0] + arrow_length * segment_direction - arrow_width * normal
    arrow = _add_generated(
        drawing,
        "polyline",
        {
            "points": [wing_a.tolist(), all_points[0].tolist(), wing_b.tolist()],
            "closed": False,
        },
        "arrowhead",
        "patent_arrowhead",
        {"stroke_width": 0.00115, "fill": "none"},
    )
    text = _add_generated(
        drawing,
        "text",
        {
            "position": np.asarray(text_position).tolist(),
            "text": str(numeral),
            "size": text_size,
            "rotation": float(rng.uniform(-2.0, 2.0)),
            "anchor": anchor,
        },
        "reference_numeral",
        "patent_reference_numeral",
        {"font_family": "DejaVu Sans", "font_weight": "normal"},
    )
    drawing.junctions_visible.append(
        Junction(
            junction_id=f"j{len(drawing.junctions_visible):03d}",
            type="leader_contact",
            position=all_points[0].tolist(),
            primitive_ids=[target_primitive.primitive_id, leader.primitive_id, arrow.primitive_id],
            component_ids=[target_primitive.component_id, ANNOTATION_COMPONENT_ID],
            visible=True,
        )
    )
    drawing.junctions_amodal.append(
        replace(
            drawing.junctions_visible[-1],
            primitive_ids=[
                _amodal_primitive_id(drawing, target_primitive),
                leader.primitive_id,
                arrow.primitive_id,
            ],
        )
    )
    drawing.processing.setdefault("patent_layers", {}).setdefault(
        "leaders", []
    ).append(
        {
            "target_primitive_id": target_primitive.primitive_id,
            "leader_id": leader.primitive_id,
            "text_id": text.primitive_id,
            "text": str(numeral),
        }
    )
    return [leader, arrow, text]


def add_dashed_hidden_circle(
    drawing: CanonicalDrawing,
    rng: np.random.Generator,
    target_component_id: str | None = None,
) -> Primitive:
    candidates = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
        and primitive.component_id != ANNOTATION_COMPONENT_ID
        and (
            target_component_id is None
            or primitive.component_id == target_component_id
        )
    ]
    circles = [primitive for primitive in candidates if primitive.kind == "circle"]
    if circles:
        source = circles[int(rng.integers(len(circles)))]
        center = np.asarray(source.geometry["center"], dtype=float)
        radius = float(source.geometry["radius"]) * float(rng.uniform(0.45, 0.78))
    else:
        lower, upper = primitive_bounds(candidates)
        center = 0.5 * (lower + upper)
        radius = max(0.018, float(np.min(upper - lower)) * float(rng.uniform(0.18, 0.32)))
    radius = min(radius, center[0] - 0.02, center[1] - 0.02, 0.98 - center[0], 0.98 - center[1])
    if radius <= 0.01:
        raise ValueError("no room for dashed hidden circle")
    primitive = _add_generated(
        drawing,
        "circle",
        {"center": center.tolist(), "radius": float(radius)},
        "object_hidden",
        "patent_hidden_circle",
        {
            "stroke_width": 0.00095,
            "dasharray": [0.013, 0.009],
            "logical_primitive": True,
        },
    )
    drawing.processing.setdefault("patent_layers", {}).setdefault(
        "hidden_geometry", []
    ).append({"primitive_id": primitive.primitive_id, "kind": "circle"})
    return primitive


def _dimension_arrow(
    drawing: CanonicalDrawing,
    tip: np.ndarray,
    inward: np.ndarray,
) -> Primitive:
    inward = inward / max(float(np.linalg.norm(inward)), 1e-12)
    normal = np.array([-inward[1], inward[0]])
    wing_a = tip + inward * 0.013 + normal * 0.0045
    wing_b = tip + inward * 0.013 - normal * 0.0045
    return _add_generated(
        drawing,
        "polyline",
        {
            "points": [wing_a.tolist(), tip.tolist(), wing_b.tolist()],
            "closed": False,
        },
        "arrowhead",
        "patent_dimension_arrowhead",
        {"stroke_width": 0.00095},
    )


def add_linear_dimension(
    drawing: CanonicalDrawing,
    rng: np.random.Generator,
    target_component_id: str | None = None,
    orientation: str | None = None,
) -> list[Primitive]:
    candidates = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
        and (
            target_component_id is None
            or primitive.component_id == target_component_id
        )
    ]
    if not candidates:
        return []
    lower, upper = primitive_bounds(candidates)
    orientation = orientation or str(rng.choice(["horizontal", "vertical"]))
    output = []
    margin = float(rng.uniform(0.035, 0.065))
    extension = 0.009
    if orientation == "horizontal":
        use_above = lower[1] - margin > 0.035
        axis = lower[1] - margin if use_above else upper[1] + margin
        axis = float(np.clip(axis, 0.03, 0.97))
        x0, x1 = float(lower[0]), float(upper[0])
        if x1 - x0 < 0.04:
            raise ValueError("horizontal dimension span is too small")
        object_y = float(lower[1] if use_above else upper[1])
        output.extend(
            [
                _add_generated(
                    drawing,
                    "line",
                    {"p0": [x0, object_y], "p1": [x0, axis + (extension if use_above else -extension)]},
                    "dimension",
                    "patent_dimension_extension",
                    {"stroke_width": 0.00085},
                ),
                _add_generated(
                    drawing,
                    "line",
                    {"p0": [x1, object_y], "p1": [x1, axis + (extension if use_above else -extension)]},
                    "dimension",
                    "patent_dimension_extension",
                    {"stroke_width": 0.00085},
                ),
                _add_generated(
                    drawing,
                    "line",
                    {"p0": [x0, axis], "p1": [x1, axis]},
                    "dimension",
                    "patent_dimension_line",
                    {"stroke_width": 0.00095},
                ),
            ]
        )
        output.extend(
            [
                _dimension_arrow(drawing, np.array([x0, axis]), np.array([1.0, 0.0])),
                _dimension_arrow(drawing, np.array([x1, axis]), np.array([-1.0, 0.0])),
            ]
        )
        text_position = [0.5 * (x0 + x1), axis - 0.007 if use_above else axis + 0.020]
        value = str(int(round((x1 - x0) * 1000)))
    else:
        use_left = lower[0] - margin > 0.035
        axis = lower[0] - margin if use_left else upper[0] + margin
        axis = float(np.clip(axis, 0.03, 0.97))
        y0, y1 = float(lower[1]), float(upper[1])
        if y1 - y0 < 0.04:
            raise ValueError("vertical dimension span is too small")
        object_x = float(lower[0] if use_left else upper[0])
        output.extend(
            [
                _add_generated(
                    drawing,
                    "line",
                    {"p0": [object_x, y0], "p1": [axis + (extension if use_left else -extension), y0]},
                    "dimension",
                    "patent_dimension_extension",
                    {"stroke_width": 0.00085},
                ),
                _add_generated(
                    drawing,
                    "line",
                    {"p0": [object_x, y1], "p1": [axis + (extension if use_left else -extension), y1]},
                    "dimension",
                    "patent_dimension_extension",
                    {"stroke_width": 0.00085},
                ),
                _add_generated(
                    drawing,
                    "line",
                    {"p0": [axis, y0], "p1": [axis, y1]},
                    "dimension",
                    "patent_dimension_line",
                    {"stroke_width": 0.00095},
                ),
            ]
        )
        output.extend(
            [
                _dimension_arrow(drawing, np.array([axis, y0]), np.array([0.0, 1.0])),
                _dimension_arrow(drawing, np.array([axis, y1]), np.array([0.0, -1.0])),
            ]
        )
        text_position = [axis - 0.008 if use_left else axis + 0.008, 0.5 * (y0 + y1)]
        value = str(int(round((y1 - y0) * 1000)))
    text_position = np.clip(text_position, 0.025, 0.975).tolist()
    text_width = 0.018 * 0.62 * len(value)
    dimension_text_box = [
        text_position[0] - text_width / 2,
        text_position[1] - 0.018,
        text_position[0] + text_width / 2,
        text_position[1] + 0.004,
    ]
    if _box_available(drawing, dimension_text_box, padding=0.004):
        _reserve_box(drawing, dimension_text_box)
    output.append(
        _add_generated(
            drawing,
            "text",
            {
                "position": text_position,
                "text": value,
                "size": 0.018,
                "rotation": 0.0 if orientation == "horizontal" else -90.0,
                "anchor": "middle",
            },
            "dimension",
            "patent_dimension_text",
            {"font_family": "DejaVu Sans"},
        )
    )
    drawing.processing.setdefault("patent_layers", {}).setdefault(
        "dimensions", []
    ).append(
        {
            "orientation": orientation,
            "target_component_id": target_component_id,
            "value": value,
            "primitive_ids": [item.primitive_id for item in output],
        }
    )
    return output


def add_text_box(
    drawing: CanonicalDrawing,
    rng: np.random.Generator,
    label: str,
) -> list[Primitive]:
    object_primitives = [
        primitive
        for primitive in drawing.primitives_visible
        if primitive.semantic == "object_visible"
    ]
    lower, upper = primitive_bounds(object_primitives)
    width = float(rng.uniform(0.105, 0.155))
    height = float(rng.uniform(0.045, 0.068))
    object_center = 0.5 * (lower + upper)
    gap = 0.045
    placements = [
        (object_center[0] - width / 2, lower[1] - height - gap),
        (upper[0] + gap, object_center[1] - height / 2),
        (object_center[0] - width / 2, upper[1] + gap),
        (lower[0] - width - gap, object_center[1] - height / 2),
        (0.035, 0.035),
        (0.965 - width, 0.965 - height),
    ]
    ordered_placements = [
        (float(x), float(y))
        for x, y in placements
        if 0.025 <= x <= 0.975 - width and 0.025 <= y <= 0.975 - height
    ]
    if not ordered_placements:
        ordered_placements = [(0.035, 0.035)]
    if rng.random() < 0.5:
        ordered_placements = list(reversed(ordered_placements))
    selected_position = next(
        (
            position
            for position in ordered_placements
            if _box_available(
                drawing,
                [position[0], position[1], position[0] + width, position[1] + height],
            )
        ),
        ordered_placements[0],
    )
    x0, y0 = selected_position
    _reserve_box(drawing, [x0, y0, x0 + width, y0 + height])
    box = _add_generated(
        drawing,
        "polyline",
        {
            "points": [
                [x0, y0],
                [x0 + width, y0],
                [x0 + width, y0 + height],
                [x0, y0 + height],
            ],
            "closed": True,
        },
        "text_box",
        "patent_text_box",
        {"stroke_width": 0.00105},
    )
    text = _add_generated(
        drawing,
        "text",
        {
            "position": [x0 + width / 2, y0 + height * 0.68],
            "text": label,
            "size": min(0.022, height * 0.42),
            "rotation": 0.0,
            "anchor": "middle",
        },
        "text",
        "patent_text_box_label",
        {"font_family": "DejaVu Sans"},
    )
    box_center = np.array([x0 + width / 2, y0 + height / 2])
    target = np.clip(object_center, lower, upper)
    delta = target - box_center
    if abs(delta[0]) / max(width, 1e-9) >= abs(delta[1]) / max(height, 1e-9):
        connector_start = np.array(
            [x0 + width if delta[0] > 0 else x0, box_center[1]]
        )
    else:
        connector_start = np.array(
            [box_center[0], y0 + height if delta[1] > 0 else y0]
        )
    connector = _add_generated(
        drawing,
        "line",
        {"p0": connector_start.tolist(), "p1": target.tolist()},
        "diagram_connector",
        "patent_text_box_connector",
        {"stroke_width": 0.0009},
    )
    drawing.processing.setdefault("patent_layers", {}).setdefault(
        "text_boxes", []
    ).append(
        {
            "label": label,
            "primitive_ids": [box.primitive_id, text.primitive_id, connector.primitive_id],
        }
    )
    return [box, text, connector]


def apply_complex_patent_layers(
    drawing: CanonicalDrawing,
    seed: int,
    difficulty: str,
) -> CanonicalDrawing:
    if difficulty not in {"medium", "hard", "very_hard"}:
        raise ValueError(
            "complex patent layers require medium, hard, or very_hard difficulty"
        )
    rng = np.random.default_rng(seed)
    source_component_ids = [
        component.component_id
        for component in drawing.components
        if component.source_dataset != "generated"
    ]
    region_candidates = []
    for component, primitives in _object_components(drawing):
        region = component_region(primitives)
        if region is not None and region.area > 1e-4:
            region_candidates.append((region.area, component.component_id))
    region_candidates.sort(reverse=True)
    hatch_count = {"medium": 1, "hard": 2, "very_hard": 3}[difficulty]
    cross_hatch_probability = {"medium": 0.0, "hard": 0.40, "very_hard": 0.65}[
        difficulty
    ]
    if region_candidates:
        for index in range(hatch_count):
            host_id = region_candidates[index % len(region_candidates)][1]
            add_hatching(
                drawing,
                angle_degrees=float(rng.choice([30.0, 45.0, 60.0, 120.0, 135.0])),
                spacing=float(rng.uniform(0.013, 0.021)),
                cross_hatch=bool(rng.random() < cross_hatch_probability),
                host_component_id=host_id,
            )

    add_dashed_centerline(
        drawing, orientation="horizontal" if rng.random() < 0.5 else "vertical"
    )
    if difficulty in {"hard", "very_hard"}:
        add_dashed_centerline(
            drawing,
            orientation="vertical" if rng.random() < 0.5 else "horizontal",
            offset=float(rng.choice([-0.035, 0.035])),
        )
    if difficulty == "very_hard":
        add_dashed_centerline(
            drawing,
            orientation="vertical" if rng.random() < 0.5 else "horizontal",
            offset=float(rng.choice([-0.065, 0.065])),
        )

    hidden_count = {"medium": 1, "hard": 2, "very_hard": 3}[difficulty]
    for index in range(hidden_count):
        target_id = source_component_ids[index % len(source_component_ids)]
        try:
            add_dashed_hidden_circle(drawing, rng, target_component_id=target_id)
        except ValueError:
            add_dashed_hidden_circle(drawing, rng)

    leader_bounds = {
        "medium": (2, 4),
        "hard": (4, 7),
        "very_hard": (7, 11),
    }[difficulty]
    leader_count = int(rng.integers(*leader_bounds))
    shuffled = list(source_component_ids)
    rng.shuffle(shuffled)
    for index in range(leader_count):
        add_leader_and_numeral(
            drawing,
            numeral=str(int(rng.integers(1, 199))),
            rng=rng,
            target_component_id=shuffled[index % len(shuffled)],
        )

    dimension_count = {"medium": 1, "hard": 2, "very_hard": 3}[difficulty]
    for index in range(dimension_count):
        add_linear_dimension(
            drawing,
            rng,
            target_component_id=None,
            orientation="horizontal" if index % 2 == 0 else "vertical",
        )

    text_box_count = 2 if difficulty == "very_hard" else 1
    for index in range(text_box_count):
        add_text_box(drawing, rng, label=f"B{index + 1}")
    drawing.processing.setdefault("patent_layers", {})["profile"] = difficulty
    return drawing


def apply_vertical_slice_layers(
    drawing: CanonicalDrawing,
    seed: int,
    numeral: str = "12",
) -> CanonicalDrawing:
    rng = np.random.default_rng(seed)
    add_hatching(
        drawing,
        angle_degrees=float(rng.choice([30.0, 45.0, 60.0, 135.0])),
        spacing=float(rng.uniform(0.016, 0.023)),
        cross_hatch=bool(rng.random() < 0.20),
    )
    add_dashed_centerline(
        drawing, orientation="horizontal" if rng.random() < 0.5 else "vertical"
    )
    add_leader_and_numeral(drawing, numeral=numeral, rng=rng)
    return drawing
