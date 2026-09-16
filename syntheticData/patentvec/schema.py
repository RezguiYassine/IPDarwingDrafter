from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import SCHEMA_VERSION


PRIMITIVE_KINDS = {
    "line",
    "polyline",
    "circle",
    "arc",
    "ellipse",
    "elliptical_arc",
    "quadratic_bezier",
    "cubic_bezier",
    "spline",
    "point",
    "text",
}

SEMANTIC_LAYERS = {
    "object_visible",
    "object_hidden",
    "object_center",
    "object_section_boundary",
    "construction",
    "hatch",
    "leader",
    "dimension",
    "arrowhead",
    "reference_numeral",
    "figure_label",
    "text_box",
    "diagram_connector",
    "text",
    "unknown",
}

JUNCTION_TYPES = {
    "endpoint",
    "sharp_corner",
    "smooth_continuation",
    "t_junction",
    "x_junction",
    "y_junction",
    "tangent_contact",
    "line_arc_transition",
    "arc_arc_transition",
    "connected_crossing",
    "disconnected_crossing",
    "leader_contact",
    "arrow_tip",
    "occlusion_boundary",
}


@dataclass
class SourceRecord:
    dataset: str
    sample_id: str
    split: str
    view: str | None = None
    license_id: str = ""


@dataclass
class TransformRecord:
    transform_id: str
    matrix: list[list[float]]
    source_frame: str
    target_frame: str
    reflection: bool = False


@dataclass
class Primitive:
    primitive_id: str
    kind: str
    geometry: dict[str, Any]
    semantic: str
    component_id: str
    source_dataset: str
    source_sample_id: str
    source_component_id: str
    source_primitive_id: str
    generated_by: str
    transform_id: str | None = None
    parent_primitive_ids: list[str] = field(default_factory=list)
    interaction_ids: list[str] = field(default_factory=list)
    visibility: str = "visible"
    source_interval: list[float] = field(default_factory=lambda: [0.0, 1.0])
    z_order: int = 0
    style: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Primitive":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Component:
    component_id: str
    source_dataset: str
    source_sample_id: str
    source_component_id: str
    source_split: str
    primitive_ids: list[str]
    transform_id: str | None = None
    parent_component_id: str | None = None
    descriptor: dict[str, Any] = field(default_factory=dict)


@dataclass
class Anchor:
    anchor_id: str
    component_id: str
    type: str
    position: list[float]
    direction: list[float] | None
    normal: list[float] | None
    primitive_ids: list[str]
    local_scale: float
    compatible_relations: list[str]
    parameter: float | None = None


@dataclass
class Interaction:
    interaction_id: str
    type: str
    host_component_id: str
    donor_component_id: str
    host_anchor_id: str | None
    donor_anchor_id: str | None
    residual: float
    intended_contacts: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Junction:
    junction_id: str
    type: str
    position: list[float]
    primitive_ids: list[str]
    component_ids: list[str]
    visible: bool = True
    interaction_id: str | None = None


@dataclass
class CanonicalDrawing:
    sample_id: str
    track: str
    difficulty: str
    seed: int
    split: str
    canvas: list[int]
    sources: list[SourceRecord]
    components: list[Component]
    transforms: list[TransformRecord]
    anchors: list[Anchor]
    interactions: list[Interaction]
    primitives_visible: list[Primitive]
    primitives_amodal: list[Primitive]
    junctions_visible: list[Junction]
    junctions_amodal: list[Junction]
    semantic_layers: dict[str, list[str]]
    images: dict[str, str] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    realism: dict[str, Any] = field(default_factory=dict)
    processing: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanonicalDrawing":
        values = dict(payload)
        values["sources"] = [SourceRecord(**x) for x in values.get("sources", [])]
        values["components"] = [Component(**x) for x in values.get("components", [])]
        values["transforms"] = [TransformRecord(**x) for x in values.get("transforms", [])]
        values["anchors"] = [Anchor(**x) for x in values.get("anchors", [])]
        values["interactions"] = [Interaction(**x) for x in values.get("interactions", [])]
        values["primitives_visible"] = [
            Primitive.from_dict(x) for x in values.get("primitives_visible", [])
        ]
        values["primitives_amodal"] = [
            Primitive.from_dict(x) for x in values.get("primitives_amodal", [])
        ]
        values["junctions_visible"] = [
            Junction(**x) for x in values.get("junctions_visible", [])
        ]
        values["junctions_amodal"] = [
            Junction(**x) for x in values.get("junctions_amodal", [])
        ]
        return cls(**values)


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    return True


def validate_primitive(primitive: Primitive) -> list[str]:
    errors: list[str] = []
    prefix = primitive.primitive_id
    if primitive.kind not in PRIMITIVE_KINDS:
        errors.append(f"{prefix}: unsupported kind {primitive.kind}")
    if primitive.semantic not in SEMANTIC_LAYERS:
        errors.append(f"{prefix}: unsupported semantic {primitive.semantic}")
    if not _finite(primitive.geometry):
        errors.append(f"{prefix}: non-finite geometry")
    if len(primitive.source_interval) != 2:
        errors.append(f"{prefix}: invalid source interval")
    else:
        lo, hi = primitive.source_interval
        if not (0.0 <= lo <= hi <= 1.0):
            errors.append(f"{prefix}: source interval outside [0, 1]")
    if not primitive.source_dataset or not primitive.source_sample_id:
        errors.append(f"{prefix}: missing source provenance")
    if not primitive.source_primitive_id or not primitive.generated_by:
        errors.append(f"{prefix}: incomplete lineage")

    geometry = primitive.geometry
    if primitive.kind == "line":
        points = np.asarray([geometry.get("p0"), geometry.get("p1")], dtype=float)
        if points.shape != (2, 2) or np.linalg.norm(points[1] - points[0]) <= 1e-9:
            errors.append(f"{prefix}: zero-length or malformed line")
    elif primitive.kind == "polyline":
        points = np.asarray(geometry.get("points", []), dtype=float)
        minimum = 3 if geometry.get("closed", False) else 2
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum:
            errors.append(f"{prefix}: malformed polyline")
    elif primitive.kind in {"circle", "arc"}:
        center = np.asarray(geometry.get("center", []), dtype=float)
        if center.shape != (2,) or float(geometry.get("radius", 0.0)) <= 1e-9:
            errors.append(f"{prefix}: invalid circular primitive")
    elif primitive.kind in {"quadratic_bezier", "cubic_bezier"}:
        names = ("p0", "p1", "p2") if primitive.kind == "quadratic_bezier" else (
            "p0", "p1", "p2", "p3"
        )
        points = np.asarray([geometry.get(name) for name in names], dtype=float)
        if points.shape != (len(names), 2):
            errors.append(f"{prefix}: malformed Bezier")
    return errors


def validate_drawing(drawing: CanonicalDrawing) -> list[str]:
    errors: list[str] = []
    if drawing.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema version {drawing.schema_version}")
    if drawing.track not in {"compose2d", "composecad"}:
        errors.append(f"invalid track {drawing.track}")

    components = {component.component_id: component for component in drawing.components}
    if len(components) != len(drawing.components):
        errors.append("duplicate component ID")
    interactions = {item.interaction_id: item for item in drawing.interactions}
    if len(interactions) != len(drawing.interactions):
        errors.append("duplicate interaction ID")
    transforms = {item.transform_id: item for item in drawing.transforms}
    if len(transforms) != len(drawing.transforms):
        errors.append("duplicate transform ID")

    for transform in drawing.transforms:
        matrix = np.asarray(transform.matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            errors.append(f"{transform.transform_id}: invalid transform")
        elif abs(float(np.linalg.det(matrix[:2, :2]))) <= 1e-12:
            errors.append(f"{transform.transform_id}: singular transform")

    for collection_name, primitives in (
        ("visible", drawing.primitives_visible),
        ("amodal", drawing.primitives_amodal),
    ):
        ids = {primitive.primitive_id for primitive in primitives}
        if len(ids) != len(primitives):
            errors.append(f"duplicate {collection_name} primitive ID")
        for primitive in primitives:
            errors.extend(validate_primitive(primitive))
            if primitive.component_id not in components:
                errors.append(
                    f"{primitive.primitive_id}: unknown component {primitive.component_id}"
                )
            if primitive.transform_id and primitive.transform_id not in transforms:
                errors.append(
                    f"{primitive.primitive_id}: unknown transform {primitive.transform_id}"
                )
            for interaction_id in primitive.interaction_ids:
                if interaction_id not in interactions:
                    errors.append(
                        f"{primitive.primitive_id}: unknown interaction {interaction_id}"
                    )

    source_splits = {source.split for source in drawing.sources}
    if source_splits and source_splits != {drawing.split}:
        errors.append("source split leakage inside composite")

    adjacency = {component_id: set() for component_id in components}
    for interaction in drawing.interactions:
        if interaction.host_component_id not in components:
            errors.append(f"{interaction.interaction_id}: unknown host component")
            continue
        if interaction.donor_component_id not in components:
            errors.append(f"{interaction.interaction_id}: unknown donor component")
            continue
        adjacency[interaction.host_component_id].add(interaction.donor_component_id)
        adjacency[interaction.donor_component_id].add(interaction.host_component_id)
        if not np.isfinite(interaction.residual):
            errors.append(f"{interaction.interaction_id}: non-finite residual")

    source_components = [
        component_id for component_id, component in components.items()
        if component.source_dataset != "generated"
    ]
    groups = [source_components]
    if drawing.processing.get("layout") == "independent_four_figure_sheet":
        figures = drawing.processing.get("figures", [])
        groups = [figure.get("source_component_ids", []) for figure in figures]
        declared = [identifier for group in groups for identifier in group]
        if (len(groups) != 4 or any(not group for group in groups)
                or len(declared) != len(set(declared)) or set(declared) != set(source_components)):
            errors.append("invalid sheet figure component partition")
            groups = [source_components]
        else:
            for group in groups:
                if any(adjacency[identifier] - set(group) for identifier in group):
                    errors.append("sheet has an undeclared cross-figure interaction")
    for group in groups:
        if len(group) <= 1:
            continue
        seen = set()
        stack = [group[0]]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current] - seen)
        if any(component_id not in seen for component_id in group):
            errors.append("disconnected component interaction graph")

    for collection_name, junctions, primitives in (
        ("visible", drawing.junctions_visible, drawing.primitives_visible),
        ("amodal", drawing.junctions_amodal, drawing.primitives_amodal),
    ):
        primitive_ids = {primitive.primitive_id for primitive in primitives}
        for junction in junctions:
            if junction.type not in JUNCTION_TYPES:
                errors.append(f"{junction.junction_id}: invalid junction type")
            missing = set(junction.primitive_ids) - primitive_ids
            if missing:
                errors.append(
                    f"{junction.junction_id}: missing {collection_name} primitives "
                    f"{sorted(missing)}"
                )
    return errors
