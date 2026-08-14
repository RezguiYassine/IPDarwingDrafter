from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from .geometry import (
    primitive_endpoints,
    primitive_length,
    primitive_tangent,
    sample_primitive,
)
from .schema import Anchor, Primitive


COMPATIBILITY: dict[tuple[str, str], tuple[str, ...]] = {
    ("endpoint", "endpoint"): ("endpoint_join", "collinear_extension"),
    ("boundary_point", "endpoint"): ("t_junction", "tangent"),
    ("circle_boundary", "endpoint"): ("tangent", "t_junction"),
    ("circle_center", "circle_center"): ("concentric",),
    ("region", "centroid"): ("containment",),
    ("region", "endpoint"): ("containment",),
}


def compatible_relations(host_type: str, donor_type: str) -> tuple[str, ...]:
    return COMPATIBILITY.get((host_type, donor_type), ())


def component_region(primitives: list[Primitive]):
    regions = []
    lines = []
    for primitive in primitives:
        if primitive.kind == "circle":
            center = primitive.geometry["center"]
            regions.append(
                Point(float(center[0]), float(center[1])).buffer(
                    float(primitive.geometry["radius"]), quad_segs=96
                )
            )
            continue
        points = sample_primitive(primitive, 128)
        if primitive.kind == "polyline" and primitive.geometry.get("closed", False):
            polygon = Polygon(points)
            if polygon.is_valid and polygon.area > 1e-8:
                regions.append(polygon)
        elif len(points) >= 2:
            lines.append(LineString(points))
    if lines:
        merged = unary_union(lines)
        regions.extend(poly for poly in polygonize(merged) if poly.area > 1e-8)
    if not regions:
        return None
    return max(regions, key=lambda region: region.area)


def detect_anchors(component_id: str, primitives: list[Primitive]) -> list[Anchor]:
    anchors: list[Anchor] = []
    serial = 0
    endpoint_records: list[tuple[np.ndarray, np.ndarray, Primitive, int]] = []
    for primitive in primitives:
        endpoints = primitive_endpoints(primitive)
        for endpoint_index, position in enumerate(endpoints):
            try:
                direction = primitive_tangent(primitive, endpoint_index)
            except ValueError:
                continue
            endpoint_records.append((position, direction, primitive, endpoint_index))

    # Only topological degree-one endpoints are attachment anchors. Shared
    # internal vertices remain corners/junctions but are not loose connectors.
    clusters: dict[tuple[int, int], list] = defaultdict(list)
    for record in endpoint_records:
        key = tuple(np.round(record[0] / 1e-5).astype(int))
        clusters[key].append(record)
    for records in clusters.values():
        if len(records) != 1:
            continue
        position, direction, primitive, endpoint_index = records[0]
        normal = np.array([-direction[1], direction[0]])
        anchors.append(
            Anchor(
                anchor_id=f"{component_id}_a{serial:03d}",
                component_id=component_id,
                type="endpoint",
                position=position.tolist(),
                direction=direction.tolist(),
                normal=normal.tolist(),
                primitive_ids=[primitive.primitive_id],
                local_scale=max(primitive_length(primitive), 1e-6),
                compatible_relations=["endpoint_join", "collinear_extension"],
                parameter=float(endpoint_index),
            )
        )
        serial += 1

    for primitive in primitives:
        if primitive.kind == "line":
            p0 = np.asarray(primitive.geometry["p0"], dtype=float)
            p1 = np.asarray(primitive.geometry["p1"], dtype=float)
            direction = p1 - p0
            length = float(np.linalg.norm(direction))
            if length > 1e-8:
                direction /= length
                anchors.append(
                    Anchor(
                        anchor_id=f"{component_id}_a{serial:03d}",
                        component_id=component_id,
                        type="boundary_point",
                        position=(0.5 * (p0 + p1)).tolist(),
                        direction=direction.tolist(),
                        normal=[float(-direction[1]), float(direction[0])],
                        primitive_ids=[primitive.primitive_id],
                        local_scale=length,
                        compatible_relations=["t_junction", "tangent", "shared_edge"],
                        parameter=0.5,
                    )
                )
                serial += 1
        if primitive.kind in {"circle", "arc"}:
            center = np.asarray(primitive.geometry["center"], dtype=float)
            radius = float(primitive.geometry["radius"])
            anchors.append(
                Anchor(
                    anchor_id=f"{component_id}_a{serial:03d}",
                    component_id=component_id,
                    type="circle_center",
                    position=center.tolist(),
                    direction=[1.0, 0.0],
                    normal=[0.0, 1.0],
                    primitive_ids=[primitive.primitive_id],
                    local_scale=radius,
                    compatible_relations=["concentric", "repetition"],
                )
            )
            serial += 1
            angle = math.pi * 0.25
            position = center + radius * np.array([math.cos(angle), math.sin(angle)])
            tangent = np.array([-math.sin(angle), math.cos(angle)])
            anchors.append(
                Anchor(
                    anchor_id=f"{component_id}_a{serial:03d}",
                    component_id=component_id,
                    type="circle_boundary",
                    position=position.tolist(),
                    direction=tangent.tolist(),
                    normal=(position - center).tolist(),
                    primitive_ids=[primitive.primitive_id],
                    local_scale=radius,
                    compatible_relations=["tangent", "t_junction"],
                    parameter=angle / (2.0 * math.pi),
                )
            )
            serial += 1

    region = component_region(primitives)
    if region is not None:
        centroid = np.asarray([region.centroid.x, region.centroid.y])
        anchors.append(
            Anchor(
                anchor_id=f"{component_id}_a{serial:03d}",
                component_id=component_id,
                type="region",
                position=centroid.tolist(),
                direction=[1.0, 0.0],
                normal=[0.0, 1.0],
                primitive_ids=[primitive.primitive_id for primitive in primitives],
                local_scale=math.sqrt(float(region.area)),
                compatible_relations=["containment", "insertion"],
            )
        )
        serial += 1

    all_points = np.concatenate(
        [sample_primitive(primitive, 64) for primitive in primitives], axis=0
    )
    centroid = all_points.mean(axis=0)
    anchors.append(
        Anchor(
            anchor_id=f"{component_id}_a{serial:03d}",
            component_id=component_id,
            type="centroid",
            position=centroid.tolist(),
            direction=[1.0, 0.0],
            normal=[0.0, 1.0],
            primitive_ids=[primitive.primitive_id for primitive in primitives],
            local_scale=float(np.max(np.ptp(all_points, axis=0))),
            compatible_relations=["containment"],
        )
    )
    return anchors
