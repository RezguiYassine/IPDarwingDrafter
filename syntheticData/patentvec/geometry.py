from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Iterable

import numpy as np

from .schema import Primitive


def similarity_matrix(
    scale: float,
    angle: float,
    translation: Iterable[float] = (0.0, 0.0),
    reflect: bool = False,
) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("similarity scale must be positive and finite")
    cosine, sine = math.cos(angle), math.sin(angle)
    reflection = -1.0 if reflect else 1.0
    linear = scale * np.array(
        [[cosine, -reflection * sine], [sine, reflection * cosine]],
        dtype=np.float64,
    )
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = np.asarray(tuple(translation), dtype=np.float64)
    return matrix


def compose_matrices(*matrices: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=np.float64)
    for matrix in matrices:
        result = np.asarray(matrix, dtype=np.float64) @ result
    return result


def apply_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    flat = array.reshape(-1, 2)
    homogeneous = np.column_stack([flat, np.ones(len(flat))])
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return transformed[:, :2].reshape(array.shape)


def _linear_scale(matrix: np.ndarray) -> float:
    linear = np.asarray(matrix, dtype=np.float64)[:2, :2]
    first = float(np.linalg.norm(linear[:, 0]))
    second = float(np.linalg.norm(linear[:, 1]))
    if not np.isclose(first, second, rtol=1e-6, atol=1e-9):
        raise ValueError("nonuniform transform cannot preserve analytic primitive")
    return 0.5 * (first + second)


def transform_primitive(
    primitive: Primitive,
    matrix: np.ndarray,
    primitive_id: str | None = None,
    component_id: str | None = None,
    transform_id: str | None = None,
) -> Primitive:
    geometry = copy.deepcopy(primitive.geometry)
    kind = primitive.kind
    if kind == "line":
        geometry["p0"] = apply_points(np.asarray(geometry["p0"]), matrix).tolist()
        geometry["p1"] = apply_points(np.asarray(geometry["p1"]), matrix).tolist()
    elif kind in {"polyline", "spline"}:
        geometry["points"] = apply_points(
            np.asarray(geometry["points"]), matrix
        ).tolist()
    elif kind in {"circle", "arc"}:
        center = np.asarray(geometry["center"], dtype=float)
        geometry["center"] = apply_points(center, matrix).tolist()
        geometry["radius"] = float(geometry["radius"]) * _linear_scale(matrix)
        if kind == "arc":
            linear = np.asarray(matrix, dtype=float)[:2, :2]
            for key in ("start_angle", "end_angle"):
                vector = np.array([math.cos(geometry[key]), math.sin(geometry[key])])
                transformed = linear @ vector
                geometry[key] = float(math.atan2(transformed[1], transformed[0]))
            if np.linalg.det(linear) < 0:
                geometry["ccw"] = not bool(geometry.get("ccw", True))
    elif kind in {"ellipse", "elliptical_arc"}:
        geometry["center"] = apply_points(
            np.asarray(geometry["center"]), matrix
        ).tolist()
        geometry["radii"] = (
            np.asarray(geometry["radii"], dtype=float) * _linear_scale(matrix)
        ).tolist()
        axis = np.array([
            math.cos(float(geometry.get("rotation", 0.0))),
            math.sin(float(geometry.get("rotation", 0.0))),
        ])
        axis = np.asarray(matrix, dtype=float)[:2, :2] @ axis
        geometry["rotation"] = float(math.atan2(axis[1], axis[0]))
    elif kind in {"quadratic_bezier", "cubic_bezier"}:
        names = ("p0", "p1", "p2") if kind == "quadratic_bezier" else (
            "p0", "p1", "p2", "p3"
        )
        for key in names:
            geometry[key] = apply_points(np.asarray(geometry[key]), matrix).tolist()
    elif kind == "point":
        geometry["position"] = apply_points(
            np.asarray(geometry["position"]), matrix
        ).tolist()
    elif kind == "text":
        geometry["position"] = apply_points(
            np.asarray(geometry["position"]), matrix
        ).tolist()
        geometry["size"] = float(geometry.get("size", 0.02)) * _linear_scale(matrix)
    return replace(
        primitive,
        primitive_id=primitive_id or primitive.primitive_id,
        component_id=component_id or primitive.component_id,
        transform_id=transform_id if transform_id is not None else primitive.transform_id,
        geometry=geometry,
        parent_primitive_ids=list(primitive.parent_primitive_ids),
        interaction_ids=list(primitive.interaction_ids),
        source_interval=list(primitive.source_interval),
        style=copy.deepcopy(primitive.style),
    )


def sample_primitive(primitive: Primitive, count: int = 128) -> np.ndarray:
    geometry = primitive.geometry
    kind = primitive.kind
    if kind == "line":
        return np.linspace(geometry["p0"], geometry["p1"], max(2, count))
    if kind in {"polyline", "spline"}:
        points = np.asarray(geometry["points"], dtype=np.float64)
        if geometry.get("closed", False) and not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        return resample_polyline(points, max(len(points), count))
    if kind == "circle":
        angles = np.linspace(0.0, 2.0 * math.pi, max(32, count), endpoint=False)
        center = np.asarray(geometry["center"], dtype=float)
        radius = float(geometry["radius"])
        return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    if kind == "arc":
        start = float(geometry["start_angle"])
        end = float(geometry["end_angle"])
        ccw = bool(geometry.get("ccw", True))
        if ccw:
            while end <= start:
                end += 2.0 * math.pi
        else:
            while end >= start:
                end -= 2.0 * math.pi
        angles = np.linspace(start, end, max(12, count))
        center = np.asarray(geometry["center"], dtype=float)
        radius = float(geometry["radius"])
        return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    if kind in {"quadratic_bezier", "cubic_bezier"}:
        t = np.linspace(0.0, 1.0, max(12, count))[:, None]
        mt = 1.0 - t
        p0 = np.asarray(geometry["p0"], dtype=float)
        p1 = np.asarray(geometry["p1"], dtype=float)
        p2 = np.asarray(geometry["p2"], dtype=float)
        if kind == "quadratic_bezier":
            return mt * mt * p0 + 2.0 * mt * t * p1 + t * t * p2
        p3 = np.asarray(geometry["p3"], dtype=float)
        return (
            mt**3 * p0
            + 3.0 * mt * mt * t * p1
            + 3.0 * mt * t * t * p2
            + t**3 * p3
        )
    if kind == "point":
        return np.asarray([geometry["position"]], dtype=float)
    if kind == "text":
        return np.asarray([geometry["position"]], dtype=float)
    raise ValueError(f"sampling is not implemented for {kind}")


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], max(2, count))
    out = np.empty((len(targets), 2), dtype=np.float64)
    out[:, 0] = np.interp(targets, cumulative, points[:, 0])
    out[:, 1] = np.interp(targets, cumulative, points[:, 1])
    return out


def primitive_endpoints(primitive: Primitive) -> list[np.ndarray]:
    if primitive.kind == "line":
        return [
            np.asarray(primitive.geometry["p0"], dtype=float),
            np.asarray(primitive.geometry["p1"], dtype=float),
        ]
    if primitive.kind in {"polyline", "spline"}:
        if primitive.geometry.get("closed", False):
            return []
        points = np.asarray(primitive.geometry["points"], dtype=float)
        return [points[0], points[-1]]
    if primitive.kind in {"arc", "quadratic_bezier", "cubic_bezier"}:
        points = sample_primitive(primitive, 24)
        return [points[0], points[-1]]
    return []


def primitive_tangent(primitive: Primitive, at_end: int = 0) -> np.ndarray:
    points = sample_primitive(primitive, 64)
    tangent = points[1] - points[0] if at_end == 0 else points[-2] - points[-1]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        raise ValueError("undefined primitive tangent")
    return tangent / norm


def primitive_length(primitive: Primitive) -> float:
    if primitive.kind == "circle":
        return 2.0 * math.pi * float(primitive.geometry["radius"])
    points = sample_primitive(primitive, 256)
    if primitive.kind == "circle":
        points = np.vstack([points, points[0]])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def primitive_bounds(primitives: Iterable[Primitive]) -> tuple[np.ndarray, np.ndarray]:
    samples = [sample_primitive(primitive, 128) for primitive in primitives]
    if not samples:
        raise ValueError("cannot bound empty primitive collection")
    points = np.concatenate(samples, axis=0)
    return points.min(axis=0), points.max(axis=0)


def local_normalization(primitives: Iterable[Primitive]) -> np.ndarray:
    lower, upper = primitive_bounds(primitives)
    span = float(np.max(upper - lower))
    if span <= 1e-12:
        raise ValueError("degenerate component bounds")
    center = 0.5 * (lower + upper)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] *= 1.0 / span
    matrix[:2, 2] = -center / span
    return matrix


def split_line(
    primitive: Primitive,
    parameter: float,
    first_id: str,
    second_id: str,
    interaction_id: str,
) -> tuple[Primitive, Primitive, np.ndarray]:
    if primitive.kind != "line" or not 1e-6 < parameter < 1.0 - 1e-6:
        raise ValueError("line split requires an interior parameter")
    p0 = np.asarray(primitive.geometry["p0"], dtype=float)
    p1 = np.asarray(primitive.geometry["p1"], dtype=float)
    point = (1.0 - parameter) * p0 + parameter * p1
    lo, hi = primitive.source_interval
    middle = lo + parameter * (hi - lo)
    parents = list(dict.fromkeys([*primitive.parent_primitive_ids, primitive.primitive_id]))
    interactions = list(dict.fromkeys([*primitive.interaction_ids, interaction_id]))
    first = replace(
        primitive,
        primitive_id=first_id,
        geometry={"p0": p0.tolist(), "p1": point.tolist()},
        parent_primitive_ids=parents,
        interaction_ids=interactions,
        source_interval=[lo, middle],
    )
    second = replace(
        primitive,
        primitive_id=second_id,
        geometry={"p0": point.tolist(), "p1": p1.tolist()},
        parent_primitive_ids=parents,
        interaction_ids=interactions,
        source_interval=[middle, hi],
    )
    return first, second, point


def line_line_intersection(
    a0: Iterable[float],
    a1: Iterable[float],
    b0: Iterable[float],
    b1: Iterable[float],
) -> tuple[np.ndarray, float, float] | None:
    p = np.asarray(tuple(a0), dtype=float)
    r = np.asarray(tuple(a1), dtype=float) - p
    q = np.asarray(tuple(b0), dtype=float)
    s = np.asarray(tuple(b1), dtype=float) - q
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) <= 1e-12:
        return None
    qp = q - p
    t = float((qp[0] * s[1] - qp[1] * s[0]) / cross)
    u = float((qp[0] * r[1] - qp[1] * r[0]) / cross)
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return p + t * r, t, u
    return None
