from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import (
    compose_matrices,
    local_normalization,
    primitive_endpoints,
    transform_primitive,
)
from .schema import Primitive, SourceRecord


@dataclass
class SourceComponent:
    component_key: str
    source: SourceRecord
    primitives: list[Primitive]
    source_to_component: np.ndarray
    descriptor: dict


def _component_descriptor(primitives: list[Primitive]) -> dict:
    kinds: dict[str, int] = {}
    endpoints = 0
    for primitive in primitives:
        kinds[primitive.kind] = kinds.get(primitive.kind, 0) + 1
        endpoints += len(primitive_endpoints(primitive))
    return {
        "primitive_count": len(primitives),
        "primitive_types": kinds,
        "endpoint_anchor_count": endpoints,
        "closed_count": sum(
            primitive.kind == "circle"
            or (primitive.kind == "polyline" and primitive.geometry.get("closed", False))
            for primitive in primitives
        ),
        "circle_count": kinds.get("circle", 0),
        "arc_count": kinds.get("arc", 0),
        "complexity": float(len(primitives) + 0.5 * endpoints),
    }


def _normalise_component(
    primitives: list[Primitive], component_key: str, source: SourceRecord
) -> SourceComponent:
    matrix = local_normalization(primitives)
    normalized: list[Primitive] = []
    for index, primitive in enumerate(primitives):
        normalized.append(
            transform_primitive(
                primitive,
                matrix,
                primitive_id=f"{component_key}_p{index:03d}",
                component_id=component_key,
            )
        )
    return SourceComponent(
        component_key=component_key,
        source=source,
        primitives=normalized,
        source_to_component=matrix,
        descriptor=_component_descriptor(normalized),
    )


def _connected_groups(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """Return stable components regardless of set insertion or hash order."""
    groups: list[list[str]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        stack = [root]
        group = []
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            group.append(current)
            neighbors = sorted(adjacency[current] & remaining, reverse=True)
            stack.extend(neighbors)
        groups.append(sorted(group))
    return groups


def procedural_polyline_component(
    sample_id: str,
    split: str,
    seed: int,
    primitive_count: int = 2,
) -> SourceComponent:
    """Build a deterministic shallow-chain motif with exact polyline labels."""
    if not 1 <= primitive_count <= 6:
        raise ValueError("procedural polyline primitive_count must be in [1, 6]")
    rng = np.random.default_rng(seed)
    dataset = "PatentVec-Procedural"
    source_sample_id = f"{sample_id}_poly_{seed}"
    component_key = f"proc_poly_{seed}"
    source = SourceRecord(
        dataset=dataset,
        sample_id=source_sample_id,
        split=split,
        license_id="PatentVec-Apache-2.0",
    )
    primitives: list[Primitive] = []
    current = np.zeros(2, dtype=float)
    heading = float(rng.uniform(-math.pi, math.pi))
    for primitive_index in range(primitive_count):
        if primitive_index:
            heading += math.radians(
                float(rng.choice([-1.0, 1.0]) * rng.uniform(48.0, 72.0))
            )
        turn_sign = float(rng.choice([-1.0, 1.0]))
        points = [current.copy()]
        segment_count = int(rng.integers(3, 6))
        for segment_index in range(segment_count):
            if segment_index:
                heading += math.radians(
                    turn_sign * float(rng.uniform(12.0, 20.0))
                )
            length = float(rng.uniform(0.14, 0.24))
            current = current + length * np.array(
                [math.cos(heading), math.sin(heading)], dtype=float
            )
            points.append(current.copy())
        primitive_id = f"proc_poly_{seed}_{primitive_index:02d}"
        primitives.append(
            Primitive(
                primitive_id=primitive_id,
                kind="polyline",
                geometry={"points": np.asarray(points).tolist(), "closed": False},
                semantic="object_visible",
                component_id=component_key,
                source_dataset=dataset,
                source_sample_id=source_sample_id,
                source_component_id=component_key,
                source_primitive_id=primitive_id,
                generated_by="procedural_polyline_motif",
            )
        )
    component = _normalise_component(primitives, component_key, source)
    component.descriptor.update(
        {
            "procedural": True,
            "motif": "shallow_polyline_chain",
            "polyline_count": primitive_count,
            "internal_turn_degrees": [12.0, 20.0],
            "boundary_turn_degrees": [48.0, 72.0],
        }
    )
    return component


class SketchGraphsAdapter:
    def __init__(self, raw_path: Path, split: str = "train"):
        self.raw_path = Path(raw_path)
        self.split = split
        self._data = None

    def _open(self):
        if self._data is None:
            from sketchgraphs.data import flat_array

            self._data = flat_array.load_dictionary_flat(str(self.raw_path))
        return self._data

    def __len__(self) -> int:
        return len(self._open()["sequences"])

    @staticmethod
    def _arc_angles(entity) -> tuple[float, float]:
        base = math.atan2(entity.yDir, entity.xDir)
        if entity.clockwise:
            start = base - entity.endParam
            end = base - entity.startParam
        else:
            start = base + entity.startParam
            end = base + entity.endParam
        while end <= start:
            end += 2.0 * math.pi
        return start, end

    @lru_cache(maxsize=128)
    def load_source_sample(self, source_id: int) -> list[SourceComponent]:
        from sketchgraphs.data import Arc, Circle, Line, sketch_from_sequence

        sketch = sketch_from_sequence(self._open()["sequences"][int(source_id)])
        source = SourceRecord(
            dataset="SketchGraphs",
            sample_id=str(source_id),
            split=self.split,
            license_id="SketchGraphs-data-Onshape-terms",
        )
        primitives: dict[str, Primitive] = {}
        for entity_id, entity in sketch.entities.items():
            if getattr(entity, "isConstruction", False):
                continue
            geometry = None
            kind = None
            if isinstance(entity, Line):
                kind = "line"
                geometry = {
                    "p0": np.asarray(entity.start_point, dtype=float).tolist(),
                    "p1": np.asarray(entity.end_point, dtype=float).tolist(),
                }
            elif isinstance(entity, Arc):
                kind = "arc"
                start, end = self._arc_angles(entity)
                geometry = {
                    "center": [float(entity.xCenter), float(entity.yCenter)],
                    "radius": float(entity.radius),
                    "start_angle": float(start),
                    "end_angle": float(end),
                    "ccw": True,
                }
            elif isinstance(entity, Circle):
                kind = "circle"
                geometry = {
                    "center": [float(entity.xCenter), float(entity.yCenter)],
                    "radius": float(entity.radius),
                }
            if kind is None:
                continue
            key = str(entity_id)
            primitives[key] = Primitive(
                primitive_id=f"sg_{source_id}_{key}",
                kind=kind,
                geometry=geometry,
                semantic="object_visible",
                component_id="source",
                source_dataset=source.dataset,
                source_sample_id=source.sample_id,
                source_component_id="constraint_graph",
                source_primitive_id=key,
                generated_by="source_transform",
            )
        if not primitives:
            return []

        adjacency = {key: set() for key in primitives}
        for constraint in sketch.constraints.values():
            references = {
                str(parameter.referenceMain)
                for parameter in constraint.parameters
                if getattr(parameter, "referenceMain", None) is not None
                and str(parameter.referenceMain) in primitives
            }
            for first in references:
                adjacency[first].update(references - {first})

        # Coincident geometry occasionally lacks an explicit retained constraint.
        keys = list(primitives)
        endpoint_sets = {
            key: primitive_endpoints(primitives[key]) for key in keys
        }
        for i, first in enumerate(keys):
            for second in keys[i + 1 :]:
                if any(
                    np.linalg.norm(a - b) <= 1e-7
                    for a in endpoint_sets[first]
                    for b in endpoint_sets[second]
                ):
                    adjacency[first].add(second)
                    adjacency[second].add(first)

        groups = _connected_groups(adjacency)

        # Source Y-up is reflected into the composite Y-down frame explicitly.
        reflection = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        output: list[SourceComponent] = []
        for group_index, group in enumerate(groups):
            selected = [primitives[key] for key in group]
            if not selected:
                continue
            reflected = [transform_primitive(item, reflection) for item in selected]
            key = f"sg_{source_id}_c{group_index:02d}"
            component = _normalise_component(reflected, key, source)
            component.source_to_component = compose_matrices(
                reflection, component.source_to_component
            )
            output.append(component)
        return output

    def validate_source_sample(self, source_id: int) -> dict:
        try:
            components = self.load_source_sample(source_id)
            return {
                "valid": bool(components),
                "component_count": len(components),
                "primitive_count": sum(len(c.primitives) for c in components),
            }
        except Exception as exc:
            return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN_RE = re.compile(rf"[MLCZ]|{_NUM}")
_PATH_RE = re.compile(r"<path\b[^>]*\bd=\"([^\"]*)\"", re.IGNORECASE)
_VIEWBOX_RE = re.compile(
    rf"viewBox=\"\s*({_NUM})[\s,]+({_NUM})[\s,]+({_NUM})[\s,]+({_NUM})\s*\""
)


class CADVGDrawingAdapter:
    VIEWS = ("Front", "Top", "Right", "FrontTopRight")

    def __init__(self, root: Path, split: str = "train"):
        self.root = Path(root)
        self.split = split
        split_file = self.root / "train_val_test_split.json"
        split_data = json.loads(split_file.read_text())
        self.sample_ids = list(split_data[split])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def svg_path(self, source_id: str, view: str) -> Path:
        outer, inner = source_id.split("/")
        return self.root / "svg_raw" / outer / inner / f"{inner}_{view}.svg"

    @lru_cache(maxsize=128)
    def load_source_sample(
        self, source_id: str, view: str = "Front"
    ) -> list[SourceComponent]:
        path = self.svg_path(source_id, view)
        text = path.read_text()
        viewbox = _VIEWBOX_RE.search(text)
        min_x, min_y, width, height = (
            tuple(map(float, viewbox.groups())) if viewbox else (0.0, 0.0, 200.0, 200.0)
        )
        source = SourceRecord(
            dataset="CAD-VGDrawing",
            sample_id=source_id,
            split=self.split,
            view=view,
            license_id="Drawing2CAD-CAD-VGDrawing-research",
        )
        components: list[SourceComponent] = []
        primitive_serial = 0
        for path_index, d_attr in enumerate(_PATH_RE.findall(text)):
            tokens = _TOKEN_RE.findall(d_attr)
            command = None
            current = None
            start = None
            primitives: list[Primitive] = []
            index = 0
            while index < len(tokens):
                if tokens[index] in {"M", "L", "C", "Z"}:
                    command = tokens[index]
                    index += 1
                if command == "M":
                    if index + 1 >= len(tokens):
                        break
                    current = np.array(
                        [float(tokens[index]), float(tokens[index + 1])], dtype=float
                    )
                    index += 2
                    start = current.copy()
                    command = "L"
                    continue
                if command == "L" and current is not None:
                    if index + 1 >= len(tokens):
                        break
                    end = np.array(
                        [float(tokens[index]), float(tokens[index + 1])], dtype=float
                    )
                    index += 2
                    if np.linalg.norm(end - current) > 1e-10:
                        primitives.append(
                            self._primitive(
                                source, path_index, primitive_serial, "line",
                                {"p0": current.tolist(), "p1": end.tolist()},
                            )
                        )
                        primitive_serial += 1
                    current = end
                    continue
                if command == "C" and current is not None:
                    if index + 5 >= len(tokens):
                        break
                    controls = np.asarray(
                        [float(value) for value in tokens[index : index + 6]],
                        dtype=float,
                    ).reshape(3, 2)
                    index += 6
                    primitives.append(
                        self._primitive(
                            source,
                            path_index,
                            primitive_serial,
                            "cubic_bezier",
                            {
                                "p0": current.tolist(),
                                "p1": controls[0].tolist(),
                                "p2": controls[1].tolist(),
                                "p3": controls[2].tolist(),
                            },
                        )
                    )
                    primitive_serial += 1
                    current = controls[2]
                    continue
                if command == "Z":
                    if (
                        current is not None
                        and start is not None
                        and np.linalg.norm(current - start) > 1e-10
                    ):
                        primitives.append(
                            self._primitive(
                                source,
                                path_index,
                                primitive_serial,
                                "line",
                                {"p0": current.tolist(), "p1": start.tolist()},
                            )
                        )
                        primitive_serial += 1
                        current = start.copy()
                    command = None
                    continue
                index += 1
            if not primitives:
                continue
            # Preserve SVG viewBox coordinates as the source frame, then create a
            # component-local, centered similarity frame for composition.
            offset = np.eye(3, dtype=float)
            offset[:2, 2] = [-min_x, -min_y]
            view_scale = 1.0 / max(width, height)
            view_matrix = np.diag([view_scale, view_scale, 1.0]) @ offset
            view_primitives = [transform_primitive(item, view_matrix) for item in primitives]
            key = f"cadvg_{source_id.replace('/', '_')}_{view}_p{path_index:03d}"
            component = _normalise_component(view_primitives, key, source)
            component.source_to_component = compose_matrices(
                view_matrix, component.source_to_component
            )
            components.append(component)
        return components

    @staticmethod
    def _primitive(
        source: SourceRecord,
        path_index: int,
        serial: int,
        kind: str,
        geometry: dict,
    ) -> Primitive:
        source_id = f"path_{path_index:03d}_segment_{serial:04d}"
        return Primitive(
            primitive_id=f"cadvg_{serial:04d}",
            kind=kind,
            geometry=geometry,
            semantic="object_visible",
            component_id="source",
            source_dataset=source.dataset,
            source_sample_id=source.sample_id,
            source_component_id=f"path_{path_index:03d}",
            source_primitive_id=source_id,
            generated_by="source_transform",
        )

    def list_source_samples(self, split: str | None = None) -> Iterable[str]:
        if split is not None and split != self.split:
            return []
        return iter(self.sample_ids)

    def validate_source_sample(self, source_id: str, view: str = "Front") -> dict:
        try:
            components = self.load_source_sample(source_id, view)
            return {
                "valid": bool(components),
                "component_count": len(components),
                "primitive_count": sum(len(c.primitives) for c in components),
            }
        except Exception as exc:
            return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
