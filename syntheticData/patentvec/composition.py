from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
from shapely.geometry import LineString, Point

from .anchors import component_region, detect_anchors
from .geometry import (
    compose_matrices,
    primitive_bounds,
    primitive_endpoints,
    primitive_length,
    primitive_tangent,
    sample_primitive,
    split_line,
    transform_primitive,
)
from .schema import (
    Anchor,
    CanonicalDrawing,
    Component,
    Interaction,
    Junction,
    SourceRecord,
    TransformRecord,
)
from .sources import SourceComponent


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def _alignment_matrix(
    source_position: np.ndarray,
    source_direction: np.ndarray,
    target_position: np.ndarray,
    target_direction: np.ndarray,
    scale: float,
    reflect: bool = False,
) -> np.ndarray:
    source_direction = _unit(source_direction)
    target_direction = _unit(target_direction)
    source_angle = math.atan2(source_direction[1], source_direction[0])
    target_angle = math.atan2(target_direction[1], target_direction[0])
    angle = target_angle - source_angle
    cosine, sine = math.cos(angle), math.sin(angle)
    reflection = -1.0 if reflect else 1.0
    linear = scale * np.array(
        [[cosine, -reflection * sine], [sine, reflection * cosine]], dtype=float
    )
    translation = np.asarray(target_position, dtype=float) - linear @ np.asarray(
        source_position, dtype=float
    )
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return matrix


def _intersection_points(geometry) -> tuple[list[np.ndarray], bool]:
    if geometry.is_empty:
        return [], False
    if geometry.geom_type == "Point":
        return [np.asarray(geometry.coords[0], dtype=float)], False
    if geometry.geom_type == "MultiPoint":
        return [
            np.asarray(item.coords[0], dtype=float) for item in geometry.geoms
        ], False
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [], geometry.length > 1e-7
    if hasattr(geometry, "geoms"):
        points = []
        overlap = False
        for item in geometry.geoms:
            item_points, item_overlap = _intersection_points(item)
            points.extend(item_points)
            overlap = overlap or item_overlap
        return points, overlap
    return [], False


class CompositionBuilder:
    """Build one split-safe Track-A composite with exact interaction geometry."""

    def __init__(
        self,
        sample_id: str,
        seed: int,
        split: str = "train",
        difficulty: str = "easy",
        canvas: tuple[int, int] = (1024, 1024),
    ):
        self.sample_id = sample_id
        self.seed = int(seed)
        self.split = split
        self.difficulty = difficulty
        self.canvas = list(canvas)
        self.rng = np.random.default_rng(seed)
        self.sources: list[SourceRecord] = []
        self.components: list[Component] = []
        self.transforms: list[TransformRecord] = []
        self.anchors: list[Anchor] = []
        self.interactions: list[Interaction] = []
        self.visible = []
        self.amodal = []
        self.junctions_visible: list[Junction] = []
        self.junctions_amodal: list[Junction] = []
        self._component_visible: dict[str, list] = {}
        self._component_amodal: dict[str, list] = {}

    def _next_component_id(self) -> str:
        return f"c{len(self.components):03d}"

    def _next_interaction_id(self) -> str:
        return f"i{len(self.interactions):03d}"

    def _record_source(self, source: SourceRecord) -> None:
        if source.split != self.split:
            raise ValueError(
                f"split leakage: source is {source.split}, composite is {self.split}"
            )
        key = (source.dataset, source.sample_id, source.split, source.view)
        existing = {
            (item.dataset, item.sample_id, item.split, item.view) for item in self.sources
        }
        if key not in existing:
            self.sources.append(copy.deepcopy(source))

    def _install_component(
        self, source: SourceComponent, placement: np.ndarray
    ) -> tuple[str, list]:
        self._record_source(source.source)
        component_id = self._next_component_id()
        source_transform_id = f"{component_id}_source_to_component"
        placement_transform_id = f"{component_id}_component_to_composite"
        complete_transform_id = f"{component_id}_source_to_composite"
        complete = compose_matrices(source.source_to_component, placement)
        self.transforms.extend(
            [
                TransformRecord(
                    transform_id=source_transform_id,
                    matrix=np.asarray(source.source_to_component).tolist(),
                    source_frame=f"{source.source.dataset}:{source.source.sample_id}",
                    target_frame=f"{component_id}:local",
                    reflection=float(np.linalg.det(source.source_to_component[:2, :2])) < 0,
                ),
                TransformRecord(
                    transform_id=placement_transform_id,
                    matrix=np.asarray(placement).tolist(),
                    source_frame=f"{component_id}:local",
                    target_frame="composite_normalized",
                    reflection=float(np.linalg.det(placement[:2, :2])) < 0,
                ),
                TransformRecord(
                    transform_id=complete_transform_id,
                    matrix=np.asarray(complete).tolist(),
                    source_frame=f"{source.source.dataset}:{source.source.sample_id}",
                    target_frame="composite_normalized",
                    reflection=float(np.linalg.det(complete[:2, :2])) < 0,
                ),
            ]
        )

        placed = []
        for index, primitive in enumerate(source.primitives):
            primitive_id = f"{self.sample_id}_{component_id}_p{index:03d}"
            item = transform_primitive(
                primitive,
                placement,
                primitive_id=primitive_id,
                component_id=component_id,
                transform_id=complete_transform_id,
            )
            item = replace(
                item,
                source_component_id=source.component_key,
                parent_primitive_ids=[primitive.primitive_id],
            )
            placed.append(item)

        self.components.append(
            Component(
                component_id=component_id,
                source_dataset=source.source.dataset,
                source_sample_id=source.source.sample_id,
                source_component_id=source.component_key,
                source_split=source.source.split,
                primitive_ids=[item.primitive_id for item in placed],
                transform_id=placement_transform_id,
                descriptor=copy.deepcopy(source.descriptor),
            )
        )
        self.visible.extend(placed)
        amodal = copy.deepcopy(placed)
        self.amodal.extend(amodal)
        self._component_visible[component_id] = placed
        self._component_amodal[component_id] = amodal
        return component_id, placed

    def add_base(
        self,
        source: SourceComponent,
        scale: float = 0.58,
        center: tuple[float, float] = (0.50, 0.50),
        angle: float = 0.0,
    ) -> str:
        if self.components:
            raise ValueError("a base component has already been installed")
        local_center = 0.5 * sum(primitive_bounds(source.primitives))
        cosine, sine = math.cos(angle), math.sin(angle)
        linear = scale * np.array([[cosine, -sine], [sine, cosine]], dtype=float)
        placement = np.eye(3, dtype=float)
        placement[:2, :2] = linear
        placement[:2, 2] = np.asarray(center) - linear @ local_center
        component_id, _ = self._install_component(source, placement)
        return component_id

    def _anchors_for(self, component_id: str, amodal: bool = False) -> list[Anchor]:
        primitives = (
            self._component_amodal[component_id]
            if amodal
            else self._component_visible[component_id]
        )
        return detect_anchors(component_id, primitives)

    def _remember_anchors(self, *anchors: Anchor) -> None:
        existing = {item.anchor_id for item in self.anchors}
        for anchor in anchors:
            if anchor.anchor_id not in existing:
                self.anchors.append(copy.deepcopy(anchor))
                existing.add(anchor.anchor_id)

    def _contact_positions(self) -> list[np.ndarray]:
        return [
            np.asarray(position, dtype=float)
            for interaction in self.interactions
            for position in interaction.intended_contacts
        ]

    def _is_clear_contact(self, position: np.ndarray, clearance: float = 0.025) -> bool:
        return all(
            np.linalg.norm(np.asarray(position, dtype=float) - existing) >= clearance
            for existing in self._contact_positions()
        )

    def component_ids(self) -> list[str]:
        return [component.component_id for component in self.components]

    def components_with(self, capability: str) -> list[str]:
        output = []
        for component_id in self.component_ids():
            primitives = self._component_visible[component_id]
            if capability == "line" and any(item.kind == "line" for item in primitives):
                output.append(component_id)
            elif capability == "long_line" and any(
                item.kind == "line" and primitive_length(item) > 0.045
                for item in primitives
            ):
                output.append(component_id)
            elif capability == "circle" and any(
                item.kind in {"circle", "arc"} for item in primitives
            ):
                output.append(component_id)
            elif capability == "region" and component_region(primitives) is not None:
                output.append(component_id)
            elif capability == "endpoint" and any(
                anchor.type == "endpoint"
                and self._is_clear_contact(np.asarray(anchor.position))
                for anchor in self._anchors_for(component_id)
            ):
                output.append(component_id)
        return output

    def endpoint_join(
        self,
        host_component_id: str,
        donor: SourceComponent,
        scale: float = 0.30,
        turn_degrees: float | None = None,
    ) -> str:
        host_candidates = [
            anchor
            for anchor in self._anchors_for(host_component_id)
            if anchor.type == "endpoint"
            and self._is_clear_contact(np.asarray(anchor.position))
        ]
        donor_candidates = [
            anchor for anchor in detect_anchors("donor", donor.primitives)
            if anchor.type == "endpoint"
        ]
        if not host_candidates or not donor_candidates:
            raise ValueError("endpoint_join requires a loose endpoint on both components")
        host = host_candidates[int(self.rng.integers(len(host_candidates)))]
        donor_anchor = donor_candidates[int(self.rng.integers(len(donor_candidates)))]
        turn = turn_degrees
        if turn is None:
            turn = float(self.rng.choice([0.0, 45.0, 90.0, -45.0, -90.0]))
        host_inward = _unit(np.asarray(host.direction))
        angle = math.radians(turn)
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        donor_target_direction = rotation @ (-host_inward)
        placement = _alignment_matrix(
            np.asarray(donor_anchor.position),
            np.asarray(donor_anchor.direction),
            np.asarray(host.position),
            donor_target_direction,
            scale,
        )
        donor_component_id, donor_primitives = self._install_component(donor, placement)
        donor_anchors = [
            anchor for anchor in self._anchors_for(donor_component_id)
            if anchor.type == "endpoint"
        ]
        placed_donor_anchor = min(
            donor_anchors,
            key=lambda item: np.linalg.norm(np.asarray(item.position) - np.asarray(host.position)),
        )
        residual = float(
            np.linalg.norm(
                np.asarray(placed_donor_anchor.position) - np.asarray(host.position)
            )
        )
        interaction_id = self._next_interaction_id()
        host_primitive_id = host.primitive_ids[0]
        donor_primitive_id = placed_donor_anchor.primitive_ids[0]
        amodal_primitive_ids = self._amodal_primitive_ids(
            [host_primitive_id, donor_primitive_id]
        )
        self._add_interaction_to_primitives(
            interaction_id,
            [host_primitive_id, donor_primitive_id, *amodal_primitive_ids],
        )
        self.interactions.append(
            Interaction(
                interaction_id=interaction_id,
                type="endpoint_join",
                host_component_id=host_component_id,
                donor_component_id=donor_component_id,
                host_anchor_id=host.anchor_id,
                donor_anchor_id=placed_donor_anchor.anchor_id,
                residual=residual,
                intended_contacts=[host.position],
                metadata={"turn_degrees": turn},
            )
        )
        junction_type = "smooth_continuation" if abs(turn) < 1e-6 else "sharp_corner"
        junction = Junction(
            junction_id=f"j{len(self.junctions_visible):03d}",
            type=junction_type,
            position=list(host.position),
            primitive_ids=[host_primitive_id, donor_primitive_id],
            component_ids=[host_component_id, donor_component_id],
            interaction_id=interaction_id,
        )
        self.junctions_visible.append(junction)
        self.junctions_amodal.append(
            replace(junction, primitive_ids=amodal_primitive_ids)
        )
        self._remember_anchors(host, placed_donor_anchor)
        return donor_component_id

    def t_junction(
        self,
        host_component_id: str,
        donor: SourceComponent,
        scale: float = 0.28,
        parameter: float | None = None,
    ) -> str:
        host_lines = [
            primitive for primitive in self._component_visible[host_component_id]
            if primitive.kind == "line" and primitive_length(primitive) > 0.045
        ]
        if not host_lines:
            raise ValueError("t_junction requires a line host")
        host_line = max(host_lines, key=primitive_length)
        donor_candidates = [
            anchor for anchor in detect_anchors("donor", donor.primitives)
            if anchor.type == "endpoint"
        ]
        if not donor_candidates:
            raise ValueError("t_junction requires a loose donor endpoint")
        if len(donor_candidates) == 1:
            donor_anchor = donor_candidates[0]
        else:
            separation = {
                anchor.anchor_id: max(
                    np.linalg.norm(
                        np.asarray(anchor.position, dtype=float)
                        - np.asarray(other.position, dtype=float)
                    )
                    for other in donor_candidates
                    if other.anchor_id != anchor.anchor_id
                )
                for anchor in donor_candidates
            }
            maximum = max(separation.values())
            reusable = [
                anchor
                for anchor in donor_candidates
                if separation[anchor.anchor_id] >= maximum - 1e-9
            ]
            donor_anchor = reusable[int(self.rng.integers(len(reusable)))]
        u = float(parameter if parameter is not None else self.rng.uniform(0.25, 0.75))
        p0 = np.asarray(host_line.geometry["p0"], dtype=float)
        p1 = np.asarray(host_line.geometry["p1"], dtype=float)
        target = (1.0 - u) * p0 + u * p1
        if parameter is None:
            candidates = []
            for candidate_line in sorted(host_lines, key=primitive_length, reverse=True):
                c0 = np.asarray(candidate_line.geometry["p0"], dtype=float)
                c1 = np.asarray(candidate_line.geometry["p1"], dtype=float)
                for _ in range(5):
                    candidate_u = float(self.rng.uniform(0.25, 0.75))
                    candidate_target = (1.0 - candidate_u) * c0 + candidate_u * c1
                    if self._is_clear_contact(candidate_target, clearance=0.035):
                        candidates.append(
                            (candidate_line, candidate_u, candidate_target, c0, c1)
                        )
            if not candidates:
                raise ValueError("t_junction has no clear host-line interval")
            host_line, u, target, p0, p1 = candidates[
                int(self.rng.integers(len(candidates)))
            ]
        host_direction = _unit(p1 - p0)
        normal = np.array([-host_direction[1], host_direction[0]])
        region = component_region(self._component_visible[host_component_id])
        if region is not None:
            if region.buffer(1e-8).covers(Point(*(target + normal * 0.008))):
                normal *= -1.0
        elif bool(self.rng.integers(2)):
            normal *= -1.0
        placement = _alignment_matrix(
            np.asarray(donor_anchor.position),
            np.asarray(donor_anchor.direction),
            target,
            normal,
            scale,
        )
        donor_component_id, _ = self._install_component(donor, placement)
        placed_donor_anchor = min(
            [a for a in self._anchors_for(donor_component_id) if a.type == "endpoint"],
            key=lambda item: np.linalg.norm(np.asarray(item.position) - target),
        )
        interaction_id = self._next_interaction_id()
        amodal_host_id = self._amodal_primitive_id(host_line)
        first_id = f"{host_line.primitive_id}_s0"
        second_id = f"{host_line.primitive_id}_s1"
        first, second, contact = split_line(
            host_line, u, first_id, second_id, interaction_id
        )
        self._replace_visible_primitive(host_component_id, host_line, [first, second])
        donor_primitive_id = placed_donor_anchor.primitive_ids[0]
        self._add_interaction_to_primitives(
            interaction_id,
            [host_line.primitive_id, amodal_host_id, donor_primitive_id],
        )
        residual = float(np.linalg.norm(np.asarray(placed_donor_anchor.position) - contact))
        host_anchor = Anchor(
            anchor_id=f"{host_component_id}_interaction_{interaction_id}",
            component_id=host_component_id,
            type="boundary_point",
            position=contact.tolist(),
            direction=host_direction.tolist(),
            normal=normal.tolist(),
            primitive_ids=[first_id, second_id],
            local_scale=primitive_length(host_line),
            compatible_relations=["t_junction"],
            parameter=u,
        )
        self.interactions.append(
            Interaction(
                interaction_id=interaction_id,
                type="t_junction",
                host_component_id=host_component_id,
                donor_component_id=donor_component_id,
                host_anchor_id=host_anchor.anchor_id,
                donor_anchor_id=placed_donor_anchor.anchor_id,
                residual=residual,
                intended_contacts=[contact.tolist()],
                metadata={
                    "host_parent_primitive_id": host_line.primitive_id,
                    "host_parameter": u,
                    "visible_children": [first_id, second_id],
                },
            )
        )
        junction = Junction(
            junction_id=f"j{len(self.junctions_visible):03d}",
            type="t_junction",
            position=contact.tolist(),
            primitive_ids=[first_id, second_id, donor_primitive_id],
            component_ids=[host_component_id, donor_component_id],
            interaction_id=interaction_id,
        )
        self.junctions_visible.append(junction)
        # The amodal host remains analytically complete and unsplit.
        self.junctions_amodal.append(
            replace(junction, primitive_ids=[amodal_host_id, donor_primitive_id])
        )
        self._remember_anchors(host_anchor, placed_donor_anchor)
        return donor_component_id

    def containment(
        self,
        host_component_id: str,
        donor: SourceComponent,
        fill_fraction: float = 0.34,
    ) -> str:
        region = component_region(self._component_visible[host_component_id])
        if region is None:
            raise ValueError("containment requires a closed host region")
        donor_anchors = detect_anchors("donor", donor.primitives)
        donor_centroid = next(anchor for anchor in donor_anchors if anchor.type == "centroid")
        host_center = np.asarray([region.centroid.x, region.centroid.y])
        region_scale = min(region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1])
        scale = max(0.04, fill_fraction * region_scale)
        placement = _alignment_matrix(
            np.asarray(donor_centroid.position),
            np.array([1.0, 0.0]),
            host_center,
            np.array([1.0, 0.0]),
            scale,
        )
        donor_component_id, _ = self._install_component(donor, placement)
        placed_centroid = next(
            anchor for anchor in self._anchors_for(donor_component_id) if anchor.type == "centroid"
        )
        interaction_id = self._next_interaction_id()
        host_anchor = Anchor(
            anchor_id=f"{host_component_id}_interaction_{interaction_id}",
            component_id=host_component_id,
            type="region",
            position=host_center.tolist(),
            direction=[1.0, 0.0],
            normal=[0.0, 1.0],
            primitive_ids=[item.primitive_id for item in self._component_visible[host_component_id]],
            local_scale=float(math.sqrt(region.area)),
            compatible_relations=["containment"],
        )
        donor_points = [
            point
            for primitive in self._component_visible[donor_component_id]
            for point in sample_primitive(primitive, 96)
        ]
        inside_fraction = (
            float(np.mean([region.buffer(1e-7).covers(Point(*p)) for p in donor_points]))
            if donor_points
            else 1.0
        )
        residual = 1.0 - inside_fraction
        self.interactions.append(
            Interaction(
                interaction_id=interaction_id,
                type="containment",
                host_component_id=host_component_id,
                donor_component_id=donor_component_id,
                host_anchor_id=host_anchor.anchor_id,
                donor_anchor_id=placed_centroid.anchor_id,
                residual=residual,
                intended_contacts=[],
                metadata={"inside_fraction": inside_fraction},
            )
        )
        self._add_interaction_to_primitives(
            interaction_id,
            [item.primitive_id for item in self._component_visible[donor_component_id]],
        )
        self._remember_anchors(host_anchor, placed_centroid)
        return donor_component_id

    def concentric(
        self,
        host_component_id: str,
        donor: SourceComponent,
        radius_ratio: float = 0.55,
    ) -> str:
        host_circles = [
            item for item in self._component_visible[host_component_id] if item.kind == "circle"
        ]
        donor_circles = [item for item in donor.primitives if item.kind == "circle"]
        if not host_circles or not donor_circles:
            raise ValueError("concentric requires circles in host and donor")
        host_circle = max(host_circles, key=lambda item: item.geometry["radius"])
        donor_circle = max(donor_circles, key=lambda item: item.geometry["radius"])
        host_radius = float(host_circle.geometry["radius"])
        donor_radius = float(donor_circle.geometry["radius"])
        scale = radius_ratio * host_radius / donor_radius
        target = np.asarray(host_circle.geometry["center"], dtype=float)
        source_center = np.asarray(donor_circle.geometry["center"], dtype=float)
        placement = _alignment_matrix(
            source_center,
            np.array([1.0, 0.0]),
            target,
            np.array([1.0, 0.0]),
            scale,
        )
        donor_component_id, donor_primitives = self._install_component(donor, placement)
        placed_circle = next(
            item for item in donor_primitives if item.source_primitive_id == donor_circle.source_primitive_id
        )
        interaction_id = self._next_interaction_id()
        host_anchor = self._circle_center_anchor(host_component_id, host_circle)
        donor_anchor = self._circle_center_anchor(donor_component_id, placed_circle)
        residual = float(
            np.linalg.norm(
                np.asarray(host_circle.geometry["center"])
                - np.asarray(placed_circle.geometry["center"])
            )
        )
        self.interactions.append(
            Interaction(
                interaction_id=interaction_id,
                type="concentric",
                host_component_id=host_component_id,
                donor_component_id=donor_component_id,
                host_anchor_id=host_anchor.anchor_id,
                donor_anchor_id=donor_anchor.anchor_id,
                residual=residual,
                intended_contacts=[],
                metadata={"radius_ratio": radius_ratio},
            )
        )
        self._add_interaction_to_primitives(
            interaction_id, [host_circle.primitive_id, placed_circle.primitive_id]
        )
        self._remember_anchors(host_anchor, donor_anchor)
        return donor_component_id

    def tangent(
        self,
        host_component_id: str,
        donor: SourceComponent,
        scale: float = 0.27,
        contact_angle: float | None = None,
    ) -> str:
        host_circles = [
            item for item in self._component_visible[host_component_id]
            if item.kind in {"circle", "arc"}
        ]
        donor_endpoints = [
            anchor for anchor in detect_anchors("donor", donor.primitives)
            if anchor.type == "endpoint"
        ]
        if not host_circles or not donor_endpoints:
            raise ValueError("tangent requires a circular host and donor endpoint")
        host_curve = max(host_circles, key=primitive_length)
        donor_anchor = donor_endpoints[int(self.rng.integers(len(donor_endpoints)))]
        angle = float(contact_angle if contact_angle is not None else self.rng.uniform(0, 2 * math.pi))
        center = np.asarray(host_curve.geometry["center"], dtype=float)
        radius = float(host_curve.geometry["radius"])
        radial = np.array([math.cos(angle), math.sin(angle)])
        contact = center + radius * radial
        tangent_direction = np.array([-radial[1], radial[0]])
        placement = _alignment_matrix(
            np.asarray(donor_anchor.position),
            np.asarray(donor_anchor.direction),
            contact,
            tangent_direction,
            scale,
        )
        donor_component_id, _ = self._install_component(donor, placement)
        placed_anchor = min(
            [item for item in self._anchors_for(donor_component_id) if item.type == "endpoint"],
            key=lambda item: np.linalg.norm(np.asarray(item.position) - contact),
        )
        interaction_id = self._next_interaction_id()
        donor_primitive_id = placed_anchor.primitive_ids[0]
        amodal_primitive_ids = self._amodal_primitive_ids(
            [host_curve.primitive_id, donor_primitive_id]
        )
        residual = float(np.linalg.norm(np.asarray(placed_anchor.position) - contact))
        host_anchor = Anchor(
            anchor_id=f"{host_component_id}_interaction_{interaction_id}",
            component_id=host_component_id,
            type="circle_boundary",
            position=contact.tolist(),
            direction=tangent_direction.tolist(),
            normal=radial.tolist(),
            primitive_ids=[host_curve.primitive_id],
            local_scale=radius,
            compatible_relations=["tangent"],
            parameter=angle / (2 * math.pi),
        )
        self.interactions.append(
            Interaction(
                interaction_id=interaction_id,
                type="tangent",
                host_component_id=host_component_id,
                donor_component_id=donor_component_id,
                host_anchor_id=host_anchor.anchor_id,
                donor_anchor_id=placed_anchor.anchor_id,
                residual=residual,
                intended_contacts=[contact.tolist()],
                metadata={"contact_angle": angle},
            )
        )
        self._add_interaction_to_primitives(
            interaction_id,
            [host_curve.primitive_id, donor_primitive_id, *amodal_primitive_ids],
        )
        junction = Junction(
            junction_id=f"j{len(self.junctions_visible):03d}",
            type="tangent_contact",
            position=contact.tolist(),
            primitive_ids=[host_curve.primitive_id, donor_primitive_id],
            component_ids=[host_component_id, donor_component_id],
            interaction_id=interaction_id,
        )
        self.junctions_visible.append(junction)
        self.junctions_amodal.append(
            replace(junction, primitive_ids=amodal_primitive_ids)
        )
        self._remember_anchors(host_anchor, placed_anchor)
        return donor_component_id

    def connected_crossing(
        self,
        host_component_id: str,
        donor: SourceComponent,
        scale: float = 0.16,
        parameter: float | None = None,
        crossing_degrees: float | None = None,
    ) -> str:
        host_lines = [
            primitive
            for primitive in self._component_visible[host_component_id]
            if primitive.kind == "line" and primitive_length(primitive) > 0.06
        ]
        donor_lines = [primitive for primitive in donor.primitives if primitive.kind == "line"]
        if not host_lines or len(donor.primitives) != 1 or len(donor_lines) != 1:
            raise ValueError("connected_crossing requires a line host and single-line donor")

        candidates = []
        for host_line in host_lines:
            p0 = np.asarray(host_line.geometry["p0"], dtype=float)
            p1 = np.asarray(host_line.geometry["p1"], dtype=float)
            values = [float(parameter)] if parameter is not None else [
                float(self.rng.uniform(0.28, 0.72)) for _ in range(5)
            ]
            for u in values:
                target = (1.0 - u) * p0 + u * p1
                if parameter is not None or self._is_clear_contact(target, 0.04):
                    candidates.append((host_line, p0, p1, u, target))
        if not candidates:
            raise ValueError("connected_crossing has no clear host interval")
        host_line, p0, p1, host_u, target = candidates[
            int(self.rng.integers(len(candidates)))
        ]
        host_direction = _unit(p1 - p0)
        turn = float(
            crossing_degrees
            if crossing_degrees is not None
            else self.rng.choice([55.0, 70.0, 90.0, 110.0, 125.0])
        )
        angle = math.radians(turn)
        target_direction = np.array(
            [
                math.cos(angle) * host_direction[0] - math.sin(angle) * host_direction[1],
                math.sin(angle) * host_direction[0] + math.cos(angle) * host_direction[1],
            ]
        )
        donor_line = donor_lines[0]
        donor_p0 = np.asarray(donor_line.geometry["p0"], dtype=float)
        donor_p1 = np.asarray(donor_line.geometry["p1"], dtype=float)
        donor_middle = 0.5 * (donor_p0 + donor_p1)
        placement = _alignment_matrix(
            donor_middle,
            donor_p1 - donor_p0,
            target,
            target_direction,
            scale,
        )
        donor_component_id, donor_primitives = self._install_component(donor, placement)
        placed_line = next(
            primitive
            for primitive in donor_primitives
            if primitive.source_primitive_id == donor_line.source_primitive_id
        )
        placed_p0 = np.asarray(placed_line.geometry["p0"], dtype=float)
        placed_p1 = np.asarray(placed_line.geometry["p1"], dtype=float)
        denominator = float(np.dot(placed_p1 - placed_p0, placed_p1 - placed_p0))
        donor_u = float(np.dot(target - placed_p0, placed_p1 - placed_p0) / denominator)
        if not 0.05 < donor_u < 0.95:
            raise ValueError("crossing fell outside donor interior")

        interaction_id = self._next_interaction_id()
        amodal_host_id = self._amodal_primitive_id(host_line)
        amodal_donor_id = self._amodal_primitive_id(placed_line)
        host_first_id = f"{host_line.primitive_id}_x0"
        host_second_id = f"{host_line.primitive_id}_x1"
        donor_first_id = f"{placed_line.primitive_id}_x0"
        donor_second_id = f"{placed_line.primitive_id}_x1"
        host_first, host_second, host_contact = split_line(
            host_line, host_u, host_first_id, host_second_id, interaction_id
        )
        donor_first, donor_second, donor_contact = split_line(
            placed_line, donor_u, donor_first_id, donor_second_id, interaction_id
        )
        self._replace_visible_primitive(
            host_component_id, host_line, [host_first, host_second]
        )
        self._replace_visible_primitive(
            donor_component_id, placed_line, [donor_first, donor_second]
        )
        self._add_interaction_to_primitives(
            interaction_id,
            [
                host_line.primitive_id,
                placed_line.primitive_id,
                amodal_host_id,
                amodal_donor_id,
            ],
        )
        residual = float(np.linalg.norm(host_contact - donor_contact))
        host_anchor = Anchor(
            anchor_id=f"{host_component_id}_interaction_{interaction_id}",
            component_id=host_component_id,
            type="boundary_point",
            position=host_contact.tolist(),
            direction=host_direction.tolist(),
            normal=[float(-host_direction[1]), float(host_direction[0])],
            primitive_ids=[host_first_id, host_second_id],
            local_scale=primitive_length(host_line),
            compatible_relations=["connected_crossing"],
            parameter=host_u,
        )
        donor_anchor = Anchor(
            anchor_id=f"{donor_component_id}_interaction_{interaction_id}",
            component_id=donor_component_id,
            type="boundary_point",
            position=donor_contact.tolist(),
            direction=_unit(placed_p1 - placed_p0).tolist(),
            normal=None,
            primitive_ids=[donor_first_id, donor_second_id],
            local_scale=primitive_length(placed_line),
            compatible_relations=["connected_crossing"],
            parameter=donor_u,
        )
        self.interactions.append(
            Interaction(
                interaction_id=interaction_id,
                type="connected_crossing",
                host_component_id=host_component_id,
                donor_component_id=donor_component_id,
                host_anchor_id=host_anchor.anchor_id,
                donor_anchor_id=donor_anchor.anchor_id,
                residual=residual,
                intended_contacts=[host_contact.tolist()],
                metadata={
                    "crossing_degrees": turn,
                    "host_parameter": host_u,
                    "donor_parameter": donor_u,
                    "host_children": [host_first_id, host_second_id],
                    "donor_children": [donor_first_id, donor_second_id],
                },
            )
        )
        junction = Junction(
            junction_id=f"j{len(self.junctions_visible):03d}",
            type="x_junction",
            position=host_contact.tolist(),
            primitive_ids=[
                host_first_id,
                host_second_id,
                donor_first_id,
                donor_second_id,
            ],
            component_ids=[host_component_id, donor_component_id],
            interaction_id=interaction_id,
        )
        self.junctions_visible.append(junction)
        self.junctions_amodal.append(
            replace(
                junction,
                primitive_ids=[amodal_host_id, amodal_donor_id],
            )
        )
        self._remember_anchors(host_anchor, donor_anchor)
        return donor_component_id

    def bridge(
        self,
        donor: SourceComponent,
        minimum_span: float = 0.10,
        maximum_span: float = 0.42,
    ) -> str:
        donor_lines = [primitive for primitive in donor.primitives if primitive.kind == "line"]
        if len(donor.primitives) != 1 or len(donor_lines) != 1:
            raise ValueError("bridge requires a single-line donor")
        anchors = []
        for component_id in self.component_ids():
            anchors.extend(
                anchor
                for anchor in self._anchors_for(component_id)
                if anchor.type == "endpoint"
                and self._is_clear_contact(np.asarray(anchor.position), 0.03)
            )
        pairs = []
        for index, first in enumerate(anchors):
            for second in anchors[index + 1 :]:
                if first.component_id == second.component_id:
                    continue
                distance = float(
                    np.linalg.norm(
                        np.asarray(first.position) - np.asarray(second.position)
                    )
                )
                if minimum_span <= distance <= maximum_span:
                    target_p0 = np.asarray(first.position, dtype=float)
                    target_p1 = np.asarray(second.position, dtype=float)
                    candidate = LineString([target_p0, target_p1])
                    clear = True
                    for primitive in self.visible:
                        if primitive.semantic != "object_visible":
                            continue
                        points = sample_primitive(primitive, 256)
                        if primitive.kind == "circle":
                            points = np.vstack([points, points[0]])
                        if len(points) < 2:
                            continue
                        contacts, overlap = _intersection_points(
                            candidate.intersection(LineString(points))
                        )
                        if overlap:
                            clear = False
                            break
                        allowed = []
                        if primitive.component_id == first.component_id:
                            allowed.append(target_p0)
                        if primitive.component_id == second.component_id:
                            allowed.append(target_p1)
                        if any(
                            not allowed
                            or min(np.linalg.norm(point - target) for target in allowed)
                            > 0.004
                            for point in contacts
                        ):
                            clear = False
                            break
                    if clear:
                        pairs.append((first, second, distance))
        if not pairs:
            raise ValueError("bridge has no compatible pair of free endpoints")
        first, second, target_distance = pairs[int(self.rng.integers(len(pairs)))]

        donor_line = donor_lines[0]
        source_p0 = np.asarray(donor_line.geometry["p0"], dtype=float)
        source_p1 = np.asarray(donor_line.geometry["p1"], dtype=float)
        source_distance = float(np.linalg.norm(source_p1 - source_p0))
        if source_distance <= 1e-9:
            raise ValueError("bridge donor is degenerate")
        target_p0 = np.asarray(first.position, dtype=float)
        target_p1 = np.asarray(second.position, dtype=float)
        placement = _alignment_matrix(
            source_p0,
            source_p1 - source_p0,
            target_p0,
            target_p1 - target_p0,
            target_distance / source_distance,
        )
        donor_component_id, donor_primitives = self._install_component(donor, placement)
        placed_line = donor_primitives[0]
        placed_anchors = [
            anchor
            for anchor in self._anchors_for(donor_component_id)
            if anchor.type == "endpoint"
        ]
        donor_first = min(
            placed_anchors,
            key=lambda anchor: np.linalg.norm(np.asarray(anchor.position) - target_p0),
        )
        donor_second = min(
            placed_anchors,
            key=lambda anchor: np.linalg.norm(np.asarray(anchor.position) - target_p1),
        )
        bridge_contacts = (
            (first, donor_first, target_p0),
            (second, donor_second, target_p1),
        )
        for bridge_endpoint, (host_anchor, donor_anchor, target) in enumerate(
            bridge_contacts
        ):
            interaction_id = self._next_interaction_id()
            visible_primitive_ids = [
                host_anchor.primitive_ids[0],
                placed_line.primitive_id,
            ]
            amodal_primitive_ids = self._amodal_primitive_ids(
                visible_primitive_ids
            )
            residual = float(
                np.linalg.norm(np.asarray(donor_anchor.position) - target)
            )
            self._add_interaction_to_primitives(
                interaction_id,
                [*visible_primitive_ids, *amodal_primitive_ids],
            )
            self.interactions.append(
                Interaction(
                    interaction_id=interaction_id,
                    type="bridge",
                    host_component_id=host_anchor.component_id,
                    donor_component_id=donor_component_id,
                    host_anchor_id=host_anchor.anchor_id,
                    donor_anchor_id=donor_anchor.anchor_id,
                    residual=residual,
                    intended_contacts=[target.tolist()],
                    metadata={"bridge_endpoint": bridge_endpoint},
                )
            )
            junction = Junction(
                junction_id=f"j{len(self.junctions_visible):03d}",
                type="endpoint",
                position=target.tolist(),
                primitive_ids=[host_anchor.primitive_ids[0], placed_line.primitive_id],
                component_ids=[host_anchor.component_id, donor_component_id],
                interaction_id=interaction_id,
            )
            self.junctions_visible.append(junction)
            self.junctions_amodal.append(
                replace(junction, primitive_ids=amodal_primitive_ids)
            )
            self._remember_anchors(host_anchor, donor_anchor)
        return donor_component_id

    def _circle_center_anchor(self, component_id: str, primitive) -> Anchor:
        center = list(primitive.geometry["center"])
        return Anchor(
            anchor_id=f"{component_id}_circle_{primitive.primitive_id}",
            component_id=component_id,
            type="circle_center",
            position=center,
            direction=[1.0, 0.0],
            normal=[0.0, 1.0],
            primitive_ids=[primitive.primitive_id],
            local_scale=float(primitive.geometry["radius"]),
            compatible_relations=["concentric"],
        )

    def _amodal_primitive_id(self, visible_primitive) -> str:
        matches = [
            primitive
            for primitive in self._component_amodal[visible_primitive.component_id]
            if primitive.source_primitive_id == visible_primitive.source_primitive_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"cannot resolve amodal parent for {visible_primitive.primitive_id}"
            )
        return matches[0].primitive_id

    def _amodal_primitive_ids(self, visible_primitive_ids: list[str]) -> list[str]:
        visible_by_id = {
            primitive.primitive_id: primitive for primitive in self.visible
        }
        missing = set(visible_primitive_ids) - set(visible_by_id)
        if missing:
            raise ValueError(f"unknown visible primitives {sorted(missing)}")
        return [
            self._amodal_primitive_id(visible_by_id[primitive_id])
            for primitive_id in visible_primitive_ids
        ]

    def _replacement_ids_at_position(self, replacements, position) -> list[str]:
        point = np.asarray(position, dtype=float)
        matches = []
        for replacement in replacements:
            if any(
                np.linalg.norm(endpoint - point) <= 1e-7
                for endpoint in primitive_endpoints(replacement)
            ):
                matches.append(replacement.primitive_id)
        return matches

    def _replace_visible_primitive(self, component_id: str, old, replacements) -> None:
        old_id = old.primitive_id
        for junction in self.junctions_visible:
            if old_id not in junction.primitive_ids:
                continue
            replacement_ids = self._replacement_ids_at_position(
                replacements, junction.position
            )
            if not replacement_ids:
                raise ValueError(
                    f"split retired {old_id} without preserving junction {junction.junction_id}"
                )
            junction.primitive_ids = [
                primitive_id
                for existing_id in junction.primitive_ids
                for primitive_id in (
                    replacement_ids if existing_id == old_id else [existing_id]
                )
            ]
        for anchor in self.anchors:
            if old_id not in anchor.primitive_ids:
                continue
            replacement_ids = self._replacement_ids_at_position(
                replacements, anchor.position
            )
            if replacement_ids:
                anchor.primitive_ids = [
                    primitive_id
                    for existing_id in anchor.primitive_ids
                    for primitive_id in (
                        replacement_ids if existing_id == old_id else [existing_id]
                    )
                ]
        self.visible[:] = [item for item in self.visible if item.primitive_id != old_id]
        self.visible.extend(replacements)
        component_items = self._component_visible[component_id]
        component_items[:] = [item for item in component_items if item.primitive_id != old_id]
        component_items.extend(replacements)
        component = next(item for item in self.components if item.component_id == component_id)
        component.primitive_ids = [item.primitive_id for item in component_items]

    def _add_interaction_to_primitives(
        self, interaction_id: str, primitive_ids: list[str]
    ) -> None:
        selected = set(primitive_ids)
        for collection in (self.visible, self.amodal):
            for index, primitive in enumerate(collection):
                if primitive.primitive_id in selected:
                    collection[index] = replace(
                        primitive,
                        interaction_ids=list(
                            dict.fromkeys([*primitive.interaction_ids, interaction_id])
                        ),
                    )
        for mapping in (self._component_visible, self._component_amodal):
            for component_id, primitives in mapping.items():
                by_id = {item.primitive_id: item for item in (
                    self.visible if mapping is self._component_visible else self.amodal
                )}
                mapping[component_id] = [by_id.get(item.primitive_id, item) for item in primitives]

    def finalize(self) -> CanonicalDrawing:
        layers: dict[str, list[str]] = {}
        for primitive in self.visible:
            layers.setdefault(primitive.semantic, []).append(primitive.primitive_id)
        return CanonicalDrawing(
            sample_id=self.sample_id,
            track="compose2d",
            difficulty=self.difficulty,
            seed=self.seed,
            split=self.split,
            canvas=self.canvas,
            sources=copy.deepcopy(self.sources),
            components=copy.deepcopy(self.components),
            transforms=copy.deepcopy(self.transforms),
            anchors=copy.deepcopy(self.anchors),
            interactions=copy.deepcopy(self.interactions),
            primitives_visible=copy.deepcopy(self.visible),
            primitives_amodal=copy.deepcopy(self.amodal),
            junctions_visible=copy.deepcopy(self.junctions_visible),
            junctions_amodal=copy.deepcopy(self.junctions_amodal),
            semantic_layers=layers,
            processing={"coordinate_frame": "normalized_y_down", "deterministic": True},
        )
