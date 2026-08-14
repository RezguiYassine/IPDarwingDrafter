from __future__ import annotations

import io
import json

import numpy as np

from syntheticData.patentvec.composition import CompositionBuilder
from syntheticData.patentvec.generator import PreparedDrawing, compact_sample_payload
from syntheticData.patentvec.geometry import similarity_matrix, transform_primitive
from syntheticData.patentvec.patent_layers import (
    apply_complex_patent_layers,
    apply_vertical_slice_layers,
)
from syntheticData.patentvec.quality import evaluate_quality
from syntheticData.patentvec.render import render_clean, render_masks
from syntheticData.patentvec.schema import Primitive, SourceRecord, validate_drawing
from syntheticData.patentvec.sources import (
    SourceComponent,
    _connected_groups,
    procedural_polyline_component,
)
from syntheticData.patentvec.training import free2cad_arrays, puhachov_arrays


def _primitive(
    primitive_id: str,
    kind: str,
    geometry: dict,
    dataset: str,
    sample_id: str,
    component_id: str,
) -> Primitive:
    return Primitive(
        primitive_id=primitive_id,
        kind=kind,
        geometry=geometry,
        semantic="object_visible",
        component_id=component_id,
        source_dataset=dataset,
        source_sample_id=sample_id,
        source_component_id=component_id,
        source_primitive_id=primitive_id,
        generated_by="test_fixture",
    )


def _component(
    dataset: str,
    sample_id: str,
    component_id: str,
    primitives: list[Primitive],
) -> SourceComponent:
    return SourceComponent(
        component_key=component_id,
        source=SourceRecord(
            dataset=dataset,
            sample_id=sample_id,
            split="train",
            license_id="test-only",
        ),
        primitives=primitives,
        source_to_component=np.eye(3),
        descriptor={"primitive_count": len(primitives)},
    )


def _rectangle(dataset: str = "SketchGraphs", sample_id: str = "rect"):
    component_id = f"{sample_id}_component"
    points = [(-0.5, -0.35), (0.5, -0.35), (0.5, 0.35), (-0.5, 0.35)]
    primitives = []
    for index, (p0, p1) in enumerate(zip(points, points[1:] + points[:1])):
        primitives.append(
            _primitive(
                f"{sample_id}_p{index}",
                "line",
                {"p0": list(p0), "p1": list(p1)},
                dataset,
                sample_id,
                component_id,
            )
        )
    return _component(dataset, sample_id, component_id, primitives)


def _open_line(dataset: str = "CAD-VGDrawing", sample_id: str = "line"):
    component_id = f"{sample_id}_component"
    primitive = _primitive(
        f"{sample_id}_p0",
        "line",
        {"p0": [-0.5, 0.0], "p1": [0.5, 0.0]},
        dataset,
        sample_id,
        component_id,
    )
    return _component(dataset, sample_id, component_id, [primitive])


def _circle(dataset: str = "SketchGraphs", sample_id: str = "circle"):
    component_id = f"{sample_id}_component"
    primitive = _primitive(
        f"{sample_id}_p0",
        "circle",
        {"center": [0.0, 0.0], "radius": 0.5},
        dataset,
        sample_id,
        component_id,
    )
    return _component(dataset, sample_id, component_id, [primitive])


def test_similarity_transform_preserves_circle():
    source = _circle().primitives[0]
    matrix = similarity_matrix(0.4, np.pi / 3, (0.6, 0.2))
    transformed = transform_primitive(source, matrix)
    assert transformed.kind == "circle"
    assert np.allclose(transformed.geometry["center"], [0.6, 0.2])
    assert np.isclose(transformed.geometry["radius"], 0.2)


def test_t_junction_splits_visible_but_preserves_amodal_source():
    builder = CompositionBuilder("unit_t", seed=7, canvas=(256, 256))
    host_id = builder.add_base(_rectangle(), scale=0.55)
    builder.t_junction(host_id, _open_line(), scale=0.18, parameter=0.4)
    drawing = builder.finalize()
    assert validate_drawing(drawing) == []
    interaction = drawing.interactions[0]
    parent_id = interaction.metadata["host_parent_primitive_id"]
    children = interaction.metadata["visible_children"]
    assert parent_id in {item.primitive_id for item in drawing.primitives_amodal}
    assert parent_id not in {item.primitive_id for item in drawing.primitives_visible}
    visible_by_id = {item.primitive_id: item for item in drawing.primitives_visible}
    assert np.allclose(visible_by_id[children[0]].source_interval, [0.0, 0.4])
    assert np.allclose(visible_by_id[children[1]].source_interval, [0.4, 1.0])
    assert drawing.junctions_visible[0].type == "t_junction"


def test_all_core_interactions_validate():
    builders = []

    endpoint = CompositionBuilder("unit_endpoint", seed=2, canvas=(256, 256))
    host_id = endpoint.add_base(_open_line("SketchGraphs", "host_line"), scale=0.35)
    endpoint.endpoint_join(host_id, _open_line(), scale=0.18, turn_degrees=90.0)
    builders.append(endpoint)

    containment = CompositionBuilder("unit_containment", seed=3, canvas=(256, 256))
    host_id = containment.add_base(_rectangle("CAD-VGDrawing", "box"), scale=0.55)
    containment.containment(host_id, _circle(sample_id="inner"), fill_fraction=0.20)
    builders.append(containment)

    concentric = CompositionBuilder("unit_concentric", seed=4, canvas=(256, 256))
    host_id = concentric.add_base(_circle(sample_id="outer"), scale=0.48)
    concentric.concentric(host_id, _circle(sample_id="inner"), radius_ratio=0.5)
    builders.append(concentric)

    tangent = CompositionBuilder("unit_tangent", seed=5, canvas=(256, 256))
    host_id = tangent.add_base(_circle(sample_id="tangent_circle"), scale=0.40)
    tangent.tangent(host_id, _open_line(), scale=0.14, contact_angle=0.0)
    builders.append(tangent)

    for builder in builders:
        assert validate_drawing(builder.finalize()) == []


def test_connected_crossing_and_bridge_create_exact_cycle():
    crossing = CompositionBuilder("unit_crossing", seed=31, canvas=(256, 256))
    host_id = crossing.add_base(_rectangle(), scale=0.50)
    crossing.connected_crossing(
        host_id,
        _open_line("CAD-VGDrawing", "cross_line"),
        scale=0.14,
        parameter=0.5,
        crossing_degrees=90.0,
    )
    crossing_drawing = crossing.finalize()
    assert validate_drawing(crossing_drawing) == []
    assert crossing_drawing.junctions_visible[0].type == "x_junction"
    assert len(crossing_drawing.junctions_visible[0].primitive_ids) == 4
    amodal_ids = {
        primitive.primitive_id for primitive in crossing_drawing.primitives_amodal
    }
    assert set(crossing_drawing.junctions_amodal[0].primitive_ids) <= amodal_ids

    cycle = CompositionBuilder("unit_bridge", seed=37, canvas=(256, 256))
    first_id = cycle.add_base(
        _open_line("SketchGraphs", "cycle_host"), scale=0.28
    )
    cycle.endpoint_join(
        first_id,
        _open_line("CAD-VGDrawing", "cycle_branch"),
        scale=0.20,
        turn_degrees=90.0,
    )
    cycle.bridge(_open_line("SketchGraphs", "cycle_bridge"))
    cycle_drawing = cycle.finalize()
    assert validate_drawing(cycle_drawing) == []
    assert len(cycle_drawing.components) == 3
    assert len(cycle_drawing.interactions) == 3
    assert sum(item.type == "bridge" for item in cycle_drawing.interactions) == 2
    assert [
        item.metadata["bridge_endpoint"]
        for item in cycle_drawing.interactions
        if item.type == "bridge"
    ] == [0, 1]


def test_source_connected_groups_have_canonical_order():
    forward = {
        "b": {"a", "c"},
        "a": {"b"},
        "c": {"b"},
        "z": set(),
    }
    reverse = {
        key: set(reversed(sorted(neighbors)))
        for key, neighbors in reversed(list(forward.items()))
    }
    expected = [["a", "b", "c"], ["z"]]
    assert _connected_groups(forward) == expected
    assert _connected_groups(reverse) == expected


def test_free2cad_aggregates_shallow_line_chain_without_conflicting_lines():
    primitives = [
        _primitive(
            "s0",
            "line",
            {"p0": [0.0, 0.0], "p1": [1.0, 0.0]},
            "CAD-VGDrawing",
            "shallow",
            "chain",
        ),
        _primitive(
            "s1",
            "line",
            {"p0": [1.0, 0.0], "p1": [2.0, 0.2]},
            "CAD-VGDrawing",
            "shallow",
            "chain",
        ),
        _primitive(
            "s2",
            "line",
            {"p0": [2.0, 0.2], "p1": [3.0, 0.6]},
            "CAD-VGDrawing",
            "shallow",
            "chain",
        ),
    ]
    builder = CompositionBuilder("unit_polyagg", seed=53, canvas=(256, 256))
    builder.add_base(
        _component("CAD-VGDrawing", "shallow", "chain", primitives), scale=0.5
    )
    arrays = free2cad_arrays(
        builder.finalize(), source_index=7, project_stage2=False
    )
    assert arrays["types"].tolist() == [3]
    assert arrays["edge_origins"].tolist() == [1]
    assert arrays["member_primitive_ids"][0].count("|") == 2
    assert arrays["source_index"].tolist() == [7]


def test_procedural_polyline_motif_has_exact_class_supply():
    component = procedural_polyline_component(
        sample_id="unit_supply",
        split="train",
        seed=59,
        primitive_count=3,
    )
    assert component.source.dataset == "PatentVec-Procedural"
    assert component.descriptor["polyline_count"] == 3
    assert all(primitive.kind == "polyline" for primitive in component.primitives)
    builder = CompositionBuilder("unit_supply", seed=59, canvas=(256, 256))
    builder.add_base(component, scale=0.5)
    arrays = free2cad_arrays(
        builder.finalize(), source_index=0, project_stage2=False
    )
    assert arrays["types"].tolist() == [3, 3, 3]
    assert arrays["edge_origins"].tolist() == [0, 0, 0]


def test_compact_payload_contains_training_contract_and_audit_subset():
    component = procedural_polyline_component(
        sample_id="unit_compact",
        split="train",
        seed=61,
        primitive_count=3,
    )
    builder = CompositionBuilder("unit_compact", seed=61, canvas=(256, 256))
    builder.add_base(component, scale=0.5)
    drawing = builder.finalize()
    clean = render_clean(drawing)
    masks = render_masks(drawing)
    quality = evaluate_quality(drawing, clean=clean, masks=masks)
    assert quality["accepted"], quality["failures"]
    payload, row = compact_sample_payload(
        PreparedDrawing(drawing, clean, masks, quality),
        source_index=13,
        audit=True,
    )
    assert {
        "sample.json",
        "quality.json",
        "masks.npz",
        "puhachov.npz",
        "free2cad_edges.npz",
        "clean.png",
        "degraded.png",
        "preview.png",
    } <= set(payload)
    with np.load(io.BytesIO(payload["free2cad_edges.npz"])) as free2cad:
        assert free2cad["types"].tolist() == [3, 3, 3]
        assert free2cad["source_index"].tolist() == [13, 13, 13]
    assert row["audit"] is True
    assert row["training_targets"]["free2cad_edges"]["polyline"] == 3


def test_complex_patent_layers_remain_semantically_separate():
    builder = CompositionBuilder("unit_m6", seed=41, canvas=(256, 256))
    host_id = builder.add_base(_rectangle(), scale=0.50)
    builder.t_junction(
        host_id,
        _open_line("CAD-VGDrawing", "m6_t"),
        scale=0.14,
        parameter=0.35,
    )
    builder.connected_crossing(
        host_id,
        _open_line("SketchGraphs", "m6_x"),
        scale=0.12,
        parameter=0.65,
    )
    drawing = apply_complex_patent_layers(
        builder.finalize(), seed=43, difficulty="hard"
    )
    required = {
        "hatch",
        "object_center",
        "object_hidden",
        "leader",
        "reference_numeral",
        "dimension",
        "text_box",
        "diagram_connector",
    }
    assert required <= set(drawing.semantic_layers)
    object_ids = set(drawing.semantic_layers["object_visible"])
    annotation_ids = {
        primitive_id
        for semantic, primitive_ids in drawing.semantic_layers.items()
        if semantic != "object_visible"
        for primitive_id in primitive_ids
    }
    assert object_ids.isdisjoint(annotation_ids)
    assert validate_drawing(drawing) == []


def test_vertical_slice_renders_masks_and_training_contracts():
    builder = CompositionBuilder("unit_vertical", seed=11, canvas=(256, 256))
    host_id = builder.add_base(_rectangle(), scale=0.52)
    builder.t_junction(host_id, _open_line(), scale=0.15, parameter=0.5)
    drawing = apply_vertical_slice_layers(builder.finalize(), seed=21, numeral="12")
    clean = render_clean(drawing)
    masks = render_masks(drawing)
    quality = evaluate_quality(
        drawing, clean=clean, masks=masks, require_vertical_slice=True
    )
    assert quality["accepted"], quality["failures"]
    assert clean.shape == (256, 256)
    assert masks["hatch"].max() == 255
    assert masks["text_numeral"].max() == 255

    puhachov = puhachov_arrays(drawing, clean)
    assert puhachov["skeleton"].dtype == np.uint8
    assert puhachov["kps"].ndim == 2 and puhachov["kps"].shape[1] == 3
    kp_meta = json.loads(puhachov["meta"])
    assert kp_meta["snapped_keypoint_count"] > 0
    assert kp_meta["dropped_keypoint_count"] < kp_meta["exact_keypoint_count"]

    free2cad = free2cad_arrays(drawing, source_index=0)
    assert free2cad["points"].shape[1:] == (64, 2)
    assert free2cad["mask"].dtype == bool
    assert free2cad["params"].shape[1] == 6
    assert set(free2cad["types"]) == {0}
