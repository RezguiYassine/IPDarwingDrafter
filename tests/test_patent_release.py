import copy
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from syntheticData.partition_sources import assigned_split, content_digest, freeze
from syntheticData.patentvec.annotations import text_bounds, annotation_failures
from syntheticData.patentvec.composition import CompositionBuilder
from syntheticData.patentvec.generator import SourcePool, PreparedDrawing, compact_sample_payload
from syntheticData.patentvec.patent_layers import add_linear_dimension, add_leader_and_numeral, add_text_box
from syntheticData.patentvec.quality import evaluate_quality
from syntheticData.patentvec.render import render_clean, render_masks
from syntheticData.patentvec.schema import validate_drawing
from syntheticData.patentvec.sheets import compose_sheet
from syntheticData.patentvec.sources import procedural_polyline_component


def drawing():
    source = procedural_polyline_component("source", "train", seed=1, primitive_count=4)
    builder = CompositionBuilder("layout", seed=1, canvas=(1024, 1024))
    builder.add_base(source, scale=.45)
    return builder.finalize()


def test_rotated_label_bounds_follow_glyph_rotation():
    geometry = {"position": [.5, .5], "text": "123456", "size": .02, "anchor": "middle"}
    horizontal = text_bounds(geometry)
    vertical = text_bounds({**geometry, "rotation": -90})
    assert horizontal[2]-horizontal[0] == pytest.approx(vertical[3]-vertical[1])
    assert horizontal[3]-horizontal[1] == pytest.approx(vertical[2]-vertical[0])


@pytest.mark.parametrize("kind", ["dimension", "reference", "textbox"])
def test_full_annotation_area_fails_closed_without_appending_primitives(kind):
    item = drawing()
    item.processing["patent_layout"] = {"reserved_boxes": [[0, 0, 1, 1]]}
    before = len(item.primitives_visible)
    with pytest.raises(ValueError, match="collision-free"):
        if kind == "dimension":
            add_linear_dimension(item, np.random.default_rng(1))
        elif kind == "reference":
            add_leader_and_numeral(item, "123", np.random.default_rng(1))
        else:
            add_text_box(item, np.random.default_rng(1), "B1")
    assert len(item.primitives_visible) == before


def test_repeated_dimensions_find_non_overlapping_label_positions():
    item = drawing()
    rng = np.random.default_rng(3)
    for orientation in ("horizontal", "vertical", "horizontal"):
        add_linear_dimension(item, rng, orientation=orientation)
    assert not annotation_failures(item)


def test_indexed_retrieval_never_uses_unrestricted_pool():
    pool = object.__new__(SourcePool)
    pool.index_path = "frozen.json"
    pool._index_buckets = {}
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        pool.pick_profile(np.random.default_rng(1), "SketchGraphs", "circle")
    with pytest.raises(RuntimeError, match="forbidden"):
        pool.pick_sketchgraphs(np.random.default_rng(1), lambda c: True)
    with pytest.raises(RuntimeError, match="forbidden"):
        pool.pick_cadvg(np.random.default_rng(1), lambda c: True)


def test_siblings_share_partition_and_snapshots_are_bound_to_content():
    identity = "CAD-VGDrawing:model:0000/00123"
    assert len({assigned_split(identity, 160926) for _ in range(4)}) == 1
    source = procedural_polyline_component("snapshot", "train", seed=1, primitive_count=3)
    frozen = freeze(source, "test")
    before = content_digest(frozen)
    frozen["primitives"][0]["geometry"]["points"][0][0] += .01
    assert content_digest(frozen) != before
    assert source.source.split == "train"


def test_sheet_preserves_provenance_and_has_no_cross_figure_contacts():
    figures = [drawing() for _ in range(4)]
    before = copy.deepcopy(figures[0].to_dict())
    sheet = compose_sheet(figures, "sheet", 1, 2048)
    assert not validate_drawing(sheet)
    assert not annotation_failures(sheet)
    assert figures[0].to_dict() == before
    assert len(sheet.primitives_visible) == 4 * len(figures[0].primitives_visible) + 4
    for item in sheet.primitives_visible:
        if item.semantic == "object_visible":
            assert item.source_interval == [0., 1.]
            assert item.source_sample_id == figures[0].primitives_visible[0].source_sample_id
    assert evaluate_quality(sheet)["accepted"]


def test_binary_observation_retains_exact_clean_label_contract():
    item = drawing()
    item.canvas = [256, 256]
    clean, masks = render_clean(item), render_masks(item)
    prepared = PreparedDrawing(item, clean, masks, evaluate_quality(item, clean=clean, masks=masks))
    payload, row = compact_sample_payload(prepared, 0, retain_rasters=True, acquisition="binary", noise_budget=.5,
                                         label_contract="reference-free-raster-topology-v1")
    for name in ("degraded.png", "reference_free_degraded.png"):
        with Image.open(io.BytesIO(payload[name])) as image:
            assert set(np.unique(np.asarray(image))) == {0, 255}
    support = masks["object"] | masks["hidden_center"] | masks["hatch"]
    probability = row["degradation"]["degradation_parameters"]["dark_speckle_probability"]
    assert probability * support.size == pytest.approx(.5 * np.count_nonzero(support) / 1000)
