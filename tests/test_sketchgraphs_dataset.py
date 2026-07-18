import numpy as np
from scipy.spatial import cKDTree

from tools.sketchgraphs_dataset import (
    PrepareConfig,
    Primitive,
    TYPE_TO_ID,
    _stage3_sample,
)


def _circle_primitive() -> Primitive:
    angles = np.linspace(0.0, 2.0 * np.pi, 129)
    points = np.column_stack([50.0 + 20.0 * np.cos(angles),
                              50.0 + 20.0 * np.sin(angles)])
    return Primitive(
        entity_id="circle-1",
        kind="CIRCLE",
        world_points=points.copy(),
        pixel_points=points,
        center_px=np.array([50.0, 50.0]),
        radius_px=20.0,
        closed=True,
        tree=cKDTree(points),
    )


def test_open_circle_fragment_becomes_arc_training_sample():
    primitive = _circle_primitive()
    fragment = primitive.pixel_points[:33]
    edge = {"pixels": fragment.tolist(), "smooth_pts": fragment.tolist(),
            "is_closed": False}

    sample = _stage3_sample(edge, primitive, PrepareConfig(), source_index=4)

    assert sample is not None
    assert sample["type"] == TYPE_TO_ID["ARC"]


def test_closed_circle_uses_extracted_edge_geometry():
    primitive = _circle_primitive()
    extracted = primitive.pixel_points[::2]
    edge = {"pixels": extracted.tolist(), "smooth_pts": [], "is_closed": True}

    sample = _stage3_sample(edge, primitive, PrepareConfig(), source_index=5)

    assert sample is not None
    assert sample["type"] == TYPE_TO_ID["CIRCLE"]
    assert sample["mask"].sum() == 64
    np.testing.assert_allclose(sample["params"][:3], [0.5, 0.5, 0.5], atol=1e-5)


def test_incomplete_closed_circle_is_rejected():
    primitive = _circle_primitive()
    fragment = primitive.pixel_points[:20]
    edge = {"pixels": fragment.tolist(), "smooth_pts": [], "is_closed": True}

    assert _stage3_sample(
        edge, primitive, PrepareConfig(), source_index=6
    ) is None


def test_raster_straight_circle_fragment_becomes_line():
    primitive = _circle_primitive()
    fragment = np.column_stack([np.linspace(40.0, 60.0, 20),
                                np.full(20, 50.0)])
    edge = {"pixels": fragment.tolist(), "smooth_pts": fragment.tolist(),
            "is_closed": False}

    sample = _stage3_sample(edge, primitive, PrepareConfig(), source_index=7)

    assert sample is not None
    assert sample["type"] == TYPE_TO_ID["LINE"]
