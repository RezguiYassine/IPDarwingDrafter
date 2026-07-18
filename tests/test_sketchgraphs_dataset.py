import numpy as np
from scipy.spatial import cKDTree

from tools.sketchgraphs_dataset import (
    PrepareConfig,
    Primitive,
    TYPE_TO_ID,
    _chunk_progress_report,
    _merge_chunk_stats,
    _new_chunk_aggregate,
    _stage3_sample,
    _write_stage3_shard,
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


def test_restart_safe_shard_and_progress_aggregation(tmp_path):
    points = np.column_stack([np.linspace(0.0, 1.0, 5), np.zeros(5)])
    sample = {
        "points": points.astype(np.float32),
        "mask": np.ones(5, dtype=bool),
        "type": TYPE_TO_ID["LINE"],
        "params": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "source_index": 9,
    }
    path = tmp_path / "shard_000000.npz"

    size = _write_stage3_shard(path, [sample])

    assert size == path.stat().st_size
    assert not path.with_suffix(".npz.tmp").exists()
    with np.load(path) as data:
        assert data["source_index"].tolist() == [9]
        assert data["types"].tolist() == [TYPE_TO_ID["LINE"]]

    aggregate = _new_chunk_aggregate()
    _merge_chunk_stats(aggregate, {
        "selected": 1,
        "accepted": 1,
        "edges": 1,
        "unmatched_edges": 0,
        "short_edges": 0,
        "errors": 0,
        "stage3_samples": 1,
        "keypoints": [2, 0, 0],
        "stage3_classes": [1, 0, 0, 0],
        "error_counts": {},
    })
    report = _chunk_progress_report(
        "train", tmp_path / "raw.npy", 1, 1, 42, 0, {}, 1, 1,
        aggregate,
    )
    assert report["complete"] is True
    assert report["stage3_samples"] == 1
    assert report["stage3_classes"]["LINE"] == 1
