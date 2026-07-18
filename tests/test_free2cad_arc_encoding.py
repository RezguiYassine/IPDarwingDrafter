import numpy as np

from stage3_primitivesfitting.research.stage3_primitive_fit_free2cad import (
    Free2CADFitter,
)
from stage3_primitivesfitting.research.train_free2cad_v3 import (
    CMD_TYPES,
    EncodedDataset,
    _apply_arc_encoding,
    _filter_inconsistent_circles,
    _relabel_degenerate_arcs,
    _stroke_residuals,
    compute_class_weights,
)


def test_three_point_targets_come_from_ordered_stroke():
    pts = np.full((1, 5, 2), -1.0, dtype=np.float32)
    pts[0, :5] = [[1.0, 0.5], [0.96, 0.69], [0.85, 0.85], [0.69, 0.96], [0.5, 1.0]]
    data = EncodedDataset(
        pts, np.ones((1, 5), dtype=bool),
        [CMD_TYPES["ARC"]], np.zeros((1, 6), dtype=np.float32),
    )
    data = _apply_arc_encoding(data, "three_point")
    np.testing.assert_allclose(
        data.params[0], [1.0, 0.5, 0.85, 0.85, 0.5, 1.0], atol=1e-6)


def test_three_point_decode_recovers_quarter_circle():
    fitter = Free2CADFitter.__new__(Free2CADFitter)
    fitter._arc_encoding = "three_point"
    diagonal = 0.5 + np.sqrt(2.0) / 4.0
    params = np.array([1.0, 0.5, diagonal, diagonal, 0.5, 1.0])
    result = fitter._decode(
        fitter._TYPE_ARC, params, 0.9, {"id": 7},
        {"center": [0.5, 0.5], "scale": 1.0, "frac": 1.0},
    )
    assert result["type"] == "arc"
    np.testing.assert_allclose(result["center"], [0.5, 0.5], atol=1e-5)
    assert abs(result["radius"] - 0.5) < 1e-5
    assert abs(result["start_angle"] - 0.0) < 1e-4
    assert abs(result["end_angle"] - 90.0) < 1e-4


def test_sqrt_class_weights_reduce_rare_class_overweighting():
    data = EncodedDataset(
        np.zeros((10, 2, 2), dtype=np.float32),
        np.ones((10, 2), dtype=bool),
        np.array([CMD_TYPES["LINE"]] * 9 + [CMD_TYPES["ARC"]]),
        np.zeros((10, 6), dtype=np.float32),
    )
    inverse = compute_class_weights(data, max_weight=100.0, power=1.0)
    sqrt_inverse = compute_class_weights(data, max_weight=100.0, power=0.5)

    assert np.isclose(
        inverse[CMD_TYPES["ARC"]] / inverse[CMD_TYPES["LINE"]], 9.0)
    assert np.isclose(
        sqrt_inverse[CMD_TYPES["ARC"]] / sqrt_inverse[CMD_TYPES["LINE"]], 3.0)
    assert inverse[CMD_TYPES["CIRCLE"]] == 0.0


def test_inconsistent_full_circle_targets_are_filtered():
    data = EncodedDataset(
        np.zeros((2, 2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        np.array([CMD_TYPES["CIRCLE"], CMD_TYPES["CIRCLE"]]),
        np.array([[0.5, 0.5, 0.5, 0, 0, 0],
                  [2.0, -1.0, 3.0, 0, 0, 0]], dtype=np.float32),
    )

    filtered, removed = _filter_inconsistent_circles(data)

    assert removed == 1
    assert len(filtered) == 1
    np.testing.assert_allclose(filtered.params[0, :3], [0.5, 0.5, 0.5])


def test_raster_straight_arc_targets_are_relabelled_as_lines():
    pts = np.full((2, 5, 2), -1.0, dtype=np.float32)
    pts[0] = np.column_stack([np.linspace(0.0, 1.0, 5), np.full(5, 0.5)])
    pts[1] = [[0.0, 0.0], [0.25, 0.05], [0.5, 0.2],
              [0.75, 0.45], [1.0, 0.8]]
    data = EncodedDataset(
        pts, np.ones((2, 5), dtype=bool),
        np.array([CMD_TYPES["ARC"], CMD_TYPES["ARC"]]),
        np.zeros((2, 6), dtype=np.float32),
    )

    relabelled, count = _relabel_degenerate_arcs(data)

    assert count == 1
    assert relabelled.types.tolist() == [CMD_TYPES["LINE"], CMD_TYPES["ARC"]]
    np.testing.assert_allclose(relabelled.params[0, :4], [0.0, 0.5, 1.0, 0.5])


def test_stroke_residual_is_zero_for_exact_primitives():
    diagonal = 0.5 + np.sqrt(2.0) / 4.0
    params = np.array([
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, diagonal, diagonal, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    angles = np.linspace(0.0, np.pi / 2.0, 5)
    points = np.array([
        np.column_stack([np.linspace(0.0, 1.0, 5), np.zeros(5)]),
        np.column_stack([0.5 + 0.5 * np.cos(angles),
                         0.5 + 0.5 * np.sin(angles)]),
        np.column_stack([0.5 + 0.5 * np.cos(angles),
                         0.5 + 0.5 * np.sin(angles)]),
    ], dtype=np.float32)
    types = np.array([CMD_TYPES["LINE"], CMD_TYPES["ARC"], CMD_TYPES["CIRCLE"]])

    residuals = _stroke_residuals(
        params, points, np.ones((3, 5), dtype=bool), types, "three_point")

    np.testing.assert_allclose(residuals, 0.0, atol=1e-6)
