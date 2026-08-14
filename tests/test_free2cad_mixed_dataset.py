import numpy as np

from tools.build_free2cad_mixed_dataset import (
    CMD_TYPES,
    _training_visible_arrays,
    allocate_domains,
    choose_epoch_target,
)


def test_auto_target_uses_larger_d2c_rare_class_supply():
    supply = np.array([1000, 100, 50, 180, 140])
    assert choose_epoch_target(supply, 0) == 180


def test_analytic_allocation_prefers_sketchgraphs_but_fills_supply_gap():
    sg = np.array([1000, 1000, 1000, 0, 0])
    d2c = np.array([1000, 10, 1000, 200, 150])

    allocation = allocate_domains(
        100, CMD_TYPES["ARC"], sg, d2c, sg_fraction=0.7, max_repeats=1.0
    )

    assert allocation == {"sketchgraphs": 90, "drawing2cad": 10}


def test_polyline_and_bezier_supply_comes_only_from_drawing2cad():
    sg = np.array([1000, 1000, 1000, 0, 0])
    d2c = np.array([1000, 100, 100, 40, 30])

    allocation = allocate_domains(
        50, CMD_TYPES["BEZIER"], sg, d2c, sg_fraction=0.7, max_repeats=1.5
    )

    assert allocation == {"sketchgraphs": 0, "drawing2cad": 45}


def test_training_visibility_matches_circle_and_arc_cleaning_contract():
    points = np.zeros((3, 5, 2), dtype=np.float32)
    points[0, :3] = [[0.1, 0.1], [0.5, 0.1], [0.9, 0.1]]
    points[1, :3] = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.1]]
    points[2, :3] = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    mask = np.zeros((3, 5), dtype=bool)
    mask[:, :3] = True
    types = np.array([
        CMD_TYPES["CIRCLE"], CMD_TYPES["ARC"], CMD_TYPES["ARC"]
    ])
    params = np.zeros((3, 6), dtype=np.float32)
    params[0, :3] = [0.9, 0.5, 0.2]

    keep, effective_types, effective_params, removed, relabeled = (
        _training_visible_arrays({
            "points": points,
            "mask": mask,
            "types": types,
            "params": params,
        })
    )

    assert keep.tolist() == [False, True, True]
    assert removed == 1
    assert relabeled == 1
    assert effective_types.tolist() == [
        CMD_TYPES["CIRCLE"], CMD_TYPES["ARC"], CMD_TYPES["LINE"]
    ]
    np.testing.assert_allclose(effective_params[2, :4], [0.0, 0.0, 1.0, 1.0])
