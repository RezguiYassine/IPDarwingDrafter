"""The ground-truth metric has to be validated before it validates anything.

Its whole purpose is to replace self-consistency, so it cannot be checked
against the pipeline it is meant to judge. These tests construct cases whose
answer is known by construction: drop a known number of primitives and recall
must fall by that amount; add unsupported ones and precision must fall; keep
everything and both must be 1.0.
"""

import math

import numpy as np
import pytest

from tools import gt_fidelity


CANVAS = 1000


def line(x0, y0, x1, y1, layer="object_visible"):
    """A ground-truth line in the generator's normalised coordinates."""
    return {"primitive_type": "line", "semantic_layer": layer,
            "geometry": {"p0": [x0 / CANVAS, y0 / CANVAS],
                         "p1": [x1 / CANVAS, y1 / CANVAS]}}


def fitted(x0, y0, x1, y1):
    return {"type": "line", "p1": [x0, y0], "p2": [x1, y1], "confidence": 1.0}


def truth_of(primitives):
    return {"canvas": [CANVAS, CANVAS], "primitives_visible": primitives}


GRID = [line(100, 100 + 40 * i, 900, 100 + 40 * i) for i in range(10)]
GRID_FITTED = [fitted(100, 100 + 40 * i, 900, 100 + 40 * i) for i in range(10)]


def test_perfect_reconstruction_scores_one():
    result = gt_fidelity.score(truth_of(GRID), GRID_FITTED)
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == 1.0


@pytest.mark.parametrize("dropped", [1, 3, 5])
def test_recall_falls_by_exactly_what_was_dropped(dropped):
    """The self-validation the roadmap asks for."""
    result = gt_fidelity.score(truth_of(GRID), GRID_FITTED[:len(GRID) - dropped])
    assert result["missed"] == dropped
    assert result["recall"] == pytest.approx((len(GRID) - dropped) / len(GRID))
    assert result["precision"] == 1.0, "dropping output must not cost precision"


@pytest.mark.parametrize("invented", [1, 4])
def test_precision_falls_when_geometry_is_invented(invented):
    extra = [fitted(50, 600 + 30 * i, 950, 600 + 30 * i) for i in range(invented)]
    # place the inventions well away from every ground-truth line
    extra = [fitted(50 + 5 * i, 20, 950, 20) for i in range(invented)]
    result = gt_fidelity.score(truth_of(GRID), GRID_FITTED + extra)
    assert result["recall"] == 1.0, "inventing geometry must not cost recall"
    assert result["unsupported"] == invented
    assert result["precision"] == pytest.approx(len(GRID) / (len(GRID) + invented))


def test_a_primitive_that_covers_only_half_its_target_is_not_found():
    """Overlap is fractional: half a line is not the line."""
    half = [fitted(100, 100, 500, 100)]
    result = gt_fidelity.score(truth_of([line(100, 100, 900, 100)]), half)
    assert result["missed"] == 1 and result["recall"] == 0.0


def test_tolerance_is_a_distance_not_a_free_pass():
    truth = truth_of([line(100, 100, 900, 100)])
    near = gt_fidelity.score(truth, [fitted(100, 102, 900, 102)], tolerance=3.0)
    far = gt_fidelity.score(truth, [fitted(100, 140, 900, 140)], tolerance=3.0)
    assert near["recall"] == 1.0
    assert far["recall"] == 0.0


def test_removed_reference_numerals_are_not_counted_as_missed_geometry():
    """Stage 0 deletes numerals on purpose; scoring them down is wrong."""
    truth = truth_of(GRID + [line(20, 20, 30, 30, layer="reference_numeral"),
                             line(40, 20, 50, 30, layer="text")])
    result = gt_fidelity.score(truth, GRID_FITTED)
    assert result["truth_primitives"] == len(GRID)
    assert result["recall"] == 1.0


def test_polyline_and_arc_ground_truth_are_sampled():
    poly = {"primitive_type": "polyline", "semantic_layer": "object_visible",
            "geometry": {"points": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]}}
    arc = {"primitive_type": "arc", "semantic_layer": "object_visible",
           "geometry": {"center": [0.5, 0.5], "radius": 0.2,
                        "start_angle": 0.0, "end_angle": 180.0}}
    scored = gt_fidelity.score(truth_of([poly, arc]), [])
    assert scored["truth_primitives"] == 2, "both shapes must be sampled, not skipped"
    assert scored["recall"] == 0.0


def test_empty_output_is_zero_recall_not_an_error():
    result = gt_fidelity.score(truth_of(GRID), [])
    assert result["recall"] == 0.0
    assert math.isnan(result["precision"])       # nothing was claimed
