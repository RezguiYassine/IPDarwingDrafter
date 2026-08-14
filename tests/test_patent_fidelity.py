import numpy as np

from tools.evaluate_patent_fidelity import fidelity


def test_fidelity_is_exact_for_identical_geometry():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 2:18] = True

    metrics = fidelity(mask, mask.copy())

    assert metrics is not None
    assert metrics["chamfer_sym"] == 0.0
    assert metrics["chamfer_p95"] == 0.0
    assert metrics["precision_at_2px"] == 1.0
    assert metrics["recall_at_2px"] == 1.0
    assert metrics["f1_at_2px"] == 1.0


def test_fidelity_separates_missing_geometry_from_precision():
    reference = np.zeros((30, 30), dtype=bool)
    reference[5, 2:28] = True
    reference[25, 2:28] = True
    prediction = np.zeros_like(reference)
    prediction[5, 2:28] = True

    metrics = fidelity(reference, prediction)

    assert metrics is not None
    assert metrics["precision_at_2px"] == 1.0
    assert metrics["recall_at_2px"] == 0.5
    assert metrics["chamfer_sym"] > 0.0
