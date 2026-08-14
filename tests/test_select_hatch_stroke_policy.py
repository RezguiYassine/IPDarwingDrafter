import pytest

from tools.select_hatch_stroke_policy import select_joint_policy


def _evaluation(dataset, policies, checkpoint="abc"):
    return {
        "checkpoint_sha256": checkpoint,
        "dataset": dataset,
        "split": "validation",
        "policy_sweep": policies,
    }


def _policy(hatch, structural, f1, *, eligible=True, error=0.0001):
    return {
        "hatch_high_threshold": hatch,
        "structural_low_threshold": structural,
        "eligible": eligible,
        "safe_removal_f1": f1,
        "safe_removal_precision": f1 + 0.01,
        "safe_removal_recall": f1 - 0.01,
        "safe_removal_structural_error_rate": error,
    }


def test_joint_policy_requires_eligibility_on_every_dataset():
    synthetic = _evaluation("synthetic", [
        _policy(0.5, 0.1, 0.95),
        _policy(0.6, 0.2, 0.80),
    ])
    real = _evaluation("real", [
        _policy(0.5, 0.1, 0.90, eligible=False),
        _policy(0.6, 0.2, 0.70),
    ])

    result = select_joint_policy([synthetic, real])

    selected = result["selected_policy"]
    assert selected["hatch_high_threshold"] == pytest.approx(0.6)
    assert selected["structural_low_threshold"] == pytest.approx(0.2)
    assert selected["worst_case_safe_removal_f1"] == pytest.approx(0.70)


def test_joint_policy_optimizes_worst_case_f1():
    synthetic = _evaluation("synthetic", [
        _policy(0.5, 0.1, 0.99),
        _policy(0.6, 0.2, 0.85),
    ])
    real = _evaluation("real", [
        _policy(0.5, 0.1, 0.50),
        _policy(0.6, 0.2, 0.80),
    ])

    selected = select_joint_policy([synthetic, real])["selected_policy"]

    assert selected["hatch_high_threshold"] == pytest.approx(0.6)
    assert selected["worst_case_safe_removal_f1"] == pytest.approx(0.80)


def test_joint_policy_rejects_mixed_checkpoints():
    first = _evaluation("synthetic", [_policy(0.5, 0.1, 0.9)], "first")
    second = _evaluation("real", [_policy(0.5, 0.1, 0.9)], "second")

    with pytest.raises(ValueError, match="same checkpoint"):
        select_joint_policy([first, second])
