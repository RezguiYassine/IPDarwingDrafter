from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

from syntheticData.export_hatch_stroke_dataset import (
    export_hatch_stroke_dataset,
)
from syntheticData.patentvec.hatch_stroke_targets import (
    HATCH_STROKE_LABEL_CONTRACT,
    build_hatch_stroke_targets,
    validate_hatch_stroke_targets,
)
from tools.hatch_model import HatchUNet
from tools.evaluate_hatch_stroke import (
    SafeRemovalHistogram,
    _difficulty_recall_floors,
    _difficulty_recall_gate,
)
from tools.hatch_stroke_train import (
    HatchStrokePatchDataset,
    StrokeMetricAccumulator,
    allocate_difficulty_counts,
    checkpoint_rank,
    masked_multilabel_loss,
    parse_difficulty_weights,
)


def _crossing_masks(shape: tuple[int, int] = (33, 33)) -> dict[str, np.ndarray]:
    object_mask = np.zeros(shape, dtype=np.uint8)
    object_mask[16, 3:30] = 255
    hatch_mask = np.zeros(shape, dtype=np.uint8)
    hatch_mask[3:30, 16] = 255
    return {
        "object": object_mask,
        "hidden_center": np.zeros(shape, dtype=np.uint8),
        "hatch": hatch_mask,
    }


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_crossing_pixel_is_multilabel_and_arms_remain_exclusive():
    targets = build_hatch_stroke_targets(_crossing_masks(), support_radius=0)

    structural = targets["structural_target"] > 0
    hatch = targets["hatch_target"] > 0
    overlap = targets["overlap_target"] > 0
    assert structural[16, 16]
    assert hatch[16, 16]
    assert overlap[16, 16]
    assert structural[16, 8] and not hatch[16, 8]
    assert hatch[8, 16] and not structural[8, 16]
    assert np.count_nonzero(overlap) == 1


def test_target_union_is_lossless_and_annotation_masks_are_excluded():
    masks = _crossing_masks()
    masks["leader_dimension"] = np.zeros((33, 33), dtype=np.uint8)
    masks["leader_dimension"][5, 20:30] = 255

    targets = build_hatch_stroke_targets(masks, support_radius=1)
    stats = validate_hatch_stroke_targets(targets)

    skeleton = targets["input_skeleton"] > 0
    structural = targets["structural_target"] > 0
    hatch = targets["hatch_target"] > 0
    assert np.array_equal(structural | hatch, skeleton)
    assert not np.any(skeleton[5, 20:30])
    assert stats["unassigned_pixels"] == 0


def test_target_builder_rejects_invalid_radius_and_shape():
    masks = _crossing_masks()
    with pytest.raises(ValueError, match="non-negative integer"):
        build_hatch_stroke_targets(masks, support_radius=-1)

    masks["hatch"] = np.zeros((31, 33), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        build_hatch_stroke_targets(masks)


def test_validator_rejects_an_unassigned_skeleton_pixel():
    targets = build_hatch_stroke_targets(_crossing_masks(), support_radius=0)
    targets["structural_target"][16, 8] = 0
    with pytest.raises(ValueError, match="do not cover"):
        validate_hatch_stroke_targets(targets)


def test_exporter_is_deterministic_and_resumes_valid_samples(tmp_path: Path):
    input_root = tmp_path / "compact"
    output_root = tmp_path / "stroke"
    (input_root / "shards").mkdir(parents=True)
    archive_name = "shards/shard_000000.tar"
    archive_path = input_root / archive_name
    sample_id = "pv_train_0000000_medium"
    prefix = f"samples/{sample_id}"
    with tarfile.open(archive_path, mode="w") as archive:
        _add_bytes(
            archive,
            f"{prefix}/masks.npz",
            _npz_bytes(**_crossing_masks()),
        )

    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "count": 1,
        "run_fingerprint": "unit-run",
        "shards": [{
            "archive": archive_name,
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
        }],
    }
    (input_root / "manifest.json").write_text(json.dumps(manifest))
    row = {
        "archive": archive_name,
        "difficulty": "medium",
        "member_prefix": prefix,
        "relative_path": prefix,
        "sample_id": sample_id,
        "source_index": 0,
    }
    (input_root / "manifest.jsonl").write_text(json.dumps(row) + "\n")

    first = export_hatch_stroke_dataset(
        input_root,
        output_root,
        validation_modulus=1,
        validation_fold=0,
    )
    second = export_hatch_stroke_dataset(
        input_root,
        output_root,
        validation_modulus=1,
        validation_fold=0,
    )

    assert first["generated_samples_this_run"] == 1
    assert first["sample_counts"] == {"validation": 1}
    assert first["hatch_positive_samples"] == 1
    assert first["overlap_positive_samples"] == 1
    assert second["generated_samples_this_run"] == 0
    assert second["resumed_samples_this_run"] == 1
    output_path = output_root / "validation" / f"{sample_id}.npz"
    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["meta"]))
        assert metadata["label_contract"] == HATCH_STROKE_LABEL_CONTRACT
        assert data["structural_target"][16, 16] == 255
        assert data["hatch_target"][16, 16] == 255


def test_hatch_unet_keeps_legacy_default_and_supports_two_channels():
    legacy = HatchUNet(freeze_encoder=True, pretrained=False).eval()
    multilabel = HatchUNet(
        freeze_encoder=True, out_channels=2, pretrained=False
    ).eval()
    sample = torch.zeros((1, 1, 64, 64), dtype=torch.float32)

    with torch.no_grad():
        assert legacy(sample).shape == (1, 1, 64, 64)
        assert multilabel(sample).shape == (1, 2, 64, 64)


def test_masked_loss_has_finite_gradients_only_for_supplied_ink():
    logits = torch.zeros((1, 2, 8, 8), requires_grad=True)
    skeleton = torch.zeros((1, 1, 8, 8))
    skeleton[:, :, 4, 1:7] = 1.0
    targets = torch.zeros((1, 2, 8, 8))
    targets[:, 0, 4, 1:5] = 1.0
    targets[:, 1, 4, 4:7] = 1.0

    loss = masked_multilabel_loss(logits, targets, skeleton)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(logits.grad))
    assert torch.count_nonzero(logits.grad[:, :, 0, :]) == 0
    assert torch.count_nonzero(logits.grad[:, :, 4, 1:7]) > 0


def test_masked_loss_ignores_unsupervised_channel_pixels():
    logits = torch.zeros((1, 2, 8, 8), requires_grad=True)
    skeleton = torch.ones((1, 1, 8, 8))
    targets = torch.zeros((1, 2, 8, 8))
    targets[:, 0, 2, 2] = 1.0
    targets[:, 1, 5, 5] = 1.0
    supervision = torch.zeros_like(targets)
    supervision[:, 0, 2, 2] = 1.0
    supervision[:, 1, 5, 5] = 1.0

    loss = masked_multilabel_loss(
        logits, targets, skeleton, supervision_mask=supervision
    )
    loss.backward()

    assert torch.count_nonzero(logits.grad) == 2
    assert logits.grad[0, 0, 2, 2] != 0
    assert logits.grad[0, 1, 5, 5] != 0


def test_metrics_measure_safe_hatch_removal_without_structural_damage():
    skeleton = np.ones((1, 3), dtype=np.uint8)
    targets = np.asarray(
        [
            [[1, 1, 0]],
            [[0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    probabilities = np.asarray(
        [
            [[0.9, 0.9, 0.1]],
            [[0.1, 0.9, 0.9]],
        ],
        dtype=np.float32,
    )
    accumulator = StrokeMetricAccumulator()
    accumulator.update(probabilities, targets, skeleton)
    metrics = accumulator.compute()

    assert metrics["structural_recall"] == 1.0
    assert metrics["hatch_recall"] == 1.0
    assert metrics["overlap_recall"] == 1.0
    assert metrics["safe_removal_f1"] == 1.0
    assert metrics["safe_removal_structural_error_rate"] == 0.0


def test_metrics_ignore_predictions_without_channel_supervision():
    skeleton = np.ones((1, 3), dtype=np.uint8)
    targets = np.asarray(
        [[[1, 1, 1]], [[0, 0, 0]]], dtype=np.uint8
    )
    probabilities = np.asarray(
        [[[0.9, 0.1, 0.1]], [[0.1, 0.9, 0.9]]], dtype=np.float32
    )
    supervision = np.asarray(
        [[[1, 0, 0]], [[1, 0, 0]]], dtype=np.uint8
    )
    accumulator = StrokeMetricAccumulator()
    accumulator.update(probabilities, targets, skeleton, supervision)

    metrics = accumulator.compute()
    assert metrics["structural_recall"] == 1.0
    assert metrics["hatch_precision"] == 0.0
    assert metrics["skeleton_pixels"] == 1.0


def test_difficulty_recall_gate_supports_explicit_baseline_floors():
    metrics = {
        "medium": {"structural_recall": 0.9988},
        "very_hard": {"structural_recall": 0.9936},
    }

    eligible, effective = _difficulty_recall_gate(
        metrics,
        default_floor=0.995,
        floors={"very_hard": 0.9935},
    )

    assert eligible
    assert effective == {"medium": 0.995, "very_hard": 0.9935}


def test_difficulty_recall_floor_parser_rejects_invalid_or_unknown_groups():
    assert _difficulty_recall_floors("hard=0.996,very_hard=0.993") == {
        "hard": 0.996,
        "very_hard": 0.993,
    }
    with pytest.raises(ValueError, match="absent groups"):
        _difficulty_recall_gate(
            {"hard": {"structural_recall": 1.0}},
            default_floor=0.995,
            floors={"medium": 0.99},
        )


def test_checkpoint_rank_enforces_structural_recall_before_hatch_score():
    destructive = {
        "structural_recall": 0.990,
        "safe_removal_f1": 0.95,
        "overlap_recall": 0.95,
        "hatch_f1": 0.95,
    }
    conservative = {
        "structural_recall": 0.996,
        "safe_removal_f1": 0.20,
        "overlap_recall": 0.80,
        "hatch_f1": 0.70,
    }
    useful = {**conservative, "safe_removal_f1": 0.60}

    assert checkpoint_rank(conservative, 0.995) > checkpoint_rank(
        destructive, 0.995
    )
    assert checkpoint_rank(useful, 0.995) > checkpoint_rank(
        conservative, 0.995
    )


def test_structural_negative_weight_penalizes_hatch_only_false_positive():
    logits = torch.tensor([[[[5.0, 5.0]], [[-5.0, -5.0]]]])
    targets = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])
    skeleton = torch.ones((1, 1, 1, 2))
    supervision = torch.tensor([[[[1.0, 1.0]], [[0.0, 0.0]]]])

    baseline = masked_multilabel_loss(
        logits,
        targets,
        skeleton,
        supervision_mask=supervision,
        structural_weight=1.0,
        structural_negative_weight=1.0,
        hatch_weight=1.0,
        crossing_weight=1.0,
        bce_fraction=1.0,
    )
    negative_focused = masked_multilabel_loss(
        logits,
        targets,
        skeleton,
        supervision_mask=supervision,
        structural_weight=1.0,
        structural_negative_weight=4.0,
        hatch_weight=1.0,
        crossing_weight=1.0,
        bce_fraction=1.0,
    )

    assert negative_focused > baseline


def test_checkpoint_rank_enforces_structural_removal_error_gate():
    unsafe = {
        "structural_recall": 0.999,
        "safe_removal_structural_error_rate": 0.01,
        "safe_removal_f1": 0.95,
        "overlap_recall": 0.95,
        "hatch_f1": 0.95,
    }
    safe = {
        **unsafe,
        "safe_removal_structural_error_rate": 0.0005,
        "safe_removal_f1": 0.30,
    }

    assert checkpoint_rank(
        safe, 0.995, maximum_structural_error_rate=0.001
    ) > checkpoint_rank(
        unsafe, 0.995, maximum_structural_error_rate=0.001
    )


def test_safe_removal_histogram_calibrates_preservation_policy():
    skeleton = np.ones((1, 3), dtype=np.uint8)
    targets = np.asarray(
        [
            [[1, 1, 0]],
            [[0, 1, 1]],
        ],
        dtype=np.uint8,
    )
    probabilities = np.asarray(
        [
            [[0.9, 0.9, 0.1]],
            [[0.1, 0.9, 0.9]],
        ],
        dtype=np.float32,
    )
    histogram = SafeRemovalHistogram(resolution=100)
    histogram.update(probabilities, targets, skeleton)

    safe = histogram.metrics(0.8, 0.2)
    permissive = histogram.metrics(0.8, 1.0)
    assert safe["safe_removal_f1"] == 1.0
    assert safe["safe_removal_structural_error_rate"] == 0.0
    assert permissive["safe_removal_precision"] == 0.5
    assert permissive["safe_removal_structural_error_rate"] == 0.5


def test_difficulty_allocation_is_exact_and_weighted_to_very_hard():
    weights = parse_difficulty_weights(
        "medium=0.2,hard=0.3,very_hard=0.5"
    )
    counts = allocate_difficulty_counts(8_976, weights)

    assert sum(counts.values()) == 8_976
    assert counts == {"medium": 1795, "hard": 2693, "very_hard": 4488}


def test_domain_schedule_balances_real_supply_before_difficulty_supply():
    synthetic = [
        Path("a_medium.npz"), Path("b_hard.npz"), Path("c_very_hard.npz")
    ]
    dataset = HatchStrokePatchDataset(
        synthetic,
        real_paths=[Path("real.npz")],
        real_fraction=0.3,
        samples_per_epoch=10,
        difficulty_weights={"medium": 0.2, "hard": 0.3, "very_hard": 0.5},
    )

    assert dataset.source_counts == {"synthetic": 7, "real": 3}
    assert dataset.source_schedule.count("real") == 3
    assert len(dataset.difficulty_schedule) == 7
