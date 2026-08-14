from __future__ import annotations

import io
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from syntheticData.export_training_dataset import (
    _export_archive,
    _split_for,
)
from syntheticData.build_free2cad_ab_dataset import (
    _canonicalize_class_order,
    paired_allocation,
)
from syntheticData.compare_datasets import _paired_stats, _unpaired_stats
from syntheticData.build_dataset_audit import _stratified_rows


def _npz_bytes(**arrays) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def _sample_payload(source_index: int) -> dict[str, bytes]:
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    skeleton[16, 4:28] = 255
    puhachov = _npz_bytes(
        skeleton=skeleton,
        kps=np.asarray([[4, 16, 0], [27, 16, 0]], dtype=np.int32),
        meta=np.asarray("{}"),
    )
    masks = _npz_bytes(
        object=skeleton,
        hidden_center=np.zeros_like(skeleton),
        hatch=np.zeros_like(skeleton),
    )
    points = np.zeros((2, 64, 2), dtype=np.float32)
    points[:, :, 0] = np.linspace(0.0, 1.0, 64)
    points[1, :, 1] = np.linspace(0.0, 0.2, 64)
    free2cad = _npz_bytes(
        points=points,
        mask=np.ones((2, 64), dtype=bool),
        types=np.asarray([0, 3], dtype=np.uint8),
        params=np.zeros((2, 6), dtype=np.float32),
        source_index=np.full(2, source_index, dtype=np.int64),
        topology_projected=np.ones(2, dtype=np.uint8),
    )
    return {
        "puhachov.npz": puhachov,
        "masks.npz": masks,
        "free2cad_edges.npz": free2cad,
    }


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_source_index_split_is_stable_for_paired_curricula():
    actual = [_split_for(index, 10, 0) for index in range(1000)]
    assert actual == [_split_for(index, 10, 0) for index in range(1000)]
    assert set(actual) == {"train", "validation"}
    assert [_split_for(index, 1, 0) for index in range(5)] == [
        "validation"
    ] * 5


def test_free2cad_ab_allocation_uses_shared_minimum_synthetic_supply():
    base = np.asarray([100, 100, 100, 100, 100])
    synthetic_a = np.asarray([50, 4, 20, 40, 10])
    synthetic_b = np.asarray([40, 8, 10, 30, 12])

    target, base_take, synthetic_take = paired_allocation(
        base,
        synthetic_a,
        synthetic_b,
        target_per_class=100,
        synthetic_fraction=0.20,
        max_synthetic_repeats=2.0,
    )

    assert target == 100
    assert synthetic_take.tolist() == [20, 8, 20, 20, 20]
    assert (base_take + synthetic_take).tolist() == [100] * 5


def test_free2cad_ab_variants_receive_the_same_class_schedule():
    first = {
        "types": np.asarray([1, 0, 1, 0], dtype=np.uint8),
        "source_index": np.asarray([10, 11, 12, 13]),
    }
    second = {
        "types": np.asarray([0, 1, 0, 1], dtype=np.uint8),
        "source_index": np.asarray([20, 21, 22, 23]),
    }
    ordered_first = _canonicalize_class_order(first, seed=7)
    ordered_second = _canonicalize_class_order(second, seed=7)

    assert ordered_first["types"].tolist() == [0, 0, 1, 1]
    assert np.array_equal(ordered_first["types"], ordered_second["types"])
    assert sorted(ordered_first["source_index"].tolist()) == [10, 11, 12, 13]
    assert sorted(ordered_second["source_index"].tolist()) == [20, 21, 22, 23]


def test_paired_statistics_measure_within_seed_complexity_delta():
    stats = _paired_stats([4, 5, 6, 7], [6, 6, 9, 7])
    assert stats["baseline_mean"] == 5.5
    assert stats["enhanced_mean"] == 7.0
    assert stats["mean_delta"] == 1.5
    assert stats["enhanced_greater_fraction"] == 0.75


def test_unpaired_statistics_do_not_assume_retry_seed_pairing():
    stats = _unpaired_stats([1, 2, 3, 4], [3, 4, 5, 6])
    assert stats["baseline_mean"] == 2.5
    assert stats["enhanced_mean"] == 4.5
    assert stats["mean_delta"] == 2.0
    assert stats["mean_delta_ci95_low"] < 2.0 < stats["mean_delta_ci95_high"]


def test_visual_audit_stratification_does_not_alias_curriculum_period():
    rows = [
        {"source_index": index, "difficulty": difficulty}
        for index, difficulty in enumerate(
            ["medium", "hard", "hard", "very_hard"] * 10
        )
    ]
    selected = _stratified_rows(rows, per_difficulty=3)

    assert Counter(row["difficulty"] for row in selected) == {
        "medium": 3,
        "hard": 3,
        "very_hard": 3,
    }


def test_compact_archive_exports_to_both_native_training_layouts(tmp_path: Path):
    dataset = tmp_path / "compact"
    output = tmp_path / "training"
    (dataset / "shards").mkdir(parents=True)
    for path in (
        output / "stage2" / "train",
        output / "stage2" / "validation",
        output / "stage3" / "train",
        output / "stage3" / "val",
    ):
        path.mkdir(parents=True)

    archive_name = "shards/shard_000000.tar"
    rows = []
    with tarfile.open(dataset / archive_name, mode="w") as archive:
        for source_index in (0, 2):
            sample_id = f"pv_{source_index}"
            prefix = f"samples/{sample_id}"
            for filename, payload in _sample_payload(source_index).items():
                _add_bytes(archive, f"{prefix}/{filename}", payload)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "relative_path": prefix,
                    "member_prefix": prefix,
                    "difficulty": "medium" if source_index == 0 else "hard",
                }
            )

    marker = _export_archive(
        dataset,
        output,
        archive_name,
        rows,
        task_fingerprint="unit-fingerprint",
        validation_modulus=2,
        validation_fold=0,
        require_topology_projected=True,
    )

    assert marker["sample_counts"] == {"validation": 1, "train": 1}
    assert marker["difficulty_counts"] == {
        "train": {"medium": 1},
        "validation": {"hard": 1},
    }
    assert (output / "stage2" / "train" / "pv_0.npz").exists()
    assert (output / "stage2" / "validation" / "pv_2.npz").exists()
    with np.load(output / "stage3" / "val" / "shard_000000.npz") as data:
        assert data["types"].tolist() == [0, 3]
        assert data["sample_source_index"].tolist() == [2, 2]
        assert data["sample_ids"].tolist() == ["pv_2", "pv_2"]
    with np.load(output / "stage3" / "train" / "shard_000000.npz") as data:
        assert data["types"].tolist() == [0, 3]
        assert data["sample_source_index"].tolist() == [0, 0]


def test_compact_archive_can_export_relabelled_stage2_only(tmp_path: Path):
    dataset = tmp_path / "compact"
    output = tmp_path / "training"
    (dataset / "shards").mkdir(parents=True)
    (output / "stage2" / "validation").mkdir(parents=True)
    archive_name = "shards/shard_000000.tar"
    prefix = "samples/pv_0"
    with tarfile.open(dataset / archive_name, mode="w") as archive:
        for filename, payload in _sample_payload(0).items():
            _add_bytes(archive, f"{prefix}/{filename}", payload)

    marker = _export_archive(
        dataset,
        output,
        archive_name,
        [{
            "sample_id": "pv_0",
            "source_index": 0,
            "relative_path": prefix,
            "member_prefix": prefix,
            "difficulty": "medium",
        }],
        task_fingerprint="unit-topology-fingerprint",
        validation_modulus=1,
        validation_fold=0,
        require_topology_projected=True,
        stage2_label_contract="supported-raster-topology-v1",
        stage2_only=True,
    )

    assert marker["stage2_only"] is True
    assert marker["stage2_keypoint_counts"] == {
        "endpoint": 2, "junction": 0, "corner": 0, "total": 2,
    }
    assert not (output / "stage3").exists()
    with np.load(output / "stage2" / "validation" / "pv_0.npz") as data:
        assert data["kps"].tolist() == [[4, 16, 0], [27, 16, 0]]
        assert "supported-raster-topology-v1" in str(data["meta"])
