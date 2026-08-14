import sqlite3

import numpy as np

from syntheticData.patentvec.hatch_stroke_targets import (
    validate_hatch_stroke_supervision,
)
from tools import export_real_hatch_stroke_dataset as exporter
from tools.export_real_hatch_stroke_dataset import build_real_targets


def test_real_targets_keep_crossing_multilabel_and_ignore_uncertain_region_ink():
    skeleton = np.zeros((21, 21), dtype=np.uint8)
    skeleton[10, 3:18] = 1
    skeleton[3:18, 10] = 1
    valid = np.ones_like(skeleton)
    region = np.zeros_like(skeleton)
    region[2:19, 2:19] = 1
    structural_support = np.zeros_like(skeleton)
    structural_support[3:18, 10] = 1
    hatch_support = np.zeros_like(skeleton)
    hatch_support[10, 3:18] = 1

    targets = build_real_targets(
        skeleton, valid, region, structural_support, hatch_support,
        boundary_ignore_radius=0, crossing_guard_radius=1,
    )
    stats = validate_hatch_stroke_supervision(targets)

    assert targets["structural_target"][10, 10] == 255
    assert targets["hatch_target"][10, 10] == 255
    assert targets["overlap_target"][10, 10] == 255
    assert targets["structural_target"][10, 4] == 0
    assert targets["hatch_target"][10, 4] == 255
    assert stats["overlap_pixels"] >= 1


def test_real_targets_supervise_negative_figure_as_structural_only():
    skeleton = np.zeros((16, 16), dtype=np.uint8)
    skeleton[8, 2:14] = 1
    valid = np.ones_like(skeleton)
    zeros = np.zeros_like(skeleton)

    targets = build_real_targets(skeleton, valid, zeros, zeros, zeros)
    supervision = targets["supervision_mask"] > 0

    assert np.all(targets["structural_target"][skeleton > 0] == 255)
    assert not np.any(targets["hatch_target"])
    assert np.all(supervision[:, skeleton > 0])


def test_real_targets_do_not_create_hatch_negatives_for_main_only_region_ink():
    skeleton = np.zeros((16, 16), dtype=np.uint8)
    skeleton[8, 2:14] = 1
    valid = np.ones_like(skeleton)
    region = np.ones_like(skeleton)
    structural_support = skeleton.copy()
    hatch_support = np.zeros_like(skeleton)

    targets = build_real_targets(
        skeleton, valid, region, structural_support, hatch_support,
        boundary_ignore_radius=0,
    )
    supervision = targets["supervision_mask"] > 0

    assert np.all(supervision[0, skeleton > 0])
    assert not np.any(supervision[1, skeleton > 0])


def test_boundary_ignore_is_computed_from_area_not_one_pixel_ink():
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    skeleton[16, 8:24] = 1
    valid = np.ones_like(skeleton)
    region = np.zeros_like(skeleton)
    region[6:26, 6:26] = 1
    hatch_support = skeleton.copy()

    targets = build_real_targets(
        skeleton,
        valid,
        region,
        np.zeros_like(skeleton),
        hatch_support,
        boundary_ignore_radius=3,
    )

    assert np.count_nonzero(targets["hatch_target"]) == 14
    assert np.all(targets["supervision_mask"][1, 16, 9:23] == 255)


def test_manual_teacher_enables_geometric_selector_for_nohachure_graph(
    monkeypatch,
):
    observed = {}

    def fake_selector(nodes, edges, region, config, *, pass_name):
        observed["remove_hachures"] = config.get("remove_hachures")
        observed["pass_name"] = pass_name
        return nodes, edges, [{"id": 7}]

    monkeypatch.setattr(
        exporter.stage2_stroke_extract,
        "_remove_hachures_cnn_geometric",
        fake_selector,
    )
    graph = {"image_shape": [8, 8], "nodes": [], "edges": []}

    recovered = exporter._manual_geometric_hatch_edges(
        graph,
        np.ones((8, 8), dtype=np.uint8),
        {"remove_hachures": False},
    )

    assert recovered == [{"id": 7}]
    assert observed == {
        "remove_hachures": True,
        "pass_name": "reviewed_manual_geometric",
    }


def test_consensus_teacher_keeps_only_periodic_candidate_pixels():
    region = np.zeros((9, 9), dtype=np.uint8)
    region[4, 1:8] = 1
    recovered = np.zeros_like(region)
    recovered[2, 2] = 1
    hough = np.zeros_like(region)
    hough[4, 3:6] = 1
    hough[2, 2] = 1

    combined = exporter._combine_hatch_teacher_support(
        region,
        recovered,
        hough,
        policy="consensus",
        consensus_radius=0,
    )

    assert np.count_nonzero(combined) == 4
    assert np.all(combined[4, 3:6] == 1)
    assert combined[2, 2] == 1
    assert combined[4, 1] == 0


def test_region_teacher_policy_preserves_all_candidate_support():
    region = np.zeros((5, 5), dtype=np.uint8)
    recovered = np.zeros_like(region)
    hough = np.zeros_like(region)
    region[1, 1] = 1
    recovered[3, 3] = 1

    combined = exporter._combine_hatch_teacher_support(
        region,
        recovered,
        hough,
        policy="region",
        consensus_radius=2,
    )

    assert np.count_nonzero(combined) == 2
    assert combined[1, 1] == 1
    assert combined[3, 3] == 1


def test_teacher_status_rejects_quality_gates():
    assert exporter._is_teacher_success_status("ok")
    assert exporter._is_teacher_success_status("ok_stage2")
    assert not exporter._is_teacher_success_status("quality_gate_stage1")
    assert not exporter._is_teacher_success_status("quality_gate_stage2")


def test_pipeline_status_loader_reads_sharded_databases(tmp_path):
    paths = [tmp_path / "results_cuda0.db", tmp_path / "results_cuda1.db"]
    for index, path in enumerate(paths):
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE results ("
                "patent_id TEXT, sketch_id TEXT, status TEXT)"
            )
            connection.execute(
                "INSERT INTO results VALUES (?, ?, ?)",
                (f"EP{index}", "F0001", "ok_stage2"),
            )

    statuses = exporter._load_pipeline_statuses(paths)

    assert statuses == {
        ("EP0", "F0001"): "ok_stage2",
        ("EP1", "F0001"): "ok_stage2",
    }
