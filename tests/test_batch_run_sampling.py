import json
from pathlib import Path

import pytest

from tools import batch_run, results_db
from tools.batch_run import (
    _is_success_status,
    _manifest_worklist,
    _preprocessing_config,
    _select_sketches,
    _source_worklist,
    _stratified_sample,
)


def _touch_tif(root: Path, patent: str, filename: str) -> Path:
    path = root / patent / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_limit_after_filter_returns_requested_kept_count(tmp_path):
    paths = [
        _touch_tif(tmp_path, f"EP{index}", f"EP{index}_A0001.tif")
        for index in range(5)
    ]

    selected = _select_sketches(
        tmp_path,
        limit=3,
        stratified=False,
        seed=7,
        excluded_paths={str(paths[0]), str(paths[2])},
    )

    assert [path for _patent, _sketch, path in selected] == [
        paths[1], paths[3], paths[4],
    ]


def test_stage_limited_terminal_status_is_successful():
    assert _is_success_status("ok")
    assert _is_success_status("ok_stage2")
    assert not _is_success_status("quality_gate_stage2")


def test_stratified_filter_uses_first_kept_tif_per_patent(tmp_path):
    discarded = _touch_tif(tmp_path, "EP1", "EP1_A0001.tif")
    kept = _touch_tif(tmp_path, "EP1", "EP1_A0002.tif")
    _touch_tif(tmp_path, "EP2", "EP2_A0001.tif")

    selected = _stratified_sample(
        tmp_path, 2, seed=11, excluded_paths={str(discarded)},
    )

    ep1 = [path for patent, _sketch, path in selected if patent == "EP1"]
    assert ep1 == [kept]


def test_preprocessing_config_ignores_stage2_model_changes():
    base = {
        "stage0": {"enabled": True, "ocr_conf": 0.3},
        "stage1": {"threshold": 0.5},
        "sketchcleannet": {"weights": "clean.pth"},
        "puhachov": {"weights": "model_a.pth"},
    }
    candidate = {
        **base,
        "puhachov": {"weights": "model_b.pth"},
    }

    assert _preprocessing_config(base) == _preprocessing_config(candidate)


def test_source_worklist_replays_exact_rows_in_stable_order(tmp_path):
    source_db = tmp_path / "source" / "results.db"
    results_db.init_db(source_db)
    inputs = {
        "EP2": _touch_tif(tmp_path, "EP2", "EP2_F0002.tif"),
        "EP1": _touch_tif(tmp_path, "EP1", "EP1_F0001.tif"),
    }
    with results_db.connect(source_db) as connection:
        for patent_id in ("EP2", "EP1"):
            results_db.insert_row(connection, {
                "patent_id": patent_id,
                "sketch_id": inputs[patent_id].stem.split("_", 1)[1],
                "input_path": str(inputs[patent_id]),
                "status": "ok",
                "completed_at": "2026-07-31T00:00:00",
            })

    assert _source_worklist(source_db, limit=1) == [
        ("EP1", "F0001", inputs["EP1"]),
    ]


def test_manifest_worklist_preserves_order_and_derives_input_paths(tmp_path):
    patent_root = tmp_path / "patents"
    second = _touch_tif(patent_root, "EP2", "EP2_F0002.tif")
    first = _touch_tif(patent_root, "EP1", "EP1_F0001.tif")
    manifest = tmp_path / "reviewed.csv"
    manifest.write_text(
        "patent_id,sketch_id,input_path\n"
        "EP2,F0002,\n"
        f"EP1,F0001,{first}\n"
    )

    assert _manifest_worklist(manifest, patent_root) == [
        ("EP2", "F0002", second),
        ("EP1", "F0001", first),
    ]
    assert _manifest_worklist(manifest, patent_root, limit=1) == [
        ("EP2", "F0002", second),
    ]


def test_manifest_worklist_rejects_duplicate_figure_identity(tmp_path):
    manifest = tmp_path / "duplicates.csv"
    manifest.write_text(
        "patent_id,sketch_id\n"
        "EP1,F0001\n"
        "EP1,F0001\n"
    )

    with pytest.raises(ValueError, match="duplicates EP1/F0001"):
        _manifest_worklist(manifest, tmp_path)


def test_copy_reused_preprocessing_materialises_self_contained_paths(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "source"
    source_patent = source_root / "EP1"
    source_refs = source_patent / "references"
    source_crops = source_refs / "crops"
    source_cleaned = source_patent / "cleaned"
    source_crops.mkdir(parents=True)
    source_cleaned.mkdir(parents=True)

    sketch_id = "A0001"
    input_path = tmp_path / "EP1_A0001.tif"
    input_path.write_bytes(b"tif")
    for path in (
        source_refs / f"{sketch_id}_norefs.png",
        source_refs / f"{sketch_id}_references_mask.png",
        source_cleaned / f"{sketch_id}_cleaned.png",
        source_cleaned / f"{sketch_id}_skeleton.png",
    ):
        path.write_bytes(path.name.encode())
    source_crop = source_crops / f"{sketch_id}_ref_001.png"
    source_crop.write_bytes(b"crop")
    source_json = source_refs / f"{sketch_id}_references.json"
    source_json.write_text(json.dumps({
        "mask_path": str(source_refs / f"{sketch_id}_references_mask.png"),
        "reference_free_path": str(source_refs / f"{sketch_id}_norefs.png"),
        "reference_labels": [{"crop_path": str(source_crop)}],
    }))

    source_db = source_root / "results.db"
    results_db.init_db(source_db)
    with results_db.connect(source_db) as conn:
        results_db.insert_row(conn, {
            "patent_id": "EP1",
            "sketch_id": sketch_id,
            "input_path": str(input_path),
            "status": "ok",
            "completed_at": "2026-07-31T00:00:00",
            "s0_time": 2.5,
            "s0_n_labels": 1,
            "s0_n_leaders": 1,
            "s0_n_iterations": 1,
            "s0_removed_ink_ratio": 0.02,
            "s0_active_removal": 1,
            "s0_flagged": 0,
            "s1_time": 1.5,
            "s1_quality": 0.95,
            "s1_model_used": "sketchcleannet",
            "s1_flagged": 0,
        })

    monkeypatch.setattr(
        batch_run, "_WORKER_REUSE_PREPROCESSING_ROOT", source_root,
    )
    monkeypatch.setattr(
        batch_run, "_WORKER_REUSE_PREPROCESSING_DB", source_db,
    )
    target = tmp_path / "target" / "EP1"
    s0, s1 = batch_run._copy_reused_preprocessing(
        "EP1", sketch_id, input_path, target,
    )

    assert s0.processing_time_s == 2.5
    assert s1.processing_time_s == 1.5
    assert s1.skeleton_path.exists()
    copied_doc = json.loads(s0.references_json_path.read_text())
    assert copied_doc["reference_free_path"] == str(s0.reference_free_path)
    copied_crop = Path(copied_doc["reference_labels"][0]["crop_path"])
    assert copied_crop.exists()
    assert target in copied_crop.parents


def test_reused_preprocessing_preserves_source_terminal_failure(
    tmp_path, monkeypatch,
):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_db = source_root / "results.db"
    input_path = tmp_path / "EP1_A0001.tif"
    input_path.write_bytes(b"tif")
    results_db.init_db(source_db)
    with results_db.connect(source_db) as connection:
        results_db.insert_row(connection, {
            "patent_id": "EP1",
            "sketch_id": "A0001",
            "input_path": str(input_path),
            "status": "stage0",
            "error": "source failure",
            "total_time": 3.0,
            "completed_at": "2026-07-31T00:00:00",
        })
    monkeypatch.setattr(
        batch_run, "_WORKER_REUSE_PREPROCESSING_ROOT", source_root,
    )
    monkeypatch.setattr(
        batch_run, "_WORKER_REUSE_PREPROCESSING_DB", source_db,
    )

    with pytest.raises(batch_run._ReusedPreprocessingFailure) as caught:
        batch_run._copy_reused_preprocessing(
            "EP1", "A0001", input_path, tmp_path / "target",
        )

    assert caught.value.source_row["status"] == "stage0"
    assert caught.value.source_row["error"] == "source failure"
