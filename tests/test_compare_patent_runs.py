from tools import results_db
import csv

from tools.compare_patent_runs import _write_csv, compare


def _make_db(path, micro_values):
    results_db.init_db(path)
    with results_db.connect(path) as connection:
        for index, micro in enumerate(micro_values):
            results_db.insert_row(connection, {
                "patent_id": f"EP{index}",
                "sketch_id": "A0001",
                "input_path": f"EP{index}_A0001.tif",
                "status": "ok",
                "completed_at": "2026-07-31T00:00:00",
                "s0_n_labels": index,
                "s0_n_leaders": index,
                "s0_n_iterations": 1,
                "s0_removed_ink_ratio": 0.01,
                "s0_active_removal": 1,
                "s0_flagged": 0,
                "s1_quality": 0.95,
                "s1_model_used": "sketchcleannet",
                "s1_flagged": 0,
                "s2_keypoint_src": "puhachov_cnn_tiled",
                "s2_n_edges": 20 + index,
                "s2_median_edge_len": 10.0 + index,
                "s2_micro_edge_ratio": micro,
                "s2_short_edge_ratio": micro + 0.1,
                "s2_isolation": 0.0,
                "s3_n_primitives": 10,
                "s3_mean_conf": 0.8,
                "s3_low_conf_ratio": 0.1,
            })


def test_compare_reports_paired_direction_and_preprocessing_identity(tmp_path):
    baseline = tmp_path / "baseline.db"
    candidate = tmp_path / "candidate.db"
    _make_db(baseline, [0.4, 0.2])
    _make_db(candidate, [0.2, 0.1])

    report, rows = compare(
        [("baseline", baseline), ("candidate", candidate)],
        seed=7,
        bootstrap_samples=100,
    )

    comparison = report["comparisons"]["candidate"]
    metric = comparison["metrics"]["s2_micro_edge_ratio"]
    assert report["shared_all_runs"] == 2
    assert comparison["preprocessing_exact_match"] is True
    assert metric["mean_delta"] < 0
    assert metric["better_count"] == 2
    assert len(rows) == 2


def test_csv_schema_includes_fields_missing_from_first_row(tmp_path):
    output = tmp_path / "report.csv"

    _write_csv(output, [{"id": "first"}, {"id": "second", "delta": 1.0}])

    with output.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["delta"] == ""
    assert rows[1]["delta"] == "1.0"
