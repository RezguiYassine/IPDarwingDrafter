import sqlite3

from tools.compare_d2c_runs import compare
from tools.d2c_eval import init_db, insert_row


def _row(sample_id, chamfer, edges):
    return {
        "sample_id": sample_id,
        "view": "Front",
        "status": "ok",
        "completed_at": "2026-07-18T00:00:00",
        "n_strokes_gt": 2,
        "chamfer_sym": chamfer,
        "n_edges": edges,
        "n_prims_out": edges,
    }


def test_d2c_schema_migrates_topology_columns(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE d2c_results (sample_id TEXT, view TEXT, status TEXT)"
        )

    init_db(path)

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(d2c_results)")
        }
    assert {"n_edges", "median_edge_length", "micro_edge_ratio"} <= columns


def test_compare_d2c_runs_uses_only_paired_ok_rows(tmp_path):
    baseline_path = tmp_path / "baseline.db"
    candidate_path = tmp_path / "candidate.db"
    init_db(baseline_path)
    init_db(candidate_path)
    with sqlite3.connect(baseline_path) as connection:
        insert_row(connection, _row("a", chamfer=2.0, edges=4))
        insert_row(connection, _row("b", chamfer=1.0, edges=2))
    with sqlite3.connect(candidate_path) as connection:
        insert_row(connection, _row("a", chamfer=1.0, edges=2))
        insert_row(connection, _row("b", chamfer=2.0, edges=2))
        insert_row(connection, _row("unpaired", chamfer=0.1, edges=1))

    report = compare(baseline_path, candidate_path)

    assert report["paired"] == 2
    assert report["metrics"]["chamfer_sym"]["baseline_mean"] == 1.5
    assert report["metrics"]["chamfer_sym"]["candidate_mean"] == 1.5
    assert report["metrics"]["chamfer_sym"]["paired_win_rate"] == 0.5
    assert report["metrics"]["edge_inflation"]["candidate_mean"] == 1.0
