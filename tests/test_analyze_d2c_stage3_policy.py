import json
import sqlite3

from tools.analyze_d2c_stage3_policy import analyze


def _database(path, rows):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE d2c_results ("
            "sample_id TEXT, view TEXT, status TEXT, chamfer_sym REAL, "
            "chamfer_p95_sym REAL, iou_pixel REAL, iou_skeleton REAL, "
            "precision_pixel REAL, recall_pixel REAL)"
        )
        connection.executemany(
            "INSERT INTO d2c_results VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )


def _primitives(run, sample_id, view, primitive):
    folder = sample_id.replace("/", "_")
    target = run / folder / "primitives"
    target.mkdir(parents=True)
    (target / f"{folder}_{view}_primitives.json").write_text(json.dumps({
        "primitives": [primitive],
        "quality_metrics": {
            "n_closed_trace_source_points": 0,
            "n_closed_trace_output_vertices": 0,
        },
    }))


def test_policy_analysis_pairs_rows_and_classifies_promotions(tmp_path):
    baseline = tmp_path / "baseline.db"
    candidate = tmp_path / "candidate.db"
    values = (1.0, 2.0, 0.5, 0.6, 0.7, 0.8)
    _database(baseline, [("0000/00000001", "Front", "ok", *values)])
    _database(candidate, [
        ("0000/00000001", "Front", "ok", 0.5, 1.0, 0.6, 0.7, 0.8, 0.9)
    ])
    run = tmp_path / "candidate"
    _primitives(run, "0000/00000001", "Front", {
        "type": "path",
        "segments": [{"type": "line"}, {"type": "line"}],
        "fit_metadata": {
            "strategy": "compound_over_weak",
            "path_atoms": 2,
        },
        "fit_fidelity": {
            "max_segment_p95": 0.5,
            "max_endpoint_error": 1.0,
        },
    })

    report, outliers = analyze(
        baseline,
        candidate,
        run,
        baseline_run=None,
        bootstrap_samples=100,
        seed=7,
    )

    assert report["paired"] == 1
    assert report["groups"] == {"open_all_line": 1}
    assert report["overall"]["chamfer_sym"]["mean_delta"] == -0.5
    assert report["activation"]["open_atoms"] == 2
    assert report["outlier_counts"]["improved"] == 1
    assert report["tradeoff_counts"]["all_metrics"] == {
        "pareto_improved": 1,
    }
    assert outliers[0]["metrics"]["chamfer_sym"]["delta"] == -0.5


def test_policy_analysis_separates_chamfer_tradeoff_from_primary_failure(
    tmp_path,
):
    baseline = tmp_path / "baseline.db"
    candidate = tmp_path / "candidate.db"
    rows = [
        ("0000/00000001", "Front", "ok", 1.0, 3.0, 0.5, 0.6, 0.7, 0.8),
        ("0000/00000002", "Front", "ok", 1.0, 3.0, 0.5, 0.6, 0.7, 0.8),
    ]
    _database(baseline, rows)
    _database(candidate, [
        # Mean Chamfer worsens, but tail distance and rendered overlap improve.
        ("0000/00000001", "Front", "ok", 1.2, 2.0, 0.6, 0.5, 0.8, 0.9),
        # All three primary metrics genuinely regress.
        ("0000/00000002", "Front", "ok", 1.3, 4.0, 0.4, 0.5, 0.8, 0.9),
    ])
    run = tmp_path / "candidate"
    for sample_id, *_ in rows:
        _primitives(run, sample_id, "Front", {"type": "line"})

    report, outliers = analyze(
        baseline,
        candidate,
        run,
        baseline_run=None,
        bootstrap_samples=10,
        seed=7,
    )

    tail = report["tradeoff_counts"]["mean_chamfer_gt_0_1"]
    assert tail == {
        "views": 2,
        "p95_better": 1,
        "pixel_iou_better": 1,
        "p95_and_pixel_iou_better": 1,
        "all_three_primary_worse": 1,
    }
    assert all(
        record["metric_outcome"]["classification"] == "mixed"
        for record in outliers
    )
