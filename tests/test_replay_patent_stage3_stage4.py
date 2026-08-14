import json
import sqlite3
from types import SimpleNamespace

from tools.replay_patent_stage3_stage4 import (
    _ensure_columns,
    _select_replay_rows,
    _stage3_quality,
)


def test_replay_selection_preserves_rows_that_never_reached_stage3():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE results ("
        "patent_id TEXT, sketch_id TEXT, status TEXT, s3_time REAL, "
        "s2_n_hachure_edges_removed INTEGER)"
    )
    connection.executemany(
        "INSERT INTO results VALUES (?,?,?,?,?)",
        [
            ("p1", "s1", "quality_gate_stage1", None, None),
            ("p2", "s2", "quality_gate_stage2", None, 2),
            ("p3", "s3", "ok", 1.0, 3),
            ("p4", "s4", "quality_gate_stage3", 2.0, 4),
        ],
    )
    _ensure_columns(connection)
    connection.execute(
        "UPDATE results SET stage34_replay_digest='current' "
        "WHERE patent_id='p4'"
    )

    assert _select_replay_rows(
        connection, "current", force=False,
    ) == [("p3", "s3", 3)]
    assert _select_replay_rows(
        connection, "current", force=True,
    ) == [("p3", "s3", 3), ("p4", "s4", 4)]


def test_stage3_quality_ignores_hachures_and_applies_relaxed_gate():
    result = SimpleNamespace(
        flagged=False,
        n_primitives=3,
        mean_confidence=0.7,
        processing_time_s=0.1,
        n_hachure_primitives=1,
    )
    document = {
        "primitives": [
            {"type": "line", "confidence": 0.9},
            {"type": "line", "confidence": 0.2},
            {"type": "line", "confidence": 0.1, "style": "hachure"},
        ],
        "quality_metrics": {"n_hachure_primitives": 1},
    }
    config = {
        "stage3": {"confidence_threshold": 0.6},
        "pipeline": {
            "quality_gates": {
                "enabled": True,
                "max_low_conf_ratio": 0.25,
                "min_hachure_edges_for_low_conf_relax": 10,
                "max_low_conf_ratio_after_hachure": 0.6,
            }
        },
    }

    strict = _stage3_quality(result, document, config, 0)
    relaxed = _stage3_quality(result, document, config, 10)

    assert strict["status"] == "quality_gate_stage3"
    assert strict["s3_low_conf_ratio"] == 0.5
    assert relaxed["status"] == "ok"
    assert relaxed["s3_n_hachure_primitives"] == 1
