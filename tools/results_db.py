"""
SQLite result store for the batch-evaluation driver.

One row per sketch. Schema kept flat (no separate per-stage tables) so that
analytic queries (histograms, percentiles, failure-mode breakdowns) are a
single SELECT.

Concurrency model: workers compute metrics, the main process writes. SQLite
WAL is enabled so a future change to direct multi-writer access doesn't
crash; the current driver only calls `insert_row` from the main process.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    patent_id        TEXT NOT NULL,
    sketch_id        TEXT NOT NULL,
    input_path       TEXT NOT NULL,

    -- Overall pipeline status: 'ok' if all four stages completed;
    -- otherwise the name of the stage that errored ('stage1', 'stage2', ...).
    status           TEXT NOT NULL,
    error            TEXT,                  -- exception repr, if any
    total_time       REAL,
    completed_at     TEXT NOT NULL,         -- ISO 8601 UTC

    -- Stage 1
    s1_time          REAL,
    s1_quality       REAL,
    s1_model_used    TEXT,
    s1_flagged       INTEGER,

    -- Stage 2
    s2_time          REAL,
    s2_keypoint_src  TEXT,
    s2_n_nodes       INTEGER,
    s2_n_edges       INTEGER,
    s2_isolation     REAL,
    s2_flagged       INTEGER,

    -- Stage 3
    s3_time          REAL,
    s3_n_primitives  INTEGER,
    s3_mean_conf     REAL,
    s3_flagged       INTEGER,

    -- Stage 4
    s4_time          REAL,
    s4_n_in          INTEGER,
    s4_n_out         INTEGER,
    s4_flagged       INTEGER,

    PRIMARY KEY (patent_id, sketch_id)
);

CREATE INDEX IF NOT EXISTS idx_status ON results(status);
CREATE INDEX IF NOT EXISTS idx_patent ON results(patent_id);
"""


# Columns in insertion order — keep aligned with the dict keys produced by
# the worker so insert_row is just `INSERT INTO results VALUES(...)`.
COLUMNS = [
    "patent_id", "sketch_id", "input_path",
    "status", "error", "total_time", "completed_at",
    "s1_time", "s1_quality", "s1_model_used", "s1_flagged",
    "s2_time", "s2_keypoint_src", "s2_n_nodes", "s2_n_edges",
    "s2_isolation", "s2_flagged",
    "s3_time", "s3_n_primitives", "s3_mean_conf", "s3_flagged",
    "s4_time", "s4_n_in", "s4_n_out", "s4_flagged",
]


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("PRAGMA journal_mode = WAL;")
        conn.executescript(SCHEMA)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        yield conn
    finally:
        conn.close()


def already_processed(conn: sqlite3.Connection,
                      patent_id: str,
                      sketch_id: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM results WHERE patent_id=? AND sketch_id=? LIMIT 1",
        (patent_id, sketch_id),
    )
    return cur.fetchone() is not None


def insert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    values = [row.get(c) for c in COLUMNS]
    placeholders = ",".join(["?"] * len(COLUMNS))
    cols = ",".join(COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO results ({cols}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def summarise(db_path: Path) -> dict[str, Any]:
    """Return a few quick aggregates for a CLI report after a run."""
    with connect(db_path) as conn:
        cur = conn.cursor()

        def scalar(sql: str) -> Any:
            r = cur.execute(sql).fetchone()
            return r[0] if r else None

        total      = scalar("SELECT COUNT(*) FROM results") or 0
        n_ok       = scalar("SELECT COUNT(*) FROM results WHERE status='ok'") or 0
        by_status  = dict(cur.execute(
            "SELECT status, COUNT(*) FROM results GROUP BY status"
        ).fetchall())
        mean_total = scalar("SELECT AVG(total_time) FROM results WHERE status='ok'")
        mean_s1    = scalar("SELECT AVG(s1_time)    FROM results WHERE status='ok'")
        mean_s2    = scalar("SELECT AVG(s2_time)    FROM results WHERE status='ok'")
        mean_s3    = scalar("SELECT AVG(s3_time)    FROM results WHERE status='ok'")
        mean_s4    = scalar("SELECT AVG(s4_time)    FROM results WHERE status='ok'")
        flag_s1    = scalar("SELECT 1.0*SUM(s1_flagged)/COUNT(*) FROM results WHERE status='ok'")
        flag_s2    = scalar("SELECT 1.0*SUM(s2_flagged)/COUNT(*) FROM results WHERE status='ok'")
        flag_s3    = scalar("SELECT 1.0*SUM(s3_flagged)/COUNT(*) FROM results WHERE status='ok'")
        flag_s4    = scalar("SELECT 1.0*SUM(s4_flagged)/COUNT(*) FROM results WHERE status='ok'")

    return {
        "total": total, "ok": n_ok, "by_status": by_status,
        "mean_total_s": mean_total,
        "mean_s1_s": mean_s1, "mean_s2_s": mean_s2,
        "mean_s3_s": mean_s3, "mean_s4_s": mean_s4,
        "flag_rate_s1": flag_s1, "flag_rate_s2": flag_s2,
        "flag_rate_s3": flag_s3, "flag_rate_s4": flag_s4,
    }
