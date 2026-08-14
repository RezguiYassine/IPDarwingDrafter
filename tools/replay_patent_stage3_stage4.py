"""Replay PatentData Stages 3-4 from archived Stage 2 graphs.

The normal batch runner intentionally recomputes Stage 2 after reusing Stage
0/1 artifacts. That is useful for end-to-end evaluation, but it cannot isolate
a Stage 3 policy when Stage 2 code has changed. This tool clones the source
database and freezes the graph, skeleton, and reference metadata instead.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import shutil
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from tools import batch_run


stage0_handle_references = batch_run.stage0_handle_references
stage3_primitive_fit = batch_run.stage3_primitive_fit
stage4_export = batch_run.stage4_export


def _clone_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_connection, sqlite3.connect(
        destination
    ) as destination_connection:
        source_connection.backup(destination_connection)


def _implementation_digest(config_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        config_path,
        Path(stage3_primitive_fit.__file__),
        Path(stage4_export.__file__),
        Path(stage0_handle_references.__file__),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in connection.execute("PRAGMA table_info(results)")
    }
    additions = {
        "stage34_replay_digest": "TEXT",
        "stage34_replayed_at": "TEXT",
        "stage34_replay_error": "TEXT",
    }
    for column, column_type in additions.items():
        if column not in existing:
            connection.execute(
                f"ALTER TABLE results ADD COLUMN {column} {column_type}"
            )
    connection.commit()


def _select_replay_rows(
    connection: sqlite3.Connection,
    replay_digest: str,
    *,
    force: bool,
) -> list[tuple[str, str, int]]:
    """Select only rows that reached Stage 3 in the frozen source run.

    The cloned database can also contain terminal Stage 1/2 quality-gate rows.
    Advancing those rows through Stage 3 would no longer be an exact replay and
    could incorrectly turn an upstream failure into an ``ok`` result.
    """
    sql = (
        "SELECT patent_id,sketch_id,"
        "COALESCE(s2_n_hachure_edges_removed,0) FROM results "
        "WHERE s3_time IS NOT NULL"
    )
    parameters: tuple = ()
    if not force:
        sql += " AND COALESCE(stage34_replay_digest,'') != ?"
        parameters = (replay_digest,)
    sql += " ORDER BY patent_id,sketch_id"
    return [
        (str(patent_id), str(sketch_id), int(hachures))
        for patent_id, sketch_id, hachures
        in connection.execute(sql, parameters).fetchall()
    ]


def _copy_matching(source: Path, target: Path, pattern: str) -> None:
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob(pattern):
        if path.is_file():
            shutil.copy2(path, target / path.name)


def _stage3_quality(
    result,
    primitives_doc: dict,
    config: dict,
    hachure_edges_removed: int,
) -> dict:
    stage3_config = config.get("stage3", {}) or {}
    confidence_threshold = float(
        stage3_config.get("confidence_threshold", 0.60)
    )
    quality_primitives = [
        primitive for primitive in primitives_doc.get("primitives", [])
        if primitive.get("style") != "hachure"
        and primitive.get("source") != "removed_hachure"
    ]
    confidences = [
        float(primitive.get("confidence", 0.0))
        for primitive in quality_primitives
    ]
    low_confidence_ratio = (
        sum(value < confidence_threshold for value in confidences)
        / len(confidences)
        if confidences else 0.0
    )
    quality_metrics = primitives_doc.get("quality_metrics", {})

    gates = (config.get("pipeline", {}) or {}).get("quality_gates", {}) or {}
    gates_enabled = bool(gates.get("enabled", False))
    max_primitives = int(gates.get("max_primitives", 0) or 0)
    max_low_confidence_ratio = float(
        gates.get("max_low_conf_ratio", 0.0) or 0.0
    )
    relax_min_edges = int(
        gates.get("min_hachure_edges_for_low_conf_relax", 0) or 0
    )
    relaxed_ratio = float(
        gates.get("max_low_conf_ratio_after_hachure", 0.0) or 0.0
    )
    effective_max_ratio = max_low_confidence_ratio
    if relaxed_ratio and relax_min_edges and hachure_edges_removed >= relax_min_edges:
        effective_max_ratio = max(effective_max_ratio, relaxed_ratio)

    failed_gate = gates_enabled and (
        result.flagged
        or (max_primitives and result.n_primitives > max_primitives)
        or (
            effective_max_ratio
            and low_confidence_ratio > effective_max_ratio
        )
    )
    error = None
    if failed_gate:
        error = (
            "primitive set not suitable for accurate CAD export: "
            f"n={result.n_primitives}, "
            f"mean_conf={result.mean_confidence:.3f}, "
            f"low_conf_ratio={low_confidence_ratio:.3f}, "
            f"max_low_conf_ratio={effective_max_ratio:.3f}"
        )
    return {
        "status": "quality_gate_stage3" if failed_gate else "ok",
        "error": error,
        "s3_time": result.processing_time_s,
        "s3_n_primitives": result.n_primitives,
        "s3_n_hachure_primitives": int(
            quality_metrics.get(
                "n_hachure_primitives", result.n_hachure_primitives,
            )
        ),
        "s3_mean_conf": result.mean_confidence,
        "s3_low_conf_ratio": low_confidence_ratio,
        "s3_flagged": int(result.flagged),
    }


def _replay_one(job: tuple[str, str, str, str, str, int]) -> dict:
    logging.getLogger().setLevel(logging.ERROR)
    source_value, target_value, config_value, patent_id, sketch_id, hachures = job
    source_work = Path(source_value) / patent_id
    target_work = Path(target_value) / patent_id
    config_path = Path(config_value)
    graph_path = source_work / "graphs" / f"{sketch_id}_graph.json"
    started = time.perf_counter()

    try:
        if not graph_path.exists():
            raise FileNotFoundError(f"missing Stage 2 graph: {graph_path}")
        config = yaml.safe_load(config_path.read_text()) or {}
        _copy_matching(
            source_work / "graphs", target_work / "graphs",
            f"{sketch_id}_graph.json",
        )
        _copy_matching(
            source_work / "cleaned", target_work / "cleaned",
            f"{sketch_id}_*",
        )
        _copy_matching(
            source_work / "references", target_work / "references",
            f"{sketch_id}_*",
        )

        stage3_result = stage3_primitive_fit.run(
            graph_path=graph_path,
            output_dir=target_work,
            sketch_id=sketch_id,
            config=config,
        )
        primitives_doc = json.loads(stage3_result.primitives_path.read_text())
        row = _stage3_quality(
            stage3_result, primitives_doc, config, hachures,
        )

        export_json = stage3_result.primitives_path
        references_path = (
            target_work / "references" / f"{sketch_id}_references.json"
        )
        if references_path.exists():
            export_json = (
                stage0_handle_references.attach_references_to_primitives(
                    primitives_path=stage3_result.primitives_path,
                    references_json_path=references_path,
                )
            )
        stage4_result = stage4_export.run(
            input_json=export_json,
            output_dir=target_work,
            sketch_id=sketch_id,
            formats=("svg", "dxf"),
            dxf_mode="patent",
        )
        row.update({
            "patent_id": patent_id,
            "sketch_id": sketch_id,
            "s4_time": stage4_result.processing_time_s,
            "s4_n_in": stage4_result.n_primitives_in,
            "s4_n_out": stage4_result.n_primitives_out,
            "s4_flagged": int(stage4_result.flagged),
            "elapsed": time.perf_counter() - started,
            "replay_error": None,
        })
        return row
    except Exception as exc:
        return {
            "patent_id": patent_id,
            "sketch_id": sketch_id,
            "elapsed": time.perf_counter() - started,
            "replay_error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay PatentData Stage 3/4 from archived Stage 2 graphs."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    output_run = args.output_run.resolve()
    config_path = args.config.resolve()
    source_db = (args.source_db or source_run / "results.db").resolve()
    output_db = (args.output_db or output_run / "results.db").resolve()
    if not source_db.exists():
        parser.error(f"source database does not exist: {source_db}")
    if not config_path.exists():
        parser.error(f"config does not exist: {config_path}")
    if source_db == output_db:
        parser.error("--output-db must differ from --source-db")

    output_run.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_run / "stage34_replay_config.yaml"
    config_snapshot.write_bytes(config_path.read_bytes())
    if not output_db.exists():
        _clone_database(source_db, output_db)

    replay_digest = _implementation_digest(config_snapshot)
    with sqlite3.connect(output_db) as connection:
        _ensure_columns(connection)
        source_rows = int(
            connection.execute("SELECT count(*) FROM results").fetchone()[0]
        )
        preserved_terminal_statuses = dict(connection.execute(
            "SELECT status,count(*) FROM results WHERE s3_time IS NULL "
            "GROUP BY status ORDER BY status"
        ).fetchall())
        rows = _select_replay_rows(
            connection, replay_digest, force=args.force,
        )

    if args.limit > 0:
        rows = rows[:args.limit]
    jobs = [
        (
            str(source_run), str(output_run), str(config_snapshot),
            patent_id, sketch_id, int(hachures),
        )
        for patent_id, sketch_id, hachures in rows
    ]

    completed = failed = 0
    errors = []
    replayed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(output_db) as connection, ProcessPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        futures = {executor.submit(_replay_one, job): job for job in jobs}
        with tqdm(total=len(futures), unit="view") as progress:
            for future in as_completed(futures):
                result = future.result()
                key = (result["patent_id"], result["sketch_id"])
                if result["replay_error"] is not None:
                    failed += 1
                    errors.append({"key": key, "error": result["replay_error"]})
                    connection.execute(
                        "UPDATE results SET stage34_replay_error=? "
                        "WHERE patent_id=? AND sketch_id=?",
                        (result["replay_error"], *key),
                    )
                else:
                    completed += 1
                    prior = connection.execute(
                        "SELECT s0_time,s1_time,s2_time FROM results "
                        "WHERE patent_id=? AND sketch_id=?", key,
                    ).fetchone()
                    total_time = sum(float(value or 0.0) for value in prior)
                    total_time += result["s3_time"] + result["s4_time"]
                    connection.execute(
                        "UPDATE results SET status=?,error=?,total_time=?,"
                        "s3_time=?,s3_n_primitives=?,s3_n_hachure_primitives=?,"
                        "s3_mean_conf=?,s3_low_conf_ratio=?,s3_flagged=?,"
                        "s4_time=?,s4_n_in=?,s4_n_out=?,s4_flagged=?,"
                        "stage34_replay_digest=?,stage34_replayed_at=?,"
                        "stage34_replay_error=NULL "
                        "WHERE patent_id=? AND sketch_id=?",
                        (
                            result["status"], result["error"], total_time,
                            result["s3_time"], result["s3_n_primitives"],
                            result["s3_n_hachure_primitives"],
                            result["s3_mean_conf"], result["s3_low_conf_ratio"],
                            result["s3_flagged"], result["s4_time"],
                            result["s4_n_in"], result["s4_n_out"],
                            result["s4_flagged"], replay_digest, replayed_at,
                            *key,
                        ),
                    )
                connection.commit()
                progress.update()
                progress.set_postfix(ok=completed, err=failed)

        status_counts = dict(connection.execute(
            "SELECT status,count(*) FROM results GROUP BY status"
        ).fetchall())
        marked_total = int(connection.execute(
            "SELECT count(*) FROM results WHERE stage34_replay_digest=?",
            (replay_digest,),
        ).fetchone()[0])

    report = {
        "source_run": str(source_run),
        "output_run": str(output_run),
        "source_db": str(source_db),
        "output_db": str(output_db),
        "config": str(config_path),
        "config_snapshot": str(config_snapshot),
        "replay_digest": replay_digest,
        "source_rows": source_rows,
        "eligible_rows": source_rows - sum(preserved_terminal_statuses.values()),
        "preserved_terminal_statuses": preserved_terminal_statuses,
        "requested": len(jobs),
        "completed": completed,
        "failed": failed,
        "marked_total": marked_total,
        "status_counts": status_counts,
        "errors": errors[:100],
    }
    report_path = output_run / "stage34_replay_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
