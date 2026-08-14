"""Replay Drawing2CAD Stages 3-4 from archived Stage 2 graphs.

This evaluator is for rapid, paired Stage 3/4 iteration. It preserves the
source run's frozen raster inputs and stroke graphs, writes fresh primitives
and SVGs to a separate run directory, and updates a cloned evaluation DB with
new raster-fidelity metrics. A digest of the config and Stage 3/4 source code
makes the run safely resumable.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from tools import d2c_eval

stage3_primitive_fit = d2c_eval.stage3_primitive_fit
stage4_export = d2c_eval.stage4_export


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRIC_COLUMNS = (
    "iou_pixel",
    "iou_skeleton",
    "precision_pixel",
    "recall_pixel",
    "chamfer_sym",
    "chamfer_gt2out",
    "chamfer_out2gt",
    "chamfer_p95_sym",
)


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
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in connection.execute("PRAGMA table_info(d2c_results)")
    }
    additions = {
        "stage34_replay_digest": "TEXT",
        "stage34_replayed_at": "TEXT",
        "stage34_replay_error": "TEXT",
    }
    for column, column_type in additions.items():
        if column not in existing:
            connection.execute(
                f"ALTER TABLE d2c_results ADD COLUMN {column} {column_type}"
            )
    connection.commit()


def _absolute(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _replay_one(job: tuple[str, str, str, str, str, str | None]) -> dict:
    logging.getLogger().setLevel(logging.ERROR)
    source_str, target_str, config_str, sample_id, view, raster_str = job
    source = Path(source_str)
    target = Path(target_str)
    config_path = Path(config_str)
    folder_id = sample_id.replace("/", "_")
    sketch_id = f"{folder_id}_{view}"
    source_work = source / folder_id
    target_work = target / folder_id
    graph_path = source_work / "graphs" / f"{sketch_id}_graph.json"
    source_primitives = (
        source_work / "primitives" / f"{sketch_id}_primitives.json"
    )
    raster_path = _absolute(raster_str)
    output_raster = target_work / f"{sketch_id}_output.png"
    started = time.perf_counter()

    try:
        if not graph_path.exists():
            raise FileNotFoundError(f"missing Stage 2 graph: {graph_path}")
        if raster_path is None or not raster_path.exists():
            raise FileNotFoundError(f"missing GT raster: {raster_path}")

        stroke_width = None
        if source_primitives.exists():
            source_doc = json.loads(source_primitives.read_text())
            stroke_width = source_doc.get("stroke_width")
        config = yaml.safe_load(config_path.read_text())

        stage3_result = stage3_primitive_fit.run(
            graph_path,
            target_work,
            sketch_id,
            config,
            stroke_width=stroke_width,
        )
        stage4_result = stage4_export.run(
            stage3_result.primitives_path,
            target_work,
            sketch_id,
            formats=("svg",),
        )
        output_binary = d2c_eval._rasterize_svg_to_binary(
            stage4_result.svg_path,
            output_raster,
        )
        gt_binary = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
        if gt_binary is None:
            raise ValueError(f"could not read GT raster: {raster_path}")
        metrics = d2c_eval._compare_binary_rasters(gt_binary, output_binary)
        output_raster.unlink(missing_ok=True)
        return {
            "sample_id": sample_id,
            "view": view,
            "metrics": metrics,
            "s3_time": stage3_result.processing_time_s,
            "s4_time": stage4_result.processing_time_s,
            "n_prims_out": stage4_result.n_primitives_out,
            "output_svg_path": str(stage4_result.svg_path.resolve()),
            "elapsed": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        output_raster.unlink(missing_ok=True)
        return {
            "sample_id": sample_id,
            "view": view,
            "elapsed": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _metric_summary(
    connection: sqlite3.Connection,
    replay_digest: str,
) -> dict:
    summary = {}
    for column in METRIC_COLUMNS:
        count, mean = connection.execute(
            f"SELECT count({column}),avg({column}) FROM d2c_results "
            "WHERE status='ok' AND stage34_replay_digest=?",
            (replay_digest,),
        ).fetchone()
        summary[column] = {"n": int(count), "mean": mean}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay D2C Stage 3/4 from a completed run's graphs."
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_run = args.source_run.resolve()
    output_run = args.output_run.resolve()
    config_path = args.config.resolve()
    source_db = (args.source_db or source_run / "d2c_results.db").resolve()
    output_db = (args.output_db or output_run / "d2c_results.db").resolve()
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
        sql = (
            "SELECT sample_id,view,raster_path FROM d2c_results "
            "WHERE status='ok'"
        )
        parameters: tuple = ()
        if not args.force:
            sql += " AND COALESCE(stage34_replay_digest,'') != ?"
            parameters = (replay_digest,)
        sql += " ORDER BY sample_id,view"
        rows = connection.execute(sql, parameters).fetchall()

    if args.limit > 0:
        rows = rows[:args.limit]
    jobs = [
        (
            str(source_run),
            str(output_run),
            str(config_snapshot),
            sample_id,
            view,
            raster_path,
        )
        for sample_id, view, raster_path in rows
    ]

    completed = failed = 0
    errors = []
    replayed_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    metric_assignments = ",".join(
        f"{column}=?" for column in METRIC_COLUMNS
    )
    update_sql = (
        f"UPDATE d2c_results SET {metric_assignments},"
        "s3_time=?,s4_time=?,total_time=COALESCE(s1_time,0)+"
        "COALESCE(s2_time,0)+?+?,n_prims_out=?,output_svg_path=?,"
        "stage34_replay_digest=?,stage34_replayed_at=?,"
        "stage34_replay_error=NULL WHERE sample_id=? AND view=?"
    )

    logging.getLogger().setLevel(logging.WARNING)
    if jobs:
        with sqlite3.connect(output_db) as connection, ProcessPoolExecutor(
            max_workers=max(1, args.workers)
        ) as pool:
            futures = [pool.submit(_replay_one, job) for job in jobs]
            progress = tqdm(as_completed(futures), total=len(futures))
            for future in progress:
                result = future.result()
                if result["error"] is not None:
                    failed += 1
                    errors.append({
                        "sample_id": result["sample_id"],
                        "view": result["view"],
                        "error": result["error"],
                    })
                    connection.execute(
                        "UPDATE d2c_results SET stage34_replay_error=? "
                        "WHERE sample_id=? AND view=?",
                        (
                            result["error"],
                            result["sample_id"],
                            result["view"],
                        ),
                    )
                else:
                    completed += 1
                    metrics = result["metrics"]
                    connection.execute(
                        update_sql,
                        (
                            *(metrics[column] for column in METRIC_COLUMNS),
                            result["s3_time"],
                            result["s4_time"],
                            result["s3_time"],
                            result["s4_time"],
                            result["n_prims_out"],
                            result["output_svg_path"],
                            replay_digest,
                            replayed_at,
                            result["sample_id"],
                            result["view"],
                        ),
                    )
                if (completed + failed) % 100 == 0:
                    connection.commit()
                if (completed + failed) % 25 == 0:
                    progress.set_postfix(ok=completed, err=failed)
            connection.commit()

    with sqlite3.connect(output_db) as connection:
        summary = _metric_summary(connection, replay_digest)
        marked = connection.execute(
            "SELECT count(*) FROM d2c_results WHERE stage34_replay_digest=?",
            (replay_digest,),
        ).fetchone()[0]

    report = {
        "source_run": str(source_run),
        "output_run": str(output_run),
        "source_db": str(source_db),
        "output_db": str(output_db),
        "config": str(config_path),
        "config_snapshot": str(config_snapshot),
        "replay_digest": replay_digest,
        "requested": len(jobs),
        "completed": completed,
        "failed": failed,
        "marked_total": int(marked),
        "metrics": summary,
        "errors": errors[:100],
    }
    report_path = output_run / "stage34_replay_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
