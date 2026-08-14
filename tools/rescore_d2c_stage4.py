"""Re-export and re-score Drawing2CAD runs without repeating Stages 1-3.

This is intended for exporter-only changes. It clones the original evaluation
database, regenerates each SVG from the archived Stage 3 primitive JSON, and
updates only Stage 4 timing and raster-fidelity metrics in the cloned database.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
from tqdm import tqdm

from tools import d2c_eval

stage4_export = d2c_eval.stage4_export


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


def _ensure_rescore_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in connection.execute("PRAGMA table_info(d2c_results)")
    }
    additions = {
        "stage4_pixel_center_offset": "REAL",
        "stage4_export_digest": "TEXT",
        "stage4_rescored_at": "TEXT",
        "stage4_rescore_error": "TEXT",
    }
    for column, column_type in additions.items():
        if column not in existing:
            connection.execute(
                f"ALTER TABLE d2c_results ADD COLUMN {column} {column_type}"
            )
    connection.commit()


def _rescore_one(job: tuple[str, str, str, str, str | None]) -> dict:
    run_dir_str, sample_id, view, raster_path_str, input_svg_path_str = job
    run_dir = Path(run_dir_str)
    folder_id = sample_id.replace("/", "_")
    sketch_id = f"{folder_id}_{view}"
    work_dir = run_dir / folder_id
    primitives_path = work_dir / "primitives" / f"{sketch_id}_primitives.json"
    raster_path = Path(raster_path_str) if raster_path_str else None
    if raster_path is None or not raster_path.exists():
        raster_path = work_dir / f"{sketch_id}_input.png"

    started = time.perf_counter()
    try:
        if not primitives_path.exists():
            raise FileNotFoundError(f"missing primitive JSON: {primitives_path}")
        if not raster_path.exists():
            if not input_svg_path_str:
                raise FileNotFoundError(f"missing GT raster: {raster_path}")
            d2c_eval._rasterize_svg_to_binary(
                Path(input_svg_path_str), raster_path
            )

        export = stage4_export.run(
            input_json=primitives_path,
            output_dir=work_dir,
            sketch_id=sketch_id,
            formats=("svg",),
            dxf_mode="basic",
        )
        output_raster = work_dir / f"{sketch_id}_output.png"
        output_binary = d2c_eval._rasterize_svg_to_binary(
            export.svg_path, output_raster
        )
        gt_binary = cv2.imread(str(raster_path), cv2.IMREAD_GRAYSCALE)
        if gt_binary is None:
            raise ValueError(f"could not read GT raster: {raster_path}")
        metrics = d2c_eval._compare_binary_rasters(gt_binary, output_binary)
        return {
            "sample_id": sample_id,
            "view": view,
            "metrics": metrics,
            "s4_time": export.processing_time_s,
            "output_svg_path": str(export.svg_path),
            "elapsed": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "view": view,
            "metrics": None,
            "s4_time": None,
            "output_svg_path": None,
            "elapsed": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _metric_summary(connection: sqlite3.Connection) -> dict:
    result = {}
    for column in METRIC_COLUMNS:
        row = connection.execute(
            f"SELECT count({column}), avg({column}) FROM d2c_results "
            "WHERE status='ok'"
        ).fetchone()
        result[column] = {"n": int(row[0]), "mean": row[1]}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-export and re-score a completed Drawing2CAD run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--output-db", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-score rows already marked with the current pixel-center offset.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    source_db = (args.source_db or run_dir / "d2c_results.db").resolve()
    output_db = (
        args.output_db or run_dir / "d2c_results_pixel_center.db"
    ).resolve()
    if not source_db.exists():
        parser.error(f"source database does not exist: {source_db}")
    if source_db == output_db:
        parser.error("--output-db must differ from --source-db")
    if not output_db.exists():
        _clone_database(source_db, output_db)

    offset = float(stage4_export.PIXEL_CENTER_OFFSET)
    export_digest = hashlib.sha256(
        Path(stage4_export.__file__).read_bytes()
    ).hexdigest()
    with sqlite3.connect(output_db) as connection:
        _ensure_rescore_columns(connection)
        before = _metric_summary(connection)
        sql = (
            "SELECT sample_id,view,raster_path,input_svg_path "
            "FROM d2c_results WHERE status='ok'"
        )
        parameters: tuple = ()
        if not args.force:
            sql += (
                " AND (stage4_pixel_center_offset IS NULL "
                "OR abs(stage4_pixel_center_offset - ?) > 1e-12 "
                "OR COALESCE(stage4_export_digest,'') != ?)"
            )
            parameters = (offset, export_digest)
        sql += " ORDER BY sample_id,view"
        rows = connection.execute(sql, parameters).fetchall()

    if args.limit > 0:
        rows = rows[: args.limit]
    jobs = [
        (str(run_dir), sample_id, view, raster_path, input_svg_path)
        for sample_id, view, raster_path, input_svg_path in rows
    ]

    completed = failed = 0
    errors = []
    rescored_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    update_sql = (
        "UPDATE d2c_results SET "
        + ",".join(f"{column}=?" for column in METRIC_COLUMNS)
        + ",s4_time=?,output_svg_path=?,stage4_pixel_center_offset=?,"
        "stage4_export_digest=?,stage4_rescored_at=?,stage4_rescore_error=NULL "
        "WHERE sample_id=? AND view=?"
    )

    if jobs:
        with sqlite3.connect(output_db) as connection, ProcessPoolExecutor(
            max_workers=max(1, args.workers)
        ) as pool:
            futures = [pool.submit(_rescore_one, job) for job in jobs]
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
                        "UPDATE d2c_results SET stage4_rescore_error=? "
                        "WHERE sample_id=? AND view=?",
                        (result["error"], result["sample_id"], result["view"]),
                    )
                else:
                    completed += 1
                    metrics = result["metrics"]
                    connection.execute(
                        update_sql,
                        (
                            *(metrics[column] for column in METRIC_COLUMNS),
                            result["s4_time"],
                            result["output_svg_path"],
                            offset,
                            export_digest,
                            rescored_at,
                            result["sample_id"],
                            result["view"],
                        ),
                    )
                if (completed + failed) % 100 == 0:
                    connection.commit()
                progress.set_postfix(ok=completed, err=failed)
            connection.commit()

    with sqlite3.connect(output_db) as connection:
        after = _metric_summary(connection)
        marked = connection.execute(
            "SELECT count(*) FROM d2c_results "
            "WHERE stage4_pixel_center_offset=?",
            (offset,),
        ).fetchone()[0]

    report = {
        "run_dir": str(run_dir),
        "source_db": str(source_db),
        "output_db": str(output_db),
        "pixel_center_offset": offset,
        "stage4_export_digest": export_digest,
        "requested": len(jobs),
        "completed": completed,
        "failed": failed,
        "marked_total": int(marked),
        "before": before,
        "after": after,
        "errors": errors[:100],
    }
    report_path = output_db.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
