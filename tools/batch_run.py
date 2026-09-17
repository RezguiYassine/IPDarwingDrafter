"""
Batch driver for evaluating the AP3 vectorization pipeline on a corpus of
patent sketches (Phase 0 + Phase 1 of the evaluation roadmap).

Reads sketches from a directory of patent subfolders, runs Stage 0 plus the
four vectorization stages
per sketch, and writes one row of intrinsic metrics per sketch to a SQLite
results DB. Resumable: sketches whose row already exists in the DB are
skipped (unless --no-resume is given).

Usage from project root:

    # Phase 0 pilot — 100 sketches, one per random patent
    python -m tools.batch_run --limit 100 --stratified

    # Exactly 100 kept drawings after applying a content filter
    python -m tools.batch_run --limit 100 --stratified \\
        --filter-manifest output/PatentData/filter_manifest_clean12.csv \\
        --limit-after-filter

    # Paired model arm: copy identical Stage-0/1 artifacts, rerun Stage 2 onward
    python -m tools.batch_run --limit 100 --stratified --limit-after-filter \\
        --filter-manifest output/PatentData/filter_manifest_clean12.csv \\
        --reuse-preprocessing-from output/PatentData100_source \\
        --reuse-preprocessing-config config_source.yaml \\
        --config config_candidate.yaml --output output/PatentData100_candidate

    # Exact paired arm: source DB defines the cohort despite later filter drift
    python -m tools.batch_run --reuse-source-worklist --limit 100 \\
        --reuse-preprocessing-from output/PatentData100_source \\
        --reuse-preprocessing-config config_source.yaml \\
        --config config_candidate.yaml --output output/PatentData100_candidate

    # Exact external cohort: CSV columns patent_id, sketch_id, optional input_path
    python -m tools.batch_run \\
        --worklist output/reviewed_patent_figures.csv \\
        --config config_candidate.yaml --output output/reviewed_patent_figures

    # Phase 1 full corpus
    python -m tools.batch_run --workers 8

    # Custom paths
    python -m tools.batch_run \\
        --patent-root data/PatentData/ReorganisedData \\
        --output      output/PatentData \\
        --db          output/PatentData/results.db \\
        --workers     8
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import os
import random
import shutil
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm

# ─── Make each stage module importable as a plain Python module ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("stage0_handling_references", "stage1_preprocessing", "stage2_strokeextraction",
            "stage3_primitivesfitting", "stage4_export"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

import stage0_handle_references    # noqa: E402
import stage1_preprocess           # noqa: E402
import stage2_stroke_extract       # noqa: E402
import stage3_primitive_fit        # noqa: E402
import stage4_export               # noqa: E402

from tools import acceptance, content_routing, deployment, results_db   # noqa: E402


logger = logging.getLogger("batch_run")


def _is_success_status(status: str) -> bool:
    return status == "ok" or status.startswith("ok_stage")


# ─── Worker globals (one set per process, lazy-loaded at first task) ─────────

_WORKER_CFG = None
_WORKER_DEPLOYMENT_IDENTITY = None
_WORKER_S1_MODEL = None
_WORKER_S2_MODEL = None
_WORKER_HATCH_MODEL = None
_WORKER_HATCH_STROKE_MODEL = None
_WORKER_REUSE_PREPROCESSING_ROOT: Path | None = None
_WORKER_REUSE_PREPROCESSING_DB: Path | None = None
_WORKER_CONTENT_MANIFEST: dict | None = None

_PREPROCESSING_ROW_FIELDS = (
    "s0_time", "s0_n_labels", "s0_n_leaders", "s0_n_iterations",
    "s0_removed_ink_ratio", "s0_active_removal", "s0_flagged",
    "s1_time", "s1_quality", "s1_model_used", "s1_flagged",
)


class _ReusedPreprocessingFailure(RuntimeError):
    def __init__(self, source_row: dict) -> None:
        super().__init__(
            f"source preprocessing ended with status={source_row['status']}"
        )
        self.source_row = source_row


def _worker_init(
    config_path: str,
    reuse_preprocessing_root: str = "",
    reuse_preprocessing_db: str = "",
    content_manifest_path: str = "",
) -> None:
    """Initialise per-process state: load config, load Stage-1/2 ML models."""
    global _WORKER_CFG, _WORKER_S1_MODEL, _WORKER_S2_MODEL, _WORKER_HATCH_MODEL
    global _WORKER_DEPLOYMENT_IDENTITY
    global _WORKER_HATCH_STROKE_MODEL
    global _WORKER_REUSE_PREPROCESSING_ROOT, _WORKER_REUSE_PREPROCESSING_DB
    global _WORKER_CONTENT_MANIFEST

    # Loaded per process: a malformed manifest must stop the run, not silently
    # leave every figure's content pending.
    _WORKER_CONTENT_MANIFEST = (content_routing.load_manifest(Path(content_manifest_path))
                                if content_manifest_path else None)

    with open(config_path) as f:
        _WORKER_CFG = yaml.safe_load(f) or {}
    _WORKER_DEPLOYMENT_IDENTITY = None
    if deployment.strict_models(_WORKER_CFG):
        _WORKER_DEPLOYMENT_IDENTITY = deployment.preflight(Path(config_path))["identity"]

    # Resolve relative weight paths against the config file's directory so
    # the path works regardless of the worker's cwd (mirrors stage1's CLI).
    config_dir = Path(config_path).resolve().parent
    for section in ("sketchcleannet", "puhachov"):
        block = _WORKER_CFG.get(section)
        if isinstance(block, dict):
            w = block.get("weights", "")
            if w and not Path(w).is_absolute():
                block["weights"] = str(config_dir / w)
    stage2_block = _WORKER_CFG.get("stage2")
    if isinstance(stage2_block, dict):
        for key in ("hachure_cnn_model", "hachure_stroke_cnn_model"):
            value = stage2_block.get(key, "")
            if value and not Path(value).is_absolute():
                stage2_block[key] = str(config_dir / value)

    # Silence per-stage chatter inside workers; only the driver logs.
    logging.basicConfig(level=logging.ERROR, force=True)

    _WORKER_REUSE_PREPROCESSING_ROOT = (
        Path(reuse_preprocessing_root) if reuse_preprocessing_root else None
    )
    _WORKER_REUSE_PREPROCESSING_DB = (
        Path(reuse_preprocessing_db) if reuse_preprocessing_db else None
    )
    _WORKER_S1_MODEL = (
        None
        if _WORKER_REUSE_PREPROCESSING_ROOT is not None
        else stage1_preprocess.load_model(_WORKER_CFG)
    )
    _WORKER_S2_MODEL = stage2_stroke_extract.load_model(_WORKER_CFG)
    _WORKER_HATCH_MODEL = stage2_stroke_extract.load_hatch_model(_WORKER_CFG)
    _WORKER_HATCH_STROKE_MODEL = (
        stage2_stroke_extract.load_hatch_stroke_model(_WORKER_CFG)
    )
    if deployment.strict_models(_WORKER_CFG):
        if _WORKER_S2_MODEL is None or _WORKER_HATCH_MODEL is None:
            raise deployment.DeploymentError("Required Puhachov/hatch model failed to load; fallback is disabled.")


def _resolved_artifact(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _copy_reused_preprocessing(
    patent_id: str,
    sketch_id: str,
    input_path: Path,
    output_dir: Path,
) -> tuple[stage0_handle_references.Stage0Result,
           stage1_preprocess.Stage1Result]:
    """Materialise validated Stage-0/1 artifacts from an earlier paired run."""
    if (_WORKER_REUSE_PREPROCESSING_ROOT is None
            or _WORKER_REUSE_PREPROCESSING_DB is None):
        raise RuntimeError("preprocessing reuse was not initialised")

    with sqlite3.connect(_WORKER_REUSE_PREPROCESSING_DB) as conn:
        conn.row_factory = sqlite3.Row
        source_result = conn.execute(
            "SELECT * FROM results WHERE patent_id=? AND sketch_id=?",
            (patent_id, sketch_id),
        ).fetchone()
    if source_result is None:
        raise FileNotFoundError(
            f"no preprocessing source row for {patent_id}/{sketch_id}"
        )
    source_row = dict(source_result)
    if source_row["status"] in {"stage0", "stage1"}:
        raise _ReusedPreprocessingFailure(source_row)
    if Path(source_row["input_path"]).resolve() != input_path.resolve():
        raise RuntimeError(
            f"source input mismatch for {patent_id}/{sketch_id}: "
            f"{source_row['input_path']} != {input_path}"
        )

    source_dir = _WORKER_REUSE_PREPROCESSING_ROOT / patent_id
    source_references = source_dir / "references"
    source_cleaned = source_dir / "cleaned"
    target_references = output_dir / "references"
    target_cleaned = output_dir / "cleaned"
    target_crops = target_references / "crops"
    target_references.mkdir(parents=True, exist_ok=True)
    target_cleaned.mkdir(parents=True, exist_ok=True)
    target_crops.mkdir(parents=True, exist_ok=True)

    source_norefs = source_references / f"{sketch_id}_norefs.png"
    source_ref_json = source_references / f"{sketch_id}_references.json"
    source_mask = source_references / f"{sketch_id}_references_mask.png"
    source_cleaned_png = source_cleaned / f"{sketch_id}_cleaned.png"
    source_skeleton = source_cleaned / f"{sketch_id}_skeleton.png"
    required = (
        source_norefs, source_ref_json, source_mask,
        source_cleaned_png, source_skeleton,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing preprocessing source artifacts: " + ", ".join(missing)
        )

    target_norefs = target_references / source_norefs.name
    target_ref_json = target_references / source_ref_json.name
    target_mask = target_references / source_mask.name
    target_cleaned_png = target_cleaned / source_cleaned_png.name
    target_skeleton = target_cleaned / source_skeleton.name
    for source, target in (
        (source_norefs, target_norefs),
        (source_mask, target_mask),
        (source_cleaned_png, target_cleaned_png),
        (source_skeleton, target_skeleton),
    ):
        shutil.copy2(source, target)

    with open(source_ref_json) as fh:
        reference_doc = json.load(fh)
    reference_doc["mask_path"] = str(target_mask)
    reference_doc["reference_free_path"] = str(target_norefs)
    for label in reference_doc.get("reference_labels", []):
        crop_value = label.get("crop_path")
        if not crop_value:
            continue
        source_crop = _resolved_artifact(crop_value)
        if not source_crop.exists():
            raise FileNotFoundError(f"missing reference crop: {source_crop}")
        target_crop = target_crops / source_crop.name
        shutil.copy2(source_crop, target_crop)
        label["crop_path"] = str(target_crop)
    with open(target_ref_json, "w") as fh:
        json.dump(reference_doc, fh, indent=2)

    s0 = stage0_handle_references.Stage0Result(
        sketch_id=sketch_id,
        reference_free_path=target_norefs,
        references_json_path=target_ref_json,
        mask_path=target_mask,
        n_labels=int(source_row["s0_n_labels"] or 0),
        n_leaders=int(source_row["s0_n_leaders"] or 0),
        n_iterations=int(source_row["s0_n_iterations"] or 0),
        removed_ink_ratio=float(source_row["s0_removed_ink_ratio"] or 0.0),
        active_removal=bool(source_row["s0_active_removal"]),
        flagged=bool(source_row["s0_flagged"]),
        processing_time_s=float(source_row["s0_time"] or 0.0),
    )
    s1 = stage1_preprocess.Stage1Result(
        sketch_id=sketch_id,
        cleaned_path=target_cleaned_png,
        skeleton_path=target_skeleton,
        skeleton_quality=float(source_row["s1_quality"] or 0.0),
        processing_time_s=float(source_row["s1_time"] or 0.0),
        model_used=str(source_row["s1_model_used"] or "unknown"),
        flagged=bool(source_row["s1_flagged"]),
    )
    return s0, s1


def _run_stages(job: tuple[str, str, str, str]) -> dict:
    """
    Run all four stages on a single sketch. Returns a dict suitable for
    `results_db.insert_row`. Never raises: errors are captured into the dict.
    """
    patent_id, sketch_id, input_path_str, output_dir_str = job
    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)

    row: dict = {
        "patent_id":    patent_id,
        "sketch_id":    sketch_id,
        "input_path":   input_path_str,
        "status":       "ok",
        "error":        None,
        "completed_at": _dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    t0 = time.perf_counter()
    reused_preprocessing_time = 0.0

    def total_time() -> float:
        return reused_preprocessing_time + time.perf_counter() - t0

    gates = _WORKER_CFG.get("pipeline", {}).get("quality_gates", {})
    gates_enabled = bool(gates.get("enabled", False))
    stage0_enabled = bool((_WORKER_CFG.get("stage0", {}) or {}).get("enabled", False))
    stage1_input = input_path
    references_json_path = None

    try:
        if _WORKER_REUSE_PREPROCESSING_ROOT is not None:
            stage1_started = True
            s0, s1 = _copy_reused_preprocessing(
                patent_id=patent_id,
                sketch_id=sketch_id,
                input_path=input_path,
                output_dir=output_dir,
            )
            stage1_input = s0.reference_free_path
            references_json_path = s0.references_json_path
            reused_preprocessing_time = s0.processing_time_s + s1.processing_time_s
            t0 = time.perf_counter()
        elif stage0_enabled:
            s0 = stage0_handle_references.run(
                input_path=input_path, output_dir=output_dir,
                sketch_id=sketch_id, config=_WORKER_CFG,
            )
            stage1_input = s0.reference_free_path
            references_json_path = s0.references_json_path
            row.update({
                "s0_time":              s0.processing_time_s,
                "s0_n_labels":          s0.n_labels,
                "s0_n_leaders":         s0.n_leaders,
                "s0_n_iterations":      s0.n_iterations,
                "s0_removed_ink_ratio": s0.removed_ink_ratio,
                "s0_active_removal":    int(s0.active_removal),
                "s0_flagged":           int(s0.flagged),
            })

        if _WORKER_REUSE_PREPROCESSING_ROOT is None:
            stage1_started = True
            s1 = stage1_preprocess.run(
                input_path=stage1_input, output_dir=output_dir,
                sketch_id=sketch_id, config=_WORKER_CFG, model=_WORKER_S1_MODEL,
            )
        if stage0_enabled and _WORKER_REUSE_PREPROCESSING_ROOT is not None:
            row.update({
                "s0_time":              s0.processing_time_s,
                "s0_n_labels":          s0.n_labels,
                "s0_n_leaders":         s0.n_leaders,
                "s0_n_iterations":      s0.n_iterations,
                "s0_removed_ink_ratio": s0.removed_ink_ratio,
                "s0_active_removal":    int(s0.active_removal),
                "s0_flagged":           int(s0.flagged),
            })
        row.update({
            "s1_time":       s1.processing_time_s,
            "s1_quality":    s1.skeleton_quality,
            "s1_model_used": s1.model_used,
            "s1_flagged":    int(s1.flagged),
        })
    except _ReusedPreprocessingFailure as exc:
        source_row = exc.source_row
        row.update({
            field: source_row.get(field) for field in _PREPROCESSING_ROW_FIELDS
        })
        row["status"] = source_row["status"]
        row["error"] = source_row.get("error")
        row["total_time"] = source_row.get("total_time") or total_time()
        return row
    except Exception as exc:
        row["status"] = "stage1" if locals().get("stage1_started") else "stage0"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = total_time()
        return row

    if gates_enabled and gates.get("stop_on_stage1", True) and s1.flagged:
        row["status"] = "quality_gate_stage1"
        row["error"] = (
            f"skeleton quality {s1.skeleton_quality:.3f} below threshold; "
            "not suitable for accurate vectorization"
        )
        row["total_time"] = total_time()
        return row

    try:
        s2 = stage2_stroke_extract.run(
            skeleton_path=s1.skeleton_path, output_dir=output_dir,
            sketch_id=sketch_id, config=_WORKER_CFG, model=_WORKER_S2_MODEL,
            source_image_path=stage1_input, hatch_model=_WORKER_HATCH_MODEL,
            hatch_stroke_model=_WORKER_HATCH_STROKE_MODEL,
        )
        row.update({
            "s2_time":         s2.processing_time_s,
            "s2_keypoint_src": s2.keypoint_source,
            "s2_n_nodes":      s2.n_nodes,
            "s2_n_edges":      s2.n_edges,
            "s2_n_closed_edges": s2.n_closed_edges,
            "s2_n_hachure_edges_removed": s2.n_hachure_edges_removed,
            "s2_median_edge_len": s2.median_edge_length,
            "s2_micro_edge_ratio": s2.micro_edge_ratio,
            "s2_short_edge_ratio": s2.short_edge_ratio,
            "s2_isolation":    s2.isolation_ratio,
            "s2_flagged":      int(s2.flagged),
        })
    except Exception as exc:
        row["status"] = "stage2"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = total_time()
        return row

    if gates_enabled and gates.get("stop_on_stage2", True) and s2.flagged:
        row["status"] = "quality_gate_stage2"
        row["error"] = (
            f"graph fragmentation/isolation too high: edges={s2.n_edges}, "
            f"median_len={s2.median_edge_length:.1f}, "
            f"micro_ratio={s2.micro_edge_ratio:.3f}, "
            f"short_ratio={s2.short_edge_ratio:.3f}, "
            f"isolation={s2.isolation_ratio:.3f}, "
            "noncycle_residuals="
            f"{getattr(s2, 'n_unclaimed_noncycle_components', 0)}, "
            "max_noncycle_residual_pixels="
            f"{getattr(s2, 'max_unclaimed_noncycle_pixels', 0)}"
        )
        row["total_time"] = total_time()
        return row

    if bool((_WORKER_CFG.get("pipeline", {}) or {}).get(
        "stop_after_stage2", False
    )):
        row["status"] = "ok_stage2"
        row["total_time"] = total_time()
        return row

    try:
        s3 = stage3_primitive_fit.run(
            graph_path=s2.graph_path, output_dir=output_dir,
            sketch_id=sketch_id, config=_WORKER_CFG,
            # stroke_width intentionally omitted: patent SVG output uses ISO 128
            # lineweights, not the original scan's ink thickness.
        )
        with open(s3.primitives_path) as fh:
            prim_doc = json.load(fh)
        conf_thresh = _WORKER_CFG.get("stage3", {}).get("confidence_threshold", 0.60)
        prims = prim_doc.get("primitives", [])
        quality_prims = [
            p for p in prims
            if p.get("style") != "hachure"
            and p.get("source") != "removed_hachure"
        ]
        prim_confs = [
            float(p.get("confidence", 0.0))
            for p in quality_prims
        ]
        low_conf_ratio = (
            sum(c < conf_thresh for c in prim_confs) / len(prim_confs)
            if prim_confs else 0.0
        )
        quality_metrics = prim_doc.get("quality_metrics", {})
        row.update({
            "s3_time":         s3.processing_time_s,
            "s3_n_primitives": s3.n_primitives,
            "s3_n_budget_primitives": sum(stage4_export.primitive_budget_cost(p) for p in prims),
            "s3_n_hachure_primitives": int(
                quality_metrics.get(
                    "n_hachure_primitives",
                    getattr(s3, "n_hachure_primitives", 0),
                )
            ),
            "s3_mean_conf":    s3.mean_confidence,
            "s3_low_conf_ratio": low_conf_ratio,
            "s3_flagged":      int(s3.flagged),
        })
    except Exception as exc:
        row["status"] = "stage3"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["total_time"] = total_time()
        return row

    max_primitives = int(gates.get("max_primitives", 0) or 0)
    max_low_conf_ratio = float(gates.get("max_low_conf_ratio", 0.0) or 0.0)
    hachure_relax_min_edges = int(
        gates.get("min_hachure_edges_for_low_conf_relax", 0) or 0
    )
    hachure_max_low_conf_ratio = float(
        gates.get("max_low_conf_ratio_after_hachure", 0.0) or 0.0
    )
    effective_max_low_conf_ratio = max_low_conf_ratio
    if (
        hachure_max_low_conf_ratio
        and hachure_relax_min_edges
        and getattr(s2, "n_hachure_edges_removed", 0) >= hachure_relax_min_edges
    ):
        effective_max_low_conf_ratio = max(
            max_low_conf_ratio,
            hachure_max_low_conf_ratio,
        )
    if gates_enabled and (
        s3.flagged
        or (max_primitives and row["s3_n_budget_primitives"] > max_primitives)
        or (effective_max_low_conf_ratio
            and row.get("s3_low_conf_ratio", 0.0) > effective_max_low_conf_ratio)
    ):
        row["status"] = "quality_gate_stage3"
        row["error"] = (
            f"primitive set not suitable for accurate CAD export: "
            f"n={s3.n_primitives}, budget_n={row['s3_n_budget_primitives']}, mean_conf={s3.mean_confidence:.3f}, "
            f"low_conf_ratio={row.get('s3_low_conf_ratio', 0.0):.3f}, "
            f"max_low_conf_ratio={effective_max_low_conf_ratio:.3f}"
        )
        row["total_time"] = total_time()
        return row

    try:
        export_json = s3.primitives_path
        if references_json_path is not None:
            export_json = stage0_handle_references.attach_references_to_primitives(
                primitives_path=s3.primitives_path,
                references_json_path=references_json_path,
            )
        s4 = stage4_export.run(
            input_json=export_json, output_dir=output_dir,
            sketch_id=sketch_id, formats=("svg", "dxf"), dxf_mode="patent",
        )
        row.update({
            "s4_time":    s4.processing_time_s,
            "s4_n_in":    s4.n_primitives_in,
            "s4_n_out":   s4.n_primitives_out,
            "s4_flagged": int(s4.flagged),
        })
    except Exception as exc:
        row["status"] = "stage4"
        row["error"] = f"{type(exc).__name__}: {exc}"

    row["total_time"] = total_time()
    return row


def _process_one(job: tuple[str, str, str, str]) -> dict:
    row = _run_stages(job)
    started = time.perf_counter()
    try:
        report, path = acceptance.record(
            row, Path(job[3]), _WORKER_DEPLOYMENT_IDENTITY,
            stage0_enabled=bool((_WORKER_CFG.get("stage0", {}) or {}).get("enabled", False)),
            content_manifest=_WORKER_CONTENT_MANIFEST,
        )
        row.update({
            "acceptance_status": report["acceptance_status"],
            "acceptance_policy_version": report["policy_version"],
            "acceptance_reason_codes": json.dumps(report["reason_codes"]),
            "acceptance_path": str(path.resolve()),
            "acceptance_sha256": acceptance.file_digest(path),
            "training_eligible": int(report["training_eligible"]),
        })
    except Exception as exc:
        logger.error("Acceptance recording failed for %s/%s: %s", job[0], job[1], exc)
        row.update(acceptance_status="error", training_eligible=0,
                   acceptance_reason_codes=json.dumps(["acceptance_recording_failed"]))
    row["total_time"] = (row.get("total_time") or 0.0) + time.perf_counter() - started
    return row


# ─── Dataset walker ──────────────────────────────────────────────────────────

def _iter_sketches(patent_root: Path) -> Iterator[tuple[str, str, Path]]:
    """Yield (patent_id, sketch_id, tif_path) over the whole corpus."""
    for patent_dir in sorted(patent_root.iterdir()):
        if not patent_dir.is_dir():
            continue
        patent_id = patent_dir.name
        for f in sorted(patent_dir.iterdir()):
            if f.suffix.lower() in (".tif", ".tiff"):
                # sketch_id = filename stem with patent prefix stripped
                stem = f.stem
                if stem.startswith(patent_id + "_"):
                    sketch_id = stem[len(patent_id) + 1:]
                else:
                    sketch_id = stem
                yield patent_id, sketch_id, f


def _stratified_sample(patent_root: Path, n: int,
                       seed: int = 42,
                       excluded_paths: set[str] | None = None,
                       ) -> list[tuple[str, str, Path]]:
    """One sketch from each of `n` random patents (those with >=1 sketch)."""
    rng = random.Random(seed)
    candidates = [d for d in patent_root.iterdir() if d.is_dir()]
    rng.shuffle(candidates)
    excluded_paths = excluded_paths or set()

    picks: list[tuple[str, str, Path]] = []
    for patent_dir in candidates:
        tifs = [f for f in sorted(patent_dir.iterdir())
                if f.suffix.lower() in (".tif", ".tiff")
                and str(f) not in excluded_paths]
        if not tifs:
            continue
        f = tifs[0]
        stem = f.stem
        patent_id = patent_dir.name
        sketch_id = (stem[len(patent_id) + 1:]
                     if stem.startswith(patent_id + "_") else stem)
        picks.append((patent_id, sketch_id, f))
        if len(picks) >= n:
            break
    return picks


def _select_sketches(
    patent_root: Path,
    limit: int | None,
    stratified: bool,
    seed: int,
    excluded_paths: set[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """Select a deterministic pilot, optionally filtering before limiting."""
    excluded_paths = excluded_paths or set()
    if limit and stratified:
        return _stratified_sample(
            patent_root, limit, seed, excluded_paths=excluded_paths,
        )

    sketches = []
    for sketch in _iter_sketches(patent_root):
        if str(sketch[2]) in excluded_paths:
            continue
        sketches.append(sketch)
        if limit and len(sketches) >= limit:
            break
    return sketches


def _source_worklist(
    source_db: Path,
    limit: int | None = None,
) -> list[tuple[str, str, Path]]:
    """Load an exact paired cohort from a completed source results DB."""
    query = (
        "SELECT patent_id, sketch_id, input_path FROM results "
        "ORDER BY patent_id, sketch_id"
    )
    params: tuple[int, ...] = ()
    if limit:
        query += " LIMIT ?"
        params = (limit,)
    with results_db.connect(source_db) as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        (str(patent_id), str(sketch_id), Path(input_path))
        for patent_id, sketch_id, input_path in rows
    ]


def _manifest_worklist(
    manifest_path: Path,
    patent_root: Path,
    limit: int | None = None,
) -> list[tuple[str, str, Path]]:
    """Load an exact ordered cohort from a small CSV manifest."""
    with manifest_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        required = {"patent_id", "sketch_id"}
        if not required.issubset(fields):
            missing = ", ".join(sorted(required - fields))
            raise ValueError(f"worklist is missing required column(s): {missing}")

        rows: list[tuple[str, str, Path]] = []
        seen: set[tuple[str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            patent_id = str(row.get("patent_id") or "").strip()
            sketch_id = str(row.get("sketch_id") or "").strip()
            if not patent_id or not sketch_id:
                raise ValueError(
                    f"worklist row {line_number} has an empty patent/sketch id"
                )
            identity = (patent_id, sketch_id)
            if identity in seen:
                raise ValueError(
                    f"worklist row {line_number} duplicates "
                    f"{patent_id}/{sketch_id}"
                )
            seen.add(identity)

            raw_path = str(row.get("input_path") or "").strip()
            if raw_path:
                input_path = Path(raw_path).expanduser()
                if not input_path.is_absolute():
                    manifest_relative = manifest_path.parent / input_path
                    input_path = (
                        manifest_relative
                        if manifest_relative.exists()
                        else PROJECT_ROOT / input_path
                    )
            else:
                patent_dir = patent_root / patent_id
                candidates = (
                    patent_dir / f"{patent_id}_{sketch_id}.tif",
                    patent_dir / f"{patent_id}_{sketch_id}.tiff",
                )
                input_path = next(
                    (candidate for candidate in candidates if candidate.exists()),
                    candidates[0],
                )
            rows.append((patent_id, sketch_id, input_path))
            if limit and len(rows) >= limit:
                break
    return rows


def _preprocessing_config(cfg: dict) -> dict:
    """Return only configuration that can affect reusable Stage-0/1 output."""
    return {
        section: cfg.get(section, {}) or {}
        for section in ("stage0", "stage1", "sketchcleannet")
    }


# ─── Driver ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate the AP3 pipeline over a patent corpus.",
    )
    parser.add_argument("--patent-root", type=Path,
                        default=PROJECT_ROOT / "data" / "PatentData"
                                              / "ReorganisedData",
                        help="Directory containing per-patent subfolders.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "output" / "PatentData",
                        help="Root directory for stage outputs (cleaned/, "
                             "graphs/, primitives/, vectors/ are created "
                             "under <output>/<patent_id>/).")
    parser.add_argument("--db", type=Path, default=None,
                        help="SQLite results DB. Default: "
                             "<output>/results.db")
    parser.add_argument("--config", type=Path,
                        default=deployment.CANONICAL_CONFIG,
                        help="Pipeline config file.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers. Default: "
                             "config.pipeline.workers or os.cpu_count().")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N sketches (Phase 0 pilot).")
    parser.add_argument("--stratified", action="store_true",
                        help="With --limit: one sketch per random patent. "
                             "Without: take the first N sketches of the corpus.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess sketches even if they already have "
                             "a row in the DB.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for stratified sampling.")
    parser.add_argument("--filter-manifest", type=Path, default=None,
                        help="CSV produced by tools/filter_patent_data.py. "
                             "TIFs with label='discard' are skipped.")
    parser.add_argument(
        "--worklist", type=Path, default=None,
        help=("Exact ordered CSV cohort with patent_id, sketch_id, and an "
              "optional input_path column. --limit truncates this cohort."),
    )
    parser.add_argument(
        "--limit-after-filter", action="store_true",
        help=("apply --filter-manifest before --limit, so a filtered pilot "
              "contains exactly N kept drawings when enough are available"),
    )
    parser.add_argument(
        "--reuse-preprocessing-from", type=Path, default=None,
        help=("reuse and copy Stage-0/1 artifacts from this completed paired "
              "run, then execute Stage 2 onward"),
    )
    parser.add_argument(
        "--reuse-preprocessing-db", type=Path, default=None,
        help=("results DB for --reuse-preprocessing-from; default: "
              "<reuse-root>/results.db"),
    )
    parser.add_argument(
        "--reuse-preprocessing-config", type=Path, default=None,
        help=("config used by the preprocessing source run; required so "
              "Stage-0/1 compatibility can be validated"),
    )
    parser.add_argument(
        "--content-manifest", type=Path, default=None,
        help=("explicit per-figure content decisions "
              "(tools/build_content_manifest.py); without it every figure's "
              "content check stays pending and nothing is training-eligible"),
    )
    parser.add_argument(
        "--reuse-source-worklist", action="store_true",
        help=("select the exact patent/sketch/input rows from the reuse DB "
              "instead of resampling the current corpus; --limit selects "
              "the first rows in stable patent/sketch order"),
    )
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be positive")
    if args.limit_after_filter and not args.limit:
        parser.error("--limit-after-filter requires --limit")
    if args.worklist is not None and not args.worklist.exists():
        parser.error(f"worklist does not exist: {args.worklist}")
    if args.limit_after_filter and not args.filter_manifest:
        parser.error("--limit-after-filter requires --filter-manifest")
    if args.filter_manifest is not None and not args.filter_manifest.is_file():
        parser.error(
            f"filter manifest does not exist: "
            f"{args.filter_manifest}"
        )
    if args.reuse_preprocessing_from is None and (
        args.reuse_preprocessing_db is not None
        or args.reuse_preprocessing_config is not None
    ):
        parser.error(
            "--reuse-preprocessing-db/config require "
            "--reuse-preprocessing-from"
        )
    if (args.reuse_preprocessing_from is not None
            and args.reuse_preprocessing_config is None):
        parser.error(
            "--reuse-preprocessing-from requires "
            "--reuse-preprocessing-config"
        )
    if args.reuse_source_worklist and args.reuse_preprocessing_from is None:
        parser.error(
            "--reuse-source-worklist requires --reuse-preprocessing-from"
        )
    if args.reuse_source_worklist and (
        args.stratified or args.filter_manifest is not None
        or args.limit_after_filter or args.worklist is not None
    ):
        parser.error(
            "--reuse-source-worklist cannot be combined with stratified or "
            "filter-based selection"
        )
    if args.worklist is not None and (
        args.stratified or args.filter_manifest is not None
        or args.limit_after_filter
    ):
        parser.error(
            "--worklist cannot be combined with stratified or filter-based "
            "selection"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.patent_root.exists():
        logger.error("Patent root does not exist: %s", args.patent_root)
        return 2

    db_path: Path = args.db or (args.output / "results.db")
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    workers = args.workers or cfg.get("pipeline", {}).get("workers") \
              or max(1, (os.cpu_count() or 2) - 1)
    strict_deployment = deployment.strict_models(cfg)
    if strict_deployment:
        try:
            report = deployment.preflight(args.config)
            report["initial_execution"] = {
                "workers": workers,
                "reuse_preprocessing_from": (
                    str(args.reuse_preprocessing_from.resolve())
                    if args.reuse_preprocessing_from is not None else None
                ),
            }
            deployment.record_run(args.output / "deployment_run.json", report, db_path)
        except (deployment.DeploymentError, OSError, ValueError) as exc:
            logger.error("Deployment preflight failed: %s", exc)
            return 2
    args.output.mkdir(parents=True, exist_ok=True)
    results_db.init_db(db_path)
    reuse_root = args.reuse_preprocessing_from
    reuse_db = args.reuse_preprocessing_db
    if reuse_root is not None:
        reuse_db = reuse_db or (reuse_root / "results.db")
        for label, path in (
            ("reuse root", reuse_root),
            ("reuse DB", reuse_db),
            ("reuse config", args.reuse_preprocessing_config),
        ):
            if path is None or not path.exists():
                parser.error(f"{label} does not exist: {path}")
        with open(args.reuse_preprocessing_config) as fh:
            source_cfg = yaml.safe_load(fh) or {}
        if _preprocessing_config(source_cfg) != _preprocessing_config(cfg):
            parser.error(
                "Stage-0/1 configuration differs from the preprocessing "
                "source; refusing unsafe artifact reuse"
            )
        if not bool((cfg.get("stage0", {}) or {}).get("enabled", False)):
            parser.error("preprocessing reuse currently requires Stage 0")
    discard_paths: set[str] = set()
    if args.filter_manifest and args.filter_manifest.exists():
        with open(args.filter_manifest, newline="") as fh:
            discard_paths = {
                row["path"]
                for row in csv.DictReader(fh)
                if row.get("label") == "discard"
            }

    # ── Build the job list ────────────────────────────────────────────────
    if args.reuse_source_worklist:
        sketches = _source_worklist(reuse_db, args.limit)
        missing_inputs = [
            path for _patent, _sketch, path in sketches if not path.exists()
        ]
        if missing_inputs:
            parser.error(
                "reuse source worklist contains missing input: "
                f"{missing_inputs[0]}"
            )
        logger.info(
            "Loaded %d exact rows from reuse source worklist.", len(sketches),
        )
    elif args.worklist is not None:
        try:
            sketches = _manifest_worklist(
                args.worklist, args.patent_root, args.limit
            )
        except ValueError as exc:
            parser.error(str(exc))
        missing_inputs = [
            path for _patent, _sketch, path in sketches if not path.exists()
        ]
        if missing_inputs:
            parser.error(f"worklist contains missing input: {missing_inputs[0]}")
        logger.info(
            "Loaded %d exact rows from %s.", len(sketches), args.worklist,
        )
    else:
        prefilter = discard_paths if args.limit_after_filter else None
        sketches = _select_sketches(
            args.patent_root,
            args.limit,
            args.stratified,
            args.seed,
            excluded_paths=prefilter,
        )

    if not sketches:
        logger.error("No sketches found under %s", args.patent_root)
        return 2

    # ── Apply content-filter manifest ─────────────────────────────────────
    if args.filter_manifest and args.filter_manifest.exists():
        if args.limit_after_filter:
            logger.info(
                "Filter manifest applied before limit: selected %d kept TIFs.",
                len(sketches),
            )
        else:
            before = len(sketches)
            sketches = [s for s in sketches if str(s[2]) not in discard_paths]
            logger.info(
                "Filter manifest: skipped %d non-drawing TIFs, %d remain.",
                before - len(sketches), len(sketches),
            )

    # ── Skip already-processed (resume) ──────────────────────────────────
    if not args.no_resume:
        with results_db.connect(db_path) as conn:
            before = len(sketches)
            sketches = [
                s for s in sketches
                if not results_db.already_processed(conn, s[0], s[1])
            ]
            skipped = before - len(sketches)
            if skipped:
                logger.info("Resume: skipping %d sketches already in DB.", skipped)

    execution_errors = 0
    if not sketches:
        logger.info("Nothing to do.")
    else:
        logger.info(
            "Processing %d sketches with %d workers. Output → %s, DB → %s",
            len(sketches), workers, args.output, db_path,
        )

        # Build the (patent_id, sketch_id, input_path_str, output_dir_str) tuples.
        # Per-patent output dir keeps the cleaned/graphs/... convention scoped.
        jobs = [
            (patent_id, sketch_id, str(tif_path),
             str(args.output / patent_id))
            for (patent_id, sketch_id, tif_path) in sketches
        ]

        with results_db.connect(db_path) as conn:
            results_db.invalidate_acceptance(conn, [(patent, sketch) for patent, sketch, *_ in jobs])

        # ── Execute ──────────────────────────────────────────────────────
        n_ok = 0
        n_err = 0
        with results_db.connect(db_path) as conn, \
             ProcessPoolExecutor(
                 max_workers=workers,
                 initializer=_worker_init,
                 initargs=(
                     str(args.config),
                     str(reuse_root) if reuse_root is not None else "",
                     str(reuse_db) if reuse_db is not None else "",
                     str(args.content_manifest) if args.content_manifest else "",
                 ),
             ) as pool:

            futures = [pool.submit(_process_one, j) for j in jobs]
            bar = tqdm(as_completed(futures), total=len(futures),
                       smoothing=0.1, mininterval=0.5)
            for fut in bar:
                try:
                    row = fut.result()
                except Exception as exc:                # process died mid-task
                    bar.write(f"Worker crashed: {exc!r}")
                    bar.write(traceback.format_exc())
                    n_err += 1
                    execution_errors += 1
                    continue
                results_db.insert_row(conn, row)
                if row.get("acceptance_status") == "error" and _is_success_status(str(row["status"])):
                    execution_errors += 1
                if _is_success_status(str(row["status"])):
                    n_ok += 1
                else:
                    n_err += 1
                    if not str(row["status"]).startswith("quality_gate_"):
                        execution_errors += 1
                bar.set_postfix(ok=n_ok, err=n_err)

        logger.info("Done. ok=%d err=%d", n_ok, n_err)

    # ── Summary ──────────────────────────────────────────────────────────
    s = results_db.summarise(db_path)
    print()
    print("── DB summary ──────────────────────────────────────────────")
    print(f"  Total rows         : {s['total']}")
    print(f"  Pipeline 'ok'      : {s['ok']}  "
          f"({100.0 * s['ok'] / s['total']:.1f}%)" if s["total"] else "")
    print(f"  Status breakdown   : {s['by_status']}")
    with results_db.connect(db_path) as conn:
        acceptance_counts = dict(conn.execute(
            "SELECT COALESCE(acceptance_status, 'unassessed'), COUNT(*) "
            "FROM results GROUP BY acceptance_status"
        ))
    print(f"  Acceptance         : {acceptance_counts}")
    if s["mean_total_s"]:
        print(f"  Mean total time/ok : {s['mean_total_s']:.2f} s")
        if s.get("mean_s0_s") is not None:
            print(f"    Stage 0          : {s['mean_s0_s']:.2f} s")
        print(f"    Stage 1          : {s['mean_s1_s']:.2f} s")
        print(f"    Stage 2          : {s['mean_s2_s']:.2f} s")
        print(f"    Stage 3          : {s['mean_s3_s']:.2f} s")
        print(f"    Stage 4          : {s['mean_s4_s']:.2f} s")
        if s.get("mean_s0_labels") is not None:
            print(f"  Stage 0 refs/ok   : labels {s['mean_s0_labels']:.1f}, "
                  f"leaders {s['mean_s0_leaders']:.1f}, "
                  f"iters {s['mean_s0_iterations']:.1f}, "
                  f"ink removed {100.0 * s['mean_s0_removed_ink_ratio']:.2f}%")
        if s.get("mean_s2_edges") is not None:
            print(f"  Stage 2 edges/ok  : mean {s['mean_s2_edges']:.1f}, "
                  f"max {s['max_s2_edges']}")
        if s.get("mean_s2_hachure_edges_removed") is not None:
            print(f"  Hachure edges rm  : mean "
                  f"{s['mean_s2_hachure_edges_removed']:.1f}")
        if s.get("mean_s2_micro_edge_ratio") is not None:
            print(f"  Micro-edge ratio  : mean "
                  f"{100.0 * s['mean_s2_micro_edge_ratio']:.1f}%")
        if s.get("mean_s3_primitives") is not None:
            print(f"  Stage 3 prims/ok  : mean {s['mean_s3_primitives']:.1f}, "
                  f"max {s['max_s3_primitives']}")
        if s.get("mean_s3_hachure_primitives") is not None:
            print(f"  Hachure prims/ok  : mean "
                  f"{s['mean_s3_hachure_primitives']:.1f}")
        if s.get("mean_s3_low_conf_ratio") is not None:
            print(f"  Low-conf prims/ok : mean "
                  f"{100.0 * s['mean_s3_low_conf_ratio']:.1f}%, "
                  f"max {100.0 * s['max_s3_low_conf_ratio']:.1f}%")
    for k in ("flag_rate_s0", "flag_rate_s1", "flag_rate_s2",
              "flag_rate_s3", "flag_rate_s4"):
        v = s[k]
        if v is not None:
            print(f"  {k:18s}: {100.0 * v:.1f}%")
    print()
    return 1 if strict_deployment and execution_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
