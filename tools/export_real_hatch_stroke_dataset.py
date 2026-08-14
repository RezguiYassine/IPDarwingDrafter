"""Export reviewed PatentData figures as confidence-masked stroke labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from stage2_strokeextraction import stage2_stroke_extract
from syntheticData.patentvec.hatch_stroke_targets import (
    REAL_HATCH_STROKE_LABEL_CONTRACT,
    SUPERVISION_KEY,
    TARGET_KEYS,
    validate_hatch_stroke_supervision,
)


EXPORTER_VERSION = "reviewed-patent-hatch-stroke-export-1.1"
HATCH_LABEL_POLICIES = ("consensus", "region")


@dataclass
class FigureAnnotation:
    patent_id: str
    sketch_id: str
    crop_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    polygons: list[list[list[float]]] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    @property
    def sample_id(self) -> str:
        return f"{self.patent_id}__{self.sketch_id}"

    @property
    def is_positive(self) -> bool:
        return bool(self.polygons)


def load_reviewed_annotations(gt_dir: Path) -> list[FigureAnnotation]:
    grouped: dict[tuple[str, str], FigureAnnotation] = {}
    for path in sorted(gt_dir.glob("*_mask.json")):
        document = json.loads(path.read_text())
        if not document.get("reviewed"):
            continue
        crop_box = document.get("crop_box")
        if not isinstance(crop_box, list) or len(crop_box) != 4:
            continue
        patent_id = str(document.get("patent") or "").strip()
        sketch_id = str(document.get("sketch") or "").strip()
        if not patent_id or not sketch_id:
            raise ValueError(f"{path}: missing patent or sketch id")
        key = (patent_id, sketch_id)
        annotation = grouped.setdefault(
            key, FigureAnnotation(patent_id=patent_id, sketch_id=sketch_id)
        )
        annotation.crop_boxes.append(tuple(int(value) for value in crop_box))
        annotation.polygons.extend(document.get("polygons") or [])
        annotation.source_files.append(path.name)
    return [grouped[key] for key in sorted(grouped)]


def _input_path(record: FigureAnnotation, patent_root: Path) -> Path:
    patent_dir = patent_root / record.patent_id
    for suffix in (".tif", ".tiff"):
        candidate = patent_dir / (
            f"{record.patent_id}_{record.sketch_id}{suffix}"
        )
        if candidate.exists():
            return candidate
    return patent_dir / f"{record.patent_id}_{record.sketch_id}.tif"


def _is_teacher_success_status(status: str) -> bool:
    return status == "ok" or status.startswith("ok_stage")


def _load_pipeline_statuses(database_paths: list[Path]) -> dict[tuple[str, str], str]:
    statuses: dict[tuple[str, str], str] = {}
    for database_path in database_paths:
        if not database_path.exists():
            raise FileNotFoundError(f"pipeline results DB not found: {database_path}")
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                "SELECT patent_id, sketch_id, status FROM results"
            ).fetchall()
        for patent_id, sketch_id, status in rows:
            key = (str(patent_id), str(sketch_id))
            value = str(status)
            previous = statuses.get(key)
            if previous is not None and previous != value:
                raise ValueError(
                    f"conflicting pipeline statuses for {key}: "
                    f"{previous!r} versus {value!r}"
                )
            statuses[key] = value
    return statuses


def write_worklists(
    records: list[FigureAnnotation],
    output_path: Path,
    patent_root: Path,
    *,
    shards: int = 1,
) -> dict:
    if shards < 1:
        raise ValueError("worklist shard count must be positive")
    missing = [record for record in records if not _input_path(record, patent_root).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing TIF for {missing[0].patent_id}/{missing[0].sketch_id}"
        )

    def write(path: Path, selected: list[FigureAnnotation]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("patent_id", "sketch_id", "input_path")
            )
            writer.writeheader()
            for record in selected:
                writer.writerow({
                    "patent_id": record.patent_id,
                    "sketch_id": record.sketch_id,
                    "input_path": str(_input_path(record, patent_root).resolve()),
                })
        temporary.replace(path)

    ordered = sorted(records, key=lambda item: (item.patent_id, item.sketch_id))
    write(output_path, ordered)

    by_patent: dict[str, list[FigureAnnotation]] = defaultdict(list)
    for record in ordered:
        by_patent[record.patent_id].append(record)
    shard_rows: list[list[FigureAnnotation]] = [[] for _ in range(shards)]
    for patent_id in sorted(by_patent, key=lambda key: (-len(by_patent[key]), key)):
        target = min(range(shards), key=lambda index: (len(shard_rows[index]), index))
        shard_rows[target].extend(by_patent[patent_id])

    shard_paths = []
    if shards > 1:
        for index, selected in enumerate(shard_rows):
            shard_path = output_path.with_name(
                f"{output_path.stem}_part{index}{output_path.suffix}"
            )
            write(shard_path, sorted(
                selected, key=lambda item: (item.patent_id, item.sketch_id)
            ))
            shard_paths.append(str(shard_path))
    return {
        "figures": len(ordered),
        "positive_figures": sum(record.is_positive for record in ordered),
        "negative_figures": sum(not record.is_positive for record in ordered),
        "worklist": str(output_path),
        "shards": shard_paths,
        "shard_counts": [len(selected) for selected in shard_rows],
    }


def _morph(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    function = cv2.dilate if operation == "dilate" else cv2.erode
    return function(mask.astype(np.uint8), kernel) > 0


def build_real_targets(
    skeleton: np.ndarray,
    valid_mask: np.ndarray,
    hatch_region: np.ndarray,
    structural_support: np.ndarray,
    hatch_support: np.ndarray,
    *,
    boundary_ignore_radius: int = 3,
    crossing_guard_radius: int = 2,
) -> dict[str, np.ndarray]:
    """Build conservative real labels while leaving uncertain ink unsupervised."""
    skeleton = np.asarray(skeleton) > 0
    for name, array in (
        ("valid mask", valid_mask),
        ("hatch region", hatch_region),
        ("structural support", structural_support),
        ("hatch support", hatch_support),
    ):
        if np.asarray(array).shape != skeleton.shape:
            raise ValueError(f"{name} shape disagrees with skeleton")

    valid_area = np.asarray(valid_mask) > 0
    region_area = (np.asarray(hatch_region) > 0) & valid_area
    valid = valid_area & skeleton
    region = region_area & skeleton
    structural_support = (np.asarray(structural_support) > 0) & valid
    hatch_support = (np.asarray(hatch_support) > 0) & valid
    if np.any(region_area):
        boundary = _morph(
            region_area, boundary_ignore_radius, "dilate"
        ) ^ _morph(
            region_area, boundary_ignore_radius, "erode"
        )
    else:
        boundary = np.zeros_like(skeleton)
    trusted_hatch = hatch_support & region & ~boundary
    structural_guard = _morph(
        structural_support, crossing_guard_radius, "dilate"
    ) & skeleton
    trusted_hatch_only = trusted_hatch & ~structural_guard

    structural_target = skeleton.copy()
    structural_target[trusted_hatch_only] = False
    hatch_target = trusted_hatch
    overlap_target = structural_target & hatch_target

    trusted_outside = valid & ~region_area & ~boundary
    structural_supervision = (
        trusted_outside | structural_support | trusted_hatch
    ) & valid & ~boundary
    hatch_supervision = (trusted_outside | trusted_hatch) & valid & ~boundary
    supervision = np.stack(
        [structural_supervision, hatch_supervision], axis=0
    )

    targets = {
        "input_skeleton": skeleton.astype(np.uint8) * 255,
        "structural_target": structural_target.astype(np.uint8) * 255,
        "hatch_target": hatch_target.astype(np.uint8) * 255,
        "overlap_target": overlap_target.astype(np.uint8) * 255,
        SUPERVISION_KEY: supervision.astype(np.uint8) * 255,
    }
    validate_hatch_stroke_supervision(targets)
    return targets


def _annotation_masks(
    record: FigureAnnotation, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    valid = np.zeros(shape, dtype=np.uint8)
    region = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in record.crop_boxes:
        x0, x1 = sorted((max(0, x0), min(width, x1)))
        y0, y1 = sorted((max(0, y0), min(height, y1)))
        valid[y0:y1, x0:x1] = 1
    for polygon in record.polygons:
        if len(polygon) < 3:
            continue
        points = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(region, [points.reshape(-1, 1, 2)], 1)
    region &= valid
    return valid, region


def _edge_mask(
    edges: list[dict],
    shape: tuple[int, int],
    *,
    stage2_scale: float,
    support_radius: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    scale = stage2_scale if stage2_scale > 0 else 1.0
    for edge in edges:
        pixels = edge.get("pixels") or []
        if not pixels:
            continue
        points = np.asarray(pixels, dtype=np.float32) / scale
        points = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
        if len(points) == 1:
            x, y = points[0, 0]
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                mask[y, x] = 1
        else:
            cv2.polylines(
                mask, [points], bool(edge.get("is_closed")), 1, thickness=1
            )
    return _morph(mask, support_radius, "dilate").astype(np.uint8)


def _manual_geometric_hatch_edges(
    graph: dict,
    hatch_region: np.ndarray,
    stage2_config: dict,
) -> list[dict]:
    geometric_config = dict(stage2_config)
    # The paired full-graph run deliberately disables online hachure removal.
    # Re-enable only this offline selector used to build reviewed labels.
    geometric_config["remove_hachures"] = True
    graph_shape = tuple(int(value) for value in graph.get("image_shape", ()))
    if len(graph_shape) != 2:
        raise ValueError("full graph is missing image_shape")
    if graph_shape == hatch_region.shape:
        region = hatch_region.astype(np.uint8)
    else:
        region = cv2.resize(
            hatch_region.astype(np.uint8),
            (graph_shape[1], graph_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    _nodes, _kept, recovered = (
        stage2_stroke_extract._remove_hachures_cnn_geometric(
            graph.get("nodes") or [],
            graph.get("edges") or [],
            region,
            geometric_config,
            pass_name="reviewed_manual_geometric",
        )
    )
    return recovered


def _combine_hatch_teacher_support(
    region_support: np.ndarray,
    recovered_support: np.ndarray,
    hough_support: np.ndarray,
    *,
    policy: str,
    consensus_radius: int,
) -> np.ndarray:
    """Combine independent weak teachers into trusted hatch-stroke pixels."""
    if policy not in HATCH_LABEL_POLICIES:
        raise ValueError(f"unsupported hatch label policy: {policy}")
    shapes = {
        np.asarray(region_support).shape,
        np.asarray(recovered_support).shape,
        np.asarray(hough_support).shape,
    }
    if len(shapes) != 1:
        raise ValueError("hatch teacher support shapes disagree")
    candidate = (
        (np.asarray(region_support) > 0)
        | (np.asarray(recovered_support) > 0)
    )
    if policy == "region":
        return candidate.astype(np.uint8)
    periodic = _morph(hough_support, consensus_radius, "dilate")
    return (candidate & periodic).astype(np.uint8)


def _split_for(sample_id: str, modulus: int, fold: int) -> str:
    digest = hashlib.blake2b(
        sample_id.encode(), digest_size=8, person=b"RealHatchSplit1"
    ).digest()
    bucket = int.from_bytes(digest, byteorder="little") % modulus
    return "validation" if bucket == fold else "train"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def export_real_dataset(
    records: list[FigureAnnotation],
    pipeline_run: Path,
    output: Path,
    *,
    pipeline_dbs: list[Path] | None = None,
    full_graph_run: Path | None = None,
    teacher_config: Path | None = None,
    support_radius: int = 2,
    boundary_ignore_radius: int = 3,
    crossing_guard_radius: int = 2,
    hatch_label_policy: str = "consensus",
    consensus_radius: int = 0,
    validation_modulus: int = 5,
    validation_fold: int = 0,
) -> dict:
    if validation_modulus < 2 or not 0 <= validation_fold < validation_modulus:
        raise ValueError("invalid validation modulus/fold")
    if hatch_label_policy not in HATCH_LABEL_POLICIES:
        raise ValueError(
            f"hatch_label_policy must be one of {HATCH_LABEL_POLICIES}"
        )
    if consensus_radius < 0:
        raise ValueError("consensus_radius must be non-negative")
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        (output / split).mkdir(parents=True, exist_ok=True)
    resolved_pipeline_dbs = list(pipeline_dbs or sorted(
        pipeline_run.glob("results*.db")
    ))
    pipeline_filter_enabled = bool(resolved_pipeline_dbs)
    pipeline_statuses = _load_pipeline_statuses(resolved_pipeline_dbs)
    stage2_config: dict = {}
    if teacher_config is not None:
        configuration = yaml.safe_load(teacher_config.read_text()) or {}
        stage2_config = dict(configuration.get("stage2", {}) or {})
    elif full_graph_run is not None:
        raise ValueError("teacher_config is required with full_graph_run")

    rows = []
    totals = Counter()
    skipped = Counter()
    for record in records:
        pipeline_status = pipeline_statuses.get(
            (record.patent_id, record.sketch_id)
        )
        if pipeline_filter_enabled and pipeline_status is None:
            skipped["missing_pipeline_result"] += 1
            continue
        if pipeline_status is not None and not _is_teacher_success_status(
            pipeline_status
        ):
            skipped[f"pipeline_status_{pipeline_status}"] += 1
            continue
        skeleton_path = (
            pipeline_run / record.patent_id / "cleaned"
            / f"{record.sketch_id}_skeleton.png"
        )
        graph_path = (
            pipeline_run / record.patent_id / "graphs"
            / f"{record.sketch_id}_graph.json"
        )
        full_graph_path = (
            full_graph_run / record.patent_id / "graphs"
            / f"{record.sketch_id}_graph.json"
            if full_graph_run is not None else None
        )
        if not skeleton_path.exists():
            skipped["missing_skeleton"] += 1
            continue
        if not graph_path.exists():
            skipped["missing_graph"] += 1
            continue
        if full_graph_path is not None and not full_graph_path.exists():
            skipped["missing_full_graph"] += 1
            continue
        skeleton_image = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton_image is None or not np.any(skeleton_image):
            skipped["empty_skeleton"] += 1
            continue
        skeleton = skeleton_image > 0
        graph = json.loads(graph_path.read_text())
        scale = float(graph.get("stage2_scale", 1.0) or 1.0)
        valid, region = _annotation_masks(record, skeleton.shape)
        structural_raw = _edge_mask(
            graph.get("edges") or [], skeleton.shape,
            stage2_scale=scale, support_radius=0,
        )
        region_hatch_raw = _edge_mask(
            graph.get("removed_hachures") or [], skeleton.shape,
            stage2_scale=scale, support_radius=0,
        )
        recovered_edges: list[dict] = []
        recovered_raw = np.zeros_like(region_hatch_raw)
        if full_graph_path is not None:
            full_graph = json.loads(full_graph_path.read_text())
            recovered_edges = _manual_geometric_hatch_edges(
                full_graph, region, stage2_config
            )
            recovered_raw = _edge_mask(
                recovered_edges,
                skeleton.shape,
                stage2_scale=float(full_graph.get("stage2_scale", 1.0) or 1.0),
                support_radius=0,
            )
        hough_raw, hough_families = (
            stage2_stroke_extract._hough_hachure_stroke_mask(
                skeleton_image, region, stage2_config
            )
        )
        hough_raw = (
            (hough_raw > 0) & skeleton & (region > 0)
        ).astype(np.uint8)
        hatch_raw = _combine_hatch_teacher_support(
            region_hatch_raw,
            recovered_raw,
            hough_raw,
            policy=hatch_label_policy,
            consensus_radius=consensus_radius,
        )
        structural_raw = structural_raw.astype(bool) & ~_morph(
            hatch_raw, 1, "dilate"
        )
        structural_support = _morph(
            structural_raw, support_radius, "dilate"
        ).astype(np.uint8)
        hatch_support = _morph(
            hatch_raw, support_radius, "dilate"
        ).astype(np.uint8)
        targets = build_real_targets(
            skeleton, valid, region, structural_support, hatch_support,
            boundary_ignore_radius=boundary_ignore_radius,
            crossing_guard_radius=crossing_guard_radius,
        )
        stats = validate_hatch_stroke_supervision(targets)
        split = _split_for(record.patent_id, validation_modulus, validation_fold)
        relative_path = Path(split) / f"{record.sample_id}_real.npz"
        metadata = {
            "sample_id": record.sample_id,
            "patent_id": record.patent_id,
            "sketch_id": record.sketch_id,
            "domain": "real_patent_reviewed",
            "difficulty": "real_patent",
            "is_positive_figure": record.is_positive,
            "annotation_files": record.source_files,
            "annotation_subdrawings": len(record.crop_boxes),
            "annotation_polygons": len(record.polygons),
            "source_skeleton": str(skeleton_path),
            "source_graph": str(graph_path),
            "source_pipeline_status": pipeline_status,
            "source_full_graph": (
                str(full_graph_path) if full_graph_path is not None else None
            ),
            "label_contract": REAL_HATCH_STROKE_LABEL_CONTRACT,
            "exporter_version": EXPORTER_VERSION,
            "support_radius": support_radius,
            "boundary_ignore_radius": boundary_ignore_radius,
            "crossing_guard_radius": crossing_guard_radius,
            "hatch_label_policy": hatch_label_policy,
            "consensus_radius": consensus_radius,
            "split": split,
        }
        payload = {
            **targets,
            "annotation_hatch_region": region.astype(np.uint8) * 255,
            "valid_annotation_mask": valid.astype(np.uint8) * 255,
            "teacher_region_hatch_support": (
                region_hatch_raw.astype(np.uint8) * 255
            ),
            "teacher_hough_hatch_support": hough_raw.astype(np.uint8) * 255,
            "meta": np.asarray(json.dumps(metadata, sort_keys=True)),
        }
        _atomic_npz(output / relative_path, payload)
        row = {
            **metadata,
            "path": str(relative_path),
            "pixel_counts": stats,
            "teacher_main_edges": len(graph.get("edges") or []),
            "teacher_hatch_edges": len(graph.get("removed_hachures") or []),
            "teacher_recovered_hatch_edges": len(recovered_edges),
            "teacher_region_hatch_pixels": int(np.count_nonzero(region_hatch_raw)),
            "teacher_hough_hatch_pixels": int(np.count_nonzero(hough_raw)),
            "teacher_selected_hatch_pixels": int(np.count_nonzero(hatch_raw)),
            "teacher_hough_families": len(hough_families),
        }
        rows.append(row)
        totals.update(stats)
        totals[f"split_{split}"] += 1
        totals["positive_figures"] += int(record.is_positive)
        totals["positive_figures_with_trusted_hatch"] += int(
            record.is_positive and stats["hatch_pixels"] > 0
        )
        totals["teacher_recovered_hatch_edges"] += len(recovered_edges)
        totals["teacher_region_hatch_pixels"] += int(
            np.count_nonzero(region_hatch_raw)
        )
        totals["teacher_hough_hatch_pixels"] += int(np.count_nonzero(hough_raw))
        totals["teacher_selected_hatch_pixels"] += int(
            np.count_nonzero(hatch_raw)
        )
        totals["teacher_hough_families"] += len(hough_families)

    rows.sort(key=lambda row: row["sample_id"])
    _atomic_jsonl(output / "manifest.jsonl", rows)
    manifest = {
        "schema_version": "reviewed-patent-hatch-stroke-dataset-1.0",
        "label_contract": REAL_HATCH_STROKE_LABEL_CONTRACT,
        "exporter_version": EXPORTER_VERSION,
        "source_gt_figures": len(records),
        "exported_figures": len(rows),
        "skipped": dict(skipped),
        "totals": dict(totals),
        "configuration": {
            "pipeline_run": str(pipeline_run),
            "pipeline_dbs": [str(path) for path in resolved_pipeline_dbs],
            "full_graph_run": (
                str(full_graph_run) if full_graph_run is not None else None
            ),
            "teacher_config": (
                str(teacher_config) if teacher_config is not None else None
            ),
            "support_radius": support_radius,
            "boundary_ignore_radius": boundary_ignore_radius,
            "crossing_guard_radius": crossing_guard_radius,
            "hatch_label_policy": hatch_label_policy,
            "consensus_radius": consensus_radius,
            "validation_modulus": validation_modulus,
            "validation_fold": validation_fold,
            "split_unit": "patent_id",
        },
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _max_pool(mask: np.ndarray, maximum_size: int) -> np.ndarray:
    height, width = mask.shape
    factor = max(1, math.ceil(max(height, width) / maximum_size))
    padded_height = math.ceil(height / factor) * factor
    padded_width = math.ceil(width / factor) * factor
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[:height, :width] = mask > 0
    return padded.reshape(
        padded_height // factor, factor,
        padded_width // factor, factor,
    ).max(axis=(1, 3))


def _fit_panel(rgb: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray(rgb)
    image.thumbnail((size, size), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (size, size), "white")
    panel.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return panel


def _audit_panels(path: Path, panel_size: int) -> tuple[list[Image.Image], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in TARGET_KEYS}
        supervision = np.asarray(data[SUPERVISION_KEY]) > 0
        region = np.asarray(data["annotation_hatch_region"]) > 0
        metadata = json.loads(str(np.asarray(data["meta"])))
    skeleton = _max_pool(arrays["input_skeleton"], panel_size)
    structural = _max_pool(arrays["structural_target"], panel_size)
    hatch = _max_pool(arrays["hatch_target"], panel_size)
    annotation = _max_pool(region, panel_size)
    structural_sup = _max_pool(supervision[0], panel_size)
    hatch_sup = _max_pool(supervision[1], panel_size)

    input_rgb = np.full((*skeleton.shape, 3), 255, dtype=np.uint8)
    input_rgb[skeleton] = (25, 28, 31)
    region_rgb = input_rgb.copy()
    region_rgb[annotation & ~skeleton] = (255, 226, 170)
    region_rgb[annotation & skeleton] = (194, 91, 24)
    labels_rgb = np.full((*skeleton.shape, 3), 244, dtype=np.uint8)
    labels_rgb[structural & ~hatch] = (35, 112, 190)
    labels_rgb[hatch & ~structural] = (210, 64, 64)
    labels_rgb[structural & hatch] = (235, 176, 28)
    supervision_rgb = np.full((*skeleton.shape, 3), 244, dtype=np.uint8)
    supervision_rgb[skeleton & ~(structural_sup | hatch_sup)] = (120, 124, 128)
    supervision_rgb[structural_sup & ~hatch_sup] = (35, 112, 190)
    supervision_rgb[hatch_sup & ~structural_sup] = (210, 64, 64)
    supervision_rgb[structural_sup & hatch_sup] = (124, 70, 165)
    return (
        [_fit_panel(panel, panel_size) for panel in (
            input_rgb, region_rgb, labels_rgb, supervision_rgb
        )],
        metadata,
    )


def build_audit_sheet(
    dataset_root: Path,
    output_path: Path,
    *,
    sample_count: int = 20,
    columns: int = 2,
    panel_size: int = 220,
) -> None:
    rows = [
        json.loads(line)
        for line in (dataset_root / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    positives = [row for row in rows if row["pixel_counts"]["hatch_pixels"] > 0]
    negatives = [row for row in rows if row["pixel_counts"]["hatch_pixels"] == 0]
    positive_count = min(len(positives), max(1, round(sample_count * 0.75)))
    negative_count = min(len(negatives), sample_count - positive_count)

    def spaced(items: list[dict], count: int) -> list[dict]:
        if count <= 0:
            return []
        indices = np.linspace(0, len(items) - 1, count, dtype=int)
        return [items[int(index)] for index in indices]

    selected = spaced(positives, positive_count) + spaced(negatives, negative_count)
    if not selected:
        raise ValueError("real hatch-stroke dataset is empty")
    font = ImageFont.load_default()
    margin = 10
    title_height = 38
    tile_width = panel_size * 4
    tile_height = panel_size + title_height
    row_count = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB",
        (
            margin + columns * (tile_width + margin),
            30 + row_count * (tile_height + margin),
        ),
        (236, 238, 240),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 7),
        "Skeleton | reviewed region | targets | supervision (gray=ignored)",
        fill=(20, 24, 28), font=font,
    )
    for index, row in enumerate(selected):
        column = index % columns
        row_index = index // columns
        x = margin + column * (tile_width + margin)
        y = 30 + row_index * (tile_height + margin)
        panels, _metadata = _audit_panels(dataset_root / row["path"], panel_size)
        for panel_index, panel in enumerate(panels):
            sheet.paste(panel, (x + panel_index * panel_size, y))
        counts = row["pixel_counts"]
        draw.text(
            (x, y + panel_size + 4),
            f"{row['sample_id']}  hatch={counts['hatch_pixels']} "
            f"joint={counts['jointly_supervised_pixels']}",
            fill=(20, 24, 28), font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=Path("output/PatentData/hatch_gt"))
    parser.add_argument(
        "--patent-root", type=Path,
        default=Path("data/PatentData/ReorganisedData"),
    )
    parser.add_argument("--worklist-output", type=Path, default=None)
    parser.add_argument("--worklist-shards", type=int, default=1)
    parser.add_argument("--pipeline-run", type=Path, default=None)
    parser.add_argument(
        "--pipeline-db",
        type=Path,
        action="append",
        default=None,
        help=(
            "Results DB used to admit only successful teacher rows; repeat for "
            "sharded runs. Defaults to results*.db under --pipeline-run."
        ),
    )
    parser.add_argument("--full-graph-run", type=Path, default=None)
    parser.add_argument("--teacher-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--support-radius", type=int, default=2)
    parser.add_argument("--boundary-ignore-radius", type=int, default=3)
    parser.add_argument("--crossing-guard-radius", type=int, default=2)
    parser.add_argument(
        "--hatch-label-policy",
        choices=HATCH_LABEL_POLICIES,
        default="consensus",
    )
    parser.add_argument("--consensus-radius", type=int, default=0)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--audit-samples", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_reviewed_annotations(args.gt)
    if not records:
        raise SystemExit(f"no reviewed annotations found in {args.gt}")
    report = {}
    if args.worklist_output is not None:
        report["worklists"] = write_worklists(
            records, args.worklist_output, args.patent_root,
            shards=args.worklist_shards,
        )
    if args.pipeline_run is not None or args.output is not None:
        if args.pipeline_run is None or args.output is None:
            raise SystemExit("--pipeline-run and --output must be supplied together")
        report["dataset"] = export_real_dataset(
            records, args.pipeline_run, args.output,
            pipeline_dbs=args.pipeline_db,
            full_graph_run=args.full_graph_run,
            teacher_config=args.teacher_config,
            support_radius=args.support_radius,
            boundary_ignore_radius=args.boundary_ignore_radius,
            crossing_guard_radius=args.crossing_guard_radius,
            hatch_label_policy=args.hatch_label_policy,
            consensus_radius=args.consensus_radius,
            validation_modulus=args.validation_modulus,
            validation_fold=args.validation_fold,
        )
        if args.audit_output is not None:
            build_audit_sheet(
                args.output, args.audit_output, sample_count=args.audit_samples
            )
            report["audit"] = str(args.audit_output)
    if not report:
        raise SystemExit("request --worklist-output and/or --pipeline-run with --output")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
