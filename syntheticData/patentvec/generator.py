from __future__ import annotations

import copy
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .anchors import component_region, detect_anchors
from .composition import CompositionBuilder
from .geometry import primitive_bounds
from .patent_layers import apply_complex_patent_layers, apply_vertical_slice_layers
from .quality import evaluate_quality
from .render import (
    degrade_patent_scan,
    make_triptych,
    render_clean,
    render_masks,
    semantic_preview,
    svg_document,
)
from .schema import CanonicalDrawing
from .sources import (
    CADVGDrawingAdapter,
    SketchGraphsAdapter,
    SourceComponent,
    procedural_polyline_component,
)
from .training import free2cad_arrays, puhachov_arrays


@dataclass
class PreparedDrawing:
    drawing: CanonicalDrawing
    clean: np.ndarray
    masks: dict[str, np.ndarray]
    quality: dict
    free2cad: dict[str, np.ndarray] | None = None


def _rejected_attempt_reason(item: dict) -> str:
    failures = item.get("failures", [])
    if failures:
        if any("Free2CAD polyline supply" in str(failure) for failure in failures):
            return "quality:polyline_supply"
        return "quality:other"
    error = str(item.get("error", "unknown"))
    placement = re.search(r"placement step ([a-z0-9_]+) exhausted", error)
    if placement:
        return f"placement:{placement.group(1)}"
    return error.split(":", 1)[0]


def _free2cad_for_source(
    prepared: PreparedDrawing, source_index: int
) -> dict[str, np.ndarray]:
    base = prepared.free2cad
    if base is None:
        base = free2cad_arrays(
            prepared.drawing,
            source_index=source_index,
            object_mask=prepared.masks["object"],
        )
    if np.all(base["source_index"] == int(source_index)):
        return base
    return {
        **base,
        "source_index": np.full(
            len(base["types"]), int(source_index), dtype=np.int64
        ),
    }


def _npz_bytes(payload: dict[str, np.ndarray | str]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def _png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _sample_manifest_row(
    drawing: CanonicalDrawing,
    report: dict,
    puhachov: dict,
    free2cad: dict,
    source_index: int,
    relative_path: str,
    audit: bool,
) -> dict:
    keypoint_counts = np.bincount(puhachov["kps"][:, 2], minlength=3).tolist()
    free2cad_counts = np.bincount(free2cad["types"], minlength=5).tolist()
    keypoint_meta = json.loads(str(puhachov["meta"]))
    polyline_mask = free2cad["types"] == 3
    polyline_origins = np.bincount(
        free2cad["edge_origins"][polyline_mask], minlength=2
    ).tolist()
    placement_rejection_count = int(
        drawing.processing.get(
            "placement_rejection_count",
            len(drawing.processing.get("placement_rejections", [])),
        )
    )
    rejected_attempt_count = int(
        drawing.processing.get(
            "rejected_attempt_count",
            len(drawing.processing.get("rejected_attempts", [])),
        )
    )
    return {
        "sample_id": drawing.sample_id,
        "source_index": int(source_index),
        "seed": drawing.seed,
        "operator": drawing.processing.get("operator"),
        "difficulty": drawing.difficulty,
        "split": drawing.split,
        "relative_path": relative_path,
        "audit": bool(audit),
        "sources": [item.__dict__ for item in drawing.sources],
        "generation": {
            "attempt": int(drawing.processing.get("generation_attempt", 0)),
            "rejected_attempt_count": rejected_attempt_count,
            "placement_rejection_count": placement_rejection_count,
            "free2cad_supply": drawing.processing.get("free2cad_supply", {}),
        },
        "quality": report,
        "training_targets": {
            "puhachov_keypoints": {
                "endpoint": int(keypoint_counts[0]),
                "junction": int(keypoint_counts[1]),
                "corner": int(keypoint_counts[2]),
                "exact": int(keypoint_meta["exact_keypoint_count"]),
                "snapped": int(keypoint_meta["snapped_keypoint_count"]),
                "dropped": int(keypoint_meta["dropped_keypoint_count"]),
            },
            "free2cad_edges": {
                "line": int(free2cad_counts[0]),
                "arc": int(free2cad_counts[1]),
                "circle": int(free2cad_counts[2]),
                "polyline": int(free2cad_counts[3]),
                "bezier": int(free2cad_counts[4]),
                "topology_projected": bool(
                    np.all(free2cad.get("topology_projected", 0))
                ),
                "polyline_direct": int(polyline_origins[0]),
                "polyline_aggregated_lines": int(polyline_origins[1]),
                "match_p90_px_max": float(
                    np.nanmax(free2cad.get("match_p90_px", np.asarray([np.nan])))
                ),
            },
        },
    }


def compact_sample_payload(
    prepared: PreparedDrawing,
    source_index: int,
    audit: bool = False,
) -> tuple[dict[str, bytes], dict]:
    drawing = prepared.drawing
    if not prepared.quality.get("accepted"):
        raise ValueError(f"refusing to pack failed sample {drawing.sample_id}")
    puhachov = puhachov_arrays(drawing, clean=prepared.clean)
    free2cad = _free2cad_for_source(prepared, source_index)
    drawing.images = {
        "masks": "masks.npz",
        "puhachov": "puhachov.npz",
        "free2cad_edges": "free2cad_edges.npz",
    }
    payload = {
        "sample.json": (
            json.dumps(drawing.to_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "quality.json": (
            json.dumps(prepared.quality, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "masks.npz": _npz_bytes(prepared.masks),
        "puhachov.npz": _npz_bytes(puhachov),
        "free2cad_edges.npz": _npz_bytes(free2cad),
    }
    if audit:
        degraded = degrade_patent_scan(prepared.clean, seed=drawing.seed + 701)
        semantic = semantic_preview(drawing)
        payload.update(
            {
                "clean.png": _png_bytes(prepared.clean),
                "degraded.png": _png_bytes(degraded),
                "semantic.png": _png_bytes(semantic),
                "preview.png": _png_bytes(
                    np.asarray(
                        make_triptych(
                            prepared.clean, degraded, semantic, drawing.sample_id
                        )
                    )
                ),
                "visible.svg": svg_document(drawing).encode("utf-8"),
                "amodal.svg": svg_document(
                    drawing, primitives=drawing.primitives_amodal
                ).encode("utf-8"),
            }
        )
        drawing.images.update(
            {
                "clean": "clean.png",
                "degraded": "degraded.png",
                "semantic": "semantic.png",
                "preview": "preview.png",
                "visible_svg": "visible.svg",
                "amodal_svg": "amodal.svg",
            }
        )
        payload["sample.json"] = (
            json.dumps(drawing.to_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    row = _sample_manifest_row(
        drawing,
        prepared.quality,
        puhachov,
        free2cad,
        source_index=source_index,
        relative_path=f"samples/{drawing.sample_id}",
        audit=audit,
    )
    row["payload_bytes"] = int(sum(len(value) for value in payload.values()))
    return payload, row


class SourcePool:
    PROFILE_NAMES = (
        "endpoint",
        "endpoint_rich",
        "endpoint_bezier",
        "region_line",
        "rich_balanced_region_line",
        "circle",
        "single_line",
    )

    def __init__(
        self,
        sketchgraphs_path: Path,
        cadvg_root: Path,
        split: str = "train",
        index_path: Path | None = None,
    ):
        self.split = split
        self.sketchgraphs = SketchGraphsAdapter(sketchgraphs_path, split=split)
        self.cadvg = CADVGDrawingAdapter(cadvg_root, split=split)
        self.index_path = Path(index_path) if index_path else None
        self._index_buckets: dict[tuple[str, str], list[dict]] = {}
        if self.index_path and self.index_path.exists():
            payload = json.loads(self.index_path.read_text())
            if payload.get("split") != split:
                raise ValueError(
                    f"source index split {payload.get('split')} does not match {split}"
                )
            for entry in payload.get("entries", []):
                for profile in entry.get("profiles", []):
                    self._index_buckets.setdefault(
                        (entry["dataset"], profile), []
                    ).append(entry)

    @staticmethod
    def has_endpoint(component: SourceComponent) -> bool:
        return any(
            anchor.type == "endpoint"
            for anchor in detect_anchors("candidate", component.primitives)
        )

    @staticmethod
    def has_reusable_endpoint_pair(component: SourceComponent) -> bool:
        endpoints = [
            np.asarray(anchor.position, dtype=float)
            for anchor in detect_anchors("candidate", component.primitives)
            if anchor.type == "endpoint"
        ]
        return any(
            np.linalg.norm(first - second) >= 0.35
            for index, first in enumerate(endpoints)
            for second in endpoints[index + 1 :]
        )

    @staticmethod
    def has_region(component: SourceComponent) -> bool:
        return component_region(component.primitives) is not None

    @staticmethod
    def has_line(component: SourceComponent) -> bool:
        return any(primitive.kind == "line" for primitive in component.primitives)

    @staticmethod
    def has_circle(component: SourceComponent) -> bool:
        return any(primitive.kind == "circle" for primitive in component.primitives)

    @staticmethod
    def has_bezier(component: SourceComponent) -> bool:
        return any(
            primitive.kind in {"quadratic_bezier", "cubic_bezier"}
            for primitive in component.primitives
        )

    @staticmethod
    def moderate(component: SourceComponent) -> bool:
        return 1 <= len(component.primitives) <= 24

    @staticmethod
    def single_line(component: SourceComponent) -> bool:
        return len(component.primitives) == 1 and component.primitives[0].kind == "line"

    @staticmethod
    def balanced_profile(component: SourceComponent) -> bool:
        lower, upper = primitive_bounds(component.primitives)
        spans = upper - lower
        return float(np.min(spans) / max(np.max(spans), 1e-9)) >= 0.28

    @classmethod
    def component_profiles(cls, component: SourceComponent) -> list[str]:
        if not cls.moderate(component):
            return []
        has_endpoint = cls.has_endpoint(component)
        endpoint_count = int(component.descriptor.get("endpoint_anchor_count", 0))
        has_line = cls.has_line(component)
        has_region = cls.has_region(component)
        profiles = []
        if has_endpoint:
            profiles.append("endpoint")
            if (
                len(component.primitives) >= 2
                and endpoint_count >= 2
                and cls.has_reusable_endpoint_pair(component)
            ):
                profiles.append("endpoint_rich")
            if cls.has_bezier(component):
                profiles.append("endpoint_bezier")
        if has_region and has_line:
            profiles.append("region_line")
            if len(component.primitives) >= 9 and cls.balanced_profile(component):
                profiles.append("rich_balanced_region_line")
        if cls.has_circle(component):
            profiles.append("circle")
        if cls.single_line(component):
            profiles.append("single_line")
        return profiles

    @classmethod
    def matches_profile(cls, component: SourceComponent, profile: str) -> bool:
        if profile not in cls.PROFILE_NAMES:
            raise ValueError(f"unknown source profile {profile}")
        return profile in cls.component_profiles(component)

    def indexed_counts(self) -> dict[str, int]:
        return {
            f"{dataset}:{profile}": len(entries)
            for (dataset, profile), entries in sorted(self._index_buckets.items())
        }

    def pick_profile(
        self,
        rng: np.random.Generator,
        dataset: str,
        profile: str,
        max_attempts: int = 64,
    ) -> SourceComponent:
        entries = self._index_buckets.get((dataset, profile), [])
        for _ in range(max_attempts if entries else 0):
            entry = entries[int(rng.integers(len(entries)))]
            try:
                if dataset == "SketchGraphs":
                    components = self.sketchgraphs.load_source_sample(
                        int(entry["source_id"])
                    )
                elif dataset == "CAD-VGDrawing":
                    components = self.cadvg.load_source_sample(
                        entry["source_id"], view=entry["view"]
                    )
                else:
                    raise ValueError(f"unsupported indexed dataset {dataset}")
            except (FileNotFoundError, ValueError):
                continue
            component = next(
                (
                    item
                    for item in components
                    if item.component_key == entry["component_key"]
                ),
                None,
            )
            if component is not None and self.matches_profile(component, profile):
                return component

        predicate = lambda item: self.matches_profile(item, profile)
        if dataset == "SketchGraphs":
            return self.pick_sketchgraphs(rng, predicate)
        if dataset == "CAD-VGDrawing":
            return self.pick_cadvg(rng, predicate)
        raise ValueError(f"unsupported source dataset {dataset}")

    def pick_sketchgraphs(
        self,
        rng: np.random.Generator,
        predicate: Callable[[SourceComponent], bool],
        max_attempts: int = 800,
    ) -> SourceComponent:
        for _ in range(max_attempts):
            source_id = int(rng.integers(len(self.sketchgraphs)))
            try:
                components = self.sketchgraphs.load_source_sample(source_id)
            except Exception:
                continue
            candidates = [item for item in components if self.moderate(item) and predicate(item)]
            if candidates:
                return candidates[int(rng.integers(len(candidates)))]
        raise RuntimeError("could not retrieve a compatible SketchGraphs component")

    def pick_cadvg(
        self,
        rng: np.random.Generator,
        predicate: Callable[[SourceComponent], bool],
        max_attempts: int = 400,
    ) -> SourceComponent:
        for _ in range(max_attempts):
            source_id = self.cadvg.sample_ids[int(rng.integers(len(self.cadvg.sample_ids)))]
            view = self.cadvg.VIEWS[int(rng.integers(len(self.cadvg.VIEWS)))]
            try:
                components = self.cadvg.load_source_sample(source_id, view=view)
            except (FileNotFoundError, ValueError):
                continue
            candidates = [item for item in components if self.moderate(item) and predicate(item)]
            if candidates:
                return candidates[int(rng.integers(len(candidates)))]
        raise RuntimeError("could not retrieve a compatible CAD-VGDrawing component")


class PilotGenerator:
    OPERATORS = (
        "t_junction",
        "endpoint_join",
        "containment",
        "concentric",
        "tangent",
    )

    def __init__(self, source_pool: SourcePool, canvas: int = 1024):
        self.source_pool = source_pool
        self.canvas = int(canvas)

    def generate(
        self,
        sample_id: str,
        seed: int,
        operator: str,
        max_attempts: int = 24,
    ) -> CanonicalDrawing:
        if operator not in self.OPERATORS:
            raise ValueError(f"unsupported pilot operator {operator}")
        rejection_log = []
        for attempt in range(max_attempts):
            # Keep the base seed parity stable across retries; it selects pilot
            # coverage variants such as the Bezier-bearing CAD donor.
            attempt_seed = int(seed + attempt * 104728)
            rng = np.random.default_rng(attempt_seed)
            try:
                drawing = self._compose(sample_id, attempt_seed, operator, rng)
                apply_vertical_slice_layers(
                    drawing, seed=attempt_seed + 17, numeral=str(10 + seed % 89)
                )
                clean = render_clean(drawing)
                masks = render_masks(drawing)
                report = evaluate_quality(
                    drawing,
                    clean=clean,
                    masks=masks,
                    require_vertical_slice=operator == "t_junction",
                )
                if report["accepted"]:
                    drawing.processing["generation_attempt"] = attempt
                    drawing.processing["operator"] = operator
                    drawing.processing["rejected_attempts"] = rejection_log
                    return drawing
                rejection_log.append({"attempt": attempt, "failures": report["failures"]})
            except Exception as exc:
                rejection_log.append(
                    {"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"}
                )
        raise RuntimeError(
            f"failed to generate {sample_id}/{operator}: "
            + json.dumps(rejection_log[-4:], indent=2)
        )

    def _compose(
        self,
        sample_id: str,
        seed: int,
        operator: str,
        rng: np.random.Generator,
    ) -> CanonicalDrawing:
        pool = self.source_pool
        builder = CompositionBuilder(
            sample_id=sample_id,
            seed=seed,
            split="train",
            difficulty="easy",
            canvas=(self.canvas, self.canvas),
        )
        engineering_angle = float(rng.choice([0.0, np.pi / 4, np.pi / 2, -np.pi / 4]))

        if operator == "t_junction":
            host = pool.pick_sketchgraphs(
                rng,
                lambda item: pool.has_region(item) and pool.has_line(item),
            )
            donor = pool.pick_cadvg(rng, pool.has_endpoint)
            host_id = builder.add_base(host, scale=0.54, angle=engineering_angle)
            builder.t_junction(host_id, donor, scale=0.21)
        elif operator == "endpoint_join":
            host = pool.pick_sketchgraphs(rng, pool.has_endpoint)
            donor_predicate = (
                (lambda item: pool.has_endpoint(item) and pool.has_bezier(item))
                if seed % 2
                else pool.has_endpoint
            )
            donor = pool.pick_cadvg(rng, donor_predicate)
            host_id = builder.add_base(host, scale=0.43, angle=engineering_angle)
            builder.endpoint_join(host_id, donor, scale=0.20)
        elif operator == "containment":
            host = pool.pick_cadvg(
                rng, lambda item: pool.has_region(item) and pool.has_line(item)
            )
            donor = pool.pick_sketchgraphs(rng, lambda item: len(item.primitives) <= 12)
            host_id = builder.add_base(host, scale=0.55, angle=engineering_angle)
            builder.containment(host_id, donor, fill_fraction=0.22)
        elif operator == "concentric":
            host = pool.pick_sketchgraphs(rng, pool.has_circle)
            donor = pool.pick_sketchgraphs(rng, pool.has_circle)
            host_id = builder.add_base(host, scale=0.48, angle=engineering_angle)
            builder.concentric(host_id, donor, radius_ratio=float(rng.uniform(0.38, 0.68)))
        else:
            host = pool.pick_sketchgraphs(rng, pool.has_circle)
            donor = pool.pick_cadvg(rng, pool.has_endpoint)
            host_id = builder.add_base(host, scale=0.44, angle=engineering_angle)
            builder.tangent(host_id, donor, scale=0.16)
        return builder.finalize()


class ComplexPilotGenerator:
    """M5/M6 graph curriculum with transactional placement and hard gates."""

    def __init__(self, source_pool: SourcePool, canvas: int = 1024):
        self.source_pool = source_pool
        self.canvas = int(canvas)

    @staticmethod
    def _source_key(component: SourceComponent) -> tuple[str, str, str]:
        return (
            component.source.dataset,
            component.source.sample_id,
            component.component_key,
        )

    @staticmethod
    def _used_sources(builder: CompositionBuilder) -> set[tuple[str, str, str]]:
        return {
            (
                component.source_dataset,
                component.source_sample_id,
                component.source_component_id,
            )
            for component in builder.components
        }

    def _pick_source(
        self,
        rng: np.random.Generator,
        builder: CompositionBuilder,
        predicate: Callable[[SourceComponent], bool] | None = None,
        dataset: str | None = None,
        profile: str | None = None,
    ) -> SourceComponent:
        if predicate is None and profile is None:
            raise ValueError("source retrieval requires a predicate or profile")
        used = self._used_sources(builder)
        for _ in range(16):
            selected_dataset = dataset or str(
                rng.choice(["SketchGraphs", "CAD-VGDrawing"])
            )
            if profile is not None:
                component = self.source_pool.pick_profile(
                    rng, selected_dataset, profile
                )
            elif selected_dataset == "SketchGraphs":
                component = self.source_pool.pick_sketchgraphs(rng, predicate)
            else:
                component = self.source_pool.pick_cadvg(rng, predicate)
            if self._source_key(component) not in used:
                return component
        raise RuntimeError("source retrieval repeatedly selected an already-used component")

    def _transaction(
        self,
        builder: CompositionBuilder,
        rng: np.random.Generator,
        name: str,
        source_factory: Callable[[CompositionBuilder], SourceComponent],
        operation: Callable[[CompositionBuilder, SourceComponent], str],
        attempts: int = 28,
    ) -> tuple[CompositionBuilder, str, list[dict]]:
        rejections = []
        for attempt in range(attempts):
            try:
                source = source_factory(builder)
                trial = copy.deepcopy(builder)
                trial.rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                donor_id = operation(trial, source)
                report = evaluate_quality(trial.finalize())
                if report["accepted"]:
                    return trial, donor_id, rejections
                rejections.append(
                    {"step": name, "attempt": attempt, "failures": report["failures"]}
                )
            except Exception as exc:
                rejections.append(
                    {
                        "step": name,
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        raise RuntimeError(
            f"placement step {name} exhausted retries: "
            + json.dumps(rejections[-3:], sort_keys=True)
        )

    def generate(
        self,
        sample_id: str,
        seed: int,
        difficulty: str,
        max_attempts: int = 64,
    ) -> CanonicalDrawing:
        return self.generate_prepared(
            sample_id=sample_id,
            seed=seed,
            difficulty=difficulty,
            max_attempts=max_attempts,
        ).drawing

    def generate_prepared(
        self,
        sample_id: str,
        seed: int,
        difficulty: str,
        max_attempts: int = 64,
    ) -> PreparedDrawing:
        if difficulty not in {"medium", "hard", "very_hard"}:
            raise ValueError(
                "M5/M6 generator supports medium, hard, and very_hard samples"
            )
        rejected_samples = []
        for attempt in range(max_attempts):
            attempt_seed = int(seed + attempt * 100003)
            rng = np.random.default_rng(attempt_seed)
            try:
                builder, placement_rejections = self._compose(
                    sample_id, attempt_seed, difficulty, rng
                )
                drawing = builder.finalize()
                apply_complex_patent_layers(
                    drawing, seed=attempt_seed + 31, difficulty=difficulty
                )
                clean = render_clean(drawing)
                masks = render_masks(drawing)
                report = evaluate_quality(
                    drawing,
                    clean=clean,
                    masks=masks,
                    difficulty_gate=difficulty,
                )
                free2cad = free2cad_arrays(
                    drawing,
                    source_index=0,
                    object_mask=masks["object"],
                )
                polyline_count = int(np.count_nonzero(free2cad["types"] == 3))
                minimum_polylines = {
                    "medium": 2,
                    "hard": 3,
                    "very_hard": 4,
                }[difficulty]
                if polyline_count < minimum_polylines:
                    report["accepted"] = False
                    report["failures"].append(
                        f"{difficulty} Free2CAD polyline supply "
                        f"{polyline_count} < {minimum_polylines}"
                    )
                if report["accepted"]:
                    rejected_reason_counts = Counter(
                        _rejected_attempt_reason(item) for item in rejected_samples
                    )
                    placement_step_counts = Counter(
                        str(item.get("step", "unknown"))
                        for item in placement_rejections
                    )
                    drawing.processing.update(
                        {
                            "generation_attempt": attempt,
                            "generator_stage": "M5_M6",
                            "placement_rejection_count": len(
                                placement_rejections
                            ),
                            "placement_rejection_counts": dict(
                                placement_step_counts
                            ),
                            "placement_rejections_tail": placement_rejections[-5:],
                            "rejected_attempt_count": len(rejected_samples),
                            "rejected_attempt_reason_counts": dict(
                                rejected_reason_counts
                            ),
                            "rejected_attempts_tail": rejected_samples[-5:],
                            "free2cad_supply": {
                                "polyline": polyline_count,
                                "minimum_polyline": minimum_polylines,
                            },
                        }
                    )
                    return PreparedDrawing(
                        drawing=drawing,
                        clean=clean,
                        masks=masks,
                        quality=report,
                        free2cad=free2cad,
                    )
                rejected_samples.append(
                    {"attempt": attempt, "failures": report["failures"]}
                )
            except Exception as exc:
                rejected_samples.append(
                    {"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"}
                )
        raise RuntimeError(
            f"failed to generate {sample_id}/{difficulty}: "
            + json.dumps(rejected_samples[-4:], indent=2)
        )

    def _compose(
        self,
        sample_id: str,
        seed: int,
        difficulty: str,
        rng: np.random.Generator,
    ) -> tuple[CompositionBuilder, list[dict]]:
        pool = self.source_pool
        target_components = {
            "medium": int(rng.integers(4, 6)),
            "hard": 8,
            "very_hard": int(rng.integers(9, 13)),
        }[difficulty]
        base_dataset = "SketchGraphs" if seed % 2 == 0 else "CAD-VGDrawing"
        base = pool.pick_profile(
            rng, base_dataset, "rich_balanced_region_line"
        )
        builder = CompositionBuilder(
            sample_id=sample_id,
            seed=seed,
            split=pool.split,
            difficulty=difficulty,
            canvas=(self.canvas, self.canvas),
        )
        base_scale = {
            "medium": float(rng.uniform(0.43, 0.49)),
            "hard": float(rng.uniform(0.38, 0.44)),
            "very_hard": float(rng.uniform(0.34, 0.40)),
        }[difficulty]
        base_id = builder.add_base(
            base,
            scale=base_scale,
            angle=float(rng.choice([0.0, np.pi / 4, np.pi / 2, -np.pi / 4])),
        )
        placement_rejections: list[dict] = []
        other_dataset = (
            "CAD-VGDrawing" if base_dataset == "SketchGraphs" else "SketchGraphs"
        )

        builder, t_component, rejected = self._transaction(
            builder,
            rng,
            "spanning_t_junction",
            lambda current: self._pick_source(
                rng,
                current,
                dataset=other_dataset,
                profile="endpoint_rich",
            ),
            lambda trial, source: trial.t_junction(
                base_id,
                source,
                scale=float(rng.uniform(0.12, 0.18))
                if difficulty == "medium"
                else float(rng.uniform(0.09, 0.14)),
            ),
        )
        placement_rejections.extend(rejected)

        endpoint_hosts = builder.components_with("endpoint")
        if not endpoint_hosts:
            raise ValueError(
                "polyline chain attachment requires a remaining loose endpoint"
            )
        polyline_host = (
            t_component if t_component in endpoint_hosts else endpoint_hosts[0]
        )
        polyline_primitive_count = {
            "medium": 3,
            "hard": 4,
            "very_hard": 5,
        }[difficulty]
        builder, _, rejected = self._transaction(
            builder,
            rng,
            "polyline_chain_attachment",
            lambda current: procedural_polyline_component(
                sample_id=sample_id,
                split=pool.split,
                seed=int(rng.integers(0, 2**31 - 1)),
                primitive_count=polyline_primitive_count,
            ),
            lambda trial, source: trial.endpoint_join(
                polyline_host,
                source,
                scale=(
                    float(rng.uniform(0.12, 0.18))
                    if difficulty == "very_hard"
                    else float(rng.uniform(0.14, 0.20))
                ),
                turn_degrees=float(
                    rng.choice(
                        [-105.0, -90.0, 90.0, 105.0]
                        if difficulty == "very_hard"
                        else [-90.0, -45.0, 45.0, 90.0]
                    )
                ),
            ),
        )
        placement_rejections.extend(rejected)

        builder, _, rejected = self._transaction(
            builder,
            rng,
            "connected_crossing",
            lambda current: self._pick_source(
                rng, current, profile="single_line"
            ),
            lambda trial, source: trial.connected_crossing(
                base_id,
                source,
                scale=float(rng.uniform(0.11, 0.16)),
            ),
        )
        placement_rejections.extend(rejected)

        circle_component = None
        if target_components >= 4:
            builder, circle_component, rejected = self._transaction(
                builder,
                rng,
                "containment",
                lambda current: self._pick_source(
                    rng, current, dataset="SketchGraphs", profile="circle"
                ),
                lambda trial, source: trial.containment(
                    base_id, source, fill_fraction=float(rng.uniform(0.14, 0.22))
                ),
            )
            placement_rejections.extend(rejected)

        include_tangent = difficulty != "very_hard" or rng.random() < 0.65
        if target_components >= 5 and circle_component is not None and include_tangent:
            try:
                builder, _, rejected = self._transaction(
                    builder,
                    rng,
                    "tangent_attachment",
                    lambda current: self._pick_source(
                        rng,
                        current,
                        profile="endpoint_rich",
                    ),
                    lambda trial, source: trial.tangent(
                        circle_component,
                        source,
                        scale=float(rng.uniform(0.09, 0.14)),
                    ),
                )
                placement_rejections.extend(rejected)
            except RuntimeError as exc:
                if difficulty != "very_hard":
                    raise
                placement_rejections.append(
                    {
                        "step": "tangent_attachment",
                        "skipped": True,
                        "error": str(exc),
                    }
                )

        if difficulty in {"hard", "very_hard"}:
            endpoint_hosts = builder.components_with("endpoint")
            if not endpoint_hosts:
                raise ValueError("hard endpoint branch requires a loose endpoint")
            preferred_host = t_component if t_component in endpoint_hosts else endpoint_hosts[0]
            builder, _, rejected = self._transaction(
                builder,
                rng,
                "endpoint_branch",
                lambda current: self._pick_source(
                    rng,
                    current,
                    dataset="CAD-VGDrawing",
                    profile="endpoint_bezier",
                ),
                lambda trial, source: trial.endpoint_join(
                    preferred_host,
                    source,
                    scale=float(rng.uniform(0.10, 0.15)),
                ),
            )
            placement_rejections.extend(rejected)

            bridge_count = 2 if difficulty == "very_hard" else 1
            while len(builder.components) < target_components - bridge_count:
                line_hosts = builder.components_with("long_line")
                if not line_hosts:
                    raise ValueError(
                        "additional T-junction requires a component with a long line"
                    )
                host_id = line_hosts[int(rng.integers(len(line_hosts)))]
                builder, _, rejected = self._transaction(
                    builder,
                    rng,
                    "additional_t_junction",
                    lambda current: self._pick_source(
                        rng,
                        current,
                        profile="endpoint_rich",
                    ),
                    lambda trial, source, selected_host=host_id: trial.t_junction(
                        selected_host,
                        source,
                        scale=float(rng.uniform(0.08, 0.13)),
                    ),
                )
                placement_rejections.extend(rejected)

            for bridge_index in range(bridge_count):
                builder, _, rejected = self._transaction(
                    builder,
                    rng,
                    f"cycle_bridge_{bridge_index + 1}",
                    lambda current: self._pick_source(
                        rng, current, profile="single_line"
                    ),
                    lambda trial, source: trial.bridge(source),
                    attempts=8,
                )
                placement_rejections.extend(rejected)

        return builder, placement_rejections


def write_sample_artifacts(
    drawing: CanonicalDrawing,
    output_dir: Path,
    source_index: int,
    prepared: PreparedDrawing | None = None,
) -> dict:
    sample_dir = Path(output_dir) / "samples" / drawing.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    clean_svg = svg_document(drawing)
    amodal_svg = svg_document(drawing, primitives=drawing.primitives_amodal)
    if prepared is not None and prepared.drawing is not drawing:
        raise ValueError("prepared artifact does not belong to drawing")
    clean = prepared.clean if prepared is not None else render_clean(drawing)
    degraded = degrade_patent_scan(clean, seed=drawing.seed + 701)
    masks = prepared.masks if prepared is not None else render_masks(drawing)
    semantic = semantic_preview(drawing)
    report = (
        prepared.quality
        if prepared is not None
        else evaluate_quality(
            drawing,
            clean=clean,
            masks=masks,
            require_vertical_slice=drawing.processing.get("operator") == "t_junction",
            difficulty_gate=(
                drawing.difficulty
                if drawing.difficulty in {"medium", "hard", "very_hard"}
                else None
            ),
        )
    )
    if not report["accepted"]:
        raise ValueError(f"refusing to write failed sample {drawing.sample_id}: {report['failures']}")

    (sample_dir / "visible.svg").write_text(clean_svg)
    (sample_dir / "amodal.svg").write_text(amodal_svg)
    from PIL import Image

    Image.fromarray(clean).save(sample_dir / "clean.png")
    Image.fromarray(degraded).save(sample_dir / "degraded.png")
    Image.fromarray(semantic).save(sample_dir / "semantic.png")
    make_triptych(clean, degraded, semantic, drawing.sample_id).save(
        sample_dir / "preview.png"
    )
    np.savez_compressed(sample_dir / "masks.npz", **masks)
    puhachov = puhachov_arrays(drawing, clean=clean)
    if prepared is not None:
        free2cad = _free2cad_for_source(prepared, source_index)
    else:
        free2cad = free2cad_arrays(
            drawing,
            source_index=source_index,
            object_mask=masks["object"],
        )
    np.savez_compressed(sample_dir / "puhachov.npz", **puhachov)
    np.savez_compressed(sample_dir / "free2cad_edges.npz", **free2cad)

    visible_graph = {
        "sample_id": drawing.sample_id,
        "primitives": [item.to_dict() for item in drawing.primitives_visible],
        "junctions": [item.__dict__ for item in drawing.junctions_visible],
    }
    amodal_graph = {
        "sample_id": drawing.sample_id,
        "primitives": [item.to_dict() for item in drawing.primitives_amodal],
        "junctions": [item.__dict__ for item in drawing.junctions_amodal],
    }
    interaction_graph = {
        "sample_id": drawing.sample_id,
        "nodes": [
            item.__dict__ for item in drawing.components if item.source_dataset != "generated"
        ],
        "edges": [item.__dict__ for item in drawing.interactions],
    }
    (sample_dir / "visible_graph.json").write_text(
        json.dumps(visible_graph, indent=2, sort_keys=True) + "\n"
    )
    (sample_dir / "amodal_graph.json").write_text(
        json.dumps(amodal_graph, indent=2, sort_keys=True) + "\n"
    )
    (sample_dir / "interaction_graph.json").write_text(
        json.dumps(interaction_graph, indent=2, sort_keys=True) + "\n"
    )
    (sample_dir / "quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    drawing.images = {
        "clean": "clean.png",
        "degraded": "degraded.png",
        "semantic": "semantic.png",
        "preview": "preview.png",
        "visible_svg": "visible.svg",
        "amodal_svg": "amodal.svg",
        "masks": "masks.npz",
        "puhachov": "puhachov.npz",
        "free2cad_edges": "free2cad_edges.npz",
    }
    (sample_dir / "sample.json").write_text(
        json.dumps(drawing.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    keypoint_counts = np.bincount(puhachov["kps"][:, 2], minlength=3).tolist()
    free2cad_counts = np.bincount(free2cad["types"], minlength=5).tolist()
    return {
        "sample_id": drawing.sample_id,
        "seed": drawing.seed,
        "operator": drawing.processing.get("operator"),
        "difficulty": drawing.difficulty,
        "relative_path": str(sample_dir.relative_to(output_dir)),
        "sources": [item.__dict__ for item in drawing.sources],
        "quality": report,
        "training_targets": {
            "puhachov_keypoints": {
                "endpoint": int(keypoint_counts[0]),
                "junction": int(keypoint_counts[1]),
                "corner": int(keypoint_counts[2]),
            },
            "free2cad_edges": {
                "line": int(free2cad_counts[0]),
                "arc": int(free2cad_counts[1]),
                "circle": int(free2cad_counts[2]),
                "polyline": int(free2cad_counts[3]),
                "bezier": int(free2cad_counts[4]),
            },
        },
    }
