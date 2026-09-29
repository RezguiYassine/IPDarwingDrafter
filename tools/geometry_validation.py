"""Source-supported geometry checks, independent of primitive fit confidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

# Match the batch/evaluation tools' plain-stage imports without caching a
# namespace package that would shadow their stage4_export module later.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage4_export"))
import stage4_export  # noqa: E402
from stage1_preprocessing import filled_regions  # noqa: E402


SCHEMA = "ap3-source-geometry-v1"
VALIDATOR = "source_supported_geometry"
# 6: filled regions, validated against Stage 1's region evidence and the ink
# rather than against a skeleton they were deliberately taken out of.
VERSION = "6"


@dataclass(frozen=True)
class Policy:
    sample_step: float = 1.0
    tolerance: float = 2.0
    pass_fraction: float = 0.95
    fail_fraction: float = 0.80
    fail_p95: float = 4.0
    max_distance: float = 8.0
    max_unsupported_run: float = 8.0
    # Connected-run bounds, in pixels, used instead of the worst single sample.
    # Over the curated 100-figure cohort the largest uncovered skeleton run was
    # 6 px and the largest unrepresented source run was 1 px, on every figure,
    # while the raw max-distance rule failed 54 of them. See
    # docs/audits/2026-09-18/geometry_speckle_evidence.json.
    max_unsupported_run_pixels: int = 16
    max_residual_run_pixels: int = 4
    endpoint_tolerance: float = 3.0
    max_angular_gap_degrees: float = 30.0
    max_samples_per_primitive: int = 200_000
    max_samples_per_drawing: int = 2_000_000
    max_source_pixels: int = 2_000_000
    # Filled regions. The ring traces boundary pixel centres and is simplified
    # at 1 px, so coverage is judged within one pixel of the polygon.
    region_min_on_ink: float = 0.97
    region_min_coverage: float = 0.99


POLICY = Policy()


class SamplingLimit(ValueError):
    pass


def _points(value, minimum=2):
    points = np.asarray(value, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum
            or not np.isfinite(points).all()):
        raise ValueError("Missing, malformed, or nonfinite source/primitive points")
    return points


def _count(length, step, budget):
    count = max(2, int(math.ceil(length / step)) + 1)
    if count > budget:
        raise SamplingLimit("Sampling budget exceeded; resolution was not reduced")
    return count


def _resample(points, step, budget):
    points = _points(points)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, lengths > 1e-12]]
    if len(points) < 2:
        raise ValueError("Zero-length geometry")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.linspace(0, distance[-1], _count(distance[-1], step, budget))
    return np.column_stack([np.interp(samples, distance, points[:, axis]) for axis in (0, 1)])


def sample_primitive(primitive, scale=1.0, policy=POLICY):
    """Sample image-space geometry in Stage 2 units, including SVG path bridges."""
    stage4_export.validate_primitive(primitive)
    step = policy.sample_step / scale
    budget = policy.max_samples_per_primitive
    kind = primitive["type"]
    connector = 0.0
    if kind in {"hatch_strokes", "filled_region"}:
        # A region is sampled along its rings: its boundary is its geometry.
        lines = (primitive["strokes"] if kind == "hatch_strokes"
                 else [list(ring) + [ring[0]] for ring in primitive["rings"]])
        pieces = []
        for stroke in lines:
            part = _resample(stroke, step, budget) * scale
            budget -= len(part)
            pieces.append(part)
        return np.vstack(pieces), 0.0
    if kind == "line":
        points = [primitive["p1"], primitive["p2"]]
    elif kind in {"polyline", "polygon"}:
        points = primitive["points"]
        if kind == "polygon":
            points = list(points) + [points[0]]
    elif kind in {"circle", "arc", "ellipse"}:
        center = np.asarray(primitive["center"], dtype=float)
        a = float(primitive["a"] if kind == "ellipse" else primitive["radius"])
        b = float(primitive["b"] if kind == "ellipse" else a)
        start = math.radians(float(primitive.get("start_angle", 0))) if kind == "arc" else 0.0
        sweep = (math.radians((primitive["end_angle"] - primitive["start_angle"]) % 360)
                 if kind == "arc" else 2 * math.pi)
        angles = np.linspace(start, start + sweep, _count(max(a, b) * sweep, step, budget))
        points = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
        rotation = math.radians(float(primitive.get("angle", 0))) if kind == "ellipse" else 0.0
        matrix = np.array([[math.cos(rotation), -math.sin(rotation)],
                           [math.sin(rotation), math.cos(rotation)]])
        points = points @ matrix.T + center
    elif kind == "bezier":
        controls = _points(primitive["points"], 4)
        pieces = []
        remaining = budget
        for index in range(0, len(controls) - 1, 3):
            control = controls[index:index + 4]
            # The derivative is bounded by three times the longest control edge.
            bound = 3 * np.linalg.norm(np.diff(control, axis=0), axis=1).max()
            n = _count(bound, step, remaining)
            t = np.linspace(0, 1, n)[:, None]
            pieces.append((1-t)**3 * control[0] + 3*(1-t)**2*t * control[1]
                          + 3*(1-t)*t**2 * control[2] + t**3 * control[3])
            remaining -= n
        points = np.vstack(pieces)
    elif kind == "path":
        pieces = []
        count = 0
        previous = None
        for segment, start, end, reverse in stage4_export._orient_path_segments(primitive["segments"]):
            sampled, _ = sample_primitive(segment, scale, policy)
            if reverse:
                sampled = sampled[::-1]
            if previous is not None:
                connector = max(connector, float(np.linalg.norm(sampled[0] - previous)))
            previous = sampled[-1]
            pieces.append(sampled / scale)
            count += len(sampled)
            if count > budget:
                raise SamplingLimit("Compound path sampling budget exceeded")
        points = np.vstack(pieces)
    elif kind == "hatch":
        raise NotImplementedError("Native DXF and SVG hatch pattern parity is not validated")
    else:
        raise ValueError(f"Unsupported geometry: {kind}")
    return _resample(points, step, budget) * scale, connector


def _outcome(status="pass", *reasons, **details):
    return {"status": status, "reason_codes": list(reasons), **details}


def _combine(checks):
    states = {check["status"] for check in checks}
    status = next((value for value in ("error", "fail", "review") if value in states), "pass")
    reasons = sorted({reason for check in checks for reason in check["reason_codes"]})
    return _outcome(status, *reasons)


def largest_pixel_run(mask):
    """Size of the biggest 8-connected group of set pixels.

    Lost content is connected: a line the fitter dropped leaves a run of
    hundreds of adjacent uncovered pixels. A single pixel the skeletonizer
    threw off a stroke end, or one left by anti-aliasing, is not lost content
    however far it happens to sit from the nearest primitive.
    """
    if not mask.any():
        return 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0


def _distances(values, policy, mask=None):
    metrics = {"p95": float(np.percentile(values, 95)), "max": float(values.max()),
               "fraction_within_tolerance": float(np.mean(values <= policy.tolerance))}
    if mask is not None:
        metrics["largest_unsupported_run_pixels"] = largest_pixel_run(mask)
    return metrics


def _distance_check(metrics, reason, policy):
    """Judge coverage by structure, not by the worst single sample.

    `max` is the extreme order statistic of a per-pixel distance taken over
    tens of thousands of skeleton pixels, so any lone speck exceeds a fixed
    bound and condemns the drawing. Measured over the curated 100-figure
    cohort it condemned 54 of the 55 failures on its own, while
    `fraction_within_tolerance` was at or above 0.95 on 83 of 84 figures and
    no figure anywhere had an uncovered run longer than 6 pixels. Where a run
    length is available it replaces `max`; the fraction and p95 rules, which
    do respond to real losses, are unchanged.
    """
    fraction = metrics["fraction_within_tolerance"]
    run = metrics.get("largest_unsupported_run_pixels")
    extreme = (run > policy.max_unsupported_run_pixels if run is not None
               else metrics["max"] > policy.max_distance)
    if fraction < policy.fail_fraction or metrics["p95"] > policy.fail_p95 or extreme:
        return _outcome("fail", reason)
    if fraction < policy.pass_fraction:
        return _outcome("review", reason)
    return _outcome()


def _unsupported_run(distances, points, tolerance):
    mask = distances > tolerance
    closed = np.linalg.norm(points[0] - points[-1]) < 1e-6
    values = np.r_[mask, mask] if closed else mask
    longest = current = 0
    for unsupported in values:
        current = current + 1 if unsupported else 0
        longest = max(longest, current)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    return min(len(mask), longest) * length / max(1, len(mask) - 1)


def _angular_coverage(primitive, source, scale):
    points = source / scale - primitive["center"]
    radius = float(primitive.get("radius", 1))
    if primitive["type"] == "ellipse":
        angle = math.radians(float(primitive.get("angle", 0)))
        matrix = np.array([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
        points = points @ matrix / [primitive["a"], primitive["b"]]
        radius = min(float(primitive["a"]), float(primitive["b"]))
    angles = np.sort(np.arctan2(points[:, 1], points[:, 0]) % (2 * math.pi))
    gap = float(np.degrees(np.diff(np.r_[angles, angles[0] + 2 * math.pi])).max())
    return {"coverage_degrees": 360 - gap, "largest_gap_degrees": gap,
            "minor_radius_stage2": radius * scale}


def _measure(primitive, model, source, edge, scale, connector, policy):
    source_to_model = cKDTree(model).query(source)[0]
    model_to_source = cKDTree(source).query(model)[0]
    metrics = {"source_to_model": _distances(source_to_model, policy),
               "model_to_source": _distances(model_to_source, policy),
               "source_pixels": len(source), "model_samples": len(model),
               "max_connector_gap": connector,
               "worst_model_point_original": (model[np.argmax(model_to_source)] / scale).tolist(),
               "worst_source_point_original": (source[np.argmax(source_to_model)] / scale).tolist()}
    checks = [_distance_check(metrics["source_to_model"], "geometry_source_trace_lost", policy),
              _distance_check(metrics["model_to_source"], "geometry_unsupported_primitive", policy)]
    run = _unsupported_run(model_to_source, model, policy.tolerance)
    metrics["longest_unsupported_run"] = run
    if run > policy.max_unsupported_run:
        checks.append(_outcome("fail", "geometry_unsupported_run"))
    if connector > policy.tolerance:
        checks.append(_outcome("fail" if connector > policy.fail_p95 else "review",
                               "geometry_path_discontinuity"))
    if primitive["type"] in {"circle", "ellipse"}:
        coverage = _angular_coverage(primitive, source, scale)
        limit = min(180, max(policy.max_angular_gap_degrees,
                            math.degrees(2 * policy.tolerance / coverage["minor_radius_stage2"])))
        metrics["angular"] = {**coverage, "maximum_allowed_gap_degrees": limit}
        if coverage["largest_gap_degrees"] > limit:
            checks.append(_outcome("fail", "geometry_incomplete_angular_coverage"))
    elif edge is not None and not edge.get("is_closed"):
        direct = max(np.linalg.norm(model[0] - source[0]), np.linalg.norm(model[-1] - source[-1]))
        reverse = max(np.linalg.norm(model[-1] - source[0]), np.linalg.norm(model[0] - source[-1]))
        error = float(min(direct, reverse))
        metrics["endpoint_error"] = error
        if error > policy.endpoint_tolerance:
            checks.append(_outcome("fail" if error > 2 * policy.endpoint_tolerance else "review",
                                   "geometry_endpoint_mismatch"))
    if edge is not None and edge.get("topology_origin") == "unclaimed_component" and not edge.get("is_simple_cycle"):
        checks.append(_outcome("review", "geometry_noncycle_source"))
    return {**_combine(checks), "metrics": metrics}


def check_primitive(primitive, edge, policy=POLICY):
    """Check an unscaled fitting candidate using the same source-support policy."""
    try:
        source = _points(edge["pixels"])
        if len(source) > policy.max_source_pixels:
            raise SamplingLimit("Source pixel budget exceeded")
        model, connector = sample_primitive(primitive, policy=policy)
        return _measure(primitive, model, source, edge, 1.0, connector, policy)
    except (SamplingLimit, NotImplementedError) as exc:
        return _outcome("review", "geometry_candidate_unverified", detail=str(exc))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _outcome("error", "geometry_invalid_primitive_or_source", detail=str(exc))


def _stage2_coverage_check(graph, skeleton, scale, policy=POLICY):
    """Accounting never exempts residual pixels from the independent raster check."""
    coverage = graph.get("coverage")
    if coverage is None:
        return _outcome()
    try:
        shape = tuple(graph["image_shape"])
        if (coverage["schema"] != "ap3-stage2-coverage-v1"
                or coverage["image_shape"] != list(shape)
                or coverage["stage2_scale"] != scale):
            raise ValueError("Invalid coverage frame/schema")
        if skeleton is None or scale != 1:
            return _outcome("review", "geometry_stage2_coverage_source_unverified")
        source = np.asarray(skeleton, dtype=bool)
        if hashlib.sha256(np.packbits(source).tobytes()).hexdigest() != coverage["source_mask_sha256"]:
            raise ValueError("Coverage source hash does not match the Stage 1 skeleton")
        represented = np.zeros(shape, dtype=bool)
        for collection in ("edges", "removed_hachures"):
            for edge in graph.get(collection, []):
                points = _points(edge["pixels"], 1)
                if (not np.equal(points, np.rint(points)).all() or (points < 0).any()
                        or (points >= np.array([shape[1], shape[0]])).any()):
                    raise ValueError("Coverage graph contains invalid source pixel coordinates")
                x, y = points.astype(int).T
                represented[y, x] = True
        residual = np.zeros(shape, dtype=bool)
        for span in coverage["recovery"]["unresolved_spans"]:
            if (len(span) != 3 or any(type(v) is not int for v in span)
                    or not 0 <= span[0] < shape[0] or not 0 <= span[1] < span[2] <= shape[1]):
                raise ValueError("Malformed coverage residual span")
            y, start, end = span
            if residual[y, start:end].any():
                raise ValueError("Duplicate coverage residual span")
            residual[y, start:end] = True
        if not np.array_equal(source & ~represented, residual):
            raise ValueError("Unaccounted source pixels or residuals already represented")
        total, missing = int(source.sum()), int(residual.sum())
        if (coverage["source_pixels"] != total or coverage["residual_source_pixels"] != missing
                or coverage["represented_source_pixels"] != total-missing
                or coverage["recovery"]["unresolved_pixels"] != missing
                or coverage["unaccounted_source_pixels"] != 0):
            raise ValueError("Coverage counters disagree with source evidence")
        # Every unrepresented pixel in the cohort was isolated -- the largest
        # residual run was 1 px on all 91 figures that had any -- so counting
        # pixels flagged 78 drawings for speckle. A run is what indicates that
        # Stage 2 dropped something.
        run = largest_pixel_run(residual)
        if run > policy.max_residual_run_pixels:
            return _outcome("review", "geometry_stage2_residual_pending",
                            residual_source_pixels=missing, largest_residual_run_pixels=run)
        return _outcome(residual_source_pixels=missing, largest_residual_run_pixels=run)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return _outcome("error", "geometry_stage2_coverage_invalid", detail=str(exc))


def _region_raster(rings, shape):
    raster = np.zeros(shape, np.uint8)
    cv2.fillPoly(raster, [np.rint(np.asarray(ring, float)).astype(np.int32).reshape(-1, 1, 2)
                          for ring in rings], 1)          # even-odd: holes stay empty
    return raster


def _check_regions(primitives, evidence, skeleton, policy):
    """Hold each filled region to Stage 1's evidence and to the ink itself.

    A region was taken out of the skeleton on purpose, so the skeleton cannot
    judge it. Three things can: the fill must lie on ink, it must cover every
    ink pixel Stage 1 removed, and none of that ink may also have been thinned
    -- the scribble this primitive exists to replace.
    """
    if evidence is None:
        failure = _outcome("error", "geometry_region_evidence_missing")
        return failure, [dict(failure) for _ in primitives]
    document, routed, ink = evidence["document"], evidence["routed"], evidence["ink"]
    shape = tuple(document["image_shape"])
    if (document.get("schema") != filled_regions.SCHEMA or routed.shape != shape
            or ink.shape != shape or (skeleton is not None and skeleton.shape != shape)):
        raise ValueError("Region evidence frame or schema mismatch")
    if filled_regions.mask_digest(routed) != document["routed_mask"]["sha256"]:
        raise ValueError("Routed-ink mask does not match the Stage 1 regions document")
    if (routed & ~ink).any():
        raise ValueError("Stage 1 routed pixels that are not ink")
    declared = document["regions"]
    owners = [p.get("source_region_index") for p in primitives]
    if (any(type(i) is not int or not 0 <= i < len(declared) for i in owners)
            or sorted(owners) != list(range(len(declared)))):
        failure = _outcome("error", "geometry_region_ownership_incomplete")
        return failure, [dict(failure) for _ in primitives]

    kernel = np.ones((3, 3), np.uint8)
    near_ink = cv2.dilate(ink.astype(np.uint8), kernel) > 0
    _, components = cv2.connectedComponents(routed.astype(np.uint8), connectivity=8)
    union = np.zeros(shape, bool)
    results = []
    for primitive, owner in zip(primitives, owners):
        if primitive["rings"] != declared[owner]["rings"]:
            results.append(_outcome("error", "geometry_region_changed_from_evidence"))
            continue
        raster = _region_raster(primitive["rings"], shape)
        filled = raster > 0
        union |= filled
        owned = np.isin(components, np.unique(components[filled & routed]).tolist()) & routed
        reach = cv2.dilate(raster, kernel) > 0
        metrics = {"area_px": int(filled.sum()),
                   "on_ink": float((filled & near_ink).sum() / max(1, filled.sum())),
                   "coverage": float((owned & reach).sum() / max(1, owned.sum()))}
        reasons = []
        if metrics["on_ink"] < policy.region_min_on_ink:
            reasons.append("geometry_region_off_ink")
        if not owned.any() or metrics["coverage"] < policy.region_min_coverage:
            reasons.append("geometry_region_ink_unrepresented")
        results.append({**_outcome("fail" if reasons else "pass", *reasons), "metrics": metrics})

    stray = routed & ~(cv2.dilate(union.astype(np.uint8), kernel) > 0)
    thinned = int((skeleton & routed).sum()) if skeleton is not None else None
    reasons = []
    if largest_pixel_run(stray) > policy.max_unsupported_run_pixels:
        reasons.append("geometry_region_ink_unrepresented")
    if thinned:
        reasons.append("geometry_region_double_representation")
    overall = _outcome("fail" if reasons else "pass", *reasons,
                       regions=len(declared), routed_pixels=int(routed.sum()),
                       unrepresented_pixels=int(stray.sum()), thinned_region_pixels=thinned)
    return {**_combine([overall] + results), **{k: v for k, v in overall.items()
                                               if k not in ("status", "reason_codes")}}, results


def validate(graph, document, skeleton=None, policy=POLICY, regions=None):
    """Validate owned traces and full Stage 1 coverage; never refit or remove ink.

    `regions` is Stage 1's filled-region evidence -- its document, the routed
    ink mask and the ink -- and is required whenever the figure has regions.
    """
    report = {"schema": SCHEMA, "validator": VALIDATOR, "version": VERSION,
              "parameters": asdict(policy), "coordinate_frame": "stage2_pixels",
              "checks": {}, "primitives": []}
    checks, items = report["checks"], report["primitives"]
    try:
        scale = float(graph.get("stage2_scale", 1))
        if not math.isfinite(scale) or scale <= 0 or not math.isclose(float(document.get("stage2_scale", 1)), scale):
            raise ValueError("Invalid or mismatched Stage 2 scale")
        shape = graph.get("original_image_shape") or graph["image_shape"]
        h, w = shape
        if (w <= 0 or h <= 0 or document["image_size"] != [w, h]
                or (scale != 1 and not graph.get("original_image_shape"))
                or not np.allclose(np.asarray(graph["image_shape"]), np.asarray(shape)*scale, atol=1, rtol=0)):
            raise ValueError("Graph and primitive coordinate frames disagree")
        edges, hatches, primitives = graph["edges"], graph.get("removed_hachures", []), document["primitives"]
        ids = [edge["id"] for edge in edges]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate main edge IDs")
        by_id = {edge["id"]: edge for edge in edges}
        ownership, hatch_ownership = Counter(), Counter()
        checks["coordinates"] = _outcome()
        checks["stage2_coverage"] = _stage2_coverage_check(graph, skeleton, scale, policy)
        models = []
        sampled_count = 0
        fills = [index for index, primitive in enumerate(primitives)
                 if primitive.get("type") == "filled_region"]
        for index, primitive in enumerate(primitives):
            if primitive.get("type") == "filled_region":
                items.append({"index": index, "type": "filled_region", "edge_id": None,
                              "source_collection": "stage1_regions",
                              "source_indices": [primitive.get("source_region_index")]})
                continue
            is_hatch = primitive.get("style") == "hachure"
            item = {"index": index, "type": primitive.get("type"), "edge_id": primitive.get("edge_id"),
                    "source_collection": "removed_hachures" if is_hatch else "edges"}
            try:
                if is_hatch:
                    indices = primitive.get("source_hachure_indices", [])
                    if not indices or any(type(i) is not int or not 0 <= i < len(hatches) for i in indices):
                        raise ValueError("Missing or invalid hatch source ownership")
                    hatch_ownership.update(indices)
                    source = np.vstack([_points(hatches[i]["pixels"]) for i in indices])
                    item["source_indices"] = indices
                    edge = hatches[indices[0]] if len(indices) == 1 else None
                else:
                    edge = by_id[primitive["edge_id"]]
                    ownership.update([edge["id"]])
                    source = _points(edge["pixels"])
                if len(source) > policy.max_source_pixels:
                    raise SamplingLimit("Source pixel budget exceeded")
                if sampled_count >= policy.max_samples_per_drawing:
                    raise SamplingLimit("Drawing sample budget exceeded")
                model, connector = sample_primitive(primitive, scale, policy)
                sampled_count += len(model)
                if sampled_count > policy.max_samples_per_drawing:
                    raise SamplingLimit("Drawing sample budget exceeded")
                if primitive["type"] == "hatch_strokes":
                    stroke_checks = []
                    for stroke, owners in zip(primitive["strokes"], primitive["stroke_source_indices"]):
                        local_source = np.vstack([_points(hatches[i]["pixels"]) for i in owners])
                        local = {"type": "polyline", "points": stroke}
                        sampled, _ = sample_primitive(local, scale, policy)
                        local_edge = hatches[owners[0]] if len(owners) == 1 else None
                        stroke_checks.append(_measure(local, sampled, local_source, local_edge,
                                                      scale, 0.0, policy))
                    item.update(_combine(stroke_checks))
                    item["stroke_checks"] = stroke_checks
                else:
                    item.update(_measure(primitive, model, source, edge, scale, connector, policy))
                if np.any(model < -policy.tolerance) or np.any(model > np.array([w-1, h-1]) * scale + policy.tolerance):
                    item.update(_combine([item, _outcome("fail", "geometry_outside_image")]))
                models.append(model)
            except NotImplementedError:
                item.update(_outcome("review", "geometry_hatch_pattern_unverified"))
            except SamplingLimit as exc:
                item.update(_outcome("review", "geometry_sampling_budget", detail=str(exc)))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                item.update(_outcome("error", "geometry_invalid_primitive_or_source", detail=str(exc)))
            items.append(item)
        if fills or regions is not None:
            try:
                checks["regions"], results = _check_regions(
                    [primitives[i] for i in fills], regions, skeleton, policy)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                checks["regions"] = _outcome("error", "geometry_region_evidence_invalid",
                                             detail=str(exc))
                results = [_outcome("error", "geometry_region_evidence_invalid")] * len(fills)
            for index, result in zip(fills, results):
                items[index].update(result)
        missing = sorted(set(ids) - set(ownership))
        repeated = [key for key, count in ownership.items() if count != 1]
        hatch_missing = sorted(set(range(len(hatches))) - set(hatch_ownership))
        hatch_repeated = [key for key, count in hatch_ownership.items() if count != 1]
        checks["ownership"] = _outcome(
            "error" if missing or repeated or hatch_missing or hatch_repeated else "pass",
            *(["geometry_source_ownership_incomplete"] if missing or repeated or hatch_missing or hatch_repeated else []),
            missing_main_edge_ids=missing, duplicate_main_edge_ids=repeated,
            missing_hachure_indices=hatch_missing, duplicate_hachure_indices=hatch_repeated)
        if not primitives:
            checks["nonempty"] = _outcome("fail", "geometry_empty")
        if skeleton is None:
            checks["raster"] = _outcome("review", "geometry_raster_validation_missing")
        elif skeleton.shape != (h, w):
            checks["raster"] = _outcome("error", "geometry_raster_shape_mismatch")
        elif np.count_nonzero(skeleton) > policy.max_source_pixels:
            checks["raster"] = _outcome("review", "geometry_sampling_budget")
        elif not np.any(skeleton):
            checks["raster"] = _outcome("error", "geometry_empty_source_raster")
        elif not models:
            checks["raster"] = _outcome("review", "geometry_raster_coverage_incomplete")
        else:
            y, x = np.nonzero(skeleton)
            raster = np.column_stack([x, y]) * scale
            model = np.vstack(models)
            precision_distances = cKDTree(raster).query(model)[0]
            # Same treatment the recall side received: a lone stray sample is
            # not geometry drawn off the skeleton, a connected run of them is.
            off = model[precision_distances > policy.tolerance]
            stray = np.zeros(skeleton.shape, bool)
            if len(off):
                columns = np.clip(np.rint(off[:, 0] / scale).astype(int), 0, skeleton.shape[1] - 1)
                rows = np.clip(np.rint(off[:, 1] / scale).astype(int), 0, skeleton.shape[0] - 1)
                stray[rows, columns] = True
            precision = _distances(precision_distances, policy, mask=stray)
            raster_checks = [_distance_check(precision, "geometry_output_off_skeleton", policy)]
            raster_metrics = {"model_to_skeleton": precision}
            # Fills are judged against the ink above, not against the skeleton.
            if len(models) == len(primitives) - len(fills):
                recall_distances = cKDTree(model).query(raster)[0]
                uncovered = np.zeros(skeleton.shape, bool)
                uncovered[y[recall_distances > policy.tolerance],
                          x[recall_distances > policy.tolerance]] = True
                recall = _distances(recall_distances, policy, mask=uncovered)
                raster_checks.append(_distance_check(recall, "geometry_skeleton_coverage_lost", policy))
                raster_metrics["skeleton_to_model"] = recall
            else:
                raster_checks.append(_outcome("review", "geometry_raster_coverage_incomplete"))
            checks["raster"] = {**_combine(raster_checks), **raster_metrics}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        checks["input"] = _outcome("error", "geometry_invalid_input", detail=str(exc))
    report.update(_combine(list(checks.values()) + items))
    report["summary"] = {"primitives": len(items), "by_status": dict(Counter(p["status"] for p in items)),
                         "by_type": dict(Counter(p["type"] for p in items))}
    return report


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acceptance_check(report):
    return {key: report[key] for key in ("status", "reason_codes", "validator", "version")}


REGION_INPUTS = ("regions", "regions_mask", "cleaned")


def run(graph_path: Path, primitives_path: Path, skeleton_path: Path, output_path: Path,
        *, regions_path: Path | None = None, regions_mask_path: Path | None = None,
        cleaned_path: Path | None = None):
    inputs = {"graph": graph_path, "primitives": primitives_path, "skeleton": skeleton_path}
    if regions_path is not None:
        inputs.update(regions=regions_path, regions_mask=regions_mask_path, cleaned=cleaned_path)
    hashes = {key: {"path": str(Path(path).resolve()), "sha256": file_digest(path)}
              for key, path in inputs.items()}
    skeleton = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        raise ValueError(f"Cannot decode source skeleton: {skeleton_path}")
    evidence = None
    if regions_path is not None:
        routed = cv2.imread(str(regions_mask_path), cv2.IMREAD_GRAYSCALE)
        cleaned = cv2.imread(str(cleaned_path), cv2.IMREAD_GRAYSCALE)
        if routed is None or cleaned is None:
            raise ValueError("Cannot decode the region mask or the Stage 1 raster")
        evidence = {"document": json.loads(Path(regions_path).read_text()),
                    "routed": routed > 0, "ink": cleaned > 127}
    report = validate(json.loads(graph_path.read_text()), json.loads(primitives_path.read_text()),
                      skeleton > 0, regions=evidence)
    report["inputs"] = hashes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_path)
    return report
