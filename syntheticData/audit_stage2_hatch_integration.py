"""Audit the Stage-2 C3 hatch-suppressed topology integration.

This is an oracle test: exact C3 structural keypoints are supplied to the
production tracer on the full object-plus-hatch skeleton.  It compares the
historical sparse-keypoint walk with the hatch-bridge path before model error
is introduced.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from syntheticData.audit_stage2_topology_contract import _iter_samples
from syntheticData.patentvec.stage2_targets import (
    HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
    relabel_puhachov_payload,
)
from stage2_strokeextraction.stage2_stroke_extract import (
    KP_CORNER,
    KP_ENDPOINT,
    KP_JUNCTION,
    _chain_length,
    _clusters_from_points,
    _extract_topology,
    _hatch_bridge_junction_clusters,
    _remove_hachures_cnn,
    _run_hachure_topology_prepass,
    _simplify_graph,
)


KEYPOINT_TYPES = (KP_ENDPOINT, KP_JUNCTION, KP_CORNER)


def _structural_clusters(
    skeleton: np.ndarray, keypoints: np.ndarray
) -> list[dict]:
    points = [
        {
            "x": int(x),
            "y": int(y),
            "type": KEYPOINT_TYPES[int(kind)],
            "confidence": 1.0,
        }
        for x, y, kind in keypoints.tolist()
    ]
    clusters = _clusters_from_points(points, skeleton)
    for cluster in clusters:
        cluster["structural_seed"] = True
        cluster["topology_origin"] = "oracle_structural"
    return clusters


def _trace(
    skeleton: np.ndarray,
    keypoints: np.ndarray,
    hatch_mask: np.ndarray,
    *,
    bridged: bool,
    strong_inside_frac: float,
) -> tuple[list[dict], list[dict], int]:
    structural = _structural_clusters(skeleton, keypoints)
    clusters = list(structural)
    if bridged:
        clusters.extend(
            _hatch_bridge_junction_clusters(
                skeleton, hatch_mask, structural, dedup_radius=5.0
            )
        )
    nodes, edges = _extract_topology(
        skeleton,
        clusters,
        unclaimed_mode="closed_only" if bridged else "all",
    )
    nodes, edges = _simplify_graph(
        nodes,
        edges,
        spur_min_len=6.0,
        collinear_max_angle=28.0,
        junction_merge_radius=0.0,
    )
    removal_config = {
        "hachure_cnn_inside_frac": 0.60,
        "hachure_cnn_min_length": 4.0,
        "hachure_cnn_max_removed_edge_ratio": 1.0,
        "hachure_cnn_protect_structural_seed_edges": bridged,
        "hachure_cnn_main_require_line_like": bridged,
        "hachure_cnn_main_min_straightness": 0.70,
        "hachure_cnn_main_max_residual_rms": 2.2,
        "hachure_cnn_strong_inside_frac": strong_inside_frac,
    }
    nodes, edges, removed = _remove_hachures_cnn(
        nodes,
        edges,
        hatch_mask,
        removal_config,
        pass_name="cnn",
    )
    if removed:
        nodes, edges = _simplify_graph(
            nodes,
            edges,
            spur_min_len=6.0,
            collinear_max_angle=28.0,
            junction_merge_radius=0.0,
        )
    return nodes, edges, len(removed)


def _graph_mask(
    shape: tuple[int, int], nodes: list[dict], edges: list[dict]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for edge in edges:
        for x, y in edge.get("pixels", []):
            if 0 <= int(x) < width and 0 <= int(y) < height:
                mask[int(y), int(x)] = True
    for node in nodes:
        x, y = int(node.get("x", -1)), int(node.get("y", -1))
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = True
    return mask


def _pixel_counts(
    prediction: np.ndarray, target: np.ndarray, radius: float
) -> dict[str, int]:
    predicted = int(np.count_nonzero(prediction))
    expected = int(np.count_nonzero(target))
    distance_to_target = distance_transform_edt(~target)
    distance_to_prediction = distance_transform_edt(~prediction)
    return {
        "predicted": predicted,
        "expected": expected,
        "precision_supported": int(
            np.count_nonzero(prediction & (distance_to_target <= radius))
        ),
        "recall_supported": int(
            np.count_nonzero(target & (distance_to_prediction <= radius))
        ),
    }


def _edge_stats(edges: list[dict]) -> dict[str, float]:
    lengths = [
        _chain_length(edge.get("pixels", []))
        for edge in edges
        if not edge.get("is_closed")
    ]
    return {
        "edges": float(len(edges)),
        "open_edges": float(len(lengths)),
        "median_open_edge_length": float(np.median(lengths)) if lengths else 0.0,
        "short_open_edge_ratio": (
            float(np.mean(np.asarray(lengths) < 15.0)) if lengths else 0.0
        ),
    }


def _finalize(counts: Counter, sample_stats: list[dict[str, float]]) -> dict:
    precision = counts["precision_supported"] / max(counts["predicted"], 1)
    recall = counts["recall_supported"] / max(counts["expected"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    keys = sample_stats[0].keys() if sample_stats else ()
    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_counts": dict(counts),
        "mean_per_sample": {
            key: float(np.mean([sample[key] for sample in sample_stats]))
            for key in keys
        },
    }


def run(
    dataset: Path,
    sample_limit: int,
    match_radius: float,
    hatch_mask_dilation: int,
    strong_inside_frac: float,
) -> dict[str, Any]:
    shards = sorted((dataset / "shards").glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"no tar shards under {dataset / 'shards'}")

    pixel_counts = {"sparse": Counter(), "bridged": Counter()}
    sample_stats: dict[str, list[dict[str, float]]] = {
        "sparse": [],
        "bridged": [],
    }
    hatch_leakage = {"sparse": Counter(), "bridged": Counter()}
    side_pixel_counts = Counter()
    side_sample_stats: list[dict[str, float]] = []
    side_overlap = Counter()
    samples = 0
    for _sample_id, puhachov_payload, masks_payload in _iter_samples(shards):
        relabelled = relabel_puhachov_payload(
            puhachov_payload,
            masks_payload,
            label_contract=HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
        )
        with np.load(io.BytesIO(relabelled), allow_pickle=False) as data:
            skeleton = np.asarray(data["skeleton"])
            topology = np.asarray(data["topology_skeleton"]) > 0
            keypoints = np.asarray(data["kps"])
        with np.load(io.BytesIO(masks_payload), allow_pickle=False) as data:
            exact_hatch_mask = (np.asarray(data["hatch"]) > 0).astype(np.uint8)
        if hatch_mask_dilation > 0:
            size = hatch_mask_dilation * 2 + 1
            hatch_mask = cv2.dilate(
                exact_hatch_mask,
                np.ones((size, size), dtype=np.uint8),
            )
        else:
            hatch_mask = exact_hatch_mask

        hatch_only = (exact_hatch_mask > 0) & ~topology
        hatch_target = skeletonize(exact_hatch_mask > 0)
        _cleaned, side_edges, _removed_pixels = _run_hachure_topology_prepass(
            skeleton,
            hatch_mask,
            {
                "remove_hachures": True,
                "hachure_cnn_inside_frac": 0.60,
                "hachure_cnn_min_length": 4.0,
                "hachure_cnn_max_removed_edge_ratio": 0.92,
                "hachure_topology_prepass_max_removed_pixel_ratio": 0.95,
                "simplify_graph": True,
                "spur_min_length": 6.0,
                "merge_collinear_max_angle": 28.0,
                "junction_merge_radius": 0.0,
            },
            max_search_radius=60,
        )
        side_prediction = _graph_mask(skeleton.shape, [], side_edges)
        side_pixel_counts.update(
            _pixel_counts(side_prediction, hatch_target, match_radius)
        )
        side_stats = _edge_stats(side_edges)
        side_stats["removed_hatch_edges"] = float(len(side_edges))
        side_sample_stats.append(side_stats)
        side_overlap.update(
            {
                "side_structural_pixels": int(
                    np.count_nonzero(side_prediction & topology)
                ),
                "side_pixels": int(np.count_nonzero(side_prediction)),
            }
        )
        for name, bridged in (("sparse", False), ("bridged", True)):
            nodes, edges, removed = _trace(
                skeleton,
                keypoints,
                hatch_mask,
                bridged=bridged,
                strong_inside_frac=strong_inside_frac,
            )
            prediction = _graph_mask(skeleton.shape, nodes, edges)
            pixel_counts[name].update(
                _pixel_counts(prediction, topology, match_radius)
            )
            stats = _edge_stats(edges)
            stats["removed_hatch_edges"] = float(removed)
            sample_stats[name].append(stats)
            hatch_leakage[name].update(
                {
                    "predicted_hatch_only": int(
                        np.count_nonzero(prediction & hatch_only)
                    ),
                    "predicted": int(np.count_nonzero(prediction)),
                }
            )
        samples += 1
        if samples >= sample_limit:
            break

    arms = {
        name: _finalize(pixel_counts[name], sample_stats[name])
        for name in ("sparse", "bridged")
    }
    for name in arms:
        leakage = hatch_leakage[name]
        arms[name]["hatch_only_leakage_rate"] = (
            leakage["predicted_hatch_only"] / max(leakage["predicted"], 1)
        )
        arms[name]["hatch_only_counts"] = dict(leakage)
    side_layer = _finalize(side_pixel_counts, side_sample_stats)
    side_layer["structural_overlap_rate"] = (
        side_overlap["side_structural_pixels"]
        / max(side_overlap["side_pixels"], 1)
    )
    side_layer["structural_overlap_counts"] = dict(side_overlap)
    return {
        "dataset": str(dataset),
        "label_contract": HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
        "samples_audited": samples,
        "match_radius_px": match_radius,
        "hatch_mask_dilation_px": hatch_mask_dilation,
        "strong_inside_fraction": strong_inside_frac,
        "arms": arms,
        "side_layer": side_layer,
        "bridged_minus_sparse": {
            key: arms["bridged"][key] - arms["sparse"][key]
            for key in (
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
                "hatch_only_leakage_rate",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--match-radius", type=float, default=2.0)
    parser.add_argument("--hatch-mask-dilation", type=int, default=2)
    parser.add_argument("--strong-inside-frac", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hatch_mask_dilation < 0:
        parser.error("--hatch-mask-dilation must be non-negative")
    if not 0.0 <= args.strong_inside_frac <= 1.0:
        parser.error("--strong-inside-frac must be between zero and one")
    report = run(
        args.dataset,
        args.sample_limit,
        args.match_radius,
        args.hatch_mask_dilation,
        args.strong_inside_frac,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
