"""Audit whether PatentVec Stage-2 labels supervise raster topology support.

The production graph tracer needs endpoints and junctions for every semantic
layer that survives into Stage 2. This audit compares classical crossing-number
clusters in each exported skeleton with exact Puhachov labels and attributes
unmatched clusters to semantic masks such as hatching or annotations.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from scipy.spatial import cKDTree

from syntheticData.patentvec.stage2_targets import (
    ARCHIVE_LABEL_CONTRACT,
    DEFAULT_TOPOLOGY_MASKS,
    STAGE2_LABEL_CONTRACTS,
    SUPPORTED_RASTER_TOPOLOGY_CONTRACT,
    relabel_puhachov_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage2_strokeextraction"))
from stage2_stroke_extract import _cn_map_vectorized  # noqa: E402


def _clusters(mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8,
    )
    clusters = []
    for index in range(1, count):
        x, y, width, height = stats[index, :4]
        local_y, local_x = np.where(
            labels[y:y + height, x:x + width] == index
        )
        clusters.append(np.column_stack((local_y + y, local_x + x)))
    return clusters


def _near_label(
    cluster: np.ndarray,
    tree: cKDTree | None,
    radius: float,
) -> bool:
    if tree is None:
        return False
    # Cluster coordinates are y,x; query in x,y order.
    distances, _indices = tree.query(cluster[:, ::-1], distance_upper_bound=radius)
    return bool(np.isfinite(distances).any())


def audit_sample(
    skeleton: np.ndarray,
    keypoints: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    match_radius: float = 8.0,
    mask_radius: int = 2,
    topology_masks: tuple[str, ...] = DEFAULT_TOPOLOGY_MASKS,
) -> dict[str, Any]:
    binary = (skeleton > 0).astype(np.uint8)
    cn = _cn_map_vectorized(binary)
    cluster_groups = {
        "endpoint": (0, _clusters((cn == 1) & (binary > 0))),
        "junction": (1, _clusters((cn >= 3) & (binary > 0))),
    }
    label_trees = {}
    for kind_id in (0, 1):
        candidates = (
            keypoints[keypoints[:, 2] == kind_id]
            if len(keypoints) else keypoints
        )
        label_trees[kind_id] = (
            cKDTree(candidates[:, :2].astype(float)) if len(candidates) else None
        )
    kernel_size = mask_radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    support_masks = {
        name: cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0
        for name, mask in masks.items()
    }
    report: dict[str, Any] = {
        "total": Counter(),
        "matched": Counter(),
        "semantic": defaultdict(Counter),
    }
    topology_support = np.zeros(binary.shape, dtype=bool)
    for name in topology_masks:
        if name in masks:
            topology_support |= np.asarray(masks[name]) > 0
    foreground_pixels = int(np.count_nonzero(binary))
    outside_pixels = int(np.count_nonzero((binary > 0) & ~topology_support))
    report["skeleton"] = {
        "foreground_pixels": foreground_pixels,
        "outside_topology_support_pixels": outside_pixels,
        "outside_topology_support_rate": (
            outside_pixels / foreground_pixels if foreground_pixels else 0.0
        ),
    }
    for kind_name, (kind_id, clusters) in cluster_groups.items():
        for cluster in clusters:
            report["total"][kind_name] += 1
            matched = _near_label(cluster, label_trees[kind_id], match_radius)
            if matched:
                report["matched"][kind_name] += 1
            ys, xs = cluster[:, 0], cluster[:, 1]
            for semantic, support in support_masks.items():
                if bool(support[ys, xs].any()):
                    report["semantic"][semantic][f"{kind_name}_total"] += 1
                    if matched:
                        report["semantic"][semantic][f"{kind_name}_matched"] += 1
                    else:
                        report["semantic"][semantic][f"{kind_name}_unmatched"] += 1
    return {
        "total": dict(report["total"]),
        "matched": dict(report["matched"]),
        "semantic": {
            name: dict(counts) for name, counts in report["semantic"].items()
        },
        "skeleton": report["skeleton"],
    }


def _iter_samples(shards: list[Path]) -> Iterator[tuple[str, bytes, bytes]]:
    for shard in shards:
        with tarfile.open(shard) as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile()
            }
            puhachov_names = sorted(
                name for name in members if name.endswith("/puhachov.npz")
            )
            for puhachov_name in puhachov_names:
                prefix = puhachov_name.rsplit("/", 1)[0]
                masks_name = f"{prefix}/masks.npz"
                if masks_name not in members:
                    continue
                puhachov_file = archive.extractfile(members[puhachov_name])
                masks_file = archive.extractfile(members[masks_name])
                if puhachov_file is None or masks_file is None:
                    continue
                yield (
                    prefix.rsplit("/", 1)[-1],
                    puhachov_file.read(),
                    masks_file.read(),
                )


def run(
    dataset: Path,
    *,
    sample_limit: int,
    match_radius: float,
    mask_radius: int,
    label_contract: str = ARCHIVE_LABEL_CONTRACT,
) -> dict[str, Any]:
    shards = sorted((dataset / "shards").glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"no tar shards under {dataset / 'shards'}")
    aggregate_total = Counter()
    aggregate_matched = Counter()
    aggregate_semantic: dict[str, Counter] = defaultdict(Counter)
    aggregate_skeleton = Counter()
    aggregate_input_total = Counter()
    aggregate_input_matched = Counter()
    aggregate_input_semantic: dict[str, Counter] = defaultdict(Counter)
    aggregate_input_skeleton = Counter()
    separate_input_samples = 0
    target_semantics = Counter()
    distractor_semantics = Counter()
    samples = 0
    for _sample_id, puhachov_bytes, masks_bytes in _iter_samples(shards):
        if label_contract != ARCHIVE_LABEL_CONTRACT:
            puhachov_bytes = relabel_puhachov_payload(
                puhachov_bytes, masks_bytes, label_contract=label_contract
            )
        with np.load(io.BytesIO(puhachov_bytes), allow_pickle=False) as puhachov:
            input_skeleton = np.asarray(puhachov["skeleton"])
            has_separate_topology = "topology_skeleton" in puhachov.files
            skeleton = (
                puhachov["topology_skeleton"]
                if has_separate_topology
                else input_skeleton
            )
            keypoints = puhachov["kps"]
            metadata = json.loads(str(puhachov["meta"]))
        with np.load(io.BytesIO(masks_bytes), allow_pickle=False) as mask_archive:
            masks = {name: mask_archive[name] for name in mask_archive.files}
        sample = audit_sample(
            skeleton,
            keypoints,
            masks,
            match_radius=match_radius,
            mask_radius=mask_radius,
            topology_masks=tuple(
                metadata.get("topology_support_masks", DEFAULT_TOPOLOGY_MASKS)
            ),
        )
        aggregate_total.update(sample["total"])
        aggregate_matched.update(sample["matched"])
        for semantic, counts in sample["semantic"].items():
            aggregate_semantic[semantic].update(counts)
        aggregate_skeleton.update(
            {
                key: int(value)
                for key, value in sample["skeleton"].items()
                if key.endswith("_pixels")
            }
        )
        if has_separate_topology:
            input_sample = audit_sample(
                input_skeleton,
                keypoints,
                masks,
                match_radius=match_radius,
                mask_radius=mask_radius,
                topology_masks=tuple(
                    metadata.get("input_support_masks", DEFAULT_TOPOLOGY_MASKS)
                ),
            )
            aggregate_input_total.update(input_sample["total"])
            aggregate_input_matched.update(input_sample["matched"])
            for semantic, counts in input_sample["semantic"].items():
                aggregate_input_semantic[semantic].update(counts)
            aggregate_input_skeleton.update(
                {
                    key: int(value)
                    for key, value in input_sample["skeleton"].items()
                    if key.endswith("_pixels")
                }
            )
            separate_input_samples += 1
        target_semantics.update(metadata.get("target_semantics", []))
        distractor_semantics.update(metadata.get("distractor_semantics_in_skeleton", []))
        samples += 1
        if samples >= sample_limit:
            break

    def semantic_report(
        aggregate: dict[str, Counter],
    ) -> dict[str, dict[str, float | int]]:
        report = {}
        for semantic, counts in sorted(aggregate.items()):
            values = dict(counts)
            for kind in ("endpoint", "junction"):
                total = counts[f"{kind}_total"]
                unmatched = counts[f"{kind}_unmatched"]
                values[f"{kind}_unmatched_rate"] = (
                    unmatched / total if total else 0.0
                )
            report[semantic] = values
        return report

    def skeleton_report(aggregate: Counter) -> dict[str, float | int]:
        report = dict(aggregate)
        foreground_pixels = aggregate["foreground_pixels"]
        outside_pixels = aggregate["outside_topology_support_pixels"]
        report["outside_topology_support_rate"] = (
            outside_pixels / foreground_pixels if foreground_pixels else 0.0
        )
        return report

    result = {
        "dataset": str(dataset),
        "label_contract": label_contract,
        "sample_limit": sample_limit,
        "samples_audited": samples,
        "match_radius_px": match_radius,
        "semantic_mask_radius_px": mask_radius,
        "cn_clusters": dict(aggregate_total),
        "cn_clusters_matched_by_exact_label": dict(aggregate_matched),
        "target_semantics_sample_frequency": dict(target_semantics),
        "distractor_semantics_sample_frequency": dict(distractor_semantics),
        "semantic_support": semantic_report(aggregate_semantic),
        "skeleton_support": skeleton_report(aggregate_skeleton),
    }
    if separate_input_samples:
        result["input_skeleton_view"] = {
            "samples": separate_input_samples,
            "cn_clusters": dict(aggregate_input_total),
            "cn_clusters_matched_by_structural_label": dict(
                aggregate_input_matched
            ),
            "cn_clusters_intentionally_unlabelled": {
                kind: aggregate_input_total[kind] - aggregate_input_matched[kind]
                for kind in ("endpoint", "junction")
            },
            "semantic_support": semantic_report(aggregate_input_semantic),
            "skeleton_support": skeleton_report(aggregate_input_skeleton),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--sample-limit", type=int, default=500)
    parser.add_argument("--match-radius", type=float, default=8.0)
    parser.add_argument("--mask-radius", type=int, default=2)
    parser.add_argument(
        "--label-contract",
        choices=STAGE2_LABEL_CONTRACTS,
        default=ARCHIVE_LABEL_CONTRACT,
        help="Audit archived labels or the deterministic topology relabeling.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.dataset,
        sample_limit=args.sample_limit,
        match_radius=args.match_radius,
        mask_radius=args.mask_radius,
        label_contract=args.label_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    hatch = report["semantic_support"].get("hatch", {})
    print(f"samples audited: {report['samples_audited']}")
    print(
        "hatch support: "
        f"endpoints={hatch.get('endpoint_total', 0)} "
        f"unmatched={hatch.get('endpoint_unmatched', 0)}; "
        f"junctions={hatch.get('junction_total', 0)} "
        f"unmatched={hatch.get('junction_unmatched', 0)}"
    )
    input_view = report.get("input_skeleton_view")
    if input_view:
        input_hatch = input_view["semantic_support"].get("hatch", {})
        print(
            "full-input hatch negatives: "
            f"endpoints={input_hatch.get('endpoint_unmatched', 0)}/"
            f"{input_hatch.get('endpoint_total', 0)}; "
            f"junctions={input_hatch.get('junction_unmatched', 0)}/"
            f"{input_hatch.get('junction_total', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
