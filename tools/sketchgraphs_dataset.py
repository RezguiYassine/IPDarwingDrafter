"""Download and prepare SketchGraphs supervision for Stages 2 and 3.

The official filtered SketchGraphs release stores construction sequences in a
memory-mapped flat-array format.  This tool preserves the official split, renders
clean Stage-2 skeleton/keypoint labels, then runs the repository's actual Stage-2
topology code to produce realistic per-edge Stage-3 examples.

Outputs
-------
Stage 2: ``<output>/stage2/<split>/*.npz`` with ``skeleton`` and ``kps``.
Stage 3: ``<output>/stage3/<split>/shard_*.npz`` with encoded point sequences.

The Stage-3 labels are matched to source primitives after topology extraction.
This is important: complete source entities are not equivalent to the fragments
that Stage 3 receives after intersections and graph simplification.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SG_REPO = PROJECT_ROOT / "stage2_strokeextraction" / "research" / "repos" / "SketchGraphs"
for path in (SG_REPO, PROJECT_ROOT / "stage2_strokeextraction"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from sketchgraphs.data import Arc, Circle, EntityType, Line
    from sketchgraphs.data import flat_array, sketch_from_sequence
except ImportError as exc:  # pragma: no cover - exercised by setup failures
    raise SystemExit(
        "SketchGraphs is not importable. Clone PrincetonLIPS/SketchGraphs into "
        f"{SG_REPO} and run `.venv/bin/pip install lz4 -e {SG_REPO}`."
    ) from exc

import stage2_stroke_extract as s2  # noqa: E402


DOWNLOADS = {
    "train": (
        "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_train.npy",
        6_151_102_626,
    ),
    "validation": (
        "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_validation.npy",
        211_121_676,
    ),
    "test": (
        "https://sketchgraphs.cs.princeton.edu/sequence/sg_t16_test.npy",
        209_307_405,
    ),
}

KP_TO_ID = {s2.KP_ENDPOINT: 0, s2.KP_JUNCTION: 1, s2.KP_CORNER: 2}
TYPE_TO_ID = {"LINE": 0, "ARC": 1, "CIRCLE": 2, "POLYLINE": 3}


@dataclass
class Primitive:
    entity_id: str
    kind: str
    world_points: np.ndarray
    pixel_points: np.ndarray
    center_px: np.ndarray | None
    radius_px: float | None
    closed: bool
    tree: cKDTree


@dataclass
class PrepareConfig:
    canvas: int = 512
    margin: int = 24
    stroke_width: int = 2
    max_pts: int = 64
    include_construction: bool = False
    match_p90_px: float = 2.5
    min_edge_px: float = 5.0
    write_stage2: bool = True


def _download_one(url: str, target: Path, expected_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    current = target.stat().st_size if target.exists() else 0
    if current == expected_size:
        print(f"verified {target} ({current:,} bytes)")
        return
    if current > expected_size:
        raise RuntimeError(f"{target} is larger than expected; remove it and retry")

    request = urllib.request.Request(url)
    if current:
        request.add_header("Range", f"bytes={current}-")
    with urllib.request.urlopen(request) as response:
        partial = response.status == 206
        mode = "ab" if current and partial else "wb"
        if mode == "wb":
            current = 0
        total = expected_size - current
        with target.open(mode) as stream, tqdm(
            total=total, initial=0, unit="B", unit_scale=True, desc=target.name
        ) as progress:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                progress.update(len(chunk))
    actual = target.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"incomplete {target}: {actual:,} != {expected_size:,}")


def download(raw_dir: Path, splits: Iterable[str]) -> None:
    for split in splits:
        url, size = DOWNLOADS[split]
        _download_one(url, raw_dir / f"sg_t16_{split}.npy", size)


def _arc_angles(entity: Arc) -> tuple[float, float]:
    base = math.atan2(entity.yDir, entity.xDir)
    if entity.clockwise:
        start = base - entity.endParam
        end = base - entity.startParam
    else:
        start = base + entity.startParam
        end = base + entity.endParam
    while end <= start:
        end += 2.0 * math.pi
    if end - start > 2.0 * math.pi + 1e-6:
        end = start + 2.0 * math.pi
    return start, end


def _coarse_entity_points(entity) -> tuple[str, np.ndarray, bool] | None:
    if isinstance(entity, Line):
        return "LINE", np.stack([entity.start_point, entity.end_point]), False
    if isinstance(entity, Arc):
        start, end = _arc_angles(entity)
        angles = np.linspace(start, end, 129)
        points = np.column_stack([
            entity.xCenter + entity.radius * np.cos(angles),
            entity.yCenter + entity.radius * np.sin(angles),
        ])
        return "ARC", points, False
    if isinstance(entity, Circle):
        angles = np.linspace(0.0, 2.0 * math.pi, 257)
        points = np.column_stack([
            entity.xCenter + entity.radius * np.cos(angles),
            entity.yCenter + entity.radius * np.sin(angles),
        ])
        return "CIRCLE", points, True
    return None


def _world_to_pixel(points: np.ndarray, center: np.ndarray, scale: float, canvas: int) -> np.ndarray:
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = (points[:, 0] - center[0]) * scale + (canvas - 1) / 2.0
    out[:, 1] = (center[1] - points[:, 1]) * scale + (canvas - 1) / 2.0
    return out


def _build_primitives(sketch, cfg: PrepareConfig) -> tuple[list[Primitive], np.ndarray]:
    coarse = []
    for entity_id, entity in sketch.entities.items():
        if entity.isConstruction and not cfg.include_construction:
            continue
        item = _coarse_entity_points(entity)
        if item is not None:
            coarse.append((entity_id, entity, *item))
    if not coarse:
        raise ValueError("no supported non-construction primitives")

    all_points = np.concatenate([item[3] for item in coarse])
    if not np.isfinite(all_points).all():
        raise ValueError("non-finite geometry")
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    span = float((upper - lower).max())
    if span <= 1e-9:
        raise ValueError("degenerate geometry bounds")
    center = (lower + upper) / 2.0
    scale = (cfg.canvas - 2.0 * cfg.margin) / span

    primitives = []
    for entity_id, entity, kind, _points, closed in coarse:
        if isinstance(entity, Line):
            world = np.stack([entity.start_point, entity.end_point])
            center_px, radius_px = None, None
        elif isinstance(entity, Arc):
            start, end = _arc_angles(entity)
            length_px = max(1.0, abs(end - start) * entity.radius * scale)
            angles = np.linspace(start, end, int(np.clip(math.ceil(length_px * 1.5), 12, 2048)))
            world = np.column_stack([
                entity.xCenter + entity.radius * np.cos(angles),
                entity.yCenter + entity.radius * np.sin(angles),
            ])
            center_px = _world_to_pixel(
                np.array([[entity.xCenter, entity.yCenter]]), center, scale, cfg.canvas
            )[0]
            radius_px = float(entity.radius * scale)
        else:
            length_px = max(1.0, 2.0 * math.pi * entity.radius * scale)
            angles = np.linspace(
                0.0, 2.0 * math.pi,
                int(np.clip(math.ceil(length_px * 1.5), 32, 2048)),
                endpoint=False,
            )
            world = np.column_stack([
                entity.xCenter + entity.radius * np.cos(angles),
                entity.yCenter + entity.radius * np.sin(angles),
            ])
            world = np.vstack([world, world[0]])
            center_px = _world_to_pixel(
                np.array([[entity.xCenter, entity.yCenter]]), center, scale, cfg.canvas
            )[0]
            radius_px = float(entity.radius * scale)

        pixels = _world_to_pixel(world, center, scale, cfg.canvas)
        primitives.append(Primitive(
            entity_id=str(entity_id), kind=kind, world_points=world,
            pixel_points=pixels, center_px=center_px, radius_px=radius_px,
            closed=closed, tree=cKDTree(pixels),
        ))
    return primitives, np.array([center[0], center[1], scale], dtype=np.float64)


def _render_skeleton(primitives: list[Primitive], cfg: PrepareConfig) -> np.ndarray:
    canvas = np.zeros((cfg.canvas, cfg.canvas), dtype=np.uint8)
    for primitive in primitives:
        pts = np.round(primitive.pixel_points).astype(np.int32)
        cv2.polylines(
            canvas, [pts], primitive.closed, 255, cfg.stroke_width,
            lineType=cv2.LINE_8,
        )
    return s2._skeletonize(canvas > 0).astype(np.uint8) * 255


def _analytic_corners(primitives: list[Primitive], skeleton: np.ndarray) -> list[dict]:
    endpoint_records = []
    for primitive in primitives:
        if primitive.closed or len(primitive.pixel_points) < 2:
            continue
        pts = primitive.pixel_points
        for point, neighbour in ((pts[0], pts[1]), (pts[-1], pts[-2])):
            direction = neighbour - point
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                endpoint_records.append((point, direction / norm))
    if len(endpoint_records) < 2:
        return []

    points = np.array([r[0] for r in endpoint_records])
    tree = cKDTree(points)
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in tree.query_pairs(r=2.5):
        union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)

    candidates = []
    cos_threshold = math.cos(math.radians(180.0 - 35.0))
    for members in groups.values():
        if len(members) != 2:
            continue
        d1, d2 = endpoint_records[members[0]][1], endpoint_records[members[1]][1]
        if float(np.dot(d1, d2)) <= cos_threshold:
            continue
        point = points[members].mean(axis=0)
        candidates.append({
            "x": int(round(point[0])), "y": int(round(point[1])),
            "type": s2.KP_CORNER, "confidence": 1.0,
        })
    return s2._clusters_from_points(candidates, skeleton, snap_radius=4)


def _keypoints(primitives: list[Primitive], skeleton: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Build Stage-2 keypoint supervision without tracing the stroke graph."""
    cn = [
        cluster for cluster in s2._cn_keypoint_clusters(skeleton)
        if cluster["type"] in (s2.KP_ENDPOINT, s2.KP_JUNCTION)
    ]
    corners = _analytic_corners(primitives, skeleton)
    if cn and corners:
        base = np.array([[c["x"], c["y"]] for c in cn], dtype=np.float64)
        corners = [
            c for c in corners
            if np.min((base[:, 0] - c["x"]) ** 2 + (base[:, 1] - c["y"]) ** 2) > 25.0
        ]
    clusters = cn + corners
    kps = np.array(
        [(c["x"], c["y"], KP_TO_ID[c["type"]]) for c in clusters],
        dtype=np.int32,
    ) if clusters else np.zeros((0, 3), dtype=np.int32)
    return kps, clusters


def _keypoints_and_graph(
    primitives: list[Primitive], skeleton: np.ndarray
) -> tuple[np.ndarray, list[dict], list[dict]]:
    kps, clusters = _keypoints(primitives, skeleton)

    nodes, edges = s2._extract_topology(skeleton, clusters)
    nodes, edges = s2._simplify_graph(
        nodes, edges, spur_min_len=6.0,
        collinear_max_angle=28.0, junction_merge_radius=0.0,
    )
    edges = s2._smooth_edges(edges)
    return kps, nodes, edges


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _sagitta_ratio(points: np.ndarray) -> float:
    """Maximum deviation from the endpoint chord, normalized by chord length."""
    if len(points) < 3:
        return 0.0
    chord = points[-1] - points[0]
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return 0.0
    normal = np.array([-chord[1], chord[0]], dtype=np.float64) / length
    sagitta = float(np.abs((points - points[0]) @ normal).max())
    return sagitta / length


def _match_primitive(edge: dict, primitives: list[Primitive], threshold: float) -> Primitive | None:
    raw = edge["pixels"] if edge.get("is_closed") else (edge.get("smooth_pts") or edge["pixels"])
    points = np.asarray(raw, dtype=np.float64)
    if len(points) < 2:
        return None
    scores = []
    for primitive in primitives:
        distances = primitive.tree.query(points, k=1)[0]
        scores.append(float(np.quantile(distances, 0.90)))
    best = int(np.argmin(scores))
    return primitives[best] if scores[best] <= threshold else None


def _normalise_edge(points: np.ndarray, max_pts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.0
    scale = float((upper - lower).max())
    if scale < 1e-9:
        raise ValueError("degenerate edge")
    norm = ((points - center) / scale + 0.5).astype(np.float32)
    if len(norm) > max_pts:
        idx = np.round(np.linspace(0, len(norm) - 1, max_pts)).astype(int)
        norm = norm[idx]
    mask = np.zeros(max_pts, dtype=bool)
    mask[:len(norm)] = True
    if len(norm) < max_pts:
        norm = np.vstack([
            norm, np.full((max_pts - len(norm), 2), -1.0, dtype=np.float32)
        ])
    return norm, mask, center, scale


def _stage3_sample(
    edge: dict, primitive: Primitive, cfg: PrepareConfig, source_index: int
) -> dict | None:
    is_closed = bool(edge.get("is_closed"))
    if is_closed:
        if primitive.kind != "CIRCLE":
            return None
        raw = np.asarray(edge["pixels"], dtype=np.float64)
    else:
        raw = np.asarray(edge.get("smooth_pts") or edge["pixels"], dtype=np.float64)
    if _polyline_length(raw) < cfg.min_edge_px:
        return None

    points, mask, center, scale = _normalise_edge(raw, cfg.max_pts)
    params = np.zeros(6, dtype=np.float32)
    # Stage 2 splits circles at junctions. An open fragment is an arc for the
    # per-edge Stage 3 contract; supervising it as a complete circle produces
    # impossible center/radius targets relative to its local normalization.
    command = "ARC" if primitive.kind == "CIRCLE" and not is_closed else primitive.kind
    if command == "ARC" and _sagitta_ratio(raw) < 0.01:
        # The raster edge has no measurable curvature. This agrees with the
        # production fitter's geometric arc guard.
        command = "LINE"
    if command == "LINE":
        params[0:2] = (raw[0] - center) / scale + 0.5
        params[2:4] = (raw[-1] - center) / scale + 0.5
    elif command in ("ARC", "CIRCLE"):
        if primitive.center_px is None or primitive.radius_px is None:
            return None
        center_norm = (primitive.center_px - center) / scale + 0.5
        params[0:2] = center_norm
        params[2] = primitive.radius_px / scale
        if command == "CIRCLE" and (
            np.any(np.abs(center_norm - 0.5) > 0.30) or params[2] > 0.65
        ):
            # A closed graph edge that covers only a local part of its source
            # circle cannot determine that circle after per-edge
            # normalization. Keep these malformed loops out of supervision.
            return None
        if command == "ARC":
            start = math.atan2(raw[0, 1] - primitive.center_px[1], raw[0, 0] - primitive.center_px[0])
            end = math.atan2(raw[-1, 1] - primitive.center_px[1], raw[-1, 0] - primitive.center_px[0])
            params[3] = (math.degrees(start) % 360.0) / 360.0
            params[4] = (math.degrees(end) % 360.0) / 360.0
    else:
        return None
    return {
        "points": points, "mask": mask, "type": TYPE_TO_ID[command],
        "params": params, "source_index": source_index,
    }


_WORKER_DATA = None
_WORKER_CFG = None
_WORKER_STAGE2_DIR = None


def _worker_init(data_path: str, config: dict, stage2_dir: str) -> None:
    global _WORKER_DATA, _WORKER_CFG, _WORKER_STAGE2_DIR
    _WORKER_DATA = flat_array.load_dictionary_flat(data_path)
    _WORKER_CFG = PrepareConfig(**config)
    _WORKER_STAGE2_DIR = Path(stage2_dir)


def _process_index(index: int) -> dict:
    cfg: PrepareConfig = _WORKER_CFG
    result = {"index": index, "status": "ok", "stage3": []}
    try:
        sketch = sketch_from_sequence(_WORKER_DATA["sequences"][index])
        primitives, _transform = _build_primitives(sketch, cfg)
        skeleton = _render_skeleton(primitives, cfg)
        kps, nodes, edges = _keypoints_and_graph(primitives, skeleton)
        if not edges:
            raise ValueError("topology has no edges")

        meta = {
            "source": "SketchGraphs", "source_index": index,
            "n_primitives": len(primitives), "n_nodes": len(nodes),
            "n_edges": len(edges),
        }
        if cfg.write_stage2:
            np.savez_compressed(
                _WORKER_STAGE2_DIR / f"sg_{index:08d}.npz",
                skeleton=skeleton, kps=kps, meta=json.dumps(meta),
            )

        unmatched = short = 0
        samples = []
        for edge in edges:
            primitive = _match_primitive(edge, primitives, cfg.match_p90_px)
            if primitive is None:
                unmatched += 1
                continue
            sample = _stage3_sample(edge, primitive, cfg, index)
            if sample is None:
                short += 1
                continue
            samples.append(sample)
        result.update({
            "stage3": samples, "n_kps": np.bincount(kps[:, 2], minlength=3).tolist()
            if len(kps) else [0, 0, 0],
            "n_edges": len(edges), "n_unmatched": unmatched, "n_short": short,
        })
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return result


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer: list[dict] = []
        self.shard_index = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def add_many(self, samples: list[dict]) -> None:
        self.buffer.extend(samples)
        while len(self.buffer) >= self.shard_size:
            self._flush(self.buffer[:self.shard_size])
            del self.buffer[:self.shard_size]

    def _flush(self, samples: list[dict]) -> None:
        if not samples:
            return
        target = self.output_dir / f"shard_{self.shard_index:05d}.npz"
        np.savez(
            target,
            points=np.stack([s["points"] for s in samples]),
            mask=np.stack([s["mask"] for s in samples]),
            types=np.array([s["type"] for s in samples], dtype=np.uint8),
            params=np.stack([s["params"] for s in samples]),
            source_index=np.array([s["source_index"] for s in samples], dtype=np.int64),
        )
        self.shard_index += 1

    def close(self) -> None:
        self._flush(self.buffer)
        self.buffer.clear()


def _sample_indices(total: int, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=limit, replace=False))


def _audit_stage2(stage2_dir: Path, target: Path, limit: int = 24) -> None:
    paths = sorted(stage2_dir.glob("*.npz"))[:limit]
    if not paths:
        return
    thumbs = []
    colors = ((40, 190, 40), (40, 40, 230), (230, 120, 30))
    for path in paths:
        data = np.load(path)
        image = cv2.cvtColor(data["skeleton"], cv2.COLOR_GRAY2BGR)
        for x, y, kind in data["kps"]:
            cv2.circle(image, (int(x), int(y)), 5, colors[int(kind)], 1)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.putText(image, path.stem, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
        thumbs.append(image)
    cols = 6
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * 256, cols * 256, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y, x = divmod(i, cols)
        sheet[y * 256:(y + 1) * 256, x * 256:(x + 1) * 256] = thumb
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), sheet)


def prepare_split(
    data_path: Path, output: Path, split: str, limit: int, workers: int,
    seed: int, cfg: PrepareConfig, shard_size: int,
) -> dict:
    dataset = flat_array.load_dictionary_flat(data_path)
    total = len(dataset["sequences"])
    indices = _sample_indices(total, limit, seed)
    del dataset

    stage2_split = "validation" if split == "validation" else split
    stage3_split = "val" if split == "validation" else split
    stage2_dir = output / "stage2" / stage2_split
    stage3_dir = output / "stage3" / stage3_split
    stage2_dir.mkdir(parents=True, exist_ok=True)
    stage3_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in stage3_dir.glob("shard_*.npz"):
        old_shard.unlink()
    writer = ShardWriter(stage3_dir, shard_size)

    stats = Counter()
    kp_counts = np.zeros(3, dtype=np.int64)
    class_counts = np.zeros(4, dtype=np.int64)
    errors = Counter()
    config_dict = vars(cfg)

    if workers <= 1:
        _worker_init(str(data_path), config_dict, str(stage2_dir))
        results = map(_process_index, indices.tolist())
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(str(data_path), config_dict, str(stage2_dir)),
        )
        results = executor.map(_process_index, indices.tolist(), chunksize=8)

    try:
        for result in tqdm(results, total=len(indices), desc=f"prepare {split}"):
            stats["selected"] += 1
            if result["status"] != "ok":
                stats["errors"] += 1
                errors[result.get("error", "unknown")] += 1
                continue
            stats["accepted"] += 1
            stats["edges"] += result["n_edges"]
            stats["unmatched_edges"] += result["n_unmatched"]
            stats["short_edges"] += result["n_short"]
            kp_counts += np.asarray(result["n_kps"], dtype=np.int64)
            writer.add_many(result["stage3"])
            for sample in result["stage3"]:
                class_counts[sample["type"]] += 1
    finally:
        writer.close()
        if executor is not None:
            executor.shutdown(wait=True)

    report = {
        "split": split, "source_path": str(data_path), "source_total": total,
        "seed": seed, "limit": limit, "config": config_dict,
        **{k: int(v) for k, v in stats.items()},
        "keypoints": dict(zip(("endpoint", "junction", "corner"), kp_counts.tolist())),
        "stage3_classes": dict(zip(TYPE_TO_ID, class_counts.tolist())),
        "top_errors": errors.most_common(20),
    }
    report_path = output / f"report_{split}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if cfg.write_stage2:
        _audit_stage2(stage2_dir, output / f"audit_stage2_{split}.png")
    print(json.dumps(report, indent=2))
    return report


def inspect_split(path: Path, sample_size: int, seed: int) -> None:
    data = flat_array.load_dictionary_flat(path)
    indices = _sample_indices(len(data["sequences"]), sample_size, seed)
    entities = Counter()
    construction = Counter()
    for index in tqdm(indices, desc="inspect"):
        sketch = sketch_from_sequence(data["sequences"][int(index)])
        for entity in sketch.entities.values():
            entities[EntityType(entity.type).name] += 1
            construction["construction" if entity.isConstruction else "normal"] += 1
    print(json.dumps({
        "path": str(path), "sequences": len(data["sequences"]),
        "sample_size": len(indices), "entities": entities,
        "line_style": construction,
    }, indent=2))


def _split_list(spec: str) -> list[str]:
    values = [value.strip() for value in spec.split(",") if value.strip()]
    invalid = set(values) - set(DOWNLOADS)
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid splits: {sorted(invalid)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare official SketchGraphs filtered splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download")
    p_download.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/SketchGraphs/raw")
    p_download.add_argument("--splits", type=_split_list, default=list(DOWNLOADS))

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("path", type=Path)
    p_inspect.add_argument("--sample-size", type=int, default=10_000)
    p_inspect.add_argument("--seed", type=int, default=42)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/SketchGraphs/raw")
    p_prepare.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/SketchGraphsTraining")
    p_prepare.add_argument("--splits", type=_split_list, default=list(DOWNLOADS))
    p_prepare.add_argument("--limit", type=int, default=100_000, help="sketches per split; <=0 means all")
    p_prepare.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.add_argument("--canvas", type=int, default=512)
    p_prepare.add_argument("--margin", type=int, default=24)
    p_prepare.add_argument("--stroke-width", type=int, default=2)
    p_prepare.add_argument("--max-pts", type=int, default=64)
    p_prepare.add_argument("--include-construction", action="store_true")
    p_prepare.add_argument("--match-p90-px", type=float, default=2.5)
    p_prepare.add_argument("--min-edge-px", type=float, default=5.0)
    p_prepare.add_argument("--shard-size", type=int, default=50_000)
    p_prepare.add_argument(
        "--stage3-only", action="store_true",
        help="rebuild Stage 3 shards without rewriting Stage 2 label files",
    )

    args = parser.parse_args()
    if args.command == "download":
        download(args.raw_dir, args.splits)
    elif args.command == "inspect":
        inspect_split(args.path, args.sample_size, args.seed)
    else:
        cfg = PrepareConfig(
            canvas=args.canvas, margin=args.margin, stroke_width=args.stroke_width,
            max_pts=args.max_pts, include_construction=args.include_construction,
            match_p90_px=args.match_p90_px, min_edge_px=args.min_edge_px,
            write_stage2=not args.stage3_only,
        )
        for offset, split in enumerate(args.splits):
            path = args.raw_dir / f"sg_t16_{split}.npy"
            if not path.exists():
                raise SystemExit(f"missing {path}; run the download command first")
            prepare_split(
                path, args.output, split, args.limit, args.workers,
                args.seed + offset, cfg, args.shard_size,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
