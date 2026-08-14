"""Build inference-shaped Free2CAD supervision from Drawing2CAD.

The source SVG command stream is used only to supervise edges produced by the
real Stage 2 topology code.  A cubic SVG command is not blindly called a
Bezier: circular cubics are relabelled ARC/CIRCLE by a radial-residual guard.
Likewise, adjacent linear commands become POLYLINE only when one extracted
edge spans several source segments and is measurably non-straight.

The generator consumes the cached 1024 px skeleton/keypoint labels produced by
``tools.d2c_keypoint_labels`` and writes restart-safe NPZ shards compatible
with ``train_free2cad_v3.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2_ROOT = PROJECT_ROOT / "stage2_strokeextraction"
if str(STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE2_ROOT))

import stage2_stroke_extract as s2  # noqa: E402
from stage3_primitivesfitting.stage3_primitive_fit import (  # noqa: E402
    _reorder_loop_pixels,
)


TYPE_TO_ID = {
    "LINE": 0,
    "ARC": 1,
    "CIRCLE": 2,
    "POLYLINE": 3,
    "BEZIER": 4,
}
ID_TO_KP = {0: s2.KP_ENDPOINT, 1: s2.KP_JUNCTION, 2: s2.KP_CORNER}
ALL_VIEWS = ("Front", "Top", "Right", "FrontTopRight")

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN_RE = re.compile(rf"[MLCZ]|{_NUM}")
_DATTR_RE = re.compile(r'\bd="([^"]*)"')
_VIEWBOX_RE = re.compile(
    rf'viewBox="\s*({_NUM})[\s,]+({_NUM})[\s,]+({_NUM})[\s,]+({_NUM})\s*"'
)


@dataclass
class SourceSegment:
    path_id: int
    order: int
    command: str
    samples: np.ndarray


@dataclass
class SourceIndex:
    segments: list[SourceSegment]
    tree: cKDTree
    owners: np.ndarray


@dataclass
class PrepareConfig:
    max_pts: int = 64
    match_p90_px: float = 2.5
    path_coverage: float = 0.70
    min_edge_px: float = 5.0
    line_p90_px: float = 1.5
    circle_p90_px: float = 1.5
    circle_relative_residual: float = 0.03
    min_arc_sagitta_ratio: float = 0.01


def _cubic_points(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
    scale: float,
) -> np.ndarray:
    polygon_len = sum(
        np.linalg.norm(b - a) for a, b in ((p0, p1), (p1, p2), (p2, p3))
    )
    count = int(np.clip(math.ceil(polygon_len * scale / 1.25), 12, 384))
    t = np.linspace(0.0, 1.0, count + 1)
    mt = 1.0 - t
    return (
        mt[:, None] ** 3 * p0
        + 3.0 * mt[:, None] ** 2 * t[:, None] * p1
        + 3.0 * mt[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )


def _line_points(p0: np.ndarray, p1: np.ndarray, scale: float) -> np.ndarray:
    count = int(np.clip(math.ceil(np.linalg.norm(p1 - p0) * scale), 2, 2048))
    return np.linspace(p0, p1, count + 1)


def parse_svg_segments(svg_path: Path, render_width: int) -> list[SourceSegment]:
    """Parse Drawing2CAD's absolute M/L/C/Z paths into dense pixel segments."""
    text = svg_path.read_text()
    match = _VIEWBOX_RE.search(text)
    min_x, min_y, width = (0.0, 0.0, 200.0)
    if match:
        min_x, min_y, width = map(float, (match.group(1), match.group(2), match.group(3)))
    scale = render_width / width

    segments: list[SourceSegment] = []
    path_id = -1
    for d_attr in _DATTR_RE.findall(text):
        tokens = _TOKEN_RE.findall(d_attr)
        command = None
        current = None
        start = None
        order = 0
        i = 0
        while i < len(tokens):
            if tokens[i] in {"M", "L", "C", "Z"}:
                command = tokens[i]
                i += 1
            if command == "M":
                if i + 1 >= len(tokens):
                    break
                current = np.array([float(tokens[i]), float(tokens[i + 1])])
                i += 2
                start = current.copy()
                path_id += 1
                order = 0
                command = "L"
                continue
            if command == "L" and current is not None:
                if i + 1 >= len(tokens):
                    break
                end = np.array([float(tokens[i]), float(tokens[i + 1])])
                i += 2
                points = _line_points(current, end, scale)
                segments.append(SourceSegment(path_id, order, "L", points))
                current = end
                order += 1
                continue
            if command == "C" and current is not None:
                if i + 5 >= len(tokens):
                    break
                values = np.asarray([float(v) for v in tokens[i:i + 6]]).reshape(3, 2)
                i += 6
                points = _cubic_points(current, values[0], values[1], values[2], scale)
                segments.append(SourceSegment(path_id, order, "C", points))
                current = values[2]
                order += 1
                continue
            if command == "Z":
                if current is not None and start is not None and not np.allclose(current, start):
                    points = _line_points(current, start, scale)
                    segments.append(SourceSegment(path_id, order, "L", points))
                    current = start.copy()
                command = None
                continue
            # Unsupported or malformed input. Drawing2CAD is expected to use
            # absolute M/L/C/Z, so skipping is safer than silently mislabelling.
            i += 1

    offset = np.array([min_x, min_y], dtype=np.float64)
    for segment in segments:
        segment.samples = (segment.samples - offset) * scale
    return segments


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _edge_points(edge: dict) -> np.ndarray:
    """Return topologically ordered raw pixels for open and closed edges."""
    raw = np.asarray(edge.get("pixels") or [], dtype=np.float64)
    if edge.get("is_closed", False) and len(raw) >= 3:
        return _reorder_loop_pixels(raw)
    return raw


def _line_residual(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    if len(points) < 3 or not np.any(centered):
        return 0.0
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = np.array([-vt[0, 1], vt[0, 0]])
    return float(np.quantile(np.abs(centered @ normal), 0.90))


def _circle_fit(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    x, y = points[:, 0], points[:, 1]
    matrix = np.column_stack([x, y, np.ones(len(points))])
    rhs = -(x * x + y * y)
    a, b, c = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    center = np.array([-a / 2.0, -b / 2.0])
    radius_sq = float(center @ center - c)
    if radius_sq <= 1e-6:
        raise ValueError("degenerate circle")
    radius = math.sqrt(radius_sq)
    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    return center, radius, float(np.quantile(residual, 0.90))


def _normalise_edge(
    points: np.ndarray, max_pts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.0
    scale = float((upper - lower).max())
    if scale < 1e-9:
        raise ValueError("degenerate edge")
    normalised = ((points - center) / scale + 0.5).astype(np.float32)
    if len(normalised) > max_pts:
        indices = np.round(np.linspace(0, len(normalised) - 1, max_pts)).astype(int)
        normalised = normalised[indices]
    mask = np.zeros(max_pts, dtype=bool)
    mask[:len(normalised)] = True
    if len(normalised) < max_pts:
        normalised = np.vstack([
            normalised,
            np.full((max_pts - len(normalised), 2), -1.0, dtype=np.float32),
        ])
    return normalised, mask, center, scale


def _source_match(
    edge_points: np.ndarray,
    source: SourceIndex | list[SourceSegment],
    cfg: PrepareConfig,
) -> tuple[list[SourceSegment] | None, float]:
    index = source if isinstance(source, SourceIndex) else _source_index(source)
    segments = index.segments
    if not segments:
        return None, math.inf
    distances, nearest = index.tree.query(edge_points, k=1)
    p90 = float(np.quantile(distances, 0.90))
    if p90 > cfg.match_p90_px:
        return None, p90

    segment_ids = index.owners[nearest]
    path_votes: Counter[int] = Counter(
        segments[int(index)].path_id for index in segment_ids
    )
    path_id, path_count = path_votes.most_common(1)[0]
    if path_count / len(edge_points) < cfg.path_coverage:
        return None, p90
    min_votes = max(2, int(math.ceil(len(edge_points) * 0.03)))
    votes = Counter(int(index) for index in segment_ids)
    selected = [
        segments[index] for index, count in votes.items()
        if count >= min_votes and segments[index].path_id == path_id
    ]
    if not selected:
        selected = [segments[votes.most_common(1)[0][0]]]
    selected.sort(key=lambda segment: segment.order)
    if len(selected) > 1:
        orders = [segment.order for segment in selected]
        if max(orders) - min(orders) + 1 > len(orders) + 1:
            return None, p90
    return selected, p90


def _source_index(segments: list[SourceSegment]) -> SourceIndex:
    if not segments:
        return SourceIndex([], cKDTree(np.zeros((1, 2))), np.zeros(1, dtype=np.int32))
    clouds = [segment.samples for segment in segments]
    owners = np.concatenate([
        np.full(len(points), index, dtype=np.int32)
        for index, points in enumerate(clouds)
    ])
    return SourceIndex(segments, cKDTree(np.vstack(clouds)), owners)


def classify_edge(
    edge: dict,
    source: SourceIndex | list[SourceSegment],
    cfg: PrepareConfig,
) -> tuple[str | None, dict]:
    """Return a semantic class and fit metadata for one extracted edge."""
    raw = _edge_points(edge)
    if len(raw) < 2 or _polyline_length(raw) < cfg.min_edge_px:
        return None, {"reason": "short"}
    selected, match_p90 = _source_match(raw, source, cfg)
    if selected is None:
        return None, {"reason": "unmatched", "match_p90": match_p90}

    commands = {segment.command for segment in selected}
    line_p90 = _line_residual(raw)
    fit = {"match_p90": match_p90, "line_p90": line_p90}

    if commands == {"L"}:
        command = "LINE" if line_p90 <= cfg.line_p90_px else "POLYLINE"
        return command, fit
    if commands != {"C"}:
        return None, {**fit, "reason": "mixed_source_commands"}
    if line_p90 <= cfg.line_p90_px:
        return "LINE", fit

    try:
        center, radius, radial_p90 = _circle_fit(raw)
        fit.update({"circle_center": center, "circle_radius": radius,
                    "circle_p90": radial_p90})
        relative = radial_p90 / max(radius, 1e-6)
        if radial_p90 <= cfg.circle_p90_px and relative <= cfg.circle_relative_residual:
            if edge.get("is_closed", False):
                return "CIRCLE", fit
            chord = float(np.linalg.norm(raw[-1] - raw[0]))
            if chord > 1e-6:
                normal = np.array([-(raw[-1] - raw[0])[1], raw[-1, 0] - raw[0, 0]]) / chord
                sagitta = float(np.abs((raw - raw[0]) @ normal).max())
                if sagitta / chord >= cfg.min_arc_sagitta_ratio:
                    return "ARC", fit
    except (ValueError, np.linalg.LinAlgError):
        pass
    return "BEZIER", fit


def _sample_from_edge(
    edge: dict, command: str, fit: dict, cfg: PrepareConfig, source_index: int,
) -> dict:
    raw = _edge_points(edge)
    input_points = raw if edge.get("is_closed") else np.asarray(
        edge.get("smooth_pts") or edge["pixels"], dtype=np.float64
    )
    points, mask, normal_center, normal_scale = _normalise_edge(input_points, cfg.max_pts)
    params = np.zeros(6, dtype=np.float32)
    if command == "LINE":
        params[:2] = (raw[0] - normal_center) / normal_scale + 0.5
        params[2:4] = (raw[-1] - normal_center) / normal_scale + 0.5
    elif command in {"ARC", "CIRCLE"}:
        center = np.asarray(fit["circle_center"])
        params[:2] = (center - normal_center) / normal_scale + 0.5
        params[2] = float(fit["circle_radius"]) / normal_scale
        if command == "ARC":
            angles = np.degrees(np.arctan2(
                raw[[0, -1], 1] - center[1], raw[[0, -1], 0] - center[0]
            )) % 360.0
            params[3:5] = angles / 360.0
    return {
        "points": points,
        "mask": mask,
        "type": TYPE_TO_ID[command],
        "params": params,
        "source_index": source_index,
    }


_WORKER_CFG: PrepareConfig | None = None
_WORKER_D2C: Path | None = None


def _worker_init(config: dict, d2c_root: str) -> None:
    global _WORKER_CFG, _WORKER_D2C
    _WORKER_CFG = PrepareConfig(**config)
    _WORKER_D2C = Path(d2c_root)


def _process_one(job: tuple[int, str]) -> dict:
    source_index, npz_name = job
    cfg = _WORKER_CFG
    assert cfg is not None and _WORKER_D2C is not None
    result = {"source_index": source_index, "samples": [], "status": "ok"}
    try:
        with np.load(npz_name, allow_pickle=False) as data:
            skeleton = np.asarray(data["skeleton"])
            keypoints = np.asarray(data["kps"])
            meta = json.loads(str(data["meta"]))
        sample_id = meta["sample_id"]
        view = meta["view"]
        group, number = sample_id.split("/")
        svg_path = _WORKER_D2C / "svg_raw" / group / number / f"{number}_{view}.svg"
        segments = parse_svg_segments(svg_path, skeleton.shape[1])
        source_index_geometry = _source_index(segments)

        candidates = [
            {"x": int(x), "y": int(y), "type": ID_TO_KP[int(kind)], "confidence": 1.0}
            for x, y, kind in keypoints if int(kind) in ID_TO_KP
        ]
        clusters = s2._clusters_from_points(candidates, skeleton, snap_radius=4)
        nodes, edges = s2._extract_topology(skeleton, clusters)
        nodes, edges = s2._simplify_graph(
            nodes, edges, spur_min_len=6.0,
            collinear_max_angle=28.0, junction_merge_radius=0.0,
        )
        edges = s2._smooth_edges(edges)

        counts = Counter()
        skipped = Counter()
        samples = []
        for edge in edges:
            command, fit = classify_edge(edge, source_index_geometry, cfg)
            if command is None:
                skipped[fit.get("reason", "rejected")] += 1
                continue
            try:
                sample = _sample_from_edge(edge, command, fit, cfg, source_index)
            except ValueError:
                skipped["degenerate"] += 1
                continue
            samples.append(sample)
            counts[command] += 1
        result.update({
            "samples": samples, "counts": dict(counts), "skipped": dict(skipped),
            "n_edges": len(edges), "sample_id": sample_id, "view": view,
        })
    except Exception as exc:
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return result


def _atomic_npz(path: Path, samples: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            points=np.stack([sample["points"] for sample in samples]),
            mask=np.stack([sample["mask"] for sample in samples]),
            types=np.asarray([sample["type"] for sample in samples], dtype=np.uint8),
            params=np.stack([sample["params"] for sample in samples]),
            source_index=np.asarray(
                [sample["source_index"] for sample in samples], dtype=np.int64
            ),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _views(spec: str) -> set[str]:
    return set(ALL_VIEWS if spec == "all" else spec.split(","))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Drawing2CAD per-edge Stage 3 supervision",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--views", default="all")
    parser.add_argument("--labels", type=Path, default=PROJECT_ROOT / "output/Drawing2CAD/kp_labels")
    parser.add_argument("--d2c-root", type=Path, default=PROJECT_ROOT / "data/Drawing2CAD")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output/Drawing2CAD/stage3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-chunk-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--max-pts", type=int, default=64)
    parser.add_argument("--match-p90-px", type=float, default=2.5)
    args = parser.parse_args()

    source_dir = args.labels / args.split
    selected_views = _views(args.views)
    paths = []
    for path in sorted(source_dir.rglob("*.npz")):
        if any(path.stem.endswith(f"_{view}") for view in selected_views):
            paths.append(path)
    if args.limit and args.limit < len(paths):
        paths = sorted(random.Random(args.seed).sample(paths, args.limit))
    if not paths:
        raise SystemExit(f"no cached labels found in {source_dir}")

    output_split = "val" if args.split == "validation" else args.split
    output_dir = args.output / output_split
    marker_dir = output_dir / "markers"
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)
    cfg = PrepareConfig(max_pts=args.max_pts, match_p90_px=args.match_p90_px)

    totals = Counter()
    skipped = Counter()
    errors = Counter()
    total_edges = total_sources = 0
    jobs = list(enumerate(map(str, paths)))
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(cfg.__dict__, str(args.d2c_root)),
    ) as executor:
        for chunk_index, start in enumerate(range(0, len(jobs), args.source_chunk_size)):
            marker_path = marker_dir / f"chunk_{chunk_index:06d}.json"
            if marker_path.exists():
                marker = json.loads(marker_path.read_text())
            else:
                chunk = jobs[start:start + args.source_chunk_size]
                results = list(tqdm(
                    executor.map(_process_one, chunk, chunksize=1), total=len(chunk),
                    desc=f"{args.split} chunk {chunk_index + 1}", leave=False,
                ))
                samples = []
                chunk_counts = Counter()
                chunk_skipped = Counter()
                chunk_errors = Counter()
                chunk_edges = 0
                for result in results:
                    if result["status"] != "ok":
                        chunk_errors[result.get("error", "unknown")] += 1
                        continue
                    samples.extend(result["samples"])
                    chunk_counts.update(result.get("counts", {}))
                    chunk_skipped.update(result.get("skipped", {}))
                    chunk_edges += int(result.get("n_edges", 0))
                shard_names = []
                for part, offset in enumerate(range(0, len(samples), args.shard_size)):
                    shard = output_dir / f"shard_{chunk_index:06d}_{part:03d}.npz"
                    _atomic_npz(shard, samples[offset:offset + args.shard_size])
                    shard_names.append(shard.name)
                marker = {
                    "chunk": chunk_index, "sources": len(chunk), "edges": chunk_edges,
                    "samples": len(samples), "classes": dict(chunk_counts),
                    "skipped": dict(chunk_skipped), "errors": dict(chunk_errors),
                    "shards": shard_names,
                }
                _atomic_json(marker_path, marker)

            total_sources += int(marker["sources"])
            total_edges += int(marker["edges"])
            totals.update(marker.get("classes", {}))
            skipped.update(marker.get("skipped", {}))
            errors.update(marker.get("errors", {}))
            print(
                f"chunk {chunk_index + 1}/{math.ceil(len(jobs) / args.source_chunk_size)} "
                f"sources={total_sources:,} edges={total_edges:,} labels={dict(totals)}",
                flush=True,
            )

    report = {
        "source": "Drawing2CAD", "split": args.split, "output_split": output_split,
        "views": sorted(selected_views), "sources": total_sources, "edges": total_edges,
        "stage3_samples": int(sum(totals.values())), "stage3_classes": dict(totals),
        "skipped": dict(skipped), "errors": dict(errors), "config": cfg.__dict__,
    }
    _atomic_json(output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
