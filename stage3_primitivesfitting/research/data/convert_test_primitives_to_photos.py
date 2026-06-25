"""Render Stage 3 research test stroke JSON to PNG photos.

This script reads all JSON files in the local test directory and writes raster
preview images. The files in `free2cad_training_v3/test` contain normalized
strokes with a `stroke` array and `command` metadata, so the script scales them
into pixel space for rendering.

Usage:
    python convert_test_primitives_to_photos.py
    python convert_test_primitives_to_photos.py --input-dir free2cad_training_v3/test \
        --output-dir free2cad_training_v3/test_photos --resolution 1024
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_INPUT_DIR = Path("free2cad_training_v3/test")
DEFAULT_OUTPUT_DIR = Path("free2cad_training_v3/test_photos")
DEFAULT_RESOLUTION = 1024
DEFAULT_MARGIN = 32
DEFAULT_STROKE_WIDTH = 6


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_normalized_value(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return 0.0 <= value <= 1.0
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0 for x in value)
    return False


def _normalize_point(point: list[float], width: int, height: int, margin: int) -> tuple[float, float]:
    return (
        margin + point[0] * (width - 2 * margin),
        margin + point[1] * (height - 2 * margin),
    )


def _normalize_radius(radius: float, width: int, height: int, margin: int) -> float:
    return radius * min(width - 2 * margin, height - 2 * margin)


def _sample_arc_points(center: tuple[float, float], radius: float,
                       start_angle: float, end_angle: float,
                       segments: int = 80) -> list[tuple[float, float]]:
    cx, cy = center
    start = math.radians(start_angle)
    end = math.radians(end_angle)
    delta = (end - start) % (2 * math.pi)
    if delta == 0 and start_angle != end_angle:
        delta = 2 * math.pi
    return [
        (
            cx + radius * math.cos(start + delta * t / segments),
            cy + radius * math.sin(start + delta * t / segments),
        )
        for t in range(segments + 1)
    ]


def _sample_ellipse_points(center: tuple[float, float], a: float, b: float,
                           angle: float, segments: int = 100) -> list[tuple[float, float]]:
    cx, cy = center
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    points = []
    for i in range(segments + 1):
        phi = 2 * math.pi * i / segments
        x = a * math.cos(phi)
        y = b * math.sin(phi)
        points.append((cx + x * cos_t - y * sin_t, cy + x * sin_t + y * cos_t))
    return points


def _resolve_value(value: Any, width: int, height: int, margin: int, normalized: bool) -> Any:
    if normalized:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return _normalize_point([float(value[0]), float(value[1])], width, height, margin)
        if isinstance(value, (int, float)):
            return _normalize_radius(float(value), width, height, margin)
    return value


def _render_data(data: dict, width: int, height: int,
                 margin: int, stroke_width: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    normalized = "image_size" not in data

    if not normalized:
        supplied_size = data.get("image_size")
        if isinstance(supplied_size, (list, tuple)) and len(supplied_size) == 2:
            width, height = int(supplied_size[0]), int(supplied_size[1])
            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)

    def resolve_point(point: list[float]) -> tuple[float, float]:
        if normalized:
            return _normalize_point([float(point[0]), float(point[1])], width, height, margin)
        return (float(point[0]), float(point[1]))

    if "stroke" in data or "strokes" in data:
        strokes = []
        if "stroke" in data and isinstance(data["stroke"], list):
            strokes.append(data["stroke"])
        if "strokes" in data and isinstance(data["strokes"], list):
            for s in data["strokes"]:
                if isinstance(s, list):
                    strokes.append(s)

        for stroke in strokes:
            points = [resolve_point(p) for p in stroke if isinstance(p, (list, tuple)) and len(p) == 2]
            if len(points) < 2:
                continue
            draw.line(points, fill="black", width=stroke_width)
        return image

    # Fallback to primitive rendering for JSONs that contain primitives.
    for prim in data.get("primitives", []):
        ptype = prim.get("type")
        if not ptype:
            continue

        if ptype == "line":
            p1 = prim.get("p1") or prim.get("start")
            p2 = prim.get("p2") or prim.get("end")
            if not p1 or not p2:
                continue
            p1 = resolve_point(p1)
            p2 = resolve_point(p2)
            draw.line([p1, p2], fill="black", width=stroke_width)

        elif ptype == "circle":
            center = prim.get("center")
            radius = prim.get("radius")
            if center is None or radius is None:
                continue
            center = resolve_point(center)
            radius = float(radius) if not normalized else _normalize_radius(radius, width, height, margin)
            cx, cy = center
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline="black", width=stroke_width)

        elif ptype == "arc":
            center = prim.get("center")
            radius = prim.get("radius")
            start_angle = prim.get("start_angle")
            end_angle = prim.get("end_angle")
            if center is None or radius is None or start_angle is None or end_angle is None:
                continue
            center = resolve_point(center)
            radius = float(radius) if not normalized else _normalize_radius(radius, width, height, margin)
            points = _sample_arc_points(center, radius, float(start_angle), float(end_angle))
            draw.line(points, fill="black", width=stroke_width)

        elif ptype in {"polyline", "polygon"}:
            points = prim.get("points") or prim.get("vertices") or []
            if len(points) < 2:
                continue
            resolved = [resolve_point(p) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
            if ptype == "polygon":
                resolved.append(resolved[0])
            draw.line(resolved, fill="black", width=stroke_width)

        elif ptype == "ellipse":
            center = prim.get("center")
            a = prim.get("a")
            b = prim.get("b")
            if center is None or a is None or b is None:
                continue
            center = resolve_point(center)
            a = float(a) if not normalized else _normalize_radius(a, width, height, margin)
            b = float(b) if not normalized else _normalize_radius(b, width, height, margin)
            angle = float(prim.get("angle", 0.0))
            points = _sample_ellipse_points(center, a, b, angle)
            draw.line(points, fill="black", width=stroke_width)

    return image


def _render_file(input_path: Path, output_path: Path,
                 resolution: int, margin: int, stroke_width: int) -> None:
    data = _load_json(input_path)
    image = _render_data(data, resolution, resolution, margin, stroke_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Stage 3 research test primitives JSON to PNG photos."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing test primitive JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where rendered PNG files will be saved.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Output resolution for normalized primitive files.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help="Pixel margin to leave around normalized primitives.",
    )
    parser.add_argument(
        "--stroke-width",
        type=int,
        default=DEFAULT_STROKE_WIDTH,
        help="Stroke width in pixels for rendered primitives.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {args.input_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        output_path = args.output_dir / f"{file_path.stem}.png"
        try:
            _render_file(file_path, output_path, args.resolution, args.margin, args.stroke_width)
            print(f"Rendered {file_path.name} -> {output_path.name}")
        except Exception as exc:
            print(f"Failed {file_path.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
