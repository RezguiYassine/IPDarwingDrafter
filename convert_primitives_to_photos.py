"""Convert Stage 3 primitive JSON files into raster preview images.

This script reads all primitive JSONs in a folder and renders each sketch as a
PNG image. It supports both pixel-space primitives (with image_size) and
normalized primitives (coords/radii in [0, 1]).

Usage examples:
    python convert_primitives_to_photos.py
    python convert_primitives_to_photos.py --input-dir my_results/primitives \
        --output-dir my_results/primitives_photos --resolution 1024
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_INPUT_DIR = Path("my_results/primitives")
DEFAULT_OUTPUT_DIR = Path("my_results/primitives_photos")
DEFAULT_RESOLUTION = 1024
DEFAULT_MARGIN = 32
DEFAULT_STROKE_WIDTH = 8


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_normalized_value(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return bool(value) and all(isinstance(v, (int, float)) and abs(v) <= 1.0 for v in value)
    if isinstance(value, (int, float)):
        return abs(value) <= 1.0
    return False


def _normalize_coord(value: float, length: int, margin: int) -> float:
    return margin + value * (length - 2 * margin)


def _normalize_point(point: list[float], width: int, height: int, margin: int) -> tuple[float, float]:
    return (
        _normalize_coord(point[0], width, margin),
        _normalize_coord(point[1], height, margin),
    )


def _normalize_radius(radius: float, width: int, height: int, margin: int) -> float:
    return radius * min(width - 2 * margin, height - 2 * margin)


def _resolve_primitives(data: dict, width: int, height: int, margin: int) -> dict:
    normalized = "image_size" not in data
    if normalized:
        canvas_size = (width, height)
    else:
        image_size = data["image_size"]
        if not (isinstance(image_size, (list, tuple)) and len(image_size) == 2):
            raise ValueError("image_size must be [width, height]")
        canvas_size = (int(image_size[0]), int(image_size[1]))

    resolved = {
        "sketch_id": data.get("sketch_id", "unknown"),
        "image_size": canvas_size,
        "primitives": [],
        "annotations": data.get("annotations", []),
    }

    for prim in data.get("primitives", []):
        prim_type = prim.get("type")
        if prim_type is None:
            continue

        item = {"type": prim_type, "style": prim.get("style", "visible")}

        if prim_type == "line":
            p1 = prim.get("p1") or prim.get("start")
            p2 = prim.get("p2") or prim.get("end")
            if p1 is None or p2 is None:
                continue
            if normalized:
                item["p1"] = _normalize_point(p1, width, height, margin)
                item["p2"] = _normalize_point(p2, width, height, margin)
            else:
                item["p1"] = tuple(p1)
                item["p2"] = tuple(p2)

        elif prim_type == "circle":
            center = prim.get("center")
            radius = prim.get("radius")
            if center is None or radius is None:
                continue
            if normalized:
                item["center"] = _normalize_point(center, width, height, margin)
                item["radius"] = _normalize_radius(radius, width, height, margin)
            else:
                item["center"] = tuple(center)
                item["radius"] = float(radius)

        elif prim_type == "arc":
            center = prim.get("center")
            radius = prim.get("radius")
            start_angle = prim.get("start_angle")
            end_angle = prim.get("end_angle")
            if center is None or radius is None or start_angle is None or end_angle is None:
                continue
            if normalized:
                item["center"] = _normalize_point(center, width, height, margin)
                item["radius"] = _normalize_radius(radius, width, height, margin)
            else:
                item["center"] = tuple(center)
                item["radius"] = float(radius)
            item["start_angle"] = float(start_angle)
            item["end_angle"] = float(end_angle)

        elif prim_type in {"polyline", "polygon"}:
            points = prim.get("points") or prim.get("vertices") or []
            if len(points) < 2:
                continue
            mapped = [
                _normalize_point(p, width, height, margin) if normalized else tuple(p)
                for p in points
                if isinstance(p, (list, tuple)) and len(p) == 2
            ]
            if len(mapped) < 2:
                continue
            item["points"] = mapped

        elif prim_type == "ellipse":
            center = prim.get("center")
            a = prim.get("a")
            b = prim.get("b")
            angle = prim.get("angle", 0.0)
            if center is None or a is None or b is None:
                continue
            if normalized:
                item["center"] = _normalize_point(center, width, height, margin)
                item["a"] = a * (width - 2 * margin)
                item["b"] = b * (height - 2 * margin)
            else:
                item["center"] = tuple(center)
                item["a"] = float(a)
                item["b"] = float(b)
            item["angle"] = float(angle)

        else:
            continue

        resolved["primitives"].append(item)

    return resolved


def _sample_arc_points(center: tuple[float, float], radius: float,
                       start_angle: float, end_angle: float,
                       segments: int = 64) -> list[tuple[float, float]]:
    cx, cy = center
    start = math.radians(start_angle)
    end = math.radians(end_angle)
    delta = ((end - start) % (2 * math.pi))
    if delta == 0 and start_angle != end_angle:
        delta = 2 * math.pi
    points = []
    for i in range(segments + 1):
        theta = start + delta * (i / segments)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        points.append((x, y))
    return points


def _sample_ellipse_points(center: tuple[float, float], a: float, b: float,
                           angle: float, segments: int = 96) -> list[tuple[float, float]]:
    cx, cy = center
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    points = []
    for i in range(segments + 1):
        t = 2 * math.pi * i / segments
        x = a * math.cos(t)
        y = b * math.sin(t)
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        points.append((cx + rx, cy + ry))
    return points


def _draw_primitives(data: dict, resolution: int, stroke_width: int) -> Image.Image:
    width, height = data["image_size"]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for prim in data["primitives"]:
        ptype = prim["type"]
        if ptype == "line":
            draw.line([prim["p1"], prim["p2"]], fill="black", width=stroke_width)

        elif ptype == "circle":
            cx, cy = prim["center"]
            r = prim["radius"]
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw.ellipse(bbox, outline="black", width=stroke_width)

        elif ptype == "arc":
            pts = _sample_arc_points(
                prim["center"], prim["radius"],
                prim["start_angle"], prim["end_angle"],
                segments=120,
            )
            draw.line(pts, fill="black", width=stroke_width)

        elif ptype in {"polyline", "polygon"}:
            pts = prim["points"]
            draw.line(pts, fill="black", width=stroke_width)
            if ptype == "polygon" and len(pts) > 2:
                draw.line([pts[-1], pts[0]], fill="black", width=stroke_width)

        elif ptype == "ellipse":
            pts = _sample_ellipse_points(
                prim["center"], prim["a"], prim["b"], prim.get("angle", 0.0),
                segments=120,
            )
            draw.line(pts, fill="black", width=stroke_width)

    return image


def _render_file(input_path: Path, output_path: Path,
                 resolution: int, margin: int, stroke_width: int) -> bool:
    data = _load_json(input_path)
    resolved = _resolve_primitives(data, resolution, resolution, margin)
    image = _draw_primitives(resolved, resolution, stroke_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert primitives JSON files into raster preview images."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing primitives JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG preview images will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help="Output image resolution (used when primitives are normalized).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help="Margin in pixels around the rendered primitives.",
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
    input_dir = args.input_dir
    output_dir = args.output_dir
    files = sorted(input_dir.glob("*_primitives.json"))
    if not files:
        print(f"No primitive JSON files found in: {input_dir}")
        return 1

    for path in files:
        sketch_id = path.stem.replace("_primitives", "")
        out_path = output_dir / f"{sketch_id}.png"
        try:
            _render_file(path, out_path, args.resolution, args.margin, args.stroke_width)
            print(f"Rendered {path.name} → {out_path.name}")
        except Exception as exc:
            print(f"Failed to render {path.name}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
