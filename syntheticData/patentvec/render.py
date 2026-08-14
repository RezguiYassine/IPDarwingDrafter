from __future__ import annotations

import html
import io
import math
from dataclasses import replace

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import sample_primitive
from .schema import CanonicalDrawing, Primitive


DEFAULT_STROKE_WIDTH = {
    "object_visible": 0.00165,
    "object_hidden": 0.00135,
    "object_center": 0.00105,
    "object_section_boundary": 0.0018,
    "construction": 0.0009,
    "hatch": 0.00072,
    "leader": 0.00105,
    "dimension": 0.0010,
    "arrowhead": 0.00105,
    "text_box": 0.0012,
    "diagram_connector": 0.0011,
}

MASK_GROUPS = {
    "object": {"object_visible", "object_section_boundary"},
    "hidden_center": {"object_hidden", "object_center", "construction"},
    "hatch": {"hatch"},
    "leader_dimension": {"leader", "dimension", "arrowhead", "diagram_connector"},
    "text_numeral": {"reference_numeral", "figure_label", "text"},
    "text_box": {"text_box"},
}


def _fmt(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".")


def _point(values) -> str:
    return " ".join(_fmt(value) for value in values)


def _arc_path(primitive: Primitive) -> str:
    geometry = primitive.geometry
    center = np.asarray(geometry["center"], dtype=float)
    radius = float(geometry["radius"])
    start = float(geometry["start_angle"])
    end = float(geometry["end_angle"])
    ccw = bool(geometry.get("ccw", True))
    if ccw:
        while end <= start:
            end += 2.0 * math.pi
    else:
        while end >= start:
            end -= 2.0 * math.pi
    delta = end - start
    p0 = center + radius * np.array([math.cos(start), math.sin(start)])
    p1 = center + radius * np.array([math.cos(end), math.sin(end)])
    large_arc = int(abs(delta) > math.pi)
    sweep = int(delta > 0.0)
    return (
        f"M {_point(p0)} A {_fmt(radius)} {_fmt(radius)} 0 "
        f"{large_arc} {sweep} {_point(p1)}"
    )


def _primitive_element(
    primitive: Primitive,
    stroke: str = "#111111",
    force_width: float | None = None,
) -> str:
    geometry = primitive.geometry
    style = primitive.style
    width = float(
        force_width
        if force_width is not None
        else style.get("stroke_width", DEFAULT_STROKE_WIDTH.get(primitive.semantic, 0.0015))
    )
    attributes = [
        f'stroke="{stroke}"',
        f'stroke-width="{_fmt(width)}"',
        'stroke-linecap="round"',
        'stroke-linejoin="round"',
        'fill="none"',
        f'data-primitive-id="{html.escape(primitive.primitive_id)}"',
        f'data-semantic="{html.escape(primitive.semantic)}"',
    ]
    dasharray = style.get("dasharray")
    if dasharray:
        attributes.append(
            'stroke-dasharray="' + " ".join(_fmt(value) for value in dasharray) + '"'
        )
    attrs = " ".join(attributes)
    kind = primitive.kind
    if kind == "line":
        return (
            f'<line x1="{_fmt(geometry["p0"][0])}" y1="{_fmt(geometry["p0"][1])}" '
            f'x2="{_fmt(geometry["p1"][0])}" y2="{_fmt(geometry["p1"][1])}" {attrs}/>'
        )
    if kind in {"polyline", "spline"}:
        tag = "polygon" if geometry.get("closed", False) else "polyline"
        points = " ".join(_point(point) for point in geometry["points"])
        return f'<{tag} points="{points}" {attrs}/>'
    if kind == "circle":
        return (
            f'<circle cx="{_fmt(geometry["center"][0])}" '
            f'cy="{_fmt(geometry["center"][1])}" r="{_fmt(geometry["radius"])}" {attrs}/>'
        )
    if kind == "arc":
        return f'<path d="{_arc_path(primitive)}" {attrs}/>'
    if kind == "quadratic_bezier":
        path = (
            f'M {_point(geometry["p0"])} Q {_point(geometry["p1"])} '
            f'{_point(geometry["p2"])}'
        )
        return f'<path d="{path}" {attrs}/>'
    if kind == "cubic_bezier":
        path = (
            f'M {_point(geometry["p0"])} C {_point(geometry["p1"])} '
            f'{_point(geometry["p2"])} {_point(geometry["p3"])}'
        )
        return f'<path d="{path}" {attrs}/>'
    if kind == "text":
        position = geometry["position"]
        size = float(geometry.get("size", 0.025))
        rotation = float(geometry.get("rotation", 0.0))
        anchor = geometry.get("anchor", "start")
        font_family = html.escape(style.get("font_family", "DejaVu Sans"))
        content = html.escape(str(geometry.get("text", "")))
        return (
            f'<text x="{_fmt(position[0])}" y="{_fmt(position[1])}" '
            f'font-family="{font_family}" font-size="{_fmt(size)}" '
            f'text-anchor="{anchor}" fill="{stroke}" stroke="none" '
            f'transform="rotate({_fmt(rotation)} {_point(position)})" '
            f'data-primitive-id="{html.escape(primitive.primitive_id)}" '
            f'data-semantic="{html.escape(primitive.semantic)}">{content}</text>'
        )
    # Exact source geometry remains in JSON. Unsupported display primitives are
    # previewed as a dense polyline without changing their training label.
    sampled = sample_primitive(primitive, 192)
    points = " ".join(_point(point) for point in sampled)
    return f'<polyline points="{points}" {attrs}/>'


def svg_document(
    drawing: CanonicalDrawing,
    primitives: list[Primitive] | None = None,
    background: str = "#ffffff",
    stroke: str = "#111111",
    semantic_colors: dict[str, str] | None = None,
    force_width: float | None = None,
) -> str:
    primitives = drawing.primitives_visible if primitives is None else primitives
    elements = []
    for primitive in sorted(primitives, key=lambda item: item.z_order):
        color = semantic_colors.get(primitive.semantic, stroke) if semantic_colors else stroke
        elements.append(_primitive_element(primitive, color, force_width=force_width))
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{drawing.canvas[0]}" '
                f'height="{drawing.canvas[1]}" viewBox="0 0 1 1">'
            ),
            f'<rect width="1" height="1" fill="{background}"/>',
            *elements,
            "</svg>",
        ]
    )


def rasterize_svg(svg: str, width: int, height: int) -> np.ndarray:
    payload = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=width, output_height=height
    )
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("L"))


def render_clean(drawing: CanonicalDrawing) -> np.ndarray:
    return rasterize_svg(
        svg_document(drawing), width=drawing.canvas[0], height=drawing.canvas[1]
    )


def render_masks(drawing: CanonicalDrawing) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, semantics in MASK_GROUPS.items():
        selected = [
            primitive for primitive in drawing.primitives_visible
            if primitive.semantic in semantics
        ]
        if not selected:
            masks[name] = np.zeros((drawing.canvas[1], drawing.canvas[0]), dtype=np.uint8)
            continue
        svg = svg_document(
            drawing,
            primitives=selected,
            background="#000000",
            stroke="#ffffff",
        )
        antialiased = rasterize_svg(
            svg, width=drawing.canvas[0], height=drawing.canvas[1]
        )
        masks[name] = (antialiased > 0).astype(np.uint8) * 255
    return masks


def degrade_patent_scan(clean: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = clean.shape
    image = clean.astype(np.float32) / 255.0
    sigma = float(rng.uniform(0.35, 0.75))
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    y_grid, x_grid = np.mgrid[0:height, 0:width]
    illumination = (
        0.035 * np.sin(x_grid / width * math.pi * rng.uniform(0.7, 1.4))
        + 0.025 * np.cos(y_grid / height * math.pi * rng.uniform(0.8, 1.6))
    )
    image = image + illumination + rng.normal(0.0, 0.009, image.shape)
    image = np.clip(image, 0.0, 1.0)

    # Mild local fading and sparse scan dirt preserve geometry while exposing
    # the raster model to patent-like acquisition noise.
    for _ in range(int(rng.integers(2, 5))):
        cx = int(rng.integers(width))
        cy = int(rng.integers(height))
        radius = float(rng.uniform(0.04, 0.12) * min(width, height))
        distance = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        fade = np.exp(-distance / max(2.0 * radius * radius, 1.0))
        image += fade * float(rng.uniform(0.015, 0.045))
    dark_speckles = rng.random(image.shape) < 0.00012
    light_speckles = rng.random(image.shape) < 0.00008
    image[dark_speckles] = rng.uniform(0.0, 0.45, dark_speckles.sum())
    image[light_speckles] = 1.0
    return np.uint8(np.clip(image * 255.0, 0, 255))


def semantic_preview(drawing: CanonicalDrawing) -> np.ndarray:
    colors = {
        "object_visible": "#111111",
        "object_hidden": "#7c3aed",
        "object_center": "#2563eb",
        "hatch": "#d97706",
        "leader": "#059669",
        "arrowhead": "#059669",
        "dimension": "#0891b2",
        "reference_numeral": "#dc2626",
        "text": "#dc2626",
        "text_box": "#9333ea",
    }
    svg = svg_document(drawing, semantic_colors=colors)
    rgb_payload = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=drawing.canvas[0],
        output_height=drawing.canvas[1],
    )
    with Image.open(io.BytesIO(rgb_payload)) as image:
        output = np.asarray(image.convert("RGB")).copy()
    for junction in drawing.junctions_visible:
        x = int(round(junction.position[0] * (drawing.canvas[0] - 1)))
        y = int(round(junction.position[1] * (drawing.canvas[1] - 1)))
        cv2.circle(output, (x, y), 5, (220, 38, 38), 2, lineType=cv2.LINE_AA)
    return output


def make_triptych(
    clean: np.ndarray,
    degraded: np.ndarray,
    semantic: np.ndarray,
    sample_id: str,
) -> Image.Image:
    clean_rgb = Image.fromarray(clean).convert("RGB")
    degraded_rgb = Image.fromarray(degraded).convert("RGB")
    semantic_rgb = Image.fromarray(semantic).convert("RGB")
    width, height = clean_rgb.size
    header = 38
    preview = Image.new("RGB", (3 * width, height + header), "white")
    preview.paste(clean_rgb, (0, header))
    preview.paste(degraded_rgb, (width, header))
    preview.paste(semantic_rgb, (2 * width, header))
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    labels = [f"{sample_id} | clean", "patent degraded", "semantic + junctions"]
    for index, label in enumerate(labels):
        draw.text((index * width + 12, 12), label, fill="#111111", font=font)
    return preview
