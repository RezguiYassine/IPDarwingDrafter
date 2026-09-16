"""Font-aware, rotation-aware bounds for generated annotation text."""

from functools import lru_cache
import math

import numpy as np
from PIL import ImageFont
from shapely.geometry import LineString, box

from .geometry import sample_primitive


@lru_cache(maxsize=128)
def _font(size):
    return ImageFont.truetype("DejaVuSans.ttf", size)


def text_bounds(geometry):
    scale = 10000
    font = _font(max(1, round(float(geometry["size"]) * scale)))
    text = str(geometry["text"])
    left, top, right, bottom = np.asarray(font.getbbox(text, anchor="ls"), dtype=float) / scale
    width = font.getlength(text) / scale
    offset = {"start": 0.0, "middle": width / 2, "end": width}[geometry.get("anchor", "start")]
    points = np.array([[left-offset, top], [right-offset, top], [right-offset, bottom], [left-offset, bottom]])
    angle = math.radians(geometry.get("rotation", 0.0))
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = points @ rotation.T + geometry["position"]
    # Cover subpixel rasterization and font hinting differences. The margin is
    # proportional to the text size, not a fixed value in normalized units: a
    # figure is placed in its own frame and then composed into a sheet at 0.45
    # scale, and a fixed margin does not shrink with the geometry. That made
    # boxes up to 14% taller relative to content at sheet scale (entirely the
    # margin -- the underlying font metrics scale linearly to within 0.4%), so
    # annotations that were legal in the figure frame failed the sheet gate.
    # 0.028 reproduces the previous 0.0005 margin at the common 0.018 size.
    margin = float(geometry["size"]) * .028
    return [*(points.min(axis=0) - margin), *(points.max(axis=0) + margin)]


def annotation_failures(drawing):
    labels = [(p, box(*text_bounds(p.geometry))) for p in drawing.primitives_visible if p.kind == "text"]
    failures = []
    strokes = annotation_strokes(drawing)
    for i, (primitive, bounds) in enumerate(labels):
        if not box(.015, .015, .985, .985).covers(bounds):
            failures.append(f"annotation outside page: {primitive.primitive_id}")
        for other, other_bounds in labels[:i]:
            if bounds.intersects(other_bounds):
                failures.append(f"overlapping annotation text: {primitive.primitive_id}/{other.primitive_id}")
        for stroke, curve in strokes:
            if curve.intersects(bounds):
                failures.append(f"annotation line crosses text: {stroke.primitive_id}/{primitive.primitive_id}")
    return failures


def annotation_strokes(drawing):
    return [(p, LineString(sample_primitive(p, 64))) for p in drawing.primitives_visible
            if p.kind != "text" and p.semantic in {"leader", "dimension", "arrowhead", "diagram_connector"}]
