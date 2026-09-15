"""
stage4_export.py
================
AP3 Vectorization Pipeline — Stage 4: Vector Export

Transforms Stage 3's RANSAC-fitted primitives JSON into deliverable vector files.

Outputs (selected by user via --format):
  - <sketch_id>.svg                 : SVG (always available, view-friendly)
  - <sketch_id>.dxf                 : DXF, either basic or patent-ready
                                        (selected by --dxf-mode)

Two DXF compliance levels:
  basic   — single layer, AutoCAD-readable, no styling. For quick QA.
  patent  — ISO 128 layered (visible / hidden / center / construction /
            hachure / text),
            standard linetypes & lineweights, Bezugszeichen rendered as MTEXT
            with leader lines per EPO Rule 46. AP6 supplies the Bezugszeichen
            via the optional `annotations` block in the input JSON; if absent,
            patent mode still produces a valid layered DXF without numerals.

Input JSON schema (from Stage 3):
    {
      "sketch_id"  : "...",
      "image_size" : [W, H],
      "primitives" : [
        {"type": "line",   "p1": [x,y], "p2": [x,y], "confidence": 0..1,
         "style": "visible|hidden|center|construction"},          # style optional
        {"type": "circle", "center": [x,y], "radius": r, "confidence": 0..1,
         "style": "..."},
        {"type": "arc",    "center": [x,y], "radius": r,
         "start_angle": deg, "end_angle": deg, "confidence": 0..1,
         "style": "..."}
      ],
      "annotations": [                                             # optional, AP6
        {"id": "1", "text": "Welle",
         "position":  [x, y],          # text anchor, image space
         "leader_to": [x, y]}          # tip of leader arrow, image space
      ]
    }

Coordinate convention:
    Input  : image pixel space, Y-down (origin top-left)
    SVG    : same as input (SVG natively uses Y-down)
    DXF    : Y is flipped (origin bottom-left, Y-up — CAD convention).
             Arc angles are mirrored accordingly so visual orientation
             matches the source sketch.

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ezdxf
import svgwrite

if __package__:
    from .export_audit import (
        capture_entities, file_digest, finish_annotation, start_item,
        validate_primitive, verify_serialized, primitive_budget_cost, stroke_digest,
    )
else:
    from export_audit import (
        capture_entities, file_digest, finish_annotation, start_item,
        validate_primitive, verify_serialized, primitive_budget_cost, stroke_digest,
    )

logger = logging.getLogger(__name__)


# ─── Output contract ─────────────────────────────────────────────────────────

@dataclass
class Stage4Result:
    sketch_id: str
    svg_path:  Optional[Path] = None
    dxf_path:  Optional[Path] = None
    dxf_mode:  Optional[str]  = None        # "basic" | "patent" | None
    n_primitives_in:  int     = 0
    n_primitives_out: int     = 0
    n_annotations:    int     = 0
    processing_time_s: float  = 0.0
    flagged: bool             = False       # any primitive failed to export
    format_reports: dict      = field(default_factory=dict)
    export_report_path: Optional[Path] = None


# ─── ISO 128 layer specification (used by patent mode) ───────────────────────
#
#   Per ISO 128 / DIN 15 — line widths in mm, AutoCAD lineweight is 1/100 mm.
#   Colour 7 = ACI white/black (renders correctly on both backgrounds).
#   Style names map from primitive["style"] in the input JSON.

ISO128_LAYERS = {
    "visible":      {"linetype": "CONTINUOUS", "lineweight": 50, "color": 7},  # 0.50 mm
    "hidden":       {"linetype": "DASHED",     "lineweight": 25, "color": 7},  # 0.25 mm
    "center":       {"linetype": "CENTER",     "lineweight": 25, "color": 7},  # 0.25 mm
    "construction": {"linetype": "CONTINUOUS", "lineweight": 13, "color": 7},  # 0.13 mm
    "hachure":      {"linetype": "CONTINUOUS", "lineweight": 18, "color": 7},  # 0.18 mm
    "text":         {"linetype": "CONTINUOUS", "lineweight": 25, "color": 7},
    "leader":       {"linetype": "CONTINUOUS", "lineweight": 18, "color": 7},
}

DEFAULT_STYLE = "visible"   # primitives without an explicit style go here

# Stage 2/3 coordinates are integer raster indices.  A pixel at index (x, y)
# occupies continuous image space around (x + 0.5, y + 0.5); exporting the raw
# indices therefore moves re-rasterized SVG geometry one pixel up and left for
# common stroke widths.  Keep this conversion at the raster-to-vector boundary
# so all primitive fitters continue to work in their native pixel-index frame.
PIXEL_CENTER_OFFSET = 0.5


# ─── Coordinate helpers ──────────────────────────────────────────────────────

def _flip_y_point(p: list, image_h: float) -> tuple:
    """Raster-index image coordinates (Y-down) → CAD coordinates (Y-up)."""
    return (
        float(p[0]) + PIXEL_CENTER_OFFSET,
        float(image_h) - (float(p[1]) + PIXEL_CENTER_OFFSET),
    )


def _flip_y_arc_angles(start_deg: float, end_deg: float) -> tuple:
    """
    Mirroring across the X-axis flips the arc's CCW direction.
    ezdxf draws arcs CCW from start to end, so to keep the same visual arc
    after a Y-flip we mirror angles around 0° AND swap start/end.
    """
    new_start = (-end_deg)   % 360
    new_end   = (-start_deg) % 360
    return (new_start, new_end)


# ─── Input loading & validation ──────────────────────────────────────────────

def _load_input(json_path: Path) -> dict:
    """Load and minimally validate the Stage 3 JSON output."""
    with open(json_path) as f:
        data = json.load(f)

    for required in ("sketch_id", "image_size", "primitives"):
        if required not in data:
            raise ValueError(
                f"Input JSON missing required field '{required}': {json_path}"
            )

    if not (isinstance(data["image_size"], (list, tuple))
            and len(data["image_size"]) == 2):
        raise ValueError("image_size must be [width, height]")

    data.setdefault("annotations", [])
    return data


def _primitive_style(prim: dict) -> str:
    """Extract style with default fallback. Always returns a known key."""
    style = prim.get("style", DEFAULT_STYLE)
    return style if style in ISO128_LAYERS else DEFAULT_STYLE


# ─── SVG export ──────────────────────────────────────────────────────────────

def _svg_stroke_width(style: str) -> float:
    """Map ISO 128 lineweight (1/100 mm) → SVG stroke-width in user units."""
    return ISO128_LAYERS[style]["lineweight"] / 25.0   # tuned for visual parity


def _svg_dasharray(style: str) -> Optional[str]:
    if style == "hidden":
        return "8,4"
    if style == "center":
        return "12,3,2,3"
    return None


def _svg_data_uri(image_path: str) -> Optional[str]:
    """Return a data URI for a PNG/JPEG crop, or None if it cannot be read."""
    path = Path(image_path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _arc_endpoints(cx: float, cy: float, r: float,
                   start_deg: float, end_deg: float) -> tuple:
    """Return (start_xy, end_xy) on the circle at the given angles (image space)."""
    sx = cx + r * math.cos(math.radians(start_deg))
    sy = cy + r * math.sin(math.radians(start_deg))
    ex = cx + r * math.cos(math.radians(end_deg))
    ey = cy + r * math.sin(math.radians(end_deg))
    return (sx, sy), (ex, ey)


def _svg_bezier_d(points: list) -> str:
    """SVG `d` for a poly-cubic-Bézier control list P0,C1,C2,P1,C1',C2',P2,..."""
    if not points or len(points) < 4 or (len(points) - 1) % 3 != 0:
        return ""
    p0 = points[0]
    d = [f"M {float(p0[0]):.3f} {float(p0[1]):.3f}"]
    for i in range(1, len(points), 3):
        c1, c2, p = points[i], points[i + 1], points[i + 2]
        d.append(f"C {float(c1[0]):.3f} {float(c1[1]):.3f} "
                 f"{float(c2[0]):.3f} {float(c2[1]):.3f} "
                 f"{float(p[0]):.3f} {float(p[1]):.3f}")
    return " ".join(d)


def _seg_endpoints(seg: dict):
    """Return the two endpoints of a path segment (image space)."""
    t = seg.get("type")
    if t == "line":
        return [float(seg["p1"][0]), float(seg["p1"][1])], [float(seg["p2"][0]), float(seg["p2"][1])]
    if t == "bezier":
        pts = seg.get("points") or []
        return list(pts[0]), list(pts[-1])
    if t == "arc":
        cx, cy = seg["center"]
        (sx, sy), (ex, ey) = _arc_endpoints(cx, cy, float(seg["radius"]),
                                            seg["start_angle"], seg["end_angle"])
        return [sx, sy], [ex, ey]
    return None, None


def _svg_segment_body(seg: dict, reverse: bool) -> str:
    """SVG `d` body (no leading M) drawing a segment start→end, honouring reverse."""
    t = seg.get("type")
    if t == "line":
        p = seg["p1"] if reverse else seg["p2"]
        return f"L {float(p[0]):.3f} {float(p[1]):.3f}"
    if t == "bezier":
        pts = list(seg.get("points") or [])
        if reverse:
            pts = pts[::-1]
        out = []
        for i in range(1, len(pts), 3):
            c1, c2, p = pts[i], pts[i + 1], pts[i + 2]
            out.append(f"C {float(c1[0]):.3f} {float(c1[1]):.3f} "
                       f"{float(c2[0]):.3f} {float(c2[1]):.3f} "
                       f"{float(p[0]):.3f} {float(p[1]):.3f}")
        return " ".join(out)
    if t == "arc":
        cx, cy = seg["center"]
        r = float(seg["radius"])
        (sx, sy), (ex, ey) = _arc_endpoints(cx, cy, r, seg["start_angle"], seg["end_angle"])
        large_arc = 1 if ((seg["end_angle"] - seg["start_angle"]) % 360) > 180 else 0
        if reverse:
            return f"A {r:.3f} {r:.3f} 0 {large_arc} 0 {sx:.3f} {sy:.3f}"
        return f"A {r:.3f} {r:.3f} 0 {large_arc} 1 {ex:.3f} {ey:.3f}"
    return ""


def _orient_path_segments(segments: list) -> list[tuple[dict, list, list, bool]]:
    """Choose segment directions that minimize connector length globally.

    Arc fitting identifies the occupied angular interval but not the original
    pixel-chain direction. Treating the first arc's stored start angle as the
    path start can therefore reverse it and force a connector across the whole
    arc chord. A two-state dynamic program solves all forward/reverse choices
    together and remains deterministic on ties.
    """
    valid = []
    for segment in segments:
        a, b = _seg_endpoints(segment)
        if a is not None:
            valid.append((segment, a, b))
    if not valid:
        return []

    # State 0 traverses a -> b; state 1 traverses b -> a.
    costs = [0.0, 0.0]
    parents: list[list[int]] = []
    for index in range(1, len(valid)):
        _, prev_a, prev_b = valid[index - 1]
        _, cur_a, cur_b = valid[index]
        prev_ends = (prev_b, prev_a)
        cur_starts = (cur_a, cur_b)
        next_costs = [math.inf, math.inf]
        next_parents = [0, 0]
        for current_state in (0, 1):
            for previous_state in (0, 1):
                dx = prev_ends[previous_state][0] - cur_starts[current_state][0]
                dy = prev_ends[previous_state][1] - cur_starts[current_state][1]
                candidate = costs[previous_state] + math.hypot(dx, dy)
                if candidate < next_costs[current_state]:
                    next_costs[current_state] = candidate
                    next_parents[current_state] = previous_state
        costs = next_costs
        parents.append(next_parents)

    state = 0 if costs[0] <= costs[1] else 1
    states = [state]
    for choices in reversed(parents):
        state = choices[state]
        states.append(state)
    states.reverse()

    oriented = []
    for (segment, a, b), reverse in zip(valid, states):
        start, end = (b, a) if reverse else (a, b)
        oriented.append((segment, start, end, bool(reverse)))
    return oriented


def _svg_path_continuous(segments: list) -> str:
    """One continuous `d` across all segments — proper joins, no seam notches.

    Segments come from a contiguous corner-split chain. Each is oriented by
    nearest endpoint to the running pen position; tiny endpoint mismatches are
    bridged with an `L` so the stroke stays a single connected path.
    """
    d = []
    cur = None
    for seg, start, end, reverse in _orient_path_segments(segments):
        if cur is None:
            d.append(f"M {start[0]:.3f} {start[1]:.3f}")
        else:
            if (cur[0] - start[0]) ** 2 + (cur[1] - start[1]) ** 2 > 0.01:
                d.append(f"L {start[0]:.3f} {start[1]:.3f}")
        body = _svg_segment_body(seg, reverse)
        if body:
            d.append(body)
        cur = end
    return " ".join(d)


def _svg_add_hatch(dwg, target, prim: dict, sw: float, uid: int) -> None:
    """Render a hatch region: boundary outline + parallel line family(ies),
    clipped to the boundary. One/two angles ⇒ single/cross hatch."""
    boundary = prim.get("boundary") or []
    if len(boundary) < 3:
        return
    pts = [(float(p[0]), float(p[1])) for p in boundary]
    dstr = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    # thin boundary outline (the region edge is real geometry)
    target.add(dwg.path(d=dstr, fill="none", stroke="black", stroke_width=sw))
    angles = prim.get("angles") or []
    spacing = float(prim.get("spacing") or 0.0)
    if spacing < 1.0 or not angles:
        return
    cid = f"hclip{uid}"
    clip = dwg.defs.add(dwg.clipPath(id=cid))
    clip.add(dwg.path(d=dstr))
    xs = [x for x, _ in pts]; ys = [y for _, y in pts]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + spacing
    g = dwg.g(clip_path=f"url(#{cid})")
    for a in angles:
        th = math.radians(a)
        dx, dy = math.cos(th), math.sin(th)         # line direction
        px, py = -dy, dx                            # perpendicular (offset)
        n = int(diag / max(spacing, 1e-6)) + 2
        for k in range(-n, n + 1):
            ox, oy = cx + px * k * spacing, cy + py * k * spacing
            g.add(dwg.line(start=(ox - dx * diag, oy - dy * diag),
                           end=(ox + dx * diag, oy + dy * diag),
                           stroke="black", stroke_width=max(0.5, sw * 0.7)))
    target.add(g)


def _export_svg(data: dict, out_path: Path,
                default_sw: Optional[float] = None, audit: Optional[dict] = None) -> int:
    """
    Write SVG file. Returns the count of primitives successfully written.

    SVG keeps image coordinates Y-down. Geometry is translated by half a pixel
    because Stage 2/3 points are raster indices, while SVG uses continuous
    coordinates whose pixel centres lie at ``index + 0.5``.

    default_sw : measured stroke width in pixels from Stage 1. When provided
        it overrides the ISO 128 lineweight calculation so the output SVG
        matches the source sketch's ink thickness.
    """
    W, H = data["image_size"]
    dwg = svgwrite.Drawing(
        filename=str(out_path),
        size=(f"{W}px", f"{H}px"),
        viewBox=f"0 0 {W} {H}",
    )

    # Background (white) — keeps the SVG consistent regardless of viewer theme
    dwg.add(dwg.rect(insert=(0, 0), size=(W, H), fill="white"))

    # Apply the pixel-index → pixel-centre conversion once to every geometric
    # element, including annotations re-injected from Stage 0. The background
    # remains fixed to the viewBox.
    geometry = dwg.g(
        transform=f"translate({PIXEL_CENTER_OFFSET} {PIXEL_CENTER_OFFSET})"
    )
    dwg.add(geometry)

    n_written = 0

    for index, prim in enumerate(data["primitives"]):
        item = start_item(audit, "primitives", index)
        before = len(geometry.elements)
        try:
            validate_primitive(prim)
            style    = _primitive_style(prim)
            if default_sw is not None:
                sw = max(1.0, min(float(default_sw), 30.0))
            else:
                sw = _svg_stroke_width(style)
            dash     = _svg_dasharray(style)
            stroke_kw = {"stroke": "black", "stroke_width": sw, "fill": "none"}
            if dash:
                stroke_kw["stroke_dasharray"] = dash

            ptype = prim["type"]
            if ptype == "line":
                p1, p2 = prim["p1"], prim["p2"]
                geometry.add(dwg.line(start=p1, end=p2, **stroke_kw))

            elif ptype == "circle":
                cx, cy = prim["center"]
                geometry.add(
                    dwg.circle(center=(cx, cy), r=prim["radius"], **stroke_kw)
                )

            elif ptype == "arc":
                cx, cy = prim["center"]
                r      = prim["radius"]
                s, e   = prim["start_angle"], prim["end_angle"]
                (sx, sy), (ex, ey) = _arc_endpoints(cx, cy, r, s, e)
                # SVG arc flags: large_arc = 1 if sweep > 180°, sweep_flag = 1 (CCW in img)
                sweep_deg = (e - s) % 360
                large_arc = 1 if sweep_deg > 180 else 0
                d = f"M {sx:.3f} {sy:.3f} A {r:.3f} {r:.3f} 0 {large_arc} 1 {ex:.3f} {ey:.3f}"
                geometry.add(dwg.path(d=d, **stroke_kw))

            elif ptype == "polyline":
                pts = prim.get("points") or []
                if len(pts) < 2:
                    logger.warning(
                        f"SVG: polyline with <2 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                geometry.add(dwg.polyline(
                    points=[(float(p[0]), float(p[1])) for p in pts],
                    **stroke_kw,
                ))

            elif ptype == "polygon":
                pts = prim.get("points") or []
                if len(pts) < 3:
                    logger.warning(
                        f"SVG: polygon with <3 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                geometry.add(dwg.polygon(
                    points=[(float(p[0]), float(p[1])) for p in pts],
                    **stroke_kw,
                ))

            elif ptype == "ellipse":
                cx, cy = prim["center"]
                a, b   = float(prim["a"]), float(prim["b"])
                angle  = float(prim.get("angle", 0.0))   # degrees
                el = dwg.ellipse(center=(cx, cy), r=(a, b), **stroke_kw)
                if angle:
                    el["transform"] = f"rotate({angle} {cx} {cy})"
                geometry.add(el)

            elif ptype == "bezier":
                d = _svg_bezier_d(prim.get("points") or [])
                if d:
                    geometry.add(dwg.path(d=d, **stroke_kw))

            elif ptype == "path":
                # Compound stroke rendered as ONE continuous path so segment
                # seams use proper line joins. (Separate per-segment elements
                # left a ~1px missing-ink notch at every join that
                # systematically inflated D2C Chamfer.) DXF still gets native
                # LINE/ARC/SPLINE entities per segment.
                d = _svg_path_continuous(prim.get("segments", []))
                if d:
                    skw = dict(stroke_kw)
                    skw["stroke_linejoin"] = "round"
                    geometry.add(dwg.path(d=d, **skw))

            elif ptype == "hatch":
                _svg_add_hatch(dwg, geometry, prim, sw, n_written)

            elif ptype == "hatch_strokes":
                d = " ".join("M " + " L ".join(f"{x:.6f} {y:.6f}" for x, y in stroke)
                             for stroke in prim["strokes"])
                geometry.add(dwg.path(d=d, **stroke_kw))
                import hashlib
                item["stroke_count"] = len(prim["strokes"])
                item["stroke_geometry_sha256"] = hashlib.sha256(d.encode()).hexdigest()

            else:
                logger.warning(f"SVG: skipping unknown primitive type '{ptype}'")
                continue

            entities = geometry.elements[before:]
            if not entities:
                raise ValueError("Primitive emitted no SVG geometry")
            capture_entities(item, entities, svg=True, prefix="primitive")
            item["complete"] = True
            n_written += 1
        except Exception as exc:
            capture_entities(item, geometry.elements[before:], svg=True, prefix="primitive")
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            logger.warning(f"SVG: failed to export primitive {prim}: {exc}")

    # Annotations (Bezugszeichen) — also in patent SVG previews
    for index, ann in enumerate(data.get("annotations", [])):
        item = start_item(audit, "annotations", index)
        item.update(label="missing", leaders_written=0)
        before = len(geometry.elements)
        try:
            leader_lines = ann.get("leader_lines") or []
            for leader in leader_lines:
                p1, p2 = leader.get("p1"), leader.get("p2")
                if not p1 or not p2:
                    raise ValueError("Missing reference leader endpoints")
                geometry.add(dwg.line(
                    start=(float(p1[0]), float(p1[1])),
                    end=(float(p2[0]), float(p2[1])),
                    stroke="black", stroke_width=0.7,
                ))
                item["leaders_written"] += 1
            if not leader_lines and "leader_to" in ann:
                x, y = ann["position"]
                lx, ly = ann["leader_to"]
                geometry.add(dwg.line(
                    start=(x, y), end=(lx, ly),
                    stroke="black", stroke_width=0.7,
                ))
                item["leaders_written"] += 1

            image_path = ann.get("image_path")
            crop_bbox = ann.get("crop_bbox") or ann.get("bbox")
            text = str(ann.get("text", "") or "")
            render_mode = ann.get("svg_render_mode", "crop")
            if render_mode not in {"crop", "text"}:
                raise ValueError(f"Unknown SVG annotation mode: {render_mode}")
            crop_rendered = False
            if image_path and crop_bbox and (render_mode == "crop" or not text):
                href = _svg_data_uri(str(image_path))
                if href:
                    x, y, bw, bh = crop_bbox
                    img = dwg.image(
                        href=href,
                        insert=(float(x), float(y)),
                        size=(float(bw), float(bh)),
                    )
                    geometry.add(img)
                    crop_rendered = True
                    item["label"] = "crop"

            if text and not crop_rendered:
                x, y = ann["position"]
                text_attributes = {}
                font_size = ann.get("char_height", 14)
                if ann.get("source") == "stage0_references" and ann.get("bbox"):
                    font_size = ann.get("char_height", 0.8 * float(ann["bbox"][3]))
                    text_attributes = {"text_anchor": "middle", "dominant_baseline": "central"}
                geometry.add(dwg.text(
                    text,
                    insert=(x, y),
                    font_size=font_size,
                    font_family="Arial",
                    fill="black",
                    **text_attributes,
                ))
                item["label"] = "text"
        except Exception as exc:
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            logger.warning(f"SVG: failed to render annotation {ann}: {exc}")
        capture_entities(item, geometry.elements[before:], svg=True, prefix="annotation")
        finish_annotation(item, ann)

    dwg.save()
    return n_written


# ─── DXF export — basic mode ─────────────────────────────────────────────────

def _bezier_sample(points: list, per: int = 10) -> list:
    """Sample a poly-cubic-Bézier control list into a smooth point list."""
    if not points or len(points) < 4 or (len(points) - 1) % 3 != 0:
        return [list(p) for p in points]
    out = [list(points[0])]
    for i in range(1, len(points), 3):
        p0, c1, c2, p1 = points[i - 1], points[i], points[i + 1], points[i + 2]
        for k in range(1, per + 1):
            t = k / per
            mt = 1.0 - t
            x = (mt**3 * p0[0] + 3*mt**2*t * c1[0] + 3*mt*t**2 * c2[0] + t**3 * p1[0])
            y = (mt**3 * p0[1] + 3*mt**2*t * c1[1] + 3*mt*t**2 * c2[1] + t**3 * p1[1])
            out.append([x, y])
    return out


def _dxf_add_segment(msp, seg: dict, H: float, dxfattribs: Optional[dict] = None) -> None:
    """Add one path segment to the modelspace as a native LINE / ARC / SPLINE."""
    kw = {"dxfattribs": dxfattribs} if dxfattribs else {}
    t = seg.get("type")
    if t == "line":
        msp.add_line(_flip_y_point(seg["p1"], H), _flip_y_point(seg["p2"], H), **kw)
    elif t == "arc":
        c = _flip_y_point(seg["center"], H)
        s_new, e_new = _flip_y_arc_angles(seg["start_angle"], seg["end_angle"])
        msp.add_arc(center=c, radius=seg["radius"],
                    start_angle=s_new, end_angle=e_new, **kw)
    elif t == "bezier":
        pts = [_flip_y_point(p, H) for p in _bezier_sample(seg.get("points") or [])]
        if len(pts) < 2:
            return
        try:
            msp.add_spline(fit_points=pts, **kw)        # smooth native spline
        except Exception:
            msp.add_lwpolyline(pts, **kw)               # robust fallback


def _dxf_add_path(msp, prim: dict, H: float, dxfattribs: Optional[dict] = None) -> None:
    for seg in prim.get("segments", []):
        before = len(msp)
        _dxf_add_segment(msp, seg, H, dxfattribs)
        if len(msp) == before:
            raise ValueError("Path segment emitted no DXF geometry")


def _dxf_add_hatch(msp, prim: dict, H: float, dxfattribs: Optional[dict] = None) -> None:
    """Emit a hatch region as a native HATCH (boundary + parametric pattern).

    One pattern line per detected angle (single hatch ⇒ 1, cross-hatch ⇒ 2),
    each offset perpendicular by the detected spacing. DXF Y is up vs the image's
    Y-down, so the angle is negated. The region boundary is also emitted as a
    closed polyline (it is real geometry, not just the fill border).
    """
    boundary = prim.get("boundary") or []
    if len(boundary) < 3:
        return
    pts = [_flip_y_point(p, H) for p in boundary]
    attribs = dict(dxfattribs or {})
    msp.add_lwpolyline(pts, close=True, dxfattribs=attribs)
    angles = prim.get("angles") or []
    spacing = float(prim.get("spacing") or 0.0)
    if spacing < 1e-6 or not angles:
        return
    hatch = msp.add_hatch(dxfattribs=attribs)
    hatch.paths.add_polyline_path(pts, is_closed=True)
    definition = []
    for a in angles:
        a_dxf = -float(a)
        th = math.radians(a_dxf)
        offset = (-spacing * math.sin(th), spacing * math.cos(th))  # ⟂, mag=spacing
        definition.append([a_dxf, (0.0, 0.0), offset, []])
    try:
        hatch.set_pattern_fill("CUSTOM", definition=definition)
    except Exception:
        hatch.set_pattern_fill("ANSI31", scale=max(1.0, spacing / 3.175),
                               angle=-float(angles[0]))


def _export_dxf_basic(data: dict, out_path: Path, audit: Optional[dict] = None) -> int:
    """
    Single-layer DXF for AutoCAD/SolidWorks/KiCad import.
    All primitives go on layer 0 with default linetype.
    """
    H = data["image_size"][1]
    doc = ezdxf.new(dxfversion="R2010", setup=True)
    msp = doc.modelspace()

    n_written = 0

    for index, prim in enumerate(data["primitives"]):
        item = start_item(audit, "primitives", index)
        before = len(msp)
        try:
            validate_primitive(prim)
            ptype = prim["type"]
            if ptype == "line":
                p1 = _flip_y_point(prim["p1"], H)
                p2 = _flip_y_point(prim["p2"], H)
                msp.add_line(p1, p2)

            elif ptype == "circle":
                c = _flip_y_point(prim["center"], H)
                msp.add_circle(c, prim["radius"])

            elif ptype == "arc":
                c        = _flip_y_point(prim["center"], H)
                s_new, e_new = _flip_y_arc_angles(
                    prim["start_angle"], prim["end_angle"]
                )
                msp.add_arc(
                    center=c,
                    radius=prim["radius"],
                    start_angle=s_new,
                    end_angle=e_new,
                )

            elif ptype == "polyline":
                pts = prim.get("points") or []
                if len(pts) < 2:
                    logger.warning(
                        f"DXF basic: polyline with <2 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                flipped = [_flip_y_point(p, H) for p in pts]
                msp.add_lwpolyline(flipped)

            elif ptype == "polygon":
                pts = prim.get("points") or []
                if len(pts) < 3:
                    logger.warning(
                        f"DXF basic: polygon with <3 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                flipped = [_flip_y_point(p, H) for p in pts]
                msp.add_lwpolyline(flipped, close=True)

            elif ptype == "ellipse":
                c        = _flip_y_point(prim["center"], H)
                a, b     = float(prim["a"]), float(prim["b"])
                # DXF defines the ellipse by its major-axis endpoint relative to
                # the centre. Stage 3's angle is in image-space degrees (Y-down);
                # the Y-flip negates it.
                angle    = math.radians(-float(prim.get("angle", 0.0)))
                major_end = (a * math.cos(angle), a * math.sin(angle))
                ratio    = max(min(b / a, 1.0), 1e-6) if a > 0 else 1.0
                msp.add_ellipse(center=c, major_axis=major_end, ratio=ratio)

            elif ptype == "bezier":
                _dxf_add_segment(msp, prim, H)

            elif ptype == "path":
                _dxf_add_path(msp, prim, H)

            elif ptype == "hatch":
                _dxf_add_hatch(msp, prim, H)

            elif ptype == "hatch_strokes":
                strokes = [[_flip_y_point(p, H) for p in stroke] for stroke in prim["strokes"]]
                for stroke in strokes:
                    msp.add_lwpolyline(stroke)
                item["stroke_count"] = len(strokes)
                item["stroke_geometry_sha256"] = stroke_digest(strokes)

            else:
                logger.warning(f"DXF basic: skipping unknown type '{ptype}'")
                continue

            entities = [entity for entity in msp[before:] if entity.is_alive]
            if not entities:
                raise ValueError("Primitive emitted no DXF geometry")
            capture_entities(item, entities, svg=False, prefix="primitive")
            item["complete"] = True
            n_written += 1
        except Exception as exc:
            capture_entities(item, [entity for entity in msp[before:] if entity.is_alive], svg=False, prefix="primitive")
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            logger.warning(f"DXF basic: failed to export primitive {prim}: {exc}")

    doc.saveas(str(out_path))
    return n_written


# ─── DXF export — patent-ready mode ──────────────────────────────────────────

def _ensure_layers(doc) -> None:
    """Create ISO 128 layers if not already present."""
    existing = {layer.dxf.name for layer in doc.layers}
    for name, spec in ISO128_LAYERS.items():
        layer_name = name.upper()
        if layer_name in existing:
            continue
        # Verify linetype exists; fall back to CONTINUOUS if not loaded
        if spec["linetype"] not in doc.linetypes:
            logger.warning(
                f"Linetype '{spec['linetype']}' not loaded — "
                f"layer '{layer_name}' will use CONTINUOUS"
            )
            linetype = "CONTINUOUS"
        else:
            linetype = spec["linetype"]

        doc.layers.add(
            name       = layer_name,
            color      = spec["color"],
            linetype   = linetype,
            lineweight = spec["lineweight"],
        )


def _add_bezugszeichen(msp, ann: dict, image_h: float) -> None:
    """
    Add a single reference numeral to the modelspace as MTEXT plus a leader
    line on the LEADER layer. Per EPO Rule 46, numerals must be uniform in
    size and clearly identify the referenced feature.
    """
    pos = _flip_y_point(ann["position"], image_h)

    leader_lines = ann.get("leader_lines") or []
    for leader in leader_lines:
        p1, p2 = leader.get("p1"), leader.get("p2")
        if not p1 or not p2:
            raise ValueError("Missing reference leader endpoints")
        msp.add_line(
            _flip_y_point(p1, image_h),
            _flip_y_point(p2, image_h),
            dxfattribs={"layer": "LEADER"},
        )

    if not leader_lines and "leader_to" in ann:
        tip = _flip_y_point(ann["leader_to"], image_h)
        msp.add_line(pos, tip, dxfattribs={"layer": "LEADER"})

    text = str(ann.get("text", "") or "")
    if text:
        char_h = ann.get("char_height", 12)
        attributes = {"layer": "TEXT", "char_height": char_h}
        if ann.get("source") == "stage0_references" and ann.get("bbox"):
            attributes["char_height"] = ann.get("char_height", 0.8 * float(ann["bbox"][3]))
            attributes["attachment_point"] = 5  # reference positions are box centres
        msp.add_mtext(
            text,
            dxfattribs=attributes,
        ).set_location(insert=pos)


def _export_dxf_patent(data: dict, out_path: Path, audit: Optional[dict] = None) -> tuple:
    """
    ISO 128-compliant layered DXF.

    Returns (n_primitives_written, n_annotations_written).
    """
    H = data["image_size"][1]
    doc = ezdxf.new(dxfversion="R2010", setup=True)
    _ensure_layers(doc)
    msp = doc.modelspace()

    n_written = 0

    for index, prim in enumerate(data["primitives"]):
        item = start_item(audit, "primitives", index)
        before = len(msp)
        try:
            validate_primitive(prim)
            style = _primitive_style(prim)
            layer = style.upper()
            attribs = {"layer": layer}

            ptype = prim["type"]
            if ptype == "line":
                p1 = _flip_y_point(prim["p1"], H)
                p2 = _flip_y_point(prim["p2"], H)
                msp.add_line(p1, p2, dxfattribs=attribs)

            elif ptype == "circle":
                c = _flip_y_point(prim["center"], H)
                r = prim["radius"]
                msp.add_circle(c, r, dxfattribs=attribs)
                # Patent convention: every circle gets a centre cross on CENTER
                cx, cy = c
                ext = r * 1.15        # cross extends slightly past circle
                msp.add_line((cx - ext, cy), (cx + ext, cy),
                             dxfattribs={"layer": "CENTER"})
                msp.add_line((cx, cy - ext), (cx, cy + ext),
                             dxfattribs={"layer": "CENTER"})

            elif ptype == "arc":
                c        = _flip_y_point(prim["center"], H)
                s_new, e_new = _flip_y_arc_angles(
                    prim["start_angle"], prim["end_angle"]
                )
                msp.add_arc(
                    center=c, radius=prim["radius"],
                    start_angle=s_new, end_angle=e_new,
                    dxfattribs=attribs,
                )

            elif ptype == "polyline":
                pts = prim.get("points") or []
                if len(pts) < 2:
                    logger.warning(
                        f"DXF patent: polyline with <2 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                flipped = [_flip_y_point(p, H) for p in pts]
                msp.add_lwpolyline(flipped, dxfattribs=attribs)

            elif ptype == "polygon":
                pts = prim.get("points") or []
                if len(pts) < 3:
                    logger.warning(
                        f"DXF patent: polygon with <3 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                flipped = [_flip_y_point(p, H) for p in pts]
                msp.add_lwpolyline(flipped, close=True, dxfattribs=attribs)

            elif ptype == "ellipse":
                c        = _flip_y_point(prim["center"], H)
                a, b     = float(prim["a"]), float(prim["b"])
                angle    = math.radians(-float(prim.get("angle", 0.0)))
                major_end = (a * math.cos(angle), a * math.sin(angle))
                ratio    = max(min(b / a, 1.0), 1e-6) if a > 0 else 1.0
                msp.add_ellipse(
                    center=c, major_axis=major_end, ratio=ratio,
                    dxfattribs=attribs,
                )

            elif ptype == "bezier":
                _dxf_add_segment(msp, prim, H, attribs)

            elif ptype == "path":
                _dxf_add_path(msp, prim, H, attribs)

            elif ptype == "hatch":
                _dxf_add_hatch(msp, prim, H, {"layer": "HACHURE"})

            elif ptype == "hatch_strokes":
                strokes = [[_flip_y_point(p, H) for p in stroke] for stroke in prim["strokes"]]
                for stroke in strokes:
                    msp.add_lwpolyline(stroke, dxfattribs=attribs)
                item["stroke_count"] = len(strokes)
                item["stroke_geometry_sha256"] = stroke_digest(strokes)

            else:
                logger.warning(f"DXF patent: skipping unknown type '{ptype}'")
                continue

            entities = [entity for entity in msp[before:] if entity.is_alive]
            if not entities:
                raise ValueError("Primitive emitted no DXF geometry")
            capture_entities(item, entities, svg=False, prefix="primitive")
            item["complete"] = True
            n_written += 1
        except Exception as exc:
            capture_entities(item, [entity for entity in msp[before:] if entity.is_alive], svg=False, prefix="primitive")
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            logger.warning(f"DXF patent: failed to export primitive {prim}: {exc}")

    n_ann = 0
    for index, ann in enumerate(data.get("annotations", [])):
        item = start_item(audit, "annotations", index)
        item.update(label="unknown" if not ann.get("text") else "missing", leaders_written=0)
        before = len(msp)
        try:
            _add_bezugszeichen(msp, ann, H)
        except Exception as exc:
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            logger.warning(f"DXF patent: failed annotation {ann}: {exc}")
        entities = [entity for entity in msp[before:] if entity.is_alive]
        capture_entities(item, entities, svg=False, prefix="annotation")
        item["leaders_written"] = sum(e.dxftype() == "LINE" for e in entities)
        if any(e.dxftype() == "MTEXT" for e in entities):
            item["label"] = "text"
        finish_annotation(item, ann)
        n_ann += int(item["complete"])

    doc.saveas(str(out_path))
    return n_written, n_ann


# ─── Public stage function ───────────────────────────────────────────────────

def run(
    input_json: Path,
    output_dir: Path,
    sketch_id: Optional[str] = None,
    formats: tuple = ("svg",),
    dxf_mode: str = "basic",
) -> Stage4Result:
    """
    Run Stage 4 export on a single sketch.

    Parameters
    ----------
    input_json : Path
        Stage 3 output JSON containing primitives (and optional annotations).
    output_dir : Path
        Root output directory. Stage 4 writes to output_dir/vectors/.
    sketch_id : str | None
        Override sketch ID; defaults to value inside the JSON.
    formats : tuple
        Subset of {"svg", "dxf"}. Empty selection raises.
    dxf_mode : str
        "basic" (single layer) | "patent" (ISO 128 layered + Bezugszeichen).
        Ignored when "dxf" is not in `formats`.

    Returns
    -------
    Stage4Result
        Paths and counts.
    """
    t_start = time.perf_counter()

    if not formats:
        raise ValueError("`formats` must contain at least one of 'svg', 'dxf'.")
    if dxf_mode not in ("basic", "patent"):
        raise ValueError(f"dxf_mode must be 'basic' or 'patent', got '{dxf_mode}'")

    data = _load_input(input_json)
    sid  = sketch_id or data["sketch_id"]
    n_in = len(data["primitives"])

    vec_dir = output_dir / "vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{sid}] Stage 4 — exporting {n_in} primitives "
        f"(formats={list(formats)}, dxf_mode={dxf_mode})"
    )

    result = Stage4Result(
        sketch_id        = sid,
        n_primitives_in  = n_in,
        n_annotations    = len(data["annotations"]),
        dxf_mode         = dxf_mode if "dxf" in formats else None,
    )

    if not formats or len(set(formats)) != len(formats) or set(formats) - {"svg", "dxf"}:
        raise ValueError("Export requires unique supported formats: svg, dxf")
    for format_name in formats:
        path = vec_dir / f"{sid}.{format_name}"
        report = {"path": str(path.resolve()), "primitives": [], "annotations": [],
                  "errors": [], "serialized_valid": False, "complete": False}
        result.format_reports[format_name] = report
        try:
            if format_name == "svg":
                _export_svg(data, path, default_sw=data.get("stroke_width"), audit=report)
                result.svg_path = path
            else:
                if dxf_mode == "basic":
                    _export_dxf_basic(data, path, audit=report)
                else:
                    _export_dxf_patent(data, path, audit=report)
                result.dxf_path = path
            verify_serialized(format_name, path, report)
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
            logger.warning("%s export failed: %s", format_name, exc)
        report["primitives_created"] = sum(item["complete"] for item in report["primitives"])
        report["primitives_written"] = report["primitives_created"] if report["serialized_valid"] else 0
        report["complete"] = (
            report["serialized_valid"] and not report["errors"]
            and len(report["primitives"]) == n_in
            and report["primitives_written"] == n_in
            and len(report["annotations"]) == len(data["annotations"])
            and all(item["complete"] for item in report["annotations"])
        )

    result.n_primitives_out = min(r["primitives_written"] for r in result.format_reports.values())
    result.processing_time_s = time.perf_counter() - t_start
    result.flagged = not all(r["complete"] for r in result.format_reports.values())
    result.export_report_path = vec_dir / f"{sid}_export_report.json"
    export_report = {
        "schema": "ap3-export-report-v1", "sketch_id": sid,
        "input_json": str(input_json.resolve()), "input_sha256": file_digest(input_json),
        "expected_primitives": n_in, "expected_annotations": len(data["annotations"]),
        "formats": result.format_reports,
    }
    temporary = result.export_report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(export_report, indent=2, allow_nan=False) + "\n")
    temporary.replace(result.export_report_path)

    if result.flagged:
        logger.warning(
            f"[{sid}] FLAGGED — an export or annotation is incomplete; see export report"
        )
    else:
        logger.info(f"[{sid}] Stage 4 done in {result.processing_time_s:.2f}s")

    return result


# ─── CLI for standalone testing ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 4 — Export RANSAC primitives to SVG and/or DXF."
    )
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser.add_argument("input",  type=Path,
                        help="Stage 3 primitives JSON")
    parser.add_argument("--output",   type=Path, default=PROJECT_ROOT / "output",
                        help="Output root directory (default: <project>/output)")
    parser.add_argument("--id",       type=str,  default=None,
                        help="Sketch ID (default: from JSON)")
    parser.add_argument("--format",   choices=["svg", "dxf", "both"],
                        default="both",
                        help="Output format(s) to produce (default: both)")
    parser.add_argument("--dxf-mode", choices=["basic", "patent"],
                        default="basic",
                        help="DXF compliance level (default: basic). "
                             "Use 'patent' for ISO 128 layered output with "
                             "Bezugszeichen.")
    args = parser.parse_args()

    formats = ("svg", "dxf") if args.format == "both" else (args.format,)

    result = run(
        input_json = args.input,
        output_dir = args.output,
        sketch_id  = args.id,
        formats    = formats,
        dxf_mode   = args.dxf_mode,
    )

    print(f"\n{'─'*55}")
    print(f"  Sketch ID         : {result.sketch_id}")
    print(f"  SVG               : {result.svg_path or '—'}")
    print(f"  DXF               : {result.dxf_path or '—'}")
    print(f"  DXF mode          : {result.dxf_mode or '—'}")
    print(f"  Primitives in/out : {result.n_primitives_in} → {result.n_primitives_out}")
    print(f"  Annotations       : {result.n_annotations}")
    print(f"  Flagged           : {'YES ⚠' if result.flagged else 'no'}")
    print(f"  Processing time   : {result.processing_time_s:.2f}s")
    print(f"{'─'*55}")
