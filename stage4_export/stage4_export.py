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
  patent  — ISO 128 layered (visible / hidden / center / construction / text),
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

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ezdxf
import svgwrite

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
    "text":         {"linetype": "CONTINUOUS", "lineweight": 25, "color": 7},
    "leader":       {"linetype": "CONTINUOUS", "lineweight": 18, "color": 7},
}

DEFAULT_STYLE = "visible"   # primitives without an explicit style go here


# ─── Coordinate helpers ──────────────────────────────────────────────────────

def _flip_y_point(p: list, image_h: float) -> tuple:
    """Image (Y-down) → CAD (Y-up). x unchanged."""
    return (p[0], image_h - p[1])


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


def _arc_endpoints(cx: float, cy: float, r: float,
                   start_deg: float, end_deg: float) -> tuple:
    """Return (start_xy, end_xy) on the circle at the given angles (image space)."""
    sx = cx + r * math.cos(math.radians(start_deg))
    sy = cy + r * math.sin(math.radians(start_deg))
    ex = cx + r * math.cos(math.radians(end_deg))
    ey = cy + r * math.sin(math.radians(end_deg))
    return (sx, sy), (ex, ey)


def _export_svg(data: dict, out_path: Path,
                default_sw: Optional[float] = None) -> int:
    """
    Write SVG file. Returns the count of primitives successfully written.

    SVG keeps image coordinates as-is (Y-down) so previews line up with
    the source sketch without any extra transform.

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

    n_written = 0

    for prim in data["primitives"]:
        try:
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
                dwg.add(dwg.line(start=p1, end=p2, **stroke_kw))

            elif ptype == "circle":
                cx, cy = prim["center"]
                dwg.add(dwg.circle(center=(cx, cy), r=prim["radius"], **stroke_kw))

            elif ptype == "arc":
                cx, cy = prim["center"]
                r      = prim["radius"]
                s, e   = prim["start_angle"], prim["end_angle"]
                (sx, sy), (ex, ey) = _arc_endpoints(cx, cy, r, s, e)
                # SVG arc flags: large_arc = 1 if sweep > 180°, sweep_flag = 1 (CCW in img)
                sweep_deg = (e - s) % 360
                large_arc = 1 if sweep_deg > 180 else 0
                d = f"M {sx:.3f} {sy:.3f} A {r:.3f} {r:.3f} 0 {large_arc} 1 {ex:.3f} {ey:.3f}"
                dwg.add(dwg.path(d=d, **stroke_kw))

            elif ptype == "polyline":
                pts = prim.get("points") or []
                if len(pts) < 2:
                    logger.warning(
                        f"SVG: polyline with <2 points skipped (edge_id="
                        f"{prim.get('edge_id', '?')})"
                    )
                    continue
                dwg.add(dwg.polyline(
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
                dwg.add(el)

            else:
                logger.warning(f"SVG: skipping unknown primitive type '{ptype}'")
                continue

            n_written += 1
        except Exception as exc:
            logger.warning(f"SVG: failed to export primitive {prim}: {exc}")

    # Annotations (Bezugszeichen) — also in patent SVG previews
    for ann in data.get("annotations", []):
        try:
            x, y = ann["position"]
            dwg.add(dwg.text(
                ann["text"],
                insert=(x, y),
                font_size=14,
                font_family="Arial",
                fill="black",
            ))
            if "leader_to" in ann:
                lx, ly = ann["leader_to"]
                dwg.add(dwg.line(
                    start=(x, y), end=(lx, ly),
                    stroke="black", stroke_width=0.7,
                ))
        except Exception as exc:
            logger.warning(f"SVG: failed to render annotation {ann}: {exc}")

    dwg.save()
    return n_written


# ─── DXF export — basic mode ─────────────────────────────────────────────────

def _export_dxf_basic(data: dict, out_path: Path) -> int:
    """
    Single-layer DXF for AutoCAD/SolidWorks/KiCad import.
    All primitives go on layer 0 with default linetype.
    """
    H = data["image_size"][1]
    doc = ezdxf.new(dxfversion="R2010", setup=True)
    msp = doc.modelspace()

    n_written = 0

    for prim in data["primitives"]:
        try:
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

            else:
                logger.warning(f"DXF basic: skipping unknown type '{ptype}'")
                continue

            n_written += 1
        except Exception as exc:
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
    pos    = _flip_y_point(ann["position"], image_h)
    char_h = ann.get("char_height", 12)

    msp.add_mtext(
        str(ann["text"]),
        dxfattribs={"layer": "TEXT", "char_height": char_h},
    ).set_location(insert=pos)

    if "leader_to" in ann:
        tip = _flip_y_point(ann["leader_to"], image_h)
        msp.add_line(pos, tip, dxfattribs={"layer": "LEADER"})


def _export_dxf_patent(data: dict, out_path: Path) -> tuple:
    """
    ISO 128-compliant layered DXF.

    Returns (n_primitives_written, n_annotations_written).
    """
    H = data["image_size"][1]
    doc = ezdxf.new(dxfversion="R2010", setup=True)
    _ensure_layers(doc)
    msp = doc.modelspace()

    n_written = 0

    for prim in data["primitives"]:
        try:
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

            else:
                logger.warning(f"DXF patent: skipping unknown type '{ptype}'")
                continue

            n_written += 1
        except Exception as exc:
            logger.warning(f"DXF patent: failed to export primitive {prim}: {exc}")

    n_ann = 0
    for ann in data.get("annotations", []):
        try:
            _add_bezugszeichen(msp, ann, H)
            n_ann += 1
        except Exception as exc:
            logger.warning(f"DXF patent: failed annotation {ann}: {exc}")

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

    n_out_max = 0   # track the best primitive count across formats

    if "svg" in formats:
        svg_path = vec_dir / f"{sid}.svg"
        n_svg = _export_svg(data, svg_path, default_sw=data.get("stroke_width"))
        result.svg_path = svg_path
        n_out_max = max(n_out_max, n_svg)
        logger.info(f"[{sid}] SVG  → {svg_path}  ({n_svg}/{n_in} primitives)")

    if "dxf" in formats:
        dxf_path = vec_dir / f"{sid}.dxf"
        if dxf_mode == "basic":
            n_dxf = _export_dxf_basic(data, dxf_path)
            n_ann = 0
        else:
            n_dxf, n_ann = _export_dxf_patent(data, dxf_path)
        result.dxf_path = dxf_path
        n_out_max = max(n_out_max, n_dxf)
        logger.info(
            f"[{sid}] DXF  → {dxf_path}  "
            f"({n_dxf}/{n_in} primitives, {n_ann} Bezugszeichen, mode={dxf_mode})"
        )

    result.n_primitives_out  = n_out_max
    result.processing_time_s = time.perf_counter() - t_start
    result.flagged           = (n_out_max < n_in)

    if result.flagged:
        logger.warning(
            f"[{sid}] FLAGGED — only {n_out_max}/{n_in} primitives exported"
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