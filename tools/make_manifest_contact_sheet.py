"""
Build visual audit contact sheets from a PatentData training manifest.

Each sampled row is rendered as three panels:

  1. original input raster
  2. Stage-1 skeleton image
  3. exported SVG rendered back to a bitmap

The tool is intentionally lightweight and deterministic so curation decisions can
be reviewed and reproduced after each filtering pass.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_MANIFEST = Path("output/PatentData_clean12_gated/training_manifest_clip.csv")


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _sort_rows(rows: list[dict[str, Any]], mode: str, seed: int) -> list[dict[str, Any]]:
    out = list(rows)
    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(out)
        return out
    if mode == "worst-neg":
        return sorted(out, key=lambda row: _float(row.get("clip_neg_score")), reverse=True)
    if mode == "small-margin":
        return sorted(out, key=lambda row: _float(row.get("clip_margin"), 999.0))
    if mode == "high-micro":
        return sorted(out, key=lambda row: _float(row.get("s2_micro_edge_ratio")), reverse=True)
    if mode == "slowest":
        return sorted(out, key=lambda row: _float(row.get("total_time")), reverse=True)
    if mode == "worst-low-conf":
        return sorted(out, key=lambda row: _float(row.get("s3_low_conf_ratio")), reverse=True)
    raise ValueError(f"unknown sort mode: {mode}")


def _load_raster(path: str) -> Image.Image:
    img = Image.open(path)
    try:
        img.seek(0)
    except EOFError:
        pass
    return img.convert("RGB")


def _load_svg(path: str, width: int, height: int) -> Image.Image:
    try:
        import cairosvg
    except ImportError as exc:
        raise SystemExit(
            "cairosvg is required to render SVG contact sheets. "
            "Install it with: pip install cairosvg"
        ) from exc

    png = cairosvg.svg2png(url=path, output_width=width, output_height=height)
    return Image.open(io.BytesIO(png)).convert("RGB")


def _fit(img: Image.Image, width: int, height: int, background: str = "white") -> Image.Image:
    canvas = Image.new("RGB", (width, height), background)
    work = img.copy()
    work.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - work.width) // 2
    y = (height - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas


def _panel(path: str, width: int, height: int, label: str, draw_font: ImageFont.ImageFont) -> Image.Image:
    header_h = 18
    panel = Image.new("RGB", (width, height + header_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((4, 2), label, fill=(30, 30, 30), font=draw_font)

    if not path or not Path(path).exists():
        body = Image.new("RGB", (width, height), "white")
        body_draw = ImageDraw.Draw(body)
        body_draw.text((8, 8), "missing", fill=(160, 0, 0), font=draw_font)
    elif path.lower().endswith(".svg"):
        body = _load_svg(path, width, height)
    else:
        bg = "black" if "skeleton" in label.lower() else "white"
        body = _fit(_load_raster(path), width, height, background=bg)

    panel.paste(body, (0, header_h))
    draw.rectangle((0, 0, width - 1, height + header_h - 1), outline=(225, 225, 225))
    return panel


def _caption(row: dict[str, Any]) -> str:
    ident = f"{row.get('patent_id', '')} {row.get('sketch_id', '')}".strip()
    pos = row.get("clip_pos_label") or row.get("curation_reason") or ""
    neg = row.get("clip_neg_label") or ""
    pos_score = _float(row.get("clip_pos_score"))
    neg_score = _float(row.get("clip_neg_score"))
    margin = _float(row.get("clip_margin"))
    total = _float(row.get("total_time"))
    edges = row.get("s2_n_edges", "")
    micro = _float(row.get("s2_micro_edge_ratio"))
    low = _float(row.get("s3_low_conf_ratio"))
    return (
        f"{ident} | {pos} | neg={neg} {neg_score:.2f} pos={pos_score:.2f} "
        f"margin={margin:.2f} t={total:.1f}s edges={edges} micro={micro:.2f} low={low:.2f}"
    )


def build_sheet(args: argparse.Namespace) -> None:
    rows = _sort_rows(_read_rows(args.manifest), args.sort, args.seed)[: args.limit]
    if not rows:
        raise SystemExit(f"No rows found in {args.manifest}")

    font = ImageFont.load_default()
    pad = 8
    caption_h = 34
    panel_w = args.panel_width
    panel_h = args.panel_height
    sample_w = panel_w * 3 + pad * 4
    sample_h = caption_h + panel_h + 18 + pad * 2
    cols = max(1, args.cols)
    sheet_w = sample_w * cols
    sheet_h = sample_h * ((len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(rows):
        col = idx % cols
        grid_row = idx // cols
        x0 = col * sample_w
        y0 = grid_row * sample_h
        draw.rectangle((x0, y0, x0 + sample_w - 1, y0 + sample_h - 1), outline=(210, 210, 210))
        draw.text((x0 + pad, y0 + pad), _caption(row), fill=(20, 20, 20), font=font)

        paths = [
            ("input", row.get("input_path", "")),
            ("skeleton", row.get("cleaned_skeleton_path", "")),
            ("svg", row.get("svg_path", "")),
        ]
        px = x0 + pad
        py = y0 + caption_h
        for label, path in paths:
            try:
                panel = _panel(path, panel_w, panel_h, label, font)
            except Exception as exc:
                panel = Image.new("RGB", (panel_w, panel_h + 18), "white")
                pdraw = ImageDraw.Draw(panel)
                pdraw.text((4, 2), label, fill=(30, 30, 30), font=font)
                pdraw.text((8, 28), type(exc).__name__, fill=(160, 0, 0), font=font)
                pdraw.rectangle((0, 0, panel_w - 1, panel_h + 17), outline=(225, 225, 225))
            sheet.paste(panel, (px, py))
            px += panel_w + pad

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Contact sheet: {args.output}")
    print(f"Rows         : {len(rows)}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render a visual audit sheet from a manifest CSV.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sort",
        choices=("random", "worst-neg", "small-margin", "high-micro", "slowest", "worst-low-conf"),
        default="random",
    )
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--panel-width", type=int, default=210)
    p.add_argument("--panel-height", type=int, default=240)
    return p


def main(argv: list[str] | None = None) -> None:
    build_sheet(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
