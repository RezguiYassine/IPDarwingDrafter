"""Build a contact sheet for multilabel structural/hachure stroke targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .patentvec.hatch_stroke_targets import TARGET_KEYS, validate_hatch_stroke_targets


def _max_pool(mask: np.ndarray, maximum_size: int) -> np.ndarray:
    height, width = mask.shape
    factor = max(1, math.ceil(max(height, width) / maximum_size))
    padded_height = math.ceil(height / factor) * factor
    padded_width = math.ceil(width / factor) * factor
    padded = np.zeros((padded_height, padded_width), dtype=bool)
    padded[:height, :width] = mask > 0
    return padded.reshape(
        padded_height // factor,
        factor,
        padded_width // factor,
        factor,
    ).max(axis=(1, 3))


def _fit_panel(rgb: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray(rgb)
    image.thumbnail((size, size), Image.Resampling.NEAREST)
    panel = Image.new("RGB", (size, size), "white")
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    panel.paste(image, (left, top))
    return panel


def _render_panels(path: Path, panel_size: int) -> tuple[Image.Image, Image.Image, dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in TARGET_KEYS}
        metadata = json.loads(str(np.asarray(data["meta"])))
    stats = validate_hatch_stroke_targets(arrays)
    skeleton = _max_pool(arrays["input_skeleton"], panel_size)
    structural = _max_pool(arrays["structural_target"], panel_size)
    hatch = _max_pool(arrays["hatch_target"], panel_size)
    overlap = structural & hatch

    input_rgb = np.full((*skeleton.shape, 3), 255, dtype=np.uint8)
    input_rgb[skeleton] = (20, 24, 28)
    labels_rgb = np.full((*skeleton.shape, 3), 255, dtype=np.uint8)
    labels_rgb[structural & ~hatch] = (35, 112, 190)
    labels_rgb[hatch & ~structural] = (210, 64, 64)
    labels_rgb[overlap] = (235, 176, 28)
    return (
        _fit_panel(input_rgb, panel_size),
        _fit_panel(labels_rgb, panel_size),
        {**metadata, "pixel_counts": stats},
    )


def build_audit_sheet(
    dataset_root: Path,
    output_path: Path,
    *,
    sample_count: int = 16,
    columns: int = 4,
    panel_size: int = 256,
) -> None:
    rows_path = dataset_root / "manifest.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("empty hatch-stroke manifest")
    count = min(sample_count, len(rows))
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    selected = [rows[int(index)] for index in indices]

    font = ImageFont.load_default()
    margin = 12
    label_height = 42
    tile_width = panel_size * 2 + margin
    tile_height = panel_size + label_height
    rows_count = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB",
        (
            margin + columns * (tile_width + margin),
            margin + rows_count * (tile_height + margin) + 28,
        ),
        (238, 240, 242),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (margin, 6),
        "Input skeleton | structural=blue, hatch=red, overlap=yellow",
        fill=(20, 24, 28),
        font=font,
    )
    for index, row in enumerate(selected):
        column = index % columns
        row_index = index // columns
        x = margin + column * (tile_width + margin)
        y = margin + 28 + row_index * (tile_height + margin)
        source_path = dataset_root / row["path"]
        input_panel, label_panel, metadata = _render_panels(
            source_path, panel_size
        )
        sheet.paste(input_panel, (x, y))
        sheet.paste(label_panel, (x + panel_size, y))
        counts = metadata["pixel_counts"]
        title = f"{row['sample_id']}  {row['difficulty']}"
        detail = (
            f"S={counts['structural_only_pixels']} "
            f"H={counts['hatch_only_pixels']} O={counts['overlap_pixels']}"
        )
        draw.text((x, y + panel_size + 4), title, fill=(20, 24, 28), font=font)
        draw.text((x, y + panel_size + 20), detail, fill=(70, 74, 78), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=256)
    args = parser.parse_args()
    build_audit_sheet(
        args.dataset,
        args.output,
        sample_count=args.samples,
        columns=args.columns,
        panel_size=args.panel_size,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
