"""Build side-by-side visual audit sheets for paired PatentData runs."""

from __future__ import annotations

import argparse
import io
import random
import sqlite3
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be NAME=RUN_DIR")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.joinpath("results.db").exists():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return name, path


def _load(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    with sqlite3.connect(run_dir / "results.db") as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("SELECT * FROM results").fetchall()
    return {
        (str(row["patent_id"]), str(row["sketch_id"])): dict(row)
        for row in rows
    }


def _number(row: dict[str, Any], field: str, default: float) -> float:
    value = row.get(field)
    return default if value is None else float(value)


def _ordered_keys(
    data: dict[str, dict[tuple[str, str], dict[str, Any]]],
    run_names: list[str],
    mode: str,
    focus_name: str,
    seed: int,
) -> list[tuple[str, str]]:
    keys = sorted(set.intersection(*(set(data[name]) for name in run_names)))
    baseline_name = run_names[0]
    baseline = data[baseline_name]
    focus = data[focus_name]
    if mode == "random":
        random.Random(seed).shuffle(keys)
        return keys
    if mode == "high-hachure":
        return sorted(
            keys,
            key=lambda key: _number(
                baseline[key], "s2_n_hachure_edges_removed", -1.0,
            ),
            reverse=True,
        )
    if mode in {"fragmentation-improvement", "fragmentation-regression"}:
        def score(key: tuple[str, str]) -> tuple[float, float]:
            micro_delta = (
                _number(focus[key], "s2_micro_edge_ratio", 1.0)
                - _number(baseline[key], "s2_micro_edge_ratio", 1.0)
            )
            median_delta = (
                _number(focus[key], "s2_median_edge_len", 0.0)
                - _number(baseline[key], "s2_median_edge_len", 0.0)
            )
            return micro_delta, -median_delta

        return sorted(
            keys,
            key=score,
            reverse=mode == "fragmentation-regression",
        )
    if mode == "status-change":
        return sorted(
            keys,
            key=lambda key: (
                baseline[key].get("status") == focus[key].get("status"), key,
            ),
        )
    raise ValueError(f"unknown sort mode: {mode}")


def _load_raster(path: Path) -> Image.Image:
    image = Image.open(path)
    try:
        image.seek(0)
    except EOFError:
        pass
    return image.convert("RGB")


def _load_svg(path: Path, width: int, height: int) -> Image.Image:
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("cairosvg is required for SVG rendering") from exc
    png = cairosvg.svg2png(
        url=str(path), output_width=width, output_height=height,
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def _load_graph(path: Path) -> Image.Image:
    import json

    graph = json.loads(path.read_text())
    height, width = graph["image_shape"]
    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()
    for edge in graph.get("edges", []):
        color = (30, 80, 190) if edge.get("is_closed") else (15, 15, 15)
        for x, y in edge.get("pixels", []):
            x, y = int(x), int(y)
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = color
    for edge in graph.get("removed_hachures", []):
        for x, y in edge.get("pixels", []):
            x, y = int(x), int(y)
            if 0 <= x < width and 0 <= y < height:
                pixels[x, y] = (235, 125, 20)
    return image


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = Image.new("RGB", (width, height), "white")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    result.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return result


def _run_panel(
    run_name: str,
    run_dir: Path,
    key: tuple[str, str],
    row: dict[str, Any],
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    patent_id, sketch_id = key
    svg = run_dir / patent_id / "vectors" / f"{sketch_id}.svg"
    graph = run_dir / patent_id / "graphs" / f"{sketch_id}_graph.json"
    mode = "SVG"
    try:
        if svg.exists():
            body = _load_svg(svg, width, height)
        elif graph.exists():
            body = _fit(_load_graph(graph), width, height)
            mode = "graph"
        else:
            body = Image.new("RGB", (width, height), "white")
            ImageDraw.Draw(body).text((8, 8), "no Stage-2 output", fill=(170, 0, 0), font=font)
            mode = "missing"
    except Exception as exc:
        body = Image.new("RGB", (width, height), "white")
        ImageDraw.Draw(body).text(
            (8, 8), type(exc).__name__, fill=(170, 0, 0), font=font,
        )
        mode = "render error"

    header_height = 34
    panel = Image.new("RGB", (width, height + header_height), "white")
    draw = ImageDraw.Draw(panel)
    status = str(row.get("status", "missing"))
    edges = row.get("s2_n_edges")
    micro = row.get("s2_micro_edge_ratio")
    median = row.get("s2_median_edge_len")
    draw.text((4, 2), f"{run_name} | {status} | {mode}", fill=(20, 20, 20), font=font)
    draw.text(
        (4, 17),
        f"edges={edges if edges is not None else '-'} "
        f"micro={micro:.3f} med={median:.1f}"
        if micro is not None and median is not None else "Stage-2 metrics unavailable",
        fill=(70, 70, 70),
        font=font,
    )
    panel.paste(body, (0, header_height))
    draw.rectangle(
        (0, 0, width - 1, height + header_height - 1), outline=(210, 210, 210),
    )
    return panel


def build(args: argparse.Namespace) -> None:
    runs = dict(args.run)
    run_names = list(runs)
    focus_name = args.focus_run or run_names[-1]
    if focus_name not in runs:
        raise SystemExit(f"unknown --focus-run: {focus_name}")
    data = {name: _load(path) for name, path in runs.items()}
    keys = _ordered_keys(data, run_names, args.sort, focus_name, args.seed)
    if args.sort == "status-change":
        baseline = data[run_names[0]]
        focus = data[focus_name]
        keys = [key for key in keys if baseline[key]["status"] != focus[key]["status"]]
    keys = keys[:args.limit]
    if not keys:
        raise SystemExit("no paired rows match this audit selection")

    font = ImageFont.load_default()
    pad = 8
    caption_height = 28
    panel_width = args.panel_width
    panel_height = args.panel_height
    columns = 1 + len(runs)
    row_width = pad + columns * (panel_width + pad)
    row_height = caption_height + panel_height + 34 + pad
    sheet = Image.new("RGB", (row_width, row_height * len(keys)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, key in enumerate(keys):
        patent_id, sketch_id = key
        y0 = index * row_height
        base_row = data[run_names[0]][key]
        draw.text(
            (pad, y0 + 7),
            f"{index + 1}. {patent_id}/{sketch_id}",
            fill=(20, 20, 20),
            font=font,
        )
        input_path = Path(str(base_row["input_path"]))
        try:
            input_body = _fit(_load_raster(input_path), panel_width, panel_height)
        except Exception as exc:
            input_body = Image.new("RGB", (panel_width, panel_height), "white")
            ImageDraw.Draw(input_body).text(
                (8, 8), type(exc).__name__, fill=(170, 0, 0), font=font,
            )
        input_panel = Image.new(
            "RGB", (panel_width, panel_height + 34), "white",
        )
        input_draw = ImageDraw.Draw(input_panel)
        input_draw.text((4, 9), "Original input", fill=(20, 20, 20), font=font)
        input_panel.paste(input_body, (0, 34))
        input_draw.rectangle(
            (0, 0, panel_width - 1, panel_height + 33), outline=(210, 210, 210),
        )
        x = pad
        sheet.paste(input_panel, (x, y0 + caption_height))
        x += panel_width + pad
        for run_name, run_dir in runs.items():
            panel = _run_panel(
                run_name, run_dir, key, data[run_name][key],
                panel_width, panel_height, font,
            )
            sheet.paste(panel, (x, y0 + caption_height))
            x += panel_width + pad

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"comparison sheet: {args.output}")
    print(f"selection: {args.sort}; focus={focus_name}; rows={len(keys)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", type=_parse_run, required=True,
        help="NAME=RUN_DIR; first run is the baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sort",
        choices=(
            "random", "high-hachure", "fragmentation-improvement",
            "fragmentation-regression", "status-change",
        ),
        default="random",
    )
    parser.add_argument("--focus-run")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=850725)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=360)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("at least two --run arguments are required")
    if len(dict(args.run)) != len(args.run):
        parser.error("run names must be unique")
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
