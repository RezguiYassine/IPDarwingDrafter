from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from syntheticData.patentvec.generator import PilotGenerator, SourcePool, write_sample_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "PatentVecSyntheticPreview"
FULL_DATA_ROOT = Path("/media/safe/secondary disk/IPdrawings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small PatentVec visual-approval set."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=240724)
    parser.add_argument("--canvas", type=int, default=1024)
    parser.add_argument(
        "--sketchgraphs",
        type=Path,
        default=PROJECT_ROOT / "data" / "SketchGraphs" / "raw" / "sg_t16_train.npy",
    )
    parser.add_argument(
        "--cadvg-root", type=Path, default=PROJECT_ROOT / "data" / "Drawing2CAD"
    )
    return parser.parse_args()


def _operator_schedule(count: int) -> list[str]:
    paired = [
        "t_junction",
        "t_junction",
        "endpoint_join",
        "endpoint_join",
        "containment",
        "containment",
        "concentric",
        "concentric",
        "tangent",
        "tangent",
    ]
    return [paired[index % len(paired)] for index in range(count)]


def _contact_sheet(output: Path, rows: list[dict]) -> None:
    thumbnails = []
    thumb_width = 1180
    for row in rows:
        path = output / row["relative_path"] / "preview.png"
        with Image.open(path) as image:
            image = image.convert("RGB")
            height = round(image.height * thumb_width / image.width)
            thumbnails.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    if not thumbnails:
        return
    columns = 2
    gutter = 24
    header = 52
    tile_height = max(image.height for image in thumbnails) + header
    rows_count = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width + (columns + 1) * gutter, rows_count * tile_height + gutter),
        "#eceff1",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (thumbnail, row) in enumerate(zip(thumbnails, rows)):
        x = gutter + (index % columns) * (thumb_width + gutter)
        y = gutter + (index // columns) * tile_height
        draw.text(
            (x + 4, y + 8),
            f"{row['sample_id']}  |  {row['operator']}  |  accepted",
            fill="#101418",
            font=font,
        )
        sheet.paste(thumbnail, (x, y + header))
    sheet.save(output / "CONTACT_SHEET.jpg", quality=92, optimize=True)


def _viewer(output: Path, rows: list[dict]) -> None:
    figures = []
    for row in rows:
        relative = row["relative_path"]
        quality = row["quality"]
        figures.append(
            f"""
            <figure>
              <a href="{relative}/sample.json"><img src="{relative}/preview.png" loading="lazy" alt="{row['sample_id']} preview"></a>
              <figcaption>
                <strong>{row['sample_id']}</strong>
                <span>{row['operator']}</span>
                <span>{quality['visible_primitive_count']} vectors</span>
                <span>{quality['foreground_ratio']:.3f} foreground</span>
                <nav>
                  <a href="{relative}/visible.svg">visible SVG</a>
                  <a href="{relative}/amodal.svg">amodal SVG</a>
                  <a href="{relative}/quality.json">quality</a>
                </nav>
              </figcaption>
            </figure>
            """
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PatentVec synthetic preview</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f3f5f6; color: #172027; }}
    header {{ padding: 22px 28px 16px; border-bottom: 1px solid #cbd2d6; background: white; position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    header p {{ margin: 5px 0 0; color: #52616b; font-size: 14px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 18px; padding: 22px; }}
    figure {{ margin: 0; background: white; border: 1px solid #cbd2d6; border-radius: 6px; overflow: hidden; }}
    figure img {{ display: block; width: 100%; height: auto; background: white; }}
    figcaption {{ min-height: 44px; display: flex; align-items: center; flex-wrap: wrap; gap: 10px 16px; padding: 10px 12px; border-top: 1px solid #dbe0e3; font-size: 13px; }}
    figcaption span {{ color: #52616b; }}
    nav {{ display: flex; gap: 12px; margin-left: auto; }}
    a {{ color: #075a8c; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 650px) {{ main {{ grid-template-columns: 1fr; padding: 10px; }} nav {{ width: 100%; margin-left: 0; }} }}
  </style>
</head>
<body>
  <header><h1>PatentVec synthetic preview</h1><p>Clean | degraded | semantic overlay with junction markers. Full generation is awaiting visual approval.</p></header>
  <main>{''.join(figures)}</main>
</body>
</html>
"""
    (output / "index.html").write_text(html)


def main() -> int:
    args = parse_args()
    if not 1 <= args.count <= 20:
        raise SystemExit("preview count must be between 1 and 20; this command cannot start a full run")
    args.output.mkdir(parents=True, exist_ok=True)
    pool = SourcePool(args.sketchgraphs, args.cadvg_root, split="train")
    generator = PilotGenerator(pool, canvas=args.canvas)
    rows = []
    for index, operator in enumerate(_operator_schedule(args.count)):
        sample_id = f"preview_{index:03d}_{operator}"
        seed = args.seed + index * 1009
        print(f"[{index + 1}/{args.count}] {sample_id}", flush=True)
        drawing = generator.generate(sample_id, seed=seed, operator=operator)
        rows.append(write_sample_artifacts(drawing, args.output, source_index=index))

    manifest = {
        "name": "PatentVec-Compose2D visual approval preview",
        "preview_only": True,
        "full_generation_started": False,
        "awaiting_user_approval": True,
        "planned_full_data_root": str(FULL_DATA_ROOT),
        "count": len(rows),
        "seed": args.seed,
        "canvas": args.canvas,
        "operator_counts": dict(Counter(row["operator"] for row in rows)),
        "puhachov_keypoint_counts": {
            name: sum(
                row["training_targets"]["puhachov_keypoints"][name] for row in rows
            )
            for name in ("endpoint", "junction", "corner")
        },
        "free2cad_edge_counts": {
            name: sum(row["training_targets"]["free2cad_edges"][name] for row in rows)
            for name in ("line", "arc", "circle", "polyline", "bezier")
        },
        "all_quality_gates_passed": all(row["quality"]["accepted"] for row in rows),
        "samples": rows,
        "track_a_note": (
            "Free2CAD artifacts supervise only local primitive fitting. "
            "They are not CAD operation sequences or Track-B CAD programs."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (args.output / "manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    _contact_sheet(args.output, rows)
    _viewer(args.output, rows)
    print(json.dumps({key: value for key, value in manifest.items() if key != "samples"}, indent=2))
    print(f"viewer: {args.output / 'index.html'}")
    print(f"contact sheet: {args.output / 'CONTACT_SHEET.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
