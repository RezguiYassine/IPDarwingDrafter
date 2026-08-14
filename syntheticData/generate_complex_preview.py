from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from syntheticData.patentvec.generator import (
    ComplexPilotGenerator,
    SourcePool,
    write_sample_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "PatentVecM5M6Preview"
FULL_DATA_ROOT = Path("/media/safe/secondary disk/IPdrawings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a restart-safe M5/M6 medium/hard review pilot."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=250724)
    parser.add_argument("--canvas", type=int, default=1024)
    parser.add_argument(
        "--sketchgraphs",
        type=Path,
        default=PROJECT_ROOT / "data" / "SketchGraphs" / "raw" / "sg_t16_train.npy",
    )
    parser.add_argument(
        "--cadvg-root", type=Path, default=PROJECT_ROOT / "data" / "Drawing2CAD"
    )
    parser.add_argument("--source-index", type=Path)
    return parser.parse_args()


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _schedule(count: int) -> list[str]:
    return ["medium" if index % 2 == 0 else "hard" for index in range(count)]


def _contact_sheets(output: Path, rows: list[dict], page_size: int = 10) -> list[str]:
    paths = []
    for page_start in range(0, len(rows), page_size):
        page_rows = rows[page_start : page_start + page_size]
        thumbnails = []
        thumb_width = 1160
        for row in page_rows:
            with Image.open(output / row["relative_path"] / "preview.png") as image:
                image = image.convert("RGB")
                height = round(image.height * thumb_width / image.width)
                thumbnails.append(
                    image.resize((thumb_width, height), Image.Resampling.LANCZOS)
                )
        columns = 2
        gutter = 24
        header = 48
        tile_height = max(item.height for item in thumbnails) + header
        row_count = (len(thumbnails) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (
                columns * thumb_width + (columns + 1) * gutter,
                row_count * tile_height + gutter,
            ),
            "#e9edef",
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for index, (thumbnail, row) in enumerate(zip(thumbnails, page_rows)):
            x = gutter + (index % columns) * (thumb_width + gutter)
            y = gutter + (index // columns) * tile_height
            complexity = row["quality"]["complexity"]
            label = (
                f"{row['sample_id']} | {row['difficulty']} | "
                f"C={complexity['source_component_count']} "
                f"R={complexity['interaction_count']} "
                f"T={complexity['interaction_type_count']} "
                f"J={complexity['structural_junction_count']}"
            )
            draw.text((x + 4, y + 8), label, fill="#11181c", font=font)
            sheet.paste(thumbnail, (x, y + header))
        filename = f"CONTACT_SHEET_{page_start // page_size + 1:02d}.jpg"
        sheet.save(output / filename, quality=92, optimize=True)
        paths.append(filename)
    return paths


def _viewer(output: Path, rows: list[dict], contact_sheets: list[str]) -> None:
    figures = []
    for row in rows:
        relative = row["relative_path"]
        quality = row["quality"]
        complexity = quality["complexity"]
        figures.append(
            f"""
            <figure data-difficulty="{row['difficulty']}">
              <a href="{relative}/sample.json"><img src="{relative}/preview.png" loading="lazy" alt="{row['sample_id']} preview"></a>
              <figcaption>
                <strong>{row['sample_id']}</strong>
                <span>{row['difficulty']}</span>
                <span>{complexity['source_component_count']} components</span>
                <span>{complexity['interaction_type_count']} relation types</span>
                <span>{complexity['structural_junction_count']} junctions</span>
                <span>cycle {complexity['cycle_rank']}</span>
                <nav>
                  <a href="{relative}/visible.svg">visible SVG</a>
                  <a href="{relative}/amodal.svg">amodal SVG</a>
                  <a href="{relative}/quality.json">quality</a>
                </nav>
              </figcaption>
            </figure>
            """
        )
    sheet_links = " ".join(
        f'<a href="{path}">sheet {index + 1}</a>'
        for index, path in enumerate(contact_sheets)
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PatentVec M5/M6 review pilot</title>
  <style>
    :root {{ font-family: Inter, system-ui, sans-serif; color: #172027; background: #eef1f2; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{ position: sticky; top: 0; z-index: 2; padding: 16px 24px; background: white; border-bottom: 1px solid #c5cdd1; }}
    h1 {{ margin: 0 0 5px; font-size: 21px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #56656d; font-size: 13px; }}
    header nav {{ margin-top: 8px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(620px, 1fr)); gap: 16px; padding: 18px; }}
    figure {{ margin: 0; overflow: hidden; background: white; border: 1px solid #c5cdd1; border-radius: 6px; }}
    figure img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; padding: 9px 11px; border-top: 1px solid #d9dfe2; font-size: 12px; }}
    figcaption span {{ color: #596971; }}
    figcaption nav {{ display: flex; gap: 10px; margin-left: auto; }}
    a {{ color: #075b88; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 700px) {{ main {{ grid-template-columns: 1fr; padding: 8px; }} figcaption nav {{ width: 100%; margin-left: 0; }} }}
  </style>
</head>
<body>
  <header>
    <h1>PatentVec M5/M6 medium and hard review pilot</h1>
    <p>Clean | degraded | semantic overlay. Full training-scale generation is disabled.</p>
    <nav>{sheet_links}</nav>
  </header>
  <main>{''.join(figures)}</main>
</body>
</html>
"""
    (output / "index.html").write_text(page)


def _summary(rows: list[dict]) -> dict:
    interaction_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    keypoint_counts: Counter[str] = Counter()
    free2cad_counts: Counter[str] = Counter()
    for row in rows:
        interaction_counts.update(row["quality"]["interaction_counts"])
        semantic_counts.update(row["quality"]["semantic_counts"])
        keypoint_counts.update(row["training_targets"]["puhachov_keypoints"])
        free2cad_counts.update(row["training_targets"]["free2cad_edges"])
    return {
        "difficulty_counts": dict(Counter(row["difficulty"] for row in rows)),
        "interaction_counts": dict(interaction_counts),
        "semantic_counts": dict(semantic_counts),
        "puhachov_keypoint_counts": dict(keypoint_counts),
        "free2cad_edge_counts": dict(free2cad_counts),
        "component_count_range": [
            min(row["quality"]["complexity"]["source_component_count"] for row in rows),
            max(row["quality"]["complexity"]["source_component_count"] for row in rows),
        ],
        "object_primitive_count_range": [
            min(row["quality"]["complexity"]["object_primitive_count"] for row in rows),
            max(row["quality"]["complexity"]["object_primitive_count"] for row in rows),
        ],
    }


def main() -> int:
    args = parse_args()
    if not 2 <= args.count <= 50:
        raise SystemExit("M5/M6 review count must be between 2 and 50")
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "config": vars(args)
            | {
                "output": str(args.output),
                "sketchgraphs": str(args.sketchgraphs),
                "cadvg_root": str(args.cadvg_root),
                "source_index": str(args.source_index) if args.source_index else None,
            },
            "rows": [],
            "errors": [],
        }
    )
    completed = {row["sample_id"]: row for row in progress.get("rows", [])}
    pool = SourcePool(
        args.sketchgraphs,
        args.cadvg_root,
        split="train",
        index_path=args.source_index,
    )
    generator = ComplexPilotGenerator(pool, canvas=args.canvas)
    for index, difficulty in enumerate(_schedule(args.count)):
        sample_id = f"m5m6_{index:03d}_{difficulty}"
        required = args.output / "samples" / sample_id / "sample.json"
        if sample_id in completed and required.exists():
            print(f"[{index + 1}/{args.count}] {sample_id} (resume skip)", flush=True)
            continue
        print(f"[{index + 1}/{args.count}] {sample_id}", flush=True)
        seed = args.seed + index * 1009
        try:
            prepared = generator.generate_prepared(
                sample_id, seed=seed, difficulty=difficulty
            )
            row = write_sample_artifacts(
                prepared.drawing,
                args.output,
                source_index=index,
                prepared=prepared,
            )
            completed[sample_id] = row
            progress["rows"] = [completed[key] for key in sorted(completed)]
        except Exception as exc:
            progress.setdefault("errors", []).append(
                {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            _write_json(progress_path, progress)
            raise
        _write_json(progress_path, progress)

    rows = [completed[key] for key in sorted(completed) if key.startswith("m5m6_")]
    summary = _summary(rows)
    contact_sheets = _contact_sheets(args.output, rows)
    _viewer(args.output, rows, contact_sheets)
    manifest = {
        "name": "PatentVec M5/M6 medium-hard visual review pilot",
        "preview_only": True,
        "full_generation_started": False,
        "awaiting_user_approval": True,
        "planned_full_data_root": str(FULL_DATA_ROOT),
        "count": len(rows),
        "requested_count": args.count,
        "seed": args.seed,
        "canvas": args.canvas,
        "all_quality_gates_passed": all(row["quality"]["accepted"] for row in rows),
        "summary": summary,
        "contact_sheets": contact_sheets,
        "samples": rows,
    }
    _write_json(args.output / "manifest.json", manifest)
    with (args.output / "manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "samples"}, indent=2))
    print(f"viewer: {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
