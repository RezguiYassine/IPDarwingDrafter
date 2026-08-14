from __future__ import annotations

import argparse
import html
import json
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from syntheticData.patentvec.render import (
    degrade_patent_scan,
    make_triptych,
    render_clean,
    semantic_preview,
)
from syntheticData.patentvec.schema import CanonicalDrawing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a visual audit viewer from compact PatentVec shards."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--per-sheet", type=int, default=6)
    parser.add_argument(
        "--stratified-per-difficulty",
        type=int,
        default=0,
        help=("select this many evenly spaced samples from every difficulty and "
              "re-render previews when no audit payload was stored"),
    )
    return parser.parse_args()


def _stratified_rows(rows: list[dict], per_difficulty: int) -> list[dict]:
    selected = []
    difficulties = ("medium", "hard", "very_hard")
    for difficulty in difficulties:
        candidates = [row for row in rows if row["difficulty"] == difficulty]
        count = min(per_difficulty, len(candidates))
        if not count:
            continue
        if count == 1:
            indices = [len(candidates) // 2]
        else:
            indices = [
                round(position * (len(candidates) - 1) / (count - 1))
                for position in range(count)
            ]
        selected.extend(candidates[index] for index in indices)
    return selected


def _extract_previews(dataset: Path, output: Path, rows: list[dict]) -> list[dict]:
    archives: dict[str, tarfile.TarFile] = {}
    result = []
    try:
        for row in rows:
            archive_name = row["archive"]
            archive = archives.get(archive_name)
            if archive is None:
                archive = tarfile.open(dataset / archive_name, mode="r")
                archives[archive_name] = archive
            filename = f"{row['sample_id']}.png"
            member_name = f"{row['member_prefix']}/preview.png"
            try:
                member = archive.extractfile(member_name)
            except KeyError:
                member = None
            if member is not None:
                (output / filename).write_bytes(member.read())
                preview_source = "stored"
            else:
                sample = archive.extractfile(
                    f"{row['member_prefix']}/sample.json"
                )
                if sample is None:
                    raise ValueError(f"missing sample.json for {row['sample_id']}")
                drawing = CanonicalDrawing.from_dict(json.load(sample))
                clean = render_clean(drawing)
                degraded = degrade_patent_scan(clean, seed=drawing.seed + 701)
                semantic = semantic_preview(drawing)
                preview = make_triptych(
                    clean, degraded, semantic, drawing.sample_id
                )
                preview.save(output / filename, format="PNG", optimize=True)
                preview_source = "rerendered"
            result.append(
                {**row, "audit_preview": filename, "preview_source": preview_source}
            )
    finally:
        for archive in archives.values():
            archive.close()
    return result


def _contact_sheets(output: Path, rows: list[dict], per_sheet: int) -> list[str]:
    names = []
    for sheet_index, offset in enumerate(range(0, len(rows), per_sheet), start=1):
        chunk = rows[offset : offset + per_sheet]
        thumbnails = []
        for row in chunk:
            image = Image.open(output / row["audit_preview"]).convert("RGB")
            image.thumbnail((1400, 470), Image.Resampling.LANCZOS)
            thumbnails.append((row, image.copy()))
        columns = 2
        rows_per_sheet = (len(thumbnails) + columns - 1) // columns
        cell_width = 1420
        cell_height = 520
        sheet = Image.new(
            "RGB", (columns * cell_width, rows_per_sheet * cell_height), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for index, (row, image) in enumerate(thumbnails):
            x = (index % columns) * cell_width + 10
            y = (index // columns) * cell_height + 38
            sheet.paste(image, (x, y))
            complexity = row["quality"]["complexity"]
            label = (
                f"{row['sample_id']}  C={complexity['source_component_count']} "
                f"R={complexity['interaction_count']} "
                f"J={complexity['structural_junction_count']} "
                f"cycles={complexity['cycle_rank']}"
            )
            draw.text((x, y - 24), label, fill="black")
        name = f"CONTACT_SHEET_{sheet_index:02d}.jpg"
        sheet.save(output / name, quality=90, optimize=True)
        names.append(name)
    return names


def _html(rows: list[dict], sheets: list[str]) -> str:
    items = []
    for row in rows:
        complexity = row["quality"]["complexity"]
        edges = row["training_targets"]["free2cad_edges"]
        items.append(
            f"""
            <article>
              <h2>{html.escape(row['sample_id'])}</h2>
              <p>{complexity['source_component_count']} components, {complexity['interaction_count']} interactions,
                 {complexity['structural_junction_count']} junctions, {complexity['cycle_rank']} cycles,
                 {edges['polyline']} polylines</p>
              <img src="{html.escape(row['audit_preview'])}" alt="{html.escape(row['sample_id'])}">
            </article>
            """
        )
    sheet_links = " ".join(
        f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in sheets
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PatentVec compact audit</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #171717; background: #f3f4f6; }}
    header {{ padding: 18px 24px; background: white; border-bottom: 1px solid #d4d4d4; position: sticky; top: 0; z-index: 1; }}
    header h1 {{ margin: 0 0 8px; font-size: 20px; }}
    header a {{ color: #075985; margin-right: 14px; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 20px; display: grid; gap: 18px; }}
    article {{ background: white; border: 1px solid #d4d4d4; border-radius: 6px; overflow: hidden; }}
    article h2 {{ margin: 14px 16px 4px; font-size: 16px; }}
    article p {{ margin: 0 16px 12px; color: #525252; }}
    article img {{ display: block; width: 100%; height: auto; border-top: 1px solid #e5e5e5; }}
  </style>
</head>
<body>
  <header><h1>PatentVec compact audit</h1><nav>{sheet_links}</nav></header>
  <main>{''.join(items)}</main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    if args.per_sheet < 1:
        raise SystemExit("per-sheet must be positive")
    if args.stratified_per_difficulty < 0:
        raise SystemExit("stratified-per-difficulty must be non-negative")
    output = args.output or args.dataset / "audit"
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (args.dataset / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    selected_rows = (
        _stratified_rows(rows, args.stratified_per_difficulty)
        if args.stratified_per_difficulty
        else [row for row in rows if row.get("audit")]
    )
    selected = _extract_previews(args.dataset, output, selected_rows)
    if not selected:
        raise SystemExit("dataset contains no audit samples")
    sheets = _contact_sheets(output, selected, args.per_sheet)
    (output / "index.html").write_text(_html(selected, sheets))
    print(
        json.dumps(
            {
                "audit_samples": len(selected),
                "difficulty_counts": dict(
                    Counter(row["difficulty"] for row in selected)
                ),
                "preview_sources": dict(
                    Counter(row["preview_source"] for row in selected)
                ),
                "contact_sheets": sheets,
                "viewer": str(output / "index.html"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
