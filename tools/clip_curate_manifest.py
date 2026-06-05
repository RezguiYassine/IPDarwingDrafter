"""
Optional CLIP-based visual curation for PatentData training manifests.

The deterministic curation in ``build_training_manifest.py`` is deliberately
high precision, but hand-written shape rules still miss some semantic classes:
flowcharts, block/network diagrams, chemistry pages, and charts that happen to
look geometrically clean. This tool applies a small zero-shot CLIP classifier on
top of an existing manifest and writes a second, visually-vetted manifest.

It is optional because it requires ``open_clip_torch`` and downloads a pretrained
checkpoint the first time it runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image


DEFAULT_INPUT = Path("output/PatentData_clean12_gated/training_manifest_strict.csv")
DEFAULT_OUTPUT = Path("output/PatentData_clean12_gated/training_manifest_clip.csv")
DEFAULT_REJECTS = Path("output/PatentData_clean12_gated/training_manifest_clip_rejects.csv")


POSITIVE_PROMPTS = [
    ("mechanical_part", "a black and white patent drawing of a mechanical part"),
    ("physical_object", "a technical CAD line drawing of a physical object"),
    ("machine_component", "an engineering drawing of a machine component"),
    ("product_exploded", "an exploded view patent drawing of a product"),
]

NEGATIVE_PROMPTS = [
    ("chart_axes", "a scientific chart or graph with axes"),
    ("line_plot", "a line chart with axes and plotted curves"),
    ("xy_plot", "an x y coordinate graph with plotted data curves and tick labels"),
    ("frequency_plot", "a frequency response plot with gain and frequency axes"),
    ("bar_chart", "a bar chart or line plot"),
    ("timeline_chart", "a timeline chart or waveform timing diagram"),
    ("flowchart", "a flowchart with boxes and arrows"),
    ("block_diagram", "a block diagram or network architecture diagram"),
    ("chemistry", "a chemical molecule formula diagram"),
    ("table_text", "a table or form with text"),
    ("ui_screen", "a user interface screen diagram"),
]

HARD_NEGATIVE_THRESHOLDS = {
    # These classes are semantically bad LLM CAD targets even when the positive
    # prompt also scores high, which happens for clean plotted curves and tidy
    # table/grid layouts. Values are calibrated on the strict PatentData seed.
    "line_plot": 0.04,
    "xy_plot": 0.045,
    "frequency_plot": 0.035,
    "chart_axes": 0.06,
    "bar_chart": 0.12,
    "timeline_chart": 0.03,
    "table_text": 0.08,
    "ui_screen": 0.09,
    "block_diagram": 0.075,
    "flowchart": 0.075,
}

CLIP_FIELDS = (
    "clip_keep",
    "clip_reason",
    "clip_pos_label",
    "clip_pos_score",
    "clip_neg_label",
    "clip_neg_score",
    "clip_margin",
)


def _load_open_clip():
    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit(
            "open_clip_torch is required for CLIP curation. "
            "Install it with: pip install open_clip_torch"
        ) from exc
    return open_clip


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_fields = list(fields)
    for field in CLIP_FIELDS:
        if field not in out_fields:
            out_fields.append(field)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in out_fields})


def _image_from_row(row: dict[str, Any]) -> Image.Image:
    return Image.open(row["input_path"]).convert("RGB")


def curate(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    open_clip = _load_open_clip()
    fields, rows = _read_rows(args.input_csv)
    if not rows:
        return [], []

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, args.torch_threads))

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS
    labels = [label for label, _ in prompts]
    texts = [text for _, text in prompts]
    n_pos = len(POSITIVE_PROMPTS)
    with torch.no_grad():
        text_tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    batch_rows: list[dict[str, Any]] = []
    batch_imgs: list[torch.Tensor] = []

    def flush() -> None:
        if not batch_rows:
            return
        images = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu()
        for row, scores in zip(batch_rows, probs):
            pos_idx = int(torch.argmax(scores[:n_pos]).item())
            neg_rel = int(torch.argmax(scores[n_pos:]).item())
            neg_idx = n_pos + neg_rel
            pos_score = float(scores[pos_idx].item())
            neg_score = float(scores[neg_idx].item())
            margin = pos_score - neg_score
            row["clip_pos_label"] = labels[pos_idx]
            row["clip_pos_score"] = f"{pos_score:.6f}"
            row["clip_neg_label"] = labels[neg_idx]
            row["clip_neg_score"] = f"{neg_score:.6f}"
            row["clip_margin"] = f"{margin:.6f}"

            # Strict high-precision policy: strong negative visual evidence is
            # enough to reject even if deterministic rules passed.
            hard_threshold = HARD_NEGATIVE_THRESHOLDS.get(labels[neg_idx])
            if (
                args.enable_hard_negatives
                and hard_threshold is not None
                and neg_score >= hard_threshold
            ):
                row["clip_keep"] = "0"
                row["clip_reason"] = f"clip_hard_{labels[neg_idx]}"
                rejected.append(row)
            elif neg_score >= args.reject_neg_score and margin <= args.keep_margin:
                row["clip_keep"] = "0"
                row["clip_reason"] = f"clip_{labels[neg_idx]}"
                rejected.append(row)
            else:
                row["clip_keep"] = "1"
                row["clip_reason"] = "keep"
                kept.append(row)
        batch_rows.clear()
        batch_imgs.clear()

    for row in rows:
        try:
            batch_imgs.append(preprocess(_image_from_row(row)))
            batch_rows.append(row)
        except Exception as exc:
            row["clip_keep"] = "0"
            row["clip_reason"] = f"clip_image_error_{type(exc).__name__}"
            row["clip_pos_label"] = ""
            row["clip_pos_score"] = ""
            row["clip_neg_label"] = ""
            row["clip_neg_score"] = ""
            row["clip_margin"] = ""
            rejected.append(row)
            continue
        if len(batch_rows) >= args.batch_size:
            flush()
    flush()

    _write_rows(args.output_csv, fields, kept)
    _write_rows(args.rejects_csv, fields, rejected)
    return kept, rejected


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply optional CLIP zero-shot curation to a training manifest."
    )
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--rejects-csv", type=Path, default=DEFAULT_REJECTS)
    p.add_argument("--model", default="RN50")
    p.add_argument("--pretrained", default="openai")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--reject-neg-score", type=float, default=0.05)
    p.add_argument("--keep-margin", type=float, default=0.45)
    p.add_argument(
        "--disable-hard-negatives",
        dest="enable_hard_negatives",
        action="store_false",
        help="Disable calibrated class-specific hard-negative thresholds.",
    )
    p.set_defaults(enable_hard_negatives=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kept, rejected = curate(args)
    print(f"CLIP manifest: {args.output_csv}")
    print(f"CLIP rejects : {args.rejects_csv}")
    print(f"Kept         : {len(kept)}")
    print(f"Rejected     : {len(rejected)}")
    reasons: dict[str, int] = {}
    for row in rejected:
        reasons[row["clip_reason"]] = reasons.get(row["clip_reason"], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda item: item[1], reverse=True):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
