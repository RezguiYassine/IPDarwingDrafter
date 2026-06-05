"""
Build a high-precision SVG/DXF training manifest from a completed batch run.

PatentData's raw ``status='ok'`` rows are only "pipeline completed" examples.
For LLM training targets we also need to exclude semantic non-CAD pages that can
vectorize successfully but are bad targets: plots, charts, flowcharts, text-box
block diagrams, UI mockups, dense hatching, and very fragmented outputs.

This tool joins:

  - ``tools.batch_run`` SQLite metrics
  - the PatentData content-filter manifest
  - expected Stage 1/2/3/4 output artifact paths

It writes a CSV/JSONL manifest for training examples plus a reject CSV with
reason codes for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("output/PatentData_clean12_gated/results.db")
DEFAULT_RUN_OUTPUT = Path("output/PatentData_clean12_gated")
DEFAULT_FILTER_MANIFEST = Path("output/PatentData/filter_manifest_clean12.csv")
DEFAULT_OUTPUT_CSV = Path("output/PatentData_clean12_gated/training_manifest_strict.csv")
DEFAULT_REJECTS_CSV = Path("output/PatentData_clean12_gated/training_manifest_strict_rejects.csv")


FILTER_NUMERIC_COLUMNS = (
    "text_density",
    "n_cc",
    "large_cc_frac",
    "long_lines",
    "line_hv_ratio",
    "line_diag_ratio",
    "line_median_length",
    "skel_density",
    "skel_n_cc",
    "skel_tiny_cc_frac",
    "skel_median_cc_area",
)

MANIFEST_FIELDS = (
    "patent_id",
    "sketch_id",
    "input_path",
    "cleaned_skeleton_path",
    "graph_path",
    "primitives_json_path",
    "svg_path",
    "dxf_path",
    "filter_reason",
    "curation_reason",
    "total_time",
    "s1_quality",
    "s2_n_edges",
    "s2_median_edge_len",
    "s2_micro_edge_ratio",
    "s2_short_edge_ratio",
    "s3_n_primitives",
    "s3_mean_conf",
    "s3_low_conf_ratio",
    "text_density",
    "n_cc",
    "large_cc_frac",
    "long_lines",
    "line_hv_ratio",
    "line_diag_ratio",
    "skel_n_cc",
    "skel_tiny_cc_frac",
    "skel_median_cc_area",
)

REJECT_FIELDS = MANIFEST_FIELDS + (
    "status",
    "error",
)


def _float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _load_filter_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            for col in FILTER_NUMERIC_COLUMNS:
                row[col] = _float(row.get(col), default=0.0)
            rows[(row["patent"], row["filename"])] = row
    return rows


def _artifact_paths(run_output: Path, patent_id: str, sketch_id: str) -> dict[str, str]:
    base = run_output / patent_id
    return {
        "cleaned_skeleton_path": str(base / "cleaned" / f"{sketch_id}_skeleton.png"),
        "graph_path": str(base / "graphs" / f"{sketch_id}_graph.json"),
        "primitives_json_path": str(base / "primitives" / f"{sketch_id}_primitives.json"),
        "svg_path": str(base / "vectors" / f"{sketch_id}.svg"),
        "dxf_path": str(base / "vectors" / f"{sketch_id}.dxf"),
    }


def _missing_artifact_reason(row: dict[str, Any]) -> str | None:
    for key in ("cleaned_skeleton_path", "graph_path", "primitives_json_path", "svg_path", "dxf_path"):
        if not Path(row[key]).exists():
            return f"missing_{key}"
    return None


def _content_reason(feats: dict[str, Any]) -> str | None:
    """Return a semantic non-CAD reject reason from filter-manifest features."""
    dens = _float(feats.get("text_density"))
    n_cc = _int(feats.get("n_cc"))
    large = _float(feats.get("large_cc_frac"))
    lines = _int(feats.get("long_lines"))
    hv = _float(feats.get("line_hv_ratio"))
    diag = _float(feats.get("line_diag_ratio"))
    skel_cc = _int(feats.get("skel_n_cc"))
    tiny = _float(feats.get("skel_tiny_cc_frac"))
    med = _float(feats.get("skel_median_cc_area"))

    # Sparse rectangular boxes with mostly H/V strokes are usually UI/block
    # diagrams. They vectorize cleanly but teach the model bad CAD targets.
    if (
        n_cc <= 110
        and lines >= 5
        and large >= 0.35
        and hv >= 0.90
        and diag <= 0.04
        and med >= 28.0
    ):
        return "sparse_hv_block_or_ui"

    # Chemical formula grids can appear in ordinary _F buckets. They have many
    # ring components, very few true engineering long lines, and a high H/V
    # ratio from bonds/text baselines.
    if (
        80 <= n_cc <= 220
        and lines <= 15
        and large >= 0.55
        and hv >= 0.80
        and diag <= 0.06
        and tiny <= 0.20
        and med >= 14.0
    ):
        return "chemistry_formula_grid"

    if (
        90 <= n_cc <= 280
        and lines <= 12
        and large >= 0.55
        and med >= 10.0
        and dens <= 0.050
    ):
        return "sparse_formula_or_symbol_grid"

    # Flowcharts, text boxes, and architecture diagrams: many boxes, very H/V,
    # little diagonal geometry, moderate text component load.
    if (
        n_cc >= 80
        and lines >= 40
        and large >= 0.50
        and hv >= 0.84
        and diag <= 0.13
        and dens >= 0.015
    ):
        return "orthogonal_block_flow_or_chart"

    if (
        n_cc <= 130
        and lines >= 20
        and large >= 0.70
        and hv >= 0.80
        and diag <= 0.20
        and med <= 18.0
        and dens <= 0.050
    ):
        return "sparse_hv_block_or_network_diagram"

    if (
        n_cc <= 130
        and lines >= 60
        and large >= 0.80
        and hv >= 0.50
        and diag <= 0.35
        and med <= 15.0
        and dens <= 0.050
    ):
        return "sparse_ui_or_device_block_diagram"

    if (
        n_cc <= 120
        and lines >= 20
        and large >= 0.80
        and hv >= 0.60
        and diag <= 0.10
        and med >= 15.0
        and dens <= 0.040
    ):
        return "waveform_or_timing_chart"

    if (
        n_cc <= 80
        and lines >= 30
        and large >= 0.80
        and hv >= 0.60
        and diag <= 0.35
        and med >= 12.0
        and dens <= 0.050
    ):
        return "sparse_flowchart_or_decision_tree"

    # Mixed block/flow diagrams often have curved arrows or slanted connectors,
    # so they escape the mostly-H/V rule above. They still have medium component
    # counts, long connectors, and large rectangular/text-box components.
    if (
        120 <= n_cc <= 260
        and lines >= 35
        and large >= 0.45
        and med >= 22.0
        and 0.14 <= diag <= 0.65
        and dens <= 0.055
    ):
        return "mixed_flowchart_or_timing_diagram"

    # Moderate-density grid/axis diagrams: plots with labels are often not
    # "dense" in pixel space but have many long H/V/grid strokes.
    if (
        n_cc >= 120
        and lines >= 45
        and large >= 0.64
        and hv >= 0.60
        and diag <= 0.26
        and dens <= 0.055
    ):
        return "moderate_axis_plot_or_grid"

    if (
        220 <= n_cc <= 360
        and lines >= 25
        and large >= 0.58
        and hv >= 0.65
        and diag <= 0.35
        and med <= 12.0
        and dens <= 0.055
    ):
        return "flow_schematic_component_pattern"

    if (
        n_cc >= 300
        and lines >= 35
        and large >= 0.60
        and hv >= 0.55
        and diag <= 0.30
        and med <= 10.0
        and dens >= 0.045
    ):
        return "dense_block_network_diagram"

    if (
        180 <= n_cc <= 300
        and lines >= 50
        and large >= 0.70
        and 0.25 <= diag <= 0.45
        and med >= 25.0
        and dens >= 0.055
    ):
        return "dense_flowchart_text_boxes"

    if (
        n_cc >= 300
        and lines >= 50
        and diag >= 0.55
        and med <= 12.0
        and dens <= 0.060
    ):
        return "diagonal_lattice_network_diagram"

    # Bar charts, grid plots, and hatching fields. Hatching is a real drafting
    # convention, but the current geometry-only exporter turns it into many low
    # value short primitives; keep it out until hatch-aware export exists.
    if dens >= 0.055 and (
        (lines >= 120 and hv >= 0.75 and diag <= 0.08)
        or (n_cc >= 150 and large >= 0.75 and hv >= 0.80 and diag <= 0.10)
    ):
        return "dense_hv_chart_or_hatching"

    if dens >= 0.080 and lines >= 80 and diag <= 0.08:
        return "dense_bar_chart_or_table"

    if dens >= 0.070 and n_cc >= 100 and lines >= 20:
        return "dense_chart_hatching_or_barplot"

    # Text/legend-heavy scientific plots and circular label maps tend to have
    # hundreds of tiny connected components and near-zero median skeleton size.
    if (
        n_cc >= 500
        and skel_cc >= 500
        and tiny >= 0.55
        and med <= 3.5
    ):
        return "label_heavy_text_or_plot"

    if (
        n_cc >= 350
        and skel_cc >= 350
        and tiny >= 0.50
        and med <= 8.0
        and lines >= 30
        and dens <= 0.065
    ):
        return "legend_heavy_chart_or_plot"

    if (
        n_cc >= 450
        and tiny >= 0.25
        and med <= 12.0
        and dens <= 0.080
    ):
        return "dense_label_or_plot_page"

    if (
        n_cc >= 100
        and lines >= 20
        and tiny >= 0.25
        and med <= 12.0
        and dens <= 0.060
        and (diag >= 0.25 or hv >= 0.55)
    ):
        return "axis_or_legend_component_pattern"

    if (
        n_cc >= 500
        and large <= 0.50
        and lines >= 30
        and tiny >= 0.35
        and diag >= 0.45
        and dens <= 0.060
    ):
        return "dense_diagonal_scientific_plot"

    if (
        dens >= 0.065
        and lines >= 70
        and large >= 0.78
        and (diag >= 0.60 or hv >= 0.75)
    ):
        return "dense_axis_chart_or_hatching"

    # Sparse single/multi-axis scientific curves: few components, long smooth
    # curves, strong axis geometry. This catches plots that are too sparse for
    # the pre-vectorization filter's dense text/chart rules.
    if (
        n_cc <= 120
        and lines >= 20
        and large >= 0.75
        and med >= 30.0
        and (hv >= 0.55 or diag <= 0.30)
    ):
        return "sparse_axis_plot"

    # Multi-panel plot pages with legends: not necessarily very H/V, but they
    # have many tiny glyph components and low median skeleton component area.
    if (
        n_cc >= 300
        and dens <= 0.075
        and large <= 0.85
        and tiny >= 0.55
        and med <= 6.0
        and diag >= 0.25
    ):
        return "multi_panel_plot_or_legend"

    if (
        170 <= n_cc <= 280
        and lines >= 12
        and large >= 0.60
        and tiny >= 0.30
        and med <= 12.0
        and diag >= 0.50
        and dens <= 0.055
    ):
        return "sparse_diagonal_plot_or_legend"

    if (
        100 <= n_cc <= 230
        and lines >= 10
        and large >= 0.50
        and tiny >= 0.35
        and med <= 20.0
        and dens <= 0.055
        and (diag >= 0.25 or hv >= 0.55)
    ):
        return "small_axis_plot_or_legend"

    if (
        n_cc <= 100
        and lines >= 25
        and large >= 0.80
        and hv >= 0.80
        and med <= 20.0
        and dens <= 0.060
    ):
        return "sparse_hv_profile_or_plot"

    if (
        n_cc <= 100
        and lines >= 40
        and large >= 0.75
        and diag >= 0.35
        and med <= 25.0
        and dens <= 0.045
    ):
        return "sparse_curve_plot_low_components"

    if (
        dens >= 0.065
        and n_cc <= 100
        and large >= 0.90
        and lines >= 20
    ):
        return "dense_small_chart_or_symbol_page"

    if (
        n_cc <= 80
        and lines >= 8
        and hv >= 0.95
        and diag <= 0.05
        and med >= 15.0
        and dens <= 0.050
    ):
        return "sparse_hv_table_or_timeline"

    if (
        n_cc <= 120
        and lines >= 40
        and large >= 0.90
        and tiny >= 0.40
        and med <= 5.0
        and diag >= 0.40
        and dens <= 0.070
    ):
        return "single_curve_plot_high_large_frac"

    if (
        120 <= n_cc <= 220
        and lines >= 18
        and large >= 0.55
        and hv >= 0.80
        and med <= 18.0
        and dens <= 0.045
    ):
        return "hv_component_table_or_diagram"

    if (
        140 <= n_cc <= 300
        and lines >= 6
        and large <= 0.55
        and tiny >= 0.40
        and med <= 10.0
        and dens <= 0.040
        and (hv >= 0.45 or diag <= 0.10)
    ):
        return "sparse_low_large_plot_or_legend"

    if (
        120 <= n_cc <= 220
        and lines >= 8
        and large <= 0.60
        and hv >= 0.60
        and diag <= 0.30
        and med <= 12.0
        and dens <= 0.040
    ):
        return "moderate_sparse_axis_plot"

    if (
        240 <= n_cc <= 360
        and large <= 0.65
        and lines <= 18
        and tiny >= 0.40
        and med <= 12.0
        and diag >= 0.45
        and dens <= 0.040
    ):
        return "sparse_scientific_diagram"

    if (
        n_cc <= 130
        and lines >= 25
        and large >= 0.65
        and diag >= 0.65
        and med <= 14.0
        and dens <= 0.040
    ):
        return "sparse_diagonal_axis_plot"

    # Sparse axis/curve plots and timing diagrams. These are hard for the
    # pre-vectorization filter because they contain clean long strokes, but they
    # are semantically charts rather than CAD parts.
    if (
        n_cc <= 180
        and lines >= 10
        and large >= 0.60
        and med >= 20.0
        and dens <= 0.055
        and (hv >= 0.48 or diag >= 0.30)
    ):
        return "sparse_curve_axis_or_timing_plot"

    # Axis diagrams with legend/tick text: component count is higher than the
    # sparse-curve case, but median skeleton components remain small.
    if (
        120 <= n_cc <= 260
        and lines >= 20
        and large >= 0.62
        and tiny >= 0.12
        and med <= 30.0
        and dens <= 0.055
        and (hv >= 0.60 or diag <= 0.25)
    ):
        return "text_axis_or_legend_plot"

    return None


def _curation_reason(row: dict[str, Any], args: argparse.Namespace) -> str:
    status = row.get("status")
    if status != "ok":
        return f"pipeline_{status}"

    if row.get("filter_label") != "drawing":
        return f"filter_{row.get('filter_label') or 'missing'}"
    if args.reject_ambiguous_filter and row.get("filter_reason") != "long_engineering_lines":
        return f"filter_{row.get('filter_reason') or 'missing'}"

    missing = _missing_artifact_reason(row)
    if missing:
        return missing

    content = _content_reason(row)
    if content:
        return content

    if _float(row.get("total_time")) >= args.max_total_time:
        return "slow_outlier"
    if _int(row.get("s2_n_edges")) < args.min_edges:
        return "too_sparse"
    if _int(row.get("s2_n_edges")) >= args.max_edges:
        return "too_complex"
    if _float(row.get("s2_micro_edge_ratio")) >= args.max_micro_edge_ratio:
        return "too_fragmented_micro_edges"
    if _float(row.get("s2_short_edge_ratio")) >= args.max_short_edge_ratio:
        return "too_fragmented_short_edges"
    if _float(row.get("s3_low_conf_ratio")) >= args.max_low_conf_ratio:
        return "too_many_low_conf_primitives"
    if _float(row.get("s3_mean_conf")) < args.min_mean_confidence:
        return "low_mean_primitive_confidence"

    return "keep"


def _format_row(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: row.get(field) for field in fields}


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    filter_rows = _load_filter_manifest(args.filter_manifest)
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    keep: list[dict[str, Any]] = []
    reject: list[dict[str, Any]] = []
    reasons: Counter = Counter()

    for db_row in con.execute("select * from results order by patent_id, sketch_id"):
        row = dict(db_row)
        input_name = Path(row["input_path"]).name
        frow = filter_rows.get((row["patent_id"], input_name), {})
        row["filter_label"] = frow.get("label")
        row["filter_reason"] = frow.get("reason")
        for col in FILTER_NUMERIC_COLUMNS:
            row[col] = frow.get(col)
        row.update(_artifact_paths(args.run_output, row["patent_id"], row["sketch_id"]))

        reason = _curation_reason(row, args)
        row["curation_reason"] = reason
        reasons[reason] += 1
        if reason == "keep":
            keep.append(row)
        else:
            reject.append(row)

    con.close()
    return keep, reject, reasons


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_format_row(row, fields))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(_format_row(row, MANIFEST_FIELDS), sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a strict PatentData SVG/DXF training manifest from batch results."
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--run-output", type=Path, default=DEFAULT_RUN_OUTPUT)
    p.add_argument("--filter-manifest", type=Path, default=DEFAULT_FILTER_MANIFEST)
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    p.add_argument("--output-jsonl", type=Path, default=None)
    p.add_argument("--rejects-csv", type=Path, default=DEFAULT_REJECTS_CSV)
    p.add_argument("--max-total-time", type=float, default=10.0)
    p.add_argument("--min-edges", type=int, default=50)
    p.add_argument("--max-edges", type=int, default=700)
    p.add_argument("--max-micro-edge-ratio", type=float, default=0.28)
    p.add_argument("--max-short-edge-ratio", type=float, default=0.70)
    p.add_argument("--max-low-conf-ratio", type=float, default=0.22)
    p.add_argument("--min-mean-confidence", type=float, default=0.67)
    p.add_argument("--keep-ambiguous-filter", action="store_true",
                   help="Keep rows whose original filter reason was sparse_ambiguous_drawing.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    args.reject_ambiguous_filter = not args.keep_ambiguous_filter

    keep, reject, reasons = build(args)
    _write_csv(args.output_csv, keep, MANIFEST_FIELDS)
    _write_csv(args.rejects_csv, reject, REJECT_FIELDS)
    if args.output_jsonl:
        _write_jsonl(args.output_jsonl, keep)

    print(f"Training manifest: {args.output_csv}")
    if args.output_jsonl:
        print(f"Training JSONL    : {args.output_jsonl}")
    print(f"Reject audit CSV  : {args.rejects_csv}")
    print(f"Kept              : {len(keep)}")
    print(f"Rejected          : {len(reject)}")
    print("Reason breakdown:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
