# IP DrawingDrafter — AP3 Vectorization Pipeline

> Hand-drawn engineering sketch → editable vector files (SVG / DXF).
> HAW Landshut · IP DrawingDrafter project · AP3.

This repository implements the Stage 0 + four-stage vectorization pipeline of
the IP DrawingDrafter project. It takes a raster image of a hand-drawn or patent
engineering sketch as input and produces clean, ISO 128–styled vector files
(SVG and DXF) that can be opened in any CAD application.

```
   raster PNG/TIF          no-reference raster       cleaned + skeleton
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐         ┌──────────────┐           ┌──────────────────┐
│   Stage 0    │ ──────▶ │   Stage 1    │ ────────▶ │     Stage 2      │
│ References   │         │ Preprocessing│           │ Stroke Extraction│
└──────────────┘         └──────────────┘           └──────────────────┘
       │                                                     │
       │ reference JSON                                      ▼
       │                                            ┌──────────────────┐
       └──────────────────────────────────────────▶ │     Stage 3      │
                                                    │ Primitive Fitting│
                                                    └──────────────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────────┐
                                                    │     Stage 4      │
                                                    │ Export + Reinject│
                                                    │  (SVG, DXF)      │
                                                    └──────────────────┘
```

| Stage | In                  | Out                         | Tech                       |
|------:|---------------------|-----------------------------|----------------------------|
| 0     | Raw patent TIF/PNG   | Reference-free PNG + reference JSON + mask/crops | Conservative connected-component + Hough leader detector; removes reference numerals/help lines before skeletonisation; reinjects annotations during export |
| 1     | Raw raster (PNG/TIF) | Cleaned image + 1-px skeleton + stroke-width estimate | SketchCleanNet (or classical fallback) |
| 2     | 1-px skeleton       | Stroke graph + side-layer hachures (JSON) | **Vectorised CN path** (NumPy ring-slice shifts, ~1 000× vs Python loop); incompatible Puhachov weights are rejected instead of silently ignored; parallel-edge walk; **graph de-fragmentation** (spur prune + degree-2 dissolve + collinear through-merge); patent hachure extraction; fragmentation metrics/gates; deterministic scale metadata |
| 3     | Stroke graph        | Geometric primitives (JSON) | RANSAC cascade (line / circle / arc / ellipse / **polygon** / polyline); primitives are rescaled back to the original image frame after Stage 2 downsampling; removed hachures are reinserted as `style: "hachure"` primitives and excluded from main-geometry confidence gates |
| 4     | Geometric primitives + optional annotations | SVG and/or DXF              | `svgwrite`, `ezdxf` (ISO 128 layered, including `HACHURE`); polygon primitive rendered as closed shape; Stage 0 references are reinserted as SVG crop overlays + leader lines and DXF leader geometry |

---

## Quick start

### 1. Clone & install

```bash
git clone <this-repo-url> Vectorization
cd Vectorization
```

**Linux / macOS:**

```bash
bash setup.sh
source .venv/bin/activate
```

**Windows (Command Prompt):**

```bat
setup.bat
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
.\setup.bat
.venv\Scripts\Activate.ps1
```

The setup script creates a virtualenv, installs requirements, and downloads
the one large model weight (`sketchcleannet.pth`, 124 MB) that doesn't ship
in the repository — see [Model weights](#model-weights) below. It is
idempotent: safe to re-run.

### 2. Run the pipeline on a sample

```bash
# Optional Stage 0 — remove patent reference numerals/leaders before Stage 1
python stage0_handling_references/stage0_handle_references.py \
    data/PatentData/ReorganisedData/EP1705282B1/EP1705282B1_F0002.tif \
    --output output/EP1705282B1 --id F0002

# Stage 1 — clean + skeletonize
python stage1_preprocessing/stage1_preprocess.py \
    data/samples/Picture1.png

# Stage 2 — build stroke graph from the skeleton
python stage2_strokeextraction/stage2_stroke_extract.py \
    output/cleaned/Picture1_skeleton.png

# Stage 3 — fit geometric primitives via RANSAC
python stage3_primitivesfitting/stage3_primitive_fit.py \
    output/graphs/Picture1_skeleton_graph.json

# Stage 4 — export to SVG + DXF (ISO 128 patent style)
python stage4_export/stage4_export.py \
    output/primitives/Picture1_skeleton_primitives.json \
    --format both --dxf-mode patent
```

Everything lands under `output/`:

```
output/
├── references/   ← Stage 0 (reference-free PNG + JSON + mask + crops)
├── cleaned/      ← Stage 1 (cleaned PNG + skeleton PNG)
├── graphs/       ← Stage 2 (stroke graph JSON)
├── primitives/   ← Stage 3 (primitives JSON)
└── vectors/      ← Stage 4 (final .svg / .dxf)
```

---

## Repository layout

```
Vectorization/
├── README.md                        ← this file
├── LICENSE                          ← Apache 2.0
├── config.yaml                      ← single source of truth for all stages
├── requirements.txt
├── setup.sh                         ← one-shot install + weight download
│
├── stage0_handling_references/
│   ├── stage0_handle_references.py  ← patent reference detection/removal
│   └── __init__.py
│
├── stage1_preprocessing/
│   ├── stage1_preprocess.py
│   └── research/                    ← optional: re-train SketchCleanNet
│       ├── train_sketchcleannet.py
│       └── README.md
│
├── stage2_strokeextraction/
│   ├── stage2_stroke_extract.py
│   ├── visualize_graph.py
│   └── …
│
├── stage3_primitivesfitting/
│   ├── stage3_primitive_fit.py      ← production (RANSAC)
│   └── research/                    ← optional: Free2CAD experiment (retired)
│       ├── stage3_primitive_fit_free2cad.py
│       ├── train_free2cad_v3.py
│       ├── generate_sketches_v3.py
│       └── README.md
│
├── stage4_export/
│   └── stage4_export.py
│
├── docs/                            ← per-stage architectural notes
│   ├── pipeline_overview.md
│   ├── stage1_preprocess.md
│   ├── stage2_stroke_extract.md
│   ├── stage3_primitive_fit.md
│   ├── stage4_export.md
│   └── archive/                     ← historical: Free2CAD investigation
│
├── tools/
│   ├── batch_run.py                 ← batch driver for PatentData corpus (SQLite results)
│   ├── filter_patent_data.py        ← content classifier: keep drawings, discard chemistry/text
│   ├── build_training_manifest.py   ← strict SVG/DXF training manifest builder
│   ├── clip_curate_manifest.py      ← optional zero-shot CLIP visual curation layer
│   ├── make_manifest_contact_sheet.py ← render visual audit sheets from any manifest
│   ├── d2c_eval.py                  ← Drawing2CAD ground-truth eval harness
│   ├── results_db.py                ← SQLite schema shared by batch_run
│   └── __init__.py
│
├── data/
│   ├── samples/                     ← a couple of example PNGs
│   ├── Drawing2CAD/                 ← D2C dataset (svg_raw/, svg_vec/, cad_vec/, split JSON)
│   └── PatentData/                  ← partner patent TIF corpus (gitignored)
│
├── models/                          ← weights (some shipped, some downloaded)
│   └── README.md
│
└── output/                          ← (gitignored) all pipeline outputs
```

The stage scripts are intentionally **decoupled**: each one reads its
input from disk, writes its output to disk, and has no Python imports
between stages. You can run any stage standalone, swap a stage's
implementation, or insert a new stage between two existing ones.

---

## Model weights

Three model weights are referenced by the pipeline:

| Weight                       | Size  | Used by              | Status                      |
|------------------------------|------:|----------------------|-----------------------------|
| `puhachov_d2c.pth`           | 30 MB | Stage 2 *(default)*  | keypoint CNN trained on Drawing2CAD (val peak-F1 0.83); shipped and used by the default **fusion** seeding. Set `puhachov.weights: ""` to force the pure-CN path |
| `free2cad_v3_best.pth`       | 3 MB  | Stage 3 *(research)* | shipped in `models/`        |
| `sketchcleannet.pth`         | 124 MB | Stage 1             | **must be downloaded** (>100 MB GitHub limit) |

`setup.sh` will print the OneDrive folder URL where `sketchcleannet.pth`
is hosted. If you skip the download, Stage 1 automatically falls back to
its **classical cleaning mode** (Otsu + adaptive threshold + morphology) —
the pipeline still produces valid output, with somewhat noisier
skeletonization on photographed or shaded sketches.

See [`models/README.md`](models/README.md) for details.

---

## Configuration

All production stages read from the single [`config.yaml`](config.yaml) at the
project root. All paths in it are relative to the project root, so the
config works on any clone without editing.

The most common knobs:

| Key                                | Default | Effect |
|------------------------------------|--------:|--------|
| `sketchcleannet.weights`           | `models/sketchcleannet.pth` | empty `""` ⇒ force classical cleaning mode |
| `sketchcleannet.device`            | `cpu`   | `cuda` for GPU |
| `puhachov.weights`                 | `models/puhachov_d2c.pth` | Default fusion seeding (CN topology + CNN corners). `""` ⇒ pure-CN path (recommended for large PatentData batches, where fusion is neutral) |
| `puhachov.fusion`                  | `true`  | Fuse CN endpoints/junctions with CNN corners. `false` ⇒ raw CNN keypoints (worse than both) |
| `puhachov.device`                  | `cpu`   | `cuda` for GPU if a compatible detector is supplied later |
| `stage0.enabled`                   | `true`  | PatentData batch mode removes reference numerals/help lines before Stage 1 and reinjects them during Stage 4 |
| `stage0.max_iterations`            | `10`    | Bounded repeated Stage 0 pass; catches labels that become detectable only after earlier removals |
| `stage0.require_leader`            | `true`  | Prefer text-like clusters attached to nearby leader/help lines; unleadered and caption text use stricter shape filters |
| `stage0.remove_unleadered_labels`  | `true`  | Remove compact numeric/text clusters without detected leaders, including margin-only single-character labels |
| `stage0.remove_figure_labels`      | `true`  | Remove compact `FIG`/panel caption text, including captions between subfigures |
| `stage0.max_removed_ink_ratio`     | `0.70`  | Guard against masks that erase too much ink in one pass |
| `stage0.max_total_removed_ink_ratio` | `0.82` | Total ink-removal guard across iterative Stage 0 passes |
| `stage0.leader_min_support_ratio`  | `0.50`  | Reject broken/dashed Hough segments that are unlikely to be solid callout leaders |
| `stage0.repair_close_kernel`       | `7`     | Local gap repair after reference removal; reconnects small true-stroke breaks inside the removal mask |
| `stage2.max_input_resolution`      | `1000`  | Skeleton images with long edge > this are downsampled before Stage 2. Prevents CNN over-segmentation on large patent TIFs (2000–2700 px). Set to `0` to disable. |
| `stage2.isolation_threshold`       | `0.30`  | Flag sketch if > this fraction of foreground pixels are unreached by any extracted stroke. Calibrated for patent TIF scan noise (p75 isolation ≈ 0.16). |
| `stage2.nms_reference_resolution`  | `512`   | Training resolution of the Puhachov model; NMS radius scales as `nms_radius × max(H,W) / this value` on larger inputs (0 = fixed radius) |
| `stage2.spline_overshoot_limit`    | `5.0`   | Max px a B-spline may exceed the raw pixel bbox; prevents scipy end-effect oscillations |
| `stage2.min_closed_loop_pixels`    | `80`    | Closed loops shorter than this are treated as noise and removed before Stage 3 |
| `stage2.remove_hachures`           | `true`  | Patent mode: extract dense short hatch clusters as a side layer so they do not split long outline strokes |
| `stage2.hachure_trigger_micro_edge_ratio` | `0.20` | Run hatch cleanup only when the simplified graph still has many micro-fragments |
| `stage2.hachure_trigger_short_edge_ratio` | `0.80` | Secondary hatch trigger for graphs dominated by short open edges |
| `stage1.quality_threshold`         | `0.70`  | Sketches below this skeleton quality are flagged for review |
| `stage3.confidence_threshold`      | `0.60`  | Primitives below this are flagged for review |
| `stage3.confidence_threshold_after_hachure` | `0.50` | Hachure-heavy graphs use this main-geometry threshold because easy hatch-line primitives are no longer part of the confidence average |
| `pipeline.quality_gates.enabled`   | `true`  | Batch mode stops bad examples before export using Stage 1/2/3 metrics |
| `pipeline.quality_gates.max_low_conf_ratio_after_hachure` | `0.65` | Hachure-heavy graphs use this relaxed low-confidence ratio after the hatch side layer is extracted |

---

## Per-stage documentation

Each stage has its own architecture document:

- **Pipeline overview** — [`docs/pipeline_overview.md`](docs/pipeline_overview.md)
- **Stage 0** — [`stage0_handling_references/stage0_handle_references.py`](stage0_handling_references/stage0_handle_references.py)
- **Stage 1** — [`docs/stage1_preprocess.md`](docs/stage1_preprocess.md)
- **Stage 2** — [`docs/stage2_stroke_extract.md`](docs/stage2_stroke_extract.md)
- **Stage 3** — [`docs/stage3_primitive_fit.md`](docs/stage3_primitive_fit.md)
- **Stage 4** — [`docs/stage4_export.md`](docs/stage4_export.md)

Historical notes (Free2CAD experiment, training methodology, results) live
in [`docs/archive/`](docs/archive/).

---

## Research extensions

Two stages have a `research/` subdirectory with optional code for
re-training the underlying ML model. These are **not** required to run the
pipeline:

- [`stage1_preprocessing/research/`](stage1_preprocessing/research/README.md)
  — re-train SketchCleanNet on new data.
- [`stage3_primitivesfitting/research/`](stage3_primitivesfitting/research/README.md)
  — Free2CAD Transformer experiment (retired; preserved for reference).

---

## Evaluation

### PatentData strict content filter

PatentData is not a clean CAD corpus. It includes mechanical figures, tables,
flowcharts, dense text, formulas, chemistry, halftones, scientific charts, and
plot pages. For LLM training on SVG/DXF targets we now use a **precision-first**
filter: it intentionally sacrifices recall to keep only pages that are plausible
CAD/vectorization targets.

```bash
# Pilot on the first 1 000 TIFs
TQDM_DISABLE=1 python -m tools.filter_patent_data \
    --limit 1000 --workers 4 \
    --output output/investigation_patent_filter_clean12/filter_manifest.csv

# Full corpus manifest for production
TQDM_DISABLE=1 python -m tools.filter_patent_data \
    --workers 8 \
    --output output/PatentData/filter_manifest_clean12.csv
```

The clean12 filter is deterministic (`probabilistic_hough_line(..., rng=0)`) and
uses EPO filename buckets plus connected-component, skeleton, Hough-line, density,
and orientation features. It rejects `_D` formula/text buckets, `_C` chemistry,
orthogonal tables/flowcharts, dense hatching, dense text, halftones, chart/axis
plots, scientific line plots, sparse scientific plots, multi-panel plot pages,
and text-box block/circuit diagrams.

Clean12 pilot result on the first 1 000 PatentData TIFs:

| Label | Count | Share |
|-------|------:|------:|
| Kept as drawing candidate | 304 | 30.4 % |
| Discarded before vectorization | 696 | 69.6 % |

Top discard reasons:

| Reason | Count |
|--------|------:|
| `orthogonal_text_table_or_flowchart` | 342 |
| `letter_d_text_or_formula` | 134 |
| `text_heavy_plot_table_or_block_diagram` | 77 |
| `block_diagram_text_boxes` | 19 |
| `multi_panel_scientific_plot` | 18 |
| `too_dense_for_clean_cad` | 16 |
| `fragmented_tiny_skeleton_components` | 14 |
| `letter_c_chemistry` | 14 |
| `plot_or_flowchart` | 14 |
| `dense_text_or_flowchart` | 11 |
| `chart_or_axis_plot` | 10 |
| `scientific_line_plot` | 8 |

Clean12 full-corpus result on `data/PatentData/ReorganisedData`:

| Label | Count | Share |
|-------|------:|------:|
| Kept as drawing candidate | 56 971 | 20.7 % |
| Discarded before vectorization | 218 778 | 79.3 % |
| Load/preprocess error | 55 | 0.0 % |
| Total TIFs scanned | 275 804 | 100.0 % |

Top full-corpus discard reasons include `letter_d_text_or_formula` (88 396),
`orthogonal_text_table_or_flowchart` (58 733), `letter_c_chemistry` (17 569),
`fragmented_tiny_skeleton_components` (14 846),
`text_heavy_plot_table_or_block_diagram` (12 906), and explicit chart/plot gates
(`chart_or_axis_plot`, `plot_or_flowchart`, `scientific_line_plot`,
`single_axis_curve_plot`, `sparse_scientific_plot`, `multi_panel_scientific_plot`).

### Batch driver — PatentData corpus

[`tools/batch_run.py`](tools/batch_run.py) runs Stage 0 plus the full
vectorization pipeline over
the partner patent corpus and writes one row of intrinsic metrics per sketch to a
resumable SQLite database:

```bash
# Phase 0 pilot — 100 random sketches, one per patent
python -m tools.batch_run --limit 100 --stratified

# 1 000-TIF clean12 pilot (recommended before full corpus)
python -m tools.batch_run \
    --limit 1000 --workers 4 \
    --config config.yaml \
    --filter-manifest output/investigation_patent_filter_clean12/filter_manifest.csv \
    --output output/investigation_patent_clean12_1k_gated \
    --db output/investigation_patent_clean12_1k_gated/results.db

# Full corpus, resumable, 8 parallel workers
python -m tools.batch_run \
    --workers 8 \
    --config config.yaml \
    --filter-manifest output/PatentData/filter_manifest_clean12.csv \
    --output output/PatentData_clean12_gated \
    --db output/PatentData_clean12_gated/results.db
```

Full clean12 gated batch result:

| Status | Count | Share |
|--------|------:|------:|
| `ok` | 13 904 | 24.4 % |
| `quality_gate_stage3` | 38 145 | 66.9 % |
| `quality_gate_stage2` | 3 558 | 6.2 % |
| `quality_gate_stage1` | 1 364 | 2.4 % |
| `stage1` | 55 | 0.1 % |
| Total manifest rows | 57 026 | 100.0 % |

Accepted `ok` rows average 3.33 s total runtime, 327 Stage-2 edges, 24.5 %
micro-edge ratio, and 20.4 % low-confidence primitive ratio. The gate is
intentionally strict: most visually or geometrically unsuitable drawing candidates
stop before export, rather than becoming noisy SVG/DXF training targets.

### PatentData training manifest curation

`status='ok'` means "the vectorization pipeline completed"; it does **not** mean
"good LLM training target". [`tools/build_training_manifest.py`](tools/build_training_manifest.py)
adds a second high-precision deterministic curation layer and writes auditable
training/reject manifests:

```bash
python -m tools.build_training_manifest \
    --db output/PatentData_clean12_gated/results.db \
    --filter-manifest output/PatentData/filter_manifest_clean12.csv \
    --run-output output/PatentData_clean12_gated \
    --output-csv output/PatentData_clean12_gated/training_manifest_strict.csv \
    --output-jsonl output/PatentData_clean12_gated/training_manifest_strict.jsonl \
    --rejects-csv output/PatentData_clean12_gated/training_manifest_strict_rejects.csv
```

Current strict manifest:

| Manifest | Rows |
|----------|-----:|
| Strict kept examples | 598 |
| Strict rejects | 56 428 |

The strict reject reasons include pipeline quality gates plus semantic filters for
axis/legend patterns, sparse plots, waveform/timing charts, block/flow/network
diagrams, dense hatching/bar charts, chemistry/formula grids, UI-like box layouts,
non-figure patent pages (`A0001`, etc.), diagonal chart/projection patterns,
polar/axis plots, and residual text-heavy pages.

An optional zero-shot CLIP layer is available for visual semantic cleanup:

```bash
python -m tools.clip_curate_manifest \
    --input-csv output/PatentData_clean12_gated/training_manifest_strict.csv \
    --output-csv output/PatentData_clean12_gated/training_manifest_clip.csv \
    --rejects-csv output/PatentData_clean12_gated/training_manifest_clip_rejects.csv \
    --batch-size 16 --device cpu
```

Current CLIP-vetted seed:

| Manifest | Rows |
|----------|-----:|
| CLIP kept examples | 259 |
| CLIP rejects | 339 |

CLIP rejects are mostly hard-negative semantic classes: `ui_screen` (57),
`block_diagram` (54), `frequency_plot` (52), `timeline_chart` (30),
`flowchart` (25), `xy_plot` (25), `table_text` (21), `line_plot` (16),
`chart_axes` (8), `chemistry` (5), and `bar_chart` (3), plus lower-margin
non-hard rejections from the same negative classes.

Audit contact sheets are generated with
[`tools/make_manifest_contact_sheet.py`](tools/make_manifest_contact_sheet.py):

```bash
python -m tools.make_manifest_contact_sheet \
    --manifest output/PatentData_clean12_gated/training_manifest_clip.csv \
    --output output/PatentData_clean12_gated/contact_clip_random.png \
    --sort random --limit 24 --seed 42
```

Current audit sheets live under `output/PatentData_clean12_gated/`:

- `contact_clip_random.png`
- `contact_clip_worst_neg.png`
- `contact_clip_high_micro.png`
- `contact_clip_slowest.png`

**Current assessment:** the CLIP-vetted seed is now suitable as a small,
high-precision bootstrap corpus: the random, worst-negative, high-fragmentation,
and slowest audit sheets are dominated by physical/mechanical drawings rather
than tables, chemistry, plots, or dense text. It is still not a broad final
PatentData corpus; recall is intentionally very low. The next priority is a
supervised visual curation layer trained from audited positives/negatives so we
can recover safe mechanical drawings while preserving this precision.

### Drawing2CAD ground-truth harness

[`tools/d2c_eval.py`](tools/d2c_eval.py) evaluates against the public
[Drawing2CAD](https://drawing2cad.github.io/) dataset. It rasterizes each
ground-truth SVG to a binary PNG (mimicking a patent-office TIF), runs the full
pipeline, then compares the output SVG to the ground truth.

**Headline metric: Chamfer distance on skeletons** — measures geometric placement
accuracy independent of stroke-width rendering. Secondary metrics (pixel IoU,
precision, recall) are also recorded.

```bash
# 100-sample pilot on the test split, Front view only
python -m tools.d2c_eval --limit 100 --views Front --workers 4

# All four views, 25 samples
python -m tools.d2c_eval --limit 25 --views all

# Re-run from scratch (ignore prior results)
python -m tools.d2c_eval --limit 100 --no-resume
```

Results are written to `output/Drawing2CAD/d2c_results.db`; a summary table is
printed at the end of each run.

### Drawing2CAD results (1 000 test-set samples, Front view)

Two configurations are compared: the original CNN path (before this sprint) and
the optimised **vectorised CN path** after all improvements (1 000 samples, seed 42).

| Metric | CNN path (baseline) | **CN path (all fixes)** | Change |
|--------|--------------------:|------------------------:|--------|
| Pixel IoU — mean | 0.619 | **0.681** | **+10.0 %** |
| Pixel IoU — median (p50) | 0.622 | **0.690** | +11 % |
| Pixel IoU — p05 (worst 5 %) | 0.427 | **0.553** | +30 % |
| Pixel IoU — p95 | 0.750 | **0.798** | +6 % |
| Skeleton IoU — mean | 0.505 | **0.547** | +8 % |
| Chamfer sym — mean | 3.83 px | **0.95 px** | **−75 %** |
| Chamfer sym — p95 | 21.3 px | **1.84 px** | **−91 %** |
| Precision | 0.740 | **0.756** | +2 % |
| Recall | 0.868 | **0.872** | +0 % |
| Zero-output samples | 12 / 1 000 | **0 / 1 000** | **−100 %** |
| Stage 2 time — mean | 13.2 s | **10.2 s** | **−23 %** |

Run with: `python -m tools.d2c_eval --limit 1000 --views Front --workers 8 --config config_d2c_eval.yaml --seed 42`

> **Note on stroke-width handling.** Stage 1 estimates the original ink thickness
> via distance transform and stores it in `Stage1Result.mean_stroke_width`. The
> D2C eval harness passes this to Stage 3 so Stage 4 renders SVG strokes at the
> correct visual thickness for metric comparison. The patent production path
> (`batch_run.py`) omits this — it keeps ISO 128 standard lineweights for CAD output.

> **config_d2c_eval.yaml** disables `max_input_resolution` (sets it to 0) so the
> pipeline operates at the same 1 024 × 1 024 px as the GT rasters, eliminating
> a systematic 1–2 px skeleton position error that was causing ≈ 10 % IoU loss on
> thin-stroke shapes. The production `config.yaml` keeps `max_input_resolution: 1000`
> which is required for large (2 000–2 700 px) patent TIFs.

### Stroke de-fragmentation (Stage 2 graph simplification)

The CN map splits a single logical stroke wherever a phantom `CN≥3` pixel appears
(Zhang-Suen staircase) or wherever the stroke crosses a junction, so a long line
fragmented into many primitives. A graph-simplification pass (`_simplify_graph`,
run after topology extraction, before smoothing) fixes this with three operations
iterated to a fixed point: **(a)** prune short dead-end spurs off junctions,
**(b)** dissolve degree-2 phantom junctions (genuine corners are `CN=2`, never
`CN≥3` junctions — so degree-2 junctions are always artefacts and safe to merge),
**(c)** merge collinear edges that pass straight through a real (degree ≥ 3)
junction. Closed loops pass through untouched.

| Metric | Before | **After de-frag** | Change |
|--------|-------:|------------------:|--------|
| **D2C** primitives — mean (1 000 samples, seed 42) | 4.35 | **2.73** | **−37 %** |
| D2C primitives — p95 | 12 | **7** | −42 % |
| D2C prim/stroke ratio — mean | 1.87 | **1.27** | −32 % |
| D2C pixel IoU — mean | 0.681 | **0.683** | +0.3 % |
| D2C Chamfer sym — mean | 0.949 px | **0.937 px** | −1.3 % |
| D2C recall — mean | 0.871 | **0.873** | +0.2 % |
| D2C precision — mean | 0.756 | **0.757** | +0.1 % |
| **PatentData** primitives — mean (30 stratified, seed 42) | 2 364 | **771** | **−67 %** |
| PatentData total edges (30 samples) | 70 927 | **23 133** | −67 % |

Quality (IoU, Chamfer, precision, recall) is unchanged-to-slightly-better on D2C —
the pass removes redundant fragments without moving any geometry, so where it merges
collinear segments the fit gets marginally cleaner. A complex 30-part patent assembly
drops from 2 497 to 848 edges while rendering visually identical. Stage-2 time rises
~3 % (the extra graph pass), offset by Stage 3 now fitting far fewer edges.

Run with: `python -m tools.d2c_eval --limit 1000 --views Front --workers 8 --config config_d2c_eval.yaml --seed 42`

`junction_merge_radius` (collapse junction clusters within N px) is a fourth,
**OFF-by-default** operation: it welds parallel strokes that run close together
(concentric circles, washers, thin-ring / double-wall outlines) and so destroyed a
two-circle D2C sample (IoU −0.246). Enable it only on corpora known to be free of
close parallel lines.

### Patent hachure handling

Hachures were diagnosed as a **Stage 2 topology problem**, not primarily a
Stage 3 fitting problem. Dense section hatching creates many short, parallel
open edges and extra junctions where hatches touch outlines; Stage 3 then
faithfully fits those already-fragmented chains.

The production fix is a side-layer design:

1. Stage 2 first simplifies the graph to reconnect true outlines.
2. If the simplified graph is still micro/short-edge heavy, Stage 2 extracts
   dense local clusters of short, straight, similarly angled strokes as
   `graph.removed_hachures`.
3. The main graph is re-simplified and sent to Stage 3 without hatch clutter.
4. Stage 3 converts `removed_hachures` back into `style: "hachure"` primitives,
   excluded from main-geometry confidence gates.
5. Stage 4 exports them on the SVG/DXF hachure layer, so the final output keeps
   the hatch/detail strokes without letting them break long outlines.

Filtered 100-sample PatentData probe:

| Metric | Stage 0 baseline | Hachure side-layer fix |
|--------|-----------------:|-----------------------:|
| Output folder | `output/PatentData100_stage0refs_iter5` | `output/PatentData100_hachures_v4` |
| `ok` rows | 91/100 overlapping old-ok rows | 94/100 total |
| Hachure-cleaned sketches | n/a | 35/100 |
| Main Stage 2 edges on hachure-cleaned sketches | 186.2 mean | **64.1 mean** |
| Median main edge length on hachure-cleaned sketches | 17.2 px | **75.1 px** |
| Micro-edge ratio on hachure-cleaned sketches | 26.8 % | **0.0 %** |
| Hachure primitives exported / ok | n/a | 43.2 mean |

Visual audit sheet: `output/hachure_final_audit_v4.png`.

Drawing2CAD smoke check with `config_d2c_eval.yaml` (`stage2.remove_hachures:
false`) remains stable on the same 50-sample Front-view overlap:
Chamfer `0.908 → 0.906`, pixel IoU `0.677 → 0.678`, 50/50 ok.

### Puhachov status

The previous "Puhachov" path was not actually improving topology:

- The shipped/local checkpoint is incompatible with the current detector
  architecture. The loader now strips common prefixes and checks same-shape tensor
  matches; incompatible checkpoints raise `ModelNotAvailableError`.
- The old topology builder also computed its own CN topology internally, so CNN
  keypoints did not change the output graph.
- Production config therefore keeps `puhachov.weights: ""` and uses the guarded CN
  path. A true Puhachov path requires a compatible model and explicit topology
  integration/retraining.

**Update — a true CNN path was built, trained on D2C, measured, and a fusion
seeding beat the CN path.** Topology extraction now genuinely consumes keypoint
clusters, and a `_build_stacked_hourglass` model trained on Drawing2CAD (val
peak-F1 0.83; corners 0.96) was compared head-to-head against CN. Pure-CNN
keypoints are slightly *worse* than CN on Chamfer, but **fusion** (CN
endpoints/junctions + only the CNN's corners, `puhachov.fusion: true`) **beats CN
on every accuracy metric** — mean Chamfer −1.1 %, p95 −33 %, IoU +2 %, −10 %
primitives — and is ≈6–8× faster in Stage 2 (corner-splitting avoids B-spline
smoothing of giant closed loops). Validated across **all 4 D2C views** (fusion
wins every aggregate metric) and on **PatentData** (neutral, out-of-distribution),
fusion is **now the production default** (`puhachov.weights: models/puhachov_d2c.pth`,
`puhachov.fusion: true`) with automatic pure-CN fallback. Full table + analysis in
[Roadmap → Results](#results--cn-vs-cnn-vs-fusion-executed).

### PatentData strict gated pilot (first 1 000 corpus TIFs)

After clean12 filtering, Stage 0 plus the vectorization pipeline runs with
quality gates enabled:

| Status | Count |
|--------|------:|
| `ok` | 93 |
| `quality_gate_stage1` | 9 |
| `quality_gate_stage2` | 55 |
| `quality_gate_stage3` | 147 |

Accepted `ok` exports:

| Metric | Value |
|--------|------:|
| Mean total time / ok | 2.72 s |
| Mean Stage 2 edges / ok | 342.8 |
| Max Stage 2 edges / ok | 774 |
| Mean micro-edge ratio / ok | 21.1 % |
| Mean low-confidence primitive ratio / ok | 17.4 % |
| Max low-confidence primitive ratio / ok | 25.0 % |

Visual audit contact sheets:

- `output/investigation_patent_clean12_1k_gated/contact_worst_ok.png`
- `output/investigation_patent_clean12_1k_gated/contact_slowest_ok.png`

The accepted worst/slowest samples are now mostly mechanical, device, and geometric
engineering drawings. Obvious tables, dense text, formulas, chemistry pages,
halftones, flowcharts, and scientific plots are rejected in this pilot. This is a
high-precision pilot result, not a full-corpus claim.

---

## Project status

### Completed

- **Stage 0 + four-stage pipeline** end-to-end, configurable via `config.yaml`
- **Stage 0** — patent reference handling: detects leadered reference numeral
  clusters, segment-adjacent labels, compact unleadered labels, and `FIG`/panel
  captions; writes
  `references/<id>_references.json`, `references/<id>_references_mask.png`,
  transparent label crops, and a `references/<id>_norefs.png` raster for Stage 1;
  includes bounded iterative removal (`stage0.max_iterations: 10`), leader-tip
  trimming, strict single-character margin rules, panel-caption removal, and local
  mask repair to reduce broken true strokes; Stage 4 reinjects references as SVG
  crop overlays + leader lines and DXF leader geometry
- **Stage 1** — SketchCleanNet inference; classical fallback; bottom-edge
  ghost-ink artefact fixed; stroke-width estimation via distance transform
- **Stage 2** — production CN-cluster skeleton tracing; topology extraction now
  consumes keypoint *clusters* (`_extract_topology(kp_clusters=…)`) so the
  detector controls the graph; **default fusion seeding** (`puhachov.fusion: true`,
  shipped `models/puhachov_d2c.pth`) — CN endpoints/junctions + the keypoint CNN's
  corners — beats the pure-CN path on Drawing2CAD all-views (Chamfer, IoU,
  primitives) and is much faster on cornered closed shapes, with automatic pure-CN
  fallback; guarded loader rejects incompatible checkpoints;
  **resolution cap** (`max_input_resolution: 1000`) — skeletons larger than
  1 000 px are dilated, downsampled, and re-skeletonized (Zhang-Suen) before
  topology extraction runs, preventing 100×–300× over-segmentation on large patent TIFs;
  **PyTorch thread cap** (`torch.set_num_threads(1)`) forces process-level
  parallelism so batch workers don't contend for CPU cores; **adaptive NMS
  radius** scales junction suppression with image size; **B-spline overshoot
  guard**; **noise closed-loop filter**; **graph de-fragmentation** (`_simplify_graph`
  — spur prune + degree-2 phantom-junction dissolve + collinear through-merge;
  D2C primitives −37 %, PatentData −67 %, IoU/Chamfer unchanged-to-better);
  **patent hachure extraction** separates dense hatch clusters into
  `removed_hachures` so long outlines fit as main geometry
- **Stage 3** — RANSAC cascade (line / circle / arc / ellipse / **polygon** /
  polyline); closed polygon fitter; sparse-smooth_pts guard; geometric arc guard;
  hachure side-layer reinjection as styled primitives; Free2CAD Transformer
  evaluated and retired
- **Stage 4** — SVG and DXF export; ISO 128 layered patent DXF, including a
  dedicated `HACHURE` layer
- **Content classifier** (`tools/filter_patent_data.py`) — deterministic,
  feature-based strict filter (Hough lines + CC/skeleton analysis + density/orientation
  gates + EPO letter codes); clean12 full-corpus pass scanned 275 804 TIFs and
  kept 56 971 drawing candidates while rejecting tables, formulas, chemistry,
  dense text, flowcharts, halftones, and plot pages
- **Batch evaluation driver** (`tools/batch_run.py`) — resumable, multi-worker,
  SQLite results; `--filter-manifest` flag to skip non-drawing TIFs; Stage 1/2/3
  quality gates prevent bad examples from entering SVG/DXF training targets; full
  clean12 gated run produced 13 904 `ok` exports and rejected 43 522 candidates
  through pipeline/quality gates
- **Training manifest curation** (`tools/build_training_manifest.py`,
  `tools/clip_curate_manifest.py`, `tools/make_manifest_contact_sheet.py`) —
  strict auditable manifest builder plus optional CLIP semantic curation and
  reproducible visual audit sheets; current seed is 598 deterministic examples,
  reduced to 259 CLIP-vetted bootstrap examples after visual semantic cleanup
- **Drawing2CAD eval harness** (`tools/d2c_eval.py`) — Chamfer distance as
  headline metric; pixel IoU / precision / recall as secondary

### Bug-fix root-cause log

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Lines → wiggly curves in SVG | `scipy splprep` Runge-phenomenon oscillations on long near-straight skeleton edges | Bounding-box overshoot guard: reject spline if any sample exits the raw-pixel bbox by > 5 px; fall back to RDP points |
| 2 | Spurious small circles in SVG | Sub-80-px closed skeleton loops from scan noise fit as circles by Stage 3 | Noise filter: remove small closed loops whose radial RMS > 30 % of mean radius; genuine small circles (low RMS) are preserved |
| 3 | Rectangles → round polylines | Right-angle corners prevent circle/ellipse fit → raw 300-pt pixel trace used | Closed polygon fitter: try RDP-simplified N-vertex polygon before polyline fallback |
| 4 | Rectangular open chains → arcs | B-spline overshoot → RDP corner-point fallback (6 pts); all corners lie on circumscribed circle → arc RANSAC wins | Sparse-smooth_pts guard: if `len(smooth_pts) < max(10, 5 % of raw pixels)` skip arc, emit corner-polyline |
| 5 | Small circles missing | `min_closed_loop_pixels=80` removed genuine small circles (radius ~7–13 px) as noise | Added circularity check: only remove loops that are geometrically non-circular (RMS radial deviation > 30 % of mean radius) |
| 6 | Stage 2 hangs on large patent TIFs (2 000–2 700 px) | Puhachov CNN trained on ~512 px; at 2 700 px it detects 100 k–300 k keypoints; pixel graph has millions of nodes | Resolution cap: dilate 1-px skeleton → resize to ≤ 1 000 px → re-skeletonize (restores 1-px width); reduces edges from ~20 k → ~1 500 per sketch (15×), time from ~190 s → ~13 s |
| 7 | Stage 2 slow under multi-worker batch despite resolution cap | PyTorch uses all CPU cores per process by default; 8 workers × all cores = severe contention | `torch.set_num_threads(1)` in `load_model()`: forces process-level parallelism |
| 8 | S2 isolation flag rate 93 % on patent data | `isolation_threshold: 0.05` calibrated for clean SVG inputs; patent TIF scan noise produces 10–20 % orphaned pixels on valid drawings | Recalibrated to `0.30` (flags only top ~5 % of sketches — genuinely noisy scans) |
| 9 | CN map O(H×W) Python loop bottleneck | `_cn_map_vectorized` and `_extract_topology` step 1 used a Python loop over every pixel | Replaced with NumPy ring-slice shifts: ~1 000× speedup at 1 000 px resolution |
| 10 | O(n_clusters × H×W) cluster groupby bottleneck | `np.where(labels == c)` inside `_add_cluster` loop scanned full image once per cluster | Single-pass argsort groupby: O(n_fg × log n_fg) total; _extract_topology time 95 s → 0.3 s |
| 11 | Bulk label dilation bottleneck | Per-cluster `cv2.dilate` called N times on full H×W image | Single float32 label-image dilation: one `cv2.dilate` call, max-pooling assigns each extended pixel to nearest keypoint |
| 12 | D2C zero-output: tiny circles (< 40 px) dropped | Hardcoded `min_loop_pixels = 40` in `_extract_topology` step 5 silently removed circles with radius < 6 px before they could be preserved by the circularity guard | Lowered to `min_loop_pixels = 8` so the circularity guard (`_is_circular_loop`) actually gets to decide |
| 13 | D2C zero-output: small circles fail circularity check | `_is_circular_loop` used a fixed 30 % RMS threshold; Zhang-Suen staircasing on tiny circles (r < 12 px) produces ≥ 38 % relative RMS, so genuine circles were rejected as noise | Size-adaptive threshold: 55 % for r_mean < 12 px, 30 % otherwise |
| 14 | Parallel paths between same two nodes lost | Walk used `frozenset({src, dst})` dict key — only the first path found between any pair of nodes was stored; parallel arcs and two-arc circles lost remaining paths | Replaced with list + `all_edge_pix` entry check: deduplicates reverse-direction walks while allowing genuinely parallel paths with disjoint pixel sets |
| 15 | Long strokes fragmented into many primitives | CN map emits a phantom `CN≥3` junction at every Zhang-Suen staircase bend and at every stroke crossing, so one logical stroke is split at each — dense patent scans produce thousands of 2–3 px junction stubs | `_simplify_graph` pass: prune short spurs, dissolve degree-2 phantom junctions, merge collinear edges straight through real junctions. D2C prims −37 % (1 000-sample), PatentData prims −67 %, IoU/Chamfer unchanged-to-better |
| 16 | Reference numerals/help lines split long patent strokes | Patent reference labels and leader lines enter Stage 1 as normal ink; every leader/feature contact becomes a skeleton junction or tiny post-removal gap | Added Stage 0 reference handling. It now removes leadered labels, segment-adjacent labels, compact unleadered labels, figure/panel captions, and heavy annotation cases before Stage 1, then reinjects the captured crops/leaders during export. On the filtered 100-sample PatentData probe, all-row mean Stage 2 edges moved 307.95 baseline → 270.71 two-pass Stage 0 → 199.31 final Stage 0. Final residual scan on `*_norefs.png`: 100/100 active removals, 0 Stage 0 flags, 99/100 with zero residual detections, max residual 1 |
| 17 | Hachures reduce long-stroke quality | Dense section hatching creates real Stage 2 junctions and hundreds of short parallel edges before primitive fitting; Stage 3 mostly fits the fragmented graph it receives | Added adaptive Stage 2 hachure side-layer extraction after graph simplification, then re-simplifies the main graph. Stage 3 reinjects extracted hachures as `style: "hachure"` primitives excluded from main-geometry confidence gates; Stage 4 exports them on a `HACHURE` layer. On the filtered 100-sample probe, hachure-cleaned sketches moved from 186.2 → 64.1 main edges, median main edge length 17.2 → 75.1 px, micro-edge ratio 26.8 % → 0.0 %, while v4 exports all hatch primitives (`output/PatentData100_hachures_v4`) |

### Known limitations / next steps

1. **Supervised visual curation layer** — deterministic rules + zero-shot CLIP now
   produce a small high-precision seed. Build an active-learning classifier from
   audited positives/negatives sampled from `training_manifest_strict.csv`,
   `training_manifest_clip.csv`, and the CLIP reject set. Target: preserve the
   current chart/table/chemistry precision while recovering safe mechanical
   drawings currently rejected by the very strict hand rules.

2. **Scale the LLM training manifest after classifier validation** — use
   `training_manifest_clip.csv` only as a bootstrap set for now. Once supervised
   curation is validated on held-out PatentData audit sheets, write the final
   training manifest with input image path, SVG path, DXF path, primitive JSON path,
   metrics, filter reason/status, and visual classifier confidence so every example
   remains traceable.

3. **Stage 0 reference refinement** — the filtered 100-sample probe is now
   effectively reference-free before Stage 1 (`output/PatentData100_stage0refs_iter5`:
   100/100 active removals, 0 Stage 0 flags, 99/100 zero residual detections,
   max residual 1). Remaining priorities:
   - separate dashed/hidden construction geometry from annotation/dimension
     geometry more reliably; annotation-heavy dimension figures can require
     60-75 % ink removal;
   - add OCR for numeric `text` fields so DXF can reinject real MTEXT, not only
     crop overlays in SVG and leader geometry in DXF;
   - add a post-reference stub/gap cleanup pass after Stage 2, because removing a
     leader at a feature contact can leave tiny remnants even when total edge
     count improves;
   - broaden the residual visual audit from 100 samples to a larger filtered
     PatentData slice before generating the final LLM training manifest.

4. **Keypoint CNN — done; fusion is the default, patent gap is open.** The
   detector was retrained on Drawing2CAD and the **fusion** seeding (CN
   endpoints/junctions + CNN corners) now beats the pure-CN path on D2C all-views
   and is the production default (see
   [Roadmap → Results](#results--cn-vs-cnn-vs-fusion-executed)). It is **neutral on
   PatentData** (out-of-distribution). Domain-randomized fine-tuning narrowed the
   gap at the *detector* (−59 % spurious corners on patent skeletons) but did not
   change patent *output* metrics. Remaining work to make fusion win on patents:
   - a **small gold corner-labelled patent eval set** (~50–100 sketches) — the
     intrinsic metrics used so far are insensitive to corner-only changes, so this
     is needed just to *measure* progress;
   - **real-patent pseudo-labels** (cleaned-graph nodes + fitted-primitive corners)
     to fine-tune on the true patent skeleton distribution;
   - check whether the bottleneck is **downstream** (fusion corner-dedup +
     `_simplify_graph` discard the corner gains before Stage 3 sees them).

5. **Stroke fragmentation** *(largely addressed — see "Stroke de-fragmentation"
   above)* — the `_simplify_graph` pass cut D2C primitives −37 % and PatentData
   primitives −67 % by pruning spurs, dissolving degree-2 phantom junctions, and
   merging collinear edges straight through real junctions. Remaining over-segmentation
   comes from harder, still-open cases:
   - **Thin outline rectangles / concentric rings → double-wall ladders.** A thin
     *outline* rectangle (or a washer / concentric-circle pair) skeletonizes to two
     close parallel rails joined by short rungs; each rail segment between rungs is
     its own primitive. Junction-cluster merging would fix it but unsafely welds the
     two rails — needs a ladder-aware rung-removal that preserves both rails.
   - **Hachure/dashed-detail policy.** Hatch side-layer extraction now prevents
     hatches from fragmenting long outlines and reinjects them into SVG/DXF, but
     visually similar dashed detail borders can also enter the hachure layer. This
     preserves the strokes, yet a future training manifest should decide whether
     these belong on `HACHURE`, `hidden`, or `construction` layers.

6. **Dense drawings and runtime outliers** — sketches with extreme hatch/detail can
   still take much longer than the median. Stage 2/3 gates now reject most of these,
   but the full-corpus run should be monitored for long-tail workers and may need a
   per-sketch timeout.

   Dense drawings now de-fragment to ~⅓ the edge count (`_simplify_graph`), and
   hachure extraction cuts hatch-heavy main graphs substantially, but the slowest
   highly detailed figures can still exceed the time budget — a `max_edges` guard
   remains worthwhile.

7. **Thin-stroke position error (1 px)** — Zhang-Suen skeletonisation places the
   skeleton 1 px off-centre for thin strokes (radius ≤ 3 px), causing a systematic
   IoU loss of ~15–20 % for these shapes. Could be improved by distance-transform
   centroid refinement, but requires changes to Stage 1.

8. **Spur-pruning coverage trade-off** — `_simplify_graph`'s spur prune removes
   dead-end edges shorter than `spur_min_length` (6 px), which raises the patent
   isolation ratio by ~0.05 on average (some pruned stubs are real short ticks, not
   just barbs). Still well under the 0.30 flag threshold; lower `spur_min_length`
   if a corpus has many genuine short features.

---

## Roadmap — Puhachov keypoint CNN: retraining & CN-vs-CNN comparison

**Goal.** Retrain the keypoint detector on the **Drawing2CAD** dataset, then
measure whether a *learned* keypoint detector beats the production classical
**CN** path inside this pipeline (Chamfer / IoU / fragmentation / runtime).

### Critical precondition

Retraining alone changes **nothing**. `_extract_topology()` in
[`stage2_stroke_extract.py`](stage2_strokeextraction/stage2_stroke_extract.py)
accepts a `keypoints` argument but **ignores it** — topology is always rebuilt
from the internal CN map, so today the "Puhachov path" and the "CN path" emit
identical graphs downstream of keypoint detection. **Wiring keypoints into
topology (Phase 3) is the load-bearing work; without it the comparison measures
nothing.**

### Design decisions

| # | Decision | Choice |
|---|----------|--------|
| A | Which network | In-repo lightweight `_build_stacked_hourglass` (3 channels: endpoint / junction / **corner**) — matches the existing guarded loader, has no frame-field machinery the pipeline can't consume, and avoids re-introducing the checkpoint-incompatibility problem |
| B | How keypoints drive topology | Start by **replacing** CN seeds with CNN keypoints; ablate by **fusing** (CNN suppresses phantom `CN≥3` staircase junctions; corners deliberately split polylines) |
| C | `_simplify_graph` in the comparison | Run the CNN arm **with** and **without** it — tests whether a learned detector can *replace* the heuristic de-fragmentation pass |

### Phases

0. **Baseline lock** — re-run the CN path on the D2C `test` split with
   [`config_d2c_eval.yaml`](config_d2c_eval.yaml) on the current machine/seed and
   freeze it as the comparison baseline DB. Freeze the keypoint→topology contract
   (classes, coordinate frame, resolution).
1. **Ground-truth labels** (new `tools/d2c_keypoint_labels.py`) — D2C `svg_raw`
   `M`/`L` paths give exact keypoints with no manual annotation: vertex degree
   1 → endpoint, ≥ 3 → junction, degree-2 with a sharp turn → corner. Rasterize
   each view, **Stage-1 skeletonize** (train on real skeleton artefacts, not the
   clean GT raster), snap each keypoint to the nearest skeleton pixel, and emit
   3-channel Gaussian heatmap targets. Visually audit ~50 overlays before scaling
   to the full 141 831-sample train split (× 4 views).
2. **Training** (new `stage2_strokeextraction/research/train_puhachov.py`,
   mirroring [`stage1_preprocessing/research/train_sketchcleannet.py`](stage1_preprocessing/research/train_sketchcleannet.py))
   — train on the `train` split only, weighted BCE/focal heatmap loss with
   small rotation/scale augmentation, checkpoint best-on-`validation` by per-class
   peak F1, and export `models/puhachov_d2c.pth` whose tensor names/shapes satisfy
   the loader's matched-tensor check. **Gate:** the CNN must beat CN on intrinsic
   keypoint precision/recall before integrating.
3. **Topology integration** *(critical path)* — refactor `_extract_topology` to
   seed clusters from a supplied keypoint source (snapped to skeleton pixels,
   reusing the existing 1-px halo + walk machinery), driven by the already-tracked
   `kp_source` switch. Unit-test on rectangle / two-arc circle / T-junction shapes
   before any batch run.
4. **Comparison matrix** — add `config_d2c_eval_puhachov.yaml` and run
   [`tools/d2c_eval.py`](tools/d2c_eval.py) on the `test` split per arm into
   separate result DBs:

   | Arm | Keypoints | `_simplify_graph` |
   |-----|-----------|-------------------|
   | A (baseline) | CN | on |
   | B | CNN | on |
   | C | CNN | off |
   | D *(optional)* | CN + CNN fused | on |

   Same seed/workers/split. Diff Chamfer (headline), pixel/skeleton IoU,
   precision, recall, primitive count & prims/stroke ratio, zero-output count,
   and Stage-2 runtime.
5. **Analysis & verdict** — per-metric delta table + win/loss audit contact
   sheets; record the outcome (including a clean null result) in the
   [Puhachov status](#puhachov-status) and the bug-fix / limitations log.

**De-risking.** Do a thin Phase-3 spike *first*: make `_extract_topology` consume
*CN-derived* keypoints passed as an argument and confirm the output is
byte-identical to today. That proves the plumbing before any GPU time is spent on
training.

### Results — CN vs CNN vs fusion (executed)

All phases were run. Labels: 567 324 train + 31 516 val skeletons from D2C
`svg_raw` (99.8 % zero-drop). The CNN (`_build_stacked_hourglass`, 3 channels)
trained 60 k steps to **val peak-F1 0.83** — corner **0.96**, junction 0.86,
endpoint 0.66. Comparison on **1 000 test-split Front samples, seed 42**, paired
(1000/1000 ok in every arm):

| Metric | A · CN | B · CNN (0.3) | C · CNN no-simplify | **D · fusion** |
|--------|-------:|--------------:|--------------------:|---------------:|
| **Chamfer sym — mean** | 0.937 | 1.007 (+7.4 %) | 1.040 (+11.0 %) | **0.927 (−1.1 %)** |
| **Chamfer sym — p95** | 3.00 | 2.24 | 2.24 | **2.00 (−33 %)** |
| Pixel IoU — mean | 0.683 | 0.695 | 0.688 | **0.697 (+2.0 %)** |
| Precision / Recall | 0.757 / 0.873 | 0.764 / 0.881 | 0.763 / 0.872 | **0.767 / 0.881** |
| Primitives / sketch | 2.73 | 2.27 | 6.93 (+144 %) | **2.45 (−10 %)** |

**Verdict — the fusion arm wins.** Pure CNN keypoints (arm B) are *worse* than CN
on Chamfer at every threshold (its endpoint/junction localisation, F1 0.66/0.86,
is less exact than crossing number on clean D2C skeletons). But **fusion — CN's
near-exact endpoints + junctions plus *only* the CNN's confident corners (F1
0.96, which CN structurally cannot detect)** — beats CN on **every** accuracy
metric: mean Chamfer −1.1 %, **p95 Chamfer −33 %**, IoU +2.0 %, precision +1.3 %,
and −10 % primitives. Arm C confirms the **CNN does _not_ replace
`_simplify_graph`** (no-simplify fragments to +144 % primitives).

**Why it also gets faster.** Controlled single-process Stage 2 (40 samples):
median **2.32 s → 0.27 s (≈8×)**, mean 9.36 s → 1.47 s (≈6×). Same mechanism as
the accuracy win: a closed cornered shape (rectangle/polygon) has *no* CN
endpoints/junctions, so the CN path traces it as **one giant closed loop** and
spends seconds B-spline-smoothing 2000+ points; the CNN corners **split it into
short arcs** → trivial smoothing *and* clean line fits instead of one wiggly
loop. (The per-sketch `s2_time` in the eval DBs is wall-clock under different
worker concurrency across runs, so it is not used for the speed claim.)

**Validation across all 4 views + PatentData.** Re-run on the test split, **all
four views** (2400 paired samples) and on a paired **PatentData** sample (clean12
filter):

| Domain | Chamfer | IoU | Primitives | Verdict |
|--------|--------:|----:|-----------:|---------|
| D2C all-views | 1.434 → **1.425** | 0.655 → **0.671** | 4.60 → **4.16** | fusion wins every metric |
| · orthographic (F/T/R) | 0.959 → **0.926** (−3.4 %) | 0.677 → **0.696** | — | fusion wins |
| · isometric (FrontTopRight) | 2.857 → 2.923 | 0.591 → 0.595 | 11.27 → **10.24** | near-tie (hard for both; fewer prims) |
| PatentData (paired, n=54) | — *(no GT)* | — | 142.8 → 143.5 | **neutral** (identical accept-rate; ±2 % conf) |

Fusion is **≥ CN on every D2C view** (clearly better on the three orthographic
views and on IoU/primitives everywhere; the isometric view is inherently hard for
both) and **neutral on out-of-distribution PatentData** (the D2C-only detector
adds no benefit there, but does no harm — its CN backbone dominates and patent
scans lack the giant-closed-loop pattern fusion exploits).

**Now the default.** Because fusion never regresses and wins where it is
in-distribution, `config.yaml` now defaults to `puhachov.weights:
models/puhachov_d2c.pth` + `puhachov.fusion: true` (shipped 30 MB weight). The
loader falls back to the pure-CN path automatically if the weight is absent or
PyTorch/GPU is unavailable. For large **PatentData** batches — where fusion is
neutral and the CNN forward is wasted work — set `puhachov.weights: ""` to force
the faster pure-CN path. Remaining levers: keypoint-threshold sweep, richer CNN
cluster cores in `_clusters_from_points` (currently single-pixel), corner-only
training, and patent-domain fine-tuning to make fusion *win* (not just tie) on
PatentData.

### Patent-domain fine-tuning (domain randomization — attempted)

To try to make fusion *win* on PatentData, the detector was fine-tuned with
**domain randomization**: clean D2C skeletons degraded to look patent-scan-like
(additive speckle, spurs, hachure clusters, mild gaps — no structure removal, so
labels stay valid), 60 % degraded / 40 % clean, initialised from
`puhachov_d2c.pth` (`train_puhachov.py --init-weights … --patent-aug 0.6`).

Outcome — a **detector-level success that did not reach the pipeline output**:

| `detect()` on real patent skeletons | endpoints | junctions | corners |
|-------------------------------------|----------:|----------:|--------:|
| D2C model | 121.6 | 68.0 | **355.8** |
| patent-FT model | 103.7 | 64.9 | **145.8** |

The D2C model **over-fires ~356 spurious corners/sketch on patent clutter**; the
fine-tuned model fires **−59 %** (and *improved* clean-D2C val-F1, 0.828 → 0.832 —
the degradation acts as regularisation, no forgetting). But the paired PatentData
*output* metrics were unchanged (primitives 143.5 → 143.0, conf 0.740 → 0.740):
the fusion corner-dedup + `_simplify_graph` already absorb the D2C model's excess
corners downstream, and the aggregate intrinsics (~140 prims/sketch) are
insensitive to corner-only changes. **Conclusion:** domain randomization narrowed
the OOD gap at the detector but did not change patent vectorisation quality, so
the patent default is unchanged. Closing it for real needs either real-patent
pseudo-labels (graph nodes + fitted-primitive corners) or a small gold corner-
labelled patent eval set to measure the effect directly. The
`puhachov_d2c_patentft.pth` weight is not shipped.

Reproduce: train `models/puhachov_d2c.pth` via
`stage2_strokeextraction/research/train_puhachov.py`; run arms with
`tools/d2c_eval.py --split test --limit 1000 --views Front --seed 42` and
`config_d2c_eval_puhachov.yaml` (B), `…_nosimplify.yaml` (C),
`…_fusion.yaml` (D). Baseline DB frozen at `output/Drawing2CAD/cn_baseline/`.

> **Training note.** Focal loss requires the RetinaNet/CenterNet output-head
> init (small weights + negative prior bias); without it the model collapses to
> "background everywhere" (val F1 0, loss frozen at the −log(1e-6) clamp). Fixed
> in `train_puhachov.py`.

---

## License

Apache License, Version 2.0 — see [`LICENSE`](LICENSE) for the full text.
This permits commercial use, modification, and redistribution, and
includes a patent grant.

---

## Acknowledgements

- **SketchCleanNet** — Simo-Serra et al., *"Mastering Sketching"*, ACM TOG 2018.
- **Puhachov keypoint CNN** — Puhachov et al., keypoint extraction for line drawings.
- **Free2CAD** — Li et al., Transformer-based stroke-to-CAD primitive fitting.
- HAW Landshut — IP DrawingDrafter project (AP3).
