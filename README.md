# IP DrawingDrafter — AP3 Vectorization Pipeline

> Hand-drawn engineering sketch → editable vector files (SVG / DXF).
> HAW Landshut · IP DrawingDrafter project · AP3.

This repository implements the four-stage vectorization pipeline of the
IP DrawingDrafter project. It takes a raster image of a hand-drawn
engineering sketch as input and produces clean, ISO 128–styled vector
files (SVG and DXF) that can be opened in any CAD application.

```
   raster PNG               cleaned + skeleton           stroke graph
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────┐         ┌──────────────────┐       ┌──────────────────┐
│   Stage 1    │  ─────▶ │     Stage 2      │ ────▶ │     Stage 3      │
│ Preprocessing│         │ Stroke Extraction│       │ Primitive Fitting│
└──────────────┘         └──────────────────┘       └──────────────────┘
                                                            │
                                                            ▼
                                                   ┌──────────────────┐
                                                   │     Stage 4      │
                                                   │     Export       │
                                                   │  (SVG, DXF)      │
                                                   └──────────────────┘
```

| Stage | In                  | Out                         | Tech                       |
|------:|---------------------|-----------------------------|----------------------------|
| 1     | Raw raster (PNG/TIF) | Cleaned image + 1-px skeleton + stroke-width estimate | SketchCleanNet (or classical fallback) |
| 2     | 1-px skeleton       | Stroke graph (JSON)         | **Vectorised CN path** (NumPy ring-slice shifts, ~1 000× vs Python loop); incompatible Puhachov weights are rejected instead of silently ignored; parallel-edge walk; **graph de-fragmentation** (spur prune + degree-2 dissolve + collinear through-merge); fragmentation metrics/gates; deterministic scale metadata |
| 3     | Stroke graph        | Geometric primitives (JSON) | RANSAC cascade (line / circle / arc / ellipse / **polygon** / polyline); primitives are rescaled back to the original image frame after Stage 2 downsampling; low-confidence ratio gate |
| 4     | Geometric primitives | SVG and/or DXF              | `svgwrite`, `ezdxf` (ISO 128 layered); polygon primitive rendered as closed shape |

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

The four stage scripts are intentionally **decoupled**: each one reads its
input from disk, writes its output to disk, and has no Python imports
between stages. You can run any stage standalone, swap a stage's
implementation, or insert a new stage between two existing ones.

---

## Model weights

Three model weights are referenced by the pipeline:

| Weight                       | Size  | Used by              | Status                      |
|------------------------------|------:|----------------------|-----------------------------|
| `puhachov_keypoints.pth`     | 22 MB | Stage 2 *(disabled)* | local checkpoint is incompatible with the current detector; `config.yaml` leaves `puhachov.weights: ""` so Stage 2 uses the CN path |
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

All four stages read from the single [`config.yaml`](config.yaml) at the
project root. All paths in it are relative to the project root, so the
config works on any clone without editing.

The most common knobs:

| Key                                | Default | Effect |
|------------------------------------|--------:|--------|
| `sketchcleannet.weights`           | `models/sketchcleannet.pth` | empty `""` ⇒ force classical cleaning mode |
| `sketchcleannet.device`            | `cpu`   | `cuda` for GPU |
| `puhachov.weights`                 | `""`    | Keep empty for production; the local Puhachov checkpoint is incompatible and is rejected by the loader |
| `puhachov.device`                  | `cpu`   | `cuda` for GPU if a compatible detector is supplied later |
| `stage2.max_input_resolution`      | `1000`  | Skeleton images with long edge > this are downsampled before Stage 2. Prevents CNN over-segmentation on large patent TIFs (2000–2700 px). Set to `0` to disable. |
| `stage2.isolation_threshold`       | `0.30`  | Flag sketch if > this fraction of foreground pixels are unreached by any extracted stroke. Calibrated for patent TIF scan noise (p75 isolation ≈ 0.16). |
| `stage2.nms_reference_resolution`  | `512`   | Training resolution of the Puhachov model; NMS radius scales as `nms_radius × max(H,W) / this value` on larger inputs (0 = fixed radius) |
| `stage2.spline_overshoot_limit`    | `5.0`   | Max px a B-spline may exceed the raw pixel bbox; prevents scipy end-effect oscillations |
| `stage2.min_closed_loop_pixels`    | `80`    | Closed loops shorter than this are treated as noise and removed before Stage 3 |
| `stage1.quality_threshold`         | `0.70`  | Sketches below this skeleton quality are flagged for review |
| `stage3.confidence_threshold`      | `0.60`  | Primitives below this are flagged for review |
| `pipeline.quality_gates.enabled`   | `true`  | Batch mode stops bad examples before export using Stage 1/2/3 metrics |

---

## Per-stage documentation

Each stage has its own architecture document:

- **Pipeline overview** — [`docs/pipeline_overview.md`](docs/pipeline_overview.md)
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

[`tools/batch_run.py`](tools/batch_run.py) runs the full four-stage pipeline over
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
| Strict kept examples | 683 |
| Strict rejects | 56 343 |

The strict reject reasons include pipeline quality gates plus semantic filters for
axis/legend patterns, sparse plots, waveform/timing charts, block/flow/network
diagrams, dense hatching/bar charts, chemistry/formula grids, UI-like box layouts,
and residual text-heavy pages.

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
| CLIP kept examples | 412 |
| CLIP rejects | 271 |

CLIP rejects are mostly `ui_screen` (66), `line_plot` (60), `block_diagram` (56),
`flowchart` (25), `chart_axes` (23), `table_text` (22), `chemistry` (14), and
`bar_chart` (5). Audit contact sheets are generated under
`output/PatentData_clean12_gated/`:

- `contact_clip_random.png`
- `contact_clip_worst_neg.png`
- `contact_clip_high_micro.png`
- `contact_clip_slowest.png`

**Current assessment:** the CLIP-vetted seed is useful for a small,
high-precision bootstrap set, but it is not yet the final PatentData training
corpus. The latest visual audit still shows a few chart/axis pages and
timeline-like diagrams surviving the filter. The next priority is a supervised
visual curation layer trained from audited positives/negatives so we can remove
those residual classes without shrinking recall blindly.

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

### PatentData strict gated pilot (first 1 000 corpus TIFs)

After clean12 filtering, the four-stage pipeline runs with quality gates enabled:

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

- **Full four-stage pipeline** end-to-end, configurable via `config.yaml`
- **Stage 1** — SketchCleanNet inference; classical fallback; bottom-edge
  ghost-ink artefact fixed; stroke-width estimation via distance transform
- **Stage 2** — production CN-cluster skeleton tracing; guarded Puhachov loader
  rejects incompatible checkpoints instead of silently using bad weights;
  **resolution cap** (`max_input_resolution: 1000`) — skeletons larger than
  1 000 px are dilated, downsampled, and re-skeletonized (Zhang-Suen) before
  topology extraction runs, preventing 100×–300× over-segmentation on large patent TIFs;
  **PyTorch thread cap** (`torch.set_num_threads(1)`) forces process-level
  parallelism so batch workers don't contend for CPU cores; **adaptive NMS
  radius** scales junction suppression with image size; **B-spline overshoot
  guard**; **noise closed-loop filter**; **graph de-fragmentation** (`_simplify_graph`
  — spur prune + degree-2 phantom-junction dissolve + collinear through-merge;
  D2C primitives −37 %, PatentData −67 %, IoU/Chamfer unchanged-to-better)
- **Stage 3** — RANSAC cascade (line / circle / arc / ellipse / **polygon** /
  polyline); closed polygon fitter; sparse-smooth_pts guard; geometric arc guard;
  Free2CAD Transformer evaluated and retired
- **Stage 4** — SVG and DXF export; ISO 128 layered patent DXF
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
  `tools/clip_curate_manifest.py`) — strict auditable manifest builder plus optional
  CLIP semantic curation; current seed is 683 deterministic examples, reduced to
  412 CLIP-vetted examples after visual semantic cleanup
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

### Known limitations / next steps

1. **Supervised visual curation layer** — deterministic rules + zero-shot CLIP now
   produce a small high-precision seed, but residual chart/axis pages still survive.
   Build an active-learning classifier from audited positives/negatives sampled from
   `training_manifest_strict.csv`, `training_manifest_clip.csv`, and the CLIP reject
   set. Target: remove charts, plots, flowcharts, UI/block diagrams, chemistry, and
   tabular pages while recovering safe mechanical drawings currently rejected by the
   very strict hand rules.

2. **Scale the LLM training manifest after classifier validation** — use
   `training_manifest_clip.csv` only as a bootstrap set for now. Once supervised
   curation is validated on held-out PatentData audit sheets, write the final
   training manifest with input image path, SVG path, DXF path, primitive JSON path,
   metrics, filter reason/status, and visual classifier confidence so every example
   remains traceable.

3. **Text/annotation handling** — accepted patent drawings still include figure
   labels, dimensions, reference numerals, and short annotations. For text-to-CAD
   training this may be useful metadata; for geometry-only training it needs OCR
   masking or separate layers before primitive fitting/export.

4. **True Puhachov replacement** — the local checkpoint is incompatible and the old
   code path did not affect topology. Port/retrain a compatible keypoint detector
   only after the CN/gated baseline has a clean full-corpus manifest.

5. **Stroke fragmentation** *(largely addressed — see "Stroke de-fragmentation"
   above)* — the `_simplify_graph` pass cut D2C primitives −37 % and PatentData
   primitives −67 % by pruning spurs, dissolving degree-2 phantom junctions, and
   merging collinear edges straight through real junctions. Remaining over-segmentation
   comes from two harder, still-open cases:
   - **Thin outline rectangles / concentric rings → double-wall ladders.** A thin
     *outline* rectangle (or a washer / concentric-circle pair) skeletonizes to two
     close parallel rails joined by short rungs; each rail segment between rungs is
     its own primitive. Junction-cluster merging would fix it but unsafely welds the
     two rails — needs a ladder-aware rung-removal that preserves both rails.
   - **Extreme over-segmentation on hatching** — a hatched fill is many genuine short
     parallel lines crossing a boundary, producing hundreds of edges. Needs a
     hatch-region detector that collapses or tags the fill rather than vectorising
     every hatch line.

6. **Dense drawings and runtime outliers** — sketches with extreme hatch/detail can
   still take much longer than the median. Stage 2/3 gates now reject most of these,
   but the full-corpus run should be monitored for long-tail workers and may need a
   per-sketch timeout.

   Dense drawings now de-fragment to ~⅓ the edge count (`_simplify_graph`), but the
   slowest hatched figures can still exceed the time budget — a `max_edges` guard
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
