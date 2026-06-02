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
| 1     | Raw raster (PNG)    | Cleaned image + 1-px skeleton + stroke-width estimate | SketchCleanNet (or classical fallback) |
| 2     | 1-px skeleton       | Stroke graph (JSON)         | Puhachov keypoint CNN + graph builder; adaptive NMS radius; B-spline overshoot guard; noise closed-loop filter |
| 3     | Stroke graph        | Geometric primitives (JSON) | RANSAC cascade (line / circle / arc / ellipse / **polygon** / polyline) |
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
| `puhachov_keypoints.pth`     | 22 MB | Stage 2              | shipped in `models/`        |
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

| Key                                | Default                       | Effect                                          |
|------------------------------------|-------------------------------|-------------------------------------------------|
| `sketchcleannet.weights`           | `models/sketchcleannet.pth`   | empty `""` ⇒ force classical cleaning mode     |
| `sketchcleannet.device`            | `cpu`                         | `cuda` for GPU                                  |
| `puhachov.device`                  | `cpu`                         | `cuda` for GPU                                  |
| `stage2.nms_reference_resolution`  | `512`                         | training resolution of the Puhachov model; NMS radius scales as `nms_radius × max(H,W) / this value` on larger inputs (0 = fixed radius) |
| `stage2.spline_overshoot_limit`    | `5.0`                         | max px a B-spline may exceed the raw pixel bbox; prevents scipy end-effect oscillations turning straight edges into curves |
| `stage2.min_closed_loop_pixels`    | `80`                          | closed loops shorter than this are treated as noise and removed before Stage 3 |
| `stage1.quality_threshold`         | `0.70`                        | sketches below this are flagged for review      |
| `stage3.confidence_threshold`      | `0.60`                        | primitives below this are flagged for review    |

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

### Batch driver — PatentData corpus

[`tools/batch_run.py`](tools/batch_run.py) runs the full four-stage pipeline over
the partner patent corpus and writes one row of intrinsic metrics per sketch to a
resumable SQLite database:

```bash
# Phase 0 pilot — 100 random sketches, one per patent
python -m tools.batch_run --limit 100 --stratified

# Full corpus, 8 parallel workers
python -m tools.batch_run --workers 8
```

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
printed at the end of each run:

```
──────────────────────────────────────────────────────
  Drawing2CAD eval — 1000/1000 ok  0 errors
──────────────────────────────────────────────────────
  Chamfer distance on skeletons (px, lower = better)
    mean       4.71    p75      1.19
    median     1.00    p95     34.50
──────────────────────────────────────────────────────
  Secondary (pixel IoU)
    iou_pixel    0.62    iou_skel    0.52
    recall       0.80    precision   0.70
──────────────────────────────────────────────────────
```

### Drawing2CAD baseline results (1 000 test-set samples, Front view, stroke-width fix applied)

| Metric | Value | Notes |
|--------|------:|-------|
| Chamfer sym — mean | 4.71 px | headline; pulled up by a tail of outliers |
| Chamfer sym — median | 1.00 px | typical sample is within 1 px of GT skeleton |
| Chamfer sym — p75 | 1.19 px | |
| Chamfer sym — p95 | 34.5 px | ~5 % of samples have poorly placed strokes |
| Pixel IoU | 0.62 | up from 0.42 before stroke-width fix |
| Skeleton IoU | 0.52 | |
| Recall | 0.80 | up from 0.44 before stroke-width fix |
| Precision | 0.70 | |

> **Note on stroke-width handling.** Stage 1 estimates the original ink thickness
> via distance transform and stores it in `Stage1Result.mean_stroke_width`. The
> D2C eval harness passes this to Stage 3 (embedded in the primitives JSON) so
> Stage 4 renders SVG strokes at the correct visual thickness for metric
> comparison. The patent production path (`batch_run.py`) omits this — it keeps
> ISO 128 standard lineweights, which is correct for CAD output.

### PatentData corpus results (1 000 stratified samples, 4 workers)

| Metric | Baseline (no fix) | Notes |
|--------|------------------:|-------|
| Stage 2 time — mean | 84.7 s | Puhachov CNN on full-size 1400–2700 px images |
| Stage 2 nodes — median | ~272 k | ~350× over-segmentation at high resolution |
| Primitives / sketch — median | 1 617 | expected 20–200 for typical patent drawings |

The over-segmentation root cause: the CNN's NMS radius (5 px) was fixed regardless
of image size. On 2500 px patent TIFs that is proportionally 5× too small compared
to the ~512 px images the model was trained on, so thousands of duplicate junctions
survive suppression.

**Current fix — adaptive NMS radius** (see [Project status](#project-status)):
the radius is scaled as `nms_radius × max(H,W) / 512` so suppression remains
geometrically consistent at any input resolution. The CNN still runs at full
resolution; only the duplicate-suppression window is widened.
Re-run `python -m tools.batch_run --limit 1000 --stratified --no-resume` to
record updated numbers.

---

## Project status

### Completed

- **Full four-stage pipeline** end-to-end, configurable via `config.yaml`
- **Stage 1** — SketchCleanNet inference; classical fallback; bottom-edge
  ghost-ink artefact fixed; stroke-width estimation via distance transform
- **Stage 2** — Puhachov keypoint model inference; topological closed-loop
  reordering; **adaptive NMS radius** (scales with image size) keeps junction
  suppression proportional on large patent TIFs; **B-spline overshoot guard**
  prevents scipy end-effect oscillations from turning straight skeleton edges
  into curves (was the main cause of lines → wiggly polylines in SVG output);
  **noise closed-loop filter** removes sub-80-px skeleton blobs before Stage 3
  (was the cause of spurious small circles in SVG output)
- **Stage 3** — RANSAC cascade (line / circle / arc / ellipse / **polygon** /
  polyline); **closed polygon fitter** fits rectangular and other angular closed
  loops as clean N-vertex polygons instead of raw 300-pt pixel traces (was the
  cause of rectangles rendering as round-cornered polylines); geometric arc guard
  prevents straight skeletons being fit as high-radius arcs; Free2CAD Transformer
  evaluated and retired (6× slower, no accuracy gain — see [`docs/archive/`](docs/archive/))
- **Stage 4** — SVG and DXF export; ISO 128 layered patent DXF with Bezugszeichen
  (EPO Rule 46); SVG stroke-width driven by measured source thickness for D2C eval;
  **polygon** primitive exported as `<polygon>` in SVG and closed `lwpolyline` in DXF
- **Batch evaluation driver** (`tools/batch_run.py`) — resumable, multi-worker,
  SQLite results
- **Drawing2CAD eval harness** (`tools/d2c_eval.py`) — rasterize → pipeline →
  compare; Chamfer distance as headline metric; pixel IoU / precision / recall
  as secondary

### Next steps

1. **Re-run the 1 000-sample patent batch eval** (`--no-resume`) to measure the
   combined effect of the B-spline fix, noise filter, and polygon fitter on
   primitive counts and mean confidence across the corpus.

2. **Investigate zero-output samples** — six D2C test samples produce no primitives
   (empty stroke graph despite no crash). Check whether Stage 1 produces a blank
   skeleton on these and tune the binarization threshold if so.

3. **Reduce the Chamfer p95 outliers** — the p95 is ~35 px while the median is
   1 px, indicating a small fraction of samples with badly placed strokes.
   Profile these to determine whether the failure is in Stage 1 (skeleton
   quality), Stage 2 (missed strokes), or Stage 3 (bad primitive fit).

4. **Speed up Stage 2** — the Puhachov CNN is the dominant runtime cost on large
   patent TIFs. Switch `puhachov.device` to `cuda` in `config.yaml` if a GPU is
   available.

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
