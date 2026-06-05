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
| 2     | 1-px skeleton       | Stroke graph (JSON)         | **Vectorised CN path** (NumPy ring-slice shifts, ~1 000× vs Python loop); optional Puhachov CNN (ignored by topology — CNN weights can be empty); parallel-edge walk; size-adaptive circularity guard; resolution cap + re-skeletonize; B-spline overshoot guard; noise closed-loop filter |
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
│   ├── filter_patent_data.py        ← content classifier: keep drawings, discard chemistry/text
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

| Key                                | Default | Effect |
|------------------------------------|--------:|--------|
| `sketchcleannet.weights`           | `models/sketchcleannet.pth` | empty `""` ⇒ force classical cleaning mode |
| `sketchcleannet.device`            | `cpu`   | `cuda` for GPU |
| `puhachov.device`                  | `cpu`   | `cuda` for GPU |
| `stage2.max_input_resolution`      | `1000`  | Skeleton images with long edge > this are downsampled before Stage 2. Prevents CNN over-segmentation on large patent TIFs (2000–2700 px). Set to `0` to disable. |
| `stage2.isolation_threshold`       | `0.30`  | Flag sketch if > this fraction of foreground pixels are unreached by any extracted stroke. Calibrated for patent TIF scan noise (p75 isolation ≈ 0.16). |
| `stage2.nms_reference_resolution`  | `512`   | Training resolution of the Puhachov model; NMS radius scales as `nms_radius × max(H,W) / this value` on larger inputs (0 = fixed radius) |
| `stage2.spline_overshoot_limit`    | `5.0`   | Max px a B-spline may exceed the raw pixel bbox; prevents scipy end-effect oscillations |
| `stage2.min_closed_loop_pixels`    | `80`    | Closed loops shorter than this are treated as noise and removed before Stage 3 |
| `stage1.quality_threshold`         | `0.70`  | Sketches below this skeleton quality are flagged for review |
| `stage3.confidence_threshold`      | `0.60`  | Primitives below this are flagged for review |

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

### PatentData content filter

Before running the batch driver, filter the corpus to keep only mechanical
and electrical drawings (discard chemistry formulas, equation pages, dense text):

```bash
# Classify all TIFs — writes output/PatentData/filter_manifest.csv (~30 min, 8 workers)
python -m tools.filter_patent_data --workers 8

# Optionally move discarded TIFs to a quarantine folder
python -m tools.filter_patent_data --workers 8 --move
```

The classifier uses EPO filename letter codes (`_F`/`_A` = figure/assembly → keep;
`_C` = chemistry → likely discard) combined with Hough line detection and connected
component analysis. No GPU or ML model required. On the full 334 835-TIF corpus it
keeps **82.4 %** and discards **17.6 %**.

### Batch driver — PatentData corpus

[`tools/batch_run.py`](tools/batch_run.py) runs the full four-stage pipeline over
the partner patent corpus and writes one row of intrinsic metrics per sketch to a
resumable SQLite database:

```bash
# Phase 0 pilot — 100 random sketches, one per patent
python -m tools.batch_run --limit 100 --stratified

# 1 000-sketch filtered pilot (recommended)
python -m tools.batch_run \
    --limit 1000 --stratified \
    --filter-manifest output/PatentData/filter_manifest.csv \
    --workers 8

# Full corpus, 8 parallel workers
python -m tools.batch_run \
    --filter-manifest output/PatentData/filter_manifest.csv \
    --workers 8
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

### CN vs Puhachov parity check (10 PatentData samples, seed 42)

Direct head-to-head on the same 10 stratified patent sketches
(`output/PatentData_CN/` vs `output/PatentData_Puhachov/`):

| Metric | Puhachov CNN | CN classical | Delta |
|--------|------------:|-------------:|-------|
| Nodes — mean | 994 | 994 | **0** |
| Edges — mean | 1 437 | 1 437 | **0** |
| Primitives — mean | 1 437 | 1 437 | **0** |
| S3 mean confidence | 0.941 | 0.941 | **0** |
| Stage 2 time — mean | 3.33 s | **1.04 s** | **−69 %** |
| Success rate | 100 % | 100 % | 0 |

Every quality metric is identical because `_extract_topology()` computes its own
CN map internally and ignores whatever keypoints the CNN emits. The CNN adds
pure latency (3.3 s vs 1.0 s) with no quality benefit. **CN path is recommended
for all evaluations** (`puhachov.weights: ""`).

Run with:
```bash
# CN path
python -m tools.batch_run --limit 10 --stratified --seed 42 \
    --output output/PatentData_CN --db output/PatentData_CN/results.db \
    --config /tmp/config_patent_cn.yaml   # puhachov.weights: ""

# Puhachov path (identical output, 3× slower)
python -m tools.batch_run --limit 10 --stratified --seed 42 \
    --output output/PatentData_Puhachov --db output/PatentData_Puhachov/results.db
```

### PatentData corpus results (1 000 stratified filtered samples)

Three independent runs on the filtered corpus (one sketch per randomly selected
patent, content-filter applied). All runs used 8 parallel workers.

| Metric | Run 1 seed=42 | Run 2 seed=99 | Run 3 seed=7 |
|--------|-------------:|-------------:|------------:|
| Success rate | **100 %** | **100 %** | **100 %** |
| Total time — mean | 13.5 s | 9.6 s | 10.8 s |
| Total time — median | 7.9 s | 6.2 s | 6.7 s |
| Total time — p90 | 28.3 s | 17.8 s | 18.1 s |
| Total time — p99 | 125.8 s | 76.4 s | 81.7 s |
| S2 edges — median | 858 | 901 | 867 |
| S2 edges — >5 k (dense outliers) | 4.0 % | 4.3 % | 4.0 % |
| S2 isolation flag rate | 4.9 % | 4.8 % | 4.6 % |
| S1 flag rate | 4.9 % | 4.5 % | 4.5 % |
| S3 / S4 flag rate | 0 % | 0 % | 0 % |

Run 1 was the first run after all three Stage 2 fixes were applied (the
`isolation_threshold` was still at the old default of 0.05 for Run 1,
which explains its 93.3 % S2 flag rate — corrected to 0.30 before Runs 2 & 3).

**Full-corpus projection** (275 904 filtered drawings, 8 workers): ~60–90 hours.

---

## Project status

### Completed

- **Full four-stage pipeline** end-to-end, configurable via `config.yaml`
- **Stage 1** — SketchCleanNet inference; classical fallback; bottom-edge
  ghost-ink artefact fixed; stroke-width estimation via distance transform
- **Stage 2** — Puhachov keypoint CNN; CN-cluster skeleton tracing;
  **resolution cap** (`max_input_resolution: 1000`) — skeletons larger than
  1 000 px are dilated, downsampled, and re-skeletonized (Zhang-Suen) before
  the CNN runs, preventing 100×–300× over-segmentation on large patent TIFs;
  **PyTorch thread cap** (`torch.set_num_threads(1)`) forces process-level
  parallelism so batch workers don't contend for CPU cores; **adaptive NMS
  radius** scales junction suppression with image size; **B-spline overshoot
  guard**; **noise closed-loop filter**
- **Stage 3** — RANSAC cascade (line / circle / arc / ellipse / **polygon** /
  polyline); closed polygon fitter; sparse-smooth_pts guard; geometric arc guard;
  Free2CAD Transformer evaluated and retired
- **Stage 4** — SVG and DXF export; ISO 128 layered patent DXF
- **Content classifier** (`tools/filter_patent_data.py`) — feature-based
  (Hough lines + CC analysis + EPO letter codes); classifies 334 835 patent
  TIFs in ~30 min; 82.4 % kept, 17.6 % discarded (chemistry, equations, text)
- **Batch evaluation driver** (`tools/batch_run.py`) — resumable, multi-worker,
  SQLite results; `--filter-manifest` flag to skip non-drawing TIFs
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

### Known limitations / next steps

1. **Stroke fragmentation** *(priority — confirmed on D2C 1 000-sample eval)*
   — 312 / 1 000 D2C cases have `n_prims_out > n_strokes_gt` (31 %). Root causes
   identified so far:
   - **Spurious short halo edges (2–5 px)** at adjacent junction clusters
     (parallel-edge fix creates chains that immediately enter a neighbouring
     keypoint's 1-px extension region). Fix: merge junction clusters within
     `merge_radius` pixels; or filter open edges shorter than a configurable
     minimum before smoothing.
   - **Extreme over-segmentation on hatching paths** — one SVG `<path>` with
     many disconnected `M…L` subpaths (e.g. hatch lines) produces hundreds of
     graph edges that map 1:1 to primitives (`out=404, gt=2`). Fix: a pre-walk
     CC-size gate or a post-walk edge-merge step that joins collinear/near-collinear
     short edges.
   - **Long strokes split at phantom junctions** — CN≥3 pixels appearing at
     near-straight skeleton bends (Zhang-Suen staircase artefact) falsely terminate
     the walk and fragment a single logical stroke into 3–6 pieces. Fix: a post-walk
     chain-merge that stitches edges sharing a junction with CN=3 and a
     near-180° turning angle.

2. **Dense drawings (4 % of corpus)** — sketches with >5 000 edges can take
   60–150+ seconds. Add a `max_edges` guard: if Stage 2 exceeds N edges, flag
   and skip Stage 3/4 rather than processing for minutes.

3. **Thin-stroke position error (1 px)** — Zhang-Suen skeletonisation places the
   skeleton 1 px off-centre for thin strokes (radius ≤ 3 px), causing a systematic
   IoU loss of ~15–20 % for these shapes. Could be improved by distance-transform
   centroid refinement, but requires changes to Stage 1.

4. **Spurious short edges at adjacent junctions** — see item 1 above; the
   parallel-edge fix (bug 14) is the source; merge or length-gate is the fix.

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
