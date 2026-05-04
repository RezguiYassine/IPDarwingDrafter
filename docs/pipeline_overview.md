# IP DrawingDrafter — Vectorization Pipeline
## Claude Code Handoff Context

> **Purpose:** This file gives Claude Code full context to continue building the
> AP3 vectorization pipeline exactly where the Claude.ai session left off.
> Read this entire file before writing any code.

---

## 1. Project Overview

**Project:** IP DrawingDrafter — BayVFP-funded research project (Dec 2025 – Nov 2028)
**Partners:** HAW Landshut (research) × infoapps GmbH, Munich (industry)
**Goal:** Automate generation of editable, patent-compliant technical drawings
from hand sketches and invention disclosures.
**Developer:** Yassine Rezgui, Research Engineer, HAW Landshut

The project has multiple work packages (AP). This pipeline belongs to **AP3 — Vectorization**.

---

## 2. AP3 Pipeline — Full Architecture

The pipeline converts a raw hand sketch into editable SVG and DXF vector files
through four sequential steps. Each step is an independent Python module.

```
Raw sketch (PNG/JPG/PDF)
        │
        ▼
┌─────────────────────┐
│  Stage 0 — Ingest   │  stage0_ingest.py       ← NOT YET BUILT
│  PDF/PNG/JPG → PNG  │
└─────────────────────┘
        │  output/sketches/<id>.png
        ▼
┌─────────────────────────┐
│  Stage 1 — Preprocess   │  stage1_preprocess.py   ✅ BUILT & TESTED
│  SketchCleanNet + thin  │
└─────────────────────────┘
        │  output/cleaned/<id>_cleaned.png
        │  output/cleaned/<id>_skeleton.png
        ▼
┌──────────────────────────────┐
│  Stage 1b — Bezugszeichen   │  stage1b_bezugszeichen.py  ← NOT YET BUILT
│  OCR detect → inpaint →     │
│  store metadata for Stage 4 │
└──────────────────────────────┘
        │  output/cleaned/<id>_skeleton.png  (text-free)
        │  output/graphs/<id>_bezugszeichen.json
        ▼
┌──────────────────────────────┐
│  Stage 2 — Stroke Extraction │  stage2_stroke_extract.py  ← NOT YET BUILT
│  Puhachov et al. CNN →      │
│  stroke graph               │
└──────────────────────────────┘
        │  output/graphs/<id>_graph.json
        ▼
┌──────────────────────────────┐
│  Stage 3 — Primitive Fitting │  stage3_primitive_fit.py   ← NOT YET BUILT
│  Free2CAD → lines/arcs/     │
│  circles with parameters    │
└──────────────────────────────┘
        │  output/primitives/<id>_primitives.json
        ▼
┌──────────────────────────────┐
│  Stage 4 — Export           │  stage4_export.py           ← NOT YET BUILT
│  ezdxf + svgwrite →         │
│  .dxf, .svg, preview image  │
└──────────────────────────────┘
        │  output/svg/<id>.svg
        │  output/dxf/<id>.dxf
        │  output/previews/<id>_preview.png
        ▼
    output/review/              (flagged sketches for manual review)
    logs/pipeline_run_YYYYMMDD.log
```

**Batch runner** (not yet built): `run_pipeline.py --input ./input --output ./output --config config.yaml [--resume] [--workers 4]`

---

## 3. Repository Structure

```
ip-drawing-drafter/
├── src/
│   └── pipeline/
│       ├── run_pipeline.py          ← NOT YET BUILT (CLI batch runner)
│       ├── stage0_ingest.py         ← NOT YET BUILT
│       ├── stage1_preprocess.py     ← ✅ BUILT (see Section 5)
│       ├── stage1b_bezugszeichen.py ← NOT YET BUILT
│       ├── stage2_stroke_extract.py ← NOT YET BUILT
│       ├── stage3_primitive_fit.py  ← NOT YET BUILT
│       ├── stage4_export.py         ← NOT YET BUILT
│       ├── models/
│       │   ├── sketchcleannet.py    ← model wrapper lives inside stage1_preprocess.py
│       │   └── puhachov.py          ← NOT YET BUILT
│       └── utils/
│           ├── logger.py            ← NOT YET BUILT
│           ├── review_queue.py      ← NOT YET BUILT
│           └── geometry.py          ← NOT YET BUILT (shared primitive dataclasses)
├── weights/
│   ├── sketchcleannet.pth           ← MISSING — must be trained (see Section 6)
│   └── puhachov_keypoints.pth       ← MISSING — must be downloaded (see Section 7)
├── input/                           ← drop input PDFs/PNGs here
├── output/                          ← pipeline writes here
│   ├── sketches/
│   ├── cleaned/
│   ├── graphs/
│   ├── primitives/
│   ├── svg/
│   ├── dxf/
│   ├── previews/
│   └── review/
├── logs/
├── config.yaml                      ← ✅ EXISTS (see Section 5)
└── requirements.txt                 ← NOT YET CREATED
```

---

## 4. Technology Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Preprocessing / cleaning | SketchCleanNet (U-Net) | Trained on engineering CAD sketches |
| Stroke extraction | Puhachov et al. (SIGGRAPH Asia 2021) | Best junction accuracy; matches our Step 2/Step 3 split |
| Primitive fitting | Free2CAD (SIGGRAPH 2022) | Only DL model trained on engineering sketches → editable CAD output |
| DXF export | ezdxf | Full DXF entity support, actively maintained |
| SVG export | svgwrite | Lightweight, direct primitive mapping |
| Bezugszeichen OCR | pytesseract or easyocr | Detect reference numerals before vectorization |

**Hardware:** CUDA GPU available. All DL inference runs on GPU.
**Language:** Python 3.10+
**Key dependencies:** torch, torchvision, opencv-python, scikit-image, ezdxf, svgwrite, pdf2image, pytesseract/easyocr, pyyaml, tqdm

---

## 5. What Is Already Built

### `stage1_preprocess.py` — ✅ Complete and tested

**Location:** `src/pipeline/stage1_preprocess.py`

**What it does:**
- Loads SketchCleanNet (U-Net) from weights file if available; falls back to
  classical cleaning (Otsu + morphological ops) if weights are missing
- Applies Zhang-Suen thinning (scikit-image `skeletonize`) to produce 1px skeleton
- Computes a `skeleton_quality` confidence score in [0, 1]
- Flags sketches below `stage1.quality_threshold` (default 0.70) for manual review

**Public API:**
```python
from stage1_preprocess import run, load_model, Stage1Result

model = load_model(config)   # call once at batch start; returns None if no weights

result: Stage1Result = run(
    input_path = Path("output/sketches/sketch_001.png"),
    output_dir = Path("output"),
    sketch_id  = "sketch_001",
    config     = config,       # parsed config.yaml dict
    model      = model,        # SketchCleanNet instance or None
)
# result.cleaned_path     → output/cleaned/sketch_001_cleaned.png
# result.skeleton_path    → output/cleaned/sketch_001_skeleton.png
# result.skeleton_quality → float in [0, 1]
# result.flagged          → bool
```

**CLI usage (standalone test):**
```bash
python stage1_preprocess.py path/to/sketch.png --output ./output --config config.yaml --id sketch_001
```

**Test result (verified):**
- 300×400px synthetic sketch with lines, circle, 2% noise
- Skeleton: 1039 foreground pixels, 99.4% genuinely 1px wide
- Quality score: 1.000 (no flagging)
- Processing time: 0.03s on CPU (classical fallback)

**Current state:** Runs in classical fallback mode (no weights). SketchCleanNet
inference path is fully wired — it activates automatically once weights are placed
at the path specified in `config.yaml`.

---

### `config.yaml` — ✅ Complete

```yaml
sketchcleannet:
  weights: ""          # ← SET THIS once weights are ready
  device: "cuda"

stage1:
  quality_threshold: 0.70
  classical:
    blur_kernel: 3
    adaptive_block: 35
    adaptive_C: 10
    blend_alpha: 0.6
    morph_kernel: 2
    min_cc_size: 30

stage2:
  isolation_threshold: 0.05

stage3:
  confidence_threshold: 0.60

pipeline:
  workers: 4
  resume: true
  log_level: "INFO"
```

---

## 6. ⚠️ PRIORITY TASK: SketchCleanNet — Download, Train, Produce Weights

`stage1_preprocess.py` contains the full U-Net architecture and inference code,
but **has no pretrained weights**. The classical fallback is active until weights
are produced. This must be completed before the pipeline can use DL-based cleaning.

### Step-by-step instructions

#### Step 1 — Clone the SketchCleanNet repository
```bash
git clone https://github.com/BardOfCodes/SketchCleanNet.git
cd SketchCleanNet
pip install -r requirements.txt
```

> If that repository is unavailable or has changed, the U-Net architecture
> defined inside `stage1_preprocess.py` (class `_UNet`) is self-contained and
> can be trained independently. It matches the architecture described in:
> Manda et al., "SketchCleanNet", Computers & Graphics 107 (2022), pp. 73–83.

#### Step 2 — Prepare training data
SketchCleanNet requires pairs of (rough sketch, clean sketch). Suitable datasets:

| Dataset | Content | Link |
|---------|---------|------|
| **CADSketchNet** | Engineering CAD hand sketches | Zenodo / GitHub |
| **OpenSketch** | Professional design sketches | OpenSketch project page |
| **SketchyScene** | General sketches (supplementary) | GitHub |
| **Project data** | Invention disclosure hand sketches from partners (ThyssenKrupp Presta, Elmos, Biedermann) | Internal — infoapps GmbH |

For training pairs: synthesize clean targets from CAD vector data by rendering
clean line drawings, then augment with noise, variable stroke width, and scan
artefacts to create the rough-sketch input side.

#### Step 3 — Train the model
```bash
# Using the SketchCleanNet repo training script (adapt paths as needed):
python train.py \
    --data_dir   ./data/pairs/ \
    --output_dir ./checkpoints/ \
    --epochs     100 \
    --batch_size 8 \
    --lr         1e-4 \
    --device     cuda

# Or train the _UNet defined in stage1_preprocess.py directly:
# (see training script skeleton below)
```

Minimal training script targeting the U-Net in `stage1_preprocess.py`:
```python
import torch
import torch.nn as nn
from stage1_preprocess import _UNet
from torch.utils.data import DataLoader
# ... define SketchDataset, training loop, save checkpoint ...
# Loss: L1 + perceptual (VGG feature) loss is recommended for sketch cleanup
# Save: torch.save({"model_state_dict": model.state_dict()}, "weights/sketchcleannet.pth")
```

#### Step 4 — Place weights and update config
```bash
cp checkpoints/best_model.pth weights/sketchcleannet.pth
```

In `config.yaml`:
```yaml
sketchcleannet:
  weights: "weights/sketchcleannet.pth"
  device: "cuda"
```

Stage 1 will automatically switch from classical fallback to SketchCleanNet
on the next run — no code changes needed.

#### Validation checklist after training
- [ ] Run `python stage1_preprocess.py test_sketch.png --config config.yaml`
- [ ] Confirm `model_used: sketchcleannet` in output (not `classical`)
- [ ] Check that `skeleton_quality` score is ≥ 0.70 on clean engineering sketches
- [ ] Visually inspect `_cleaned.png` — strokes should be crisp, noise removed

---

## 7. Next Stages to Build (in order)

### Priority order
1. **`stage1b_bezugszeichen.py`** — OCR + inpainting (blocks Stage 2 correctness)
2. **`stage0_ingest.py`** — PDF/PNG/JPG normalisation (needed for real batch runs)
3. **`stage2_stroke_extract.py`** — Puhachov et al. keypoint CNN → stroke graph
4. **`stage3_primitive_fit.py`** — Free2CAD → geometric primitives
5. **`stage4_export.py`** — ezdxf + svgwrite → .dxf, .svg, preview
6. **`run_pipeline.py`** — CLI batch runner tying all stages together
7. **`utils/logger.py`** + **`utils/review_queue.py`** + **`utils/geometry.py`**

---

### Stage 1b — Bezugszeichen (`stage1b_bezugszeichen.py`)

**Input:** `output/cleaned/<id>_cleaned.png` (cleaned raster, before thinning)
**Output:**
- `output/cleaned/<id>_skeleton.png` — skeleton with text regions inpainted
- `output/graphs/<id>_bezugszeichen.json` — detected text metadata

**What it must do:**
1. Run OCR (pytesseract or easyocr) on the cleaned raster
2. Filter detections to patent reference numeral patterns:
   - Single or double digits (1–99 typical range)
   - Single uppercase letters (A, B, C…)
   - Reject full words (those are titles/labels, not Bezugszeichen)
3. For each accepted detection: store bounding box, text content, centroid
4. Inpaint detected regions using `cv2.inpaint` (INPAINT_TELEA method)
5. Apply Zhang-Suen thinning to the inpainted image → clean skeleton
6. Save Bezugszeichen JSON sidecar for Stage 4 re-insertion

**Bezugszeichen JSON schema:**
```json
{
  "sketch_id": "sketch_001",
  "items": [
    {
      "id": 0,
      "text": "12",
      "bbox": {"x": 145, "y": 87, "w": 18, "h": 14},
      "centroid": [154, 94],
      "anchor_edge_id": null
    }
  ]
}
```
`anchor_edge_id` is filled by Stage 4 (nearest stroke graph edge).

**Confidence signal:** fraction of detected text regions with OCR confidence < 0.5 → flag

---

### Stage 2 — Stroke Extraction (`stage2_stroke_extract.py`)

**Model:** Puhachov et al., "Keypoint-Driven Line Drawing Vectorization via
PolyVector Flow", ACM Trans. Graph. 40(6), SIGGRAPH Asia 2021.

**Weights download:**
```bash
# Repository with pretrained weights:
git clone https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow
# Weights are available in the releases section of the repository
```

**Input:** `output/cleaned/<id>_skeleton.png` (1px binary, text-free)
**Output:** `output/graphs/<id>_graph.json`

**Graph JSON schema:**
```json
{
  "sketch_id": "sketch_001",
  "image_shape": [300, 400],
  "nodes": [
    {"id": 0, "x": 142, "y": 87, "type": "junction"},
    {"id": 1, "x": 20,  "y": 87, "type": "endpoint"}
  ],
  "edges": [
    {
      "id": 0,
      "source": 0,
      "target": 1,
      "pixels": [[142,87],[141,87],[140,87]],
      "is_closed": false
    }
  ]
}
```

**Confidence signal:** ratio of orphaned foreground pixels (not in any edge).
Threshold: `stage2.isolation_threshold` = 0.05 (5%).

---

### Stage 3 — Primitive Fitting (`stage3_primitive_fit.py`)

**Model:** Free2CAD (Li, Pan, Bousseau, Mitra — SIGGRAPH 2022)
- Paper: https://arxiv.org/abs/2204.01977
- Repository: https://github.com/Enigma-li/Free2CAD

**Input:** `output/graphs/<id>_graph.json` + `output/cleaned/<id>_cleaned.png`
**Output:** `output/primitives/<id>_primitives.json`

**Primitives JSON schema:**
```json
{
  "sketch_id": "sketch_001",
  "primitives": [
    {"edge_id": 0, "type": "line",   "start": [142.3, 87.1], "end": [312.7, 87.4], "confidence": 0.95},
    {"edge_id": 5, "type": "arc",    "center": [200.0, 150.0], "radius": 48.2, "start_angle": 0.0, "end_angle": 90.0, "confidence": 0.88},
    {"edge_id": 9, "type": "circle", "center": [400.0, 300.0], "radius": 30.1, "confidence": 0.91}
  ]
}
```

**Confidence signal:** mean primitive confidence across all edges.
Threshold: `stage3.confidence_threshold` = 0.60.

**Fallback:** If Free2CAD is unavailable or fails on an edge, fall back to
RANSAC fitting (line/arc/circle) on the pixel chain from the stroke graph.
This ensures the pipeline always produces output even if the DL model fails.

---

### Stage 4 — Export (`stage4_export.py`)

**Input:** `output/primitives/<id>_primitives.json` + `output/graphs/<id>_bezugszeichen.json`
**Output:**
- `output/svg/<id>.svg`     — SVG with `<line>`, `<path>`, `<circle>`, `<text>`
- `output/dxf/<id>.dxf`     — DXF with LINE, ARC, CIRCLE, TEXT entities
- `output/previews/<id>_preview.png` — coloured overlay on original sketch

**DXF entity mapping:**
| Primitive | DXF entity | SVG element |
|-----------|-----------|-------------|
| line      | LINE      | `<line>`    |
| arc       | ARC       | `<path>` (arc command) |
| circle    | CIRCLE    | `<circle>`  |
| Bezugszeichen text | TEXT / MTEXT | `<text>` |

**Standards compliance:**
- ISO 128 (general principles of technical drawing)
- IEC 60617 (electrical/electronic symbols)
- EPO Rule 46 (patent drawing requirements)
- DXF version: R2010 (AC1024) — compatible with AutoCAD, SolidWorks, FreeCAD

**Preview colour scheme:**
- Lines → blue `#2196F3`
- Arcs → green `#4CAF50`
- Circles → orange `#FF9800`
- Bezugszeichen boxes → red `#F44336`

---

### Batch Runner (`run_pipeline.py`)

```bash
python run_pipeline.py \
    --input   ./input \
    --output  ./output \
    --config  config.yaml \
    --resume \
    --workers 4
```

**Per-sketch status:** `SUCCESS` | `FLAGGED` | `FAILED`
**Resume logic:** skip sketches where all output files already exist
**Review report:** `output/review/review_report.html` — browser-viewable list of
flagged sketches with preview images and per-signal confidence scores

---

## 8. Confidence & Flagging System

Three independent signals, each with a configurable threshold:

| Signal | Stage | Config Key | Default | Meaning |
|--------|-------|-----------|---------|---------|
| `skeleton_quality` | 1 | `stage1.quality_threshold` | 0.70 | Fraction of thin pixels in skeleton |
| `isolation_ratio` | 2 | `stage2.isolation_threshold` | 0.05 | Fraction of orphaned fg pixels |
| `primitive_confidence` | 3 | `stage3.confidence_threshold` | 0.60 | Mean Free2CAD sequence probability |

If **any** signal breaches its threshold → sketch is copied to `output/review/`
and marked `FLAGGED` in the log. The batch continues processing remaining sketches.

---

## 9. Key Domain Requirements (Patent-Specific)

These must be respected throughout the pipeline:

- **Bezugszeichen (reference numerals):** Must be detected and removed before
  vectorization, then re-inserted as text entities in the final SVG/DXF.
  They typically appear as single/double digits near strokes (EPO Rule 46).

- **No dimensions:** Patent drawings must not contain dimension lines or
  measurements. Do not generate dimension entities in DXF output.

- **No colours:** Patent office drawings are black and white only.
  SVG strokes must be `stroke="black"`, `fill="none"`.

- **Orthographic strokes:** Lines are predominantly horizontal, vertical,
  or at 45°. The primitive fitter should use this as a prior where applicable.

- **Closed contours:** Circles and rectangles are common (gear profiles,
  cross-sections). The stroke extractor must detect these as closed loop
  edges, not open chains.

- **Editability:** All output must be editable in standard CAD tools
  (AutoCAD, SolidWorks, FreeCAD, Inkscape).

---

## 10. Testing Strategy

Each stage should be tested independently before integration:

```bash
# Stage 0 (once built)
python stage0_ingest.py --input ./input --output ./output

# Stage 1 (already working)
python stage1_preprocess.py ./input/test.png --output ./output --config config.yaml

# Stage 1b (once built)
python stage1b_bezugszeichen.py ./output/cleaned/test_cleaned.png --output ./output

# Stage 2 (once built)
python stage2_stroke_extract.py ./output/cleaned/test_skeleton.png --output ./output

# Stage 3 (once built)
python stage3_primitive_fit.py --graph ./output/graphs/test_graph.json \
       --image ./output/cleaned/test_cleaned.png --output ./output

# Stage 4 (once built)
python stage4_export.py --primitives ./output/primitives/test_primitives.json \
       --bezugszeichen ./output/graphs/test_bezugszeichen.json --output ./output

# Full batch
python run_pipeline.py --input ./input --output ./output --config config.yaml
```

Use the synthetic test sketch generator from the Claude.ai session for initial
testing (lines, circles, noise — no need for real patent sketches yet):
```python
import numpy as np, cv2
img = np.ones((300, 400), dtype=np.uint8) * 255
cv2.line(img, (20, 150), (380, 150), 0, 3)
cv2.line(img, (200, 20), (200, 280), 0, 3)
cv2.circle(img, (100, 230), 40, 0, 2)
cv2.line(img, (260, 60), (360, 260), 0, 3)
rng = np.random.default_rng(42)
img[rng.random(img.shape) < 0.02] = 0
cv2.imwrite("input/test_sketch.png", img)
```

---

## 11. Session History Summary

This handoff was prepared from a Claude.ai conversation covering:

1. Clarification of the role of each vectorization step (confusion between
   topology/geometry/preprocessing was resolved explicitly)
2. Full literature review of AI-based vectorization (Steps 1–4), including
   SketchCleanNet, Puhachov et al., Deep Sketch Vectorization (Yan 2024 SIGGRAPH),
   Free2CAD, DeepSVG, and classical baselines — saved as a Word document
3. Architecture design of the full pipeline (all 7 stages + batch runner)
4. Implementation and testing of `stage1_preprocess.py`
5. This handoff document

**The immediate next task is:**
1. Set up SketchCleanNet training (Section 6 of this document)
2. Build `stage1b_bezugszeichen.py`
3. Continue with remaining stages in the order listed in Section 7
