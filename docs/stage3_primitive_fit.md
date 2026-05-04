# IP DrawingDrafter — Stage 3: Primitive Fitting
## Claude Code Handoff Context

> **Purpose:** This file gives Claude Code full context to continue building
> and validating Stage 3 (Primitive Fitting) of the AP3 Vectorization Pipeline.
> Stages 1 and 2 are already implemented and tested. Read this entire file
> before writing or running any code.

---

## 1. Pipeline Status

```
Stage 0 — Ingest            ← NOT YET BUILT
Stage 1 — Preprocessing     ✅ BUILT & TESTED  (stage1_preprocess.py)
Stage 2 — Stroke Extraction ✅ BUILT & TESTED  (stage2_stroke_extract.py)
Stage 3 — Primitive Fitting ⚠  PARTIALLY BUILT (stage3_primitive_fit.py)
                               RANSAC fallback: done & tested
                               Free2CAD DL path: wrapper stub only
                               Egiazarian et al.: NOT YET BUILT
Stage 4 — Export            ← NOT YET BUILT
```

**Your task in this session:**
1. Integrate Free2CAD (Li et al., SIGGRAPH 2022) as the primary DL fitter
2. Implement Egiazarian et al. (ECCV 2020) as a second DL fitter
3. Benchmark the three approaches (RANSAC / Free2CAD / Egiazarian) on
   a shared test set and document results

---

## 2. What Stage 3 receives (Stage 2 output)

Stage 2 (Puhachov CNN + CN-cluster skeleton tracing) produces one JSON file
per sketch:

```
output/graphs/<sketch_id>_graph.json
```

### Graph JSON schema

```json
{
  "sketch_id": "test_001",
  "image_shape": [300, 400],
  "nodes": [
    {"id": 0, "x": 200, "y": 21,  "type": "endpoint",     "confidence": 1.0},
    {"id": 6, "x": 201, "y": 151, "type": "junction",     "confidence": 1.0},
    {"id": 9, "x": 100, "y": 230, "type": "loop_anchor",  "confidence": 1.0}
  ],
  "edges": [
    {
      "id": 0,
      "source": 0,
      "target": 6,
      "pixels": [[200,21],[200,22],"..."],
      "smooth_pts": [[200.0,21.0],[200.1,45.3],"..."],
      "is_closed": false
    },
    {
      "id": 8,
      "source": 9,
      "target": 9,
      "pixels": [[60,190],[61,190],"..."],
      "smooth_pts": [[60.0,190.0],"..."],
      "is_closed": true
    }
  ]
}
```

**Key field notes for Stage 3:**
- `smooth_pts`: RDP-simplified + B-spline smoothed coordinates (float).
  Use for open edges (lines, arcs, ellipses).
- `pixels`: raw 1px skeleton pixel chain (integer). Use for closed loops
  (circles, ellipses) — smooth_pts are unordered for loops and
  produce unreliable spline fits.
- `is_closed`: True = the edge is a closed contour (circle, rectangle,
  ellipse). False = open stroke.
- `source == target` for closed loops (self-loop in the graph).

---

## 3. What Stage 3 must produce

One JSON file per sketch:

```
output/primitives/<sketch_id>_primitives.json
```

### Primitives JSON schema

```json
{
  "sketch_id": "test_001",
  "primitives": [
    {
      "edge_id": 0,
      "type": "line",
      "start": [200.969, 22.0],
      "end":   [201.015, 150.0],
      "confidence": 0.957
    },
    {
      "edge_id": 5,
      "type": "arc",
      "center": [200.0, 150.0],
      "radius": 48.2,
      "start_angle": 0.0,
      "end_angle": 90.0,
      "confidence": 0.88
    },
    {
      "edge_id": 8,
      "type": "circle",
      "center": [100.704, 230.674],
      "radius": 39.975,
      "confidence": 0.877
    },
    {
      "edge_id": 12,
      "type": "ellipse",
      "center": [300.0, 200.0],
      "a": 60.0,
      "b": 35.0,
      "angle": 15.0,
      "confidence": 0.72
    },
    {
      "edge_id": 7,
      "type": "polyline",
      "points": [[10.0,20.0],[15.5,25.3]],
      "confidence": 0.50
    }
  ]
}
```

**Primitive types and parameters:**

| Type | Parameters | Notes |
|------|-----------|-------|
| `line` | `start [x,y]`, `end [x,y]` | Most common in patent drawings |
| `arc` | `center [x,y]`, `radius`, `start_angle`, `end_angle` (degrees) | Angles: 0=east, CCW positive |
| `circle` | `center [x,y]`, `radius` | Always from `is_closed=True` edges |
| `ellipse` | `center [x,y]`, `a` (semi-major), `b` (semi-minor), `angle` (degrees) | Rotation of major axis |
| `polyline` | `points [[x,y],...]` | Fallback; also for freeform curves |

---

## 4. What is already implemented in stage3_primitive_fit.py

### RANSAC fallback (fully working, tested)

Five fitting functions, all pure Python + NumPy:

- `_fit_line_ransac(pts)` — SVD-based line fit, handles vertical lines
- `_fit_circle_ransac(pts)` / `_fit_circle_algebraic(pts)` — Taubin method
- `_arc_from_circle_fit(pts, circle)` — extracts arc angles from circle fit
- `_fit_ellipse_ransac(pts)` / `_fit_ellipse_algebraic(pts)` — Fitzgibbon method
- `fit_edge_ransac(edge)` — priority selector: circle (closed) → line → arc →
  ellipse → polyline

**Critical design decision:** for `is_closed=True` edges, `fit_edge_ransac`
uses `edge["pixels"]` (raw), NOT `edge["smooth_pts"]`. This is because
smooth_pts are ordered by coordinate sort for closed loops, not by arc angle,
making the spline fit scatter across the loop rather than tracing it.

### Verified test results (synthetic sketch: 4 lines + 1 circle)

```
edge  0  LINE    conf=0.957    ✓ vertical line top half
edge  1  LINE    conf=0.867    ✓ diagonal
edge  2  LINE    conf=1.000    ✓ horizontal left
edge  3  LINE    conf=1.000    ✓ horizontal right
edge  4  LINE    conf=0.822    ✓ diagonal right half
edge  5  LINE    conf=1.000    ✓ vertical line bottom half
edge  6  LINE    conf=1.000    ✓ horizontal middle segment
edge  7  POLYLINE pts=2        ✓ micro-edge (junction artefact)
edge  8  CIRCLE  r=39.975      ✓ drawn at r=40; center=(100.7, 230.7)
Mean confidence: 0.869  Flagged: no
```

### Free2CAD wrapper (stub — needs full implementation)

`Free2CADFitter` class exists with:
- Constructor that validates `repo_path` and `weights_path` from config
- `_load()` that imports `from network.model import CADLModel`
- `fit_edge()` that rasterises the stroke and calls the model
- `_parse_free2cad_output()` — **placeholder parser, must be updated**
  once the actual Free2CAD output format is confirmed from the repo

`load_model(config)` follows the weights-optional pattern: returns `None`
if Free2CAD is unavailable → all edges use RANSAC.

---

## 5. Task A — Complete the Free2CAD integration

### Step 1: Clone and install Free2CAD

```bash
git clone https://github.com/Enigma-li/Free2CAD.git
cd Free2CAD
pip install -r requirements.txt
```

Free2CAD may need a specific PyTorch version — check its README.
It may also require its own conda environment:

```bash
conda create -n free2cad python=3.8
conda activate free2cad
pip install -r requirements.txt
```

### Step 2: Download weights

Check the Free2CAD releases page or README for the pretrained checkpoint.
Expected filename pattern: `*.pth` or `*.pkl`.

```bash
# Place weights at:
weights/free2cad_model.pth
```

### Step 3: Update config.yaml

Add the following block:

```yaml
free2cad:
  repo_path: "path/to/Free2CAD"          # absolute or relative to project root
  weights:   "weights/free2cad_model.pth"
  device:    "cuda"                        # or "cpu"
```

### Step 4: Inspect the actual output format

Before updating `_parse_free2cad_output`, run the Free2CAD demo on a test
image to understand what the model returns:

```python
import sys
sys.path.insert(0, "path/to/Free2CAD")
from network.model import CADLModel
import torch

model = CADLModel()
ckpt  = torch.load("weights/free2cad_model.pth", map_location="cpu")
model.load_state_dict(ckpt.get("model_state_dict") or ckpt)
model.eval()

# Feed a test tensor and inspect output type and shape
dummy = torch.zeros(1, 1, 256, 256)
with torch.no_grad():
    out = model(dummy)
print(type(out), out.shape if hasattr(out, "shape") else "no shape")
```

Update `_parse_free2cad_output()` in `stage3_primitive_fit.py` to match
the real output format. The current stub uses `output.logits` or
`output[0]` as a placeholder.

### Step 5: Validate

```bash
python src/pipeline/stage3_primitive_fit.py \
    output/graphs/test_001_graph.json \
    --output output \
    --config src/pipeline/config.yaml

# Expected: fitter_used = "free2cad"  (not "ransac")
```

---

## 6. Task B — Implement Egiazarian et al. (ECCV 2020)

### Paper reference

> Egiazarian, V. et al. **Deep Vectorization of Technical Drawings**.
> ECCV 2020. Springer LNCS 12358, pp. 582–598.
> https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579.pdf

### Why implement it

Egiazarian et al. is the most directly relevant work for the IP DrawingDrafter
domain — it was specifically designed for **technical line drawings** (floor
plans, 2D CAD images), not artistic sketches. It outperforms general-purpose
vectorizers on this exact domain. Implementing it alongside Free2CAD allows a
controlled benchmark on patent sketch data.

### Architecture summary (for implementation reference)

The pipeline has three stages that map to our Stage 3:

1. **Deep CNN cleaning stage** — a convolutional network that eliminates
   background noise and fills missing parts. For our pipeline this is already
   done by Stage 1 (SketchCleanNet), so this stage can be skipped.

2. **Transformer-based primitive regression** — a patch-based Transformer
   that estimates primitive placements per image patch (512×512 tiles with
   overlap). Each patch outputs a set of candidate line-segment endpoints.
   Input: clean binary raster. Output: candidate primitive parameters per patch.

3. **Iterative optimisation** — an optimisation step that refines the primitive
   configurations across the full image, resolving conflicts between overlapping
   patch predictions.

**For our pipeline, Stage 3 only needs the Transformer (step 2) + optimisation
(step 3)**, feeding them the already-clean skeleton from Stage 1.

### Repository

```bash
git clone https://github.com/egiazarian/deep-vectorization-of-technical-drawings.git
cd deep-vectorization-of-technical-drawings
pip install -r requirements.txt
```

If the original repo is unavailable, the paper describes the architecture in
sufficient detail for re-implementation. The core idea is:
- Split the skeleton image into 512×512 overlapping patches (64px overlap)
- For each patch: a Transformer outputs N candidate line segments as
  (x1,y1,x2,y2) normalised to [0,1] in the patch coordinate system
- Merge candidates across overlapping patches (NMS on endpoint proximity)
- Run a final L-BFGS / Adam optimisation pass that adjusts endpoint positions
  to minimise a pixel-distance energy

### Integration plan

Add a second DL fitter class to `stage3_primitive_fit.py`:

```python
class EgiazarianFitter:
    """
    Wrapper around Egiazarian et al. (ECCV 2020) patch-based Transformer.
    Designed for technical line drawings — the most relevant model for patents.
    Input: full cleaned skeleton image (not individual edges).
    Output: set of line segments for the entire image, then matched to edges.
    """

    def __init__(self, repo_path, weights_path, device="cuda"):
        ...  # same weights-optional pattern

    def fit_sketch(self, skeleton_image, edges):
        """
        Run Egiazarian on the full skeleton image.
        Returns a dict mapping edge_id -> primitive dict.

        Note: Egiazarian works on the full image, not per-edge.
        After fitting, match each output primitive to the nearest edge
        in the stroke graph by proximity of endpoints.
        """
        ...
```

**Key difference from Free2CAD:** Egiazarian operates on the **full image**
and outputs a global set of primitives, which must then be matched back to
graph edges. Free2CAD and RANSAC both operate per-edge.

Update `config.yaml` with:

```yaml
egiazarian:
  repo_path: "path/to/deep-vectorization-of-technical-drawings"
  weights:   "weights/egiazarian_model.pth"
  device:    "cuda"
  tile_size: 512
  overlap:   64
```

Update `load_model()` and `run()` in `stage3_primitive_fit.py` to accept
and use the Egiazarian fitter when available.

---

## 7. Task C — Benchmark the three approaches

Once both DL fitters are integrated, run a benchmark comparing:

| Approach | Fitter class | Trigger |
|----------|-------------|---------|
| RANSAC | `fit_edge_ransac()` | always available |
| Free2CAD | `Free2CADFitter` | `free2cad.weights` set in config |
| Egiazarian | `EgiazarianFitter` | `egiazarian.weights` set in config |

### Evaluation metrics (per sketch, per primitive type)

Implement a `benchmark.py` script at the project root:

```python
# benchmark.py
# Runs all three fitters on a shared test set and reports:
#   - Chamfer Distance: mean distance between fitted primitive and raw pixels
#   - Type accuracy: % of edges classified as the correct primitive type
#     (requires a manually labelled ground truth for a small test set)
#   - Mean confidence: as reported by each fitter
#   - Processing time per sketch (ms)
```

**Chamfer Distance** (primary geometric metric):
For each edge, compute the mean minimum distance between the raw pixel chain
and the fitted primitive (sampled at 1px intervals):

```
CD(edge) = (1/N) * sum_i min_j dist(pixel_i, sample_j)
```

Lower = better fit to the original sketch.

**Suggested test set:**
- The synthetic test sketch (4 lines + 1 circle) — ground truth known
- 5–10 real patent sketch images from the project partner data
  (ThyssenKrupp Presta, Elmos, Biedermann via infoapps GmbH)
- Annotate ground truth manually (expected primitive type per edge)
  using a small annotation script or LabelImg

### Expected benchmark output format

```
Approach     | Lines (CD) | Arcs (CD) | Circles (CD) | Conf  | Time/sketch
-------------|-----------|-----------|-------------|-------|------------
RANSAC       | 0.31 px   | 0.58 px   | 0.31 px     | 0.869 | 0.04s
Free2CAD     | ?         | ?         | ?           | ?     | ?
Egiazarian   | ?         | ?         | ?           | ?     | ?
```

---

## 8. Pipeline orchestration (Stage 3 in context)

Stage 3 sits between Stage 2 and Stage 4:

```python
# How run() is called from the batch runner (not yet built)
from stage3_primitive_fit import run, load_model

model = load_model(config)   # called once per batch

result = run(
    graph_path         = Path("output/graphs/test_001_graph.json"),
    output_dir         = Path("output"),
    sketch_id          = "test_001",
    config             = config,
    model              = model,             # Free2CADFitter | None
    cleaned_image_path = Path("output/cleaned/test_001_cleaned.png"),
)
# result.primitives_path → output/primitives/test_001_primitives.json
# result.fitter_used     → "ransac" | "free2cad" | "egiazarian" | "mixed"
# result.mean_confidence → float in [0,1]
# result.flagged         → True if mean_confidence < stage3.confidence_threshold
```

When the Egiazarian fitter is added, extend `run()` to accept a second model
parameter and apply the matching logic described in Task B.

---

## 9. config.yaml — full Stage 3 section

The current `config.yaml` has:

```yaml
stage3:
  confidence_threshold: 0.60  # mean confidence below this -> flag
```

Add the following when integrating the DL fitters:

```yaml
free2cad:
  repo_path: ""               # set to cloned repo path
  weights:   ""               # set to .pth file path
  device:    "cuda"

egiazarian:
  repo_path: ""               # set to cloned repo path
  weights:   ""               # set to .pth file path
  device:    "cuda"
  tile_size: 512              # patch size for tiled inference
  overlap:   64               # overlap between adjacent patches
```

---

## 10. Dependencies

Already installed (from Stage 1 & 2 setup):

```
torch torchvision   # PyTorch with CUDA
numpy scipy         # numerical core
opencv-python       # image I/O and rasterisation
scikit-image        # skeletonization (Stage 1)
networkx            # graph operations (Stage 2)
rdp                 # Ramer-Douglas-Peucker (Stage 2)
pyyaml              # config
```

New for Stage 3 RANSAC (already installed):

```
scikit-learn        # used indirectly for potential RANSAC extensions
```

New for Free2CAD / Egiazarian — install from their respective requirements.txt
files after cloning. Likely additional dependencies:

```
einops              # tensor manipulation (Free2CAD Transformer)
timm                # Vision Transformer backbone (possibly Egiazarian)
```

---

## 11. Key design decisions (do not change without good reason)

**Weights-optional architecture:** `load_model()` returns `None` if the
weights are missing. `run()` then uses RANSAC for every edge. This means
Stage 3 always produces output, even before any DL weights are available.

**Closed loop pixel selection:** `fit_edge_ransac()` uses raw `pixels`
(not `smooth_pts`) for `is_closed=True` edges. smooth_pts ordering is
based on coordinate sort, not arc angle, which causes the spline to scatter
rather than trace the loop. This was verified and fixed during testing.

**Priority order for RANSAC:** circle (closed) → line → arc → ellipse
→ polyline. Line is tried before arc because most patent sketch strokes
are straight. Arc is tried second because arc fitting reuses the circle
RANSAC infrastructure. Ellipse is last because it is the most expensive
and least common in mechanical/electrical patent drawings.

**Confidence formula:**
```
confidence = inlier_ratio * max(0, 1 - rms / max_rms)
```
Combines both inlier coverage and residual quality into a single [0,1] score.

**Free2CAD vs Egiazarian input difference:**
- Free2CAD: per-edge (stroke-level input)
- Egiazarian: full-image (must match output back to graph edges)

**Patent domain requirement:** output primitives feed Stage 4 (ezdxf + svgwrite)
which writes DXF LINE, ARC, CIRCLE, ELLIPSE, SPLINE entities. All primitive
parameters must be in pixel coordinates (will be scaled to mm in Stage 4
using the sketch DPI from Stage 0 metadata).

---

## 12. Immediate next steps for Claude Code

```
1. Verify the current stage3_primitive_fit.py runs correctly:
   python src/pipeline/stage3_primitive_fit.py \
       output/graphs/test_001_graph.json \
       --output output --config src/pipeline/config.yaml

2. Clone Free2CAD and inspect its output format (Task A, Steps 1–4)

3. Update _parse_free2cad_output() to match actual output

4. Clone Egiazarian repo and design EgiazarianFitter class (Task B)

5. Implement benchmark.py with Chamfer Distance metric (Task C)

6. Update CONTEXT.md with benchmark results before moving to Stage 4
```
