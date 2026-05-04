# Stage 1 — Preprocessing

> AP3 Vectorization Pipeline · IP DrawingDrafter · HAW Landshut
> Module: `stage1_preprocessing/stage1_preprocess.py`

---

## 1. Position in the Pipeline

Stage 1 is the **first stage** of the AP3 vectorization pipeline. It takes a raw raster image (a scanned or photographed sketch) and produces a clean, normalised binary skeleton ready for stroke graph extraction in Stage 2.

```
Raw raster (PNG)  →  Stage 1 (Preprocessing)  →  cleaned + skeleton
                                                  ↓
                                                Stage 2 (Stroke Extraction)
```

| Stage | In         | Out                              |
|-------|------------|----------------------------------|
| **1** | Raw raster | Cleaned image + 1-px skeleton    |
| 2     | Skeleton   | Stroke graph (JSON)              |
| 3     | Graph      | Geometric primitives (JSON)      |
| 4     | Primitives | Vector files (SVG, DXF)          |

---

## 2. Input Contract

A single grayscale or BGR PNG. Any size. Stage 1 is robust to noisy backgrounds, uneven lighting, paper texture, and pencil/pen variability.

---

## 3. Output

Two PNG files written to `<output>/cleaned/`:

| File                       | Format | Purpose                                                           |
|----------------------------|--------|-------------------------------------------------------------------|
| `<id>_cleaned.png`         | PNG    | Denoised, binarised drawing — strokes black, background white     |
| `<id>_skeleton.png`        | PNG    | 1-pixel-wide medial-axis skeleton (input for Stage 2)             |

The `Stage1Result` dataclass also carries a `quality` score in `[0, 1]` and a `flagged` boolean (true when `quality < stage1.quality_threshold` from config) for downstream QC pipelines.

---

## 4. Two Cleaning Modes

Stage 1 chooses its cleaning method automatically:

| Mode                   | When                                                       | What it does                                                      |
|------------------------|------------------------------------------------------------|-------------------------------------------------------------------|
| **SketchCleanNet (DL)** | `sketchcleannet.weights` is set in config and the file exists | Tile-based U-Net inference with overlap blending — best quality   |
| **Classical fallback**  | Weights not available                                      | Otsu + adaptive threshold blend, morphological opening, CC filter |

The DL mode is roughly 5–10× slower than classical but handles photographed/shaded sketches dramatically better. Both modes produce a binary cleaned image and a 1-pixel skeleton.

**Why ship classical too:** so the pipeline runs out of the box without downloading a 124 MB weight file, and so partner sites without GPU access still get useful output.

---

## 5. Configuration

Set in [config.yaml](../config.yaml) under the `sketchcleannet:` and `stage1:` blocks:

```yaml
sketchcleannet:
  weights: "models/sketchcleannet.pth"   # leave "" to force classical mode
  device:  "cpu"                          # or "cuda"

stage1:
  quality_threshold: 0.70                 # < this ⇒ flagged for review
  classical:
    blur_kernel:    3
    adaptive_block: 35
    adaptive_C:     10
    blend_alpha:    0.6
    morph_kernel:   2
    min_cc_size:    30
```

---

## 6. CLI Reference

Run from the project root:

```bash
# Single sketch
python stage1_preprocessing/stage1_preprocess.py data/samples/Picture1.png

# Override output and ID
python stage1_preprocessing/stage1_preprocess.py data/samples/Picture1.png \
    --output ./output --id picture1
```

Outputs land in `<output>/cleaned/<id>_cleaned.png` and `<output>/cleaned/<id>_skeleton.png`.

---

## 7. Research Code

Training code for SketchCleanNet (`train_sketchcleannet.py`) lives under [stage1_preprocessing/research/](../stage1_preprocessing/research/) and is not needed to run the pipeline. See [stage1_preprocessing/research/README.md](../stage1_preprocessing/research/README.md) for how to obtain the training data and re-train the model.

---

## 8. Dependencies

| Package    | Purpose                              |
|------------|--------------------------------------|
| `numpy`    | array math                           |
| `opencv-python` | image I/O, classical filters    |
| `scikit-image`  | medial-axis skeletonization     |
| `torch`    | optional — only for SketchCleanNet   |
| `pyyaml`   | config loading                       |
