# Stage 3 Research — Free2CAD Experiment

This directory contains the **Free2CAD** Transformer-based primitive fitter that
was prototyped as an alternative to the production RANSAC implementation. The
original synthetic-data experiment was retired because RANSAC was more reliable
on real Stage-2 graphs. The encoder-only model is now being re-evaluated with
corrected SketchGraphs supervision; production remains RANSAC until paired
Drawing2CAD and filtered-PatentData integration gates pass.

This research code is preserved here for two reasons:

1. **Reproducibility** — the training pipeline + best checkpoint are intact, so a future team can re-evaluate the approach on improved data.
2. **Knowledge transfer** — see [docs/archive/free2cad_handoff.md](../../docs/archive/free2cad_handoff.md) for the full investigation log (what was tried, what failed, and why).

The production Stage 3 (`stage3_primitivesfitting/stage3_primitive_fit.py`) does
not load anything from this directory by default.

## SketchGraphs revival

`tools/sketchgraphs_dataset.py` runs the repository's actual Stage 2 topology
path and matches extracted edges back to exact SketchGraphs line/arc/circle
entities. Full training attempted all 9,179,789 train sketches and produced
27,023,994 matched edges. The selected epoch-2 model reaches 0.9993 supported
macro-F1 on all 926,940 untouched test edges; details are in
[`TRAIN_STAGE2_STAGE3.md`](../../TRAIN_STAGE2_STAGE3.md).

The restart-safe full-corpus extraction, shard-streaming warm-start training,
and full test evaluation are launched together with:

```bash
tools/run_stage3_full_training_queue.sh
```

The selected local checkpoint is `models/free2cad_sketchgraphs_full.pth`
(SHA-256 `bed27ed08ad794757bb93192c9347caf0af9d18f6f62a15a39537f0fc4d97d3f`).
It remains a research deployment candidate; the production fitter is still
RANSAC until Drawing2CAD and filtered-PatentData integration comparisons pass.

## Mixed Drawing2CAD revival

SketchGraphs contains no polyline labels. The completed mixed experiment therefore
balances **edge-label supply**, not drawing count:

```bash
python -m tools.d2c_stage3_dataset --split train --workers 56 \
  --output output/Drawing2CAD/stage3
python -m tools.build_free2cad_mixed_dataset \
  --drawing2cad output/Drawing2CAD/stage3 \
  --output output/Free2CADMixedSketchGraphsD2C --split train
```

Drawing2CAD cubics are guarded by circular residual before receiving BEZIER,
because SVG exporters also represent arcs/circles with `C` commands. The model
has five classes and can warm-start a four-class v3 checkpoint while preserving
all existing head rows. POLYLINE/BEZIER output is fitted deterministically from
the dense edge after classification. The selected checkpoint is
`models/free2cad_sketchgraphs_d2c_mixed.pth`: validation macro-F1 0.9683,
Drawing2CAD test macro-F1 0.8603, and full SketchGraphs test macro-F1 0.9936.
It remains a research candidate pending paired end-to-end pipeline evaluation.
Full methodology and results live in
[`TRAIN_STAGE2_STAGE3.md`](../../TRAIN_STAGE2_STAGE3.md#mixed-sketchgraphs--drawing2cad-stage-3-2026-07-19).

## Files

| File                                | Purpose                                                                   |
|-------------------------------------|---------------------------------------------------------------------------|
| `generate_sketches_v3.py`           | Synthetic data generator (rebalanced families incl. `short_stubs`, `noisy_circle`). |
| `train_free2cad_v3.py`              | Trainer (encoder-only architecture, 5 classes, class-weighted CE).        |
| `stage3_primitive_fit_free2cad.py`  | Inference script using the Free2CAD model (drop-in replacement for the production RANSAC stage). |

Supporting repository tools are
[`tools/d2c_stage3_dataset.py`](../../tools/d2c_stage3_dataset.py),
[`tools/build_free2cad_mixed_dataset.py`](../../tools/build_free2cad_mixed_dataset.py),
and [`tools/evaluate_free2cad_v3.py`](../../tools/evaluate_free2cad_v3.py).

## Required external repositories

The trainer references the upstream Free2CAD reference implementation for some helpers. To set this up:

```bash
mkdir -p stage3_primitivesfitting/research/repos
cd stage3_primitivesfitting/research/repos
git clone https://github.com/Enigma-li/Free2CAD.git
# DeepCAD (used for ABC dataset preprocessing — only needed if generating training data
# from the ABC step files):
git clone https://github.com/ChrisWu1997/DeepCAD.git
```

## Training data

Two corpora are referenced by the data pipeline:

- **`free2cad_training/`** — synthetic stroke-graphs produced by `generate_sketches_v3.py`. Generate locally, do not commit (large).
- **`abc_dataset/step_files/`** and **`deepcad/`** — upstream parametric-CAD datasets used by some experiment branches. Download links:
  - ABC dataset: https://deep-geometry.github.io/abc-dataset/
  - DeepCAD: https://github.com/ChrisWu1997/DeepCAD (see their `data/` instructions).

All three live under `stage3_primitivesfitting/research/data/` and are gitignored.

The active SketchGraphs raw corpus lives under `data/SketchGraphs/raw/`, and
derived Stage 3 shards live under `output/SketchGraphsStage3Full/`. Both are
local, generated artifacts and must not be committed.

## Re-running the experiment

Quick path on a fresh clone:

```bash
# 1. Generate synthetic data (~30k samples by default; takes ~30 minutes)
python stage3_primitivesfitting/research/generate_sketches_v3.py

# 2. Train (set device: "cuda" in config.yaml first if you have a GPU)
python stage3_primitivesfitting/research/train_free2cad_v3.py

# 3. Run inference on a Stage-2 graph
python stage3_primitivesfitting/research/stage3_primitive_fit_free2cad.py \
    output/graphs/Picture1_skeleton_graph.json \
    --output output --fitter free2cad
```

Best epoch checkpoint lands at `models/free2cad_v3_best.pth`.

## Why the synthetic approach was retired

Short version (long version in `docs/archive/free2cad_handoff.md`):

- **Class collapse** in v1/v2 — model heavily biased toward whichever primitive type was over-represented in training data.
- **Distribution gap** — synthetic training graphs had ~3× more vertices per stroke than real Stage-2 output (which has many 2-point edges). Inference looked nothing like training.
- **Geometric inexactness** — even when the class was right, the predicted geometry (centers, radii, endpoints) was less accurate than RANSAC's least-squares fit on the same edge points.
- **Performance** — Free2CAD inference was ~6× slower than RANSAC on the same input, with no quality compensation.

On the canonical test sketch (`Picture1_skeleton_graph.json`), RANSAC achieved 86.9% type-agreement with hand-labelled ground truth and recovered 4/4 circles; the best Free2CAD checkpoint recovered 1/4 circles and frequently mis-classified arcs as polylines.

This does **not** mean Free2CAD-style approaches are wrong for this problem — only that this particular synthetic-only training pipeline did not close the gap to a tuned classical fitter. With a real-world labelled stroke-graph corpus it would be worth revisiting.
