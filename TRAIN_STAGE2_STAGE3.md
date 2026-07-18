# Training Stages 2 and 3 on Real CAD Vector Ground Truth

> **Repository audit (2026-07-17):** The original proposal below assumed a
> different, flat checkout at `~/ip-drawing-drafter` and an external
> `~/datasets` tree. Those paths and several data/model contracts do not match
> this repository. Use the verified workflow in this section. The longer
> proposal is retained after it as design history, not as executable commands.

## Verified SketchGraphs workflow for this repository

Run every command from:

```bash
cd /home/safe/Desktop/yassine/Vectorization
```

The real training entry points are:

- Stage 2: `stage2_strokeextraction/research/train_puhachov.py`
- Stage 3: `stage3_primitivesfitting/research/train_free2cad_v3.py`
- SketchGraphs conversion: `tools/sketchgraphs_dataset.py`

The official filtered SketchGraphs release is three split files, not
`sg_filtered_unique.npy`:

```text
data/SketchGraphs/raw/sg_t16_train.npy
data/SketchGraphs/raw/sg_t16_validation.npy
data/SketchGraphs/raw/sg_t16_test.npy
```

Download and verify them with:

```bash
git clone --depth 1 https://github.com/PrincetonLIPS/SketchGraphs.git \
  stage2_strokeextraction/research/repos/SketchGraphs
.venv/bin/pip install lz4
.venv/bin/pip install -e \
  stage2_strokeextraction/research/repos/SketchGraphs --no-deps
.venv/bin/python -m tools.sketchgraphs_dataset download
```

Create the first leakage-free pilot. Random index sampling avoids the strong
ordering bias visible in first-N examples:

```bash
.venv/bin/python -m tools.sketchgraphs_dataset prepare \
  --splits train --limit 100000 --workers 16
.venv/bin/python -m tools.sketchgraphs_dataset prepare \
  --splits validation --limit 10000 --workers 16
.venv/bin/python -m tools.sketchgraphs_dataset prepare \
  --splits test --limit 10000 --workers 16
```

This creates:

```text
output/SketchGraphsTraining/stage2/{train,validation,test}/*.npz
output/SketchGraphsTraining/stage3/{train,val,test}/shard_*.npz
output/SketchGraphsTraining/report_*.json
output/SketchGraphsTraining/audit_stage2_*.png
```

The Stage 2 contract is three heatmaps (`endpoint`, `junction`, `corner`), not
primitive-membership labels. Stage 3 consumes ordered edge points and exact
normalized primitive parameters. Its SketchGraphs examples are generated only
after the repository's real Stage 2 topology extraction and simplification,
then matched back to source line/arc/circle entities. This prevents whole CAD
entities from being used where inference actually sees graph edges.

Train the pilot models with:

```bash
.venv/bin/python -m stage2_strokeextraction.research.train_puhachov \
  --labels output/SketchGraphsTraining/stage2 \
  --init-weights models/puhachov_d2c.pth \
  --out models/puhachov_sketchgraphs_pilot.pth \
  --steps 10000 --batch 24 --workers 8 --device cuda:0 --amp \
  --val-subset 1000 --val-every 1000

# Preserve the Drawing2CAD domain while learning SketchGraphs. Checkpoints are
# selected by the mean validation macro-F1 across both domains.
.venv/bin/python -m stage2_strokeextraction.research.train_puhachov \
  --labels output/Drawing2CAD/kp_labels \
  --secondary-labels output/SketchGraphsTraining/stage2 \
  --mix 0.7 --mix-size 100000 \
  --init-weights models/puhachov_d2c.pth \
  --out models/puhachov_sketchgraphs_rehearsal70.pth \
  --steps 10000 --batch 24 --workers 8 --device cuda:0 --amp \
  --val-subset 500 --secondary-val-subset 500 --val-every 1000

.venv/bin/python stage3_primitivesfitting/research/train_free2cad_v3.py \
  --data_dir output/SketchGraphsTrainingV2/stage3 \
  --output_dir output/SketchGraphsTrainingV2/checkpoints/stage3_threepoint_sqrt_clean_v2 \
  --epochs 40 --batch_size 512 --max_pts 64 \
  --arc_encoding three_point --class_weight_power 0.5 --device cuda:1
```

### Full Stage 2 streaming workflow

Step 1 of the full-corpus roadmap is implemented. Do **not** materialize the
9,179,789 training sketches as 512x512 NPZ rasters. The trainer memory-maps the
official flat-array file in each DataLoader worker, renders only the current
batch, and uses a constant-memory distributed permutation. One mixed epoch
contains every SketchGraphs source record exactly once plus a balanced sample
of cached Drawing2CAD labels.

Run Phase A on both RTX 4090 GPUs with 70% SketchGraphs / 30% Drawing2CAD:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  -m stage2_strokeextraction.research.train_puhachov \
  --labels output/Drawing2CAD/kp_labels \
  --sketchgraphs-raw data/SketchGraphs/raw/sg_t16_train.npy \
  --sketchgraphs-val-labels output/SketchGraphsTraining/stage2 \
  --mix 0.30 --steps 0 --epochs 1 \
  --init-weights models/puhachov_sketchgraphs_rehearsal70.pth \
  --out models/puhachov_sketchgraphs_full_phaseA.pth \
  --state-out models/puhachov_sketchgraphs_full_phaseA_last.pth \
  --coverage-file output/SketchGraphsFull/stage2_train.coverage.i8 \
  --require-full-coverage \
  --batch 12 --workers 6 --prefetch-factor 2 --amp \
  --val-subset 500 --secondary-val-subset 500 \
  --val-every 10000 --save-every 5000 --log-every 100
```

`--amp` defaults to BF16 on the RTX 4090s. FP16 remains available through
`--amp-dtype fp16`, with overflow retries, but BF16 is the verified stable path
for this sparse focal loss.

The atomic `*_last.pth` file contains model, optimizer, scaler, epoch, and the
per-rank source offset. Resume with the identical world size and per-rank batch:

```bash
# Repeat all Phase-A arguments and add:
--resume models/puhachov_sketchgraphs_full_phaseA_last.pth
```

The compact int8 coverage map records one status per official source index
(`-1` unattempted, `0` accepted, positive values are rejection categories).
Inspect it or export rejected indices with:

```bash
.venv/bin/python -m tools.sketchgraphs_coverage \
  output/SketchGraphsFull/stage2_train.coverage.i8 \
  --rejections-csv output/SketchGraphsFull/stage2_train_rejections.csv \
  --require-complete
```

The keypoint-only streaming renderer deliberately skips Stage 2 topology and
Stage 3 edge fitting. A 200-record probe took 1.45 seconds in one process,
compared with 11.36 seconds through the offline graph-building path, while
returning the same 193 accepted and 7 unsupported source records.

#### Phase A completion (2026-07-18)

The full two-GPU run completed one exact mixed epoch in 546,416 optimizer
steps (1,027.2 minutes). All 9,179,789 official SketchGraphs training records
were attempted: 8,809,417 were accepted, 369,486 had no supported geometry,
and 886 had degenerate geometry. There were zero unattempted, decode, or
unknown-error records.

Checkpoint selection used the mean Drawing2CAD/SketchGraphs validation
macro-F1. The selected model is
`models/puhachov_sketchgraphs_full_phaseA.pth` from step 330,000:

| checkpoint | Drawing2CAD val | SketchGraphs val | dual-domain score |
|---|---:|---:|---:|
| 70/30 pilot rehearsal | 0.8454 | 0.8948 | 0.8701 |
| **full Phase A, step 330,000** | **0.8607** | **0.9435** | **0.9021** |

`models/puhachov_sketchgraphs_full_phaseA_last.pth` is the completed epoch
state, not the deployment candidate.

#### Full Stage 2 evaluation (2026-07-18)

The Phase A checkpoint and the previous production checkpoint were evaluated
with the same renderer, peak extraction, NMS, and greedy matching contract on
the complete untouched SketchGraphs test split. Every one of the 313,271 source
records was attempted. Both runs accepted the same 300,321 records and rejected
the same 12,950 records (12,922 unsupported and 28 degenerate); there were no
decode or unknown errors.

| checkpoint | endpoint F1 | junction F1 | corner F1 | macro-F1 |
|---|---:|---:|---:|---:|
| production `puhachov_d2c.pth` | 0.0039 | 0.4216 | 0.8695 | 0.4317 |
| **full Phase A, step 330,000** | **0.9965** | **0.8244** | **0.9838** | **0.9349** |

Detailed Phase A class statistics:

| class | precision | recall | F1 | support | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| endpoint | 0.9983 | 0.9948 | 0.9965 | 127,701 | 127,036 | 217 | 665 |
| junction | 0.9942 | 0.7041 | 0.8244 | 293,609 | 206,743 | 1,216 | 86,866 |
| corner | 0.9780 | 0.9896 | 0.9838 | 756,906 | 749,061 | 16,862 | 7,845 |

The remaining learned-keypoint weakness is junction recall, not precision. The
full reports are:

```text
output/SketchGraphsFull/evaluation/phaseA_test.json
output/SketchGraphsFull/evaluation/production_test.json
```

Generalization was measured without additional tuning on all cached
Drawing2CAD validation views and on the complete ArchCAD validation set:

| checkpoint | Drawing2CAD macro-F1 (31,516) | ArchCAD macro-F1 (1,159) |
|---|---:|---:|
| production `puhachov_d2c.pth` | 0.8350 | 0.3676 |
| **full Phase A, step 330,000** | **0.8649** | **0.5971** |

| dataset/model | endpoint F1 | junction F1 | corner F1 |
|---|---:|---:|---:|
| Drawing2CAD, production | 0.6872 | **0.8663** | 0.9516 |
| Drawing2CAD, Phase A | **0.7693** | 0.8649 | **0.9605** |
| ArchCAD, production | 0.0363 | 0.6334 | 0.4329 |
| ArchCAD, Phase A | **0.5815** | **0.7413** | **0.4685** |

The final Stage 2 gate was a paired end-to-end run on 7,881 untouched
Drawing2CAD test models in all four views. Both configurations completed all
31,524 matched views with zero pipeline errors. Only the Stage 2 checkpoint
changed; Stage 1, learned/classical fusion, Stage 3, and export were held fixed.

| end-to-end mean metric | production | Phase A | relative change |
|---|---:|---:|---:|
| symmetric Chamfer (lower) | 0.9115 | **0.9031** | **-0.92%** |
| symmetric Chamfer p95 (lower) | 1.5378 | **1.5289** | **-0.58%** |
| pixel IoU (higher) | 0.6961 | **0.6988** | **+0.39%** |
| skeleton IoU (higher) | 0.5635 | **0.5668** | **+0.59%** |
| pixel precision (higher) | 0.7725 | **0.7747** | **+0.29%** |
| pixel recall (higher) | 0.8752 | **0.8768** | **+0.18%** |
| output primitives (lower) | 7.5793 | **7.5377** | **-0.55%** |
| primitive inflation (lower) | 3.2244 | **3.2139** | **-0.33%** |
| median edge length (higher) | 243.9067 | **244.0215** | **+0.05%** |
| micro-edge ratio (lower) | **0.02691** | 0.02709 | +0.65% |
| short-edge ratio (lower) | **0.07439** | 0.07501 | +0.84% |
| Stage 2 time, seconds (lower) | 0.18759 | **0.18737** | **-0.12%** |
| total time, seconds (lower) | **0.45879** | 0.45924 | +0.10% |

Chamfer metrics have 31,497 finite pairs; all other metrics have 31,524. The
large tie rates (56-98%, depending on metric) explain why the substantial
keypoint-domain gain becomes a modest end-to-end gain: fusion and downstream
geometry fitting produce identical results for most views. Phase A nevertheless
improves every primary geometry/raster metric and slightly reduces primitive
inflation at effectively unchanged runtime. Micro/short-edge ratios regress by
small absolute amounts (0.00017 and 0.00062).

The comparison report and reproducibility manifests are:

```text
output/Drawing2CAD/full_test_phaseA_vs_production.json
output/Drawing2CAD/full_test_phaseA/evaluation_manifest.json
output/Drawing2CAD/full_test_production/evaluation_manifest.json
```

**Stage 2 decision:** `models/puhachov_sketchgraphs_full_phaseA.pth` supersedes
the pilot rehearsal as the deployment candidate. Complete subgroup/outlier
analysis and a filtered-PatentData visual regression before changing the
default production configuration. Junction recall and the small short-edge
regression are the two explicit follow-up checks.

The `V2` Stage 3 corpus is a corrected rebuild from the same sampled source
indices. It was generated without rewriting Stage 2 labels:

```bash
.venv/bin/python -m tools.sketchgraphs_dataset prepare \
  --output output/SketchGraphsTrainingV2 --splits train \
  --limit 100000 --workers 16 --stage3-only
.venv/bin/python -m tools.sketchgraphs_dataset prepare \
  --output output/SketchGraphsTrainingV2 --splits validation \
  --limit 10000 --workers 16 --stage3-only
.venv/bin/python -m tools.sketchgraphs_dataset prepare \
  --output output/SketchGraphsTrainingV2 --splits test \
  --limit 10000 --workers 16 --stage3-only
```

### Pilot audit (2026-07-17, superseded by the full evaluation above)

The official split files and exact sequence counts are:

| split | file size | sequences |
|---|---:|---:|
| train | 6,151,102,626 bytes | 9,179,789 |
| validation | 211,121,676 bytes | 315,228 |
| test | 209,307,405 bytes | 313,271 |

The deterministic pilot accepted 95,853 train, 9,597 validation, and 9,565
test sketches. The corrected Stage 3 train set contains 293,870 extracted
edges before quality filtering. Its loader removes 96 incomplete-circle labels
and relabels 2,722 raster-straight source arcs as lines. These corrections are
part of the inference contract, not test-set tuning: open pieces of a source
circle are arcs, incomplete closed loops are not identifiable full circles,
and edges with no measurable sagitta are lines at raster resolution.

Untouched test metrics:

| model | metric | old checkpoint | SketchGraphs checkpoint |
|---|---|---:|---:|
| Stage 2 | supported macro-F1, 9,565 sketches | 0.4331 | 0.9194 |
| Stage 3 | supported macro-F1, 29,420 edges | 0.3733 | 0.9987 |
| Stage 3 | accuracy | 0.7693 | 0.9996 |

Final Stage 3 per-class test results:

| class | F1 | parameter L1 | stroke residual |
|---|---:|---:|---:|
| line | 0.9998 | 0.0015 | 0.0014 |
| arc | 0.9965 | 0.0056 | 0.0232 |
| circle | 0.9999 | 0.0029 | 0.0058 |

The pure SketchGraphs Stage 2 pilot was **not** a production replacement. On a
fixed 1,000-view Drawing2CAD validation sample it drops from 0.8335 to 0.5252
macro-F1, mainly because endpoint recall collapses. A lower confidence
threshold does not recover the signal (0.5558 macro-F1 at 0.05). The
dual-domain rehearsal model above is therefore the deployment candidate.

Stage 2 cross-domain results:

| checkpoint | SketchGraphs test (9,565) | Drawing2CAD val sample (1,000) | ArchCAD val (1,159) |
|---|---:|---:|---:|
| original `puhachov_d2c.pth` | 0.4331 | 0.8335 | 0.3688 |
| pure SketchGraphs | 0.9194 | 0.5252 | not selected |
| 50/50 rehearsal, best step 3,000 | 0.8802 | 0.8199 | not selected |
| **70/30 rehearsal, best step 8,000** | **0.8813** | **0.8370** | **0.5416** |

The pilot recommendation was `models/puhachov_sketchgraphs_rehearsal70.pth`;
the full evaluation above supersedes it with
`models/puhachov_sketchgraphs_full_phaseA.pth`. The current Stage 3 checkpoint,
`models/free2cad_sketchgraphs.pth`, is still a 100,000-source pilot. Both model
families are ignored by Git; their evaluation JSON reports are under
`output/SketchGraphsTraining*` and `output/SketchGraphsFull`.

The Stage 3 checkpoint loads and decodes through the research
`Free2CADFitter`, but the main production Stage 3 entry point still defaults to
the guarded deterministic/RANSAC fitter. Do not switch production configuration
until PatentData integration renders have been compared visually.

SketchGraphs code is MIT licensed. The sketch data is different: the official
release states that the original sketch creators retain copyright and points to
the Onshape terms. Do not publish or commit the downloaded/derived corpus.

CAD-VGDrawing is intentionally out of scope until both SketchGraphs-only models
have been evaluated on their untouched test split and on the existing
Drawing2CAD/filtered PatentData pipeline benchmarks.

**Decision after evaluation:** defer CAD-VGDrawing. Corrected SketchGraphs
already solves the Stage 3 line/arc/circle task on the untouched pilot test,
and 70/30 rehearsal improves SketchGraphs and ArchCAD while retaining
Drawing2CAD. The next evidence gate is visual integration on filtered
PatentData, not another large dataset download. Reconsider CAD-VGDrawing only
if that integration exposes mechanical-view failure modes not represented by
the existing Drawing2CAD labels.

---

## Historical proposal (paths and commands below are not authoritative)

**Project:** IP DrawingDrafter / AP3 Vectorization Pipeline
**Author:** Yassine Rezgui — HAW Landshut
**Purpose:** End-to-end recipe for replacing the current synthetic-only training with real CAD vector ground truth from SketchGraphs and CAD-VGDrawing, then training the Puhachov CNN (Stage 2) and Free2CAD v3 encoder-only model (Stage 3).

**Assumed target environment:** Linux/Ubuntu, NVIDIA GPU with CUDA, Miniconda already installed, existing `ip-drawing-drafter/` repo layout intact.

**Rationale for this work.** The current `generate_sketches_v3.py` produces synthetic per-edge data whose distribution does not match real hand-drawn CAD input (uniform angular spacing, axis-aligned bias, artificial noise). SketchGraphs supplies 15M real Onshape 2D sketches with parametric primitive ground truth (line/arc/circle/ellipse) exactly matching Stage 3's fitter. CAD-VGDrawing supplies 161K mechanical CAD models with multi-view engineering-drawing SVGs, which is structurally closer to utility-patent Maschinenbau figures. This directly addresses the "training distribution must match inference distribution" failure mode already identified in the project memory.

---

## Table of contents

1. [Preflight and directory layout](#1-preflight-and-directory-layout)
2. [Environment setup](#2-environment-setup)
3. [Dataset 1 — SketchGraphs](#3-dataset-1--sketchgraphs)
4. [Dataset 2 — CAD-VGDrawing](#4-dataset-2--cad-vgdrawing)
5. [Stage 2 training data preparation](#5-stage-2-training-data-preparation)
6. [Stage 2 — Puhachov CNN training](#6-stage-2--puhachov-cnn-training)
7. [Stage 3 training data preparation](#7-stage-3-training-data-preparation)
8. [Stage 3 — Free2CAD v3 training](#8-stage-3--free2cad-v3-training)
9. [Integration into the AP3 pipeline](#9-integration-into-the-ap3-pipeline)
10. [Validation](#10-validation)
11. [Notes and known caveats](#11-notes-and-known-caveats)

---

## 1. Preflight and directory layout

Before doing anything else, verify the working state of the repo and create the dataset root.

```bash
cd ~/ip-drawing-drafter
git status                    # expect: clean working tree on the current branch
git checkout -b feat/real-cad-training

# Expected repo layout that this recipe relies on:
# ip-drawing-drafter/
# ├── stage1_preprocess.py
# ├── stage2_stroke_extract.py
# ├── stage3_primitive_fit.py
# ├── stage3_primitive_fit_PATCHED.py
# ├── generate_sketches_v3.py
# ├── train_free2cad_v3.py
# ├── config.yaml
# └── ...
```

Create the dataset root outside the repo (large files should not be in Git):

```bash
mkdir -p ~/datasets/{sketchgraphs,cad_vgdrawing,derived}
mkdir -p ~/datasets/derived/{stage2_pairs,stage3_pairs}
mkdir -p ~/ip-drawing-drafter/checkpoints/{stage2,stage3}
mkdir -p ~/ip-drawing-drafter/mlruns
```

Free-disk-space check before proceeding — you need at least:

- ~40 GB for SketchGraphs filtered subset
- ~30 GB for CAD-VGDrawing SVGs
- ~50 GB for derived training pairs
- ~20 GB for checkpoints and MLflow runs

```bash
df -h ~/datasets
```

If any of these are tight, stop and reroute paths before continuing.

---

## 2. Environment setup

Extend the existing IP DrawingDrafter conda env or create a dedicated training env. A separate env keeps the training deps (torch-scatter, geometric libs) from polluting the inference env.

```bash
conda create -n ipdd-train python=3.10 -y
conda activate ipdd-train

# Core
pip install --upgrade pip
pip install torch==2.1.* torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install numpy scipy scikit-image opencv-python pillow tqdm pyyaml

# Training utilities
pip install mlflow==2.14.* tensorboard
pip install ezdxf svgwrite svgpathtools shapely

# For SketchGraphs data loading
pip install networkx

# Verify GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is not available, stop and diagnose — training on CPU is not feasible at this scale.

---

## 3. Dataset 1 — SketchGraphs

### 3.1 Clone the reference repo

```bash
cd ~/datasets/sketchgraphs
git clone https://github.com/PrincetonLIPS/SketchGraphs.git repo
cd repo
pip install -e .
```

### 3.2 Download the data

SketchGraphs is distributed in several forms. Get the **filtered sequence dataset** (`sg_filtered_unique.npy`, ~7 GB), which is deduplicated and already screened for sketches with at least one primitive and one constraint. The full dataset (~50 GB) is only needed if you want to build a custom filter.

```bash
# The maintainers host the artifacts on their group page.
# The current canonical URL should be verified from the repo README —
# if it has moved, check github.com/PrincetonLIPS/SketchGraphs README.md.

cd ~/datasets/sketchgraphs
# Example command (verify URL before running):
# wget -c https://sketchgraphs.cs.princeton.edu/data/sg_filtered_unique.npy

# If the URL fails, fall back to the raw dataset (larger):
# wget -c https://sketchgraphs.cs.princeton.edu/data/sg_all.npy
```

**Verification:**

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path.home() / "datasets/sketchgraphs/sg_filtered_unique.npy"
data = np.load(p, allow_pickle=True)
print(f"Loaded {len(data)} sketches")
print(f"First sketch type: {type(data[0])}")
PY
```

Expect on the order of 2–15 million sketches depending on which artifact was fetched.

### 3.3 Understand the data structure

Each sketch is a `Sketch` object with:

- `entities`: dict of primitives, each carrying its parametric definition
  - `Line`: start point (x1, y1), end point (x2, y2)
  - `Arc`: center (cx, cy), radius r, start angle, end angle, clockwise flag
  - `Circle`: center (cx, cy), radius r
  - `Ellipse`: center, major/minor radii, angle
  - `Point`: (x, y)
  - `Spline`: control points (skip these — <3% of the corpus)
- `constraints`: list of relations (coincident, tangent, parallel, perpendicular, distance, etc.) — not needed for our supervision but useful for downstream constraint inference work

The `sketchgraphs.data` module already provides the parsing.

### 3.4 Filter to Stage 3-compatible sketches

Discard sketches containing splines (Stage 3 does not fit splines). Keep sketches with 2–16 primitives — matches the working set used by recent SketchGraphs derivative work and keeps memory reasonable.

Create `~/datasets/sketchgraphs/filter_for_stage3.py`:

```python
"""
Filter SketchGraphs sketches to those Stage 3 can supervise on.
Outputs a smaller .npy of Sketch objects ready for rendering.
"""
from pathlib import Path
import numpy as np
from sketchgraphs.data import sketch as sk

SRC = Path.home() / "datasets/sketchgraphs/sg_filtered_unique.npy"
DST = Path.home() / "datasets/sketchgraphs/sg_stage3_ready.npy"

MIN_PRIMS, MAX_PRIMS = 2, 16
ALLOWED = {"Line", "Arc", "Circle", "Ellipse"}   # no splines, no bare points

def keep(sketch):
    types = [type(e).__name__ for e in sketch.entities.values()]
    non_point = [t for t in types if t != "Point"]
    if not (MIN_PRIMS <= len(non_point) <= MAX_PRIMS):
        return False
    return all(t in ALLOWED or t == "Point" for t in types)

def main():
    data = np.load(SRC, allow_pickle=True)
    kept = [s for s in data if keep(s)]
    print(f"Kept {len(kept):,}/{len(data):,} sketches")
    np.save(DST, np.array(kept, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()
```

Run it:

```bash
python ~/datasets/sketchgraphs/filter_for_stage3.py
```

Expect roughly 2–4 million surviving sketches.

---

## 4. Dataset 2 — CAD-VGDrawing

### 4.1 Clone the Drawing2CAD repo

CAD-VGDrawing is released as part of the Drawing2CAD project (arXiv:2508.18733, August 2025).

```bash
cd ~/datasets/cad_vgdrawing
git clone https://github.com/lllssc/Drawing2CAD.git repo   # verify current URL
cd repo
```

Check the repo `README.md` for the current dataset download link — as of writing, the maintainers distribute the SVG files as a compressed archive on a shared drive. If the direct download is not yet public, the fallback is to **regenerate** the SVGs from DeepCAD using their provided FreeCAD script:

```bash
# Fallback path:
# 1. Download DeepCAD dataset (~5 GB parametric CAD)
#    https://github.com/ChrisWu1997/DeepCAD
# 2. Install FreeCAD:
#      sudo apt install freecad
# 3. Run the Drawing2CAD projection script (in the repo):
#      python scripts/generate_svg_from_deepcad.py \
#             --deepcad_root ~/datasets/deepcad \
#             --output ~/datasets/cad_vgdrawing/svg
#
# This produces 4 views (front / top / side / iso) per CAD model as SVG.
```

Expected final layout:

```
~/datasets/cad_vgdrawing/
├── svg/
│   ├── 00000000/
│   │   ├── front.svg
│   │   ├── top.svg
│   │   ├── side.svg
│   │   └── iso.svg
│   └── ...
└── metadata.json
```

### 4.2 Parse the SVGs to primitive lists

Create `~/datasets/cad_vgdrawing/parse_svg.py` using `svgpathtools`:

```python
"""
Convert CAD-VGDrawing SVGs to Stage 3-compatible primitive lists.
Each SVG path element becomes one or more (type, params) tuples.
"""
from pathlib import Path
import json
from svgpathtools import svg2paths, Line, Arc, CubicBezier, QuadraticBezier

ROOT = Path.home() / "datasets/cad_vgdrawing/svg"
OUT  = Path.home() / "datasets/cad_vgdrawing/primitives.jsonl"

def parse_one(svg_path):
    paths, attrs = svg2paths(str(svg_path))
    prims = []
    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                prims.append({
                    "type": "line",
                    "p1": [seg.start.real, seg.start.imag],
                    "p2": [seg.end.real, seg.end.imag],
                })
            elif isinstance(seg, Arc):
                prims.append({
                    "type": "arc",
                    "center": [seg.center.real, seg.center.imag],
                    "radius": abs(seg.radius),
                    "start_angle": seg.theta,
                    "sweep_angle": seg.delta,
                })
            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                # Approximate curved segments as polylines for Stage 3
                pts = [seg.point(t/16) for t in range(17)]
                prims.append({
                    "type": "polyline",
                    "points": [[p.real, p.imag] for p in pts],
                })
    return prims

def main():
    with open(OUT, "w") as f:
        for model_dir in sorted(ROOT.iterdir()):
            if not model_dir.is_dir():
                continue
            for view in ["front", "top", "side", "iso"]:
                svg = model_dir / f"{view}.svg"
                if not svg.exists():
                    continue
                prims = parse_one(svg)
                if not prims:
                    continue
                f.write(json.dumps({
                    "model_id": model_dir.name,
                    "view": view,
                    "primitives": prims,
                }) + "\n")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python ~/datasets/cad_vgdrawing/parse_svg.py
wc -l ~/datasets/cad_vgdrawing/primitives.jsonl   # expect ~600K lines (150K models × 4 views)
```

---

## 5. Stage 2 training data preparation

Stage 2 needs (raster input, ground-truth stroke graph) pairs. The stroke graph is a set of (skeleton pixels → primitive-membership label) mappings. Two derivation paths:

### 5.1 Renderer that emits paired (raster, per-primitive skeleton) data

Create `~/ip-drawing-drafter/data/render_stage2_pairs.py`:

```python
"""
Render SketchGraphs primitives to:
  - A raster image (input to Stage 2)
  - A per-primitive skeleton mask (target: which pixel belongs to which primitive)

Uses the existing Stage 1 degradation pipeline to make inputs realistic.
"""
from pathlib import Path
import numpy as np
import cv2
import json
from sketchgraphs.data import sketch as sk
from sketchgraphs.pipeline.render import render_sketch  # from SketchGraphs

# Import Stage 1 degradation
import sys
sys.path.insert(0, str(Path.home() / "ip-drawing-drafter"))
from stage1_preprocess import _classical_clean   # reuse existing pipeline

SG_PATH   = Path.home() / "datasets/sketchgraphs/sg_stage3_ready.npy"
OUT_DIR   = Path.home() / "datasets/derived/stage2_pairs"
IMG_SIZE  = 512
LINE_WIDTH = 2

def primitive_to_pixels(prim, size=IMG_SIZE, width=LINE_WIDTH):
    """Return a (size, size) mask where this primitive is drawn."""
    mask = np.zeros((size, size), dtype=np.uint8)
    # Use OpenCV drawing routines per primitive type:
    if isinstance(prim, sk.Line):
        p1 = (int(prim.pntA.x * size), int(prim.pntA.y * size))
        p2 = (int(prim.pntB.x * size), int(prim.pntB.y * size))
        cv2.line(mask, p1, p2, 255, width, cv2.LINE_AA)
    elif isinstance(prim, sk.Arc):
        c  = (int(prim.center.x * size), int(prim.center.y * size))
        r  = int(prim.radius * size)
        a0 = int(np.degrees(prim.startAngle))
        a1 = int(np.degrees(prim.endAngle))
        cv2.ellipse(mask, c, (r, r), 0, a0, a1, 255, width, cv2.LINE_AA)
    elif isinstance(prim, sk.Circle):
        c = (int(prim.center.x * size), int(prim.center.y * size))
        r = int(prim.radius * size)
        cv2.circle(mask, c, r, 255, width, cv2.LINE_AA)
    # Ellipse handled similarly
    return mask

def render_pair(sketch, idx):
    """Produce (input_image, target_labels) for one sketch."""
    per_prim_masks = []
    prim_ids       = []
    for i, prim in enumerate(sketch.entities.values()):
        if type(prim).__name__ == "Point":
            continue
        m = primitive_to_pixels(prim)
        if m.sum() == 0:
            continue
        per_prim_masks.append(m)
        prim_ids.append(i)

    # Union → clean binary raster
    clean = np.max(np.stack(per_prim_masks), axis=0) if per_prim_masks else None
    if clean is None:
        return None

    # Per-pixel primitive label map (0 = background, k = primitive index)
    label_map = np.zeros_like(clean, dtype=np.int32)
    for k, m in enumerate(per_prim_masks, start=1):
        label_map[m > 0] = k

    # Degrade input to look like a scanned/hand-drawn sketch
    degraded = simulate_degradation(clean)

    return degraded, clean, label_map, prim_ids

def simulate_degradation(img):
    """Simulate scan/hand-drawn appearance from a clean rendering."""
    # Random blur
    k = np.random.choice([1, 3, 5])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    # Additive noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Random line-thickness jitter via morphological ops
    if np.random.rand() < 0.3:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    return img

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(SG_PATH, allow_pickle=True)
    for idx, sketch in enumerate(data):
        result = render_pair(sketch, idx)
        if result is None:
            continue
        degraded, clean, label_map, prim_ids = result

        stem = OUT_DIR / f"sg_{idx:08d}"
        cv2.imwrite(str(stem) + "_input.png",  degraded)
        cv2.imwrite(str(stem) + "_clean.png",  clean)
        np.save(str(stem) + "_labels.npy",     label_map)

        if idx % 5000 == 0:
            print(f"[{idx}] wrote {stem}")

if __name__ == "__main__":
    main()
```

**Run in chunks** — the full 2–4 M pass will take many hours. Start with a 100K subset for the first training run:

```bash
python ~/ip-drawing-drafter/data/render_stage2_pairs.py --limit 100000
```

### 5.2 Sanity check

```bash
python - <<'PY'
from pathlib import Path
import cv2, numpy as np
d = Path.home() / "datasets/derived/stage2_pairs"
files = sorted(d.glob("*_input.png"))[:5]
for f in files:
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    lbl = np.load(str(f).replace("_input.png", "_labels.npy"))
    print(f.name, img.shape, "n_primitives=", int(lbl.max()))
PY
```

Expect between 2 and 16 unique primitive labels per sample.

---

## 6. Stage 2 — Puhachov CNN training

### 6.1 Get the Puhachov reference implementation

Puhachov et al. 2021 ("Keypoint-Driven Line Drawing Vectorization via PolyVector Flow", SIGGRAPH Asia 2021) publish their code:

```bash
cd ~/ip-drawing-drafter/models
git clone https://github.com/dli7319/Puhachov-Vectorization.git puhachov   # verify current URL
# If unavailable: check the SIGGRAPH Asia 2021 project page or supplementary
```

The relevant piece is the keypoint-prediction CNN. Copy the network definition into `models/puhachov_net.py` and adapt the training loop to consume the pair format from Section 5.

### 6.2 Training script

Create `~/ip-drawing-drafter/train_puhachov.py`:

```python
"""
Train the Puhachov CNN on real CAD stroke ground truth.
Input:  degraded raster (from stage2_pairs/*_input.png)
Target: per-pixel keypoint heatmap + primitive-membership labels
Loss:   L_keypoint (heatmap MSE) + λ * L_membership (cross-entropy)
"""
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import mlflow

from models.puhachov_net import PuhachovNet

class Stage2Dataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        stems = sorted({p.name.replace("_input.png","")
                        for p in self.root.glob("*_input.png")})
        # 90/10 split
        cut = int(len(stems) * 0.9)
        self.stems = stems[:cut] if split == "train" else stems[cut:]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.root / self.stems[i]
        img  = cv2.imread(str(stem) + "_input.png", cv2.IMREAD_GRAYSCALE)
        lbl  = np.load(str(stem) + "_labels.npy")
        img  = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        # Keypoint heatmap = binary skeleton
        keypoints = torch.from_numpy((lbl > 0).astype(np.float32)).unsqueeze(0)
        membership = torch.from_numpy(lbl.astype(np.int64))
        return img, keypoints, membership

def train(args):
    mlflow.set_experiment("ap3-stage2-puhachov")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = PuhachovNet(n_classes=args.max_prims + 1).to(device)
        opt    = optim.Adam(model.parameters(), lr=args.lr)
        crit_h = nn.MSELoss()
        crit_m = nn.CrossEntropyLoss(ignore_index=0)

        train_ds = Stage2Dataset(args.data_root, "train")
        val_ds   = Stage2Dataset(args.data_root, "val")
        train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4)
        val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4)

        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for img, kp, mem in train_ld:
                img, kp, mem = img.to(device), kp.to(device), mem.to(device)
                pred_kp, pred_mem = model(img)
                loss = crit_h(pred_kp, kp) + args.lam * crit_m(pred_mem, mem)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
            avg = total / len(train_ld)
            mlflow.log_metric("train_loss", avg, step=epoch)
            print(f"epoch {epoch}: train_loss={avg:.4f}")

            # Checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "version": "puhachov_v1",
            }
            torch.save(ckpt, f"{args.ckpt_dir}/puhachov_e{epoch:03d}.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(Path.home()/"datasets/derived/stage2_pairs"))
    ap.add_argument("--ckpt_dir",  default=str(Path.home()/"ip-drawing-drafter/checkpoints/stage2"))
    ap.add_argument("--epochs",    type=int, default=30)
    ap.add_argument("--batch",     type=int, default=16)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--lam",       type=float, default=0.5)
    ap.add_argument("--max_prims", type=int, default=16)
    args = ap.parse_args()
    train(args)
```

Kick off training:

```bash
cd ~/ip-drawing-drafter
mlflow ui --backend-store-uri ./mlruns &   # background: browse at http://localhost:5000
python train_puhachov.py --epochs 30 --batch 16
```

Expect ~4–8 hours per epoch on a single GPU with 100K samples; scale batch/workers based on VRAM.

**Early stopping signal:** validation membership accuracy plateaus around 0.85–0.9 on the SketchGraphs distribution. If it stays under 0.6, first check that the label maps are aligned with the input rasters (common bug).

---

## 7. Stage 3 training data preparation

Stage 3 needs (per-edge raster, primitive parameters) pairs. The existing `generate_sketches_v3.py` already produces this shape — the task is to **replace the synthetic generator with a SketchGraphs consumer** so the training distribution matches real CAD.

### 7.1 Extend `generate_sketches_v3.py`

Create `~/ip-drawing-drafter/generate_sketches_v4_sketchgraphs.py`:

```python
"""
Stage 3 training-pair generator sourced from SketchGraphs.

Per-edge output (one training sample per primitive):
  - raster patch (small crop around the primitive with local context)
  - primitive class label   (0=line, 1=arc, 2=circle, 3=ellipse, 4=polyline)
  - parameter vector        (padded, class-conditional)

Realistic distribution: no axis-aligned bias, no uniform angular spacing,
mixed primitive counts, real designer-chosen dimensions.
"""
from pathlib import Path
import numpy as np
import cv2
from sketchgraphs.data import sketch as sk

SG_PATH = Path.home() / "datasets/sketchgraphs/sg_stage3_ready.npy"
OUT_DIR = Path.home() / "datasets/derived/stage3_pairs"
PATCH   = 128
CONTEXT_MULT = 1.3     # crop size = bbox × context_mult

def primitive_bbox(prim):
    if isinstance(prim, sk.Line):
        return (min(prim.pntA.x, prim.pntB.x), min(prim.pntA.y, prim.pntB.y),
                max(prim.pntA.x, prim.pntB.x), max(prim.pntA.y, prim.pntB.y))
    if isinstance(prim, sk.Circle):
        c, r = prim.center, prim.radius
        return (c.x-r, c.y-r, c.x+r, c.y+r)
    if isinstance(prim, sk.Arc):
        c, r = prim.center, prim.radius
        return (c.x-r, c.y-r, c.x+r, c.y+r)   # loose but sufficient
    return None

def encode_params(prim):
    """Return (class_id, param_vec[12] padded)."""
    v = np.zeros(12, dtype=np.float32)
    if isinstance(prim, sk.Line):
        v[:4] = [prim.pntA.x, prim.pntA.y, prim.pntB.x, prim.pntB.y]
        return 0, v
    if isinstance(prim, sk.Arc):
        v[:5] = [prim.center.x, prim.center.y, prim.radius,
                 prim.startAngle, prim.endAngle]
        return 1, v
    if isinstance(prim, sk.Circle):
        v[:3] = [prim.center.x, prim.center.y, prim.radius]
        return 2, v
    return None, v

def render_patch(sketch, target_prim):
    """Render the whole sketch, then crop around the target primitive."""
    # ... reuse the render code from Section 5.1 to produce a full raster ...
    # ... then compute bbox → crop → resize to PATCH×PATCH ...
    # Returns a uint8 array of shape (PATCH, PATCH)
    raise NotImplementedError("copy render logic from render_stage2_pairs.py")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(SG_PATH, allow_pickle=True)
    n = 0
    for si, sketch in enumerate(data):
        for pi, prim in enumerate(sketch.entities.values()):
            enc = encode_params(prim)
            if enc[0] is None:
                continue
            cls, params = enc
            patch = render_patch(sketch, prim)
            if patch is None:
                continue
            stem = OUT_DIR / f"sg_{si:08d}_{pi:02d}"
            cv2.imwrite(str(stem) + ".png", patch)
            np.save(str(stem) + ".npy", np.concatenate([[cls], params]))
            n += 1
        if si % 10000 == 0:
            print(f"[{si}] wrote {n} samples")

if __name__ == "__main__":
    main()
```

### 7.2 Augment with CAD-VGDrawing multi-view engineering drawings

Add a second data source in the same output format. Multi-view drawings introduce hidden lines and centre lines that single-sketch SketchGraphs does not have. Even a small mix (10–20% CAD-VGDrawing samples) helps generalise to patent-style drawings.

Create `~/ip-drawing-drafter/generate_sketches_v4_vgdrawing.py` that reads `~/datasets/cad_vgdrawing/primitives.jsonl` and emits the same per-edge format.

---

## 8. Stage 3 — Free2CAD v3 training

The existing `train_free2cad_v3.py` and encoder-only architecture (per the project memory: ~1.5M params, classifier + regressor heads, RANSAC fallback below 0.55 confidence) is already correct. **The only change needed is the data source and a two-phase schedule.**

### 8.1 Modify `train_free2cad_v3.py`

Two changes:

1. **Data loader:** point at `~/datasets/derived/stage3_pairs` (combines SketchGraphs and CAD-VGDrawing outputs).
2. **Two-phase schedule:**
   - **Phase A (pretrain):** 100% SketchGraphs, 20 epochs. Gets the fitter into the right ballpark on clean parametric primitives.
   - **Phase B (fine-tune):** 80% SketchGraphs + 20% CAD-VGDrawing, 10 epochs, lower learning rate. Bridges to multi-view engineering drawing style.

```python
# In train_free2cad_v3.py:

def get_dataloader(phase, batch, sg_root, vg_root):
    sg_files = list(Path(sg_root).glob("sg_*.png"))
    vg_files = list(Path(vg_root).glob("vg_*.png"))
    if phase == "A":
        files = sg_files
    else:   # phase B
        n_vg  = int(0.2 * len(sg_files))
        files = sg_files + np.random.choice(vg_files, n_vg, replace=False).tolist()
    return DataLoader(
        Stage3Dataset(files), batch_size=batch, shuffle=True, num_workers=4
    )

# Then in main():
for epoch in range(20):
    train_one_epoch(model, get_dataloader("A", args.batch, ...), lr=1e-4)
    save_ckpt(model, epoch, version="v4_phaseA")

# Load best Phase A checkpoint, drop LR
model.load_state_dict(torch.load(best_A_ckpt)["model_state_dict"])
for epoch in range(10):
    train_one_epoch(model, get_dataloader("B", args.batch, ...), lr=2e-5)
    save_ckpt(model, epoch, version="v4_phaseB")
```

### 8.2 Kick off training

```bash
cd ~/ip-drawing-drafter
python train_free2cad_v3.py \
       --sg_root  ~/datasets/derived/stage3_pairs \
       --vg_root  ~/datasets/derived/stage3_pairs \
       --ckpt_dir ~/ip-drawing-drafter/checkpoints/stage3 \
       --batch    64
```

Expect ~2–3 hours per epoch on 100K training samples with the encoder-only model (batch size 64, single GPU).

---

## 9. Integration into the AP3 pipeline

Both `stage2_stroke_extract.py` and `stage3_primitive_fit.py` already support version-tagged checkpoints (per project memory). Update `config.yaml`:

```yaml
stage2:
  puhachov:
    weights: "~/ip-drawing-drafter/checkpoints/stage2/puhachov_e029.pt"
    version: "puhachov_v1"

stage3:
  free2cad:
    weights: "~/ip-drawing-drafter/checkpoints/stage3/free2cad_v4_phaseB_e009.pt"
    version: "v4_phaseB"
    ransac_confidence_threshold: 0.55   # unchanged fallback threshold
```

Then run the full pipeline on the standard smoke-test input:

```bash
python -m ip_drawing_drafter.run \
       --input tests/fixtures/smoke_sketch.tif \
       --output out/smoke_run \
       --config config.yaml \
       --format both
```

Expected outputs unchanged in structure: `vectors/smoke.svg`, `vectors/smoke.dxf`, plus per-stage `flagged` flags in the pipeline log.

---

## 10. Validation

Three validation sets, in ascending order of ecological validity:

### 10.1 Held-out synthetic (regression)

Reserve 10% of SketchGraphs samples not seen in training. Report:
- Stage 2: keypoint IoU, membership accuracy
- Stage 3: per-class primitive classification accuracy, parameter L2 error

Target: no regression versus the current v3 baseline on the current `generate_sketches_v3.py` synthetic held-out.

### 10.2 OpenSketch (real freehand)

```bash
# Download OpenSketch (small, 900 MB, CC0)
wget -c https://repo-sam.inria.fr/d3/OpenSketch/OpenSketch.zip \
     -P ~/datasets/opensketch/
cd ~/datasets/opensketch && unzip OpenSketch.zip
```

Run the pipeline on all OpenSketch concept sketches. Compare Stage 4 SVG output with the paired designer-drawn *presentation drawings* (OpenSketch supplies both).

Target: qualitative similarity should be visibly better than the current v3 model. This is the closest publicly available proxy for real hand-drawn engineering input.

### 10.3 infoapps sample sketches

Run on 10–20 real invention-disclosure sketches from the infoapps corpus. This is the ecological test that matters.

Target: primitive-fit failure rate (flagged QC field in `Stage3Result`) drops relative to the v3 baseline. Track the exact number in MLflow.

---

## 11. Notes and known caveats

- **SketchGraphs sketches are single-figure.** They do not contain Bezugszeichen with leader lines, multiple figures per sheet, or title blocks. Those are handled by Stage 1b (student project) and the tiling pipeline, not Stages 2–3.
- **Coordinate normalisation matters.** SketchGraphs primitives use Onshape's normalised coordinate system; CAD-VGDrawing SVGs use SVG pixel coordinates (Y-down). The renderer in Section 5.1 must project both into the same PATCH×PATCH pixel frame before saving. This is the same coordinate-discipline hazard already flagged in the project memory (SVG Y-down vs DXF Y-up).
- **Splines are excluded from Stage 3 supervision.** The Stage 3 fitter does not support splines. Any spline in the source data is either discarded (SketchGraphs — <3% of sketches contain splines) or polyline-approximated (CAD-VGDrawing curved SVG paths).
- **Free2CAD v3's encoder-only architecture is unchanged.** The current 1.5M-parameter classifier+regressor already fixes the seq2seq architectural mismatch identified in the memory. This work is about training-distribution correction, not architecture.
- **RANSAC fallback stays in place.** Below 0.55 confidence, the fitter still falls back to classical RANSAC — the weights-optional design principle is preserved.
- **Checkpoint versioning.** New checkpoints use `version: "puhachov_v1"` (Stage 2) and `version: "v4_phaseA"` / `"v4_phaseB"` (Stage 3). The existing inference branching in `stage3_primitive_fit.py` needs one new branch to handle `v4_*` (structurally identical to `v3`, so this is a one-line addition).
- **Licensing.** SketchGraphs is MIT, OpenSketch is CC0, CAD-VGDrawing/DeepCAD are research-use. All compatible with the BayVFP-funded research context of IP DrawingDrafter.
- **Do not commit derived data to Git.** Add `datasets/` and `checkpoints/` to `.gitignore` if not already present. MLflow runs (`mlruns/`) are safe to commit but bloat quickly — consider a shared MLflow tracking server for multi-machine work.

---

## Deliverables checklist

At the end of this recipe you should have:

- [ ] `~/datasets/sketchgraphs/sg_stage3_ready.npy` — filtered SketchGraphs subset
- [ ] `~/datasets/cad_vgdrawing/primitives.jsonl` — parsed CAD-VGDrawing primitives
- [ ] `~/datasets/derived/stage2_pairs/` — Stage 2 training pairs
- [ ] `~/datasets/derived/stage3_pairs/` — Stage 3 training pairs
- [ ] `~/ip-drawing-drafter/checkpoints/stage2/puhachov_e029.pt` — trained Puhachov weights
- [ ] `~/ip-drawing-drafter/checkpoints/stage3/free2cad_v4_phaseB_e009.pt` — trained Free2CAD v4 weights
- [ ] `config.yaml` updated to point at the new checkpoints
- [ ] MLflow run history covering both training phases
- [ ] Validation report on OpenSketch and infoapps sample inputs
- [ ] One-page handoff note (`CONTEXT_STAGE2_STAGE3_V4.md`) summarising the training runs and any deviations from this recipe

Once the last item is done, this branch is ready for review and merge into the main AP3 line.
