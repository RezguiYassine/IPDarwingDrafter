# Stage 2 research — Puhachov keypoint CNN

Optional code for **retraining the Stage 2 keypoint detector** and comparing a
learned keypoint path against the production classical CN path. Not required to
run the pipeline. See the README "Roadmap — Puhachov keypoint CNN" section for
the full plan.

## Why

`_extract_topology` consumes keypoint *clusters* (Phase-3 spike). The classical
path seeds them from the crossing-number map (`_cn_keypoint_clusters`); a learned
detector can supply the same contract instead. A trained checkpoint loads back
through the existing guarded `PuhachovKeypointDetector` loader because we train
the **exact** in-repo architecture (`_build_stacked_hourglass`, 3 channels:
endpoint / junction / corner).

## Pipeline

### 1. Generate ground-truth labels (Phase 1)

```bash
# all four views, train + val, on GPUs (test split held out for comparison)
python -m tools.d2c_keypoint_labels --split train      --views all --workers 8 --s1-device cuda:0
python -m tools.d2c_keypoint_labels --split validation --views all --workers 6 --s1-device cuda:1
```

Each `(sample, view)` is rasterized and run through the real Stage 1 (matching
inference), keypoints are derived from the SVG polyline topology and snapped to
the skeleton, and cached as per-sample `.npz` `{skeleton, kps (N,3)=(x,y,type)}`.
Pass `--audit N` for a colour-coded contact sheet (green=endpoint, red=junction,
blue=corner).

### 2. Train (Phase 2)

```bash
python -m stage2_strokeextraction.research.train_puhachov \
    --labels output/Drawing2CAD/kp_labels \
    --out models/puhachov_d2c.pth \
    --steps 40000 --batch 12 --device cuda:0
```

- Native-scale `512²` crops (kp-centered), lossless 90°/flip aug — keeps 1-px
  skeletons intact and matches the pixel scale of full-res inference.
- CenterNet penalty-reduced focal loss on Gaussian-splat targets.
- Checkpoints the best **per-class peak-F1** on a validation subset.
- Output state_dict matches `_build_stacked_hourglass`, so it loads via the
  guarded loader with no architecture mismatch.

### 3. Compare (Phases 0/4)

Point an eval config at the checkpoint (`puhachov.weights: models/puhachov_d2c.pth`)
and run `tools/d2c_eval.py` on the test split; diff against the CN baseline
(Chamfer / IoU / primitive count / runtime). See the README roadmap.

## Files

- `train_puhachov.py` — trainer (dataset, focal loss, val F1, checkpointing).
- label generator lives at `tools/d2c_keypoint_labels.py`.
