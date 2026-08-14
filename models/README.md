# Models

Pre-trained model weights consumed by the AP3 vectorization pipeline at runtime.

## What's in the repo

| File                       | Size  | Used by                  | Source                     |
|----------------------------|------:|--------------------------|----------------------------|
| `puhachov_keypoints.pth`   | 22 MB | Stage 2 (stroke graph)   | shipped in repo            |
| `free2cad_v3_best.pth`     | 3 MB  | Stage 3 (research only)  | shipped in repo            |

These two weights are small enough to ship inline. The pipeline runs out of the box for Stages 2–4 once the repo is cloned.

## Local trained research candidates

| File | Size | Used by | Status |
|---|---:|---|---|
| `free2cad_sketchgraphs_d2c_mixed.pth` | 9.2 MB | Stage 3 research fitter | selected five-class mixed checkpoint; generated locally and Git-ignored |
| `puhachov_patentvec_complexityA.pth` | 29 MB | Stage 2 research detector | final three-epoch PatentVec A candidate; full/PatentData gates pending |
| `puhachov_patentvec_complexityB.pth` | 29 MB | Stage 2 research detector | final three-epoch PatentVec B candidate; full/PatentData gates pending |
| `puhachov_patentvec_referencefree_topology.pth` | 29 MB | Stage 2 research detector | selected C2 reference-free topology candidate; PatentData gate pending |
| `hatch_stroke_multilabel.pth` | 165 MB | Stage 2 hatch-stroke research | synthetic warm start; rejected for real deployment |
| `hatch_stroke_multilabel_real35.pth` | 165 MB | Stage 2 hatch-stroke research | rejected: real structural recall 0.994210 |
| `hatch_stroke_multilabel_real50.pth` | 165 MB | Stage 2 hatch-stroke research | rejected: real structural recall 0.991928 and guarded E2E integration is inert |
| `hatch_stroke_multilabel_realsep_n4.pth` / `n8.pth` | 165 MB each | Stage 2 hatch-stroke research | stopped after epoch 1: structural recall collapsed |

The mixed checkpoint recognizes LINE/ARC/CIRCLE/POLYLINE/BEZIER and has
SHA-256
`dd09a1d54bda5bf178d80cf56f7200723ac132e8a532ede18a5e7c731659db34`.
It scored 0.9683 macro-F1 on mixed validation, 0.8603 on untouched
Drawing2CAD test, and 0.9936 on full SketchGraphs test. It remains a research
candidate until paired end-to-end Drawing2CAD and filtered-PatentData gates
pass. See
[`TRAIN_STAGE2_STAGE3.md`](../TRAIN_STAGE2_STAGE3.md#mixed-sketchgraphs--drawing2cad-stage-3-2026-07-19)
for the complete reproducibility record.

The PatentVec Stage 2 candidates use an exact four-pool rehearsal over
Drawing2CAD, SketchGraphs, ArchCAD, and synthetic labels. Their fixed
three-real-domain macro-F1 is `0.847668` (A) and `0.847748` (B), versus
`0.8193` for their shared warm start. They improve complete synthetic A/B
validation from `0.1746/0.1650` to `0.5047/0.4791`. The difference between A
and B on the real selectors is negligible, so neither is a default production
model until filtered-PatentData regression is complete.
See
[`TRAIN_STAGE2_STAGE3.md`](../TRAIN_STAGE2_STAGE3.md#patentvec-synthetic-ab-experiment-2026-07-25)
for checkpoint history and artifact paths.

Full cached evaluation leaves A and B tied: their equal-domain means over
Drawing2CAD test, SketchGraphs test, and ArchCAD validation are `0.840928` and
`0.840458`. A wins SketchGraphs/ArchCAD; B wins Drawing2CAD. Filtered
PatentData, rather than this `0.00047` margin, decides promotion.

The C2 checkpoint fixes a mismatch between synthetic vector keypoints and the
actual reference-free raster topology consumed by Stage 2. It was selected at
step 14,961 with `0.853213` mean macro-F1 over fixed Drawing2CAD,
SketchGraphs, and ArchCAD guards. Complete C2 validation improves from
`0.705869` to `0.845934`; full SketchGraphs test is `0.935981` and full ArchCAD
validation is `0.716139`. Its SHA-256 is
`2cce917f7db3a3942d8767af8cd0fde1aa7576bdced1556556e0c1931e6b091a`.
It remains a research candidate until full Drawing2CAD and the paired
filtered-PatentData continuity, raster-fidelity, and visual gates complete.

The hatch-stroke checkpoints are local research artifacts, not production
dependencies. The final source-separated geometric integration ties the region
detector on all 100 patent-disjoint figures through Stage 4 and selects zero
model-only edges. See the executed hatch-stroke section in
[`TRAIN_STAGE2_STAGE3.md`](../TRAIN_STAGE2_STAGE3.md#12-executed-hatch-stroke-adaptation-2026-08-07).

## What's NOT in the repo

| File                       | Size   | Used by              | How to obtain                    |
|----------------------------|-------:|----------------------|----------------------------------|
| `sketchcleannet.pth`       | 124 MB | Stage 1 (DL cleaning) | Run [`setup.sh`](../setup.sh) — or download manually and place here |

`sketchcleannet.pth` exceeds GitHub's 100 MB per-file limit and is hosted externally. The download URL is set in `setup.sh`. If unavailable, Stage 1 automatically falls back to its **classical cleaning mode** (Otsu + adaptive threshold + morphology), so the pipeline still runs end-to-end — just with somewhat noisier output on photographed/shaded sketches.

## Manual download (if `setup.sh` does not work)

1. Obtain `sketchcleannet.pth` from your project administrator or partner-shared storage.
2. Place it at `models/sketchcleannet.pth`.
3. Confirm the path in `config.yaml` matches (it should, by default).

## Re-training

If you want to re-train any of these models on new data, see the `research/` subdirectory of the corresponding stage:

- Stage 1: [stage1_preprocessing/research/](../stage1_preprocessing/research/README.md)
- Stage 3: [stage3_primitivesfitting/research/](../stage3_primitivesfitting/research/README.md)
