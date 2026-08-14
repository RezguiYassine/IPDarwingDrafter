# Training Stages 2 and 3 on Real CAD Vector Ground Truth

> **Repository audit (2026-07-17):** The original proposal below assumed a
> different, flat checkout at `~/ip-drawing-drafter` and an external
> `~/datasets` tree. Those paths and several data/model contracts do not match
> this repository. Use the verified workflow in this section. The longer
> proposal is retained after it as design history, not as executable commands.

## PatentVec synthetic A/B experiment (2026-07-25)

This controlled experiment compares the frozen baseline curriculum A with the
enhanced medium/hard/very-hard curriculum B. Both use seed `850725`, identical
optimizer budgets, fixed real holdouts, and a retained step-zero candidate.
The data and audit evidence is documented in `syntheticData/README.md`.

### Stage 2

An initial 85% real / 15% synthetic run replayed Drawing2CAD and SketchGraphs
but used ArchCAD only for validation. At step 5,000 its ArchCAD macro-F1 fell
from `0.627` to `0.501/0.507`; those runs were stopped and retained under
`output/PatentVecComplexityABTraining/stage2/diagnostic_replay3way`.

The corrected exact mixer contains 59,840 samples per epoch:

| source | samples/epoch | share |
|---|---:|---:|
| Drawing2CAD | 20,346 | 34.0% |
| cached SketchGraphs | 25,432 | 42.5% |
| ArchCAD guard replay | 5,086 | 8.5% |
| PatentVec A or B | 8,976 | 15.0% |

All synthetic labels appear exactly once per epoch. Selected real-domain
indices are unique within the epoch. Both variants trained for three epochs,
14,961 optimizer steps, at `lr=1e-5` from
`models/puhachov_sketchgraphs_cadvg40_phaseB.pth`.

| variant/checkpoint | D2C | SketchGraphs | ArchCAD | mean |
|---|---:|---:|---:|---:|
| step zero, shared | 0.8670 | 0.9640 | 0.6270 | 0.8193 |
| A, step 2,500 | 0.833 | 0.958 | 0.689 | 0.827 |
| A, step 7,500 | 0.849 | 0.961 | 0.721 | 0.844 |
| A, step 10,000 | 0.856 | 0.961 | 0.721 | 0.846 |
| **A, final 14,961** | **0.8510** | **0.9620** | **0.7301** | **0.8477** |
| B, step 2,500 | 0.829 | 0.958 | 0.687 | 0.825 |
| B, step 7,500 | 0.848 | 0.961 | 0.718 | 0.842 |
| B, step 10,000 | 0.855 | 0.961 | 0.719 | 0.845 |
| **B, final 14,961** | **0.8539** | **0.9617** | **0.7277** | **0.8477** |

The final aggregates differ by only `0.00008`; treat A and B as tied on these
real selectors. Synthetic validation confirms adaptation:

| checkpoint | synthetic A macro-F1 | synthetic B macro-F1 |
|---|---:|---:|
| shared warm start | 0.1746 | 0.1650 |
| model A | **0.5047** | 0.4678 |
| model B | 0.5008 | **0.4791** |

Complete cached real-domain evaluation gives the same conclusion:

| checkpoint | Drawing2CAD test (31,524) | SketchGraphs test (9,565) | ArchCAD val (1,159) | three-domain mean |
|---|---:|---:|---:|---:|
| model A | 0.856136 | **0.936599** | **0.730050** | **0.840928** |
| model B | **0.857240** | 0.936426 | 0.727709 | 0.840458 |

B is better on Drawing2CAD by `0.00110`; A is better on SketchGraphs by
`0.00017` and ArchCAD by `0.00234`. The full-domain mean differs by `0.00047`.
This is not a meaningful model-selection margin; filtered PatentData remains
the deployment tie-breaker.

Selected checkpoints:

```text
models/puhachov_patentvec_complexityA.pth
models/puhachov_patentvec_complexityB.pth
```

Logs, resumable states, periodic checkpoints, and evaluation JSON files are in
`output/PatentVecComplexityABTraining/stage2/guard4way`.

### Stage 3

Each paired corpus contains 914,250 trainer-visible labels, exactly 182,850 per
class. The 822,349 real rows and 48,996-label validation set are aligned; only
the 91,901 synthetic rows differ. Both variants trained for eight epochs from
`models/free2cad_sketchgraphs_d2c_mixed.pth`.

| checkpoint | real val loss | real macro-F1 | synthetic own-set macro-F1 | synthetic polyline F1 |
|---|---:|---:|---:|---:|
| shared step zero | **0.3163** | **0.9683** | A 0.6001 / B 0.6111 | A 0.1951 / B 0.2367 |
| A, epoch 8 | 0.3271 | 0.9622 | **0.7862** | **0.8419** |
| B, epoch 8 | 0.3262 | 0.9626 | **0.7793** | **0.8228** |

The adapted states learn the synthetic polyline contract but regress the fixed
real validation set. Therefore both `free2cad_v3_best*.pth` files intentionally
remain step-zero checkpoints. The epoch-8 states are retained as
`free2cad_v3_latest.pth` for Pareto analysis and are not production candidates.
All Stage 3 artifacts are under
`output/PatentVecComplexityABTraining/stage3`.

### Current decision

Stage 2 synthetic rehearsal is successful and passes full cached real-domain
evaluation. It advances to filtered-PatentData regression. Stage 3 does not justify replacing the existing
Free2CAD model; production remains on RANSAC. Full 50k generation remains
blocked until the PatentData visual gate and source-license review pass.

## Reference-free raster topology repair (2026-07-31)

The A/B experiment exposed a supervision-contract error rather than a model
capacity limit. The archived synthetic keypoints described selected vector
geometry, while production Stage 2 traces the complete reference-free raster
skeleton. On a fixed 500-drawing B audit, crossing number found 91,773
endpoints and 245,331 junctions, but only 4,920 endpoints and 5,532 junctions
were represented by the archived labels. Approximately 97-98% of hatch
topology was therefore unlabeled. Training on that contract penalized correct
hatch/contact detections and encouraged fragmented long strokes.

Two corrected contracts were evaluated:

- `supported-raster-topology-v1` (C1) exactly labels the complete rendered
  skeleton. It closes the numerical topology gap, but also teaches junctions
  created by reference numerals, leaders, dimensions, and text. C1 is retained
  as a diagnostic export and was not trained.
- `reference-free-raster-topology-v1` (C2) rebuilds the Stage 2 input from the
  object, hidden/centre-line, and hatch masks, excluding references, leaders,
  dimensions, and text. Endpoints and junctions come from the production
  crossing-number implementation; true vector corners are re-snapped to that
  skeleton. This mirrors the post-Stage-0 inference contract.

The immutable C2 export is
`output/PatentVecComplexityB10kTrainingReferenceFreeTopologyV1`. It contains
8,976 train and 1,024 validation drawings with 4,766,069 labels: 1,153,393
endpoints, 3,513,144 junctions, and 99,532 corners. An independent in-memory
audit over 500 source drawings matched all 57,582 endpoints and all 179,539
junctions, including all 4,595 hatch endpoints and 167,766 hatch junctions.
All 4,397,254 reference-free skeleton pixels are supported by the declared
semantic masks. Evidence is stored in:

```text
output/PatentVecComplexityABTraining/stage2/topology_contract_C2_reference_free_500.json
output/PatentVecComplexityB10kTrainingReferenceFreeTopologyV1/audit_reference_free_topology.png
```

Dense C2 junction supply would dominate the previous globally normalized focal
loss. The trainer therefore adds `--focal-normalization sample-class`, which
reduces every sample/channel independently before averaging. The historical
global reduction remains the default for reproducibility. C2 then used the
same exact 59,840-sample epoch and real-domain replay schedule as A/B.

| checkpoint | Drawing2CAD | SketchGraphs | ArchCAD | selector mean |
|---|---:|---:|---:|---:|
| warm start | 0.867 | 0.964 | 0.627 | 0.819 |
| step 2,500 | 0.851 | 0.961 | 0.698 | 0.837 |
| step 5,000 | 0.858 | 0.962 | 0.704 | 0.841 |
| step 7,500 | 0.871 | 0.963 | 0.709 | 0.848 |
| step 10,000 | 0.878 | 0.962 | 0.710 | 0.850 |
| step 12,500 | 0.876 | 0.962 | **0.717** | 0.852 |
| **final 14,961** | **0.880219** | **0.962832** | 0.716589 | **0.853213** |

The final checkpoint was selected and copied immutably after three exact
epochs. Its C2 validation macro-F1 improves from `0.705869` at warm start to
`0.845934`; endpoint/junction/corner F1 are
`0.913605/0.710363/0.913833`. Full SketchGraphs test is effectively unchanged
at `0.935981`. Full ArchCAD validation is `0.716139`, a `0.01157` regression
from model B, so deployment still requires a compensating paired PatentData
continuity/fidelity gain. Full Drawing2CAD and the canonical 100-drawing
PatentData comparison were still running when this subsection was written.

```text
models/puhachov_patentvec_referencefree_topology.pth
SHA-256 2cce917f7db3a3942d8767af8cd0fde1aa7576bdced1556556e0c1931e6b091a
output/PatentVecComplexityABTraining/stage2/reference_free_topology
```

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

#### Phase B: 60/40 CAD-VGDrawing retraining (started 2026-07-19)

The repository's `data/Drawing2CAD` directory is the released CAD-VGDrawing
corpus, not a separate precursor dataset. Its official split contains 141,831
train, 7,879 validation, and 7,881 test CAD models, with four drawing views per
model. The `svg_raw`, `svg_vec`, and `cad_vec` archives and
`train_val_test_split.json` match that release. Consequently, Phase A already
trained on 70% SketchGraphs / 30% CAD-VGDrawing; adding a separate "VG" input
would duplicate the same examples.

A deterministic 20,000-file sample from each cached training pool measured the
following keypoint supply:

| source | endpoints/view | junctions/view | corners/view |
|---|---:|---:|---:|
| CAD-VGDrawing | 0.8192 | 2.5328 | 4.2686 |
| SketchGraphs pilot cache | 0.4316 | 0.9406 | 2.4646 |

The next controlled increment is therefore **60% SketchGraphs / 40%
CAD-VGDrawing**. At that image ratio CAD-VGDrawing contributes approximately
56% of endpoint, 64% of junction, and 54% of corner labels. It targets Phase
A's remaining junction-recall weakness without giving up exact coverage of any
SketchGraphs source record. The full mixed epoch has 9,179,789 SketchGraphs
slots plus 6,119,859 CAD-VGDrawing slots: 15,299,648 samples and 637,486
two-GPU optimizer steps.

Phase B warm-starts the selected Phase A checkpoint at half the learning rate.
Checkpoint selection uses the mean macro-F1 over fixed 2,000-view
CAD-VGDrawing and SketchGraphs validation subsets and all 1,159 ArchCAD
validation drawings:

```bash
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/torchrun \
  --standalone --nproc_per_node=2 \
  -m stage2_strokeextraction.research.train_puhachov \
  --labels output/Drawing2CAD/kp_labels \
  --sketchgraphs-raw data/SketchGraphs/raw/sg_t16_train.npy \
  --sketchgraphs-val-labels output/SketchGraphsTraining/stage2 \
  --tertiary-val-labels output/ArchCAD/kp_labels \
  --mix 0.40 --steps 0 --epochs 1 --lr 5e-5 \
  --init-weights models/puhachov_sketchgraphs_full_phaseA.pth \
  --out models/puhachov_sketchgraphs_cadvg40_phaseB.pth \
  --state-out models/puhachov_sketchgraphs_cadvg40_phaseB_last.pth \
  --coverage-file output/SketchGraphsCADVG40/stage2_train.coverage.i8 \
  --require-full-coverage \
  --batch 12 --workers 6 --prefetch-factor 2 --amp \
  --val-subset 2000 --secondary-val-subset 2000 \
  --tertiary-val-subset 1159 \
  --val-every 10000 --save-every 5000 --log-every 100
```

The best checkpoint is not promoted from validation alone. After completion it
must pass the full SketchGraphs test, full CAD-VGDrawing validation/test,
ArchCAD validation, paired CAD-VGDrawing end-to-end, and filtered-PatentData
visual gates against Phase A.

The command above is running as the persistent user service
`puhachov-cadvg40-phaseb.service`. Its append-only log is
`output/SketchGraphsCADVG40/phaseB_training.log`. The launch smoke test passed
DDP, three-domain validation, and both checkpoint formats; the production run
reached step 300 at 8.15 optimizer steps/s with both GPUs active. At that rate,
the training portion is approximately 22 hours, excluding validation pauses.
Inspect it without interrupting training:

```bash
systemctl --user --no-pager status puhachov-cadvg40-phaseb.service
tail -50 output/SketchGraphsCADVG40/phaseB_training.log
```

The resumable checkpoint is written atomically every 5,000 steps. If the
service is interrupted, repeat the same command with `--resume
models/puhachov_sketchgraphs_cadvg40_phaseB_last.pth` in place of
`--init-weights`; world size and per-rank batch must remain unchanged.

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

### Full Stage 3 workflow

The 100,000-source Stage 3 pilot is intentionally not treated as full training.
Full-corpus extraction is more expensive than Stage 2 keypoint rendering because
it runs the production topology and simplification path, then matches each
resulting edge back to its source primitive. The chunked path bounds process-pool
submission, writes each NPZ atomically, and records a completion marker and
configuration manifest per 10,000 source sketches. `--resume` verifies those
markers and skips completed chunks without rewriting them.

Free2CAD training uses one shard at a time instead of concatenating the expected
roughly 27 million edges in RAM. It warm-starts from the corrected pilot,
deterministically shuffles shards and samples, and atomically saves model,
optimizer, scheduler, and next-shard position every ten shards. The full queue is:

```bash
systemd-run --user --unit=free2cad-stage3-full \
  --description="Full SketchGraphs Stage 3 extraction and Free2CAD training" \
  tools/run_stage3_full_training_queue.sh
```

The queue performs, in order:

1. Full train, validation, and test extraction from the official splits with 16
   CPU workers and restart-safe 10,000-source chunks.
2. Two FP32 epochs on GPU 1, batch 512, three-point arc encoding, square-root
   inverse-frequency class weights, and learning rate `3e-5`.
3. Full untouched-test evaluation of the 100,000-source pilot, the best-loss
   full checkpoint, and the best-macro-F1 full checkpoint.

The warm-start is deliberate: the corrected pilot already reaches 0.9987
macro-F1, so full training is a low-learning-rate long-tail coverage pass rather
than a random restart. A 2,000-source, 16-worker benchmark sustained about 158
sketches/second after startup, implying roughly 16 hours for the train split
plus about one hour for validation/test extraction. A real interrupted-training
probe resumed exactly from shard position 4/6 and completed validation.

Live progress and final artifacts are written under:

```text
output/SketchGraphsStage3Full/progress_{train,validation,test}.json
output/SketchGraphsStage3Full/stage3/{train,val,test}/manifest.json
output/SketchGraphsStage3Full/checkpoints/free2cad_full_phaseA/
output/SketchGraphsStage3Full/evaluation_*_full_test.json
output/SketchGraphsStage3Full/stage3_full_queue.log
```

### Full Stage 3 completion (2026-07-19)

The restart-safe queue completed all extraction, both training epochs, full
validation, and three full-test evaluations. The systemd unit exited
successfully. Training ran once; later idempotent queue invocations detected the
completed epoch-2 state and only regenerated the same evaluation reports.

#### Corpus coverage

Every official source record was attempted. `accepted` means that supported
non-construction geometry produced a non-empty Stage 2 topology; individual
edges could still be unmatched to a source primitive or shorter than the
five-pixel supervision floor.

| split | source sketches | accepted | rejected | graph edges | unmatched | short | Stage 3 samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 9,179,789 | 8,802,845 | 376,944 | 32,310,428 | 4,542,234 | 744,200 | **27,023,994** |
| validation | 315,228 | 302,166 | 13,062 | 1,113,582 | 155,843 | 24,827 | **932,912** |
| test | 313,271 | 300,093 | 13,178 | 1,108,631 | 156,226 | 25,465 | **926,940** |

| rejection reason | train | validation | test |
|---|---:|---:|---:|
| no supported non-construction geometry | 369,486 | 12,831 | 12,922 |
| topology has no edges | 6,572 | 193 | 228 |
| degenerate geometry bounds | 886 | 38 | 28 |

The exact post-matching class distribution is:

| split | line | arc | circle | polyline |
|---|---:|---:|---:|---:|
| train | 20,430,593 | 1,486,670 | 5,106,731 | 0 |
| validation | 713,668 | 49,364 | 169,880 | 0 |
| test | 708,356 | 50,303 | 168,281 | 0 |

#### Training configuration and history

The 794,762-parameter encoder-only model was warm-started from
`models/free2cad_sketchgraphs.pth`, then trained in FP32 on GPU 1. The point
encoder uses `max_pts=64`, `d_model=128`, eight heads, four encoder layers, and
dropout 0.1. Arc targets use the bounded three-point encoding. Other settings:

| setting | value |
|---|---:|
| epochs | 2 |
| batch size | 512 |
| initial learning rate | `3e-5` |
| final learning rate | `1e-6` |
| parameter-loss weight | 0.5 |
| label smoothing | 0.05 |
| class-weight power | 0.5 |
| line / arc / circle / polyline weights | 0.5750 / 2.1318 / 1.1502 / 0.0 |

| epoch | train loss | validation loss | validation accuracy | macro-F1 | ending LR | time |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.2357 | 0.2359 | 1.000 (rounded) | 0.9989 | `1.55e-5` | 1,161.1 s |
| **2** | **0.2338** | **0.2358** | **0.9998** | **0.9992** | `1e-6` | 1,168.7 s |

Epoch 2 won both checkpoint criteria. Its exact validation results are:

| class | precision | recall | F1 | parameter L1 | stroke residual | support |
|---|---:|---:|---:|---:|---:|---:|
| line | 1.0000 | 0.9997 | 0.9999 | 0.0012 | 0.0013 | 713,668 |
| arc | 0.9958 | 0.9998 | 0.9978 | 0.0057 | 0.0233 | 49,364 |
| circle | 1.0000 | 0.9998 | 0.9999 | 0.0025 | 0.0051 | 169,880 |

Validation has 221 total classification errors: 183 true lines and 26 true
circles were predicted as arcs, while 12 true arcs were predicted as lines.
There were no line/circle confusions.

#### Full untouched-test evaluation

The 100,000-source pilot and both full-checkpoint selections were evaluated on
the same 926,940 test edges. Best validation loss and best validation macro-F1
both selected epoch 2 and therefore have identical predictions.

| checkpoint | accuracy | supported macro-F1 | line F1 | arc F1 | circle F1 |
|---|---:|---:|---:|---:|---:|
| 100K pilot | 0.9997 | 0.9989 | 0.9998 | 0.9971 | 0.9999 |
| **full epoch 2** | **0.9998** | **0.9993** | **0.9999** | **0.9981** | **1.0000** |

| class | support | pilot param L1 | full param L1 | pilot residual | full residual |
|---|---:|---:|---:|---:|---:|
| line | 708,356 | 0.0015 | **0.0012** | 0.0014 | **0.0013** |
| arc | 50,303 | 0.0058 | **0.0057** | **0.0222** | 0.0233 |
| circle | 168,281 | 0.0029 | **0.0025** | 0.0058 | **0.0053** |

The full-model test confusion matrix has 162 true lines predicted as arcs, 12
true arcs predicted as lines, and 15 true circles predicted as arcs. All other
926,751 predictions are correct. Full training therefore improves classification
and normalized parameter accuracy, but arc stroke residual regresses by 0.0011
(about 5%). This is an explicit downstream integration check, not hidden by the
aggregate score.

Pilot and full `val_loss` values must not be compared: the pilot checkpoint
predates persisted label-smoothing metadata, so the evaluator uses smoothing
0.0 for the pilot and 0.05 for the full model. Accuracy, F1, parameter L1, and
stroke residual use the same contract and are directly comparable.

#### Selected model and decision

The selected checkpoint is the epoch-2 best-validation-loss model:

```text
output/SketchGraphsStage3Full/checkpoints/free2cad_full_phaseA/free2cad_v3_best.pth
models/free2cad_sketchgraphs_full.pth
```

The second path is a stable byte-identical deployment-candidate alias. File
size is 9,640,160 bytes and SHA-256 is
`bed27ed08ad794757bb93192c9347caf0af9d18f6f62a15a39537f0fc4d97d3f`.
`free2cad_v3_best_f1.pth` contains the same epoch and metrics but is a separate
serialization; use `free2cad_v3_best.pth` as the canonical artifact.

This is the best **SketchGraphs-domain** Free2CAD model, not yet the production
Stage 3 default. SketchGraphs supplies no polyline examples, and the test is
in-domain. Production remains the guarded deterministic/RANSAC fitter until a
paired Drawing2CAD end-to-end comparison measures primitive type, Chamfer,
parameter error, fragmentation, and runtime, followed by filtered-PatentData
visual regression focused on arcs and compound paths.

Full evidence files:

```text
output/SketchGraphsStage3Full/progress_{train,validation,test}.json
output/SketchGraphsStage3Full/evaluation_pilot_full_test.json
output/SketchGraphsStage3Full/evaluation_best_full_test.json
output/SketchGraphsStage3Full/evaluation_best_f1_full_test.json
output/SketchGraphsStage3Full/stage3_full_queue.log
```

### Mixed SketchGraphs + Drawing2CAD Stage 3 (2026-07-19)

#### Why sample-count mixing is rejected

SketchGraphs supplies excellent analytic supervision but no polyline class. Its
full train extraction is dominated by 20,430,593 line edges, so a nominal
"70% SketchGraphs / 30% Drawing2CAD" source-sketch ratio would still make
POLYLINE and BEZIER negligible. The mixed run is balanced by the number of
**post-Stage-2 edge labels** available to each class, not source drawings or
files.

Drawing2CAD is converted by `tools/d2c_stage3_dataset.py`. For each cached
Stage 1 skeleton and ground-truth keypoint set, the tool runs the repository's
actual Stage 2 topology/simplification/smoothing path, then matches each graph
edge back to dense SVG source geometry. Label guards are:

- one measurably straight linear source span -> `LINE`;
- several contiguous linear source segments with a non-straight extracted edge
  -> `POLYLINE`;
- SVG cubic geometry with a low radial residual -> `ARC` or `CIRCLE`, because
  CAD SVG exporters encode circles as cubic commands;
- remaining non-circular cubic geometry -> `BEZIER`;
- mixed line/cubic, short, and unmatched edges are excluded rather than given a
  noisy label.

Closed-loop pixels require special handling. Stage 2 stores them in scanline
order, which produced zigzag point tensors in the first visual audit. Dataset
construction and Free2CAD inference now both apply the production loop-ordering
routine before normalization or deterministic curve fitting. The corrected
class audit is
`output/Drawing2CAD/stage3_ordered_pilot/val/class_audit.png`.

The converter is restart-safe at source chunks, writes atomic NPZ shards, and
prebuilds one source-geometry KD-tree per view. Full train, validation, and test
conversion completed without errors:

| split | views | graph edges | accepted labels | line | arc | circle | polyline | bezier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 567,324 | 3,946,373 | **3,265,723** | 2,779,020 | 44,157 | 104,870 | **182,850** | **154,826** |
| validation | 31,516 | 219,243 | **180,889** | 153,366 | 2,632 | 6,042 | **10,049** | **8,800** |
| test | 31,524 | 220,349 | **182,428** | 155,268 | 2,396 | 5,754 | **10,412** | **8,598** |

Train excluded 552,699 unmatched, 69,330 short, and 58,621 mixed-command edges.
Validation excluded 30,972 unmatched, 4,093 short, and 3,289 mixed-command
edges. Test excluded 30,650 unmatched, 3,969 short, and 3,302 mixed-command
edges. These are edge-level quality exclusions, not failed drawings; every
view in all three splits completed with zero worker errors.

Reproduce the conversion with BLAS threading disabled inside each process:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m tools.d2c_stage3_dataset \
  --split train --workers 56 --source-chunk-size 512 \
  --output output/Drawing2CAD/stage3
```

`tools/build_free2cad_mixed_dataset.py` scans the labels visible to the trainer,
including the same incomplete-circle rejection and straight-arc relabeling.
The manifest records both raw and effective supply. Its automatic train target
is the larger of Drawing2CAD's effective POLYLINE and BEZIER counts. It
downsamples every analytic class to that target, sources analytic labels from
SketchGraphs/Drawing2CAD at 70/30 when available, and permits at most 1.5x
reuse to close a modest rare-class gap. Validation uses no repeated labels
(`--max-repeats 1.0`). Selected rows are globally shuffled before mixed
50,000-edge shards are written; class-specific batches are explicitly avoided.

The completed train mix contains **914,250 effective edges**, exactly 182,850
per class. LINE/ARC/CIRCLE each contribute 127,995 SketchGraphs and 54,855
Drawing2CAD labels. POLYLINE contributes all 182,850 unique Drawing2CAD labels.
BEZIER contributes 182,850 Drawing2CAD labels from 154,826 unique labels, a
bounded 1.181x exposure. The domain totals are 383,985 SketchGraphs and 530,265
Drawing2CAD labels. The trainer re-audit reports zero filtered or relabeled
records. The non-duplicated validation mix contains 48,996 edges: 10,049 each
for LINE/ARC/CIRCLE/POLYLINE and all 8,800 available BEZIER labels.

Exact mixed-corpus materialization commands:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m tools.build_free2cad_mixed_dataset \
  --drawing2cad output/Drawing2CAD/stage3 \
  --output output/Free2CADMixedSketchGraphsD2C \
  --split train --max-repeats 1.5 --shard-size 50000 --overwrite

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
.venv/bin/python -m tools.build_free2cad_mixed_dataset \
  --drawing2cad output/Drawing2CAD/stage3 \
  --output output/Free2CADMixedSketchGraphsD2C \
  --split val --max-repeats 1.0 --shard-size 50000 --overwrite
```

The resulting train split has 19 shards: 18 x 50,000 labels and one x 14,250.
Its domain totals are 383,985 SketchGraphs and 530,265 Drawing2CAD labels.

A 2,500-edge five-class smoke run first verified four-to-five-class checkpoint
expansion and the complete train/evaluate path. Despite only 500 train labels
per class, epoch 20 reached 0.8535 validation macro-F1, including POLYLINE F1
0.892 and BEZIER F1 0.676. This smoke checkpoint is diagnostic only.

Free2CAD v3 now has a five-row type head. Warm-starting from the selected
four-class SketchGraphs checkpoint copies the encoder, parameter head, and the
first four classifier rows exactly; only the new BEZIER row keeps its random
initialization. POLYLINE and BEZIER use deterministic dense-point geometry
after classification and do not contribute to the six-value parameter loss.
Old four-class checkpoints remain loadable because model construction and
evaluation take the checkpoint vocabulary size.

#### Full mixed training

Both runs warm-started `models/free2cad_sketchgraphs_full.pth`, used batch 1,024,
64 points, three-point arc encoding, label smoothing 0.05, and parameter weight
0.5. Equal effective class supply makes all class weights exactly 1.0.

| schedule | epochs | learning rate | GPU time | best epoch | val loss | val accuracy | val macro-F1 | polyline F1 | bezier F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conservative | 12 | 3e-5 | 572.4 s | 10 | 0.3575 | 0.9536 | 0.9528 | 0.9415 | 0.9203 |
| **adaptive** | **20** | **1e-4** | **835.0 s** | **20** | **0.3163** | **0.9687** | **0.9683** | **0.9566** | **0.9491** |

Adaptive validation F1 by class is LINE 0.9902, ARC 0.9499, CIRCLE 0.9956,
POLYLINE 0.9566, and BEZIER 0.9491. The higher-rate schedule wins every full
evaluation and is selected.

Exact training commands:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python stage3_primitivesfitting/research/train_free2cad_v3.py \
  --data_dir output/Free2CADMixedSketchGraphsD2C \
  --output_dir output/Free2CADMixedSketchGraphsD2C/checkpoints/conservative \
  --epochs 12 --batch_size 1024 --lr 3e-5 --device cuda --max_pts 64 \
  --d_model 128 --n_heads 8 --n_enc_layers 4 --dropout 0.1 \
  --arc_encoding three_point --class_weight_power 0.5 \
  --label_smoothing 0.05 --param_weight 0.5 --stream_shards \
  --save_every_shards 0 --init_weights models/free2cad_sketchgraphs_full.pth

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python stage3_primitivesfitting/research/train_free2cad_v3.py \
  --data_dir output/Free2CADMixedSketchGraphsD2C \
  --output_dir output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive \
  --epochs 20 --batch_size 1024 --lr 1e-4 --device cuda --max_pts 64 \
  --d_model 128 --n_heads 8 --n_enc_layers 4 --dropout 0.1 \
  --arc_encoding three_point --class_weight_power 0.5 \
  --label_smoothing 0.05 --param_weight 0.5 --stream_shards \
  --save_every_shards 0 --init_weights models/free2cad_sketchgraphs_full.pth
```

Complete epoch history as written to `train.log` (F1 is rounded to three
decimal places in the log):

| epoch | conservative train/val/F1 | adaptive train/val/F1 |
|---:|---:|---:|
| 1 | 0.7419 / 0.4334 / 0.922 | 0.5545 / 0.3973 / 0.936 |
| 2 | 0.4376 / 0.3979 / 0.935 | 0.4006 / 0.3765 / 0.943 |
| 3 | 0.4139 / 0.3869 / 0.939 | 0.3820 / 0.3620 / 0.950 |
| 4 | 0.4017 / 0.3818 / 0.941 | 0.3697 / 0.3535 / 0.954 |
| 5 | 0.3930 / 0.3700 / 0.947 | 0.3607 / 0.3415 / 0.958 |
| 6 | 0.3865 / 0.3658 / 0.949 | 0.3535 / 0.3412 / 0.959 |
| 7 | 0.3825 / 0.3635 / 0.950 | 0.3487 / 0.3336 / 0.962 |
| 8 | 0.3788 / 0.3639 / 0.950 | 0.3444 / 0.3321 / 0.961 |
| 9 | 0.3764 / 0.3607 / 0.951 | 0.3409 / 0.3286 / 0.964 |
| 10 | 0.3745 / 0.3575 / 0.953 | 0.3377 / 0.3272 / 0.964 |
| 11 | 0.3731 / 0.3585 / 0.953 | 0.3347 / 0.3248 / 0.965 |
| 12 | 0.3726 / 0.3570 / 0.953 | 0.3324 / 0.3233 / 0.966 |
| 13 | n/a | 0.3301 / 0.3253 / 0.965 |
| 14 | n/a | 0.3285 / 0.3231 / 0.966 |
| 15 | n/a | 0.3268 / 0.3184 / 0.967 |
| 16 | n/a | 0.3256 / 0.3199 / 0.967 |
| 17 | n/a | 0.3243 / 0.3172 / 0.968 |
| 18 | n/a | 0.3235 / 0.3171 / 0.968 |
| 19 | n/a | 0.3229 / 0.3169 / 0.968 |
| 20 | n/a | 0.3223 / 0.3163 / 0.968 |

#### Full held-out evaluation

Drawing2CAD test is the natural, line-heavy distribution. Its loader retains
181,770 of 182,428 labels after excluding 658 incomplete circle targets.
SketchGraphs test is the complete 926,940-edge holdout and contains no
POLYLINE/BEZIER support, so its supported macro-F1 covers LINE/ARC/CIRCLE.

| checkpoint | Drawing2CAD accuracy | Drawing2CAD macro-F1 | SketchGraphs accuracy | SketchGraphs macro-F1 |
|---|---:|---:|---:|---:|
| conservative mixed | 0.9330 | 0.8275 | 0.9969 | 0.9912 |
| **adaptive mixed** | **0.9488** | **0.8603** | **0.9977** | **0.9936** |
| SketchGraphs specialist | n/a | n/a | **0.9998** | **0.9993** |

Per-class F1 for every candidate and evaluation domain:

| checkpoint / split | line | arc | circle | polyline | bezier |
|---|---:|---:|---:|---:|---:|
| conservative / mixed validation | 0.9880 | 0.9223 | 0.9920 | 0.9415 | 0.9203 |
| **adaptive / mixed validation** | **0.9902** | **0.9499** | **0.9956** | **0.9566** | **0.9491** |
| conservative / Drawing2CAD test | 0.9652 | 0.6099 | 0.9754 | 0.6828 | 0.9043 |
| **adaptive / Drawing2CAD test** | **0.9728** | **0.6598** | **0.9870** | **0.7468** | **0.9351** |
| conservative / SketchGraphs test | 0.9997 | 0.9742 | 0.9997 | n/a | n/a |
| **adaptive / SketchGraphs test** | **0.9998** | **0.9810** | **0.9999** | n/a | n/a |
| SketchGraphs specialist / SketchGraphs test | **0.9999** | **0.9981** | **1.0000** | n/a | n/a |

Selected adaptive precision/recall/F1 details:

| split / class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| mixed validation / line | 0.9955 | 0.9850 | 0.9902 | 10,049 |
| mixed validation / arc | 0.9354 | 0.9648 | 0.9499 | 10,049 |
| mixed validation / circle | 0.9934 | 0.9978 | 0.9956 | 10,049 |
| mixed validation / polyline | 0.9589 | 0.9543 | 0.9566 | 10,049 |
| mixed validation / bezier | 0.9604 | 0.9380 | 0.9491 | 8,800 |
| Drawing2CAD test / line | 0.9998 | 0.9473 | 0.9728 | 155,268 |
| Drawing2CAD test / arc | 0.5017 | 0.9633 | 0.6598 | 2,396 |
| Drawing2CAD test / circle | 0.9865 | 0.9874 | 0.9870 | 5,096 |
| Drawing2CAD test / polyline | 0.6110 | 0.9601 | 0.7468 | 10,412 |
| Drawing2CAD test / bezier | 0.9356 | 0.9345 | 0.9351 | 8,598 |
| SketchGraphs test / line | 1.0000 | 0.9995 | 0.9998 | 708,356 |
| SketchGraphs test / arc | 0.9984 | 0.9641 | 0.9810 | 50,303 |
| SketchGraphs test / circle | 0.9999 | 0.9998 | 0.9999 | 168,281 |

Selected adaptive Drawing2CAD test F1 is LINE 0.9728, ARC 0.6598, CIRCLE
0.9870, POLYLINE 0.7468, and BEZIER 0.9351. The main residual error is low
precision for rare ARC (0.5017) and POLYLINE (0.6110): 1,647/6,234 true line
edges are predicted as ARC/POLYLINE. This is visible in the natural test even
though recall is 0.9633/0.9601. The inference wrapper therefore also applies a
raw-pixel straight-line fast path gated by both 1.5-pixel p90 residual and 1%
relative p90 residual; visibly curved edges continue to the learned model.

Selected checkpoint:

```text
models/free2cad_sketchgraphs_d2c_mixed.pth
size 9,642,557 bytes
SHA-256 dd09a1d54bda5bf178d80cf56f7200723ac132e8a532ede18a5e7c731659db34
```

Warm-start checkpoint:

```text
models/free2cad_sketchgraphs_full.pth
size 9,640,160 bytes
SHA-256 bed27ed08ad794757bb93192c9347caf0af9d18f6f62a15a39537f0fc4d97d3f
```

Selected checkpoint metadata is `version=3`, `architecture=encoder_only`,
zero-based epoch 19 (human epoch 20), five command rows, 64 input points,
128 hidden dimensions, eight attention heads, four encoder layers, dropout
0.1, three-point arc encoding, label smoothing 0.05, and parameter-loss weight
0.5. The saved class weights are `[1, 1, 1, 1, 1]`.

Reproduction environment:

| component | version |
|---|---|
| Python | 3.11.8 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| GPUs | 2 x NVIDIA GeForce RTX 4090, 24,564 MiB each |
| NVIDIA driver | 580.159.03 |

Implementation inventory:

| file | responsibility |
|---|---|
| `tools/d2c_stage3_dataset.py` | Drawing2CAD SVG-to-Stage-3 supervision, topology matching, circularity guard, loop ordering, atomic restartable shards |
| `tools/build_free2cad_mixed_dataset.py` | trainer-visible supply scan, bounded class/domain allocation, global shuffle, mixed manifests |
| `stage3_primitivesfitting/research/train_free2cad_v3.py` | five-class head, four-class warm start, dynamic vocabulary, streaming training/evaluation |
| `tools/evaluate_free2cad_v3.py` | vocabulary-aware checkpoint evaluation and JSON reports |
| `stage3_primitivesfitting/research/stage3_primitive_fit_free2cad.py` | five-class loading/decoding, deterministic polyline/Bezier geometry, loop ordering, straight-line guard |
| `tests/test_d2c_stage3_dataset.py` | conversion and label-guard coverage |
| `tests/test_free2cad_mixed_dataset.py` | supply allocation and trainer-visibility coverage |
| `tests/test_free2cad_arc_encoding.py` | warm-start, five-class decoding, circle/arc cleaning, line fast-path coverage |

Evaluation artifacts:

```text
output/Free2CADMixedSketchGraphsD2C/train/manifest.json
output/Free2CADMixedSketchGraphsD2C/val/manifest.json
output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/train.log
output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/free2cad_v3_best_f1.pth
output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/evaluation_mixed_val.json
output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/evaluation_d2c_test.json
output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/evaluation_sketchgraphs_test.json
output/Free2CADMixedSketchGraphsD2C/checkpoints/conservative/train.log
output/Free2CADMixedSketchGraphsD2C/checkpoints/conservative/free2cad_v3_best_f1.pth
output/Free2CADMixedSketchGraphsD2C/checkpoints/conservative/evaluation_d2c_test.json
output/Free2CADMixedSketchGraphsD2C/checkpoints/conservative/evaluation_sketchgraphs_test.json
output/Drawing2CAD/stage3/train/manifest.json
output/Drawing2CAD/stage3/val/manifest.json
output/Drawing2CAD/stage3/test/manifest.json
output/Drawing2CAD/stage3_ordered_pilot/val/class_audit.png
```

Exact selected-checkpoint evaluation commands:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/evaluate_free2cad_v3.py \
  --data-dir output/Free2CADMixedSketchGraphsD2C \
  --checkpoint models/free2cad_sketchgraphs_d2c_mixed.pth \
  --split val --batch-size 1024 --device cuda \
  --output output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/evaluation_mixed_val.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/evaluate_free2cad_v3.py \
  --data-dir output/Drawing2CAD/stage3 \
  --checkpoint models/free2cad_sketchgraphs_d2c_mixed.pth \
  --split test --batch-size 1024 --device cuda \
  --output output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/evaluation_d2c_test.json

CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/evaluate_free2cad_v3.py \
  --data-dir output/SketchGraphsStage3Full/stage3 \
  --checkpoint models/free2cad_sketchgraphs_d2c_mixed.pth \
  --split test --batch-size 1024 --device cuda \
  --output output/Free2CADMixedSketchGraphsD2C/checkpoints/adaptive/evaluation_sketchgraphs_test.json
```

Verification completed on 2026-07-19: `55 passed` from `.venv/bin/pytest -q`,
checkpoint loading and five-class inference succeeded on CPU, dense straight
edges exercised the geometric line path, curved edges continued to Free2CAD,
and `git diff --check` reported no whitespace errors.

This checkpoint remains a research deployment candidate. The next gate before
changing the production default is a paired Drawing2CAD end-to-end comparison
and filtered-PatentData visual regression against the RANSAC fitter.

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

The pilot Stage 2 recommendation was `models/puhachov_sketchgraphs_rehearsal70.pth`;
the full evaluation above supersedes it with
`models/puhachov_sketchgraphs_full_phaseA.pth`. The Stage 3 pilot
`models/free2cad_sketchgraphs.pth` is superseded for SketchGraphs-domain use by
`models/free2cad_sketchgraphs_full.pth`. Both model families are ignored by Git;
their evaluation JSON reports are under `output/SketchGraphsTraining*`,
`output/SketchGraphsFull`, and `output/SketchGraphsStage3Full`.

The Stage 3 checkpoint loads and decodes through the research
`Free2CADFitter`, but the main production Stage 3 entry point still defaults to
the guarded deterministic/RANSAC fitter. Do not switch production configuration
until PatentData integration renders have been compared visually.

SketchGraphs code is MIT licensed. The sketch data is different: the official
release states that the original sketch creators retain copyright and points to
the Onshape terms. Do not publish or commit the downloaded/derived corpus.

**Dataset identity correction (2026-07-19):** the local Drawing2CAD corpus is
the CAD-VGDrawing release. Full mixed Stage 2 Phase A therefore already used it
at 30%; Phase B above increases its image share to 40%. Stage 3 remains on the
production RANSAC fitter, so no further Free2CAD corpus or training work is
planned.

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

---

## 12. Executed hatch-stroke adaptation (2026-08-07)

This section records the current repository implementation; it supersedes the
older external-root examples above for the hachure-specific Stage 2 work.

### Synthetic model

- Dataset: `output/PatentVecHatchStroke10k` (`8,976` train, `1,024` validation).
- Selected checkpoint: `models/hatch_stroke_multilabel.pth`, epoch 30.
- SHA-256:
  `8a9b8e1d3f6d1e4dc983d45b597eb9614771aec1d195d1d310350a51526542a6`.
- Synthetic safe-removal calibration: hatch `>= 0.50`, structural `< 0.03`,
  precision `0.999507`, recall `0.923380`, structural error `0.000553`.
- Deployment decision: rejected as a synthetic-only model. Destructive
  pre-topology and replacement inference both regress real PatentData topology;
  additive inference is inert.

### Accepted non-learning fix

`stage2.hachure_region_cleanup_before_metrics: true` preserves region residue in
the hachure side layer and reconnects the structural graph before metrics. On 97
paired PatentData figures it improves 3, ties 94, and regresses 0. Stage 2 F1 is
`0.942526 -> 0.942967`; symmetric Chamfer is `1.585188 -> 1.580809`.

Evidence:

```text
output/PatentVecHatchStrokeTraining/patentdata100_residue_fix_intrinsic.json
output/PatentVecHatchStrokeTraining/patentdata100_residue_fix_fidelity.json
output/PatentVecHatchStrokeTraining/residue_fix20_audit.png
output/PatentVecHatchStrokeTraining/residue_stage3/
```

### Reviewed-real adaptation

The reviewed `output/PatentData/hatch_gt` set contains 268 subdrawing files,
grouped into 212 figures (132 positive and 80 negative). Exact patent-disjoint
worklists replayed all 212 figures through the promoted Stage 0-2 teacher path:

```text
output/PatentVecHatchStrokeReal/reviewed_figures.csv
output/PatentVecHatchStrokeReal/reviewed_figures_part0.csv
output/PatentVecHatchStrokeReal/reviewed_figures_part1.csv
```

The teacher completed 203 figures: 3 stopped at the Stage 1 quality gate and 6
at the Stage 2 gate, with no runtime failures. Full no-hachure graph recovery
was rejected because it added only 87 selected pixels across 8 figures and the
dominant delta was a false structural contour. Final labels therefore require
exact reviewed-region and teacher-Hough consensus.

`tools.export_real_hatch_stroke_dataset` exported 170 train and 33 validation
figures to `output/PatentVecHatchStrokeReal`. The split unit is `patent_id`.
Structural and hatch channels have independent supervision masks; polygon
boundaries, unexplained region ink, and unsupported crossings do not contribute
loss. Corpus totals are:

| Quantity | Pixels / figures |
|----------|-----------------:|
| Input skeleton | 8,421,818 |
| Structural target | 8,246,044 |
| Trusted hatch target | 199,789 |
| Hatch-only / overlap | 175,774 / 24,015 |
| Jointly supervised | 5,661,723 |
| Unsupervised skeleton | 1,419,036 |
| Positive figures with trusted hatch | 89 |

### First mixed-domain training result

Two full 30-epoch arms used the same synthetic checkpoint, 8,976 patches per
epoch, all 1,024 synthetic validation samples, all 33 reviewed-real validation
figures, and exact 35% or 50% real-domain supply.

| Arm | Selected epoch | Synthetic structural recall | Real structural recall | Real hatch F1 | Decision |
|-----|---------------:|----------------------------:|-----------------------:|--------------:|----------|
| real35 | 1 | 0.996378 | 0.994210 | 0.856876 | reject: misses 0.995 real floor |
| real50 | 24 | 0.996332 | 0.991928 | 0.927022 | reject: stronger hatch, weaker preservation |

Checkpoint SHA-256 values:

```text
real35 best  6b7f5a38ddd9546567a1fd71f0a028207e0b4ed8af9334e663e0270d39a8d126
real35 last  e9ad8461da3d78e87a512ee8df4842474c746a856cc126199710e56b9a1611ee
real50 best  43ab630141d19db07f1f6a03eed50b6ac289b2b3dcde0b3824f7d2258e1953ab
real50 last  bb5eaa591d1448c5027a5f045e1b0027ac218255305bcf369949202f9e461c45
```

Full threshold sweeps show a domain-specific structural-score conflict. The
synthetic safety gate selects structural probability `< 0.03` with safe-removal
F1 `0.951-0.959`; at that same veto threshold, real safe-removal recall is only
`0.00014-0.00077`. Raising the real veto to `0.50` recovers up to `0.4603`
recall and `0.6270` F1, but those checkpoints fail the real structural-recall
floor. No single policy is eligible on synthetic and real validation.

Synthetic subgroup preservation is baseline-relative plus an aggregate 0.995
floor. The untouched checkpoint starts at hard `0.996882`, medium `0.998691`,
and very-hard `0.993482`; every mixed checkpoint preserves or improves those
three subgroup recalls. `tools.select_hatch_stroke_policy` requires one exact
threshold pair to pass every dataset rather than choosing per-domain policies.

A controlled second round started from the real35 preservation-best state. Both
arms used 50% reviewed-real supply, learning rate `1e-5`, structural channel
weight 8, and structural-negative weights 4 or 8. Both were stopped after epoch
1 because they moved in the wrong direction:

| Negative weight | Synthetic structural recall | Real structural recall | Real safe F1 | Decision |
|----------------:|----------------------------:|-----------------------:|-------------:|----------|
| 4 | 0.99196 | 0.96223 | 0.00088 | stop |
| 8 | 0.98794 | 0.93579 | 0.00110 | stop |

Increasing negative loss destroys structural preservation before moving enough
real hatch pixels below the veto. No further hatch-stroke training is promoted.

### Disjoint PatentData gate and scaling fixes

`output/PatentVecHatchStrokeTraining/patentdata100_disjoint_worklist.csv`
contains 100 figures from 100 patents with zero reviewed-training patent
overlap. A fresh current-code end-to-end control has 73 `ok`, 3 Stage 1 gates,
6 Stage 2 gates, and 18 Stage 3 gates. The earlier 65/26 split was generated
before the bounded Stage 3 traversal and raw closed-curve fitting fixes.

One figure exposed a 335,922-pixel unclaimed branched residual. Stage 2 now uses
exact OpenCV connected-component labeling, marks only true simple cycles as
closed, and quality-gates noncycle residuals above 100,000 pixels. Stage 3 loop
ordering is linear for simple cycles and bounded for malformed networks. The
real residual benchmark completes Stage 2 component labeling in 2.255 seconds
and bounded Stage 3 ordering in 4.106 seconds instead of entering quadratic
work. The exact 20-case high-hachure screen is fixed at:

```text
output/PatentVecHatchStrokeTraining/patentdata20_disjoint_high_hachure_worklist.csv
```

### Final guarded integration result

The first additive experiment unioned the region and stroke masks before edge
classification. That erased source attribution and allowed EP2976579 structural
chains up to 3,891 pixels to bypass the 80-pixel hatch limit. Stage 2 now keeps
the established region pass unchanged, subtracts its mask from the stroke-model
mask, and applies the model-only evidence in a second pass requiring short,
straight, repeated hatch geometry.

A 2x2 screen separated model evidence from
`hachure_region_cleanup_before_metrics`. Region cleanup alone reproduced all
three improvements; guarded model-only inference was an exact 20/20 tie. The
fresh full end-to-end comparison then replayed 100 patent-disjoint figures under
one current code revision:

| Result | Control | real50 guarded additive |
|--------|--------:|------------------------:|
| OK / Stage 1 / Stage 2 / Stage 3 gates | 73 / 3 / 6 / 18 | 73 / 3 / 6 / 18 |
| Stage 2 metric ties | 97/97 | 97/97 |
| Stage 3 metric ties | 91/91 | 91/91 |
| Stage 2/3 raster-fidelity ties | all | all |
| Model-exclusive mask figures / pixels | - | 8 / 19,665 |
| Eligible additive edges | - | 0 |

EP2976579 contributes 18,794 of the 19,665 exclusive pixels and none pass the
geometric guard. The model is therefore rejected for deployment as safely inert,
not promoted as an improvement. The reviewed region detector plus early residue
preservation remains production.

Evidence:

```text
output/PatentVecHatchStrokeTraining/hatchscreen20_factorial/
output/PatentVecHatchStrokeTraining/hatchstroke_real50_full100_e2e/
output/PatentData100_HatchStrokeE2E_ControlCurrent/
output/PatentData100_HatchStrokeE2E_Real50AdditiveGuardedCurrent/
```

### Stage 3 confidence policy: completed and promoted (2026-08-14)

The planned Stage 3 work is complete. Weak open line/arc fits may now be
replaced by the existing corner-split compound fitter, but only when the result
passes all of these guards:

- confidence at least 0.60 and gain at least 0.05 over the weak primitive;
- at most 20,000 source points and 64 path atoms;
- every fitted segment has p95 residual at most 2 px;
- path endpoint error is at most 3 px;
- cubic Bezier handles remain inside a source-relative safety envelope.

Weak polygon/raw closed fallbacks now compete with adaptive 24, 32, 48, 64,
96, and 128 vertex closed paths. The selected path must reach confidence 0.65,
improve confidence by at least 0.05, and keep p95 residual at or below 2 px. If
no candidate passes, the exact raw trace is preserved. Quality gates were not
relaxed.

Stage 4 evaluation also exposed two independent exporter errors. Raster-index
coordinates are now translated to pixel centres (`+0.5 px`), and mixed path
segments are globally oriented with a two-state dynamic program. This prevents
first-arc chord jumps and makes SVG/DXF path continuity deterministic. The
Stage3/4 replay tools freeze archived Stage 2 graphs so policy comparisons are
not contaminated by later topology changes:

```text
tools/replay_d2c_stage3_stage4.py
tools/replay_patent_stage3_stage4.py
tools/analyze_d2c_stage3_policy.py
```

#### Full Drawing2CAD evaluation

The selected p95=2 policy replayed all 31,524 views with zero errors. Relative
to the pixel-centre-corrected control, all 10,000-resample drawing-cluster
bootstrap intervals exclude zero in the favorable direction:

| Metric | Control | Selected p95=2 | Mean delta |
|--------|--------:|----------------:|-----------:|
| Symmetric Chamfer | 0.355585 | **0.345473** | -0.010112 |
| Symmetric Chamfer p95 | 1.279391 | **1.230501** | -0.048890 |
| Pixel IoU | 0.783634 | **0.785794** | +0.002160 |
| Skeleton IoU | 0.832186 | **0.835159** | +0.002973 |
| Pixel precision | 0.811151 | **0.812492** | +0.001341 |
| Pixel recall | 0.954054 | **0.955553** | +0.001499 |

Activation is sparse and bounded: 4,622 weak open promotions use 21,206 atoms;
1,365 closed simplifications compress 969,138 source points to 36,213 vertices
(26.76x). No candidate has a mean-Chamfer regression above 1 px. Visual tail
review showed that several apparent mean-skeleton-Chamfer regressions actually
recover both sides of narrow outlined geometry; pixel IoU and p95 distance must
therefore accompany mean Chamfer in policy decisions.

The stricter endpoint=2 px ablation was rejected because it significantly
regressed all six metrics against endpoint=3 px. A closed p95=0.5 ablation
improves Drawing2CAD mean Chamfer by 0.001333 and skeleton IoU by 0.000638 over
p95=2, but p95=2 improves p95 distance by 0.004542, pixel IoU by 0.000209, and
precision by 0.000300; recall is tied. This is a real metric tradeoff rather
than a catastrophic tail.

#### Frozen filtered-PatentData evaluation

A normal rerun was discarded for Stage 3 attribution because current Stage 2
regenerated different graphs. The exact replay instead froze all Stage 0-2
artifacts from `PatentData30_Stage3OpenClosedGuarded` and reran only Stages 3-4.
Both candidates completed all 30 figures with zero replay errors.

| Result | Rejected p95=0.5 | Selected fixed p95=2 |
|--------|-----------------:|---------------------:|
| Stage 3 pass / quality gate | 22 / 8 | **30 / 0** |
| Stage 3 mean confidence | 0.753383 | **0.821098** |
| Low-confidence ratio | 0.196056 | **0.035456** |
| Chamfer p95 | 9.815303 | **9.695640** |
| Precision at 2 px | 0.944744 | **0.949599** |
| Recall at 2 px | 0.819822 | **0.827840** |
| F1 at 2 px | 0.871916 | **0.878894** |

Direct p95=2 minus p95=0.5 paired deltas are `-0.119663` Chamfer p95
(`95% CI [-0.212516, -0.041471]`), `+0.004855` precision
(`[0.002728, 0.007350]`), `+0.008018` recall
(`[0.004843, 0.011947]`), and `+0.006978` F1
(`[0.004428, 0.009912]`). Mean Chamfer is tied. The p95=2 policy is therefore
promoted in `config.yaml`, `config_deploy.yaml`, and `config_patentvec_A.yaml`;
p95=0.5 remains an explicit clean-CAD ablation only.

#### Broader 100-patent disjoint confirmation

`tools/replay_patent_stage3_stage4.py` now replays only rows that had actually
reached Stage 3. It preserves upstream terminal rows instead of incorrectly
advancing a Stage 1/2 failure through later stages. On the frozen current-code
100-patent control, 91 rows were eligible, 91 completed, and zero replay errors
occurred. The final status transitions are:

| Source status | Selected status | Count |
|---------------|-----------------|------:|
| `ok` | `ok` | 73 |
| `quality_gate_stage3` | `ok` | 18 |
| `quality_gate_stage1` | preserved | 3 |
| `quality_gate_stage2` | preserved | 6 |

All 91 Stage 3 rows improve mean confidence, from `0.764635` to `0.843326` on
average; low-confidence ratio falls from `0.188156` to `0.019506`. The paired
full-resolution raster result is:

| Metric | Control | Selected p95=2 | Mean delta, 95% CI |
|--------|--------:|----------------:|-------------------:|
| Chamfer | 3.616210 | 3.624344 | +0.008133 [-0.004633, 0.021503] |
| Chamfer p95 | 15.706233 | **15.574026** | -0.132207 [-0.222991, -0.056177] |
| Precision at 2 px | 0.907183 | **0.912172** | +0.004989 [0.003708, 0.006372] |
| Recall at 2 px | 0.852937 | **0.858209** | +0.005273 [0.003580, 0.007089] |
| F1 at 2 px | 0.873256 | **0.878608** | +0.005352 [0.003953, 0.006869] |

The 73 previously passing rows independently improve F1 by `0.004064`
(`[0.002738, 0.005531]`). The 18 recovered gates improve by `0.010574`
(`[0.007044, 0.014792]`). The worst individual F1 regression is `-0.009132`;
the ten-case side-by-side tail audit shows no structural deletion or connector
jump. Eight sampled recovered gates produce complete, usable SVGs.

The audit also reconfirms that filtering is not closed: EP3184100B1/F0004 is a
bar chart and EP3191598B1/F0001 is dense chemistry, but both were accepted by
the clean12 `long_engineering_lines` rule. They are valid Stage 3 stress tests
but must be rejected before creating training targets.

Evidence:

```text
output/Drawing2CAD/stage3_guarded_p2_fixed_full/
output/PatentData30_Stage3GuardedP2FixedFrozen/
output/PatentVecHatchStrokeTraining/stage3_guarded_p2_fixed_full_analysis.json
output/PatentVecHatchStrokeTraining/stage3_guarded_p2_fixed_vs_p05_analysis.json
output/PatentVecHatchStrokeTraining/stage3_guarded_final30/intrinsic_threeway.json
output/PatentVecHatchStrokeTraining/stage3_guarded_final30/fidelity_threeway.json
output/PatentVecHatchStrokeTraining/stage3_guarded_final30/fidelity_final_p05_vs_fixed_p2.json
output/PatentData100_Stage3GuardedP2FixedFrozen/
output/PatentVecHatchStrokeTraining/stage3_guarded_final100/intrinsic.json
output/PatentVecHatchStrokeTraining/stage3_guarded_final100/fidelity.json
output/PatentVecHatchStrokeTraining/stage3_guarded_final100/fidelity_by_source_status.json
output/PatentVecHatchStrokeTraining/stage3_guarded_final100/worst_f1_1.png
output/PatentVecHatchStrokeTraining/stage3_guarded_final100/worst_f1_2.png
output/PatentVecHatchStrokeTraining/stage3_guarded_final100/recovered_stage3_first8.png
```

#### Remaining Stage 3 priority

Accuracy promotion is complete, but dense-patent runtime remains poor. The
100-patent replay averages 73.6 seconds over 91 eligible rows, with a 27.8
second median, 343.8 second p95, and 486.1 second maximum. Profiling should next
isolate repeated open-edge compound fits, add safe memoization/early rejection,
and introduce per-edge runtime telemetry and budgets that fall back to the
already valid weak primitive without deleting geometry. In parallel, the
PatentData content filter must add a chart/chemistry veto after the permissive
`long_engineering_lines` decision. Fidelity and the existing quality gate
remain the release criteria.
