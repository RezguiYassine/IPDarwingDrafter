# PatentVec synthetic data

This directory implements the executable Track-A curriculum from
`PATENTVEC_SYNTHETIC_COMPOSITION_ROADMAP.md`.

## Current scope

- Track A: exact 2D interaction-aware compositions from SketchGraphs and
  CAD-VGDrawing/Drawing2CAD source vectors.
- M5 graph composition: medium, hard, and very-hard curricula spanning roughly
  5-12 connected source components. Samples use endpoint joins, analytically
  split T-junctions and connected crossings, containment, concentric/tangent
  placement, and cycle-forming bridges.
- M6 patent layers: clipped and cross hatching, dashed centre/hidden lines,
  leaders, arrowheads, reference numerals, dimensions, and text callouts.
- Separate visible and amodal graphs with source provenance, source intervals,
  transforms, semantic layers, and typed interactions.
- Compact restart-safe tar shards with checksums, source provenance, optional
  audit previews, Puhachov NPZ, and topology-projected Free2CAD NPZ targets.
- Native training export for both Stage 2 and Stage 3, with a BLAKE2 source
  index split that cannot alias the periodic curriculum.
- Difficulty gates cover graph connectivity/cycles, interaction variety,
  object complexity, local density, semantic-layer supply, geometric
  clearance, and unintended cross-component intersections.

Track-A composites are not CAD programs. Their Free2CAD artifact supervises
only the repository's per-edge primitive classifier/regressor. CAD operation
sequences must come from roadmap Track B and are not fabricated here.

## Validated 10k experiments

The frozen baseline and enhanced datasets use the same 10,000 source indices
and global seed:

```text
output/PatentVecComplexityA10k
output/PatentVecComplexityB10k
output/PatentVecComplexityAB.json
output/PatentVecComplexityAB.md
```

Baseline A is 50% medium and 50% hard. Enhanced B is 25% medium, 50% hard,
and 25% very hard. B passed every quality gate, projected all 10,000 Stage 3
targets through the real Stage 2 topology path, and passed an independent
500-drawing contract check for all 2,168 sampled polyline targets. The
stratified visual audit is at
`output/PatentVecComplexityB10k/audit_stratified/index.html`.

Compared with A, B increases mean source components by 19.5%, structural
junctions by 34.6%, exact Puhachov targets by 18.3%, and Free2CAD polyline
supply by 13.5%. The comparison uses conservative full-sample confidence
intervals plus paired intervals on the 6,449 indices whose accepted retry seed
is identical.

The native exports are:

```text
output/PatentVecComplexityA10kTrainingV2
output/PatentVecComplexityB10kTrainingV2
```

Each contains 8,976 train and 1,024 diagnostic validation drawings. The
validation split is deterministic but not source-disjoint because both sides
come from the source training pool; model selection therefore stays on fixed
real Drawing2CAD, SketchGraphs, and ArchCAD holdouts.

## Controlled training outcome

The equal-budget experiment uses seed `850725` and protects the real-data warm
start as a step-zero candidate. Stage 2 trains for three exact 59,840-sample
epochs. Each epoch contains 20,346 Drawing2CAD, 25,432 cached SketchGraphs,
5,086 ArchCAD, and all 8,976 synthetic labels exactly once.

The Stage 2 warm start scored `0.8193` mean macro-F1 over the three fixed real
holdouts. Final A and B reached `0.8477` and `0.8477`, respectively. A scored
`0.8510/0.9620/0.7301` on Drawing2CAD/SketchGraphs/ArchCAD; B scored
`0.8539/0.9617/0.7277`. On each complete 1,024-drawing synthetic validation
set, A improved over the warm start from `0.1746` to `0.5047`, while B improved
from `0.1650` to `0.4791`. The A/B real-domain difference is negligible; the
main validated gain is adding synthetic exposure without catastrophic
real-domain forgetting.

On complete cached test/validation splits, A/B score `0.856136/0.857240` on
31,524 Drawing2CAD views, `0.936599/0.936426` on 9,565 SketchGraphs drawings,
and `0.730050/0.727709` on all 1,159 ArchCAD drawings. Their equal-domain means
are `0.840928/0.840458`, a difference of only `0.00047`; PatentData remains the
deployment tie-breaker.

Stage 3 uses 914,250 trainer-visible labels per variant, exactly 182,850 per
class, with aligned real rows and a byte-identical 48,996-label validation set.
The real-data warm start remains selected at macro-F1 `0.9683`: final A and B
score `0.9622` and `0.9626`. The adapted states nevertheless improve synthetic
macro-F1 from `0.6001` to `0.7862` on A and from `0.6111` to `0.7793` on B;
polyline F1 rises from `0.1951` to `0.8419` and from `0.2367` to `0.8228`.
This is a diagnostic Pareto tradeoff, not a production Free2CAD promotion.

Artifacts and logs are under:

```text
output/PatentVecComplexityABTraining/stage2/guard4way
output/PatentVecComplexityABTraining/stage3
```

## Reference-free Stage 2 topology contract

The original archive labels are not suitable for dense patent-raster training:
they omit raster-induced endpoints and junctions, especially within hatching.
`reference-free-raster-topology-v1` rebuilds each Stage 2 skeleton from only
the object, hidden/centre-line, and hatch semantic masks. References, leaders,
dimensions, and text are absent before crossing-number labels are computed, so
the training input matches the post-Stage-0 inference contract.

The corrected immutable export is:

```text
output/PatentVecComplexityB10kTrainingReferenceFreeTopologyV1
```

It contains 8,976 train and 1,024 validation drawings with 1,153,393
endpoints, 3,513,144 junctions, and 99,532 corners. A separate 500-source audit
matched every endpoint and junction, including all 4,595 hatch endpoints and
167,766 hatch junctions, and found no skeleton pixels outside the declared
reference-free topology support. The complete validation set improved from
`0.705869` to `0.845934` after three exact rehearsal epochs. The selected
checkpoint and evidence are under:

```text
models/puhachov_patentvec_referencefree_topology.pth
output/PatentVecComplexityABTraining/stage2/reference_free_topology
output/PatentVecComplexityABTraining/stage2/topology_contract_C2_reference_free_500.json
```

The complete real-domain and paired PatentData deployment gates remain
authoritative; synthetic validation alone cannot release the 50k generation.

## Multilabel hachure-stroke experiment

Explicit Hough-guided junction routing was rejected on the exact paired
hachure-heavy PatentData control: even the restricted T/X-only variant reduced
Stage 2 recall/F1 and Stage 3 F1 relative to the safer intersection-preserving
cleanup. The next experiment therefore classifies existing Stage 1 skeleton
pixels instead of inserting graph nodes.

`reference-free-hatch-stroke-multilabel-v1` builds its input skeleton from
`object`, `hidden_center`, and `hatch` masks only. It emits independent
structural and hatch channels, so a true crossing is positive in both channels.
The channel union must equal the complete input skeleton; any unexplained pixel
falls back to structural preservation. Exact mask support is the default. On a
128-sample ablation it produced 60,626 true overlap pixels with zero unassigned
pixels, while a one-pixel dilation inflated overlap to 146,741 without adding
coverage.

The reviewed pilot is:

```text
output/PatentVecHatchStrokePilot128
```

It contains 114 train and 14 deterministic validation samples. All 128 contain
hatch-only, structural-only, and overlap supervision, totaling 651,209,
431,665, and 60,626 pixels respectively. The visual audit is
`output/PatentVecHatchStrokePilot128/audit_16.png`.

The two-output trainer consumes the binary skeleton rather than scan grayscale,
computes loss on ink pixels, upweights crossings and structural labels, and
hard-masks inference probabilities back to input ink. The full synthetic export
contains 8,976 train and 1,024 validation figures. Its selected 30-epoch model is
stored at `models/hatch_stroke_multilabel.pth`, but remains a warm start only:
synthetic calibration is excellent while destructive and replacement modes fail
the real PatentData topology gate.

Reviewed-real adaptation uses
`tools.export_real_hatch_stroke_dataset`. It groups the 268 reviewed mask files
into 212 figure-safe samples with patent-disjoint validation, writes exact batch
worklists, projects the current
main graph and `removed_hachures` into the Stage 1 skeleton frame, and stores a
two-channel `supervision_mask`. Reviewed polygons define valid areas; they are
not treated as exact hatch strokes. Unknown region ink and annotation boundaries
remain unsupervised. `tools.hatch_stroke_train --real-dataset ...
--real-fraction ...` then supplies a fixed real-domain share per epoch and ranks
checkpoints only after both synthetic and real structural-recall floors pass.

## Commands

```bash
.venv/bin/python -m syntheticData.build_source_index \
  --split train --output output/PatentVecSourceIndex/train.json

.venv/bin/python -m syntheticData.generate_dataset \
  --mode experiment --count 10000 --workers 24 --shard-size 100 \
  --canvas 1024 --audit-every 100 --max-sample-attempts 128 \
  --curriculum enhanced \
  --source-index output/PatentVecSourceIndex/train.json \
  --output output/PatentVecComplexityB10k --seed 850725

.venv/bin/python -m syntheticData.validate_polyline_contract \
  output/PatentVecComplexityB10k --sample-limit 500

.venv/bin/python -m syntheticData.analyze_dataset \
  output/PatentVecComplexityB10k

.venv/bin/python -m syntheticData.build_dataset_audit \
  output/PatentVecComplexityB10k \
  --output output/PatentVecComplexityB10k/audit_stratified \
  --stratified-per-difficulty 18 --per-sheet 8

.venv/bin/python -m syntheticData.export_training_dataset \
  --input output/PatentVecComplexityB10k \
  --output output/PatentVecComplexityB10kTrainingV2

.venv/bin/python -m syntheticData.export_hatch_stroke_dataset \
  --input output/PatentVecComplexityB10k \
  --output output/PatentVecHatchStrokePilot128 --limit 128

.venv/bin/python -m syntheticData.build_hatch_stroke_audit \
  output/PatentVecHatchStrokePilot128 \
  --output output/PatentVecHatchStrokePilot128/audit_16.png

.venv/bin/python -m tools.hatch_stroke_train \
  --dataset output/PatentVecHatchStroke10k \
  --real-dataset output/PatentVecHatchStrokeReal \
  --real-fraction 0.35 \
  --initial-checkpoint models/hatch_stroke_multilabel.pth \
  --difficulty-weights medium=0.2,hard=0.3,very_hard=0.5 \
  --out models/hatch_stroke_multilabel_real35.pth
```

Each compact sample contains these core members:

```text
sample.json
quality.json
masks.npz
puhachov.npz
free2cad_edges.npz
```

Audit-selected rows additionally store the rendered artifacts and preview.
Other rows can be re-rendered deterministically from `sample.json`.

## Full-generation storage

The planned full output root is:

```text
/media/safe/secondary disk/IPdrawings
```

No full generation has been started and this external root has not been
written. Measured enhanced throughput projects 50,000 drawings at about 4.9
hours and 15.8 GiB with 24 workers and one stored audit preview per 100
drawings. Scale-up remains gated on the controlled model evaluation, filtered
PatentData regression, and source-license review.

## Tests

```bash
.venv/bin/pytest -q
```

The current full repository suite passes 136 tests.
