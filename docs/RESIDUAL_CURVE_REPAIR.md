# Non-Cycle Residual and Closed-Curve Repair

Implemented 2026-09-11 in canonical Stage 2 and the RANSAC Stage 3 fitter;
paired evaluations finalized 2026-09-12.
No model training, weight promotion, quality-threshold relaxation, or acceptance
policy changes are involved. Historical outputs are untouched.

## Root Causes

1. Stage 2 classified unclaimed components as non-simple cycles, but still
   represented them as closed edges. A partial arc or a branched hatch/outline
   network could therefore enter the closed-shape fitting cascade.
2. Radial inlier confidence does not establish support for a complete circle
   or ellipse. The confirmed patent examples cover only about 31 and 28 degrees
   of their fitted circles.
3. The ellipse solver was poorly conditioned, and ellipse confidence used an
   incorrect rotation sign. Positive tests on full rotated ellipses exposed
   both problems.
4. Opening the residuals exposed another failure: smoothed or inlier-trimmed
   line/arc candidates could omit much of a recovered stroke. Preserving graph
   pixels alone was insufficient to preserve the fitted geometry.

## Implementation

### Stage 2

`repair_noncycle_residuals()` decomposes residual pixel graphs into ordered
endpoint/junction chains and actual cycles. The digital adjacency excludes
redundant diagonal chords where an orthogonal bridge exists. Each retained
local adjacency is traced once; the source pixel set is preserved exactly.

Recovered edges carry `residual_parent_edge_ids` and
`topology_origin: recovered_residual`. Straight-through and degree-two merging
can reconstruct continuous strokes. Spur pruning, junction collapse, and tiny
loop filtering cannot silently discard these protected residual traces.
Hatching remains eligible for separation into the existing preserved side layer.

Components above 100,000 pixels remain bounded, unrepaired evidence for the
existing non-cycle quality gate. The new code does not bypass that gate.

### Stage 3

- Circle and ellipse candidates must pass the existing independent source
  validator, including bidirectional support and angular coverage. Open-edge
  ellipse candidates are guarded too.
- Closed polygon fallback is checked as well, preventing an unsupported circle
  from simply becoming an unsupported polygon.
- Ellipse fitting uses centered/scaled inputs with OpenCV `fitEllipseDirect`;
  axis ordering and rotation are converted to the existing primitive schema.
- All recovered-edge candidates are checked against raw source pixels. When
  needed, line/arc fits are retried on raw points. A checked compact polyline,
  then the exact ordered trace, are the final alternatives. Trace fallback
  retains low confidence (0.3), not an artificial high-confidence score.
- Legacy non-cycle bags passed directly to Stage 3 receive a bounded adjacency
  trace, never automatic closed-curve promotion. Full topology repair belongs
  in Stage 2; legacy evidence is still reviewable by acceptance.
- Regions containing recovered hatch sources use explicit source strokes,
  not an inferred filled hull. The fresh full-chain run exposed a severe
  unsupported-fill regression, so this guard is part of the final change.
  Its decisions are recorded as `hachure_coverage.recovered_regions_traced`.

The source-check policy is unchanged: Stage 2 coordinate units, two-pixel
support tolerance, at least 95% support for a pass, bounded unsupported runs,
endpoint checks, and complete-curve angular coverage. This is a necessary
source-fidelity condition, not a proof of intended CAD semantics.

## Verification

The complete suite passes **347 tests**. The 27 new repair tests cover actual
patent phantom-circle traces, crosses, branches, staircases, loops with branches,
grids, pixel/adjacency preservation, idempotence, long-stroke continuity, partial
curve rejection, valid rotated ellipses, and raw-source fallback after damaged
smoothing, and recovered hatch-region fallback. The existing branched-residual
topology test now expects open chains.

See [JUnit evidence](audits/2026-09-11/residual_curve_tests.xml) and the
[confirmed phantom-circle comparison](audits/2026-09-11/phantom_circle_repair.png).

### Completed Paired Results

The [machine-readable evidence](audits/2026-09-11/residual_curve_evidence.json)
records aggregates, regressions, source/configuration/code hashes, and the final
hatch-guard parity check. All runs completed without process errors.

| Measurement | Before | Repaired V2 |
|---|---:|---:|
| Patent91 mean raster F1 at 2 px | 0.881590 | 0.915722 |
| Patent91 mean precision | 0.913289 | 0.978976 |
| Patent91 mean recall | 0.863338 | 0.866828 |
| Patent91 mean symmetric Chamfer, px | 3.569534 | 1.764149 |
| Patent91 failing primitive checks | 2,146 | 513 |
| Patent91 full-circle/ellipse pass / review / fail | 18 / 202 / 998 | 3,491 / 0 / 0 |
| Patent91 drawing geometry pass / review / fail | 0 / 0 / 91 | 0 / 1 / 90 |
| CAD250 mean raster F1 at 2 px | 0.984643 | 0.985261 |
| CAD250 mean symmetric Chamfer, px | 0.332285 | 0.329450 |
| CAD250 drawing geometry pass / review / fail | 211 / 14 / 25 | 212 / 14 / 24 |

The 91 patents are the historically exported subset of the prior 100-patent
batch, not a newly drawn test set. The other nine gated rows were not replayed.
All 3,799,372 main source-pixel occurrences across those drawings were retained;
historical hatch primitives are unchanged in every replay. All bounded
non-cycle residuals were repaired. The two confirmed phantom circles now fit
supported arcs; all 52 existing CAD circles remain circles, with 14 additional
source-supported ellipses recovered.

There are important counter-results:

- Patent F1 improves in 83 drawings and regresses in eight; recall improves in
  67 and regresses in 24. Worst F1 delta is -0.010096 on
  `EP3129274B1/F0001`. Source-supported output alone does not restore source ink
  already missing before fitting.
- CAD F1 improves in 11 views, ties in 238, and regresses by 0.000725 in
  `0002_00028951_FrontTopRight`. Its source-geometry outcome nevertheless changes
  from review to pass. Different quality measures are not interchangeable.
- Frozen Patent91 primitive count rises from **15,970 to 611,969**, with 51
  drawings exceeding the unchanged 900-primitive cap. These traces expose
  genuine dense networks and hatch connections formerly hidden in pixel bags.
  This is not an acceptable compact CAD representation. Per-primitive pass
  percentages would be misleading after such a large change in denominator.
- No patent reaches a full geometry pass. Remaining failures involve other
  source traces, unsupported line/arc/path fits, endpoint/path gaps, and raster
  coverage. Historical native hatch-pattern parity also remains unverified.

The first iteration (`output/PatentData100_ResidualRepair`) raised mean F1 to
0.913688 but left 1,074 failing primitives and regressed recall in 32 drawings.
V2 adds the recovered-stroke raw-source fallback, reducing those failures to
513. It does not claim to solve unsupported fitting on unrelated edges.

### Fresh Chain and Hatch Regression

`output/PatentData2_ResidualRepairSmoke` reran all stages with canonical models
and settings on `EP2565056B1/F0001` and `EP3082506B1/F0001`. Stage 0/1 evidence
remains unchanged. Hatch separation reduced the first drawing to 209 main
edges, but inferred region fills then covered unsupported white space. Its
raster F1 fell to **0.268953**, despite all its circles passing source checks.
The earlier [full-chain metrics](audits/2026-09-11/fresh_smoke_fidelity.json)
are retained as regression evidence, not the final result.

The final guard keeps explicit strokes for recovered hatch regions:

| Drawing | Original F1 | Unguarded fresh F1 | Final guarded F1 | Final primitives |
|---|---:|---:|---:|---:|
| EP2565056B1/F0001 | 0.904888 | 0.268953 | 0.995728 | 1,557 |
| EP3082506B1/F0001 | 0.976320 | 0.991199 | 0.994482 | 388 |

See the [rendered comparison](audits/2026-09-11/recovered_hatch_guard_preview.png)
and [guard metrics](audits/2026-09-11/recovered_hatch_guard.json). All 1,348 and
196 side-layer source edges are represented. The final primitive sets have
three and eight failing main-geometry checks respectively; all seven full
circles/ellipses pass. No inferred native HATCH fills remain in these two outputs.

The final canonical paired rerun is
`output/PatentData2_ResidualRepairGuardedSmoke`, reusing the same Stage 0/1
evidence. It finishes without process crashes, but **one row stops at the
Stage 3 primitive-count gate** (1,557 > 900); the other exports 388 primitives.
Both acceptance records are rejected and the manifest builder keeps **zero**.
Quality gates were not loosened to make the dense drawing export.

Complete diagnostic SVG/DXF previews for both are under
`output/PatentData2_ResidualRepairHatchGuard`. They are deliberately separate
from canonical artifacts. Exact primitive, graph, and skeleton parity was
checked against the guarded canonical run, including its gated drawing.

The final hatch guard was added while frozen V2 workers were running. Its
output was subsequently checked on all **341** saved Patent91/CAD250 graphs:
hatch primitives are exactly unchanged, because those historical side layers
contain no newly recovered regions. The evidence retains the replay hashes
and final implementation hashes separately; fresh guarded checks use the final
code. No historical summary was rewritten to pretend it used a later revision.

### Reproducible Paired Replay

`tools/replay_residual_repair.py` reads frozen source graphs and writes new
graphs, primitives, SVG, DXF, geometry reports, and a paired `summary.json`.
It repairs/simplifies only residual subgraphs; unrelated source edges are kept.
It asserts exact main source-pixel preservation and unchanged hatch primitives.
It records source/configuration/implementation hashes.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.replay_residual_repair \
  --source output/PatentData100_Priority0Preservation \
  --output output/PatentData100_ResidualRepairV2 \
  --config output/PatentData100_Priority0Preservation/stage34_replay_config.yaml \
  --raster-metrics --workers 6

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.replay_residual_repair \
  --source output/Drawing2CAD/stage3_open_closed_full \
  --output output/Drawing2CAD250_ResidualRepairV2 \
  --config output/Drawing2CAD/stage3_open_closed_full/evaluation_config.yaml \
  --limit 250 --raster-metrics --workers 2
```

Use new output paths to repeat these commands. The CAD selection uses seed
850725, matching the preceding 250-view audit. Replays are diagnostic, not
canonical acceptance runs: they do not rerun hatch detection or enforce batch
quality gates, and inherited graph metrics describe the frozen graph rather
than the repaired edge count. `residual_repair` records the repair-specific data.

Raster comparisons render the real Stage 4 SVG without annotations, skeletonize
it, and compare against the shared Stage 1 skeleton at the original image size.
They are not original-CAD ground-truth accuracy measurements. Hatch primitives
are included in this raster check, although native DXF pattern parity remains
unverified by the per-primitive source validator.

## Remaining Work

- Follow-up implemented 2026-09-13: bounded explicit hatch bundles, ordinary
  raw-source fit guards, source-supported path connectors, and legacy hatch
  trace preservation. See [final paired results](COMPACT_HATCH_FIT_REPAIR.md).
  Patent91 now has zero failing primitive checks, but whole-image coverage,
  native patterns, and true stroke complexity remain unresolved. The canonical
  900-budget gate counts contained strokes, not merely bundled JSON objects.
- Repair Stage 2 source omissions and improve actual hatch/outline continuity;
  a fitting guard cannot recover ink absent from its graph source.
- Validate and improve hatch pattern phase, boundary, and spacing fidelity.
  Correct topology exposes more hatch strokes; full-chain checks are essential.
- Assess broader fresh canonical batches for edge-count and hatch-routing
  regressions. Frozen repair can greatly increase primitive counts because it
  deliberately retains hatch strokes in their historical main-graph ownership.
- Complete serialized DXF geometry checks and content routing. Neither clean
  execution nor improved raster F1 makes a drawing training-eligible.
