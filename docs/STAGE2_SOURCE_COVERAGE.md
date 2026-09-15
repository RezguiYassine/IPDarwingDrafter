# Stage 2 Source Coverage

First implementation and evaluation: 2026-09-14. No training, checkpoint
promotion, new dataset generation, or relaxation of acceptance thresholds.

Follow-up implemented: [recovered connection integration](STAGE2_CONNECTION_INTEGRATION.md).
The results below document the initial, separate-trace recovery baseline;
the follow-up preserves those source pixels while reducing fragmentation.

## Implementation

### Coverage Accounting

Stage 2 retains an immutable working skeleton before hatch subtraction and
records coverage after topology prepasses/extraction, graph simplification,
hatch separation, residual pruning, deduplication, region cleanup, smoothing,
floating-noise pruning, dashed grouping, and final recovery.

Every graph now contains `coverage` with schema `ap3-stage2-coverage-v1`:

- Working/source coordinate frames and a packed source-mask SHA-256.
- Per-operation represented, missing, newly missing, and restored pixel counts.
- Exact newly missing coordinates as row spans `[y, first_x, exclusive_end_x]`.
- Recovered component bounding boxes, sizes, edge IDs, and decisions.
- Explicit unresolved spans and separate represented/accounted fractions.

Accounting counts actual main and hatch source pixels. A nearby node, inferred
hatch boundary, or pruned-noise label cannot claim rendered coverage. Stage 2
isolation no longer removes pruned noise from its denominator; only preserved
hatch pixels leave the main-geometry denominator.

The accounting reference is the Stage 2 working grid. Canonical tiled inference
uses the original grid; downsampled research configurations do not gain a claim
of original-resolution fidelity. Independent Stage 1 validation stays active.

### Preservation and Recovery

Unclaimed components with two or more pixels are traced rather than skipped
by the former eight-pixel minimum. Small non-circular closed shapes are retained:
their size alone is insufficient evidence of noise. The former 80-pixel filter
now supplies an audit count, not a deletion decision.

Hatch deduplication additionally requires complete source containment. The
first canonical iteration exposed 104 lost pixels in EP2565056 during overlap
deduplication; the final guard keeps that ink in the hatch layer. This prevents
the final recovery step from reclassifying those lost hatch pixels as outlines.

`recover_source_coverage()` identifies unrepresented source components and
uses the existing digital-adjacency tracer. A one-pixel halo includes only
actual, already represented source ink for boundary contacts. It never bridges
white space or invents a full circle for a point. Existing endpoint node IDs
are reused when the pixel coordinates match; interior contacts are geometric
contacts, not yet a general edge-splitting/semantic-merging implementation.

Recovery preserves existing edge arrays, uses new unique IDs, and reports its
provenance. It is idempotent. Whole components above 100,000 pixels or requiring
more than the 5,000-new-edge budget remain explicit unresolved evidence and
flag the Stage 2 quality gate. These limits bound work, not validation accuracy.

Isolated single pixels remain unresolved review items. They do not become
zero-length lines, arbitrary circles, or automatically approved noise.
Legacy hatch-region cleanup now also preserves its residue side channel and
runs before final recovery/metrics; no cleanup deletes ink afterward.

### Acceptance

Geometry validator version `3` independently verifies native-grid accounting
against the real skeleton, graph pixels, residual spans, and counters. It rejects
wrong source hashes, hidden residuals, duplicate spans, and false counters.
Unresolved pixels add `geometry_stage2_residual_pending`; they remain in the
unchanged whole-raster geometry check, so distant marks can still cause failure.
Non-native accounting receives a source-verification review, not an exemption.

Acceptance policy is `2026-09-14.1`. Historical evidence retains its previous
identity. Accounted fraction 1.0 means every pixel is represented or explicitly
unresolved; it does not mean every pixel was reconstructed or accepted.

## Verification

All **391 tests pass**, including 21 new coverage cases. Tests cover exact loss
attribution, small loops/marks, junction contacts, branches, preservation of
existing edges and hatch ownership, no white-space bridges, idempotence,
resource limits, singleton handling, scaled-grid review, deduplication, and
attempts to falsify coverage evidence.

See [JUnit](audits/2026-09-14/stage2_coverage_tests.xml),
[machine-readable evidence](audits/2026-09-14/stage2_coverage_evidence.json),
and the [recovered CAD loop preview](audits/2026-09-14/stage2_coverage_preview.png).

### Frozen-Graph Recovery Replays

These isolate final recovery and fitting, retaining all existing source edges.
They do not rerun the learned detector or exercise early small-loop preservation.
The shared Stage 1 skeleton is the raster reference, not original CAD ground truth.

| Measurement | Before | After |
|---|---:|---:|
| CAD250 geometry pass / fail | 247 / 3 | 250 / 0 |
| CAD250 mean raster F1 at 2 px | 0.988022 | 0.989766 |
| CAD250 mean symmetric Chamfer, px | 0.308788 | 0.293808 |
| CAD250 budget primitives | 1,743 | 5,581 |
| Pilot2 geometry pass / fail | 0 / 2 | 1 / 1 |
| Pilot2 mean raster F1 at 2 px | 0.996371 | 0.999875 |
| Pilot2 mean symmetric Chamfer, px | 0.174769 | 0.137972 |
| Pilot2 budget primitives | 1,945 | 2,865 |

CAD250 recovers 2,531 pixels as 3,838 additional edges, leaving no unresolved
source pixels. Pilot2 recovers 1,364 pixels as 920 additional edges, leaving two
singletons in EP3082506. All emitted primitives pass individual source checks.

Important counter-results: CAD F1 improves in 184 views, regresses in 30, and
ties in 36. Mean precision decreases slightly, 0.990922 -> 0.990837. The worst
F1 loss is **0.024703** in `0024_00248013_FrontTopRight`; its original primitives
are exactly unchanged, but 297 added traces interact with the existing
4.33-pixel export stroke width. `0089_00896586_Right` already has poor raster
fidelity (F1 0.025558) and regresses to 0.019455 despite passing centerline
geometry checks. Thus source coverage and source-supported centerlines do not
certify rendered topology, line width, or acceptable CAD semantics. These
outputs are diagnostics, not promoted training targets.

### Fresh Targeted Checks

The three CAD regressions also rerun the actual D2C Puhachov model and full
Stage 2 logic on their unchanged skeletons, followed by diagnostic Stage 3/4.
All three now have complete source coverage and passing geometry:

| CAD view | Raster F1 before | Fresh F1 |
|---|---:|---:|
| 0026_00267762_Right | 0.982131 | 0.991607 |
| 0078_00787655_FrontTopRight | 0.983207 | 1.000000 |
| 0079_00793147_Front | 0.997757 | 1.000000 |

The canonical patent run reuses verified Stage 0/1 artifacts and reruns its
actual deployed Stage 2 models. The final ledger localizes remaining loss:

| Patent | Topology loss | Simplification loss | Recovered pixels | Residual pixels |
|---|---:|---:|---:|---:|
| EP2565056B1/F0001 | 521 | 0 | 521 | 0 |
| EP3082506B1/F0001 | 323 | 4 | 325 | 2 |

The first graph represents 40,074/40,074 source pixels. The second represents
54,015/54,017, retaining singleton coordinates `(597,1340)` and `(597,2120)`
as unresolved evidence. Six small non-circular closed shapes are retained in
EP3082506. Fresh diagnostic exports have F1 0.999938 and 0.999824 respectively;
the first passes geometry, the second still fails on its distant singletons.

**Canonical regression:** both patents now stop at the unchanged Stage 2
fragmentation gate. Their main graphs have 656 and 527 edges, with micro-edge
ratios 0.689 and 0.634. The previous canonical smoke exported one drawing and
gated the other later at Stage 3. Explicit source recovery improves fidelity
but creates too many short standalone traces. The real training-manifest
builder retains **zero** rows; neither a clean geometry result nor a diagnostic
SVG bypasses those gates. Unknown reference text also flags diagnostic exports.

## Artifacts

- `output/Drawing2CAD250_Stage2CoverageV2`: final frozen CAD recovery replay.
- `output/PatentData2_Stage2CoverageV2`: final frozen patent recovery replay.
- `output/PatentData2_Stage2CoverageSmokeV2`: canonical run, graph ledgers,
  acceptance records, database, empty manifest, and rejection CSV.
- `output/Stage2CoverageFresh5`: fresh CAD and canonical-patent diagnostic
  primitives, SVG/DXF, geometry reports, and combined evidence.

The initial unsuffixed runs are retained as iteration evidence. No historical
outputs were overwritten. The full Patent91 cohort was not rerun in this first
implementation; the 250-view CAD replay and five targeted fresh checks bound
the present evidence.

```bash
.venv/bin/python -m tools.replay_fitting_repair \
  --source output/Drawing2CAD250_CompactHatchFitV2 \
  --config output/Drawing2CAD/stage3_open_closed_full/evaluation_config.yaml \
  --output output/MyCoverageReplay --recover-coverage --raster-metrics --workers 4

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/audits/2026-09-14/evaluate_coverage.py \
  --output output/MyFreshCoverageChecks
```

The targeted evaluator expects the final smoke and manifest paths above.
Diagnostic replay bypasses execution gates to measure outputs, but does not
produce an acceptance decision or training eligibility. Use new output paths.

## Next Work

1. Splice supported recovered connections into their original strokes using
   explicit junction/edge contacts, preserving topology and source ownership.
   Reduce actual micro-edge proliferation without dropping recovered pixels
   or excluding them from quality gates.
2. Add rendered stroke-width/topology checks and repair the recorded CAD
   regressions; centerline validation alone is insufficient.
3. Resolve singletons through explicit point/detail semantics or independently
   justified noise/content routing, never implicit deletion.
4. Repeat broader fresh PatentData checks, then native hatch/DXF parity and
   content/reference acceptance work. This remains a first implementation,
   not a deployment-readiness claim.
