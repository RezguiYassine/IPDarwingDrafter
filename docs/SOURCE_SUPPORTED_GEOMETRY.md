# Source-Supported Geometry Validation

Implemented 2026-09-10. Validator `source_supported_geometry`, current version
`3`, schema `ap3-source-geometry-v1`; current acceptance policy `2026-09-14.1`.
The original evaluations below used version `1` and policy `2026-09-10.2`.
Version 2 adds independently owned explicit hatch-stroke bundle validation.
Version 3 verifies Stage 2 coverage accounting and keeps unresolved residuals
under review without removing them from raster checks. Numeric support
thresholds are unchanged. See [coverage recovery](STAGE2_SOURCE_COVERAGE.md) and the
[2026-09-14 fitting and hatch follow-up](COMPACT_HATCH_FIT_REPAIR.md).

## Scope

`tools/geometry_validation.py` measures fitted geometry against its own Stage 2
raw source pixels and the Stage 1 skeleton. It does not trust RANSAC confidence
or borrow support from unrelated edges. It does not refit, remove, or replace
primitives, change model weights, or rewrite source artifacts.

The canonical batch invokes the validator through `acceptance.record` after a
completed pipeline execution. Each figure receives:

```text
<run>/<patent>/geometry/<sketch>_geometry_report.json
<run>/<patent>/acceptance/<sketch>_acceptance.json
```

The geometry report contains per-primitive outcomes, source ownership,
bidirectional residuals, worst-point coordinates in the original image,
coverage checks, policy parameters, and hashes of the graph, primitive JSON,
and skeleton. Its file hash is bound into the acceptance record. A caller's
supplied `geometry: pass` cannot override the measured result. The training
selector verifies the current validator/version, report consistency, and
unchanged artifacts. Content validation is still pending, so geometry passes
alone do not qualify real drawings for training.

Old acceptance records are not grandfathered in. Changed implementation and
policy identities require a new canonical output directory/database.

## Checks

- Main edges must have unique IDs and exactly one owning primitive. Hachures
  use `source_hachure_indices` in their separate collection: their IDs can
  legitimately collide with main-edge IDs.
- Lines, arcs, circles, ellipses, polylines, polygons, cubic Beziers, and compound
  paths are sampled in Stage 2 coordinates. Source pixels are already in that
  frame; original-image primitives are multiplied by `stage2_scale`.
- Primitive-to-source distance detects invented geometry. Source-to-primitive
  distance detects lost geometry. Both are required independently for every
  primitive, not just as a drawing-wide average.
- Compound paths reuse the exporter's orientation algorithm and include its
  SVG connector geometry. Large connector gaps are flagged separately because
  the DXF segments remain disconnected.
- Full circles/ellipses require angular coverage. Open traced edges also get
  a direction-independent endpoint check. Non-cycle unclaimed components
  remain under review even when their distance checks pass.
- Complete sampled geometry is checked against the Stage 1 skeleton in both
  directions, exposing strokes lost before fitting. With incomplete sampling,
  available model-to-skeleton support is still measured, but total raster
  coverage cannot pass.
- Malformed geometry, invalid coordinates, missing ownership, and missing or
  undecodable evidence are errors. Empty geometry and established support
  violations fail. Sampling limits or unimplemented cases require review.

## Initial Policy

These are conservative engineering thresholds, not empirically certified
acceptance thresholds for all patent domains. Distances are in Stage 2 pixels,
not millimetres or SVG stroke-width units. Stroke width cannot relax them.

| Check | Threshold |
|---|---|
| Curve sampling | At most approximately 1 pixel between samples |
| Support tolerance | 2 pixels |
| Passing support fraction | At least 95% in each direction |
| Failing support fraction | Below 80% in either direction |
| Failing p95 distance | Greater than 4 pixels |
| Failing maximum distance | Greater than 8 pixels |
| Failing unsupported continuous run | Greater than 8 pixels |
| Open endpoints | Review above 3 pixels; fail above 6 |
| Path connector gap | Review above 2 pixels; fail above 4 |
| Full-curve angular gap | At most 30 degrees, relaxed for pixel-scale radii |
| Sampling limits | 200,000 samples/primitive; 2,000,000/drawing |
| Source limit | 2,000,000 pixels per source trace or skeleton |

The angular allowance is `min(180, max(30, degrees(4 / minor_radius_stage2)))`.
Intermediate support results require review. One failing primitive fails the
drawing's geometry decision; good primitives cannot hide it. Resource limits
never silently reduce sampling resolution to produce a pass.

## Verified Results

The [machine-readable evidence](audits/2026-09-10/source_geometry_evidence.json)
binds audit summaries to their code hashes, records the fresh smoke and
preservation checks, and identifies the prior CAD ground-truth diagnostic.
The [JUnit report](audits/2026-09-10/source_geometry_tests.xml) records the tests.

- **320 tests pass**, including 50 geometry tests. Coverage includes analytic
  shapes at Stage 2 scales 1, 0.5, and 0.25; rotation/translation; reversed
  tracing; real patent regression pixels; partial-arc circle promotion;
  off-source geometry; lost source/raster strokes; path gaps; hatch ID
  collisions; provenance; and caller-supplied pass bypass attempts.
- **91 frozen PatentData drawings: 91 geometry failures, zero audit errors.**
  This cohort already contains known issues; these are new measurements of
  unchanged reconstructions, not regressions introduced by the validator.
  Among 1,190 circle primitives, 979 fail, 195 require review, and 16 pass.
  All 20 hatch-region primitives remain under review. See
  `output/PatentData100_SourceGeometryValidation/summary.json` and its per-figure
  reports. The nine earlier-stage gated drawings have no complete export in
  this frozen cohort and are not part of these 91 checks.
- **250 seeded Drawing2CAD views: 211 passes, 14 reviews, 25 failures, zero
  audit errors.** All 52 circle primitives pass. Each primitive is paired with
  its own run's graph and skeleton in `output/Drawing2CAD/stage3_open_closed_full`.
  The views come from 250 distinct CAD models. Results:
  `output/Drawing2CAD_GeometryAudit250V1/summary.json`.
- As a diagnostic, joining those outcomes to the already-recorded pixel-center
  CAD evaluation gives mean symmetric p95 distance **0.8814 pixels** for pass
  views versus **2.1635 pixels** for fail views. Mean CAD pixel recall is
  **0.9615 versus 0.9284**. Review views can still have good aggregate fidelity:
  their mean IoU is 0.8183. This is not a fresh ground-truth evaluation, a
  classifier accuracy estimate, or evidence that all flagged details are wrong.
- **Fresh canonical smoke: 2/2 execution `ok`, 2/2 acceptance `rejected`, no
  process errors**, in `output/PatentData2_SourceGeometrySmoke`. Both geometry
  checks fail; content remains pending and unknown DXF reference text still
  requires review. The training-manifest command keeps **zero** and excludes
  **two**, each with `acceptance_rejected`. The run's deployment identity
  matches current code. Main/hatch primitives, reference-free rasters, removal
  masks, and skeletons exactly match `PatentData2_AcceptanceContractSmoke`.
  Wall time is about 5 minutes 16 seconds; this is a correctness smoke, not a
  throughput benchmark.

The frozen patent audit examined 15,970 primitives: 8,885 pass, 4,939 require
review, and 2,146 fail. Four drawings hit a primitive sampling limit and remain
explicitly unverified for that geometry; other established violations already
fail those drawings. The 91-drawing audit takes approximately 49.4 seconds of
accumulated per-figure processing; the 250-view CAD audit takes 4.1 seconds on
this machine. These timings exclude model execution and export regeneration.

The confirmed circles in EP2565056B1/F0001 explain why a one-direction fit
metric was insufficient:

| Main edge | Radius (pixels) | Observed angle | Full-circle supported fraction | Model-to-source p95 |
|---|---:|---:|---:|---:|
| 118 | 326.3136 | 30.8565 degrees | 8.67% | 637.79 pixels |
| 216 | 212.6064 | 28.2549 degrees | 8.15% | 416.43 pixels |

Both have source-to-model p95 below one pixel and fit confidence above 0.84,
yet fail whole-curve support. Their exact source pixels are retained in
`tests/fixtures/unsupported_patent_circles.json` for portable regression tests.

## Reproduction

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m tools.audit_geometry_run \
  --source output/PatentData100_Priority0Preservation \
  --output output/MyPatentGeometryAudit

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m tools.audit_geometry_run \
  --source output/Drawing2CAD/stage3_open_closed_full \
  --output output/MyCADGeometryAudit --limit 250 --seed 850725
```

The audit requires a new output directory. It writes reports only and cannot
promote historical data into canonical acceptance. When a graph-source override
is necessary, it refuses artifacts that differ from the source run's own saved
graph or skeleton. Early diagnostic runs with a nonexistent patent graph root
or mismatched CAD graph root are retained under explicitly named
`*_InvalidGraphRoot` and `*_MismatchedGraphRoot` directories and are not used in
the results above.

## Remaining Work

1. **Implemented 2026-09-11, evaluated 2026-09-12:** bounded non-cycle residual
   repair, source-supported circle/ellipse promotion, and recovered-stroke
   fallbacks. See [repair evidence and limitations](RESIDUAL_CURVE_REPAIR.md).
   All emitted full curves pass source checks in the paired Patent91/CAD250
   replay. The 2026-09-13 follow-up also guards ordinary/legacy hatch fits and
   path joins: final Patent91/CAD250 replays have zero failing primitive checks.
   Whole-drawing coverage and true stroke complexity still fail; compact
   explicit hatch bundles preserve detail but do not bypass the budget gate.
   No numerical acceptance thresholds were relaxed.
2. Verify native DXF HATCH phase, fill, and boundary fidelity against SVG and
   source hatch ink. Region ownership alone is not geometric validation;
   these primitives deliberately receive `geometry_hatch_pattern_unverified`.
3. Validate serialized DXF geometry against the intended primitives, especially
   fit-point splines and exporter fallbacks. This checker measures primitive
   geometry and known SVG path bridges, not a full DXF raster roundtrip.
4. Calibrate false rejections and gray-zone thresholds using existing CAD
   ground truth and targeted patent inspections. A geometry pass establishes
   fidelity to the retained Stage 1/2 evidence, not original CAD semantics or
   correctness of Stage 0 removal and Stage 1 cleaning.
5. Implement content routing. References, chemistry, charts, formulas, and text
   semantics are outside this validator; no missing content decision is inferred.
