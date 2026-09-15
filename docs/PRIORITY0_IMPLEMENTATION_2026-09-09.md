# Priority 0: Preservation and Deployment Enforcement

Date: 2026-09-09. Follow-up to the
[project audit](PROJECT_STATUS_2026-09-09.md).

## Decision and Scope

The three highest-priority recommendations are implemented: retain recognized
reference text, stop losing unmatched hatch geometry, and enforce the canonical
batch chain. Model weights, reference detection/removal policy, main-geometry
fitting, and acceptance thresholds have not been retuned. No training, synthetic
generation, or new manual annotation was started.

This is a measured preservation improvement, not a declaration that arbitrary
PatentData figures are suitable as LLM training targets. In particular,
content routing and absolute reference/hatch accuracy remain open.

## Implementation

### Reference Round Trip

`_build_reference_doc` now retains detector `text`, `ref_class`, and
`confidence`. Stage 0 annotations carry these fields into export. SVG defaults
to the original label crop, without also drawing recognized text over it.
Explicit text rendering or a missing crop can use known text instead. Unknown
text remains unknown, not a guessed number.

DXF now receives recognized tokens as centered `MTEXT`, positioned using the
same pixel-center conversion as geometry and sized from the detected box.
Leader geometry remains unchanged. Unknown labels retain their SVG crops but
still do not have editable DXF text. This fixes serialization loss; it does not
fix OCR misrecognition or prove complete reference removal.

### Hatch Ownership and Fallback

Stage 2 regions now declare `source_hachure_indices` into `removed_hachures`.
These are array indices because historical edge IDs can repeat. Stage 3 accepts
ownership only when the complete source trace is contained by a valid region.
Legacy graphs infer containment conservatively; new graphs require explicit
membership. Unowned edges receive their own hatch line/polyline primitive.

In mixed region/fallback graphs, a line replacement must remain within 0.75
Stage 2 pixels of every source point, including endpoint bounds; otherwise the
raw trace is preserved. The existing no-region fitter behavior is retained.
Each exported hatch primitive records source ownership. The new
`quality_metrics.hachure_coverage` reports source, region, fallback, represented,
and unrepresented edges. An unrepresentable edge flags Stage 3.

Region membership is an accounting guarantee, not proof that a parametric
pattern reproduces every original hatch pixel. Convex-hull overfill, spacing,
angle, detector false positives, and strokes lost upstream remain separate
accuracy questions.

### Canonical Execution

`tools.batch_run` now defaults to `config_deploy.yaml`. Its strict profile:

- Validates selected Puhachov/hatch checkpoint bytes against
  [deployment_manifest.json](../models/deployment_manifest.json).
- Checks SketchCleanNet's hash when present. Binary passthrough is legitimate;
  grayscale inputs require a functioning cleaner.
- Refuses classical reference fallback when configured OCR is unavailable,
  keypoint inference fallback, and required hatch inference fallback.
- Records resolved settings, checkpoint hashes, selected stage-entry-point and
  batch/preflight code hashes, database identity, and initial worker count.
- Refuses adoption of a nonempty legacy database without a deployment manifest,
  or resume under changed recorded code/configuration/checkpoint identity.
- Rejects missing requested filter manifests and invalid worker counts.
- Returns failure for execution errors in strict batches. Quality-gate rejects
  remain deliberate pipeline outcomes, not process crashes.

`hachure_mode: region` and `dashed_grouping: false` make existing behavior
explicit. Device/worker changes can pass canonical behavior validation, but
remain recorded and can require a separate output identity. Research configs
can still opt out of strict deployment. Standalone research stage CLIs are not
a substitute for the canonical batch preflight.

This is not a complete environment lock or portable model package: transitive
helpers/dependencies and EasyOCR's cached weights are not all hash-pinned by
this manifest. Grayscale cleaner accuracy and CPU/GPU geometry parity are not
established by the binary CPU smoke below.

## Frozen 100-Patent Replay

Baseline: `output/PatentData100_Stage3GuardedP2FixedFrozen`.
Candidate: `output/PatentData100_Priority0Preservation`.

All 91 eligible Stage 3/4 rows were actually refitted and exported, with zero
execution errors. The three Stage 1 and six Stage 2 gated rows remain gated;
the success count is still 91/100, not 100/100. The replay took approximately
16 minutes 23 seconds with eight workers. Its saved replay digest is
`23e330cc865b9ff25fa44917f8963080fd013541d6e36735377c8d1e27e3287a`.

The [saved-artifact audit](audits/2026-09-09/priority0_replay.json) verifies:

- Main primitive dictionaries are exactly unchanged in all 91 drawings.
- Frozen graph, skeleton, and reference JSON bytes match the baseline.
- All 1,190 separated hatch edges have exactly one declared export owner.
- Eight drawings gain 341 fallback primitives; 83 drawings gain none.
- All 91 SVGs parse, all 91 DXFs pass the parser/auditor, and no reference
  crop is missing. The 2,744 historical labels remain unknown-text labels
  because frozen reference JSON intentionally predates the OCR text fix.

Full-resolution raster fidelity uses the shared reference-free Stage 1
skeleton, not manually annotated vector truth. The evaluator renders the
primitives without reference overlays so annotation reinjection does not
inflate this measurement.

| Metric, 91 matched drawings | Baseline | Preservation | Mean delta, 95% bootstrap CI |
|---|---:|---:|---|
| Symmetric Chamfer, px | 3.624344 | 3.569534 | -0.054810 [-0.104605, -0.015563] |
| Chamfer p95, px | 15.574026 | 15.158286 | -0.415740 [-0.873308, -0.072186] |
| Precision at 2 px | 0.912172 | 0.913289 | +0.001117 [0.000381, 0.001965] |
| Recall at 2 px | 0.858209 | 0.863338 | +0.005129 [0.001767, 0.009157] |
| F1 at 2 px | 0.878608 | 0.881590 | +0.002982 [0.000997, 0.005362] |

For every Stage 3 metric: eight improvements, 83 ties, zero regressions.
All Stage 2 metrics tie, as expected from frozen inputs. See
[complete metrics](audits/2026-09-09/priority0_fidelity.json) and
`output/Priority0Validation/patent100_fidelity.csv` for per-drawing values.

| Affected drawing | Additional fallback primitives |
|---|---:|
| EP2976579B1/F0003 | 15 |
| EP1467415B1/F0003 | 23 |
| EP1833535B1/F0001 | 53 |
| EP1974619B1/F0001 | 72 |
| EP2793370B1/F0001 | 20 |
| EP2565056B1/F0001 | 77 |
| EP3322902B1/F0002 | 40 |
| EP3499654A1/A0001 | 41 |

## Fresh CPU Smoke

The [two-figure worklist](audits/2026-09-09/priority0_smoke.csv) covers
EP3082506B1/F0001 and the mixed-hatch case EP2565056B1/F0001. Both ran through
Stage 0-4 from their original TIFs, without preprocessing reuse, in
`output/PatentData2_Priority0DeploySmoke`. Both passed, with no stage flags or
execution errors; wall time was approximately 5 minutes 14 seconds on two CPU
workers. Mean Stage 0/1/2/3/4 times were 193.79/0.16/62.58/27.86/0.28 seconds.

The [smoke export audit](audits/2026-09-09/priority0_smoke.json) confirms 34
reference labels: 14 recognized tokens retained in DXF and 20 unknown labels.
SVG contains 34 crop images and no duplicate text. Both drawings use binary
passthrough, `cnn_tiled` keypoints, and the CNN hatch source. All 88 hatch edges
are represented. Native membership keeps nine in the region and 79 as explicit
fallbacks; legacy replay inferred eleven region members and 77 fallbacks.
This conservative difference is expected, not a change to the main fitter.

Removal masks, reference-free rasters, and skeletons are byte-identical to
their frozen baseline counterparts. Preserving recognized tokens does not
change what Stage 1 receives. Recognition correctness has not been measured.

Visual inspection of `output/Priority0Validation/fresh_hatch_smoke.png` against
the cleaned raster still shows fragmentation and two unsupported large circles
in EP2565056B1/F0001. Their main primitives are byte-for-byte equivalent JSON
values in the old baseline: edge 118, radius 326.3136 px, confidence 0.8447;
edge 216, radius 212.6064 px, confidence 0.8680. Neither is introduced by the
hatch fix. High primitive confidence therefore does not establish whole-curve
support. Investigate circle/arc coverage and bidirectional primitive-to-source
distance before accepting this drawing as a training target; do not delete its
source traces to improve a score. The rendered SVG is nonblank (74,761 ink
pixels at 1100 x 1575), but renderability is not geometric correctness.

Final dependency-hardening recheck: **completed, 2/2 ok**, in
`output/PatentData2_Priority0DeploySmokeFinal`, after adding the explicit
missing-EasyOCR rejection. Wall time was approximately 5 minutes 10 seconds.
The [final export audit](audits/2026-09-09/priority0_smoke_final.json) again finds
14 known tokens, 20 unknown labels, 88/88 represented hatch edges, and no audit
errors. All primitive values match the first smoke; masks, reference-free
rasters, and skeletons still match the frozen baseline. The saved
[deployment identity](audits/2026-09-09/priority0_deployment_run.json) matches the
final code/configuration and records two CPU workers with no preprocessing
reuse. No validation or training job from this implementation remains running.
An actual resume with the final identity skips both completed rows and exits
successfully. Attempting resume into the earlier smoke directory, whose code
identity predates the OCR dependency guard, correctly exits with code 2 without
rerunning or overwriting its results.

## Tests and Reproduction

Full test suite: **223 passed**. Existing NumPy/RDP, Pyparsing/ezdxf, and PyTorch
warnings remain. New tests cover OCR text/class/confidence, unknown labels,
SVG no-double-rendering, DXF positioning, explicit and legacy hatch ownership,
partial containment, curved trace fallback, invalid regions, coverage audit
failures, model hashes, incompatible resume, and strict inference failures.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q
.venv/bin/python -m tools.deployment --config config_deploy.yaml

.venv/bin/python -m tools.audit_preservation_run \
  --run output/PatentData100_Priority0Preservation \
  --baseline output/PatentData100_Stage3GuardedP2FixedFrozen \
  --output output/Priority0Validation/replay_audit.json

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.evaluate_patent_fidelity \
  --run baseline=output/PatentData100_Stage3GuardedP2FixedFrozen \
  --run preservation=output/PatentData100_Priority0Preservation \
  --output-json output/Priority0Validation/patent100_fidelity.json \
  --output-csv output/Priority0Validation/patent100_fidelity.csv
```

Use a new destination for any new replay or fresh canonical batch. The baseline
audit evidence and historical evaluation outputs were not overwritten. Full
Drawing2CAD evaluation was not repeated in this implementation pass; focused
geometry tests and exact main-primitive preservation on 91 real patents provide
the current regression evidence.

## Next Priorities

1. Content routing with the existing 100 pilot-v3 labels and known chart/chemistry
   failures. Keep ambiguous figures quarantined and do not turn every successful
   reconstruction into an accepted training target.
2. Paired Stage 2 PatentVec A tiled versus historical Phase A fusion, with frozen
   preprocessing and the now-preserving Stage 3/4 implementation.
3. GPU smoke/parity and hatch threshold 0.7 evaluation on existing positive and
   negative masks. Report the known patent leakage in the historical hatch
   split; do not label these reused data an untouched gold test.
4. Target the remaining three Stage 1 and six Stage 2 gates, then dense Stage 3
   runtime, hidden-line semantics, and metamorphic geometry regressions. Include
   the confirmed unsupported-circle case above in the geometry regression set;
   positive fit confidence alone must not justify promoting an entire circle.

No change to training decisions: RANSAC remains production, Free2CAD remains
research, A/B synthetic corpora remain frozen, and the proposed 50k generation
has not been launched. These fixes do not establish an accepted hatch-detector
accuracy or solve the outstanding content/license/provenance release work.
