# Acceptance Contract

Implemented 2026-09-10. Schema `ap3-acceptance-v1`, current policy `2026-09-14.1`.
The original checks below retain their historical policy identities. Current
geometry validator version 3 checks explicit hatch bundles per contained stroke
and verifies Stage 2 source-coverage accounting, including pending residuals;
the batch primitive budget counts strokes independently of packaging.
See [coverage recovery and its remaining gates](STAGE2_SOURCE_COVERAGE.md).
See the [compact-hatch/fitting follow-up](COMPACT_HATCH_FIT_REPAIR.md).
The initial verification below used policy `2026-09-10.1`. The subsequent
[source-supported geometry validator](SOURCE_SUPPORTED_GEOMETRY.md) is now
integrated; its outcomes replace the previous pending geometry check.

## Execution Is Not Acceptance

The existing SQLite `status` retains its execution meaning. A row may have
`status: ok` because every stage returned, while its exports or training
suitability require review. The new independent decision is:

| Acceptance status | Meaning | Training eligible |
|---|---|---|
| `accepted` | All required checks explicitly pass | yes |
| `review` | A warning, unknown reference text, incomplete validation, or unverified deployment | no |
| `rejected` | A configured quality gate or validator explicitly fails | no |
| `error` | Execution, artifact integrity, or export-accounting failure | no |

Decision precedence is `error`, then `rejected`, then `review`, then `accepted`.
Review and rejection preserve diagnostic outputs; they do not delete drawings.

**Current batches intentionally remain non-training-eligible:** content routing
has not been implemented yet. Geometry is now computed from source artifacts;
an externally supplied geometry pass is ignored. The batch driver cannot infer
content acceptance from `status: ok`, RANSAC confidence, or the old content
filter. Content remains `pending`, with an in-process interface for future
versioned decisions; there is no CLI switch that marks it as passed.

This work changes accounting and selection, not model weights, reference
detection, primitive fitting, or content classification. It does not fix the
previously documented unsupported circles.

## Saved Evidence

Each processed figure now receives:

```text
<run>/<patent>/vectors/<sketch>_export_report.json
<run>/<patent>/acceptance/<sketch>_acceptance.json
<run>/<patent>/geometry/<sketch>_geometry_report.json
```

Early stage failures/gates still receive acceptance records but may have no
export report because Stage 4 was never reached. A process crash can prevent
record creation; absence of a record never qualifies an example for training.

The acceptance JSON contains execution and acceptance status, `training_eligible`,
reason codes, policy version, deployment identity, a fingerprint of relevant
SQLite fields, artifact hashes, and these required checks:

- Execution completion and all available Stage 0-4 warning flags.
- Verified canonical deployment identity.
- Input, raster, graph, primitive, reference, crop, and output artifact presence.
- Independent SVG and DXF export accounting and serialization checks.
- Reference JSON-to-annotation consistency, crop presence, leader accounting,
  and explicit treatment of unknown text.
- Exact once-only ownership of separated hatch edges, checked against the graph
  rather than trusting only its reported coverage total.
- A versioned, computed geometry decision bound to graph, primitives, skeleton,
  and report hashes; a content decision that is currently pending.

SQLite records the decision, reason codes, policy, JSON path, JSON SHA-256, and
eligibility. Schema migration leaves historical rows unassessed; old `ok` rows
are not grandfathered into acceptance.

## Export Accounting

The old maximum-of-SVG/DXF success count is removed. `n_primitives_out` is now
the minimum serialized primitive count across requested formats. A format that
does not serialize successfully contributes zero, even if entities were created
in memory. Individual format reports retain both created and written counts.

Each source primitive has its own outcome and associated SVG element IDs or
DXF entity handles. One compound path may legitimately map to several DXF
entities. Empty/unsupported primitives, malformed path segments, incomplete
Bezier controls, invalid radii, and nonfinite coordinates cannot be counted as
successful exports. Partial geometry remains recorded as a failed source item.

Each annotation separately records label representation and expected/written
leaders. Known text missing in DXF, omitted annotations, and lost leaders are
failures even when every geometric primitive exported. Unknown DXF text is
recorded as unknown, never fabricated; it requires review. SVG can retain its
original crop. Missing crops block acceptance even if known text provides a
visual fallback.

After writing, SVG is parsed, checked for recorded/unique entity IDs, and
rendered with a bounded 512-pixel preview. Blank rendering with declared
primitives is flagged. DXF is parsed and audited, including audit repairs, and
all recorded modelspace handles must survive serialization. SVG rendering is
a renderability check, not full-resolution fidelity. DXF raster parity and
geometric source-support checks remain separate work.

## Training Manifest Enforcement

`tools.build_training_manifest` now requires the acceptance decision before
applying the existing content/complexity heuristics. It verifies:

1. The selected canonical configuration passes deployment preflight and matches
   the run manifest and database path.
2. The acceptance file hash matches SQLite; its schema and policy are current.
3. Deployment identity and relevant row fields match the accepted record.
4. Recomputing the decision from all required checks still yields `accepted`.
5. Content identifies a validator/version, and geometry matches the current
   source-support validator, its bound report, and the exact checked inputs.
6. Bound artifacts are still present and byte-identical.

Permissive legacy curation flags cannot bypass this gate. Historic runs without
valid acceptance evidence produce no eligible rows. A post-hoc Stage 3/4 replay
is not automatically a new canonical accepted run.

Before submitting reprocessing jobs, the batch controller invalidates previous
acceptance seals for those rows. A worker crash therefore cannot leave an old
accepted result eligible for a new attempt. Resume skips already processed rows
as before; rerunning a figure requires explicit reprocessing. Changed canonical
code or policy requires a new output directory/database.

These hashes prevent accidental stale/mixed evidence; this is not a signed
attestation against someone who can deliberately rewrite the code, database,
and artifacts together. The deployment manifest now includes the acceptance,
export-audit, results-store, and training-selector code in its implementation
fingerprint.

## Usage

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.batch_run --config config_deploy.yaml \
  --worklist docs/audits/2026-09-09/priority0_smoke.csv \
  --output output/MyAcceptanceRun --workers 2

.venv/bin/python -m tools.build_training_manifest \
  --config config_deploy.yaml \
  --db output/MyAcceptanceRun/results.db \
  --run-output output/MyAcceptanceRun \
  --filter-manifest output/PatentData/filter_manifest_clean12.csv \
  --output-csv output/MyAcceptanceRun/training_manifest.csv \
  --rejects-csv output/MyAcceptanceRun/training_manifest_excluded.csv
```

Until content validation is integrated, an empty eligible
manifest is expected. Its excluded CSV preserves individual reason codes.

## Initial Contract Verification

- The full suite passes: **268 tests, including 45 new acceptance tests**.
  The new failure suite covers stage warnings, early errors/gates, missing
  evidence, asymmetric format failures, lost annotations/crops/leaders,
  incomplete hatch ownership, empty primitives, partial compound export,
  serialization loss, blank SVG, stale policy/identity/artifacts, database
  migration, and reprocessing invalidation. Valid controlled fixtures pass.
  [JUnit results](audits/2026-09-10/acceptance_tests.xml) record zero failures,
  errors, or skipped tests. Existing dependency deprecation warnings remain.
- Full frozen-patent Stage 4 re-export: **91/91 drawings retain every source
  primitive, 182/182 SVG/DXF files pass serialization checks, and 91/91 SVG
  thumbnails are pixel-identical to the baseline** at a maximum edge of 512
  pixels. There are no annotation export errors. Outputs are in
  `output/PatentData100_AcceptanceExportCheck`; the
  [per-drawing report](audits/2026-09-10/acceptance_export_check.json) records
  the results. The legacy frozen reference annotations contain **2,744 unknown
  DXF labels**, now explicitly accounted for rather than treated as accepted
  labels. This check does not rerun Stage 0-3, improve OCR, or measure new
  full-resolution geometric accuracy.
- Fresh canonical Stage 0-4 smoke: **2/2 execution `ok`, 2/2 acceptance
  `review`, zero training-eligible rows**, in
  `output/PatentData2_AcceptanceContractSmoke`. Both rows have unknown DXF
  reference text and pending content/geometry validation. These are review
  reasons, not process crashes. Artifact integrity and hatch accounting pass.
  The deployment identity matches the current code. Main and hatch primitives,
  reference-free rasters, removal masks, and skeletons exactly match the
  preceding smoke run. Mean Stage 4 time is 0.74 seconds per drawing; mean
  complete processing time is 290.82 seconds. Resume correctly skips both rows.
- The real training-manifest command **keeps zero and excludes two**, each
  with `curation_reason: acceptance_review`, despite execution `status: ok`.
  Results and preservation checks are saved in
  [acceptance_smoke.json](audits/2026-09-10/acceptance_smoke.json).

The previous 2026-09-09 evaluation remains a historical baseline, not evidence
that the newly required content/geometry checks have passed.

Reproduce the tests and frozen export check with:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/audits/2026-09-10/validate_acceptance_exports.py \
  --source output/PatentData100_Priority0Preservation \
  --output output/MyAcceptanceExportCheck \
  --report output/MyAcceptanceExportCheck/results.json
```

The export audit requires a new output directory to preserve previous results.

## Next Work

The geometry checker, bounded non-cycle repair, supported circle/ellipse
fitting, ordinary fit/path-join guards, and explicit hatch compaction are now
implemented and tested on CAD and patent regressions. Next address Stage 2
source omissions and true stroke complexity, validate native hatch patterns
and serialized DXF spline geometry, then
connect the existing audited content labels to accept/reject/review routing.
Only those explicit validators can turn technically intact exports into
accepted training targets. No new training, 50k generation, or manual gold-set
construction was started for this contract.
