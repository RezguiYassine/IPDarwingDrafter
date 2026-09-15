# Compact Hatches and Source-Faithful Fitting

Implemented and evaluated 2026-09-13; final audit documented 2026-09-14.
Production remains canonical PatentVec A Puhachov plus guarded RANSAC.
No model training, data generation, weight changes, or numerical acceptance
threshold relaxation. Historical outputs are retained.

## Outcome

Explicit hatches now use bounded, disconnected stroke bundles. Unsupported
main fits and implicit compound-path joins are repaired against their own
raw source pixels. The final iteration also fixes unsupported legacy hatch
line fits that previously bypassed source preservation without region metadata.

All **370 tests pass**. The fixed Patent91 and CAD250 cohorts complete without
replay errors. Every individually evaluated explicit primitive passes; native
hatch patterns remain deliberately unverified. This does **not** establish
whole-drawing acceptance, hatch-detection accuracy, or original CAD semantics.

See the [before/after crops](audits/2026-09-13/compact_hatch_fit_preview.png),
[machine-readable evidence](audits/2026-09-13/compact_hatch_fit_evidence.json),
and [JUnit report](audits/2026-09-13/compact_hatch_fit_tests.xml).

## Representation and Export

`hatch_strokes` contains explicit point arrays in `strokes`, independent
`stroke_source_indices`, and their flattened `source_hachure_indices`.
Ownership uses the hatch-array index, not potentially repeated graph edge IDs.
Bundles contain at most 32 strokes and 1,024 vertices, grouped in deterministic
256-pixel tiles in Stage 2 coordinates. Groups smaller than four strokes and
oversized individual traces retain their original representation.

Polyline simplification uses a 0.25-pixel tolerance and a source-support check.
Final validation checks each contained stroke against its own original graph
source, not the union of nearby hatch ink. Wrong ownership cannot be hidden by
a neighboring stroke. Coordinate scaling applies to every contained point.

- SVG uses separate `M` subpaths in one unfilled path. Disconnected strokes
  are not joined, and no filled hull or inferred hatch phase is introduced.
- DXF preserves each stroke as an open `LWPOLYLINE`, on `HACHURE` in patent
  mode. Consequently, fewer JSON/SVG objects do not mean fewer DXF strokes.
- Export audits verify mapped entities after serialization, SVG path-data
  hashes, and DXF stroke counts, coordinates, open state, and zero bulges.
  Missing strokes and introduced SVG connectors have failure tests.
- Existing native `hatch` regions are not silently replaced or certified.
  Their phase, clipping, spacing, and cross-format pattern fidelity still
  require a separate validator.

The 900-primitive limit remains effective: a bundle costs one budget unit per
contained stroke. `s3_n_primitives` counts JSON objects, while the new SQLite
column `s3_n_budget_primitives` records the independently recomputed gate cost.
Packaging 1,000 strokes into fewer than 900 objects cannot bypass the gate.

## Fitting and Continuity

The raw-source guard now covers ordinary RANSAC edges as well as recovered
residuals. Already supported analytic fits remain unchanged. Closed-trace
fallback uses the original sequence instead of reordering an already ordered
loop into unsupported links.

For compound paths, short SVG bridges are made explicit in both formats only
when samples every 0.25 pixels remain within two pixels of the owning source.
The whole repaired path must also pass the unchanged source validator and the
existing atom budget. This preserves valid curve segments while fixing the
SVG/DXF difference caused by implicit SVG `L` connectors.

When those joins are unsupported, the fitter tries checked raw line/arc fits,
then an endpoint-anchored compound path within the existing point/atom limits.
Finally, a checked compact trace or the exact raw trace preserves evidence at
low confidence, rather than falsely promoting it. A genuinely unsupported raw
gap remains rejected. Exact endpoint-constrained arcs are retained as arcs.

The separate legacy hatch fitter now checks its proposed line against the
complete source, even without region/residual metadata. Bent or inlier-trimmed
hatches fall back to their actual trace before bundling. Three real regression
fixtures from `EP1839350B1/F0002` cover this previously missed path.

DXF entity capture now slices only newly added modelspace entities instead of
repeatedly materializing the whole modelspace. This removes a quadratic audit
operation on dense drawings without changing emitted entities.

Geometry validator version is now `2`; acceptance policy is `2026-09-13.1`.
Historical reports keep their old identities and are not silently promoted.

## Paired Results

Baselines are the preceding residual-repair V2 outputs, on exactly the same
91 historical patent exports and 250 seeded CAD views. Graphs are unchanged,
all source ownership checks pass, and both formats export every primitive.
Raster metrics render the actual Stage 4 SVG without annotations at source
resolution, skeletonize it, and compare with the shared Stage 1 skeleton.
These are source-fidelity metrics, **not** a new CAD ground-truth evaluation.

| Measurement | Before | Final |
|---|---:|---:|
| Patent91 raster F1 at 2 px | 0.915722 | 0.920001 |
| Patent91 precision / recall | 0.978976 / 0.866828 | 0.983190 / 0.871139 |
| Patent91 symmetric Chamfer, px | 1.764149 | 1.731140 |
| Patent91 failing primitive checks | 513 | 0 |
| Patent91 review primitive checks | 1,519 | 20 native hatch patterns |
| Patent91 geometry pass / review / fail | 0 / 1 / 90 | 0 / 8 / 83 |
| CAD250 raster F1 at 2 px | 0.985261 | 0.988022 |
| CAD250 symmetric Chamfer, px | 0.329450 | 0.308788 |
| CAD250 failing / review primitive checks | 31 / 21 | 0 / 0 |
| CAD250 geometry pass / review / fail | 212 / 14 / 24 | 247 / 0 / 3 |

All 3,491 patent circles/ellipses and all 66 CAD circles/ellipses retain passing
source checks. No `geometry_path_discontinuity` flags remain in these cohorts.

There are counter-results, recorded rather than discarded:

- Patent F1 improves in 74 drawings, regresses in 15, and ties in two. Worst
  delta is -0.000414 on `EP3340025B1/F0020`. Chamfer improves in 65 and worsens
  in 26; the worst increase is 0.142442 pixels on that same drawing.
- CAD F1 improves in 21 views, ties in 226, and regresses in three. The largest
  loss is 0.000707 on `0024_00248013_FrontTopRight`; the other two losses are
  below 0.000001. Chamfer worsens in four views despite improving on average.
- Frozen Patent91 still costs **611,969 budget primitives**; 51 drawings exceed
  900. Historical main-graph ownership includes dense recovered hatch networks,
  which this frozen Stage 3/4 replay does not reclassify. JSON object count only
  falls from 611,969 to 611,770. Full minified primitive JSON actually grows from
  75,554,374 to 81,592,163 bytes as missing traces and explicit joins are retained.
  Small storage savings must not be presented as solved topology complexity.
- Patent whole-image coverage still fails in 83 drawings. Eight others require
  review because native hatch patterns prevent complete coverage validation.

### Hatch-Heavy Canonical Pilot

The two pilot graphs come from the prior canonical Stage 2 run, with its actual
hatch separation, unlike the historical frozen ownership above.

| Measurement | EP2565056B1/F0001 | EP3082506B1/F0001 |
|---|---:|---:|
| Hatch strokes retained | 1,348 / 1,348 | 196 / 196 |
| Hatch JSON objects, before -> after | 1,348 -> 55 | 196 -> 25 |
| All JSON primitive objects, before -> after | 1,557 -> 264 | 388 -> 217 |
| Minified hatch JSON bytes, before -> after | 746,497 -> 214,693 | 61,630 -> 18,450 |
| Budget primitives, unchanged | 1,557 | 388 |
| Raster F1, before -> after | 0.995728 -> 0.996607 | 0.994482 -> 0.996134 |

Combined hatch JSON shrinks **71.15%**, without dropping a hatch source edge.
Every individual primitive passes, both raster F1 scores improve, and mean
Chamfer improves from 0.178521 to 0.174769 pixels. This is compact explicit
encoding, not proof of a correct semantic hatch classification or pattern model.

The final canonical smoke reruns Stage 2 onward using verified reused Stage 0/1
artifacts. Graph, primitive, and skeleton hashes match the diagnostic pilot.
The first drawing still stops at the Stage 3 budget gate (1,557 > 900); the
second exports. Both acceptance decisions are rejected and the actual training
manifest builder keeps **zero rows**. Full diagnostics for the gated drawing
are available separately; they are not canonical accepted exports.

The remaining pilot coverage failures already exist between skeleton and graph:
34 source pixels in EP2565056 and 312 in EP3082506 lie more than eight Stage 2
pixels from any main or hatch graph point. The three remaining CAD failures
also have graph omissions, with 8, 33, and 4 such pixels respectively. These
need Stage 2 coverage repair or an explicit justified noise decision, not
fabricated fitting bridges or relaxed acceptance.

## Outputs and Reproduction

- `output/PatentData91_CompactHatchFitV2`: final fixed-graph patent replay.
- `output/Drawing2CAD250_CompactHatchFitV2`: final fixed-graph CAD replay.
- `output/PatentData2_CompactHatchFitV4`: final two-patent diagnostic SVG/DXF.
- `output/PatentData2_CompactHatchFitSmokeV2`: canonical smoke, acceptance, DB,
  empty training manifest, and rejection CSV.

Every replay includes per-drawing primitives, vectors, geometry reports,
`*_replay.json`, and `summary.json` with source/configuration/implementation
hashes. `tools.replay_fitting_repair` requires a fresh output directory.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.replay_fitting_repair \
  --source output/PatentData100_ResidualRepairV2 \
  --config output/PatentData100_Priority0Preservation/stage34_replay_config.yaml \
  --output output/MyCompactHatchPatentReplay --raster-metrics --workers 6

.venv/bin/python -m tools.replay_fitting_repair \
  --source output/Drawing2CAD250_ResidualRepairV2 \
  --config output/Drawing2CAD/stage3_open_closed_full/evaluation_config.yaml \
  --output output/MyCompactHatchCADReplay --raster-metrics --workers 2

.venv/bin/python docs/audits/2026-09-13/collect_compact_hatch_fit.py
```

The collector verifies final implementation hashes, canonical/pilot parity,
test results, and manifest exclusion, then regenerates evidence and previews.

Intermediate runs are retained. The first patent replay left 86 failing legacy
hatch primitives; the final legacy-source guard removes those failures. The
pilot V2 over-refitted some otherwise valid curves while anchoring joins; V3
adds explicit supported connectors, and V4 repeats with the final hatch guard.
Only the final paths listed above support the final numbers.

## Next Priorities

The first Stage 2 coverage implementation is now evaluated:
[accounting, preservation, residual recovery, and remaining regressions](STAGE2_SOURCE_COVERAGE.md).
The list below records the priorities at this earlier fitting checkpoint.

1. Audit Stage 2 skeleton-to-graph omissions, starting with the three remaining
   CAD views and the two pilot patents; preserve unsupported components before
   pruning or explicitly account for noise. Test meaningful long-stroke recovery.
2. Improve actual hatch/outline ownership and source-supported continuity on
   broader fresh canonical runs. Bound true stroke complexity, not only the
   number of JSON objects. Existing 900-budget gating stays in force.
3. Validate native hatch pattern phase/clipping and serialized DXF spline
   geometry. The new explicit-bundle checks are not a full DXF roundtrip audit.
4. Connect audited content labels to conservative routing for chemistry,
   formulas, charts, tables, and dense text, and address unresolved references.

The pipeline is improved, but it is not yet an automatically accepted patent
training-target generator. No training data was promoted by this work.
