# Patent-Like Synthesis Roadmap (10k)

Target: a 10,000-drawing synthetic corpus whose **raster topology** matches
filtered PatentData, so Stage 2 trains on a distribution that behaves like the
deployment domain. Deployment gates remain calibrated on real patents only.

## Implementation Review (2026-09-15)

The first **300-drawing measurement probe is complete and audited**. It is a
baseline diagnostic, not an accepted 10k training release. All new data belongs under
`/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset`.

- [x] Add all-sample full/reference-free clean and degraded PNG retention.
- [x] Add a balanced medium/hard/very-hard schedule and stratified audit sampling.
- [x] Make degradation parameters explicit, configurable, and reproducible.
- [x] Allow the established reference-free C2 label contract in generated shards.
- [x] Bind generation to configuration, source-index, implementation and runtime
  fingerprints; reject incompatible resumes and concurrent writers.
- [x] Generate and audit the balanced 300-sample Phase 0 probe.
- [x] Measure real Stage 1 results, production CN clusters and legacy raw-degree
  statistics separately; reject universal numeric bands contradicted by the probe.
- [x] Implement a fixed-source acquisition/scale probe with binary controls,
  paired speckle settings and constant-pixel versus relative lineweights.
- [x] Measure binary/grayscale acquisition, native scale and lineweights through
  450 paired Stage 1 passes on 30 frozen drawings (new Phase 0.5).
- [ ] Integrate binary acquisition as a candidate, recalibrate noise against ink
  and resolution, and close the remaining native-topology gap before release.
- [ ] Prevent colliding annotation labels and redundant dimension placement.
- [ ] Implement source-disjoint pools with no unrestricted retrieval fallback.
- [ ] Run controlled composition/noise/scale probes and freeze the 10k recipe.
- [ ] Generate and accept the 10k release. A 50k run remains out of scope.

Important corrections from code inspection:

1. The historic table has no checked-in measurement implementation or explicit
   branch/tiny-component definitions. Its numeric bands are **provisional**,
   not automatic generation acceptance thresholds. Raw 8-neighbour degree
   pixels are not production crossing-number (CN) junction clusters.
2. Counts per 1,000 skeleton pixels are **not resolution-invariant**: with fixed
   vector topology, doubling raster resolution approximately doubles stroke
   length but not graph-node count. Compare matched frames and lineweights,
   alongside native-resolution measurements and tile counts.
3. The intersection rejection concerns `object_visible` primitives belonging
   to different source components. It does not reject all hatch, leader or
   centre-line intersections, and raster contact does not necessarily imply a
   structural connection. Never turn every crossing into an amodal junction.
4. `syntheticData/configs/full_generation.yaml` is a planning-only file. The
   executable component ranges are in `ComplexPilotGenerator._compose`:
   medium 4-5, hard 8, very hard 9-12, with mandatory interaction operators.
5. The archives retain `masks.npz`, `sample.json` and targets for every sample.
   They can already support full-tier C2 topology measurement and deterministic
   re-rendering. Missing retained PNGs bias the earlier image-only comparison;
   they do not mean the hard tiers or ground truth are irrecoverable.
6. The old interaction-count proxy for `cycle_rank` includes containment and
   placement relations. It is not the cycle rank of the visible stroke graph.
7. Scan gaps can change topology. Clean structural targets, noisy observations,
   and the actual model-input skeleton must remain distinct artifacts. C2
   endpoints/junctions must exactly describe the skeleton stored with them.

The initial hypothesis came from the 2026-09-15 skeleton-space comparison
across CAD250, PatentVec A/B and the frozen 100-patent cohort. The complete
Phase 0 probe now shows that **very-hard clean targets already exceed the
matched-frame patent junction density**. More crossings in every tier is not
justified. Encoding, preprocessing and scale must be separated from composition.

## 1. Measured Gap

Previously reported medians over skeletons. These historical values require
reproduction with documented preprocessing and metric definitions. Densities
are per 1,000 skeleton pixels and must not be compared across resolutions
without a matched-frame check.

| Metric | CAD250 | PatentVec (medium) | **Patents** | Direction |
|---|---:|---:|---:|---|
| Junctions / 1k px | 3.2 | 35.4 | **131.1** | **3.7x too low** |
| Endpoints / 1k px | 0.0 | 44.5 | **13.1** | 3.4x too high |
| Tiny-component fraction | 0.000 | 0.832 | **0.250** | 3.3x too high |
| Short-branch fraction | 0.000 | 0.822 | **0.513** | 1.6x too high |
| Median branch length, px | 421.5 | 1.0 | **5.0** | 5x too short |
| Connected components | 1 | 255 | 160 | too fragmented |
| Skeleton pixels | 1,552 | 4,389 | 36,892 | scale artifact, see 4.3 |

Real Stage-2 gate metric, for reference: patents `micro_edge_ratio` median
`0.290`, p90 `0.596`; CAD250 median and p90 are both `0.000`.

**Historical hypothesis, not a release diagnosis:** additive noise may be too
strong and topology too sparse in the medium-only comparison. Section 8
supersedes this blanket claim with full-tier and real-preprocessing evidence.

## 2. Why The Current Generator Produces This

Four mechanisms to test; none alone establishes the cause of the deployment gap.

### 2.1 Incidental crossings are rejected as defects

`syntheticData/patentvec/quality.py` fails a sample on
`"{label}: {n} unintended cross-component intersections"`. Any crossing not
created by an explicit interaction operator rejects the drawing.

This gate checks cross-component object-visible strokes only. Other semantic
layers already produce raster contacts. The useful experiment is whether
additional coherent object contacts improve similarity, while preserving the
distinction between structural joins, non-connected crossings and occlusions.

### 2.2 Contacts are forced apart

`CompositionBuilder._is_clear_contact(position, clearance=0.025)` (and `0.035`
for `t_junction`) enforces a minimum separation between contact points, which
spreads junctions out instead of letting them cluster as they do in dense
mechanical views.

### 2.3 Difficulty is scaled by component count, not by connectivity

The planning YAML lists source-component ranges but is not consumed by the
generator. The executable `_compose` selects medium 4-5, hard 8 and very-hard
9-12 components plus typed interactions. Adding components with clearance adds
**ink** without adding **junctions**. `_complexity_metrics` tracks
`structural_junction_count` (junctions created by interactions) but never the
raster junction density that the measured gap is expressed in.

### 2.4 Speckle is over-applied

`syntheticData/patentvec/render.py::degrade_patent_scan` applies
`dark_speckles < 0.00012` and `light_speckles < 0.00008` per pixel. On a
1024x1024 canvas that is roughly 126 dark and 84 light isolated pixels. Each
dark speckle may survive direct thresholding and skeletonization. This is a
plausible contributor to the historical tiny-component fraction, not a proven
cause after deployment Stage 1: SketchCleanNet/binarization can suppress such
marks. A paired clean/degraded and real-preprocessing comparison is required.

## 3. Blocking Prerequisite

**Rasters are sparsely retained.** With the historical `--audit-every 100`
schedule, each 100-sample shard retains PNGs for only its first sample. Other
samples retain vector JSON, semantic masks and training arrays. Every retained exemplar is tagged
`medium`, and the retained A and B rasters are byte-identical (matching
SHA-256).

Consequences:

- The measured "PatentVec" column above is **medium-tier only**. The `hard`
  tier is unmeasured; A/B cannot be distinguished from stored rasters at all.
- Retained-PNG-only comparisons are biased. Masks and stored vector geometry
  allow other tiers to be measured or re-rendered without resampling sources.

Restore measurable raster coverage before claiming a distribution match.

## 4. Phases

### Phase 0 — Restore measurability, then re-measure

1. Use `--retain-rasters` to keep full clean/degraded and oracle reference-free
   clean/degraded PNGs for every sample. Keep semantic masks, original vector
   provenance and both training targets. Use `--audit-strategy stratified` for
   the larger visual triptychs independently of raster retention.
2. Regenerate a 300-sample stratified probe under the **current** settings,
   balanced across `medium` / `hard` / `very_hard`.
3. Re-run the skeleton-space comparison on that probe. Report per-tier native
   measurements, matched-frame measurements, raw 8-neighbour pixel statistics,
   production CN cluster counts, and actual deployed Stage 1 outputs separately.
   The oracle reference-free render is not a claim that Stage 0 OCR is perfect.
4. Audit C2 labels against independently computed production CN clusters using
   exact set equality, not an 8-pixel nearest-label tolerance. Verify mask
   support, duplicate/extra labels, raster retention, source provenance, shard
   hashes and the independent Free2CAD polyline contract.

**Acceptance:** balanced 100/100/100 retained samples; documented per-tier
metrics and clean target contracts. Recalibrate the provisional bands using
the same metric definition and scale. If hard tiers already cover the patent
regime, prefer curriculum re-weighting over unnecessary topology changes.

### Phase 0.5 - Match acquisition and protect visual coherence (next)

1. Hold source geometry, seeds and semantic masks fixed while comparing binary
   patent-office scans, grayscale degradation, and lower-speckle degradation.
   Record the actual Stage 1 route. The frozen patent cohort uses binary
   passthrough for 100/100 samples; the baseline degraded synthetics invoke
   SketchCleanNet for 300/300. Do not compare those routes as though identical.
2. Retain separate acquisition variants and apply C2 labels to the exact
   skeleton that accompanies them. Noisy-input restoration targets and clean
   raster-topology labels are different supervision contracts. Do not silently
   attach clean C2 coordinates to a changed, post-cleaning skeleton.
   **PNG encoding/noise changes alone do not change the existing C2 training
   inputs**, which are rebuilt from clean semantic masks. A Stage 2 training
   distribution change needs a measured change in scale, geometry/lineweight,
   curriculum, or an explicitly new observation-supervision contract.
3. Run native-scale/lineweight probes as well as matched-frame comparisons.
   The patent native median is 49 tiles at 512/256; the current 1024-square
   synthetic canvas has 9. Report per-tier tails and morphology, not just seven
   corpus medians. Establish route-specific acceptance bands after this probe.
   Follow up with structure-only and hatch-conditioned measurements: total CN
   density can be dominated by cross-hatching and is not evidence of coherent
   mechanical connections or long-stroke continuity.
4. Make reference/dimension placement collision-aware. The visual audit found
   colliding horizontal dimension annotations in `pv_train_0000002_very_hard`.
   Such samples currently pass geometry gates; visual coherence needs its own
   gate, including label/leader placement and redundant dimensions.
   Code inspection confirms that `add_linear_dimension` appends the label even
   when `_box_available` is false, and very-hard layering requests the overall
   horizontal extent twice. Rotated text also needs a rotation-aware box.
5. Before a training release, split original source identities (including
   sibling CAD views), prohibit unrestricted pool fallback, and verify actual
   source-content hashes. A source-index checksum alone is insufficient to
   detect edits to the underlying geometry files.

**Acceptance:** fixed-geometry paired results, explicit acquisition/label
contracts, improved geometric fidelity through the intended preprocessing
route, source-disjoint partitions and a clean stratified visual review. Never
compensate for destructive preprocessing by generating artificial crossings.

### Phase 1 — Composition: more crossings, more connected structure

Run only where Phase 0.5 still shows a structural deficit. Hard/very-hard
curriculum re-weighting is the first control, not a global crossing increase.

1. **Introduce typed incidental-contact labels, not blanket gate removal.**
   Resolve every allowed cross-component intersection and preserve source
   intervals. A structural join splits incident curves into a shared graph
   node; a non-connected crossing preserves separate structural ownership;
   occlusion changes the visible graph without joining the amodal graph.
   Retain a rejection for cases
   that genuinely break ground truth: coincident/overlapping collinear spans and
   intersections that cannot be resolved to a point. This extends the principle
   already established by the C2 contract, that the labelled topology must match
   the raster the pipeline actually sees.
2. **Relax contact clearance** to a resolution-relative floor (roughly one
   stroke width) instead of a fixed `0.025` canvas fraction, so contacts may
   cluster.
3. **Add an explicit interaction budget** to the curriculum, independent of
   component count: target `interactions_per_component` and a minimum visible
   graph cycle rank, computed from graph vertices, edges and components rather
   than the existing placement-interaction proxy.
4. **Add coherent patent-specific contacts**: clipped section hatches meeting
   boundaries, leaders crossing geometry, and centre lines through parts.
   Controlled scan bleed may be modeled as observation noise; do not invent
   physically incoherent hatch overshoots merely to increase a statistic.

**Provisional hypotheses, to revise after Phase 0:** raw junction-pixel density `120-140` per 1k skeleton pixels at
the corpus median, endpoint density falls to `10-18`, and median branch length
rises to `4-7 px`, with every added crossing represented in the exported
Puhachov and Stage-2 targets. Ground-truth exactness is non-negotiable: the
independent contract audit must still report zero unmatched endpoints,
junctions, or unsupported skeleton pixels.

### Phase 2 — Degradation: less additive speckle

1. Lower `dark_speckles` to approximately `0.00004` and `light_speckles` to
   approximately `0.00003`, then tune against the measured
   `tiny_component_fraction`, rather than fixing the constants by eye.
2. Replace a share of isolated-pixel speckle with **stroke-attached**
   degradation (short spurs, small gaps, thickness variation). Measure their
   effects rather than assuming they leave keypoints intact. Do not edit clean
   geometry labels to explain nuisance speckle or erased observations.
3. Make degradation parameters configurable and recorded in the manifest, so a
   corpus can state the noise model it was generated under.

**Provisional hypotheses, to revise after Phase 0:** `tiny_component_fraction` in `0.20-0.32` and
`short_branch_fraction` in `0.45-0.60` at the corpus median, with Stage-1
skeletons still recovering the structural graph (no loss of labelled
structure relative to the clean render).

### Phase 3 — Scale alignment

Do **not** chase `skeleton_pixels` directly; it is a canvas-size artifact.
Instead set the canvas and stroke width so that a generated drawing enters
Stage 2 at the same tiling regime as a patent scan (512/256 tiles at native
resolution), then report how density and topology change under rasterization.
Scale experiments must retain vector provenance and re-audit raster labels;
unchanged density is not a valid requirement.

**Acceptance:** comparable native tile counts and stroke widths; matched-frame
topology distributions within the bands established in Phase 0. Report the
native density change, rather than requiring an invalid +/-10% invariance.

### Phase 4 — Generate and accept the 10k

1. Freeze the parameter set from Phases 0.5-3, source-disjoint partitions and
   source-content checksums in the manifest.
2. Generate 10,000 accepted drawings, retaining rasters per Phase 0.
3. Re-run the skeleton comparison on the full corpus, the independent
   polyline/topology contract audit, and a stratified visual review.

**Acceptance:** all Phase 1-3 bands met at corpus scale; contract audit clean;
visual review finds no broken geometry or incoherent overlaps.

## 5. Guardrails

- **Do not tune to the statistic alone.** Matching seven medians does not prove
  transfer. Treat the bands as necessary, not sufficient.
- **Keep A and B frozen** as comparators. The new corpus is a third arm, not a
  replacement.
- **Run an equal-budget control.** Any downstream training claim needs a
  matched run on the existing corpus, or the effect of the new distribution
  cannot be separated from more optimization.
- **Source-disjoint validation.** The existing A/B split is *not* source
  disjoint; both sides draw from the same source pool. The new corpus must split
  at the original sketch/CAD-model identity, including sibling D2C views.
  Index-restricted retrieval must fail closed: `SourcePool.pick_profile`
  currently falls back to the full source pool when indexed retrieval fails,
  which would invalidate a partition. Phase 0 is a train-pool diagnostic probe,
  not a source-disjoint train/validation release.
- **Do not reuse this corpus to calibrate the Stage-2/3 gates.** Gates are being
  calibrated on the real patent cohort, which is the only set that exhibits the
  failure regime. Synthetic data is for training, not for setting deployment
  thresholds.

## 6. What This Does Not Establish

This roadmap targets **raster topology similarity**. It does not establish that
a topologically patent-like corpus improves real-patent Stage 2 accuracy; that
requires the equal-budget control above. It does not address semantic content
mix (patent figures include charts, flowcharts and chemistry that should be
routed out, not generated), and it does not remove the need for content routing
before any drawing becomes training-eligible.

It also does not, by itself, justify the 50k full generation. Phase 0 through
Phase 4 produce the evidence needed to make that decision; the decision remains
separate.

## 7. Evidence

Measurement source for section 1:

```text
output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14/*/cleaned/*_skeleton.png
output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14/*/graphs/*_graph.json
output/Drawing2CAD250_Stage2CoverageV2/*/cleaned/*_skeleton.png
output/PatentVecComplexityA10k/shards/*.tar   (degraded.png exemplars)
output/PatentVecComplexityB10k/shards/*.tar   (degraded.png exemplars)
```

Code referenced:

```text
syntheticData/patentvec/quality.py        unintended-intersection rejection, _complexity_metrics
syntheticData/patentvec/composition.py    _is_clear_contact, interaction operators
syntheticData/patentvec/render.py         degrade_patent_scan speckle constants
syntheticData/configs/full_generation.yaml  planning-only curriculum ranges
syntheticData/measure_patent_like.py        exact C2 audit and per-view measurements
```

## 8. Completed Phase 0 Results (2026-09-15)

Dataset: `/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300`.
The original composition and default degradation settings were intentionally
held fixed. This is not yet a newly accepted patent-like distribution.

- 300 drawings: 100 medium, 100 hard, 100 very hard; 60 checksum-verified shards.
- 258.57 seconds on 12 CPU workers, 1.160 drawings/second. Archives occupy
  278,589,440 bytes (265.68 MiB); measurement outputs are additional.
- All 1,200 full/reference-free clean/degraded PNGs retained, plus vector/source
  JSON, semantic masks and both model-target archives for every drawing.
- Exact C2 audit: 34,185 endpoints and 110,886 junctions; **zero missing, extra
  or duplicate topology labels, and zero unsupported skeleton pixels**.
  Archives also contain 3,037 corners; the exact-set audit concerns endpoints
  and junctions, not semantic corner completeness.
- Free2CAD supply: 3,471 lines, 210 arcs, 96 circles, 1,329 polylines and
  254 Beziers. All 1,329 polylines passed the independent existing contract
  across all 300 samples. Worst per-target p90 matching error was 3.606 px
  against its 4 px threshold; this is tolerance-based fitting, not zero error.
- 30 stored visual previews, ten per tier. All loaded in Chromium at
  1440x1000 and 390x844 without horizontal overflow or browser errors.
- Full repository tests: **437 passed**. All 60 resume markers independently
  revalidated against the original plan, archive sizes and SHA-256 checksums.

Medians below use 100 drawings per row. Raw junction pixels and CN junction
clusters are distinct metrics; both densities use 1,000 skeleton pixels.

| Input/view | Raw junction density | CN junction density | Tiny CC fraction | Short branch fraction | Median branch px |
|---|---:|---:|---:|---:|---:|
| Real patents, native Stage 1 | 131.11 | 36.66 | 0.317 | 0.513 | 5 |
| Real patents, matched longside 1024 | 112.07 | 32.03 | 0.385 | 0.511 | 5 |
| Medium, C2 clean target | 68.11 | 19.86 | 0.069 | 0.202 | 14 |
| Hard, C2 clean target | 115.77 | 28.42 | 0.187 | 0.270 | 11 |
| Very hard, C2 clean target | 270.64 | 50.45 | 0.194 | 0.494 | 6 |
| Medium, degraded RF through Stage 1 | 16.80 | 5.52 | 0.000 | 0.176 | 22 |
| Hard, degraded RF through Stage 1 | 42.59 | 10.91 | 0.000 | 0.248 | 12 |
| Very hard, degraded RF through Stage 1 | 130.83 | 24.12 | 0.000 | 0.485 | 6 |

RF means oracle reference-free, without running OCR. Matched-frame resampling
can itself alter topology. Tiny CC means <=4 pixels; short branches are
<=5-pixel components after removing raw degree>=3 pixels, not traced strokes.
The historical tiny-component figure 0.250 is not reproduced with this explicit
definition (native median 0.317). The raw junction figure 131.1 is reproduced.

Against its own C2 clean target at a 2 px tolerance, the actual synthetic
Stage 1 output has median precision/recall/F1 of:

| Tier | Precision | Recall | F1 |
|---|---:|---:|---:|
| Medium | 0.5897 | 0.9240 | 0.7196 |
| Hard | 0.6114 | 0.9146 | 0.7303 |
| Very hard | 0.6782 | 0.9213 | 0.7828 |

These are image-fidelity diagnostics, not Puhachov evaluation scores. No model
was trained or promoted in this run. The clean targets bracket the patent CN
density, but preprocessing changes that distribution substantially. The next
priority is Phase 0.5, followed by conditional composition/noise adjustments.

At unchanged recipe and measured throughput, 10k would take approximately
2 hours 24 minutes and 9.29 GB of archives, excluding audits and training
exports. This is a linear estimate, not a benchmark of the future recipe.
**The 10k training release and 50k generation have not been launched.**

Evidence and reproduction commands:
[`docs/audits/2026-09-15/patent_like_generation/README.md`](audits/2026-09-15/patent_like_generation/README.md).
Complete per-sample metrics and audits remain next to the external dataset;
`phase0_summary.json` in the audit directory records compact results and hashes.

## 9. Completed Acquisition/Scale Probe (2026-09-15)

Dataset: `/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase05_acquisition30`.
This reuses 30 Phase 0 drawings (10 per tier), not 450 independent drawings.
Five acquisition variants across three render settings produced **450 actual
Stage 1 passes**: 270 binary passthrough and 180 SketchCleanNet. All 90 clean
target audits passed with zero missing/extra/duplicate CN labels or unsupported
pixels (11,146 endpoints and 35,541 junctions). All 450 retained input/output
hash pairs were independently verified. Regression tests: **439 passed**.

Pooled medians over the same 30 source drawings per setting, using a 2 px
fidelity tolerance against each setting's clean C2 target:

| Render setting | Grayscale | Grayscale, lower speckle | Binary | Binary, lower speckle |
|---|---:|---:|---:|---:|
| 1024, relative widths | 0.7134 | 0.7128 | 0.9896 | **0.9963** |
| 2048, relative widths | 0.7432 | 0.7406 | 0.9813 | **0.9932** |
| 2048, fixed-pixel widths | 0.7330 | 0.7330 | 0.9824 | **0.9937** |

The binary C2 positive control returned the exact target skeleton in all 90
cases. Its perfect score is an integrity check, not learned recovery. At 2048,
the report also includes a 4 px tolerance for normalized-coordinate comparisons.

At 1024, binarizing the same default observation increased paired mean F1 by
0.2911 (exploratory paired-bootstrap 95% interval 0.2532-0.3294). Lowering
speckle alone in grayscale changed mean F1 by only 0.0002, with an interval
including zero. The dataset-side encoding/Stage 1 route is the dominant
measured acquisition effect. **No production preprocessing behavior changed.**

This does not close release acceptance:

- Binary/lower-speckle tiny-component fraction is still 0.603 at 1024 and
  0.784 at 2048 relative widths, versus the real-patent native median 0.317.
  A constant per-area dirt probability gets worse relative to ink at larger
  canvases. Test an ink-/resolution-aware noise budget and stroke-attached
  wear; do not remove true small geometry to force a match.
- 2048 achieves 49 native tiles, but pooled clean CN density is only 13.95
  versus 36.66 for native patents. Scaling alone therefore does not reproduce
  the patent topology regime. Select a curriculum using native, matched-frame,
  structure-only and hatch-conditioned evidence, not total junctions alone.
- Annotation collisions, strict source-disjoint pools and source-content
  checksums remain open. Puhachov training inputs are unchanged by PNG-only
  acquisition choices; no training-transfer claim or retraining is warranted
  from these fidelity results alone.

The paired gallery is `phase05_acquisition30/index.html`, showing one drawing
per tier at all three render settings. All 99 displayed images loaded at
1600x1000 and 390x844 without page overflow or browser errors. Complete
per-tier results, paired confidence intervals and hashes are retained in
`phase05_summary.json` alongside the Phase 0 evidence. The full report and
approximately 281 MiB of probe artifacts remain on the external disk.
