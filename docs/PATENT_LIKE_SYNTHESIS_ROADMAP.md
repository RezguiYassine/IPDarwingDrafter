# Patent-Like Synthesis Roadmap (10k)

Target: a 10,000-drawing synthetic corpus whose **raster topology** matches
filtered PatentData, so Stage 2 trains and Stage 2/3 gates calibrate on a
distribution that behaves like the deployment domain.

Derived from the 2026-09-15 skeleton-space distribution comparison across
CAD250, PatentVec A/B and the frozen 100-patent cohort. The corpus does not
need to be larger. It needs **more crossings and connected structure per
drawing, and less additive speckle**.

## 1. Measured Gap

Medians over Stage-1 skeletons. Densities are per 1,000 skeleton pixels, so
they are comparable across canvas sizes.

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

**One-sentence diagnosis:** the generator over-models acquisition noise and
under-models drawing topology. It scatters more loose fragments than a real
scan while producing far less structural tangling.

## 2. Why The Current Generator Produces This

Three specific mechanisms, all in code, all changeable.

### 2.1 Incidental crossings are rejected as defects

`syntheticData/patentvec/quality.py` fails a sample on
`"{label}: {n} unintended cross-component intersections"`. Any crossing not
created by an explicit interaction operator rejects the drawing.

Real patent figures are full of incidental crossings: hatching over section
boundaries, leaders over geometry, overlapping views, centre lines through
parts. The generator is designed to avoid exactly the structure we need.

### 2.2 Contacts are forced apart

`CompositionBuilder._is_clear_contact(position, clearance=0.025)` (and `0.035`
for `t_junction`) enforces a minimum separation between contact points, which
spreads junctions out instead of letting them cluster as they do in dense
mechanical views.

### 2.3 Difficulty is scaled by component count, not by connectivity

`syntheticData/configs/full_generation.yaml` defines the curriculum purely as
source-component ranges (`medium: [3, 5]`, `hard: [5, 8]`,
`very_hard: [8, 12]`). Adding components that are placed with clearance adds
**ink** without adding **junctions**. `_complexity_metrics` tracks
`structural_junction_count` (junctions created by interactions) but never the
raster junction density that the measured gap is expressed in.

### 2.4 Speckle is over-applied

`syntheticData/patentvec/render.py::degrade_patent_scan` applies
`dark_speckles < 0.00012` and `light_speckles < 0.00008` per pixel. On a
1024x1024 canvas that is roughly 126 dark and 84 light isolated pixels. Each
dark speckle survives skeletonization as its own tiny component, which is the
direct cause of `tiny_component_fraction = 0.832`.

## 3. Blocking Prerequisite

**Rasters are not retained.** Each shard stores 100 samples but keeps
`clean.png` / `degraded.png` / `semantic.png` for only the **first** sample;
the remaining 99 keep label arrays only. Every retained exemplar is tagged
`medium`, and the retained A and B rasters are byte-identical (matching
SHA-256).

Consequences:

- The measured "PatentVec" column above is **medium-tier only**. The `hard`
  tier is unmeasured; A/B cannot be distinguished from stored rasters at all.
- No raster-level metric, gate calibration, or pipeline replay can run on the
  existing corpora without regeneration.

Nothing else in this roadmap can be verified until this is fixed.

## 4. Phases

### Phase 0 — Restore measurability, then re-measure

1. Retain `degraded.png` (at minimum) for every sample, or add a
   `--retain-rasters` sampling mode that keeps a **stratified** subset across
   difficulty tiers rather than the first sample per shard.
2. Regenerate a 300-sample stratified probe under the **current** settings,
   balanced across `medium` / `hard` / `very_hard`.
3. Re-run the skeleton-space comparison on that probe.

**Acceptance:** per-tier medians for all seven metrics in section 1, measured
rather than extrapolated. If the `hard` tier already reaches patent junction
density, the composition changes in Phase 1 reduce to a curriculum re-weighting
and Phase 1.1/1.2 are not needed.

### Phase 1 — Composition: more crossings, more connected structure

1. **Convert unintended intersections from a rejection into a label.** Detect
   every cross-component intersection, emit it as a junction in the visible and
   amodal graphs, and keep the sample. Retain a rejection only for the cases
   that genuinely break ground truth: coincident/overlapping collinear spans and
   intersections that cannot be resolved to a point. This extends the principle
   already established by the C2 contract, that the labelled topology must match
   the raster the pipeline actually sees.
2. **Relax contact clearance** to a resolution-relative floor (roughly one
   stroke width) instead of a fixed `0.025` canvas fraction, so contacts may
   cluster.
3. **Add an explicit interaction budget** to the curriculum, independent of
   component count: target `interactions_per_component` and a minimum
   `cycle_rank`, so difficulty increases connectivity rather than only ink.
4. **Add patent-specific crossing layers**: hatching that crosses its own
   section boundary, leaders that cross object geometry, and centre lines that
   run through parts. These are the dominant incidental-crossing sources in real
   figures and are already partially present as layers.

**Acceptance:** junction density reaches `120-140` per 1k skeleton pixels at
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
   degradation (short spurs, small gaps, thickness variation), which is what
   real scans mostly produce and what leaves keypoints intact.
3. Make degradation parameters configurable and recorded in the manifest, so a
   corpus can state the noise model it was generated under.

**Acceptance:** `tiny_component_fraction` in `0.20-0.32` and
`short_branch_fraction` in `0.45-0.60` at the corpus median, with Stage-1
skeletons still recovering the structural graph (no loss of labelled
structure relative to the clean render).

### Phase 3 — Scale alignment

Do **not** chase `skeleton_pixels` directly; it is a canvas-size artifact.
Instead set the canvas and stroke width so that a generated drawing enters
Stage 2 at the same tiling regime as a patent scan (512/256 tiles at native
resolution), then confirm the **density** metrics are unchanged by the
resolution change. Absolute ink follows from correct density plus correct scale.

**Acceptance:** density metrics stable within +/- 10% across the resolution
change; Stage-2 tile counts per drawing comparable to the patent cohort.

### Phase 4 — Generate and accept the 10k

1. Freeze the parameter set from Phases 1-3 and record it in the manifest with
   checksums.
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
syntheticData/configs/full_generation.yaml  curriculum component ranges
```
