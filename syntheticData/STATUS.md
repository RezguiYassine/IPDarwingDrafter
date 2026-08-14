# PatentVec status

## Implemented

- [x] Canonical primitive, component, anchor, interaction, junction, and sample schema.
- [x] Similarity transforms and exact line/circle/arc/Bezier transformation.
- [x] Primitive sampling, intersections, analytic line splitting, and source intervals.
- [x] Split-safe SketchGraphs source adapter and connected constraint components.
- [x] Split-safe CAD-VGDrawing SVG adapter with exact `M/L/C/Z` conversion.
- [x] Endpoint, boundary, circle, centroid, and closed-region anchors.
- [x] Endpoint join, T-junction, connected X-junction, containment,
      concentric, tangent, and cycle-forming bridge operators.
- [x] Visible/amodal geometry, full transform records, and source provenance.
- [x] Hatching/cross-hatching, centreline, hidden-line, leader, arrowhead,
      reference-numeral, dimension, text-box, and connector layers.
- [x] SVG/raster rendering, seeded non-geometric patent degradation, and masks.
- [x] Puhachov and local Free2CAD primitive-fitting adapters.
- [x] Schema, residual, bounds, global unintended-intersection, interval,
      clearance, mask, density, graph-cycle, and difficulty gates.
- [x] Unit/integration tests for M0-M6, including repeated line splits and
      visible/amodal junction integrity.
- [x] Restart-safe medium/hard preview generator, manifest, paginated contact
      sheets, and HTML viewer.
- [x] Reusable SketchGraphs/CAD-VGDrawing source index with indexed retrieval.
- [x] Compact multiprocess tar sharding, atomic markers, SHA-256 checksums,
      deterministic resume, and storage/throughput forecasting.
- [x] Medium/hard/very-hard curriculum and procedural polyline supply.
- [x] Stage 2-shaped Free2CAD projection and independent polyline contract
      validator.
- [x] Native Puhachov/Free2CAD exporter with non-periodic hash splitting.
- [x] Equal-class, equal-compute Free2CAD A/B corpus builder.
- [x] Retry-aware hybrid statistical comparison and stratified audit viewer.
- [x] Exact four-pool Stage 2 rehearsal with protected step-zero selection and
      final-state validation.
- [x] Protected-step-zero Stage 3 A/B training and own/cross synthetic
      evaluation.
- [x] Production-identical Stage 2 crossing-number target builder and topology
      contract audit.
- [x] Reference-free C2 Stage 2 export that excludes reference, leader,
      dimension, and text branches before skeletonization.
- [x] Per-sample/per-class focal normalization for dense synthetic topology.
- [x] Three-epoch C2 rehearsal with immutable selected and resumable
      checkpoints.
- [x] Lossless two-label structural/hachure skeleton contract with overlap
      supervision and structural fallback.
- [x] Restart-safe hatch-stroke exporter, visual audit sheet, conservative
      checkpoint metrics, and CPU end-to-end trainer smoke test.

## Completed review

- [x] Generate the balanced 30-sample M5/M6 1024 px pilot.
- [x] Pass all automated gates and audit all three contact sheets.
- [x] Obtain user approval of the M5/M6 pilot.
- [x] Add greater complexity and explicit polyline supply from user feedback.
- [x] Generate, audit, and compare frozen 10,000-sample A/B datasets.

## Frozen 10k evidence (2026-07-25)

- Outputs: `output/PatentVecComplexityA10k` and
  `output/PatentVecComplexityB10k`.
- Acceptance: 20,000/20,000 final samples passed; all Stage 3 targets are
  topology projected.
- Enhanced composition: 2,500 medium, 5,000 hard, and 2,500 very hard.
- Enhanced labels: 336,833 exact Puhachov targets with 45 dropped; 43,401
  Free2CAD polyline labels before trainer cleaning.
- Independent contract: 500/500 samples and 2,168/2,168 polyline targets pass.
- Visual review: 18 samples per difficulty inspected across seven contact
  sheets; no broken geometry or incoherent overlaps found.
- A/B result: +19.5% components, +34.6% junctions, +13.5% polylines, and
  +18.3% exact keypoints. All comparison gates pass.
- Native Stage 2 exports: 8,976 training samples per variant, exact synthetic
  coverage, and equal 59,840-sample epochs at an 85/15 mix.
- Native Stage 3 A/B corpora: 914,250 labels per variant, exactly 182,850 per
  class, 822,349 identical real rows, aligned class/domain schedules, and
  byte-identical fixed validation.
- Stage 2 result: A/B final three-real-domain macro-F1 is 0.8477/0.8477 versus
  0.8193 at step zero; synthetic A/B validation improves from 0.1746/0.1650
  to 0.5047/0.4791.
- Full Stage 2 result: A/B score 0.856136/0.857240 on 31,524 Drawing2CAD test
  views, 0.936599/0.936426 on 9,565 SketchGraphs test drawings, and
  0.730050/0.727709 on all 1,159 ArchCAD validation drawings. The three-domain
  means differ by only 0.00047.
- Stage 3 result: the 0.9683 real-validation warm start remains selected;
  adapted A/B states reach 0.9622/0.9626 real macro-F1 but improve synthetic
  macro-F1 to 0.7862/0.7793 and synthetic polyline F1 to 0.8419/0.8228.
- Full generation was not started and the external data root was not written.

## C2 topology evidence (2026-07-31)

- C1 is diagnostic only: it labels complete raster topology but retains
  annotation-contact junctions that Stage 0 is intended to remove.
- C2 export: 10,000 drawings, split 8,976/1,024, with 4,766,069 labels.
- Independent 500-source audit: 57,582/57,582 endpoints and
  179,539/179,539 junctions matched; all hatch topology matched; zero skeleton
  pixels fall outside the reference-free semantic support.
- Selected C2 checkpoint: final step 14,961, three-domain selector `0.853213`;
  SHA-256 `2cce917f7db3a3942d8767af8cd0fde1aa7576bdced1556556e0c1931e6b091a`.
- Complete C2 validation: `0.705869 -> 0.845934` macro-F1.
- Complete SketchGraphs test: `0.935981`; complete ArchCAD validation:
  `0.716139`. Drawing2CAD and paired PatentData gates are running.

## Before a full run

- [x] Add a connected 4-8 component medium/hard M5 graph curriculum.
- [x] Add placement clearance, exact crossing, and cycle/bridge policies.
- [x] Extend the upper curriculum tail to 10-12 components.
- [ ] Add remaining roadmap operators and explicit visibility/occlusion curricula.
- [x] Build reusable component indices for high-throughput retrieval.
- [x] Add topology-equivalent shallow-chain aggregation and class-balanced
      polyline supply for the Free2CAD edge adapter.
- [ ] Calibrate distributions quantitatively against filtered PatentData.
- [x] Generate and manually audit frozen 1,000- and 10,000-sample experiments.
- [ ] Review source licenses and provenance release metadata.
- [x] Add restart-safe sharding, checksums, disk forecasting, and multiprocessing.
- [x] Complete full real-domain Stage 2 evaluation.
- [ ] Complete filtered PatentData visual regression. Equal-budget Stage 2 and
      Stage 3 A/B training/evaluation is complete; C2 paired evaluation is
      running.
- [ ] Start approved full generation under `/media/safe/secondary disk/IPdrawings`.

Track-B CAD-valid generation remains a separate later milestone.

## Hachure decomposition evidence (2026-07-31)

- Broad and solvable T/X Hough-routing variants were rejected because exact
  hachure-heavy PatentData fidelity remained below the intersection-preserving
  cleanup arm.
- The 128-sample multilabel pilot is split 114/14 and every sample contains
  structural-only, hatch-only, and true-overlap labels.
- Exact semantic support supplies 431,665 structural-only, 651,209 hatch-only,
  and 60,626 overlap pixels with zero unassigned skeleton pixels.
- Radius-one support was rejected: it increased overlap to 146,741 pixels
  without recovering any unassigned input ink.
- The two-channel trainer passes a CPU end-to-end smoke run. This verifies
  mechanics, not quality; production Stage 2 remains unchanged.
- Next gate: full 10k export and training, complete synthetic validation, then
  frozen real hachure-heavy PatentData fidelity/fragmentation comparison.

## Hachure decomposition update (2026-08-07)

- The full 10,000-drawing hatch-stroke corpus is complete at
  `output/PatentVecHatchStroke10k`: 8,976 train and 1,024 validation figures.
- The selected 30-epoch synthetic checkpoint is retained at
  `models/hatch_stroke_multilabel.pth` with SHA-256
  `8a9b8e1d3f6d1e4dc983d45b597eb9614771aec1d195d1d310350a51526542a6`.
- Synthetic calibration selected hatch probability `>= 0.50` and structural
  probability `< 0.03`: safe-removal precision `0.999507`, recall `0.923380`,
  and structural-removal error `0.000553`. These are synthetic-only metrics.
- Destructive pre-topology inference is rejected on 97 paired PatentData
  Stage 2 cases: edges increase by 12.55 on average, median edge length drops
  by 19.20 px, and short-edge ratio rises by 0.01132.
- Replacing the reviewed region detector with the stroke model is also
  rejected on 20 matched current-code cases: it classifies only 11 graph edges
  while the region path removes 617, raising main edges by 47.85 on average.
  The additive variant is an exact 20/20 tie and adds no value.
- The accepted production change is preservation-first residue handling:
  classify region-matching residue before metrics, retain it in
  `removed_hachures`, and reconnect the main graph without pruning newly
  exposed arms. On 97 paired Stage 2 cases, 3 improve, 94 tie, and 0 regress;
  F1 rises from `0.942526` to `0.942967` and symmetric Chamfer falls from
  `1.585188` to `1.580809`.
- The 269 reviewed subdrawing masks reduce to 212 unique figures: 132 positive
  and 80 negative. A current Stage 0-2 teacher replay is running across both
  RTX 4090 GPUs. Real labels use per-channel supervision masks so uncertain
  region interiors and polygon boundaries do not become false stroke labels.
- A corrected 11-figure exporter smoke has 9,774 trusted hatch pixels, 1,882
  crossing pixels, and 10/11 positive figures with safe teacher hatch support.
- On the first two real validation figures, the synthetic checkpoint preserves
  structure (`0.997592` recall) but has only `0.075201` hatch recall and
  `0.019441` safe-removal recall. This confirms a real-domain adaptation need.
- Next gate: finish the 212-figure export and audit, compare 35% and 50%
  real-domain mixed fine-tunes, then require synthetic preservation plus
  paired real PatentData fragmentation, fidelity, quality-gate, and visual
  wins before enabling the stroke CNN in Stage 2.

## Hachure reviewed-real result (2026-08-07)

- Teacher replay completed 203/212 figures with 3 Stage 1 and 6 Stage 2 quality
  gates and no runtime failures.
- Strict region-and-Hough consensus exported 170 train and 33 patent-disjoint
  validation figures. The corpus contains 199,789 trusted hatch pixels, 24,015
  overlap pixels, and 5,661,723 jointly supervised skeleton pixels.
- Full 30-epoch real35 and real50 training is complete. Selected checkpoints
  reach real hatch F1 `0.856876` and `0.927022`, respectively, but real
  structural recall is only `0.994210` and `0.991928`; neither is deployable.
- Synthetic difficulty preservation passes baseline-relative floors, but no
  single threshold policy passes both synthetic and real validation. The real
  structural channel remains overconfident on hatch-only pixels.
- Structural-negative-weight 4 and 8 continuations were stopped after epoch 1.
  Synthetic/real structural recall fell to `0.99196/0.96223` and
  `0.98794/0.93579`; neither moved toward the release gate.
- A source-separated geometric guard prevents the stroke model from deleting
  long structural contours. On the frozen 20-figure screen the guarded model is
  an exact tie; the apparent earlier gains came entirely from the already
  accepted region-residue cleanup.
- The fresh 100-figure current-code end-to-end A/B is also an exact tie through
  Stage 4 and on all raster-fidelity metrics. Both arms finish 73 OK, 3 Stage 1,
  6 Stage 2, and 18 Stage 3 gates. The model finds 19,665 region-exclusive pixels
  on eight figures but zero geometrically eligible additive edges.
- Hatch-stroke deployment is rejected as safely inert. Production remains the
  region detector plus early residue preservation.
- Giant unclaimed residual networks now quality-gate in Stage 2; OpenCV
  component labeling and bounded/linear Stage 3 traversal remove the previous
  quadratic runtime failure.
- Next priority is the 18 remaining Stage 3 confidence gates. Their 1,001 low
  primitives comprise 358 open lines, 78 open arcs, 263 closed polygons, 301
  closed polyline fallbacks, and one ellipse. Improve weak-fit routing and
  fidelity-bounded closed-path simplification; do not lower the 0.25 gate.
