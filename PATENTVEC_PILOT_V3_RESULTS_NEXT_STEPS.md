# PatentVec Pilot v3 Results: Interpretation and Recommended Next Steps

## Purpose

This document explains the results obtained from the 100 real patent drawings processed through the PatentVec pilot v3 pipeline. It focuses on:

- the missing measurements that must be completed before declaring the current manifest clean;
- the correct interpretation of the flagged samples;
- the recommended next steps;
- targeted fixes for the valid drawings that still fragment;
- how the different sample groups should be used for training;
- the evaluation and ablation plan for future improvements.

---

# 1. Current experiment summary

## 1.1 Input data

The pilot used 100 real patent drawings sampled from `filter_manifest_v3`.

The code distribution was:

| Code | Number of drawings |
|---|---:|
| F | 91 |
| A | 9 |
| C | 0 |
| D | 0 |

This means the current conclusions are strong for code `F`, partially informative for code `A`, and provide no evidence for codes `C` and `D`.

The current 100-image set should be frozen as a regression benchmark, but it is not sufficient as the final representative evaluation set.

## 1.2 Stage-2 robustness results

| Stage-2 configuration | Passed | Quality-gated |
|---|---:|---:|
| CN-fusion | 91 | 9 |
| Full-CNN phase A + tiling | 70 | 30 |
| Full-CNN phase B + tiling | 71 | 29 |

CN-fusion is clearly the strongest production configuration.

Compared with the two pure-CNN variants, CN-fusion:

- improves the accepted rate by approximately 20 percentage points;
- reduces failures from approximately 30 percent to 9 percent;
- is roughly three times more robust in terms of rejected cases.

The similarity between phase A and phase B also indicates that both CNN variants share the same underlying limitation.

## 1.3 Interpretation of the tiling result

Native-resolution tiling did not solve the pure-CNN fragmentation problem.

This suggests that the problem is not caused only by insufficient input resolution.

The remaining limitations are more likely related to:

- lack of long-range stroke continuity;
- poor grouping of separated dashes;
- confusion between hatching and structural boundaries;
- confusion between labels, boxes, and object geometry;
- weak topology preservation in dense regions;
- inability to reconnect structures across tile boundaries;
- excessive sensitivity to small local details.

Tiling may preserve local details while still losing the global relation between them.

Therefore, increasing tile resolution alone is unlikely to make the pure-CNN path competitive with CN-fusion.

CN-fusion should remain the Stage-2 production baseline.

---

# 2. Post-vectorization gate results

The final post-vectorization classification was:

| Result | Number |
|---|---:|
| Clean | 85 |
| Low quality | 8 |
| Not OK | 7 |
| Total flagged | 15 |

The clean training yield is therefore 85 percent.

The 15 flagged drawings were manually audited and divided into three meaningful groups.

---

# 3. Interpretation of the flagged groups

## 3.1 Group 1: unsuitable content correctly rejected

Approximately five drawings were not appropriate vectorization targets.

Examples included:

- shaded or photorealistic renders;
- contour or line plots;
- user-interface or text mock-ups;
- signal-flow diagrams;
- block diagrams.

These images cannot always be removed using only global image features.

For example, a block diagram and a mechanical patent drawing may have similar:

- foreground ratios;
- numbers of connected components;
- line densities;
- black-and-white distributions;
- image sizes.

The post-vectorization gate has access to stronger evidence, such as:

- graph fragmentation;
- primitive distribution;
- text proportion;
- topology;
- isolated component count;
- curve consistency.

Therefore, this group confirms that the two-stage filtering approach is working correctly:

```text
Coarse image pre-filter
    -> removes obvious unsuitable content

Vectorization

Post-vectorization quality gate
    -> removes content that looks acceptable globally
       but does not form a valid engineering vector graph
```

### Correct training role

These examples should be used as:

- invalid or out-of-domain negatives;
- quality-gate training examples;
- out-of-distribution examples;
- pre-filter and post-filter calibration samples.

They should not be used as positive samples for Puhachov or the Stage-2 vectorizer.

## 3.2 Group 2: valid patent drawings that over-fragment

Approximately five drawings were valid patent drawings, but the pipeline failed because of genuine geometric difficulty.

The main failure styles were:

- dense PCB or detailed 3D geometry;
- reference numerals mixed with geometry;
- hatched mechanical parts;
- dashed centre lines;
- circular bolt patterns;
- text inside boxes;
- cross-sections with hatching;
- dimension arcs;
- angle annotations.

This is the most important group because these are real positive samples.

The problem is not that the drawings should be rejected. The problem is that the vectorizer must improve.

### Correct training role

These examples must be treated as:

- hard positive samples;
- manual-correction candidates;
- active-learning candidates;
- targeted synthetic-generation templates;
- regression examples for fragmentation fixes.

They must not be globally labelled as negative drawings.

### Local hard-negative interpretation

Although the full drawing is a positive sample, some local regions can serve as hard negatives for specific semantic classes.

Examples:

- hatch strokes are negative examples for `object_boundary`;
- numerals are negative examples for `object_geometry`;
- separated dashes are positive examples for `same_logical_line`;
- dimension arcs are negative examples for `physical_object_contour`;
- text-box interiors are negative examples for `object_component`.

This distinction is essential.

A valid difficult drawing must not be inserted into a global invalid-image dataset.

## 3.3 Group 3: borderline valid drawings rejected aggressively

Approximately four drawings appeared valid but were slightly over-penalized by the post-vectorization gate.

Examples included:

- a clean box with a small glyph;
- sphere construction lines;
- a detailed coil;
- a shaded panel.

For training-data selection, these false positives are relatively cheap because excluding a few usable figures helps preserve label purity.

For production, however, rejecting valid user inputs is undesirable.

### Recommended policy

Use two different quality-gate modes.

#### Training-corpus mode

```text
Clean
    -> include automatically

Low quality or uncertain
    -> exclude or send for manual review

Not OK
    -> exclude
```

The objective is maximum precision.

#### Production mode

```text
ACCEPT
    -> return vectors normally

ACCEPT_WITH_WARNING
    -> return vectors with reduced-confidence warning

REJECT
    -> unsupported or clearly unsuitable input
```

The borderline Group 3 cases should usually become `ACCEPT_WITH_WARNING`, not hard rejections.

---

# 4. Missing measurement

## 4.1 The current audit measures only flagged samples

All 15 low-quality or rejected samples were manually audited.

This tells us whether the gate's rejections were reasonable.

However, it does not tell us how many incorrect samples may exist among the 85 accepted drawings.

The remaining unknown is the false-negative rate of the post-vectorization gate.

A bad vectorization may still pass the gate and enter the clean training manifest.

## 4.2 Required accepted-sample audit

Before declaring `pilotv3_train_clean.csv` fully clean, audit the following accepted samples:

- 20 randomly selected accepted drawings;
- 10 accepted drawings closest to the rejection threshold;
- all accepted drawings with unusually high primitive counts;
- all accepted drawings with unusually high connected-component counts;
- accepted drawings with high text, hatch, or line-density scores.

The audit should answer:

- Is the figure a valid patent drawing?
- Is reference text correctly removed?
- Is the main object geometry preserved?
- Is the graph excessively fragmented?
- Are dashed lines grouped correctly?
- Are hatches separated correctly?
- Are text boxes isolated from object geometry?
- Are there important missing contours?
- Are unrelated components incorrectly merged?

## 4.3 Why this measurement matters

If nearly all audited accepted samples are correct, the 85-image manifest is trustworthy.

If several accepted samples are still fragmented or unsuitable, the post-vectorization gate needs:

- additional features;
- a higher threshold;
- failure-family-specific logic;
- better calibration.

The clean count of 85 should therefore be considered provisional until this accepted-sample audit is completed.

---

# 5. Immediate recommended next steps

## Step 1: Freeze and commit the current baseline

Commit and preserve:

- the three Stage-2 configurations;
- `filter_manifest_v3`;
- the exact 100-image sample list;
- the v3 viewer builder;
- `pilotv3_train_clean.csv`;
- the complete audit CSV;
- ghost-circle fixes;
- hatch fixes;
- all quality thresholds;
- the current code commit hash.

Create a baseline report with:

```text
CN-fusion:
    accepted = 91
    rejected = 9

Full-CNN phase A:
    accepted = 70
    rejected = 30

Full-CNN phase B:
    accepted = 71
    rejected = 29

Post-vectorization:
    clean = 85
    low_quality = 8
    not_ok = 7
```

The current 100 samples should become a frozen regression set.

Every future change must be evaluated against the same inputs.

## Step 2: Split the current samples into four explicit manifests

Do not use a single generic hard-negative dataset.

Create four distinct manifests.

### A. Clean positives

Suggested file:

```text
pilotv3_positive_clean.csv
```

Contains the 85 accepted drawings after the accepted-sample audit.

Training role:

- normal real positive training;
- pseudo-label training;
- clean-domain validation.

### B. Hard positives

Suggested file:

```text
pilotv3_positive_hard.csv
```

Contains valid patent drawings that over-fragmented.

Examples:

- dense PCB;
- hatched mechanical parts;
- dashed bolt circles;
- boxed-label cross-sections;
- dimension or angle-arc drawings.

Training role:

- manually corrected hard examples;
- Puhachov fine-tuning;
- active learning;
- targeted augmentation;
- regression testing.

### C. Invalid or out-of-domain negatives

Suggested file:

```text
pilotv3_negative_invalid.csv
```

Contains:

- photorealistic renders;
- plots;
- UI mock-ups;
- unsuitable block diagrams;
- text-heavy non-object content.

Training role:

- pre-filter training;
- post-vectorization gate training;
- out-of-distribution detection.

### D. Borderline calibration samples

Suggested file:

```text
pilotv3_borderline.csv
```

Contains valid but uncertain drawings.

Training role:

- threshold calibration;
- production warning calibration;
- manual-review policy design;
- false-positive analysis.

## Step 3: Add explicit failure labels to the audit manifest

Extend the audit CSV with fields such as:

```text
sample_id
is_valid_patent_drawing
stage2_should_accept
postvec_should_accept
failure_family
failure_subtype
severity
recommended_training_role
requires_manual_vector_correction
notes
```

Recommended failure subtypes:

```text
dense_detail
dashed_line_fragmentation
hatch_boundary_fragmentation
text_box_isolation
reference_numeral_leakage
dimension_arc_confusion
centerline_fragmentation
tile_seam_fragmentation
shaded_render
plot_or_chart
ui_mockup
block_diagram
borderline_valid
```

This converts the visual audit into reusable training metadata.

---

# 6. Targeted fixes for Group 2

## 6.1 Fix 1: dashed-line and centre-line grouping

### Problem

Dashed centre lines and circular bolt patterns may be represented as many independent fragments.

Each dash is geometrically correct, but the graph treats it as a separate component.

This artificially increases:

- component count;
- isolation score;
- fragment count;
- graph complexity.

### Proposed solution

Add a dashed-line grouping stage before final quality scoring.

Candidate dash segments should be grouped when they have:

- compatible orientation or curvature;
- similar stroke widths;
- regular gap lengths;
- similar dash lengths;
- small lateral displacement;
- support from a common line, circle, or arc model.

For circular centre lines, fit a circle through the dash centres instead of fitting each dash independently.

### Desired vector representation

```json
{
  "type": "circle",
  "layer": "object_center",
  "stroke": {
    "dash_pattern": [8.0, 5.0]
  }
}
```

The gaps should remain gaps in the rendered output.

The segments should be logically grouped, not physically connected by solid strokes.

### Acceptance test

On the bolt-circle regression sample:

- the number of unrelated components should decrease;
- the circular centre line should be represented as one logical primitive;
- nearby unrelated dashes must not be merged.

## 6.2 Fix 2: separate hatching from section boundaries

### Problem

Hatching creates many short parallel segments.

These segments inflate:

- isolated primitive count;
- fragment count;
- connected-component count;
- post-vectorization quality penalties.

A correctly vectorized cross-section may therefore look like a failed fragmented graph.

### Proposed solution

Detect hatching using:

- repeated parallel orientation;
- regular spacing;
- similar short segment lengths;
- clipping inside a common closed region;
- one or two dominant hatch angles;
- proximity to a section boundary.

Separate:

```text
section outline
    -> object_section_boundary

internal repeated strokes
    -> hatch
```

### Quality-gate modification

The object-fragmentation score should either:

- exclude hatch primitives; or
- calculate an independent expected hatch-fragmentation score.

Hatch segmentation quality should be evaluated separately from object topology.

### Acceptance test

On hatched mechanical drawings:

- hatch strokes remain available in the SVG;
- the section boundary remains complete;
- hatch lines do not increase object-fragmentation failure;
- hatching does not merge into visible object contours.

## 6.3 Fix 3: handle text boxes and boxed labels separately

### Problem

A boxed label may contain:

- one valid rectangular contour;
- multiple glyph fragments;
- text touching the rectangle;
- isolated small components inside the box.

The quality gate may interpret the interior glyphs as object fragments.

### Proposed solution

Add a `text_box` or annotation-region detector.

Suggested logic:

1. detect a closed rectangular contour;
2. inspect OCR or glyph evidence inside the rectangle;
3. classify the rectangle as annotation or diagram structure;
4. assign interior glyphs to the text layer;
5. exclude interior text fragmentation from the object-quality score.

### Important limitation

Do not automatically delete every rectangle containing text.

In flowcharts or diagrams, the box may be semantically meaningful.

The box should be preserved in a separate annotation or diagram layer.

### Acceptance test

For boxed-label samples:

- text is separated from object geometry;
- the box is preserved in the appropriate semantic layer;
- glyph isolation does not cause object-fragmentation rejection.

## 6.4 Fix 4: dimension and angle-arc separation

### Problem

Dimension lines, angle arcs, and radius indicators are valid geometric strokes but are not physical object boundaries.

The vectorizer may incorrectly assign them to object geometry.

### Proposed semantic classes

Add explicit support for:

- linear dimension;
- radial dimension;
- angular dimension;
- extension line;
- dimension arrowhead;
- measurement text.

### Detection signals

Use:

- arrowheads at one or both ends;
- nearby numeric text;
- arcs centred near a vertex;
- extension lines from object boundaries;
- typical dimension-line widths;
- repeated drafting conventions.

### Output policy

These primitives may remain in the final SVG, but they must not enter the physical CAD object contour graph.

### Acceptance test

On the angle-arc regression sample:

- the physical object remains connected;
- the dimension arc is preserved separately;
- the dimension does not create a false object loop or component.

## 6.5 Fix 5: dense-detail adaptive cleanup

### Problem

PCB drawings and detailed 3D views contain many valid small structures.

A fixed global fragmentation threshold may reject them because they naturally have:

- many primitives;
- many small components;
- high junction density;
- high foreground density.

### Proposed solution

Use local-density-aware cleanup and quality thresholds.

For dense regions:

```text
if local_stroke_density is high:
    tolerate more short primitives
    require stronger evidence before deleting details
    use local graph connectivity
    adjust endpoint merging carefully
```

### Avoid a naive tolerance increase

Increasing all merge tolerances may incorrectly connect independent PCB traces or nearby mechanical edges.

Prefer:

- component-aware joining;
- orientation-aware merging;
- local scale estimation;
- semantic numeral masks;
- graph-based pruning;
- local topology scoring.

### Acceptance test

On dense PCB and detailed 3D examples:

- valid fine structures remain;
- reference numerals are removed;
- unrelated neighbouring traces remain separate;
- the global isolation score does not reject valid dense content.

---

# 7. Mine more examples of each failure family

The five Group 2 examples are enough to identify the problem but not enough to retrain a robust model.

Use the current pipeline to search the wider corpus.

## 7.1 Dashed-line candidates

Search for drawings with:

- many short collinear fragments;
- regular gap distributions;
- repeated line orientations;
- circular arrangements of short segments;
- high dashed-line probability.

## 7.2 Hatch candidates

Search for:

- strong repeated orientation;
- high density of short parallel lines;
- enclosed regions;
- one or two dominant hatch angles;
- cross-section labels.

## 7.3 Text-box candidates

Search for:

- closed rectangular contours;
- OCR boxes inside rectangles;
- high isolated-component counts inside the rectangle;
- labels touching borders.

## 7.4 Dense-detail candidates

Search for:

- high junction density;
- high primitive count;
- high local foreground density;
- many small connected components;
- complex PCB or mechanical structures.

## 7.5 Initial mining targets

Recommended initial targets:

- 200 to 500 automatically mined examples per failure family;
- 50 to 100 manually verified examples per family;
- 20 to 50 manually corrected vector labels per family.

These examples should then be complemented by synthetic PatentVec data.

---

# 8. Targeted synthetic PatentVec generation

Generate exact synthetic samples that reproduce the observed failure modes.

## 8.1 Dashed geometry

Generate:

- dashed straight centre lines;
- dashed circles;
- dashed arcs;
- irregular gaps;
- missing dashes;
- dashes interrupted by labels;
- mixed dash lengths;
- low-resolution dashed geometry.

## 8.2 Hatching

Generate:

- hatching touching section boundaries;
- cross-hatching;
- interrupted hatching;
- hatching behind component numerals;
- several hatch angles;
- dense and sparse spacing;
- partial fading;
- broken hatch strokes.

## 8.3 Text boxes

Generate:

- text inside rectangles;
- glyphs touching box boundaries;
- labels attached to leaders;
- nested boxes;
- small symbols inside boxes;
- partial OCR-like degradation.

## 8.4 Dimension and angle annotations

Generate:

- angle dimensions;
- radius dimensions;
- diameter dimensions;
- arc arrowheads;
- extension lines;
- overlapping dimensions and object contours;
- text touching dimension lines.

These samples retain exact semantic and primitive labels and are safer for supervised training than uncorrected pseudo-labels.

---

# 9. Correct training use of each group

| Dataset group | Correct role |
|---|---|
| Clean accepted drawings | Normal real positive training |
| Valid fragmented Group 2 | Hard positive training after correction |
| Invalid Group 1 | Gate and out-of-domain negative training |
| Borderline Group 3 | Threshold calibration |
| Targeted synthetic PatentVec | Exact primitive and semantic supervision |

## Important rule

Do not combine Group 1 and Group 2 under one generic negative label.

Group 1 contains invalid inputs.

Group 2 contains valid but difficult positive drawings.

Mixing them would teach the system to reject complex patent drawings instead of learning to vectorize them.

---

# 10. Puhachov training strategy

For the valid fragmented Group 2 samples, store:

- the original raster;
- the current failed pipeline output;
- manually corrected primitives;
- corrected junctions;
- corrected semantic layers;
- failure subtype.

This allows primitive-level hard-negative mining.

Example:

```text
Ground truth:
    one dashed circular centre line

Incorrect prediction:
    twenty-four unrelated short object segments
```

The model can learn that:

- the drawing is valid;
- the dashes belong to one logical centre-line structure;
- the fragments should not be interpreted as unrelated object components.

---

# 11. Controlled ablation plan

Evaluate every targeted fix independently on the frozen 100-image benchmark.

| Experiment | Dashed grouping | Hatch separation | Text-box handling | Dense-detail logic |
|---|---:|---:|---:|---:|
| Baseline v3 | No | Current | No | No |
| E1 | Yes | Current | No | No |
| E2 | No | Improved | No | No |
| E3 | No | Current | Yes | No |
| E4 | No | Current | No | Yes |
| E5 | Yes | Improved | Yes | Yes |

Track:

- Stage-2 accepted count;
- post-vectorization clean count;
- valid-drawing rejection rate;
- invalid-content acceptance rate;
- median primitive count;
- isolated primitive count;
- connected-component count;
- topology score;
- rendering agreement;
- runtime;
- memory usage.

Do not promote a change only because it increases the accepted count.

A promoted fix must not allow more invalid drawings to pass.

---

# 12. Build a larger stratified holdout

The current set lacks codes C and D and is dominated by F.

Create a second frozen evaluation set with 300 to 500 drawings.

It should include:

- F;
- A;
- C;
- D;
- clean design drawings;
- difficult utility drawings;
- invalid content;
- shaded figures;
- cross-sections;
- diagrams;
- dense mechanical figures;
- PCB drawings;
- multiple panels;
- low-quality scans.

Create two reports:

## Natural-distribution report

Preserves the real corpus class proportions.

## Balanced hard-set report

Contains equal or deliberate representation of rare and difficult styles.

Both are necessary.

A high score on abundant easy F drawings must not hide poor performance on rare C/D or hard utility figures.

---

# 13. Training-corpus gate versus production gate

Use different threshold policies.

## 13.1 Training-corpus selection

```text
clean
    -> include

low_quality
    -> exclude or manually review

not_ok
    -> exclude
```

The priority is high label precision.

## 13.2 Production inference

```text
high confidence
    -> return vectors

medium confidence
    -> return vectors with warning

low confidence
    -> reject or request review
```

The priority is avoiding unnecessary rejection while communicating uncertainty.

Group 3 samples should be used to calibrate the boundary between medium and low confidence.

---

# 14. Recommended execution order

Proceed in the following order:

1. Commit the v3 configs, sample list, viewer builder, fixes, thresholds, and results.
2. Freeze the current 100 samples as a regression set.
3. Split flagged samples into invalid negatives, hard positives, and borderline cases.
4. Audit accepted samples to estimate the false-negative rate.
5. Add detailed failure labels to the audit CSV.
6. Implement dashed-line grouping.
7. Implement hatch and section-boundary separation.
8. Implement text-box handling.
9. Implement dimension and angle-arc separation.
10. Add dense-detail adaptive cleanup.
11. Mine more examples of each failure family.
12. Manually correct a small hard-positive set.
13. Generate targeted synthetic PatentVec examples.
14. Fine-tune Puhachov using exact, clean, and hard-positive data.
15. Run controlled ablations.
16. Build a larger C/D-inclusive stratified holdout.
17. Calibrate separate training and production thresholds.
18. Promote only changes that improve both robustness and selectivity.

---

# 15. Immediate implementation checklist

- [ ] Commit all pilot v3 configurations.
- [ ] Commit the viewer builder.
- [ ] Commit the exact sample list.
- [ ] Save the baseline report.
- [ ] Freeze the 100-sample regression manifest.
- [ ] Audit 20 random accepted samples.
- [ ] Audit 10 accepted samples near the quality threshold.
- [ ] Create `pilotv3_positive_clean.csv`.
- [ ] Create `pilotv3_positive_hard.csv`.
- [ ] Create `pilotv3_negative_invalid.csv`.
- [ ] Create `pilotv3_borderline.csv`.
- [ ] Extend the audit schema with failure labels.
- [ ] Implement dashed-line grouping.
- [ ] Add bolt-circle regression tests.
- [ ] Implement hatch semantic separation.
- [ ] Add cross-section regression tests.
- [ ] Implement text-box detection.
- [ ] Add boxed-label regression tests.
- [ ] Implement dimension-arc classification.
- [ ] Add dimension regression tests.
- [ ] Add local-density-aware quality scoring.
- [ ] Mine at least 200 candidates per failure family.
- [ ] Manually verify 50 to 100 per failure family.
- [ ] Correct 20 to 50 vector labels per failure family.
- [ ] Generate targeted synthetic PatentVec samples.
- [ ] Run Puhachov fine-tuning.
- [ ] Run the ablation matrix.
- [ ] Build a 300 to 500 sample stratified holdout.
- [ ] Document training and production gate thresholds.

---

# 16. Final conclusion

The pilot v3 results demonstrate that corpus filtering is no longer the dominant problem.

The stale-manifest and obvious-junk issues have largely been resolved.

The remaining errors are concentrated in identifiable technical drawing styles:

- dashed and centre lines;
- hatching near boundaries;
- text inside boxes;
- dimension and angle arcs;
- dense mechanical and PCB details.

This is a positive result because the remaining weaknesses can now be addressed with targeted geometry logic, semantic separation, synthetic data, and manually corrected hard positives.

The most important data-design rule is:

> Invalid drawings should become negative examples for the quality gate, while valid fragmented drawings should become hard positive examples for the vectorizer.

Following this distinction will improve the pipeline without teaching it to reject the complex patent drawings it is intended to support.
