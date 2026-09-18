# Roadmap to a Deployable Vectorization Pipeline

Status: proposal, 2026-09-18. Supersedes the "next steps" sections of
`PATENTVEC_PILOT_V3_RESULTS_NEXT_STEPS.md`, `GATE_RECALIBRATION_2026-09-15.md`
and `CONTENT_ROUTING_2026-09-17.md` where they overlap.

Every number in this document was measured on the curated 100-figure cohort
(`benchmarks/curatedv1/`) unless marked **hypothesis**. Where a step depends on
something not yet measured, the step says so and names the measurement that
must come first.

---

## 0. Definition of done

"Deployable" means all of the following hold at once. Each is a number so
that the question "are we there" has a yes/no answer.

| # | criterion | today | target |
| --- | --- | --- | --- |
| D1 | training-eligible share of *executed* figures on the curated 100 | 5 / 85 (6%) | ≥ 75% |
| D2 | wrong-text rate of accepted reference labels, on a 200-label human audit | unmeasured (probe: ~35% of OCR labels absent from the description) | ≤ 2% |
| D3 | primitive recall and precision against **ground truth**, not self-consistency | unmeasured | established in Phase C; then no regression |
| D4 | wall time per executed figure | 270 s | ≤ 120 s |
| D5 | content decisions require a human per figure | yes | no — human only on low-margin predictions |
| D6 | full-corpus run completes resumably and every accepted record verifies under `eligibility_reason` | never attempted | yes |
| D7 | the canonical config states what it does (no disguised-off gates) | yes, since `bf7e4d75d2` | keep |

The targets in D1 and D4 are proposals. They are set from what Phase A and
Phase B are expected to deliver, not from an external requirement, and should
be revised once those phases report.

---

## 1. Baseline, 2026-09-18

Cohort: 100 human-curated engineering drawings, 100 distinct patents, all
labelled `in_scope` in `benchmarks/curatedv1/content_decisions.json`.

| stage | state | evidence |
| --- | --- | --- |
| execution | 85 `ok`; 13 gated at Stage 2, 1 at Stage 1, 1 at Stage 3 | `output/PatentDataCurated100_GeomFix_2026-09-18/results.db` |
| content | passes on all 85 executed | `docs/audits/2026-09-17/content_routing_evidence.json` |
| deployment | passes on 100 | same |
| geometry | 82 pass / 2 fail / 1 review | `docs/audits/2026-09-18/geometry_speckle_evidence.json` |
| references | **80 review** (`unknown_reference_text`) | `docs/audits/2026-09-18/reference_recovery_evidence.json` |
| accepted | **5** | — |
| accepted if references cleared | **82** | counterfactual on sealed records |
| per-figure time | 270 s: Stage 0 181 s, Stage 2 80 s, Stage 3 5 s, Stage 4 1 s | batch summary |

Reference labels on the cohort:

| population | count | what they are |
| --- | --- | --- |
| OCR-read numerals | 1218 | 65.5% appear in the patent's description; 34.5% do not, and the disagreements are visibly clipped or misread glyphs |
| `cc_recovered`, no text | 1175 | ~77% real glyphs, ~23% geometry, ~15% fragments of `FIG.` captions (48-crop sample) |

Gates: of nine audited, two are live (`max_edges`, `isolation_threshold`),
four are explicitly `null`, three are dormant. `max_edges` correlates 0.74
with drawing size; edge density correlates −0.02
(`docs/audits/2026-09-18/threshold_shape_audit.json`).

---

## 2. Principles

These come from what failed this month, not from preference.

1. **Measure the distribution before setting a rule.** Every threshold that
   was fixed this month — Stage-2/3 gates, geometry `max`, residual pixels —
   had been condemning the wrong thing, and each was found the same way:
   plot what the rule bounds across the cohort, then look at what it
   actually rejects.
2. **Test a discriminator against a ground-truth negative class.** Four of
   seven mechanisms proposed for the reference blocker failed this test
   (collinear runs, circularity, recogniser agreement, and my own crop
   estimate). Elongation passed. Nothing ships on a montage.
3. **Fail closed, and say what is known.** A record must never assert more
   than the evidence supports. An unread mark is not a reference; a
   disabled gate is `null`, not `0`.
4. **Bump the version when the rule changes.** Validator and policy
   versions exist so that old records do not silently verify under new
   rules. Every phase below that changes a rule bumps one.
5. **Every phase ends with a rerun on the curated 100 and a counterfactual
   on the sealed records.** That is how each phase reports what it bought.

---

## 3. Phases

Dependency order is A → B → C → D → E → F, with B and D independent of A.
Sections 4 and 5 give the graph and the decisions that only the owner can
make.

### Phase A — Clear the reference blocker

Goal: D1 from 6% toward 75%; D2 measured and brought under 2%.

The blocker is one contradiction: the recovery pass removes marks *because*
OCR could not read them, and the contract requires every removed mark to
carry text. Phase A resolves it from both sides — read more of them
correctly using a source of truth the pipeline never consulted, and stop
claiming reference status for the rest.

#### A1. Extract the reference vocabulary from each patent's description

**What.** A tool `tools/reference_vocabulary.py` that reads
`data/PatentData/ReorganisedData/<patent>/<patent>.xml` and emits
`<patent>_vocabulary.json`: every reference numeral the description or
claims name, with its surrounding noun phrase(s) and mention count.

**Why.** The description is a closed list of legitimate numerals per patent.
A probe with a naive regex found 57 tokens for EP1467415B1 and 65.5%
agreement with OCR across 98 figures; the 34.5% disagreements were visibly
wrong readings (`8, 8, 8, 8` in a 100-series patent).

**How.**
- Strip XML; keep description and claims separately.
- Patterns: `<noun> <num>[suffix]` (`housing 104`, `valve 14a`, `arm 14'`),
  parenthesised `(104)`, enumerations `104, 106 and 108`. Suffixes: one
  lowercase letter, prime, double prime.
- Reject known false positives: `claim N`, `Fig. N`, `step N`, years,
  percentages, units, `N mm`, and tokens mentioned once with no noun.
- Record `series` (the numbering style — `1x`, `10x`, `100x` — detectable
  from the token set) so that a read of `8` in a 100-series patent can be
  scored as improbable.

**Accept when.** On 20 hand-checked patents, ≥ 95% of numerals visible in
the drawings appear in the extracted vocabulary, and ≤ 2% of extracted
tokens are not reference numerals. This is a human check; it is the
calibration for everything downstream and is not optional.

**Cost.** 1–2 days. **Risk.** Descriptions that omit a numeral present in a
drawing; handled by A2 mapping absence to `review`, not rejection.

#### A2. Gate OCR-read labels on the vocabulary

**What.** In `stage0_handling_references/stage0_handle_references.py`,
`_ocr_pass()` and `_classify_reference_token()` consult the vocabulary when
one exists. A read numeral not in the vocabulary is not discarded — it is
kept as a removal candidate but marked `vocabulary: "absent"`. In
`tools/acceptance.py::_check_references`, an accepted reference whose text
is absent from the vocabulary raises a new reason code
`reference_not_in_description` → `review`.

**Why.** This is the wrong-text blind spot. The contract catches missing text
and nothing else; 420 confidently-read labels on the cohort would be written
into DXF as references the patent never mentions.

**Accept when.** On the 200-label human audit for D2, labels the gate passes
have ≤ 2% wrong text, and the gate flags ≥ 80% of the wrong ones.

**Cost.** 1 day. **Dependency.** A1.

#### A3. Closed-set recognition on recovered crops

**What.** For every `cc_recovered` candidate that survives the elongation
gate, run the easyocr recogniser on the padded, upscaled crop with a
digits+letters allowlist and rotation trials, and **accept a reading only if
it is an exact vocabulary token**. Anything else stays unread.

**Why.** Open-set recognition on these crops was rejected this month because
the recogniser reads a dash as `1` at 0.97 and a small circle as `0` at
1.00. Closed-set changes the failure mode: a dash that reads `1` is accepted
only if `1` is a reference in *this* patent, and a 100-series patent
rejects every single-digit reading outright.

**How.** Reuse the ensemble from
`docs/audits/2026-09-18/reference_recovery_evidence.json` (five
preprocessing variants, best-confidence vote). Also try tokens at edit
distance 1 from the reading when the reading itself is absent and the
neighbour is in the vocabulary, but record that as `matched_by: "edit1"` and
give it lower confidence. Merge adjacent recovered components before
reading (the earlier grouping probe) — a `1` and a `2` side by side are
`12`, and only the grouped reading can match a multi-digit token.

**Accept when.** On a 150-crop hand-labelled sample drawn from the survivors:
precision of accepted readings ≥ 97%; recall is reported, not targeted. The
sample is labelled once and kept under `benchmarks/curatedv1/` as the
standing test set for this step.

**Cost.** 2–3 days including the labelled sample. **Dependency.** A1.
**Risk.** Multi-digit tokens split across components with a gap; mitigated
by grouping. Measured before shipping, not assumed.

#### A4. Record unidentified marks as what they are

**What.** A recovered component that A3 could not match becomes
`kind: "unidentified_mark"`. It is still removed and its crop still
reinjected — nothing about the artifact changes — but the record no longer
claims reference status. Changes:
- `stage0_handle_references.py`: emit the kind; carry it through
  `_build_reference_doc`.
- `stage4_export`: an unidentified mark is a complete annotation without a
  text label (`label_required: false`), so `s4_flagged` stops firing on it.
- `tools/acceptance.py::_check_references`: `unknown_reference_text` applies
  only to labels whose kind claims to be a reference. Add an informational
  count `unidentified_marks_removed` to the record.
- `POLICY_VERSION` → `2026-09-XX.1`.

**Why.** This is the policy decision (Section 5, decision 1). The contract
currently blocks 80 figures for not knowing something the pipeline was
never able to know. The crop preserves the mark; the honest record says it
was removed unread. Downstream, an unidentified mark contributes no
text↔drawing link and pollutes none.

**Accept when.** On the curated 100, `unknown_reference_text` fires only on
labels with a claimed class, and the training-eligible count on sealed
records moves from 5 toward the counterfactual 82. Report the actual number.

**Cost.** 1–2 days. **Dependency.** A3 first, so the population this applies
to is as small as it can be.

#### A5. Suppress recovery inside rejected caption boxes

**What.** `_ocr_pass()` currently discards tokens the blacklist rejects
(`FIG`, `Fig.`, `sheet`). Keep their boxes as suppression zones and pass
them to `_recover_missed_numerals` as `taken` regions.

**Why.** ~15% of remaining unread marks are letters of figure captions that
OCR read and the blacklist rejected; the decision is thrown away and the
component scan re-adds the letters one by one.

**Accept when.** Caption fragments in a 48-crop sample of survivors fall
below 3%. **Cost.** Half a day.

#### A6. Phase A validation

Rerun the curated 100 on the canonical config. Report: accepted count,
`unknown_reference_text` count, `reference_not_in_description` count, and
the D2 audit (200 accepted labels, read by a human against the crops).
Update `docs/audits/` and this document's baseline table.

---

### Phase B — Throughput

Goal: D4 from 270 s to ≤ 120 s. Independent of Phase A; can run in parallel.

#### B1. Move Stage 0 OCR to the GPUs

**What.** Set `stage0.ocr_gpu` to a pinned device string per worker
(`_get_ocr_reader` already accepts `"cuda:0"` and avoids the DataParallel
OOM). Split workers across the two RTX 4090s.

**Why.** Stage 0 is 67% of per-figure time and both GPUs sit idle during
runs. At 12 CPU workers the corpus (~78k figures at ~5 per patent) is
roughly 20 days; the batch log shows the model-loading path exists.

**Accept when.** Stage 0 mean time on the curated 100 ≤ 40 s, **and** the
reference label sets on 20 figures match the CPU run — same count, same
texts, same boxes within 2 px. OCR on GPU can differ numerically; the
identity records the device, and the comparison decides whether the
difference matters.

**Cost.** 1 day. **Risk.** Twelve workers sharing two GPUs; measure memory,
choose the worker count from it.

#### B2. Stage 2 tiling on GPU

**What.** `puhachov.device: cpu` in the canonical config; tiled inference
exists and was validated with 14× less memory. Try `cuda` for Stage 2 as
well.

**Accept when.** Stage 2 mean ≤ 25 s with keypoint sets on 20 figures
matching CPU within the tiling seam tolerance. **Cost.** Half a day.
**Hypothesis** — not yet measured whether GPU Stage 2 changes results.

---

### Phase C — Ground truth

Goal: D3 established. Every quality decision after this phase stops being
self-referential.

#### C1. Run the pipeline over the synthetic patent-like sheets

**What.** The compositor under `syntheticData/` emits sheets with exact
semantic ground truth (`docs/PATENT_LIKE_SYNTHESIS_ROADMAP.md`). Run the
canonical pipeline over ~500 sheets stratified across the complexity tiers
and both acquisition modes.

**Why.** Training on this data gave a null result on real patents
(`docs/audits/2026-09-17/training_control_evidence.json`). But its ground
truth is the one thing the real corpus lacks, and it is exactly what the
validators need. The investment was aimed at the wrong consumer.

#### C2. A true primitive-level precision/recall metric

**What.** `tools/gt_fidelity.py`: match fitted primitives to ground-truth
primitives by sampled-point Chamfer distance in the sheet's coordinate
frame; report precision, recall, and F1 per sheet and per primitive type.
Keep the existing raster F1 alongside so the two can be compared.

**Accept when.** On sheets where a known number of primitives was
deliberately dropped from the render, recall falls by the expected amount.
That is the metric validating itself before it validates anything else.

#### C3. Calibrate with ground truth what could not be calibrated without it

- **`max_edges`.** Replace with a scale-free rule. Candidates measured this
  month: edge density (edges per 1000 ink px, density corr with size −0.02)
  and median edge length in px (fragmented ≤ 7.0, healthy ≥ 8.2 on the 13
  gated figures — but that is 13 points, not a rule). Choose the rule by
  its ROC against C2 recall on synthetic sheets, then confirm on the 13
  real gated figures that the 6 healthy ones pass and the 7 fragmented
  ones fail.
- **`geometry_output_off_skeleton`.** Replace its raw `max` with the same
  run-length rule used for recall
  (`tools/geometry_validation.py::_distance_check`). One figure today; the
  identical latent defect. Validator version → 5.
- **The fidelity proxy.** Report the correlation between raster F1 and C2
  F1; if it is weak, stop quoting raster F1 as fidelity.
- **`max_low_conf_ratio`.** Now visibly off. Measure whether RANSAC
  confidence predicts C2 precision at all. If it does not, delete the key
  rather than leave a dead gate; if it does, set it from the ROC.

**Cost.** C1 half a day of compute; C2 2 days; C3 2–3 days.

---

### Phase D — Content authority at scale

Goal: D5. Independent of A–C.

#### D1. Train a drawing / not-drawing classifier on the 158 curated labels

**What.** Image embedding from a pretrained backbone plus a linear probe,
trained on `benchmarks/curatedv1/curation_decisions.csv` (100 in-scope, 58
out-of-scope with reasons). Report cross-validated accuracy and the
precision/recall trade-off as a function of decision margin.

**Why.** Content routing today needs a human manifest per figure. That does
not scale to 15,600 patents, and the corpus filter's `drawing` label has a
~29% false-positive rate (17 of 58 offered replacements were rejected).

**Accept when.** Cross-validated accuracy ≥ 95% on the 158, **and** on 200
fresh figures from `filter_manifest_v3` labelled by the curator with
`tools/curate_pilot.py`, the classifier's high-margin predictions are
≥ 98% correct. The 158 are the training set; the 200 are the test set and
must not overlap.

**Cost.** 2 days plus one curation session for the 200.

#### D2. Register the classifier as a manifest authority

**What.** `tools/build_content_manifest.py --authority content_classifier_v1`
emits decisions with `decided_by: <model sha256>` and a `margin` field.
`tools/content_routing.py::_class_decision`: an `in_scope` decision from a
non-human authority passes only when `margin ≥ τ`; otherwise → `review`
with `content_low_margin`. τ is the margin at which D1 measured ≥ 98%
precision.

#### D3. Human review only where the model is unsure

**What.** `tools/curate_pilot.py --seed-manifest <low-margin figures>` —
the existing tool already resumes, replaces, and records reasons. The
human's decisions feed back into D1's training set on the next cycle.

**Cost.** 1 day.

---

### Phase E — Hatch styles and fragmentation

Goal: reduce the 13 Stage-2 gates by fixing their most likely shared cause.

#### E1. Verify the link before building

**What.** For the 7 genuinely fragmented figures (median edge length
1.4–7 px, micro ratio 0.38–0.95), overlay the HatchUNet mask on the
micro-edge regions.

**Why.** Thousands of 2-px edges is the signature of hatching or stipple the
detector did not remove. The owner has separately observed misses on bold
and solid-black hatching. If these are the same defect, the deferred style
work is also the fragmentation fix. **Hypothesis** until E1 reports.

**Accept when.** A one-page note with the seven overlays and a yes/no per
figure. Half a day.

#### E2. If confirmed: extend HatchUNet's training styles

**What.** Label bold, solid, and cross-hatch styles with the existing
`tools/hatch_label.py`; add them to the 268-figure dataset; retrain with
the recipe in memory (`lr 3e-4`, `pw 3.0`); re-measure test IoU and the
Stage-2 gate rate on the curated 100.

**Accept when.** Test IoU does not drop below 0.816 on the original styles,
and at least 4 of the 7 fragmented figures execute `ok`. **Cost.** 3–5
days including labelling.

---

### Phase F — Corpus run and release

Goal: D6.

#### F1. Dry run on 1,000 figures

Stratified one-per-patent from `filter_manifest_v3`, content decided by
Phase D. Report acceptance rates, time per figure, and the distribution of
every reason code. Anything that fires on more than 10% of figures gets the
Section 2 treatment before F2.

#### F2. Sampling audit

Draw 100 accepted figures at random. A human checks SVG overlay against the
TIF and reads 200 reference labels. This is the D2 measurement at corpus
scale. Any systematic failure found here is a stop.

#### F3. Freeze and run

- Freeze `config_deploy.yaml`, record the three identity hashes and the
  four model hashes in `models/README.md`.
- Run the corpus sharded by patent with `--resume`. Every figure gets a
  sealed acceptance record; `tools/build_training_manifest.py` selects only
  rows where `eligibility_reason` is `None`.
- Write `docs/RELEASE_<date>.md`: identity hashes, counts by acceptance
  status and reason code, the F2 audit numbers, and the known limitations
  (Section 6).

---

## 4. Dependency graph

```
A1 vocabulary ──► A2 OCR gate ──┐
       │                        ├──► A6 validate ──┐
       └──► A3 closed-set ──► A4 contract ──┘        │
                    A5 captions ──────────────────────┤
                                                      │
B1 GPU OCR ──► B2 GPU Stage 2 ────────────────────────┤
                                                      ├──► F1 dry run ──► F2 audit ──► F3 release
C1 synthetic run ──► C2 GT metric ──► C3 calibrate ───┤
                                                      │
D1 classifier ──► D2 authority ──► D3 review loop ────┤
                                                      │
E1 verify link ──► E2 hatch styles ───────────────────┘
```

A, B, C, D can start in parallel. E1 is half a day and should happen early
because its answer changes how much E2 is worth. F waits for all of them.

Rough effort, one person: A 6–9 days, B 2, C 5–6, D 4 plus one curation
session, E 1–6 depending on E1, F 3–5 plus compute. About five to six
working weeks serially; three to four with two people.

---

## 5. Decisions only the owner can make

1. **Is an unidentified removed mark acceptable in a training example?**
   (A4.) The crop is preserved and reinjected; the record says it was
   removed unread; no text link is claimed. If the answer is no, Phase A
   stops at A3 and D1 is bounded by A3's recall.
2. **What is the D2 tolerance for wrong reference text?** 2% is proposed.
   For the text↔drawing goal, a wrong numeral fabricates a correspondence;
   the tolerance should be set by what that costs downstream.
3. **May a model be a content authority?** (D2.) The alternative is a human
   per figure at corpus scale.
4. **Is 75% eligibility the bar, or is it the corpus count that matters?**
   82/85 on the cohort would give roughly 60k eligible figures at corpus
   scale if the rate held, which it will not exactly. Decide whether the
   target is a rate or a number.

---

## 6. What this roadmap does not fix, and what is not known

- **~23% of recovered marks are geometry** (dashes were removed; small
  circles, arrowheads and leader fragments remain). A3 will not read them
  and A4 records them as unidentified. They are removed from the vector
  output and preserved only as crops. No mechanism tested this month
  separates them from glyphs; the ink-routing check cannot see
  mis-routing because removal is a declared channel.
- **Ground truth on real patents does not exist.** Phase C uses synthetic
  ground truth; the gap between synthetic and real is exactly what the
  training-control null result measured. C3's calibrations transfer to the
  extent that gap is small, and that extent is unknown.
- **The reference contract cannot distinguish a correct numeral from a
  wrong one that happens to be in the vocabulary.** A2 narrows the blind
  spot; it does not close it. F2's human audit is the only check.
- **The curated 100 is one cohort.** F1's thousand is the first look at
  whether its rates hold.
- **Every effort estimate above is a guess by the author**, whose
  proposals this month were right four times and wrong three.

---

## 7. Evidence index

| finding | file |
| --- | --- |
| content routing design and counterfactuals | `docs/CONTENT_ROUTING_2026-09-17.md`, `docs/audits/2026-09-17/content_routing_evidence.json` |
| reference recovery: what the untexted labels are, rejected mechanisms | `docs/audits/2026-09-18/reference_recovery_evidence.json`, crop montages alongside |
| geometry speckle and the run-length change | `docs/audits/2026-09-18/geometry_speckle_evidence.json`, `geometry_uncovered_clusters.json` |
| threshold shape audit | `docs/audits/2026-09-18/threshold_shape_audit.json` |
| gate semantics | `docs/audits/2026-09-18/gate_semantics_evidence.json` |
| gate recalibration | `docs/GATE_RECALIBRATION_2026-09-15.md` |
| synthetic training null result | `docs/audits/2026-09-17/training_control_evidence.json` |
| curated cohort and decisions | `benchmarks/curatedv1/` |
| production model hashes | `models/README.md` |
