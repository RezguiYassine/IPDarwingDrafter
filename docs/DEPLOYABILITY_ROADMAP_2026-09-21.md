# Roadmap to a Deployable Vectorization Pipeline — revision 2

Status: proposal, 2026-09-21. Supersedes `DEPLOYABILITY_ROADMAP_2026-09-18.md`,
which stays in place as the record of what was planned and what happened.

Every number here was measured; the run or audit file is named beside it.
Anything not yet measured is marked **hypothesis**. Where the 18.09 roadmap
made a claim that turned out wrong, this document says so rather than
quietly replacing it.

---

## 0. Decisions taken on 2026-09-21

These four answers from the owner reshape the plan. Each is paired with the
measurement that informed it.

| # | decision | informed by |
| --- | --- | --- |
| 1 | **The corpus is the mechanical subset, selected by IPC section B/F.** | 4,654 of 15,606 patents (29.8%). Drawing rate inside it ~82% against ~71% corpus-wide. It discards 31 good electrical drawings (sections G/H) in the curated set for every 45 it keeps; that trade is accepted. `docs/audits/2026-09-21/ipc_prior_evidence.json` |
| 2 | **Content authority: label ~1,000 figures inside the B/F subset, and use the IPC code as a prior — both.** | IPC alone reaches 81.8% precision, far from the 98% an authority needs. Two image classifiers on 158 labels reached 70.9% and 64.6%. Labels are the only path with measured odds; the prior narrows where they are spent. |
| 3 | **`max_edges` is disabled.** `max_primitives` (50,000) is the sole budget guard. | 106 real figures gated by edge count alone, run with the gate off: 91 execute, 76 pass geometry, and the 8,000–20,000-edge band runs 4× *faster* than the 2,500–4,000 band. The owner judged a seven-figure montage trainable. Recovers ~10.6% of the corpus. `docs/audits/2026-09-21/max_edges_evidence.json` |
| 4 | **The fidelity target is drawing geometry only.** Arrowheads, dimension lines, leaders and connectors leave the recall target. | Ground truth put annotation recall at 67–71% against 92–98% for drawing geometry. The downstream use learns shapes, not dimensioning. They stay reported per layer so the number cannot vanish. `docs/audits/2026-09-21/fidelity_target_evidence.json` |

An earlier decision stands: an out-of-vocabulary reference reading is
**demoted to an unidentified mark** rather than written as a numeral — a
missing label over a wrong one (2026-09-19).

---

## 1. Definition of done — revised

| # | criterion | 18.09 | **21.09** | target | state |
| --- | --- | --- | --- | --- | --- |
| D1 | eligible share of *executed* figures, curated 100 | 6% | **97%** (83/86) | ≥ 75% | **met** |
| D2 | wrong reference text written into exports | ~35%, undetected | **0 written**; 311 of 1,489 readings demoted | ≤ 2% | **met by demotion**, not by correction |
| D3 | fidelity against ground truth, drawing-only target | none | **F1 0.949, recall 0.957, precision 0.951** on 498 sheets | established | **met** |
| D4 | wall time per executed figure | 270 s | **33 s** curated, **15 s** median at scale | ≤ 120 s | **met** |
| D5 | content decisions need a human per figure | yes | yes | authority + review loop | **not met** |
| D6 | corpus run completes; every accepted record verifies | never tried | 1,000-figure dry run completes, **0 accepted** | yes | **not met** |
| D7 | canonical config states what it does | yes | yes | keep | met |
| **D8** | **corpus scope** | — | — | **B/F subset, ~4,654 patents, ~23k figures** | new, from decision 1 |

D5 and D6 share one cause: no content authority exists yet. Everything else
this roadmap adds is in service of that.

---

## 2. Baseline, 2026-09-21

**Curated 100** (`output/PhaseA_v4_2026-09-19`, canonical GPU config):

| | value |
| --- | --- |
| executed `ok` | 86 |
| accepted | **83** — every executed figure but three |
| references check | 86 pass, 0 `unknown_reference_text` |
| geometry check | 83 pass |
| reference labels | 925 identified, 311 demoted, 761 unidentified marks |

**14.09 review cohort, rerun** (`output/PipelineReview_21092026`, four-column
viewer at `viewer/index.html`): exports **23 → 81**, Stage-3 gating **46 → 0**,
Stage-2 gating 28 → 16, median time ~270 s → 20 s.

**Dry run, 1,000 uncurated figures** (`output/PhaseF_dryrun_2026-09-19`):
807 execute, median 15 s, p90 72 s, worst 1,061 s; 13 CUDA OOMs found and
fixed; `geometry_skeleton_coverage_lost` at 1.8% against 55% before the
run-length change; **0 accepted, all `content_manifest_absent`**.

**Ground truth** (`docs/audits/2026-09-19/gt_fidelity_500_drawing_only.json`):
recall 0.957 / precision 0.951 / F1 0.949, p10 F1 0.895. Per layer, still
reported: hatch 98.2%, object_visible 92.1%, arrowhead 69.4%, dimension
67.3%, leader 70.7%.

**Identity:** `config_deploy.yaml` `b58dc20d…`, `config_deploy_gpu.yaml`
`453246905…`, implementation `d90ad439…`, weights `e7fdb035…`.

---

## 3. What the 18.09 roadmap got wrong

Recorded because the corrections are as useful as the results.

- **"An image embedding plus a linear probe is enough" for content.** It was
  not: 70.9%. A mechanical cross-section and a block diagram are both thin
  black lines on white; the difference is semantic and 158 examples cannot
  teach it. Structural features did worse (64.6%).
- **Closed-set OCR was proposed as safe because the vocabulary constrains
  it.** A negative control showed 73.3% of pure line fragments matched a
  vocabulary token — the dictionary was accepting noise. A two-character
  minimum fixed it, and deleting the 149 fabrications changed acceptance by
  zero: the defect was invisible in the headline.
- **`max_edges` was to be calibrated on synthetic ground truth.** Synthesis
  tops out at 714 edges against a 2,500 gate. The circularity was broken by a
  gate-off run on real figures and the owner's eye instead.
- **E1's answer was overstated as "hatch-driven".** Four of seven fragmented
  figures carry real hatch; the two most fragmented carry none. And E1 never
  tested the bold-hatch styles the owner flagged — that remains unmeasured.
- **A1's acceptance test (95% recall, hand-checked on 20 patents) was not
  met.** 92.1% by an automated proxy; the hand check was never done.

---

## 4. Phases

Phases A–C and F1 are closed. The remaining work is re-lettered G–K so it
does not collide with the 18.09 numbering.

### Phase G — Content authority for the B/F subset  ·  *critical path*

Goal: D5 and D6. Nothing else on this roadmap produces a training example
until G ships.

#### G1. Build the B/F pool

**What.** `tools/ipc_filter.py`: read the IPC code from each patent's XML,
keep patents whose first-listed section is B or F, emit
`benchmarks/mechanical/pool.csv` with one `drawing`-labelled figure per
patent.

**Accept when.** ~4,650 patents; the 158 curated figures inside the pool
reproduce the measured 81.8% drawing rate within ±5 points.

**Cost.** Half a day. **Evidence.** Already measured on 158; the tool
generalises it.

#### G2. Label ~1,000 figures inside the pool  ·  *owner*

**What.** `tools/curate_pilot.py --seed-manifest benchmarks/mechanical/pool.csv`.
The tool resumes, replaces rejects, and records reasons. Two sessions of
roughly an hour.

**Why 1,000.** The 158-label classifiers failed because the seven negative
classes had 2–21 examples each. At ~1,000 inside a pool that is ~82%
drawings, the ~180 rejects give each class 10–60 examples — enough for a
linear probe to learn from rather than memorise.

**Accept when.** ≥ 1,000 decisions with reasons, ≥ 150 rejects. Stored under
`benchmarks/mechanical/`.

#### G3. Retrain the classifier, measure the margin curve

**What.** `tools/content_classifier.py --train` on the ~1,150 labels
(158 + G2). Report cross-validated accuracy and precision-at-margin.

**Accept when.** On held-out folds, a margin τ exists at which precision on
`in_scope` predictions is **≥ 98%** with coverage ≥ 50%. If it does not
exist, G stops here and content stays human-gated — which under decision 1
is still a viable release (Section 5).

**Cost.** One day. **Hypothesis** — the 98% target is what D2 needs; whether
1,150 labels reach it is not known until measured.

#### G4. Register the authority; route the rest to a person

**What.** `tools/build_content_manifest.py --authority content_classifier_v2`
writes decisions with `decided_by: <model sha256>` and `margin`.
`content_routing._class_decision` accepts a non-human `in_scope` only when
`margin ≥ τ`; otherwise `review` with `content_low_margin`. Low-margin
figures feed `curate_pilot.py`, whose decisions return to G3's training set.

**Accept when.** On the F1 dry run's B/F figures, the classifier decides
≥ 50% at ≥ 98% precision against a 200-figure human check; the remainder
route to review, none to acceptance.

**Cost.** One day plus a 200-figure check by the owner.

### Phase H — References: close what A left open

Not blocking. A carries the corpus; H tightens it.

- **H1. The A1 hand check.** 20 patents, numerals in the drawing against the
  extracted vocabulary. Never done; the roadmap promised it. An hour.
- **H2. Demotion rate at scale.** Report `labels_demoted_unreadable` over the
  1,000-figure dry run. If it exceeds the curated cohort's 21% by a wide
  margin, the vocabulary extractor is missing series or suffix forms on
  some patent families and H1's findings say which.
- **H3. Caption suppression check.** 48-crop sample of surviving
  unidentified marks; caption fragments should now be below 3%. Half a day.

### Phase I — Hatch: measure the thing that was flagged

The owner observed on 2026-09-17 that bold and solid-black hatching goes
undetected. E1 did not test this. Phase C's 98.2% hatch recall is on
synthetic sheets drawn at a single thin stroke width and says nothing about
it.

- **I1.** Label 30–50 real figures the owner believes are hatched, across
  styles, with the existing `tools/hatch_label.py`. Run HatchUNet; report
  recall **by style**. One to two hours, mostly the owner's.
- **I2.** If bold-style recall is low: extend the training set with those
  styles and retrain (`lr 3e-4`, `pw 3.0`), accepting only if test IoU on
  the original styles stays ≥ 0.816. Three to five days. **Hypothesis** until
  I1 reports.
- **I3.** Separately, four gated figures had hatch *detected* at 47–51% and
  still fragmented — a removal failure, not detection. Half a day to look.

### Phase J — Validators: finish the audit's list

- **J1.** `geometry_output_off_skeleton` still uses the raw `max` against
  8 px — the identical defect removed from the recall side. One figure
  today; replace with the run-length rule, validator version → 5. Half a day.
- **J2.** Remeasure the self-consistency gap under the drawing-only target.
  It was 9 points (98.9% claimed vs 89.8% true); with annotation layers out
  of the recall set it should shrink. If it does not, the gap is in drawing
  geometry and matters. Half a day.
- **J3.** `max_low_conf_ratio` correlates 0.02 with true F1. Delete the key
  from the canonical config rather than leave a dead gate readers may
  re-enable. Ten minutes plus the identity bump.

### Phase K — Corpus run and release, on the B/F subset

- **K1. Scope.** G1's pool, ~4,654 patents, ~23,000 figures. At 15 s median
  and 10 workers that is roughly 10 hours of wall clock; the p90 tail makes
  it a day.
- **K2. Human audit (F2).** 100 accepted figures at random: SVG over TIF,
  plus 200 reference labels read against the crops. The D2 measurement at
  scale. Any systematic failure is a stop.
- **K3. Freeze and run (F3).** Freeze `config_deploy_gpu.yaml` as the
  deployment config (it is what every run since Phase B has used; the CPU
  config remains the behavioural reference). Record identities in
  `models/README.md`. Run sharded by patent with `--resume`. Select training
  rows with `tools/build_training_manifest.py`. Write
  `docs/RELEASE_<date>.md` with counts, the K2 numbers, and Section 6.

---

## 5. Dependency graph

```
G1 pool ──► G2 labels (owner) ──► G3 classifier ──► G4 authority ──┐
                                        │                           │
                                        └─ if 98% not reached: ─────┤  human-gated release
                                                                    │  over G2's labels only
H1 ──► H2 ──► H3  (independent, non-blocking) ─────────────────────┤
I1 (owner) ──► I2 / I3  (independent) ──────────────────────────────┤
J1, J2, J3  (independent, small) ───────────────────────────────────┤
                                                                    ▼
                                             K1 corpus run ──► K2 audit (owner) ──► K3 release
```

**Two viable releases.** If G3 reaches 98%, K1 covers the whole B/F pool.
If it does not, K1 covers exactly the ~1,000 figures from G2 — roughly 800
training examples at the measured 83% acceptance — with no classifier at
all. Both are honest; the second is smaller.

Rough effort: G 3 days plus ~3 owner-hours · H 1.5 days · I 0.5–6 days
depending on I1 · J 1.5 days · K 2 days plus compute plus ~2 owner-hours.
Two to three working weeks serially. Owner time in total: roughly five hours,
all of it on decisions a model cannot make.

---

## 6. What this roadmap does not fix

- **The demoted 21%.** 311 of 1,489 readings on the curated cohort were
  discarded rather than corrected. They contribute no text↔drawing link.
  Five recognition mechanisms were tested on this population and four
  failed; no sixth is planned.
- **The ~23% of unidentified marks that are geometry.** Small circles,
  arrowheads and leader fragments the elongation gate cannot catch are
  removed and preserved only as crops. Ink routing cannot see mis-routing.
- **Annotation geometry at 67–71% recall.** Out of the target by decision 4,
  not fixed. Reported per layer in every fidelity run.
- **Electrical drawings.** Decision 1 excludes sections G and H, which held
  31 of the 100 curated drawings. A later revision could add a G/H pool with
  its own labels; nothing here prevents it.
- **Synthetic-to-real transfer.** Every ground-truth number is on synthetic
  sheets. The September training control measured that gap as a null for
  *training*; for *validation* it is unmeasured. K2 is the only real check.
- **Every effort estimate is a guess** by an author whose 18.09 estimates
  were right about A, B and F and wrong about D.

---

## 7. Evidence index

| finding | file |
| --- | --- |
| Phase A: reference blocker, four mechanisms, negative control | `docs/audits/2026-09-19/phase_a_evidence.json` |
| Phase B: GPU acceleration and the two bugs | `docs/audits/2026-09-19/gpu_acceleration_evidence.json` |
| Phase C: first ground truth, proxy gap, dead confidence gates | `docs/audits/2026-09-19/phase_c_evidence.json`, `gt_fidelity_500.json` |
| drawing-only fidelity target | `docs/audits/2026-09-19/gt_fidelity_500_drawing_only.json`, `docs/audits/2026-09-21/fidelity_target_evidence.json` |
| Phase D: both classifiers fail | `docs/audits/2026-09-19/phase_d_evidence.json` |
| Phase E1: partial hatch link | `docs/audits/2026-09-19/phase_e_evidence.json` |
| F1: 1,000-figure dry run, OOM | `docs/audits/2026-09-19/phase_f1_evidence.json` |
| `max_edges` gate-off run and decision | `docs/audits/2026-09-21/max_edges_evidence.json` |
| IPC prior measurement | `docs/audits/2026-09-21/ipc_prior_evidence.json` |
| four-column review, 14.09 cohort | `output/PipelineReview_21092026/viewer/index.html` |
| curated cohort and decisions | `benchmarks/curatedv1/` |
| threshold audit and gate semantics | `docs/audits/2026-09-18/` |
