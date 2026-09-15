# Stage-2/3 Quality Gate Recalibration

Implemented and replayed 2026-09-15 on the frozen 100-patent disjoint cohort.
Research configuration `config_deploy_recal.yaml`; canonical `config_deploy.yaml`
is unchanged pending review.

## Problem

After the September source-coverage recovery, Stage 2 preserves far more source
ink: median graph edges rose from 125 to 577 on the same worklist. The quality
gates were calibrated for the earlier, lossier graphs and were never re-derived.
End-to-end completions fell from **91 to 23 of 100** while reconstruction itself
improved on ground-truthed CAD data.

The question this work answers is not "how do we admit more figures" but
**"do these gate metrics predict bad reconstruction?"**

## Method

For all 100 cohort figures, gate metrics were paired with an independent
quality measure: Stage-3 raster fidelity against the Stage-1 skeleton
(precision/recall/F1 at 2 px, symmetric Chamfer), computed by
`tools.evaluate_patent_fidelity`.

Gated figures normally have no Stage-3/4 output. Two sources made them
measurable without re-running anything:

- Stage 3 writes primitives **before** the batch gate is applied, so all 46
  Stage-3-gated figures already had primitives.
- The 2026-09-14 review generated labelled diagnostic outputs for figures gated
  earlier, covering 27 of the 28 Stage-2 gates.

**96 of 100 figures therefore had both gate metrics and a fidelity score**,
including the entire rejected population. The remaining 4 (3 Stage-1 gates, 1
Stage-2 gate with no diagnostic trace) are correctly unevaluable.

## Finding 1: the gates were not separating good from bad

| Group | n | Median F1 | p10 | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| `ok` | 23 | 0.9994 | 0.9978 | 0.9989 | 1.0000 |
| Stage-3 gated | 46 | **0.9982** | 0.9691 | 0.9965 | 0.9999 |
| Stage-2 gated | 27 | 0.9822 | 0.9431 | 0.9650 | 1.0000 |

The 46 Stage-3 rejections reconstruct essentially as well as the 23 that passed.
Across all 96 evaluable figures the **minimum Stage-3 F1 is 0.9333** and nothing
falls below 0.90. The worst figure the gates *rejected* (0.9333) is comparable to
the worst they *accepted* (0.9479). There was no meaningfully bad population to
separate.

## Finding 2: the primary Stage-2 trigger has no predictive value

Spearman correlation against Stage-3 F1, n = 96 (69 where Stage-3 fields exist):

| Metric | rho | p |
|---|---:|---:|
| **micro_edge_ratio** | **-0.019** | **0.85** |
| isolation | **-0.624** | 1.1e-11 |
| short_edge_ratio | -0.571 | 1.2e-09 |
| n_edges | -0.500 | 2.2e-07 |
| mean_conf | -0.523 | 3.9e-06 |
| low_conf_ratio | **+0.596** | 6.6e-08 |
| n_primitives | -0.373 | 0.0016 |

`micro_edge_ratio` is the dominant Stage-2 trigger and is uncorrelated with
reconstruction quality. It counts sub-6 px edges, which coverage recovery
necessarily increases; it measures density, not defect.

## Finding 3: the Stage-3 confidence conditions are inverted

`mean_conf` correlates **negatively** with fidelity and `low_conf_ratio`
**positively**. The gate rejects the wrong side of both.

The mechanism is confounding, not a bug in confidence itself: confidence tracks
drawing simplicity. Simple drawings reach high confidence trivially; complex
drawings generate more low-confidence primitives *and* are reconstructed more
completely. As a population-level decision rule the gate is actively harmful.

## Finding 4: the primitive budget was not a runtime guard

`n_primitives` drives Stage-3 time (rho +0.873), but the absolute cost is small:
Stage-3 median 4.5 s, p90 13.7 s, max 30.8 s, against a 215 s median total
pipeline dominated by Stage-0 OCR. `budget_primitives` had median 920 against a
fixed cap of 900, so roughly half the cohort tripped it immediately.

## Changes

`config_deploy_recal.yaml`, derived from the exact baseline configuration.
Configuration only; no pipeline algorithm, model, or threshold code was edited.

| Setting | Baseline | Recalibrated | Basis |
|---|---:|---:|---|
| `stage2.fragmentation.max_micro_edge_ratio` | 0.50 | 1.01 (never fires) | rho -0.019 |
| `stage2.fragmentation.max_short_edge_ratio` | 0.80 | 1.01 (never fires) | confounded with density |
| `pipeline.quality_gates.max_primitives` | 900 | 50000 | safety backstop; ~1.6x observed max 30,426 |
| `pipeline.quality_gates.max_low_conf_ratio` | 0.25 | 0 (disabled) | correlation inverted |
| `...max_low_conf_ratio_after_hachure` | 0.65 | 0 (disabled) | correlation inverted |
| `stage2.isolation_threshold` | 0.30 | **unchanged** | strongest correctly-signed predictor |
| `stage2.fragmentation.max_edges` | 2500 | **unchanged** | explosion guard |
| `...max_unclaimed_noncycle_pixels` | 100000 | **unchanged** | pathological-residual guard |

The demoted metrics are still recorded in SQLite and in the graph JSON, so they
remain available as warnings and to the acceptance layer.

## Replay result

Stage 0/1 artifacts were reused from the baseline run
(`--reuse-preprocessing-from`), so only Stage 2 onward re-executed.

| Status | Baseline | Recalibrated |
|---|---:|---:|
| `ok` | 23 | **81** |
| `quality_gate_stage1` | 3 | 3 |
| `quality_gate_stage2` | 28 | **16** |
| `quality_gate_stage3` | 46 | **0** |

The 16 remaining Stage-2 gates are exactly the predicted split: 6 held by
`isolation`, 10 by `max_edges` — the two conditions deliberately retained.

### Fidelity after recalibration

| Population | n | min F1 | p10 | median |
|---|---:|---:|---:|---:|
| Admitted, recalibrated | 81 | 0.9421 | 0.9798 | 0.9988 |
| — newly admitted | 58 | 0.9421 | 0.9714 | 0.9981 |
| — previously admitted | 23 | 0.9479 | - | 0.9994 |

3.5x more admitted figures for a 0.0058 reduction in worst-case fidelity.

### Non-regression

| Check | Result |
|---|---|
| Baseline-`ok` figures still `ok` | 23/23 |
| Gained any geometry defect code | 0 |
| Geometry defect codes present, before -> after | 18/23 -> 18/23 |
| Reason-code differences beyond `deployment_unverified` | 0 |

`deployment_unverified` appears only because this is a declared research
configuration (`strict_models: false`); it is not a behavioral change.

### Runtime

| | Baseline | Recalibrated |
|---|---:|---:|
| Stage-3 median | 4.5 s | 4.3 s |
| Stage-3 p90 | 13.7 s | 13.6 s |
| Stage-3 max | 30.8 s | 31.3 s |

Admitting 58 additional figures, many with several thousand primitives, did not
move the runtime distribution.

## Pre-existing defect exposed, not introduced

`geometry_skeleton_coverage_lost` is reported on 56 of the 81 admitted figures.
This is **not** a consequence of recalibration: it was already present on 18 of
the 23 figures that passed in the baseline, and 77 baseline figures carried
`geometry_validation_missing` because they were gated before Stage 4, so the
geometry validator never ran on them.

The old gates were concealing a corpus-wide coverage issue on 77% of the cohort
by rejecting figures before it could be measured. This deserves separate
investigation and is the most consequential side effect of this work.

## Limits

- **Fidelity here is self-consistency, not ground truth.** Stage-3 primitives are
  compared against the Stage-1 skeleton. A skeleton that already misrepresents
  the drawing — merged hatch, retained reference text — still scores well. These
  numbers are not absolute real-patent accuracy.
- **No figure became training-eligible.** All 100 rows remain
  `training_eligible: 0`; acceptance moved from 94 rejected / 6 review to 75
  rejected / 25 review, with 0 accepted. Content routing is still unimplemented
  and remains the blocker.
- **The thresholds were not tuned to maximize admissions.** Two hard conditions
  that still reject 16 figures were deliberately retained because they are safety
  guards rather than quality metrics.
- **`max_edges` is unresolved.** It holds 10 figures, `n_edges` correlates with
  fidelity at rho -0.500, and that correlation is confounded with complexity.
  Deciding it needs ground truth, not this self-consistency measure.
- **CAD250 cannot validate this change.** Its `micro_edge_ratio` is identically
  0.000 at median and p90, so it cannot exercise the conditions being changed. It
  remains useful only as a ground-truth guard against geometrically wrong output.

## Recommended next steps

1. Review and, if accepted, port these thresholds into `config_deploy.yaml` with
   an explicit policy version, keeping the demoted metrics as recorded warnings.
2. Investigate `geometry_skeleton_coverage_lost` on the now-visible population.
3. Resolve `max_edges` against CAD ground truth rather than self-consistency.
4. Implement content routing; throughput is no longer the constraint on building
   a training corpus.

## Evidence

```text
docs/audits/2026-09-15/gate_recalibration_evidence.json
docs/audits/2026-09-15/gate_metrics.csv              per-figure gate metrics + outcome
docs/audits/2026-09-15/fidelity_baseline_cohort.csv  baseline + diagnostic fidelity
docs/audits/2026-09-15/fidelity_recal.csv            post-recalibration fidelity
docs/audits/2026-09-15/cohort_worklist.csv           exact 100-figure cohort
config_deploy_recal.yaml                             research configuration
output/PatentData100_GateRecal_2026-09-15/           replay run
output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14/  baseline run
```
