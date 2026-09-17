# Content routing

Status: implemented. `tools/content_routing.py`, validator `content_routing`
version `1`, wired into `tools/acceptance.py` and `tools/batch_run.py`.

## What was pending, and why it could not be inferred

Since `ap3-acceptance-v1` shipped, every acceptance record has carried
`content_validation_pending`, and the contract said plainly that no CLI switch
would be provided to clear it. The reason is that execution success carries no
information about content:

> A chemical structure, a flowchart, a bar chart and a table all vectorize
> without error. Stage 2 finds strokes, Stage 3 fits primitives, Stage 4
> exports them, every quality gate passes, and the result is useless as a
> training target.

`status: ok`, RANSAC confidence, primitive counts and the old content filter
are all blind to this. The decision has to come from an authority outside the
pipeline.

The 2026-09-17 curation pass demonstrates the scale of the problem rather than
merely arguing it. Re-reviewing the frozen 100-figure pilot by hand:

| verdict | count |
| --- | --- |
| accepted as engineering drawings | 59 |
| rejected as out-of-scope content | 41 |

The rejects break down as 21 block diagrams, 13 plots or charts, 10
flowcharts, 4 chemical structures, 5 shaded renders or photographs, 2 tables
and 3 other. **Every benchmark this project has quoted over "the 100-figure
patent cohort" was computed over a cohort that was 41% non-drawings.**

A secondary finding: of the 58 replacements drawn from `filter_manifest_v3`
rows labelled `drawing`, 17 were also rejected. The corpus filter's `drawing`
label carries roughly a 29% false-positive rate.

## What the validator decides

Two independent questions, combined worst-first.

### 1. Figure class routing

Read from a decision manifest (`ap3-content-decisions-v1`) supplied by an
authority named in the manifest header. Each decision is bound to the SHA-256
of the exact image it was made about.

| condition | status | reason code |
| --- | --- | --- |
| no manifest supplied | `pending` | `content_manifest_absent` |
| figure absent from manifest | `pending` | `content_decision_missing` |
| manifest hash ≠ processed source hash | `review` | `content_source_changed` |
| `routing: out_of_scope` | `fail` | `content_out_of_scope`, `content_class_<reason>` |
| `routing: in_scope` | `pass` | — |

Binding to the source hash is what makes the decision non-transferable. If a
TIF is replaced, the decision does not silently apply to a different drawing;
it reverts to `review`.

### 2. Source ink routing

Ink leaves the source raster by exactly three declared routes: removed as a
reference numeral at Stage 0, removed as a hachure at Stage 2, or retained as
drawing ink in the Stage-1 output. Ink reaching none of them was dropped
silently, and the vector output is not a faithful record of the figure.

The check rasterises all three channels, dilates their union by 2px (the
channels are rendered at slightly different stroke widths than the source, so
an exact match would report every stroke edge as unrouted), subtracts, and
measures the residual.

**Measured on the 100-figure cohort: unrouted ink is identically zero on every
figure.** Median, p90 and maximum unrouted fraction are all 0.000000, and the
largest unrouted connected component on any figure is 0 pixels. The reason is
structural — Stage 1 routes to `passthrough_binary`, which preserves the
source raster exactly.

This check therefore has **no discriminative power over today's figures**. It
is a guard against a Stage-1 route that deletes content — SketchCleanNet is a
learned cleaner and can remove strokes — not a filter that is currently
separating good figures from bad. The thresholds
(`max_unrouted_fraction = 0.001`, `max_unrouted_component = 100`) are set at a
physical scale rather than a percentile of a distribution that is a point mass
at zero.

## Fail-closed properties

- Absent manifest, absent entry, stale hash, and uncomputable ink routing all
  resolve to `pending`. No code path turns an undecided figure into a pass.
- A malformed manifest raises rather than degrading into a permissive one:
  wrong schema, no declared authority, no decisions, duplicate entries,
  invalid routing values and decisions not bound to a source hash are all
  exceptions at load time, in the worker process, before any figure runs.
- `acceptance.eligibility_reason` re-reads the sealed content report and
  recomputes its check, so editing a decision after the record is sealed
  yields `acceptance_content_unverified` rather than passing unnoticed.
- Rejections are carried into the manifest as explicit `out_of_scope`
  decisions, which **fail** the content check. A curator's rejection is a
  decision, not an absence of one. Skips are omitted.

## Deployment identity

Adding content validation to the acceptance implementation moves
`implementation_sha256`:

```text
before  e930c1e7c91222475da9daa48b22d0cfc8716e5212f704e1be083c9021cbebb0
after   84be3c72255d5420b69e75e6e6aad66006f8a551dc314cdf78c03e8dd7f7d0e2
```

`effective_config_sha256` (`fe36dd63…`) and `weight_manifest_sha256`
(`e7fdb035…`) are unchanged. Acceptance records sealed under the old identity
no longer verify. That is the mechanism working as designed, not a regression:
the rules by which those records were produced have changed.

## Content routing is necessary but not sufficient

Measured over the 100 acceptance records of the gate-recalibration run, with
the counterfactual applied directly to the sealed checks:

> **If content passed on all 100 figures, 0 would become training-eligible.**

The acceptance contract requires nine checks. Content was one of five that
were not passing:

| blocking check | figures | root cause | state |
| --- | --- | --- | --- |
| `content` | 100 | no decision authority existed | **resolved here** |
| `deployment` | 100 | run used the research config, so no verified identity | resolved by running on canonical `config_deploy.yaml` |
| `references` / `exports` / `stage_flags` | 80 | `unknown_reference_text`: Stage 0 detects a numeral but OCR returns no text, so DXF cannot write the label and Stage 4 flags the figure | **open** |
| `geometry` | 56 | `geometry_skeleton_coverage_lost` | **open** |
| `geometry` | 75 | `geometry_stage2_residual_pending` | **open** |

`unknown_reference_text` is the highest-leverage remaining item: one root
cause blocks three checks on 80% of figures. `s4_flagged` is not an
independent defect — it is the same OCR failure surfacing as an incomplete DXF
annotation.

## Sizing the top blocker: reference OCR

Measured over the same run's Stage-0 reference documents:

| quantity | value |
| --- | --- |
| figures with reference labels | 99 |
| figures with at least one unreadable label | 98 |
| reference labels detected | 3084 |
| labels with no OCR text | 1843 (59.8%) |
| figures blocked by exactly one unreadable label | 4 |

The two populations are structurally different:

| | texted | untexted |
| --- | --- | --- |
| count | 1241 | 1843 |
| median bbox area | 4900 px | 972 px |
| median confidence | 0.994 | 0.500 (placeholder) |
| `ref_class` | numeral 1150, acronym 47, alnum 44 | `None` on all 1843 |
| with leader lines | 747 | 816 |

Visual inspection of a random sample of 32 crops from each population settles
what they are. **The untexted crops are overwhelmingly real text** — digits,
uppercase and lowercase letters, and alphanumerics — not drawing content. What
separates them from the texted population is legibility, not category:
rotated and sideways glyphs, broken or dotted strokes from raster degradation,
and lowercase letters, with a small minority of genuine leader-line and
arrowhead fragments mixed in.

Two consequences:

1. The fix is recognition, not detection. The regions are found correctly and
   removed; only the text string is unknown. Extending OCR to rotated,
   degraded and alphanumeric glyphs addresses the large majority of the 1843.
2. A minority of untexted detections are drawing fragments, and Stage 0
   removes them (`active_removal` is true on 99 of 99 figures). Ink routing
   cannot catch this: removal *is* one of the three declared channels, so
   mis-routed content is still routed content. Ink routing detects ink that
   reaches no channel, not ink that reaches the wrong one.

## Artifacts

```text
tools/content_routing.py                     validator
tools/build_content_manifest.py              curation session -> decision manifest
tools/curate_pilot.py                        interactive curation with replacement
tests/test_content_routing.py                17 tests
benchmarks/curatedv1/content_decisions.json  158 decisions, sha256 587091aa0271475e…
benchmarks/curatedv1/curation_decisions.csv  the human decision log
benchmarks/curatedv1/curated_manifest.csv    the 100 accepted figures
benchmarks/curatedv1/worklist_100.csv        run cohort
```

Usage:

```bash
python -m tools.curate_pilot --session output/PatentData/curation_v2
python -m tools.build_content_manifest \
    --session output/PatentData/curation_v2 \
    --output benchmarks/curatedv1/content_decisions.json \
    --decided-by "<name>"
python -m tools.batch_run --config config_deploy.yaml \
    --worklist benchmarks/curatedv1/worklist_100.csv \
    --content-manifest benchmarks/curatedv1/content_decisions.json \
    --output output/PatentDataCurated100_2026-09-17
```

Without `--content-manifest` the content check stays `pending` and nothing is
training-eligible — the previous behaviour, unchanged.
