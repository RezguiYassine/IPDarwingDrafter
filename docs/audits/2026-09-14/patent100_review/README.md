# Current Patent100 Visual Review

Batch completed on 2026-09-14. Final artifact and browser verification completed
on 2026-09-15. This run evaluates the existing pipeline; no pipeline algorithms,
models, quality thresholds, or acceptance rules were changed for this review.

[Open the offline comparison gallery](../../../../output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14/viewer/index.html).

## Results

All 100 original worklist entries were processed, without execution crashes.

| Canonical execution outcome | Count |
| --- | ---: |
| SVG/DXF exported | 23 |
| Stopped at Stage 1 quality gate | 3 |
| Stopped at Stage 2 quality gate | 28 |
| Stopped at Stage 3 quality gate | 46 |

Acceptance is independent of execution: **94 rejected, 6 review, 0 accepted**.
These results are not approved LLM training targets.

The gallery contains 100 original scans, 23 canonical vector outputs, 73
separately generated diagnostic SVG previews, one graph-only current preview,
and three current-output placeholders. Every diagnostic is explicitly labeled,
stored outside the canonical output folders, and marked non-training-eligible.
Continuing Stage 3/4 solely for a diagnostic does not change the canonical gate
decision or manufacture an accepted output.

The previous column preserves the 91 saved diagnostic outputs from
`output/PatentData91_CompactHatchFitV2`, the latest available cohort-sized saved
run before source-coverage recovery and connection integration. All 91 have a
current SVG comparison. Nine previous outputs were never saved and remain
explicitly missing; no replacement baseline was fabricated.

This is a current-pipeline versus saved-output visual review, not an isolated
Stage 2 ablation or a labeled-ground-truth benchmark. The previous run used
frozen intermediate graphs and its saved SVGs do not all include reinjected
reference crops. Current Stage 0/1 processing and reference overlays are part
of the current run. Historical SVGs were not retroactively altered to match.

## Missing Current SVGs

| Sample | Drawing | Reason |
| ---: | --- | --- |
| 46 | EP3131533B1/F0001 | Stage 1 gate; no completed Stage 2 graph |
| 78 | EP3494929A1/F0001 | Stage 1 gate; no completed Stage 2 graph |
| 97 | EP3502566A1/F0002 | Stage 2 gate; diagnostic trace budget exceeded |
| 99 | EP3503056A1/F0001 | Stage 1 gate; no completed Stage 2 graph |

Sample 97 contains a single 335,935-point trace among 5,137 main edges. Its
diagnostic fitting attempt was stopped after prolonged CPU processing, and
the unchanged Stage 2 graph remains available. The review renderer now bounds
uncached fitting at 50,000 points per trace. Its other review-only limits are
100,000 total strokes and 2,000,000 total point occurrences. Earlier dense
cases excluded by the initial 25,000-stroke diagnostic limit were retried
successfully when memory became available. None of these resource limits
changes the canonical pipeline.

The nine missing previous drawings are EP2005342B1/F0006, EP3117602B1/F0008,
EP3131533B1/F0001, EP3147970B1/F0001, EP3272234B1/F0001, EP3492751A1/F0002,
EP3494929A1/F0001, EP3502566A1/F0002, and EP3503056A1/F0001.

## Provenance

- Worklist: `output/PatentVecHatchStrokeTraining/patentdata100_disjoint_worklist.csv`.
- Worklist SHA-256: `8c44338a997d2d35d57fdc35f689538757f47ceee15cfe790adbc55b514d8d3a`.
- Current run: `output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14`.
- Config: `output/PatentData100_CurrentPipelineReview_2026-09-14/config_gpu1.yaml`.
- Canonical behavior is validated against `config_deploy.yaml`; Puhachov and
  hatch inference use `cuda:1`, while reference OCR remains on CPU.
- Implementation SHA-256: `e930c1e7c91222475da9daa48b22d0cfc8716e5212f704e1be083c9021cbebb0`.
- Effective config SHA-256: `5b71a6bc63543c56721961b6ff3d20a129f13fe7e9dcf0163942e5393a0ab770`.
- Registered checkpoint identities are recorded in the run's
  `deployment_run.json` and the gallery's `review.json`.

Four completed Stage 0/1 results from the initial CPU run were reused after
preprocessing-config compatibility checks; their Stage 2 onward was rerun on
GPU1. The other 96 drawings were processed from their original TIFs. CPU and
GPU identities use separate result databases. The GPU continuation started
with two workers because host RAM was constrained, then resumed with eight
workers after 44 rows completed and host memory became available. Completed
rows were retained. Process stops for those controlled resumes, and the
bounded diagnostic attempt above, were intentional rather than crashes.

## Verification

- [Artifact verification](final_artifacts.json): 100 original PNG previews are
  pixel-identical to their source TIFs; 475 preview files and 649 download links
  verified; source graph hashes and deployment/checkpoint identity match.
- [Browser verification](final_browser.json): all 100 entries checked in
  Chromium, with expected image counts and native dimensions. Navigation,
  synchronized zoom/pan, filters, feedback persistence and CSV download pass.
  Desktop 1600x1000, desktop 1280x800, and mobile 390x844 layouts pass.
- [Desktop screenshot](final_desktop.png) and [mobile screenshot](final_390.png)
  were visually inspected.
- [Full test suite](tests.xml): **424 passed**, with existing dependency warnings.
- Gallery SVG preview errors: **0**. Graph preview errors: **0**.
- `git diff --check` passed.

## Reproduce The Gallery

Run from the repository root. Lucide icons are already bundled in the finished
gallery. To build a separate gallery, supply a lucide-static icons directory.

```bash
.venv/bin/python -m tools.build_pipeline_review \
  --worklist output/PatentVecHatchStrokeTraining/patentdata100_disjoint_worklist.csv \
  --current output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14 \
  --previous output/PatentData91_CompactHatchFitV2 \
  --output output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14/viewer \
  --icons /tmp/patent-review-browser/node_modules/lucide-static/icons
```

The standalone gallery opens directly without a server. Assessments and notes
are saved in browser-local storage; its download button exports the 100-row
feedback CSV. Preserve the output directories and original data for the direct
TIF, SVG, DXF and JSON links. The HTML and its `assets`/`icons` folders provide
the offline visual review; `review.json` records the displayed provenance.

## Next Decision

Collect the user's per-sample visual feedback before changing reconstruction
behavior. This run does not establish deployment readiness or a quantitative
accuracy gain. The graph-only long trace, persistent quality-gate failures,
reference fidelity, and visible lineweight differences are candidates for
follow-up investigation, not fixes implemented as part of this review.
