# Patent-like dataset: Phase 0 evidence

Date: 2026-09-15. Scope: implement and execute the measurement prerequisite
from [the revised roadmap](../../../PATENT_LIKE_SYNTHESIS_ROADMAP.md), without
changing the canonical deployment pipeline, model weights or frozen A/B data.

## Outcome

- `phase0_smoke3`: three successful preliminary examples, one per tier.
- `phase0_baseline300`: 300 drawings, balanced 100/100/100; 60 five-sample
  shards. Generation took 258.569 s on 12 CPU workers (1.16023 drawings/s).
- Archive size: 278,589,440 bytes. Full directory including preprocessing
  artifacts and preview images occupied approximately 371 MiB at audit time.
- Four retained 1024x1024 PNGs per sample: `clean.png`, `degraded.png`,
  `reference_free_clean.png`, `reference_free_degraded.png`.
- Every sample includes `sample.json`, `quality.json`, semantic `masks.npz`,
  `puhachov.npz`, and `free2cad_edges.npz`. Source provenance is in the vector JSON.
- Exact C2 audit: 34,185 endpoints, 110,886 junctions, zero missing, extra or
  duplicate endpoint/junction labels and zero unsupported skeleton pixels.
  Stored corner supply is 3,037; semantic corner completeness is not audited.
- Free2CAD: 5,360 targets, including 1,329 polylines (1,307 direct and 22
  aggregated). Independent polyline audit passed 300/300 drawings and
  1,329/1,329 targets, with worst per-target p90 residual 3.605553 px under
  the existing 4 px bound. Passing this tolerance is not an exact vector match.
- All 60 checksum/resume markers and recorded implementation hashes verified.
- All 30 stratified gallery images loaded at desktop/mobile widths, with no
  browser errors or horizontal overflow. Three full-resolution triptychs and
  the mobile screenshot were inspected. This is not full visual acceptance.
- Repository test suite: 437 passed, with existing deprecation warnings.

**Training release status: not accepted.** Generation used the original
composition/noise recipe intentionally. No training, 10k release, 50k run,
model promotion or deployment-gate recalibration took place.

## Evidence locations

All new generated data is under the requested external root:

```text
/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/
  phase0_smoke3/
  phase0_baseline300/
    generation.json
    manifest.json
    manifest.jsonl
    shards/shard_*.tar
    markers/shard_*.json
    measurements/measurements.json
    measurements/stage1/<sample>/
    polyline_contract.json
    audit/index.html
    audit/CONTACT_SHEET_*.jpg
```

This repository directory retains `phase0_summary.json` (compact results and
SHA-256 evidence), `tests.xml`, `browser_checks.json`, desktop/mobile
screenshots, and the scripts that collected this evidence. The generator run
fingerprint is:

```text
d8b707445402d69e53ea00a3c23dd01e75bc4c8265c67d99f07adcbe2707a14c
```

## What changed in the diagnosis

The initial raw-junction median of 131.1 for real patents was reproduced,
but production CN junction-cluster density is 36.66, not 131.1. At matched
longside 1024 the patent CN median is 32.03. Clean synthetic C2 medians are
19.86, 28.42 and 50.45 for medium, hard and very hard, respectively.
Very hard is already denser on this metric. Universal densification is not
supported by the new evidence.

All 100 real-patent database rows used `passthrough_binary`; all 300 synthetic
oracle-reference-free degraded images used `sketchcleannet`. After that route,
synthetic CN medians fell to 5.52, 10.91 and 24.12. Fidelity F1 against each
sample's clean C2 skeleton at 2 px tolerance has medians 0.7196, 0.7303 and
0.7828. These are image metrics, not learned Stage 2 evaluation scores.
Native patent median tile count is 49 (512/256), versus 9 for these samples.

Tiny-component fraction is also route-dependent: native real median 0.317,
direct-threshold degraded reference-free synthetic medians 0.851/0.809/0.746,
and deployed synthetic Stage 1 medians zero. Reducing speckle by matching a
single mixed-route number would be misleading. Metric definitions and full
p10/median/p90 tables are in the JSON evidence.

The reviewed very-hard example `pv_train_0000002_very_hard` has colliding
horizontal dimension annotations near the top. Structural quality gates do
not prevent this visual defect. The corpus is still a mechanical-composition
curriculum, not a set of physically verified complete inventions.

## Next priorities

1. Fixed-geometry binary/grayscale/noise and native-scale/lineweight probes,
   preserving separate clean-topology and observation-supervision contracts.
2. Collision-aware reference/dimension placement and stratified visual checks.
3. Split original source identities, including sibling CAD views; remove the
   unrestricted retrieval fallback and record underlying source-content hashes.
4. Re-weight the hard/very-hard curriculum before trying selective extra
   contacts. Establish route-specific distribution acceptance from the probes.
5. Only then freeze and generate the 10k release, audit it, and run matched
   training controls with real-domain evaluation. Generation alone cannot
   demonstrate transfer to PatentData.

With the *unchanged* measured recipe, 10k projects to about 2h24m and 9.29 GB
of archives, excluding measurement artifacts and training exports. A different
canvas, rejection rate or recipe requires a new throughput benchmark.

## Reproduction

Run from the project root. External-disk write access is required. Generation
is CPU-only; the measured actual Stage 1 pass used CUDA 1. The output identity
is immutable: code/configuration changes require a new dataset directory.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m syntheticData.generate_dataset \
  --mode probe --count 300 --workers 12 --shard-size 5 --canvas 1024 \
  --curriculum balanced --retain-rasters --audit-every 10 \
  --audit-strategy stratified \
  --stage2-label-contract reference-free-raster-topology-v1 --seed 150926 \
  --output '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300'

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m syntheticData.measure_patent_like \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300' \
  --patent-run output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14 \
  --stage1-config config_deploy.yaml --stage1-device cuda:1 \
  --output '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300/measurements'

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m syntheticData.validate_polyline_contract \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300' \
  --sample-limit 300 \
  --output '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300/polyline_contract.json'

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m syntheticData.build_dataset_audit \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300' \
  --stratified-per-difficulty 10 --per-sheet 6 \
  --output '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300/audit'

env PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/audits/2026-09-15/patent_like_generation/summarize_phase0.py \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300' \
  --patent-run output/PatentData100_CurrentPipelineReviewGPU1_2026-09-14

env NODE_PATH=/tmp/patent-review-browser/node_modules \
  PLAYWRIGHT_BROWSERS_PATH=/tmp/patent-review-browser/browsers \
  node docs/audits/2026-09-15/patent_like_generation/verify_preview.cjs \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300/audit/index.html'

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests -q \
  --junitxml=docs/audits/2026-09-15/patent_like_generation/tests.xml
```

## Contract limitations

`puhachov.npz` labels apply to its own clean reference-free C2 skeleton, not
directly to a degraded PNG or a changed Stage 1 output. The exact audit checks
all CN endpoint/junction cluster representatives using the production
convention. The semantic preview overlays authored contacts, not all C2
raster keypoints; its sparse red markers are not evidence of missing NPZ labels.

Legacy `exact/snapped/dropped` fields in the manifest's Puhachov summary refer
to the older semantic projection counters. Use endpoint/junction/corner
counts and independent `contract_totals` for the C2 target counts.

This is a train-pool diagnostic, **not source-disjoint validation**. Current
fingerprints include the source index, implementation and runtime, not full
source-corpus content checksums. Oracle reference removal does not assess OCR
or leader removal. The Free2CAD target archive is a per-edge primitive task,
not a CAD-program or text-to-CAD dataset. No end-to-end PatentData quality
claim follows from passing these synthetic contracts.

## Paired acquisition follow-up

`syntheticData/probe_patent_acquisition.py` reuses ten frozen drawings per
tier. It compares five acquisition variants at three render settings:
1024 relative widths, 2048 relative widths, and 2048 constant-pixel stroke
widths. Geometry, dash lengths and text layout remain fixed in normalized
coordinates. This is 450 Stage 1 passes over 30 reused drawings, not 450 new
source drawings.

The five variants are a binary C2 positive control, default grayscale scan,
reduced-speckle grayscale scan, and the two scan variants binarized at 210.
The control must return the clean C2 skeleton byte-for-byte through Stage 1;
it is not a learned-recovery achievement. The 1024 grayscale re-render must
match the frozen baseline. Each changed canvas/width gets a newly rendered
mask archive and independently audited clean C2 target.

The probe retains full and oracle-reference-free PNGs, actual Stage 1 outputs,
clean target NPZ, masks and original source JSON. It does not attach clean
keypoints to the changed observation skeleton or fabricate new Free2CAD labels.
Noise/encoding changes alone leave the existing C2 training arrays unchanged.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m syntheticData.probe_patent_acquisition \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase0_baseline300' \
  --per-difficulty 10 --device cuda:1 \
  --output '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase05_acquisition30'

env PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python docs/audits/2026-09-15/patent_like_generation/summarize_acquisition.py \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase05_acquisition30'

env NODE_PATH=/tmp/patent-review-browser/node_modules \
  PLAYWRIGHT_BROWSERS_PATH=/tmp/patent-review-browser/browsers \
  node docs/audits/2026-09-15/patent_like_generation/verify_acquisition.cjs \
  '/media/safe/secondary disk/IPdrawings/Patent_Like_Dataset/phase05_acquisition30/index.html'
```

The probe requires a new output directory and does not silently resume partial
results. Full regression suite after adding it: **439 passed**; evidence is
`tests_phase05.xml`.

### Completed results

All 30 source drawings and 450 passes completed. Independent verification
found 450/450 matching input/output hash pairs, with 270 binary and 180 neural
routes. All 90 target audits passed: 11,146 endpoints and 35,541 junctions,
zero missing/extra/duplicate labels and zero unsupported skeleton pixels.
The 90 positive controls were pixel-identical to their clean targets. All
30 baseline-size default grayscale observations matched the frozen corpus.

| Render setting | Grayscale F1 | Grayscale, low speckle | Binary F1 | Binary, low speckle |
|---|---:|---:|---:|---:|
| 1024 relative | 0.7134 | 0.7128 | 0.9896 | 0.9963 |
| 2048 relative | 0.7432 | 0.7406 | 0.9813 | 0.9932 |
| 2048 fixed-pixel | 0.7330 | 0.7330 | 0.9824 | 0.9937 |

These are medians over 30 drawings, at 2 px tolerance against each render's
own clean target. The report includes 4 px scores at 2048 for normalized-frame
comparisons. The 1024 default encoding effect has paired mean F1 gain 0.2911
(bootstrap interval 0.2532-0.3294); noise reduction in grayscale alone gives
0.0002 (-0.0002 to 0.0007). Bootstrap units are source drawings within a
render setting, not the 450 correlated renderings. These remain exploratory
synthetic findings, not measured real-patent generalization.

Binary/lower-speckle tiny-component fraction remains 0.603 at 1024 and 0.784
at 2048 relative widths. Although 2048 matches the real native median of 49
tiles, pooled clean CN density falls to 13.95, below the native patent median
36.66. Binary acquisition wins on fidelity, but neither the noise distribution
nor the native topology distribution is accepted. Next: ink-/resolution-aware
noise probes, semantic-conditioned topology analysis, annotation placement
fixes, and strict source partitioning before the 10k recipe is frozen.

All 99 gallery images loaded in Chromium at 1600x1000 and 390x844. Desktop
and mobile screenshots were inspected; the comparison grid scrolls within
its own region on mobile, without horizontal page overflow. Evidence:
`phase05_summary.json`, `acquisition_browser_checks.json`,
`acquisition_1600.png`, `acquisition_390.png`. External artifacts occupy about
281 MiB. Full report SHA-256:

```text
a66abfad3838259a2b3c6da4cc2cf257d74b57d77ac006700aef5ec070eb4a0e
```

The 300-drawing baseline and this 30-source paired probe remain immutable
diagnostics. The 10k training release is still unlaunched and unaccepted.
