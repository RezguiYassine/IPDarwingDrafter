# Pilot v3 — Frozen Regression Baseline

Frozen 2026-07-24. **100 real patent drawings** sampled stratified (1/patent,
seed 2026) from `filter_manifest_v3` (drawing set). Code mix: 91 F + 9 A.
Every future pipeline change must be evaluated against this same 100-image set.

- Sample list: `frozen_sample_100.csv` (patent, filename, letter_code, path)
- Configs: `config_pilotv3_{cnfusion,phaseA,phaseB}.yaml` (repo root)
- Viewer:  `tools/pilot_build_viewer2_v3.py` -> `output/pilotv3_viewer2/index.html`
- Fixes live in this baseline: ghost-circle path-completeness guard
  (`_is_circular_loop`), separate hatch-CNN removal ceiling
  (`hachure_cnn_max_removed_edge_ratio=0.92`).
- Code commit at freeze: pre-commit (see git log around this file's add).

## Stage-2 robustness (pipeline internal quality gate, `stop_on_stage2`)
| Config | accepted | quality-gated |
|---|---:|---:|
| CN-fusion | 91 | 9 |
| Full-CNN phase A + tiling | 70 | 30 |
| Full-CNN phase B + tiling | 71 | 29 |

CN-fusion ~3x more robust; native-res tiling did NOT fix pure-CNN
over-fragmentation. **CN-fusion is the production Stage-2 baseline.**

## Post-vectorization gate (`tools/postvec_quality_gate.py`, text-load router on)
| Verdict | n |
|---|---:|
| clean | 71 |
| review_text_heavy | 14 |
| low_quality | 8 |
| not_ok | 7 |

## Content-audited 4-manifest split (`tools/pilotv3_split_manifests.py`)
| Bucket | n | audited | provisional |
|---|---:|---:|---:|
| positive_clean | 67 | 20 | 47 |
| positive_hard | 5 | 5 | 0 |
| negative_invalid | 21 | 21 | 0 |
| borderline | 7 | 7 | 0 |

**Honest yield:** the fragmentation gate's "85 clean" was optimistic. Manual
audit found **21 confirmed non-drawings** (flowcharts, block/network/UML
diagrams, tables, charts, UI, chemistry, shaded renders) that vectorize
CLEANLY (low micro/isolation) and so passed the gate. True clean-positive
yield is <= 67% (47 of the 67 positive_clean are still un-audited, doc Step 4).

**Hard-positive failure families (one exemplar each — mine more, doc §7):**
dense_detail, hatch_boundary_fragmentation, dashed_line_fragmentation,
dimension_arc_confusion, text_box_isolation.

## Key methodological finding
No single cheap feature separates clean-line non-drawings from real
orthographic drawings (tested: curve-fraction, orthogonality, OCR-label
geometry, cc_recovered fraction — all overlap). The `cc_recovered` text-load
signal *ranks* the class (~75% precision at the top) so it is used as an
ADVISORY review-router (`review_text_heavy`), never a hard reject. Clean
auto-classification needs OCR text-content or a small learned classifier
(future work).
