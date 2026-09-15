# Patent Vectorization: Project Status and Prioritized Roadmap

> Historical audit snapshot, before the fixes implemented later on 2026-09-09.
> The confirmed Priority 0 defects below motivated the work recorded in
> [Priority 0 implementation and validation](PRIORITY0_IMPLEMENTATION_2026-09-09.md).
> Keep this baseline evidence separate from the post-fix measurements.

## Executive Assessment

The project has a working, benchmarked Stage 0-4 vectorization baseline, substantial completed model training, a functioning synthetic-data generator, and credible measured improvements to primitive fitting and export. `config_deploy.yaml` is the canonical deployment configuration. Its reconciliation is complete; configuration disagreement is no longer the main unresolved decision.

However, a validated baseline is not the same as a lossless reconstruction system or an automatically trustworthy producer of LLM training targets. This investigation found two concrete preservation defects in the current code: recognized reference text is discarded during JSON serialization, and the presence of any hatch region suppresses the per-line export fallback for all separated hatch edges. Both are supported by saved production-chain artifacts and can be addressed without additional labeling or model training.[^audit][^refs][^hatchcode]

The most important conclusions are:

1. **Canonical chain established:** reference handling, conditional preprocessing, tiled PatentVec A Puhachov, hatch-region processing, guarded RANSAC/compound fitting, and SVG/DXF export. Free2CAD is not the production fitter. SketchCleanNet was not used for cleaning any of the 100 benchmark images; Stage 1 selected binary passthrough.
2. **Latest fitting improvement completed:** all 31,524 Drawing2CAD test views replayed successfully. On the separate frozen 100-patent cohort, all 91 upstream-eligible drawings passed Stage 3; three Stage 1 and six Stage 2 gates remain. This is a gate-pass count, not 91% reconstruction accuracy.
3. **Synthetic generation completed at experimental scale:** A10k plus B10k, not 50k. B includes 2,500 very-hard drawings. Both datasets have been used for Stage 2 and Stage 3 training. The designated external full-generation directory remains empty.
4. **Full SketchGraphs training really happened:** it was not limited to the earlier pilot. Later synthetic adaptation used smaller, controlled replay pools; its 9,565-image SketchGraphs evaluation must not be described as the original full 300,321-accepted-image test.
5. **Hatching remains only partially solved:** the deployed region detector is useful, but the quoted 0.816 IoU is positive-figure test IoU, not overall accuracy. The experimental stroke separator failed its real-data acceptance criteria.
6. **Filtering remains a major data-quality risk:** cleanly vectorized charts and chemistry are still admitted. Geometry gates cannot establish semantic suitability for engineering-drawing training.
7. **No new hand-annotated gold set is required to make the next improvements.** Existing audited figures, CAD ground truth, synthetic semantic masks, and deterministic preservation checks are sufficient for the immediate roadmap. They do not justify an unrestricted claim of absolute real-patent accuracy.

**Recommended direction:** preserve the canonical baseline, fix the two confirmed information-loss paths, strengthen content routing, and evaluate those changes on existing assets before committing to more training or 50k generation.

## Scope and Evidence

This is a repository and artifact audit performed on September 9, 2026, against commit `d88637361c`. Evidence includes source code, YAML configurations, checkpoint hashes, generation manifests, training coverage ledgers, evaluation JSON, SQLite result databases, existing labeled audits, and a recovered-output contact sheet. The current unit/integration suite was rerun: **188 passed**, with **35,404 warnings**, in 8.73 seconds. No training or full pipeline evaluation was launched for this report.

The accompanying [evidence snapshot](audits/2026-09-09/evidence.json) and [collector](audits/2026-09-09/collect_evidence.py) make the principal counts and preservation checks reproducible. The collector reads existing artifacts and writes only its requested JSON output. Historical visual reviews are distinguished from new measurements; this audit did not manually re-annotate or visually inspect every drawing.[^audit]

Several Markdown sections retain earlier decisions. Priority of evidence is: current code/configuration, immutable or saved experiment artifacts, dated execution results, then historical plans. In particular, old statements that CN-fusion is production, that C2 evaluation is running, or that 18 Stage 3 gates remain are not the current status.[^production][^pilot][^training]

## Canonical Deployment Chain

| Component | Current behavior | Status and limits |
|---|---|---|
| Content filtering | Separate manifest-based workflow | Not a semantic safeguard supplied merely by selecting the deployment YAML. |
| Stage 0 | OCR plus connected-component recovery, leader detection, bounded iterative removal, crossing repair, reference JSON and image crops | Operational; complete reference removal is not independently established. OCR text is currently lost at serialization. |
| Stage 1 | Binary passthrough for already-binary scans; SketchCleanNet or classical processing for other inputs | All 100 benchmark rows recorded `passthrough_binary`. Configuring a neural cleaner does not mean it was exercised. |
| Stage 2 | `puhachov_patentvec_complexityA.pth`, pure CNN rather than fusion, tiled 512/256, threshold 0.3, NMS radius 5 | All 97 rows reaching Stage 2 recorded `cnn_tiled`. |
| Hatching | `hatch_unet.pth`, threshold 0.7, early region-residue cleanup | Region aggregation is implicitly enabled. The newer two-channel stroke CNN is not enabled. |
| Stage 3 | RANSAC cascade, guarded weak-fit replacement with compound paths, guarded simplification of closed traces | Production path. The configured Free2CAD weight is not consumed by this fitter. |
| Stage 4 | Pixel-center-corrected SVG/DXF, globally oriented compound segments, reference reinjection | 91 SVG and 91 DXF files exist for the latest eligible cohort. Reference labels are raster overlays in SVG and currently lack semantic text in DXF. |

The canonical YAML and the frozen 100-patent replay configuration differ only in the Puhachov and hatch-CNN devices: deployment specifies CPU, while the replay snapshot specifies `cuda:1`. All four configured checkpoint hashes match `models/README.md`. This establishes artifact/configuration alignment, not a newly measured CPU/GPU numerical-equivalence or throughput result.[^audit][^production]

Two implicit settings deserve explicit documentation: `hachure_mode` is absent from the YAML and defaults to `region`; `dashed_grouping` is absent and defaults to false. Therefore, the canonical chain is **not** a guaranteed line-only hatch exporter, and the implemented dashed-line grouping feature is **not** active.[^hatchcode][^dash]

The model change from the older Phase A fusion path to PatentVec A tiled inference is an alignment with the measured frozen chain. A directly paired, current-code PatentData comparison of those two complete Stage 2 configurations remains missing. The frozen Stage 3 comparison held Stage 2 fixed; it cannot establish which Stage 2 alternative is superior.[^production]

## Newly Confirmed Problems

### 1. Recognized Reference Text Is Discarded

The OCR detector stores the recognized token as `text: t`. `_build_reference_doc` subsequently writes `"text": ""` unconditionally. This is a definite serialization defect, not an OCR-recall estimate.

Across the 91 copied reference JSON files in the latest replay, there are **2,744 labels and zero nonempty text values**: 1,068 marked `ocr_numeral`, 84 `ocr_reference`, and 1,592 `cc_recovered`. The last group may legitimately lack a recognized token; the first two have already passed OCR detection. The database records 3,084 labels over all 100 source rows, a different denominator because the replay copied artifacts for the 91 eligible rows.[^audit][^refs]

Consequences:

- SVG can appear visually faithful because reference crops are embedded as images, while the annotation is not actually represented as vector/text content.
- DXF's MTEXT path executes only for nonempty text, so those serialized labels cannot be restored as semantic text.
- Any future text-to-vector training dataset loses recognized reference identities even when OCR succeeded.

The fix should retain recognized text, class and confidence, and test JSON-to-SVG/DXF preservation. It must also make SVG crop-versus-text rendering an explicit choice: the current exporter independently draws both when both are present, so a text-only serialization fix could create double-rendered labels. Unknown tokens must remain unknown rather than be invented.[^export]

### 2. Hatch Regions Can Suppress Unrepresented Hatch Edges

When a graph contains at least one `hachure_region`, Stage 3 emits region primitives instead of fitting **any** of its `removed_hachures` as individual lines. Region aggregation can omit small or unsuitable groups, and its descriptors contain no explicit source-edge membership list for export accounting.

In the latest 91 graph files, eight drawings contain regions. They contain **20 regions and 955 separated hatch edges**, but only 459 region-member entries. The count difference alone is not proof of loss because residue cleanup can add edges after aggregation. A stronger geometric check finds **272 separated edges wholly outside every emitted region boundary**, and 339 at least partly outside. There are **44,416 unique side-layer pixels outside region boundaries**, summed across those eight drawings.[^audit][^hatchcode]

These are representation-coverage findings, not a claim that exactly 44,416 pixels disappear from the final rendered image: some can overlap other geometry. Nevertheless, the region branch supplies no per-line representation for these outside edges. This violates the intended preservation-through-a-side-layer contract.

Examples include `EP2565056B1/F0001` with 75 edges wholly outside its one region, and `EP1974619B1/F0001` with 61 wholly outside its three regions. The appropriate fix is explicit ownership: export each separated edge through a validated region or through a per-line fallback, with no unaccounted remainder. Region convex hulls also need overfill checks; a convex envelope does not guarantee faithful coverage of concave or disconnected hatching.[^hatchcode]

### 3. Content Quality Is Not Geometry Quality

The existing pilot-v3 audit contains 56 clean positives, five hard positives, 27 invalid targets, and 12 borderline cases. These are actual existing content labels. The 56% clean-positive yield belongs to that historical sample and filter version; it must not be projected to all current PatentData.[^pilot]

The more recent Stage 3 audit still identifies `EP3184100B1/F0004` as a bar chart and `EP3191598B1/F0001` as chemistry admitted by the permissive `long_engineering_lines` rule. The recovered-output sheet also visibly contains plots and chemistry among outputs rescued from Stage 3 gates. Better primitive fitting does not make these suitable mechanical-drawing training examples.[^training][^visual]

Content rejection should therefore be upstream of training-target acceptance and independently logged. Charts, chemistry, formulas and dense text should be routed or rejected according to the intended corpus scope, not treated as failed mechanical drawings that require more vectorization training.

### 4. Canonical Configuration Does Not Yet Enforce Canonical Execution

The batch CLI still defaults to `config.yaml`; deployment requires explicitly passing `--config config_deploy.yaml`. Model-loading paths can fall back to classical/CN/heuristic behavior when weights are unavailable. Batch workers set logging to ERROR, so warnings about such fallback can be hidden during batch execution.[^runtime]

For controlled deployment, require a startup check of checkpoint presence/hashes and record the effective model sources, configuration digest and device. Optional fallback remains useful in research, but it should not silently satisfy a production-chain acceptance check. Hashes are documented; automated enforcement and a complete reproducible model package remain to be finished.

## Data Inventory

### Real Datasets

| Dataset/artifact | Verified scope | Use and caveat |
|---|---|---|
| SketchGraphs | Official filtered training source: 9,179,789 sketches; full Stage 2 test: 313,271 attempted, 300,321 accepted | Full training and evaluation completed. Unsupported geometry is explicitly rejected. |
| Drawing2CAD / CAD-VGDrawing | 141,831 / 7,879 / 7,881 CAD models in train/val/test, four views each: 567,324 / 31,516 / 31,524 views | These names refer to the same dataset here, not two independent sources. Supplies polylines and Beziers. |
| ArchCAD | Local dataset and cached keypoint labels; 1,159 validation drawings in evaluations | Added to the synthetic-adaptation rehearsal mix. It is no longer an entirely unseen domain. |
| PatentData clean12 filter | 275,804 input records: 56,971 drawing candidates, 218,778 discarded, 55 errors | Candidate status is not verified semantic suitability. |
| PatentData filter-v3 | Same 275,804 records: 79,111 drawing candidates, 196,638 discarded, 55 errors | Different filter/version; do not mix its yields with clean12. |
| Historical clean12 gated run | 57,026 DB rows: 13,904 `ok`, 1,364 Stage 1 gates, 3,558 Stage 2 gates, 38,145 Stage 3 gates, 55 `stage1` rows | Old pipeline snapshot, not a full-corpus evaluation of the new canonical chain. The 55 intermediate rows require reconciliation, not silent counting as completed. |
| Historical curated manifests | 598 strict selections, then 259 CLIP-curated selections | Bootstrap corpus, not a large validated LLM training set. |
| Existing reviewed hatch collection | 212 unique figures; newer stroke export produced 203 usable figures, split 170 train / 33 validation by patent | Region annotations and automatically derived stroke supervision must not be confused with fully hand-labeled stroke truth. |

The main local source directories occupy approximately 6.2 GiB for SketchGraphs, 9.4 GiB for Drawing2CAD, 15 GiB for ArchCAD, and 13 GiB for PatentData. Local models occupy approximately 3.5 GiB. These sizes exclude the many derived experiment outputs.[^audit][^training]

### Synthetic Generation

| Item | A10k | Enhanced B10k |
|---|---:|---:|
| Manifest rows / tar shards | 10,000 / 100 | 10,000 / 100 |
| Medium / hard / very hard | 5,000 / 5,000 / 0 | 2,500 / 5,000 / 2,500 |
| Automated final acceptance | 10,000 | 10,000 |
| Archive bytes | 2,736,384,000 | 3,391,703,040 |
| Recorded generation time | 3,270.7 s, 12 workers | 3,553.0 s, 24 workers |
| Polyline labels before trainer cleaning | 38,229 | 43,401 |

The two experiments share the same source indices and global seed. They are 20,000 generated sample records, **not 20,000 independent underlying CAD designs**. Relative to A, B increases components by 19.5%, junctions by 34.6%, exact keypoints by 18.3%, and polyline supply by 13.5%. A separate 500-drawing B audit passed all 2,168 sampled polyline targets.[^synthetic][^audit]

Implemented generator capabilities include component/source indexing, exact geometry transforms, connected interactions, T/X junctions, containment and tangency, cycle-forming bridges, visible/amodal records, patent semantic layers, masks, provenance, seeded rendering, quality gates, atomic shard completion markers, deterministic resume, and training exports. These are substantial completed M0-M6 capabilities, although the roadmap's full operator and occlusion ambitions are not all implemented.[^synthetic][^roadmap]

Each native A/B training export contains 8,976 training and 1,024 diagnostic validation drawings. That synthetic split is deterministic but **not source-disjoint**; both sides derive from the source training pool. It is useful for contract diagnostics, not an independent generalization claim.

C2 reference-free topology, C3 hatch-suppressed topology, and the 10k hatch-stroke export are alternative labels/representations of the existing B drawings. They are not additional independently generated 10k datasets. C2 contains 4,766,069 labels and passed its separate 500-source topology audit.[^synthetic][^training]

**The 50k dataset has not been generated or used for training.** `/media/safe/secondary disk/IPdrawings` exists and is empty. Recorded B throughput projects about 4.9 hours and 15.8 GiB for 50k with that same curriculum and generation setup, before training-export expansion and extra audits. That is a projection, not a measured duration for an all-very-hard 50k dataset. The external filesystem currently has about 223 GiB available; the project filesystem has about 264 GiB available.

M7 realism calibration remains incomplete. M8's numerical 20k pilot target is met across A/B, but correlation and remaining coverage limits matter. M9 has completed training and evaluations, with mixed real-patent outcomes. M10 CAD-valid program generation is not delivered. M11 packaging/release work is partial. Track-A geometry is suitable for 2D vector supervision; it is not a parametric CAD construction program.[^roadmap]

## Model Training Status

### Stage 2 Puhachov

**Full Phase A is complete.** The 70% SketchGraphs / 30% Drawing2CAD schedule attempted all 9,179,789 official filtered training sketches: 8,809,417 accepted, 370,372 rejected, zero unattempted and zero decode errors. The selected checkpoint is step 330,000; training continued to its final coverage state. Full accepted SketchGraphs test macro-F1 is **0.934901**, versus **0.431670** for the earlier Drawing2CAD-only checkpoint.[^coverage][^sgtest]

**Full Phase B is also complete.** It used the 60% SketchGraphs / 40% Drawing2CAD schedule and reached step 637,486 with the same complete source coverage. Increasing the D2C ratio did not establish an end-to-end accuracy improvement: on 31,524 paired D2C views, mean Chamfer worsened from 0.903088 to 0.905125, pixel IoU from 0.698813 to 0.697434, and skeleton IoU from 0.566828 to 0.565956. Runtime improved by about 4%. This is a negative accuracy result, not an unfinished evaluation.[^phaseb]

**Synthetic adaptation is complete.** The successful controlled mixer uses, per 59,840-sample epoch, 20,346 D2C, 25,432 cached SketchGraphs, 5,086 ArchCAD, and 8,976 synthetic drawings. A and B each trained three epochs, 14,961 steps. The fixed three-domain selector improved from 0.8193 to approximately 0.8477. An earlier experiment without ArchCAD rehearsal was stopped after severe ArchCAD forgetting.[^training]

That selector gain is not uniform: for A, the fixed D2C selector falls from 0.8670 to 0.8510 and SketchGraphs from 0.9640 to 0.9620, while ArchCAD rises from 0.6270 to 0.7301. The mixer adds ArchCAD training exposure as well as synthetic data, so its mean improvement does not isolate the causal benefit of synthesis. A matched real-only rehearsal control is needed to measure that incremental benefit. The loader correctly draws ArchCAD rehearsal from its training directory; this is a domain-exposure caveat, not evidence that validation examples were deliberately used as training rows.[^training][^mixercode]

| Adapted checkpoint | D2C test, 31,524 | Cached SG test, 9,565 | ArchCAD val, 1,159 | Current role |
|---|---:|---:|---:|---|
| PatentVec A | 0.856136 | 0.936599 | 0.730050 | Canonical Stage 2 |
| PatentVec B | 0.857240 | 0.936426 | 0.727709 | Research alternative; no decisive overall win |
| C2 reference-free topology | 0.876543 | 0.935981 | 0.716139 | Completed evaluation; not promoted |
| C3 hatch-suppressed topology | 0.873161 | 0.936597 | 0.715059 | Completed three-epoch training/evaluation; not promoted |

These are detector macro-F1 measurements, not vector reconstruction accuracies. C2/C3 evaluation JSON records threshold 0.3 and NMS radius 3, whereas canonical inference uses NMS radius 5; a deployment decision requires an end-to-end comparison of actual inference settings.[^stage2eval]

C2 is an instructive mixed result: corrected synthetic supervision and higher D2C detector F1 did not automatically improve real patent reconstruction. In the historical matched 97-row Stage 2 comparison, C2 reduced Chamfer but reduced skeleton recall from **0.900336 to 0.882668** and F1 from **0.943766 to 0.934079** relative to A. A smaller distance is not a sufficient win when coverage worsens. C3's synthetic oracle diagnostics also retain substantial hatch/structure mixing; neither contract is a released solution to real hatch separation.[^patentmodels]

### Stage 3 Free2CAD and the Polyline Question

The local Free2CAD primitive classifier/regressor was trained on full SketchGraphs-derived supervision, not only a pilot. Full extraction supplied approximately 27.0 million training edges; the two-epoch run achieved supported test macro-F1 **0.9993** over **926,940** test samples. SketchGraphs supplies line/arc/circle supervision but no polyline class.[^free2cad]

The class-supply problem was then addressed with Drawing2CAD. Complete conversion processed 567,324 training views and supplied 182,850 polyline and 154,826 Bezier labels. The balanced five-class mixed corpus contained 914,250 labels, exactly 182,850 per class. The selected mixed model achieved **0.9683** mixed-validation macro-F1, **0.8603** on D2C test, and **0.9936** on full SketchGraphs test.[^free2cad]

This is a successful supervision/classification improvement, but not a successful replacement for the production geometric fitter. On the documented small 18-graph D2C comparison, learned fitting had mean primitive Chamfer 16.65, the research RANSAC baseline 19.39, and the production cascade 0.43. The approximately 40-fold difference is specific to that small benchmark; it is not a universal claim about learned fitting.[^fittercomparison]

Synthetic A/B Free2CAD adaptation completed eight epochs each. Synthetic polyline F1 improved to **0.8419 / 0.8228**, but real validation fell to **0.9622 / 0.9626** from 0.9683. Protected checkpoint selection correctly retained the real-data warm start; final adapted states remain diagnostic artifacts. No new Free2CAD training is recommended now, consistent with keeping RANSAC as the production fitter.[^training]

The remaining polyline issue is therefore not simply missing class labels. It is robust fitting and continuity on the actual noisy/fragmented Stage 2 output. The compound-path and closed-trace changes directly address that problem without requiring another learned-model run.

## Hatching: What Has and Has Not Worked

### Deployed Region Detector

The installed `hatch_unet.pth` matches grid-v2's selected run: 30 training epochs, best epoch 29, learning rate 0.0003, positive weight 3.0. Its recorded validation positive IoU is **0.788310**; test positive IoU **0.815879**; test all-figure IoU **0.552687**.[^hatchtrain]

The split is 170 training, 21 validation, and 21 test figures. The test contains only 13 positive and eight negative figures. Splitting was by figure, not patent: one patent occurs in train and validation, and four occur in train and test. The metric threshold is 0.5; the deployment threshold is 0.7. Therefore, the shorthand "hatch accuracy 81.6%" overstates what was measured. The existing 100-patent replay cohort has no patent overlap with this entire region-label collection, which is useful separate end-to-end regression evidence.[^audit][^hatchtrain]

The accepted residue-ordering fix preserves matching hatch residue before computing metrics and reconnects exposed structural neighbors without pruning those newly exposed ends. Its historical paired comparison improved three of 97 drawings, tied 94, and regressed none; Stage 2 F1 rose from 0.942526 to 0.942967. This is a small, useful engineering gain, not a complete solution.[^hatchstroke]

### Rejected Stroke-Level Experiments

The two-channel structural/hatch model was trained on the B-derived 10k multilabel dataset. Its conservative synthetic policy reached approximately 0.9995 hatch precision and 0.9234 recall, but real integration increased fragmentation. Synthetic performance did not transfer sufficiently.

Reviewed-real adaptation then used the 203-figure patent-disjoint export. The real35 and real50 candidates reached hatch F1 **0.856876 / 0.927022**, but structural recall **0.994210 / 0.991928**, failing the preservation objective. Additional source-separated variants were stopped after one epoch as structural recall deteriorated further. The final guarded real50 integration selected zero model-only edges and tied the region-only control through Stage 4 on the 100-patent comparison.[^hatchstroke]

That final experiment was **safely inert**: it avoided additional damage but provided no measurable reconstruction benefit. It is correctly absent from deployment. The immediate hatch priority is now deterministic side-layer export preservation and threshold-aware regression checks on existing annotations, not repeating the same separator training at larger scale.

## Reconstruction Evaluation

### Completed Full Drawing2CAD Stage 3/4 Replay

The promoted policy improves weak open fits only with bounded confidence gain, residuals, endpoint error and path complexity. Weak closed traces compete with bounded simplified alternatives; an exact trace remains available when no candidate passes. Stage 4 additionally fixes half-pixel alignment and mixed-segment path orientation.[^stage3]

| Metric | Corrected control | Selected policy |
|---|---:|---:|
| Mean symmetric Chamfer | 0.355585 | **0.345473** |
| Mean per-view p95 Chamfer | 1.279391 | **1.230501** |
| Pixel IoU | 0.783634 | **0.785794** |
| Skeleton IoU | 0.832186 | **0.835159** |
| Pixel precision | 0.811151 | **0.812492** |
| Pixel recall | 0.954054 | **0.955553** |

All 31,524 views completed with zero replay errors. Pixel metrics include all views; distance metrics have **31,344 finite pairs**, not 31,524. Nonfinite cases must remain visible in reports rather than be interpreted as successful distance measurements. Drawing-cluster bootstrap intervals exclude zero in the favorable direction for all six listed comparisons.[^d2cstage3]

The policy activated 4,622 weak-open promotions and 1,365 closed simplifications, reducing the latter's 969,138 source points to 36,213 vertices. Stricter endpoint=2-pixel and closed p95=0.5 policies were evaluated and rejected for the relevant regression/tradeoff reasons, rather than indiscriminately making thresholds tighter.[^stage3]

### Completed Frozen 100-Patent Stage 3/4 Replay

| Outcome | Count |
|---|---:|
| Previously `ok`, still `ok` | 73 |
| Previous Stage 3 gates recovered to `ok` | 18 |
| Stage 1 gates preserved | 3 |
| Stage 2 gates preserved | 6 |
| Eligible replays completed / failed | 91 / 0 |

On the 91 paired rows, F1 at two pixels improved **0.873256 -> 0.878608**, recall **0.852937 -> 0.858209**, and precision **0.907183 -> 0.912172**. Mean per-drawing p95 Chamfer improved **15.706233 -> 15.574026**. Mean Chamfer was statistically tied, **3.616210 -> 3.624344**. The F1 mean-delta 95% interval is **[0.003953, 0.006869]**.[^patentstage3]

Important measurement limits:

- This is an exact Stage 3/4 replay of frozen upstream artifacts, not a new end-to-end CPU deployment run.
- Patent fidelity uses the Stage 1 skeleton as its reference. It does not independently validate original-image structure removed incorrectly by Stage 0/1 or the semantics of restored labels.
- Stage 2 masks include both graph edges and separated hatch edges; a pixel copied from the input can score high even when its semantic class or connectivity is wrong.
- Different run-level denominators can create apparent improvements. In the final JSON, control Stage 2 has 97 measurable rows and replay Stage 2 has 91 copied rows; on the actual 91 pairs Stage 2 is identical. There is no Stage 2 gain attributable to this Stage 3-only experiment.
- Reconstruction of chemistry or a chart can improve these metrics while remaining an invalid target for the chosen engineering corpus.

These qualifications do not erase the measured Stage 3 gains. They delimit the claims those gains support.[^fidelitycode]

### Runtime

Stage 3 alone averages **73.57 seconds**, with median **27.81 seconds**, p95 **343.76 seconds**, and maximum **486.08 seconds** over 91 drawings. These are not full-pipeline timings. Dense compound fitting is a material throughput bottleneck even if accuracy remains the primary goal.[^audit]

At the host process check, no Vectorization training, generation or evaluation job was running. CUDA 0 was a 24 GiB RTX 4090 with about 12.8 GiB used and 35% utilization; CUDA 1 was another RTX 4090 with about 208 MiB used and 0% utilization. An Unreal/CARLA editor process was active. GPU utilization is not evidence that model training is continuing.

## Wins, Failures, and Remaining Work

| Category | Assessment |
|---|---|
| Full-data execution | Win: complete coverage ledgers, resumable training, explicit rejected-source accounting, completed full tests. |
| Synthetic infrastructure | Win: coherent M5/M6 compositions, enhanced difficulty, exact semantic layers, sharding, provenance and independent contract checks. |
| Stage 2 synthetic adaptation | Win with limits: protected multi-domain rehearsal improves selectors; higher complexity B is not a decisive real-domain winner. |
| Stage 3 and export | Strongest recent win: paired improvements on full D2C and frozen patents without loosening gates. |
| Free2CAD | Class-supply fix succeeded; learned end-to-end geometric replacement failed. RANSAC remains justified. |
| C2/C3 topology | Contract correction succeeded; real-patent superiority did not follow automatically. |
| Hatch-stroke CNN | Synthetic learning succeeded; real structural preservation and useful guarded integration failed. |
| Stage 0 | Removal/reinjection infrastructure works, but full recall is unproven and recognized text is discarded. |
| Hatch representation | Region processing exists, but mixed region/per-line preservation has a confirmed accounting gap. |
| Content filtering | Incomplete: invalid semantic classes can pass every geometry gate. |
| Operational reproducibility | Canonical configuration and hashes exist; default CLI selection, enforced model availability, CPU parity and deploy packaging remain incomplete. |
| LLM readiness | Vectorization support exists; a sufficiently pure, caption-paired, split-controlled training corpus and demonstrated text-to-vector model training are not established by these artifacts. |

The historical README and training log are valuable but currently conflate plans and results. A concise current-state index should link to this report and mark superseded conclusions without deleting experimental history. At audit start the tree was clean and `main` was two commits ahead of the locally known `origin/main`. No remote fetch or push was performed; no model-training or pipeline behavior is modified by this report.

## Roadmap Without a New Manual Gold Set

### Priority 0: Fix Confirmed Information Loss

1. Preserve OCR text/confidence/class in reference JSON. Add round-trip tests from known detector output to JSON and SVG/DXF, including an unknown-text case and a no-double-rendering case. Use existing synthetic reference labels for exact expected values.
2. Give hatch regions explicit source-edge membership and preserve every unmatched separated edge as geometry. Add mixed-region/ungrouped-edge tests, then replay the eight confirmed affected drawings and the full existing 100-patent cohort. Check representation coverage, overfill, structural recall and export continuity, not just primitive counts.
3. Add a deployment preflight that requires the canonical config and intended weights, records effective settings, and rejects unexpected fallback. Make region mode and other important defaults explicit without changing their selected behavior.

**Acceptance:** known OCR tokens survive serialization; unknown labels are not fabricated; no separated hatch edge lacks an export representation; required model loading cannot silently substitute another chain; existing tests and matched CAD/patent regressions pass. None of these tasks requires new manual labeling.

### Priority 1: Reuse Existing Evidence for Content and Deployment Checks

1. Reuse the 100 already-audited pilot-v3 content labels, existing positive/negative hatch figures, and known chart/chemistry failures. Evaluate content routing separately from vectorization. A small classifier or OCR-informed router can be assessed with grouped cross-validation on existing labels; keep ambiguous cases quarantined.
2. Run a paired Stage 2 comparison between canonical PatentVec A tiled and the older Phase A fusion chain, holding preprocessing and the current Stage 3/4 policy fixed. Report success transitions, structure recall, fragmentation, hatch accounting, distances and runtime. Do not infer this result from detector F1.
3. Run a small fresh Stage 0-4 canonical smoke batch on CPU and, separately, GPU. Check chosen methods, output entities, missing crops, coordinate conventions and geometry agreement within a declared tolerance.
4. Re-score the region detector at the deployed 0.7 threshold using existing masks. Report positive/negative figure behavior and structural preservation. For future retraining, regroup the existing annotation pool by patent; no new annotations are needed, although the resulting test is not independent of past experimentation.

**Acceptance:** known invalid examples are routed out of the training manifest without losing the existing hard-positive examples; uncertain items have a separate disposition; comparison claims use matched denominators and recorded inference settings. Existing test material becomes a transparent regression suite, not a newly untouched gold benchmark.

### Priority 2: Improve Targeted Geometry and Efficiency

Profile the slow Stage 3 tail; reuse repeated computations and add safe early rejection while preserving the current accepted primitive as fallback. Investigate the remaining six Stage 2 and three Stage 1 gates individually before assuming more model capacity is needed.

Evaluate the existing opt-in dashed-line/circle grouping on synthetic hidden-line masks and the already-labeled `dashed_line_fragmentation` case. Validate both appearance and style semantics before enabling it. Add metamorphic tests for translation, scale, rotation and harmless annotation insertion/removal. These expose consistency problems without requiring hand-drawn vector truth.

**Acceptance:** no material fidelity or preservation regression, bounded tail runtime, and measured improvement in the targeted failure family. A global average alone must not justify deleting hard cases.

### Priority 3: Decide Whether More Synthetic Data Is Worth Generating

Keep the A/B corpora and trained models frozen as comparators. Quantitatively compare existing synthetic and filtered-real distributions: component counts, contour lengths, crossing density, hatch spacing/angles, reference density, dashed-line supply and primitive vocabulary. Use that diagnosis to target underrepresented structures rather than defining progress as a larger sample count.

Before another training run, choose the Stage 2 input/label contract deliberately. Do not silently reuse the original A/B sparse labels for dense topology training, and do not presume C2/C3 solves real transfer because its labels are internally correct. New synthetic validation should be source-disjoint at the original sketch/CAD-model level, including sibling D2C views and source components.

Only then consider a bounded targeted increment or the planned 50k run in `/media/safe/secondary disk/IPdrawings`, with fresh throughput/storage measurements for the actual curriculum. Keep multi-domain rehearsal and protected warm-start selection. Do not restart Free2CAD training unless a new fitter architecture or demonstrated failure-specific benefit justifies it.

Include an equal-budget real-only rehearsal control with the same real domains. This separates the effect of additional synthetic examples from the effect of adding ArchCAD rehearsal or simply continuing optimization.

### Priority 4: Build the Actual Text-to-Vector Training Product

Define the target explicitly: editable 2D SVG/DXF does not require a parametric 3D CAD program, whereas CAD construction-sequence generation does require the separate Track-B effort. Decide how references, text, hatch patterns, hidden lines, dimensions, scale and units are encoded.

Build description/vector pairs with provenance, content disposition, actual quality metrics and uncertainty. Separate synthetic ground truth from real-patent pseudo-labels, split by patent and by source CAD identity as appropriate, and validate that exports parse and render. Start with the existing small curated corpus after the preservation fixes; expand only with measured filtering and reconstruction behavior. Artifact distribution permissions and source-provenance review remain separate release work, not a model-quality score.

## Practical Release Position

A large new manual gold set is not an immediate prerequisite. The next release can use three existing evidence tracks:

- **Exact contracts:** synthetic semantic masks, known vectors, reference round trips, hatch ownership, SVG/DXF parsing and coordinate/continuity checks.
- **Paired reconstruction:** full Drawing2CAD plus the existing frozen real-patent cohorts, retaining failed and nonfinite cases in denominators and reports.
- **Existing semantic review:** pilot-v3 labels and reviewed hatch annotations, with patent overlap and prior exposure disclosed.

This supports an evidence-backed, guarded deployment and continued improvement without imposing a major annotation project. It cannot establish absolute precision/recall for every rare real-patent reference, hatch, formula or mechanical configuration. The responsible near-term objective is **no known avoidable information loss, reproducible execution, explicit content routing, and measurable non-regression on existing evidence**. More training is secondary to those concrete engineering tasks.

## Sources and Reproduction

Regenerate the local evidence snapshot from the repository root with:

```bash
.venv/bin/python docs/audits/2026-09-09/collect_evidence.py \
  --output docs/audits/2026-09-09/evidence.json
```

The snapshot references local artifacts, many of which are Git-ignored. It is an inventory and diagnostic record, not a redistribution package. Reproduction requires the original datasets, saved output directories and local weights.

[^audit]: [Live evidence snapshot](audits/2026-09-09/evidence.json), generated by [the audit collector](audits/2026-09-09/collect_evidence.py). Includes weight hashes, YAML differences, dataset counts, SQLite status, reference-text counts, hatch representation coverage, split overlap and runtimes.
[^production]: [Canonical deployment configuration](../config_deploy.yaml) and [production checkpoint register](../models/README.md), sections "Production checkpoints" and "Why complexityA is the deployed Stage 2". Compare the saved [replay configuration](../output/PatentData100_Stage3GuardedP2FixedFrozen/stage34_replay_config.yaml).
[^refs]: [Stage 0 implementation](../stage0_handling_references/stage0_handle_references.py), `_build_reference_doc` around line 811, baseline unconditional empty text at line 857, OCR token retention at line 1067, and `annotations_from_reference_json` around line 1690. These line numbers describe the pre-fix audit snapshot.
[^hatchcode]: [Stage 2 hatch aggregation](../stage2_strokeextraction/stage2_stroke_extract.py), `_aggregate_hachure_regions` around line 1538, convex-hull construction and early region cleanup around line 4887; [Stage 3 hatch branch](../stage3_primitivesfitting/stage3_primitive_fit.py) around lines 1611-1628.
[^export]: [Stage 4 exporter](../stage4_export/stage4_export.py), SVG image/text rendering around lines 513-534 and DXF annotation MTEXT around lines 736-768.
[^dash]: [Stage 2 dashed grouping](../stage2_strokeextraction/stage2_stroke_extract.py), `_group_dashed_edges` around line 2352 and opt-in invocation around line 4968.
[^pilot]: [Pilot-v3 baseline report](../benchmarks/pilotv3/BASELINE_REPORT.md) and [existing 100-figure content labels](../benchmarks/pilotv3/pilotv3_labeled_audit.csv). Historical filter/version and cohort differ from the later 100-patent Stage 3 replay.
[^training]: [Training execution record](../TRAIN_STAGE2_STAGE3.md), synthetic A/B and C2 sections at the start, full-data sections in the middle, and executed hatch/Stage 3 results from section 12 onward. Historical "pending" text is superseded where completed artifacts are cited here.
[^visual]: [Recovered Stage 3 outputs](../output/PatentVecHatchStrokeTraining/stage3_guarded_final100/recovered_stage3_first8.png), inspected in this audit. It contains mechanical figures alongside plots and chemistry; it is not a content-purity certificate.
[^runtime]: [Batch worker initialization and CLI](../tools/batch_run.py), lines 131-165 and 716-719; [Puhachov and hatch model loaders](../stage2_strokeextraction/stage2_stroke_extract.py), around lines 1673 and 3990; [conditional preprocessing](../stage1_preprocessing/stage1_preprocess.py), around lines 474-511.
[^synthetic]: [Synthetic implementation/status](../syntheticData/README.md), [A10k generation manifest](../output/PatentVecComplexityA10k/manifest.json), [B10k generation manifest](../output/PatentVecComplexityB10k/manifest.json), and [B visual audit](../output/PatentVecComplexityB10k/audit_stratified/index.html).
[^roadmap]: [Synthetic composition roadmap](../syntheticData/PATENTVEC_SYNTHETIC_COMPOSITION_ROADMAP.md), milestones M0-M11, especially M7 realism calibration and M10 Track-B CAD generation.
[^coverage]: [Full Phase A coverage](../output/SketchGraphsFull/stage2_train.coverage.i8.json) and [full Phase B coverage](../output/SketchGraphsCADVG40/stage2_train.coverage.i8.json).
[^sgtest]: [Phase A full SketchGraphs test](../output/SketchGraphsFull/evaluation/phaseA_test.json) and [older production full test](../output/SketchGraphsFull/evaluation/production_test.json).
[^phaseb]: [Full paired Phase A versus Phase B D2C evaluation](../output/Drawing2CAD/full_test_phaseA_vs_phaseB.json). Pixel metrics have 31,524 pairs; distance metrics have 31,497 finite pairs in this historical experiment.
[^stage2eval]: [A/B evaluation directory](../output/PatentVecComplexityABTraining/stage2/guard4way), [C2 evaluations](../output/PatentVecComplexityABTraining/stage2/reference_free_topology), and [C3 evaluations/training log](../output/PatentVecComplexityABTraining/stage2/hatch_suppressed_topology). Scalar values and protocol fields are captured in the audit evidence JSON.
[^mixercode]: [Puhachov trainer](../stage2_strokeextraction/research/train_puhachov.py), replay and guard-pool loading around lines 759-768; both explicitly collect the `train` split.
[^patentmodels]: [Historical matched PatentData model fidelity](../output/PatentData100_model_comparison/fidelity.json) and [C3 synthetic oracle integration diagnostic](../output/PatentVecComplexityABTraining/stage2/topology_integration_C3_oracle_100.json). These experiments predate the final guarded Stage 3 policy.
[^free2cad]: [Full Stage 3 test result](../output/SketchGraphsStage3Full/evaluation_best_full_test.json) and [mixed five-class training/evaluation record](../TRAIN_STAGE2_STAGE3.md), sections "Mixed SketchGraphs + Drawing2CAD" and full Stage 3 extraction/training. The local research model is a per-edge fitter, not a trained text-to-CAD generator.
[^fittercomparison]: [README fitter comparison](../README.md), around lines 1089-1100: the small 18-graph production-cascade comparison, distinct from large classifier test evaluations.
[^hatchtrain]: [Hatch grid-v2 selected result](../models/hatch_grid_v2/best_config.json), [saved splits](../models/hatch_grid_v2/splits.json), [trainer](../tools/hatch_train_cnn.py), `figure_iou` and `validate`, and [dataset split implementation](../tools/hatch_dataset.py). The installed checkpoint metadata agrees with the selected result.
[^hatchstroke]: [Executed hatch-stroke adaptation record](../TRAIN_STAGE2_STAGE3.md), section 12; [reviewed-real export manifest](../output/PatentVecHatchStrokeReal/manifest.json); [residue-fix fidelity](../output/PatentVecHatchStrokeTraining/patentdata100_residue_fix_fidelity.json); [guarded stroke-CNN end-to-end results](../output/PatentVecHatchStrokeTraining/hatchstroke_real50_full100_e2e/fidelity.json).
[^stage3]: [Executed Stage 3 promotion record](../TRAIN_STAGE2_STAGE3.md), "Stage 3 confidence policy: completed and promoted (2026-08-14)"; [primitive-fitting implementation notes](stage3_primitive_fit.md); [export notes](stage4_export.md).
[^d2cstage3]: [Full D2C replay report](../output/Drawing2CAD/stage3_guarded_p2_fixed_full/stage34_replay_report.json) and [paired policy analysis](../output/PatentVecHatchStrokeTraining/stage3_guarded_p2_fixed_full_analysis.json).
[^patentstage3]: [100-patent replay report](../output/PatentData100_Stage3GuardedP2FixedFrozen/stage34_replay_report.json), [paired fidelity](../output/PatentVecHatchStrokeTraining/stage3_guarded_final100/fidelity.json), and [source-status subgroup analysis](../output/PatentVecHatchStrokeTraining/stage3_guarded_final100/fidelity_by_source_status.json).
[^fidelitycode]: [Patent fidelity evaluator](../tools/evaluate_patent_fidelity.py), module scope and `_graph_mask`, `_primitive_mask`, `fidelity`, and `evaluate`; its reference is the shared Stage 1 skeleton, not independent hand-vectorized patent ground truth.
