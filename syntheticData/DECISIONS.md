# PatentVec decisions

1. Similarity transforms are used so line, arc, circle, and Bezier identities
   remain exact. Patent degradation does not geometrically warp labels.
2. A T-junction splits only the visible host primitive. The original complete
   primitive remains in the amodal graph with exact parent-child intervals.
3. Every source component in one composite comes from the composite split.
   CAD views inherit the split of their object identifier.
4. Hatching, references, leaders, text, and centre lines remain separate
   semantic layers. They are not relabelled as object geometry.
5. The initial archive-v1 Puhachov inputs retain semantic distractors and store
   separate masks. This historical contract is diagnostic only; decision 28
   supersedes it for reference-free Stage 2 training.
6. Track-A output can train the local Free2CAD edge fitter only. It must never
   be described or consumed as a valid CAD command program.
7. Candidate generation rejects unintended cross-component intersections,
   missing provenance, invalid source intervals, out-of-canvas geometry, and
   blank required masks.
8. Full generation remains disabled until the visual pilot is approved.
9. Medium and hard samples are accepted by graph and class-supply gates, not
   by raster density alone. Hard samples require a cycle and at least three
   interaction types.
10. Connected crossings split both visible lines exactly. Their complete
    source primitives remain available in the amodal graph.
11. Patent annotations are generated after object composition and remain in
    dedicated semantic layers; they do not create source-component topology.
12. The M5/M6 command is a restart-safe review pilot capped at 50 samples. It
    cannot write the planned full-generation dataset.
13. Experimental generation uses compact tar shards and a reusable source
    index. Full 50k generation remains a separately gated mode and root.
14. Free2CAD targets must be projected through the real Stage 2 topology path;
    primitive-level labels that bypass extracted edges are diagnostic only.
15. Polyline supply is enforced per drawing and independently reconstructed
    from skeleton/keypoint inputs before a dataset can enter training.
16. Periodic curricula must not use modulo-based audit or validation sampling.
    Audits are stratified and native exports use a BLAKE2 source-index split.
17. Synthetic validation drawn from the training source pool is diagnostic,
    not source-disjoint. Model selection remains on fixed real-domain holdouts.
18. Complexity comparisons account for curriculum-specific retry rejection:
    full-sample unpaired intervals and exact-seed paired intervals must agree.
19. Stage 3 A/B datasets are balanced by trainer-visible label supply, not
    drawing count. Real rows, class/domain schedules, validation, and total
    optimizer budget are held equal across variants.
20. Every warm-started A/B experiment retains and evaluates step zero. A
    fine-tuned state cannot become the selected model by existing at the end
    of the run; it must improve the declared holdout criterion.
21. Stage 2 synthetic rehearsal keeps exact synthetic coverage and explicit
    Drawing2CAD, SketchGraphs, and ArchCAD replay. Real-domain forgetting is a
    failed experiment even when synthetic validation improves.
22. The Stage 3 adapted states are retained for Pareto analysis, but the shared
    real-data warm start remains selected because neither A nor B improves its
    source-disjoint validation score.
23. The 50k generation remains blocked until the selected Stage 2 candidate
    passes full real-domain evaluation and filtered-PatentData visual
    regression. Stage 3 synthetic gains alone cannot release generation.
24. Filtered pilot sizes are counted after content filtering. Use
    `tools.batch_run --limit-after-filter`; legacy pre-filter limit semantics
    remain available only for reproducing older reports.
25. Filtered-PatentData model arms share one canonical Stage-0/1 result. A
    candidate may reuse it only when the input identity and all Stage-0,
    Stage-1, and SketchCleanNet configuration blocks match; copied annotation
    paths must remain self-contained.
26. Lower edge counts are not accepted as continuity evidence by themselves.
    Deployment requires paired fragmentation metrics plus Stage-2/Stage-3
    raster fidelity against the shared Stage-1 skeleton, quality-gate
    transitions, and visual review of hachure-heavy and worst-regression cases.
27. Code licenses do not automatically clear source data or generated
    derivatives. Track-A generation remains local to the research workspace;
    unrestricted redistribution or commercial release requires written/legal
    confirmation recorded in `syntheticData/LICENSES.md` and the dataset card.
28. Stage 2 synthetic supervision must reproduce the post-Stage-0 raster, not
    selected vector keypoints. The training skeleton contains object,
    hidden/centre-line, and hatch masks and excludes references, leaders,
    dimensions, and text before topology labels are computed.
29. Complete-raster C1 labels are retained as diagnostic evidence but cannot
    train the deployable model because annotation contacts create precisely the
    false junctions and stroke fragmentation Stage 0 is designed to prevent.
30. Dense topology rehearsal uses per-sample/per-class focal reduction so
    synthetic junction volume cannot starve endpoint and corner learning. The
    historical global focal reduction remains the default for old experiments.
31. C2 promotion requires complete real-domain evaluation plus paired
    PatentData fragmentation, raster-fidelity, quality-gate, and visual checks.
    A synthetic-domain gain cannot by itself release the 50k generation.
32. Hachure handling must classify existing Stage 1 skeleton pixels; explicit
    Hough junction insertion is diagnostic only because paired controls showed
    that even restricted T/X routing damages structural continuity.
33. Structural and hatch stroke labels are multilabel at true crossings. Their
    union must reproduce every input skeleton pixel, and ambiguity defaults to
    structural preservation rather than deletion.
34. Exact semantic-mask support is the training default. A one-pixel support
    dilation was rejected after it increased 128-sample overlap labels by 142%
    while exact support already had zero unassigned pixels.
35. A hatch-stroke checkpoint cannot be selected on hatch score alone. It must
    first meet the structural-recall floor and then pass synthetic validation,
    real hachure-heavy PatentData fidelity, fragmentation, quality-gate, and
    visual comparisons before any production Stage 2 integration.
36. Synthetic hatch-stroke calibration is not a deployment gate. The selected
    synthetic checkpoint is retained as a warm start only because destructive
    pre-topology use increases real PatentData fragmentation, replacement use
    suppresses the stronger reviewed-region detector, and additive use is inert.
37. Region-based hachure separation remains the production path. Matching
    residue must be moved into `removed_hachures` before metrics and followed
    by reconnect-only simplification; residue may not be silently deleted after
    metrics or omitted from the exported side layer.
38. Reviewed hatch polygons are area annotations, not exact stroke labels.
    Real-domain targets therefore use independent structural and hatch
    supervision masks. Unexplained region ink, annotation boundaries, and
    unsupported crossings are ignored rather than assigned a destructive
    negative label.
39. Mixed hatch-stroke training is balanced by domain supply per optimizer
    epoch, not raw file count. Checkpoint selection must satisfy the structural
    floor on both synthetic and reviewed-real validation before real safe-hatch
    score can rank candidates; final release still requires paired end-to-end
    PatentData evidence.
40. Reviewed hatch labels require exact region-and-Hough teacher consensus.
    Full no-hachure graph recovery is rejected because its sparse additions are
    dominated by false structural contours.
41. Synthetic difficulty gates are baseline-relative while aggregate synthetic
    structural recall remains at least 0.995. An absolute 0.995 floor on every
    subgroup is invalid because the untouched checkpoint starts at 0.993482 on
    very-hard samples.
42. Deployment calibration must select one identical hatch/structural threshold
    pair across synthetic and reviewed-real validation. Per-domain thresholds
    are diagnostic only and cannot be combined into a deployment claim.
43. Unclaimed Stage 2 components are closed only when their pixel adjacency is
    a true simple cycle. Branched residuals above 100,000 pixels quality-gate;
    Stage 3 must use linear simple-cycle traversal and bounded malformed-network
    traversal rather than quadratic nearest-neighbour ordering.
44. First-round real35 and real50 checkpoints are rejected despite large hatch
    gains because neither reaches 0.995 reviewed-real structural recall and no
    global safe-removal policy exists. Follow-up training changes supervised
    structural-negative supply while keeping the release gates fixed.
45. Structural-negative weights 4 and 8 are rejected after one epoch because
    both collapse synthetic and reviewed-real structural recall before producing
    useful safe-removal recall. More loss reweighting is not the next hatch path.
46. Region and stroke-model masks must never be unioned before edge
    classification. Model-only evidence is source-separated and must pass the
    existing short, straight, repeated-family geometric gate; long structural
    chains are preservation vetoes.
47. The real50 guarded additive integration is rejected as safely inert after
    exact 20-case and 100-case ties through Stage 4 and raster fidelity. A model
    that contributes zero eligible edges does not justify production inference
    cost, even when it introduces no regression.
