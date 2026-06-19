# Learned hatch-region detector — scope

## Why this exists
Section hatching is ~45 % of all primitives and the worst remaining output problem.
The parametric-HATCH representation (Stage 2-4, `hachure_mode=region`) is **correct**
but currently **disabled** because the upstream detector (`_remove_hachure_edges`)
has poor precision: it flags dense mechanical line-work as hatching (e.g. 581 false
"hatch" lines on EP3120975B1, an external drill view with no section hatching),
which HATCH then smears as fills over real geometry.

**Heuristics can't fix the detector.** Five hand-crafted discriminators were tested
to separate genuine hatching from structural/texture line-work — spacing-CV, line
density, FFT periodicity, gradient orientation-coherence, 2D fill-uniformity — and
**none separate individually** (drill body FFT 64 > cross-hatch 31; drill grip
orientation-coherence 0.50 > diagonal hatching 0.31). The detector also has poor
recall (misses FIG.1 diagonal hatching). → a **learned** detector is required.

## Goal
A detector that decides, per region/pixel, **hatched-fill vs not**, with:
- High **precision** (never smear a fill over the drill / non-hatched geometry).
- Reasonable **recall** (catch real section hatching: single diagonal + cross-hatch).
- A tight **boundary** (no convex-hull over-cover of concave/disjoint fills).
Output feeds the existing `hachure_regions` → HATCH export, which is already built.

## Approach — two phases (start cheap, upgrade only if needed)

### Phase 1 — Region re-classifier (recommended first)
"Detector proposes, classifier disposes." Keep the current detector as a *proposal*
generator (good recall on cross-hatch), and train a classifier to accept/reject each
candidate region. This directly kills the false positives (the user's actual
complaint) with the **least labeling**.

- **Key bet:** the 5 features fail *individually*, but a classifier over the *joint*
  feature vector may separate the classes. De-risk early (see Validation): if
  held-out AUC is low, the classes genuinely overlap in this feature space → skip to
  Phase 2 instead of polishing a dead end.
- **Features per candidate region** (cheap, interpretable): orientation-histogram
  entropy + dominant-mode mass, FFT dominant-frequency power & sharpness, 2D
  fill-uniformity (grid-cell occupancy), spacing-CV, line count, ink density,
  bbox aspect, region area, and an **enclosure** signal (fraction of the region
  boundary that coincides with a closed graph contour — genuine hatching is bounded
  by an outline; surface texture is not).
- **Model:** gradient-boosted trees (sklearn/LightGBM) or a small MLP. Fast, runs
  CPU, no GPU. Train/val split **by figure** (never leak a figure across splits).
- **Effect:** drill regions rejected → stay per-line (safe); cross-hatch regions
  accepted → HATCH. Diagonal hatching the detector misses still falls back to
  per-line (safe, just not upgraded). This makes HATCH *safe*, if not complete.

### Phase 2 — Pixel/patch segmentation CNN (only if recall matters)
If Phase 1's "miss the diagonal hatching" recall gap is unacceptable, train a small
**patch classifier or U-Net** on the cleaned image to produce a pixel-wise hatch
mask (fixes precision AND recall, independent of the proposal detector). Reuses the
existing CNN training harness (`stage2_strokeextraction/research/train_puhachov.py`).
Heavier: more labeling (pixel masks) + GPU training.

## Data & labeling plan
- **Source:** 28,768 patent dirs; sample figures from the curated
  `output/PatentData_clean12_gated/training_manifest_clip.csv`, stratified so ~50 %
  are likely-hatched (cross-sections) and ~50 % not (external views like the drill),
  so the classifier sees both classes.
- **Volume (Phase 1):** ~7 candidate regions/figure → labeling **~150 figures ≈
  ~1,000 region true/false labels**. At ~10-20 s/region that's **~2-4 h** of your
  time. A 60-figure pilot (~400 labels, ~1-1.5 h) is enough to run the de-risk AUC
  check first.
- **Labeling tool:** new `tools/hatch_label_sample.py` + `hatch_label.py`, reusing
  the `patent_gold_sample.py` overlay pattern — render each figure with its candidate
  regions outlined and numbered; you mark each region hatch / not-hatch (one keypress
  per region). Writes `<id>_regions.json` {region_id, label}.
- **Phase 2 (if pursued):** brush/polygon hatch masks on ~150-300 figures (heavier).

## Integration
- Phase 1: in Stage 2, after `_aggregate_hachure_regions`, score each region with the
  classifier; drop rejected regions (their edges stay as geometry, fit per-line).
  Re-enable `hachure_mode=region` only when precision clears a bar on the held-out set.
- **Boundary fix (independent, needed regardless):** replace the convex hull with a
  concave hull (alpha-shape) or clip the fill to the enclosing closed contour, so the
  HATCH doesn't over-cover concave/disjoint regions.

## Validation
- **De-risk gate (do first, after the pilot):** train on pilot labels, report
  by-figure-split **ROC-AUC / precision@high-recall**. AUC ≳ 0.9 ⇒ Phase 1 viable;
  low ⇒ go to Phase 2.
- **Acceptance:** region precision ≥ ~0.95 (few false fills) at usable recall;
  end-to-end visual on held-out figures — drill stays clean (no smear), genuine
  cross-hatch filled. Add a `tests/` regression with a few known hatched/non-hatched
  figures.

## Effort estimate
- Phase 0 — labeling tool + pilot labels (~60 figs) + de-risk AUC: ~1 day eng + ~1.5 h labeling.
- Phase 1 — full labels (~150 figs) + train + integrate + boundary fix + tests: ~2-3 days eng + ~2-4 h labeling.
- Phase 2 (optional) — mask labeling + U-Net + integrate: ~1-2 weeks eng + several h labeling.

## Risks / decisions
- **Label ambiguity:** fine texture vs light hatching is sometimes genuinely unclear;
  define a rule (e.g. "regular parallel fill bounded by an outline = hatch").
- **Class overlap:** the de-risk AUC may show region features don't separate → Phase 2
  required (more cost). The pilot answers this before heavy investment.
- **Generalization:** patent drawing styles vary; stratified sampling + by-figure
  splits mitigate.
- **Decision needed from you:** approve the ~1.5-4 h labeling budget, and whether to
  cap at Phase 1 (safe HATCH, partial recall) or commit to Phase 2 (full coverage).
