# Free2CAD — Handoff for deeper investigation

> Goal of this doc: a self-contained brief for a fresh investigation into
> why our trained Free2CAD model under-performs RANSAC on real Stage-2
> output, summarising three rounds of data-generator + training-strategy
> redesign and the open problems each round failed to solve.
>
> Attached scripts: `generate_sketches_v2.py`, `train_free2cad_v2.py`,
> `stage3_primitive_fit.py`. All numbers below are measured on a real
> input graph (`Picture1_skeleton_graph.json`, 817 edges, see §3).

---

## 1. Pipeline context

Free2CAD is **Stage 3** of an AP3 sketch-vectorisation pipeline:

```
raw sketch (PNG)
   │
   ▼
Stage 1: SketchCleanNet (clean + skeletonise)            → cleaned PNG
   │
   ▼
Stage 2: Puhachov stroke graph (Stroke_Extraction)       → graph JSON
   │   { "edges": [{ "id", "pixels":[(x,y)…], "smooth_pts":[…],
   │                  "is_closed": bool, … }, …] }
   ▼
Stage 3: Free2CAD primitive fitting                      → primitives JSON
   │   For each edge → one of {LINE, ARC, CIRCLE, POLYLINE}
   ▼
Stage 4: SVG / DXF export                                (downstream)
```

**Stage 3** has two fitter implementations selectable via `--fitter`:

- `ransac` — pure NumPy geometric fit, always available, used as
  ground-truth-style baseline for evaluation here.
- `free2cad` — a custom PyTorch Transformer trained on synthetic data,
  the subject of this investigation.

The official upstream Free2CAD repo (TensorFlow, Li et al. SIGGRAPH 2022)
is not used — we build our own equivalent in PyTorch. Our model:

```
StrokeEncoder  : Transformer encoder over (stroke_0, …, stroke_S)
                  each stroke = 32 (x, y) pts → flat 64-d feature
                  positional embedding on stroke index, pad mask
CommandDecoder : autoregressive Transformer decoder
                  type embedding (6 classes) + param projection (6-d) + pos emb
                  causal mask, cross-attention to encoder output
                  outputs:  type_logits (B, T, 6),  params (B, T, 6)
```

`max_strokes = 20`, `max_pts = 32`, `d_model = 256`, 4 enc + 4 dec layers,
~7.4 M trainable params. Vocabulary:

```
LINE=0  ARC=1  CIRCLE=2  POLYLINE=3  END=4  BOS=5
```

Each training sequence is `[BOS, c₀, c₁, …, c_{K-1}, END]` length K+2.
Teacher-forcing slices: `tgt_in = seq[:-1]`, `tgt_out = seq[1:]`. Position
0 of decoder output is therefore conditioned on BOS and predicts `c₀`.

Parameter layout (6 floats, all values normalised to [0, 1]):

```
LINE   : [x0, y0, x1, y1, 0, 0]
ARC    : [cx, cy, r,  sa/360, ea/360, 0]
CIRCLE : [cx, cy, r,  0, 0, 0]
POLYLINE / END : zeros (POLYLINE shape carried by the stroke, not params)
```

---

## 2. Round 1 — v1 generator + v1 training

**v1 training data** (`generate_sketches.py`, since superseded). Five
shape families:

```
rect       25%   →  4 LINE
l_shape    20%   →  6 LINE
circle     15%   →  1–2 CIRCLE
arc_box    20%   →  1 ARC + 3 LINE
multi_line 20%   →  3–6 LINE
```

Measured class distribution on 25 500 generated training samples
(105 069 commands):

| LINE | ARC | CIRCLE | POLYLINE |
|---|---|---|---|
| 89.7 % | 4.9 % | 5.4 % | 0 % |

POLYLINE was in the vocabulary but never emitted by the generator.

**v1 training**: vanilla CE loss, no class weights, no BOS token.
Decoder off-by-one: `tgt_out = cmd_types[1:]` so position 0 of output
predicts `c₁` (the second command). At inference there's no `c₀` to seed
with — the wrapper has to fake one.

**v1 model behaviour on real data** (epoch 98, val_loss 0.0064):

| 817 edges | LINE | ARC | CIRCLE | POLYLINE | mean conf |
|---|---|---|---|---|---|
| Free2CAD | **817 (100 %)** | 0 | 0 | 0 | 1.000 |
| RANSAC | 713 | 55 | 4 | 45 | 0.867 |

**Diagnosis: class collapse.** The Bayes-optimal trivial classifier under
a 90 % LINE distribution is "always LINE", which is what the model
learned. Reaching val_loss 0.0064 was the *symptom*, not a sign of
quality — the trivial baseline is also accurate on the trivial-LINE
distribution.

---

## 3. Round 2 — v2 redesign

Two scripts replaced:

- `generate_sketches_v2.py` (rebalanced data + new shape families)
- `train_free2cad_v2.py` (BOS, class weights, label smoothing, per-class
  metrics)

### 3.1 Data: rebalanced shape mix

```
rect             0.04        polyline_curve   0.20
l_shape          0.02        rounded_rect     0.04
multi_line       0.03        short_stubs      0.14    NEW (round 3)
circle           0.10        noisy_circle     0.13    NEW (round 3)
arc_box          0.08
arc_only         0.16        + new families: arc_only, circle_cluster,
circle_cluster   0.06          polyline_curve, rounded_rect, mixed_assembly
```

(Round 3 additions are described in §4. Round 2 used the same families
*without* `short_stubs` and `noisy_circle`.)

A `mixed_assembly` composite stitches 2-3 sub-shapes into horizontal
stripes of a sample, controlled by `--mixed_fraction` (default 0.15).

### 3.2 Training: §5 of the original roadmap

- **BOS token** prepended to every sequence so position-0 decoding is
  well-defined.
- **Class-weighted CE**: weights computed from training set frequencies
  via `_compute_class_weights`, capped at `--max_class_weight`.
- **Param L1 gated** on `{LINE, ARC, CIRCLE}`; POLYLINE / END contribute
  no parameter signal (zeros by design).
- **Label smoothing** 0.05 to prevent softmax saturation.
- **Per-class validation metrics**: confusion matrix, P/R/F1 per class,
  mean L1 per param-class, prediction-distribution entropy
  (collapse detector).

### 3.3 Round 2 result (initial v2 data, 149 epochs, val_loss 0.22)

| 817 edges | LINE | ARC | CIRCLE | POLYLINE | RANSAC type agreement |
|---|---|---|---|---|---|
| Free2CAD-v2 | 0 | 33 | 0 | **784 (96 %)** | 7.8 % |

**The collapse moved from LINE to POLYLINE.** Per-class validation
metrics on v2 synthetic data were healthy (P/R 0.62-0.96 for all four
classes, prediction entropy 0.89). But predictions on real Stage-2
output were degenerate.

### 3.4 What the input graph actually looks like

Before reaching for a third redesign, we measured the input graph the
model was being asked to handle:

| Property | Value |
|---|---|
| total edges | 817 |
| closed loops | 4 |
| edge length p50 / p90 | 8 px / 32 px |
| points per edge: median | **2** |
| points per edge: distribution | 2pt → 69 % · 3-5pt → 17 % · 6-15pt → 3 % · 16+pt → 11 % |

Compare with v2-initial training distribution (same metric, 25 500
samples, ~105 k strokes):

| points per stroke | 2pt → 0 % · 3-5pt → 0 % · 6-15pt → ~30 % · 16+pt → ~70 % |

**The model had literally never seen a 2-point stroke at training time.**
Real Stage-2 output is dominated by them. With `max_pts = 32`, a 2-point
edge is encoded as `[x0, y0, x1, y1, -1, -1, …, -1]` — 30 padding
sentinels. That input is OOD for the encoder, and the model's safest
output is the highest-entropy class, which on this distribution is
POLYLINE.

---

## 4. Round 3 — v2 data updated, retrained

Two new shape families added to `generate_sketches_v2.py`:

### 4.1 `short_stubs` (14 % of samples)

5-20 short LINE stubs of 2-4 points each per sample, half forming a
chain (endpoint of one = start of next). Mimics the dense 2-point edges
that Stage 2 produces at skeleton branch points.

### 4.2 `noisy_circle` (13 % of samples)

Small circle (`r ∈ [0.04, 0.18]`), only 12-30 sample points, 1.5-3×
extra noise on top of the global `noise_std`. Mimics the small wobbly
closed loops that real skeletons produce.

`polyline_curve` length range was also widened from 20-40 → 8-40 points.

### 4.3 Updated training distribution

| LINE | ARC | CIRCLE | POLYLINE | strokes ≤ 5 pts |
|---|---|---|---|---|
| 64 % | 15 % | 17 % | 4 % | **31 %** (was 0 %) |

### 4.4 Round 3 result (updated data, 169 epochs, val_loss 0.29)

Synthetic-set validation metrics are essentially perfect:

```
LINE     : P=1.000  R=1.000  F1=1.000  param_L1=0.003
ARC      : P=0.999  R=0.999  F1=0.999  param_L1=0.010
CIRCLE   : P=1.000  R=1.000  F1=1.000  param_L1=0.003
POLYLINE : P=0.997  R=0.995  F1=0.996
pred_entropy = 0.984
```

Real-data result on the same 817-edge graph:

| 817 edges | LINE | ARC | CIRCLE | POLYLINE | mean conf | RANSAC agree |
|---|---|---|---|---|---|---|
| Free2CAD-v3 | **322** | 17 | **0** | 478 | 0.918 | 46.3 % |
| RANSAC | 713 | 55 | 4 | 45 | 0.867 | (self) |

**LINE class is alive.** 322 / 817 lines now correctly classified, vs 0 in
round 2. On 2-point edges (567 of them) it's 50/50 LINE/POLYLINE.
Confidence is also informative for the first time — mean dropped from
0.99 to 0.92, with meaningful spread.

But two failures persist:

- **CIRCLE remains 0 / 817 in deployment** despite F1 = 1.000 on
  synthetic val. The 4 closed loops (edges 813-816, 42-52 pts, span
  ~20 px, geometric circles) are all classified POLYLINE at confidence
  1.00. RANSAC correctly flags all 4 as circles.
- **LINE parameters under-predict length by ~45 %.** Spot-checks of the
  first 5 LINE predictions:

| Edge | predicted line length | actual edge span | ratio |
|---|---|---|---|
| 2 | 0.5 px | 1.0 px | 0.50 |
| 4 | 11.0 px | 20.0 px | 0.55 |
| 5 | 3.1 px | 6.0 px | 0.52 |
| 7 | 4.4 px | 8.0 px | 0.55 |
| 8 | 22.3 px | 41.0 px | 0.54 |

Predicted endpoints land at roughly the 25 % / 75 % marks of the edge's
own bbox instead of 0 % / 100 %. The *type* is right; the *parameters*
are systematically compressed.

---

## 5. Inference-side experiment (no retrain)

Hypothesis: at training time, individual strokes occupy small sub-regions
of the per-sample [0, 1]² canvas (especially `short_stubs`); at inference
each edge is normalised to fill its own canvas. That OOD scale could
explain both the CIRCLE failure and the LINE-length compression. Test:
encode each inference edge into a sub-region of [0, 1]² instead of the
full canvas.

Implemented in `Free2CADFitter._encode_edge`:

```python
pts_norm = (pts - center) * (frac / scale) + 0.5
```

`frac` is the new `_INFERENCE_FRACTION` knob; the inverse is in
`_decode`. Probed several values:

| frac | LINE | POLYLINE | ARC | line_len_ratio (median) | RANSAC agree |
|---|---|---|---|---|---|
| 0.4 | 89 | 726 | 2 | 0.60 | 16 % |
| 0.7 | 329 | 483 | 5 | 0.64 | – |
| 0.85 | 407 | 402 | 8 | 0.62 | – |
| **1.0 (centred)** | **408** | 396 | 13 | 0.62 | **55.6 %** |
| (corner-anchor, no centring) | 322 | 478 | 17 | ~0.55 | 46.3 % |

**The original hypothesis was wrong.** The model was *not* trained on
strokes confined to small sub-regions — four of eleven training shape
families (`circle`, `arc_only`, `noisy_circle`, `polyline_curve`) are
single-stroke samples that fill the canvas. Shrinking the inference
stroke pushed it *further* from the distribution.

The only thing that *did* help was the shift from corner-anchor
placement (`(pt - mn) / scale`) to **centroid-anchor**
(`(pt - centre) * 1/scale + 0.5`). LINE recognition improved from 322
→ 408 (+27 %), RANSAC agreement from 46 % → 56 %, confidence became
better calibrated (median 0.99 → 0.82). The LINE-length compression
barely moved (~0.55 → 0.62 ratio). CIRCLE recognition didn't move at all
(0 → 0).

---

## 6. Current open problems

### 6.1 CIRCLE never predicted on real closed loops

| edge | n pts | span | true topology | model output |
|---|---|---|---|---|
| 813 | 52 | 24 px | closed loop | polyline @ 1.00 |
| 814 | 45 | 19 px | closed loop | polyline @ 1.00 |
| 815 | 42 | 18 px | closed loop | polyline @ 1.00 |
| 816 | 44 | 24 px | closed loop | polyline @ 1.00 |

Synthetic CIRCLE F1 = 1.000. So the failure is not "didn't learn the
class" — it's "real closed loops don't match the training CIRCLE
distribution".

### 6.2 LINE parameters systematically under-scaled

For 408 LINE predictions, `pred_length / edge_span`:

| min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|
| 0.24 | 0.49 | 0.62 | 0.79 | 1.00 | 0.65 |

The model's output for line endpoints clusters near 0.25 / 0.75 of the
normalised canvas. This looks like a learned mean — when the encoder
input is OOD relative to training, the decoder falls back to the
distribution's centroid for endpoint position.

### 6.3 ARC count is low but the predictions look fine

13 / 817 arcs predicted (vs RANSAC's 55). When the model picks ARC, the
geometry is plausible — radii and sweeps line up with edge spans. But it
under-detects: in the 6-15 pt bucket the model picks 30 % arc / 70 %
polyline, vs RANSAC 87 % arc / 13 % polyline.

### 6.4 No collapse, but biased to POLYLINE on long strokes

| bucket | LINE | POLYLINE | ARC |
|---|---|---|---|
| 2 pts (n=567) | 63 % | 37 % | 0 % |
| 3-5 pts (n=141) | 37 % | 63 % | 0 % |
| 6-15 pts (n=23) | 0 % | 70 % | 30 % |
| 16+ pts (n=86) | 0 % | 93 % | 7 % |

Anything 6 + points becomes "polyline by default". The training
distribution has POLYLINE at only 4 % of commands — but inference reads
93 % POLYLINE on long edges.

---

## 7. Why RANSAC outperforms Free2CAD on this input

The honest summary:

1. **Geometric direct fit doesn't depend on a learned distribution.**
   RANSAC takes the actual edge points and fits a line / circle / arc by
   minimising residuals. Two points → exact line, no learned bias.
2. **2-point edges are degenerate but unambiguous.** Our skeleton graph
   has 567 edges with exactly 2 points. RANSAC: trivial line fit,
   confidence 1.0. Free2CAD: model has to *infer* "this is a line" from
   stroke features it has never seen at this scale and at inference has
   no multi-stroke context. 63 % accuracy is a real result, not chance,
   but it's worse than RANSAC's geometric certainty.
3. **Closed loops have explicit topology.** RANSAC checks
   `is_closed`; if true, it tries circle first. Free2CAD has no such
   short-circuit — it sees a noisy loop trace and must classify it from
   stroke features alone.
4. **RANSAC parameters are exact by construction.** A line fit returns
   the actual endpoints of the inlier set. Free2CAD outputs normalised
   parameters that have to be un-normalised — and the model is biased
   towards 0.25 / 0.75 instead of 0 / 1.
5. **RANSAC's confidence is informative; ours partially is.** RANSAC
   gives mean 0.91 on lines (high — most fits are clean), 0.30 on
   polyline fallbacks (low — these are cases where line / arc / circle
   all failed). Free2CAD confidence is now somewhat calibrated post-fix
   but still narrower than is useful.

Free2CAD's *theoretical* edge over RANSAC was supposed to be:
- handling noisy strokes that confuse a residual fit,
- producing semantically clean primitives (CAD-grade endpoints),
- joint reasoning across strokes (e.g. shared endpoints).

None of those advantages are realised in the current single-stroke
inference path.

---

## 8. Hypotheses and solution avenues

In rough order of expected payoff vs. effort. Each section explains the
reasoning so the receiving conversation can pick its battles.

### 8.1 Train on per-edge samples (high payoff, modest cost)

The fundamental mismatch is that we train on **multi-stroke sketches**
but infer one **single-stroke edge** at a time. The decoder learns
`stroke_i → command_i+1` with the encoder providing context across all
strokes, then at inference we feed a single stroke and the encoder has
no peer strokes to compare against.

Two ways to close the gap:

**(a) Augment the data with single-stroke samples.** Add a generator
mode that emits one stroke + one command per sample (one for each of
LINE / ARC / CIRCLE / POLYLINE). Each such stroke fills the canvas — same
distribution as inference. Mix this with the multi-stroke samples,
e.g. 50 % single-stroke / 50 % multi-stroke. This costs one new shape
family and a retrain.

**(b) Train per-edge from the start.** Drop multi-stroke training
entirely and treat each edge as its own sample. Simpler training code,
exact match between train and inference distributions. Sacrifices the
ability to do sketch-level reasoning, but our pipeline currently doesn't
exploit that anyway.

(b) is the cleanest. (a) preserves optionality.

### 8.2 Sketch-level batched inference (high payoff, high cost)

Pass groups of N ≤ max_strokes edges to the model together, with their
spatial positions normalised together (same way training samples are
constructed). Decode N commands at once. This matches the training
distribution exactly without changing the training code.

Cost: significant restructuring of `Free2CADFitter` and `run()`. The
817 edges in a real graph would need to be partitioned into ~40
chunks of 20, with attention to keeping spatially-related edges in the
same chunk (otherwise the multi-stroke context is meaningless).

### 8.3 Closed-loop fast path (low cost, fixes one specific failure)

In `Free2CADFitter.fit_edge`, before calling the model:

```python
if edge.get("is_closed", False) and len(pts) >= MIN_CIRCLE_PTS:
    # Try a fast circle fit; if residual is low, return it directly.
    ...
```

This is a hybrid path that uses the cheap circle check from RANSAC for
the unambiguous cases and only invokes Free2CAD for non-closed edges.
Trivially fixes the 4 / 4 closed-loop failure on this input. Doesn't
help LINE param quality.

### 8.4 Retrain with explicit param-loss-on-all-classes (medium cost)

Currently param L1 is gated on `{LINE, ARC, CIRCLE}` only. The decoder
only learns to predict accurate parameters for those classes when the
gold type also belongs to them. Test idea: predict params for *every*
position regardless of gold type, with the gold target being zeros for
END/POLYLINE. Forces the param head to learn the geometry signal even
when the type head is uncertain, which might also stabilise the
encoder's stroke representation.

Cost: a one-line change in the training loss + a retrain. Risk: makes
the param loss noisier.

### 8.5 Revisit `noisy_circle` realism (low cost, fixes CIRCLE)

The 4 real closed loops (edges 813-816) have 42-52 pts on a span of
~20 px. The current `_gen_noisy_circle` produces 12-30 pts on a span of
~9-40 % of [0, 1]² (roughly 0.04-0.18 radius). When normalised, both
training and real loops fill their own bbox… so why doesn't it
generalise?

Two probable culprits:

- **Noise amplitude**: `noise_std * uniform(1.5, 3.0)` ≈ 0.0075-0.015 on
  the normalised canvas. Real skeleton-traced loops are much
  noisier — their points wobble by 1-2 px on a 20 px span (= 5-10 %).
  Bump `noise_scale` to `uniform(3.0, 6.0)` so training noise covers the
  real scale.
- **Sampling pattern**: `_sample_circle` evenly samples around the
  circle. Real skeletons trace pixels ≠ angles uniformly — they have
  clusters and gaps from the skeletonisation algorithm. Adding
  per-angle gating would close the gap.

A side-by-side plot of one real closed loop next to one
`noisy_circle` sample (both normalised) would tell us in 30 seconds
which of these is the dominant gap.

### 8.6 Augmentations from the original roadmap §4.3 (low cost, broad)

Still untouched:

- **Rotation**: random ±180° rotation applied to strokes *and* command
  parameters. Currently every shape is axis-aligned (rectangles always
  have horizontal sides, etc.). Real engineering drawings aren't.
- **Variable stroke density**: sample `n_pts` from a wider range so the
  model sees both 2-point and 60-point versions of the same primitive.
- **Partial strokes**: randomly truncate 5-10 % of strokes so the
  encoder learns robustness to stroke-extraction errors.

### 8.7 Pre-Stage-3 edge merging (medium cost, sidesteps the problem)

Many of the 567 2-point edges are likely fragments of longer collinear
strokes that Stage 2 split at branch points. A preprocessing pass that
chains collinear adjacent edges into longer polylines before Stage 3
would push the input distribution closer to training.

Risk: aggressive merging breaks legitimate corner geometry. Needs
a tolerance threshold and probably a simple angle test
(`angle_change < 5°` between adjacent edges).

---

## 9. Reproducing the numbers

```bash
# 1. Generate v2 training data (≈30 k samples)
python primitivefitting/generate_sketches_v2.py \
    --output_dir primitivefitting/data/free2cad_training_v2 \
    --n_samples 30000

# 2. Train (200 epochs, ~hours on CPU, much faster on GPU)
python primitivefitting/train_free2cad_v2.py \
    --data_dir   primitivefitting/data/free2cad_training_v2 \
    --output_dir primitivefitting/weights/free2cad \
    --device     cuda \
    --epochs     200

# 3. Evaluate on the real graph
python primitivefitting/stage3_primitive_fit.py \
    Stroke_Extraction/output/graphs/Picture1_skeleton_graph.json \
    --output /tmp/eval --fitter free2cad

python primitivefitting/stage3_primitive_fit.py \
    Stroke_Extraction/output/graphs/Picture1_skeleton_graph.json \
    --output /tmp/eval_ransac --fitter ransac
```

Output: `output/primitives/<sketch_id>_primitives.json` with
`{ sketch_id, primitives: [{ edge_id, type, …, confidence }, … ] }`.

Best checkpoint goes to `Models/free2cad_best.pth` with this structure:

```python
{
  "model_state_dict": ...,
  "epoch": int,
  "val_loss": float,
  "config":   {"max_pts", "max_strokes", "n_cmd_types",
               "d_model", "n_heads", "n_enc_layers",
               "n_dec_layers", "dropout"},
  "cmd_types": {"LINE":0, "ARC":1, "CIRCLE":2, "POLYLINE":3, "END":4, "BOS":5},
  "class_weights": [w_LINE, w_ARC, w_CIRCLE, w_POLYLINE, 0, 0],
  "metrics":   {... per-class P/R/F1, param_L1, pred_entropy ...},
}
```

Stage 3 reads `cmd_types` to detect v2 (BOS) vs v1 (no BOS) checkpoints
and seeds the decoder accordingly.

---

## 10. What to investigate first

If I were picking up this problem cold with the three scripts in hand, I
would:

1. **Plot a real closed loop next to a `_gen_noisy_circle` sample** to
   determine whether §8.5 (more realistic noise) is enough to fix the
   CIRCLE failure or whether a bigger redesign is needed.
2. **Try §8.1(b) — single-stroke training data** and retrain. If the
   LINE-length compression goes from 0.62 to >0.9, that confirms the
   multi-stroke / single-stroke mismatch is the dominant issue and §8.2
   (sketch-level batched inference) is overkill.
3. **Add §8.3 (closed-loop fast path)** for free even if §8.1 succeeds —
   it's a 5-line edit and removes the worst-case CIRCLE failure mode
   regardless.

Everything else in §8 is incremental once those three are settled.
