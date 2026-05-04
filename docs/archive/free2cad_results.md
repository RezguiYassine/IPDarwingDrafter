# Free2CAD — Stage 3 Results & Roadmap

> Status as of 2026-04-23. Covers the first end-to-end training +
> inference cycle of the custom PyTorch Free2CAD model.

---

## 1. What is now wired end-to-end

- `train_free2cad.py` builds a standalone seq2seq Transformer
  (`StrokeEncoder` + `CommandDecoder`) and saves
  `Models/free2cad_best.pth` after each new best validation loss.
- `stage3_primitive_fit.py` now loads that checkpoint directly via
  `build_model()` — no `network.model.CADLModel`, no TF repo, no
  `repo_path` dependency. See `Free2CADFitter._load()`.
- A `--fitter {auto, free2cad, ransac}` CLI flag lets the user choose
  between the trained model, the RANSAC baseline, or the previous
  auto-fallback behaviour.

---

## 2. First-pass results

Input: `Stroke_Extraction/output/graphs/Picture1_skeleton_graph.json`
(817 edges).

| Mode        | Time   | Mean confidence | Type distribution              |
|-------------|--------|-----------------|--------------------------------|
| `ransac`    | 0.30 s | 0.867           | 817 line                       |
| `free2cad`  | 4.90 s | 1.000           | 817 line (0 arc, 0 circle)     |

Checkpoint stats: epoch 98, `val_loss = 0.0064`.

The wiring is correct. The **model** is the problem.

---

## 3. Root cause — class collapse

Measured on the actual training set
(`data/free2cad_training/train/*.json`, 25 500 samples,
105 069 commands):

| Type     | Count   | Share  |
|----------|---------|--------|
| LINE     | 94 238  | 89.7 % |
| CIRCLE   |  5 697  |  5.4 % |
| ARC      |  5 134  |  4.9 % |
| POLYLINE |      0  |  0.0 % |

So:
- **LINE is ~90 %** of every training gradient. A constant "LINE + mean
  geometry" classifier reaches very low loss, which is exactly what the
  model learned.
- **POLYLINE is in the vocabulary but never in the data.** The token is
  dead weight — the model will never produce it.
- Arc and circle signal is drowned out; confidence 1.0 on every edge
  confirms the softmax has saturated on a single class.

The per-shape frequencies baked into `_generate_procedural_sample` at
[generate_sketches.py:133-135](generate_sketches.py#L133-L135) make
this inevitable:

```
rect       25%  → 4 LINE
l_shape    20%  → 6 LINE
circle     15%  → 1–2 CIRCLE
arc_box    20%  → 1 ARC + 3 LINE
multi_line 20%  → 3–6 LINE
```

Three of the five shape families emit only LINE; two emit mostly LINE.

---

## 4. Roadmap — data generator (`generate_sketches.py`)

Concrete, ordered by priority.

### 4.1 Rebalance shape frequencies

Target a roughly uniform **per-command** distribution, not per-shape.
Starting point:

| Shape      | Current | Proposed |
|------------|---------|----------|
| rect       | 25 %    |  10 %    |
| l_shape    | 20 %    |   5 %    |
| multi_line | 20 %    |  10 %    |
| circle     | 15 %    |  25 %    |
| arc_box    | 20 %    |  20 %    |
| **(new)** arc_only        | —     | 15 %    |
| **(new)** circle_cluster  | —     | 10 %    |
| **(new)** polyline_curve  | —     |  5 %    |

Expected command share after rebalancing: LINE ≈ 45 %, ARC ≈ 25 %,
CIRCLE ≈ 25 %, POLYLINE ≈ 5 %.

### 4.2 Add new shape primitives

- `arc_only` — isolated arcs at random orientation/sweep (covers the
  "arc not attached to a U-channel" case).
- `circle_cluster` — 2–4 circles of varying radius (bolt patterns,
  holes).
- `polyline_curve` — hand-drawn-style polylines that are *not* well
  approximated by line/arc/circle, so POLYLINE becomes a genuine label.
- `rounded_rect` — 4 lines + 4 arcs, forces the model to switch
  primitive type within a sketch.
- `mixed_assembly` — combine 2–3 sub-shapes in one sample so the
  decoder sees longer command sequences and stronger context.

### 4.3 Augmentations

- **Rotation**: random rotation ±180° applied to both strokes *and*
  command parameters (currently everything is axis-aligned).
- **Variable stroke density**: sample `n_pts` from a wider range
  (currently 8–20) so the model sees 3-point and 40-point strokes.
- **Noise schedule**: mix samples with `noise_std ∈ {0.002, 0.005,
  0.015, 0.03}` instead of a single value, covering clean CAD exports
  to coarse hand-drawn strokes.
- **Arc sweep diversity**: today arcs are always 0-180° half-circles.
  Sample `start_angle ∈ [0, 360)` and `sweep ∈ [20°, 340°]`.
- **Partial strokes**: randomly truncate 5–10 % of strokes so the
  encoder learns robustness to stroke extraction errors.

### 4.4 STEP-file mode

The `--cad_dir` path (Open CASCADE) is implemented but disabled without
pythonocc-core. Priority-3: once procedural data is balanced, add a
small real-world slice (ABC, Fusion360 Gallery) to close the sim-to-real
gap — procedural noise ≠ scanned-sketch noise.

### 4.5 Sanity checks after regeneration

Add a small validator in the generator that prints per-class counts and
fails if any class drops below 10 %:

```
LINE:   45.2%  ARC: 23.1%  CIRCLE: 26.5%  POLYLINE:  5.2%
```

---

## 5. Roadmap — training strategy (`train_free2cad.py`)

### 5.1 Class-weighted loss

Replace the current `CrossEntropyLoss` at
[train_free2cad.py:431](train_free2cad.py#L431) with an inverse-frequency
weighted version so the gradient from an ARC example is worth ~6× a LINE:

```python
w = torch.tensor([1/.90, 1/.05, 1/.05, 1/.05, 0], device=dev)
w = w / w.sum() * 4       # normalise, keep END at 0
type_loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=CMD_TYPES["END"])
```

(Recompute the weights from the actual dataset after regeneration.)

### 5.2 Decoder start-of-sequence token

Training currently has an off-by-one: with `tgt_in = cmd[:-1]` and
`tgt_out = cmd[1:]`, the model only ever learns `cmd[i] → cmd[i+1]`. At
inference there is no `cmd[0]` to seed with, so
`Free2CADFitter.fit_edge` has to fake one.

Fix:
- Add `CMD_TYPES["BOS"] = 5`, `N_CMD_TYPES = 6`.
- Prepend BOS to `cmd_types` / zeros to `cmd_params` in
  `load_dataset()`.
- Now position 0's output legitimately predicts the first real command.
- Update `Free2CADFitter.fit_edge` to seed with BOS instead of LINE.

### 5.3 Drop the unused END parameter regression

The END token's parameter vector is always zero by construction; the L1
loss on it adds no signal but does add noise during training. Gate the
parameter loss on the gold type, not a mask of END.

### 5.4 Per-class validation metrics

During validation, log:
- Confusion matrix over {LINE, ARC, CIRCLE, POLYLINE}.
- Mean L1 on parameters **per class**.
- Percentage of samples where argmax ≠ END (collapse detector).

A single `val_loss` number hides the fact that the model is 90 % LINE
— per-class metrics would have surfaced this on day one.

### 5.5 Scheduler and capacity review

- Current `CosineAnnealingLR(T_max=epochs)` with warm start is fine.
  Keep.
- Consider label smoothing (0.05–0.1) to stop the softmax from
  saturating as aggressively.
- Model size (`d_model=256`, 4+4 layers, ~4 M params) is likely
  adequate once the data is balanced. No change recommended yet.

### 5.6 Longer training after rebalancing

With a balanced dataset the model will need more epochs to reach
comparable val loss because it can no longer take the trivial "always
LINE" shortcut. Plan for 200 epochs initially; rely on `free2cad_best`
checkpointing to stop getting worse.

---

## 6. Suggested execution order

1. **Rebalance procedural generator** (§4.1, §4.2, §4.5) — cheap, high
   impact.
2. **Regenerate training set** with the new distribution and run the
   validator.
3. **Add BOS token** (§5.2) and **class-weighted loss** (§5.1) to
   `train_free2cad.py`.
4. **Add per-class metrics** (§5.4) so the next run is measurable.
5. **Retrain** for 200 epochs; save the new `Models/free2cad_best.pth`.
6. **Re-run `stage3_primitive_fit.py --fitter free2cad`** on
   `Picture1_skeleton_graph.json`, inspect the per-class output, and
   iterate.

Augmentations (§4.3) and STEP-file mode (§4.4) are worth adding once
steps 1–6 prove the balanced model is learning non-trivially.

---

## 7. Out of scope for this roadmap

- Porting the paper's official TensorFlow architecture to PyTorch. Our
  custom Transformer is architecturally sufficient; the bottleneck is
  data.
- Multi-stroke sketch-level inference (feeding N edges, decoding N
  commands autoregressively). Current per-edge inference works, is
  simpler, and handles sketches with more edges than `max_strokes=20`
  (e.g. the 817-edge real-world case).
- Training on the real skeleton graphs coming out of Stage 2. That is a
  fine-tuning step to schedule after the synthetic model is healthy.
