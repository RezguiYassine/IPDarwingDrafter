# Stage 1 Research — Training SketchCleanNet

This directory contains the training code for **SketchCleanNet**, the deep-learning sketch cleaner used by Stage 1 of the AP3 vectorization pipeline. It is NOT required to run the pipeline — production code consumes the pre-trained `.pth` weight file. Use the contents of this directory only if you want to re-train the model on new data.

## Files

| File                        | Purpose                                                          |
|-----------------------------|------------------------------------------------------------------|
| `train_sketchcleannet.py`   | Full training loop (PyTorch). Reads paths from `config.yaml`.    |

## Training data

The model was trained on the **SketchCleanNet dataset**, a publicly released set of paired noisy/clean sketches. The original release ships as a single `.zip`.

### How to download

The dataset is **not shipped** with this repo (it is ~28 MB unzipped, ~26 MB zipped, and is freely available from the upstream authors). To obtain it:

1. **Source (upstream release).** SketchCleanNet was published by Simo-Serra et al. The training data is distributed alongside the original paper artefacts. Search:
   - Paper: *"Mastering Sketching: Adversarial Augmentation for Structured Prediction"*, ACM TOG 2018
   - Project page: https://esslab.jp/~ess/en/research/sketch_master/
   - Direct dataset link: see the project page for the `SketchCleanNet_Data.zip` mirror (link may move; check the page).

2. **Layout expected by the trainer.** Unzip into the project root so the resulting tree is:

   ```
   stage1_preprocessing/research/data/SketchCleanNet_Data/
       Train Data/
           rough/         # noisy / un-cleaned input sketches
           clean/         # paired ground-truth clean sketches
       Validation Data/
           rough/
           clean/
   ```

   (The trainer reads the paired structure from these directories. The exact layout produced by the upstream zip should already match.)

3. **Update `config.yaml`** if you place the data elsewhere — the trainer derives paths from the `data:` block.

## Re-training

Once the data is in place:

```bash
python stage1_preprocessing/research/train_sketchcleannet.py
```

Hyperparameters live at the top of the script. Best checkpoints land in `models/sketchcleannet.pth` by default (overwriting the production weight). Back up the old one first if you want to keep it.

## Why this lives in `research/`

Stage 1 has two cleaning modes (DL + classical fallback). The pipeline runs end-to-end without ever importing this directory. Keeping the trainer here makes it clear that:

- A partner cloning the repo does **not** need 26 MB of training data to use the pipeline.
- The choice of training corpus, hyperparameters, and weight file is research-time tuning, not a runtime concern.
- If a future team wants to fine-tune SketchCleanNet on their own data, everything they need is in this folder.
