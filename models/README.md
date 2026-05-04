# Models

Pre-trained model weights consumed by the AP3 vectorization pipeline at runtime.

## What's in the repo

| File                       | Size  | Used by                  | Source                     |
|----------------------------|------:|--------------------------|----------------------------|
| `puhachov_keypoints.pth`   | 22 MB | Stage 2 (stroke graph)   | shipped in repo            |
| `free2cad_v3_best.pth`     | 3 MB  | Stage 3 (research only)  | shipped in repo            |

These two weights are small enough to ship inline. The pipeline runs out of the box for Stages 2–4 once the repo is cloned.

## What's NOT in the repo

| File                       | Size   | Used by              | How to obtain                    |
|----------------------------|-------:|----------------------|----------------------------------|
| `sketchcleannet.pth`       | 124 MB | Stage 1 (DL cleaning) | Run [`setup.sh`](../setup.sh) — or download manually and place here |

`sketchcleannet.pth` exceeds GitHub's 100 MB per-file limit and is hosted externally. The download URL is set in `setup.sh`. If unavailable, Stage 1 automatically falls back to its **classical cleaning mode** (Otsu + adaptive threshold + morphology), so the pipeline still runs end-to-end — just with somewhat noisier output on photographed/shaded sketches.

## Manual download (if `setup.sh` does not work)

1. Obtain `sketchcleannet.pth` from your project administrator or partner-shared storage.
2. Place it at `models/sketchcleannet.pth`.
3. Confirm the path in `config.yaml` matches (it should, by default).

## Re-training

If you want to re-train any of these models on new data, see the `research/` subdirectory of the corresponding stage:

- Stage 1: [stage1_preprocessing/research/](../stage1_preprocessing/research/README.md)
- Stage 3: [stage3_primitivesfitting/research/](../stage3_primitivesfitting/research/README.md)
