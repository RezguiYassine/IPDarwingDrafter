# IP DrawingDrafter — Stage 1 & 2 Setup Guide
## Claude Code Handoff: Environment · Dependencies · Models · First Run

> **Purpose:** This file gives Claude Code everything it needs to set up the
> environment on a local machine, install all dependencies, download or prepare
> model weights, and run Stage 1 (Preprocessing) and Stage 2 (Stroke Extraction)
> end-to-end on a test image.
>
> Read this entire file before executing any commands.

---

## 1. What we are setting up

Two pipeline stages that together convert a raw hand sketch into a topological
stroke graph (JSON):

```
Hand sketch PNG/JPG
      ↓  [Stage 1 — stage1_preprocess.py]
      │  Cleans the image (SketchCleanNet or classical fallback)
      │  Produces a 1px-wide binary skeleton
      ↓
Skeleton PNG
      ↓  [Stage 2 — stage2_stroke_extract.py]
      │  Detects endpoints and junctions (Puhachov CNN or classical CN)
      │  Traces stroke chains between keypoints
      │  Detects closed loops (circles, rectangles)
      ↓
Stroke graph JSON  →  feeds Stage 3 (Free2CAD primitive fitting, not yet built)
```

Both stages use a **weights-optional architecture**: they run in classical
fallback mode immediately, and switch to DL inference automatically once
weights are placed at the paths in `config.yaml`. No code changes required.

---

## 2. Repository layout (expected)

```
ip-drawing-drafter/
├── src/
│   └── pipeline/
│       ├── stage1_preprocess.py     ← already written
│       ├── stage2_stroke_extract.py ← already written
│       └── config.yaml              ← already written
├── weights/
│   ├── sketchcleannet.pth           ← to be trained (see Section 5)
│   └── puhachov_keypoints.pth       ← to be downloaded (see Section 6)
├── input/                           ← place input images here
├── output/                          ← pipeline writes here
└── requirements.txt                 ← to be created (see Section 3)
```

If the repository does not yet exist, create this structure:
```bash
mkdir -p ip-drawing-drafter/src/pipeline
mkdir -p ip-drawing-drafter/weights
mkdir -p ip-drawing-drafter/input
mkdir -p ip-drawing-drafter/output
cd ip-drawing-drafter
```

Place `stage1_preprocess.py`, `stage2_stroke_extract.py`, and `config.yaml`
in `src/pipeline/`.

---

## 3. Python environment

### Requirements

Create `requirements.txt` at the project root with:

```text
# Core image processing
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-image>=0.21.0

# Graph operations (Stage 2 topology extraction)
networkx>=3.1

# Curve smoothing (Stage 2 Layer 3)
scipy>=1.11.0
rdp>=0.8

# YAML config
pyyaml>=6.0

# Deep learning (Stage 1 SketchCleanNet + Stage 2 Puhachov CNN)
# Install the CUDA version matching your local GPU driver.
# For CUDA 12.x:
torch>=2.0.0
torchvision>=0.15.0

# PDF ingestion (Stage 0, not yet built but install now)
pdf2image>=1.16.0

# Progress bar (batch runner)
tqdm>=4.65.0
```

### Create and activate a virtual environment

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

### PyTorch CUDA version

The default `pip install torch` installs the CPU version.
For GPU inference (recommended — significantly faster for SketchCleanNet),
install the CUDA-enabled build:

```bash
# CUDA 12.1 (adjust to match your driver: nvidia-smi shows CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only (fallback, no GPU needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify installation

```bash
python - <<'EOF'
import cv2, numpy, networkx, scipy, rdp, yaml, torch
print(f"OpenCV   : {cv2.__version__}")
print(f"NumPy    : {numpy.__version__}")
print(f"NetworkX : {networkx.__version__}")
print(f"SciPy    : {scipy.__version__}")
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA OK  : {torch.cuda.is_available()}")
EOF
```

Expected: all version numbers printed, `CUDA OK: True` if GPU is available.

---

## 4. Config file

Edit `src/pipeline/config.yaml` to match your machine.
The key fields to update are the `weights` paths:

```yaml
sketchcleannet:
  weights: "weights/sketchcleannet.pth"   # set once trained (Section 5)
  device: "cuda"                           # or "cpu" if no GPU

puhachov:
  weights: "weights/puhachov_keypoints.pth"  # set once downloaded (Section 6)
  device: "cuda"                              # or "cpu"
  keypoint_threshold: 0.50
  nms_radius: 5

stage1:
  quality_threshold: 0.70
  classical:
    blur_kernel: 3
    adaptive_block: 35
    adaptive_C: 10
    blend_alpha: 0.6
    morph_kernel: 2
    min_cc_size: 30

stage2:
  isolation_threshold: 0.05
  max_search_radius: 60
  rdp_epsilon: 1.5
  spline_smoothing: 2.0

pipeline:
  workers: 4
  resume: true
  log_level: "INFO"
```

Leave `weights: ""` for any model whose weights are not yet available.
The stage will print a warning and use the classical fallback automatically.

---

## 5. SketchCleanNet weights (Stage 1)

SketchCleanNet is a U-Net trained on engineering CAD sketches.
**Pretrained weights are not publicly released** — you must train them.
Until training is complete, Stage 1 runs in classical fallback mode
(Otsu + morphological opening + Zhang-Suen thinning), which already
produces good results.

### Option A — Train from scratch (recommended)

#### Step 1: Clone the SketchCleanNet repository

```bash
git clone https://github.com/BardOfCodes/SketchCleanNet.git
cd SketchCleanNet
pip install -r requirements.txt
```

If the original repository is unavailable, the U-Net architecture is
fully defined inside `stage1_preprocess.py` (class `_UNet`) and can be
trained independently using the training script below.

#### Step 2: Prepare training data

Training requires paired images: (rough/defective sketch, clean sketch).

**Recommended datasets:**

| Dataset | Description | Link |
|---------|-------------|------|
| CADSketchNet (Dataset-B) | 801 hand-drawn sketches of ESB CAD models across 42 categories | https://github.com/bharadwaj-manda/CADSketchNet |
| ESB (Engineering Shape Benchmark) | 3D CAD models — render clean edge images as ground truth | http://datarepository.wolframcloud.com/resources/Engineering-Shape-Benchmark |
| OpenSketch | Professional design sketches with clean/rough pairs | https://ns.inria.fr/d3/OpenSketch/ |
| Project partner data | Invention disclosure sketches from ThyssenKrupp Presta, Elmos, Biedermann (internal) | Contact infoapps GmbH |

**Generating training pairs from CADSketchNet:**
- Clean side: render the 3D ESB models as edge images using Blender or
  OpenCASCADE (weighted Canny on the CAD projection).
- Rough side: the hand-drawn CADSketchNet-B sketches directly.
- Augment rough side: add Gaussian noise, dilation, random gaps, brightness jitter.

#### Step 3: Minimal training script

If training with the U-Net defined in `stage1_preprocess.py`:

```python
# train_sketchcleannet.py  — place at project root
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2, numpy as np, yaml
from src.pipeline.stage1_preprocess import _UNet, ModelNotAvailableError

class SketchPairDataset(Dataset):
    def __init__(self, rough_dir, clean_dir, size=512):
        self.pairs = sorted(Path(rough_dir).glob("*.png"))
        self.clean_dir = Path(clean_dir)
        self.size = size
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        rough = cv2.imread(str(self.pairs[idx]), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(self.clean_dir / self.pairs[idx].name), cv2.IMREAD_GRAYSCALE)
        rough = cv2.resize(rough, (self.size, self.size)) / 255.0
        clean = cv2.resize(clean, (self.size, self.size)) / 255.0
        return (torch.FloatTensor(rough).unsqueeze(0),
                torch.FloatTensor(clean).unsqueeze(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = _UNet().to(device)
optim  = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

ds     = SketchPairDataset("data/rough", "data/clean")
loader = DataLoader(ds, batch_size=8, shuffle=True)

for epoch in range(100):
    total_loss = 0
    for rough, clean in loader:
        rough, clean = rough.to(device), clean.to(device)
        pred  = model(rough)
        loss  = loss_fn(pred, clean)
        optim.zero_grad(); loss.backward(); optim.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:3d}  loss={total_loss/len(loader):.4f}")
    if (epoch+1) % 10 == 0:
        torch.save({"model_state_dict": model.state_dict()},
                   f"weights/sketchcleannet_ep{epoch+1}.pth")

torch.save({"model_state_dict": model.state_dict()},
           "weights/sketchcleannet.pth")
print("Saved: weights/sketchcleannet.pth")
```

```bash
python train_sketchcleannet.py
```

#### Step 4: Activate in config

Once training is complete and `weights/sketchcleannet.pth` exists:

```yaml
# config.yaml
sketchcleannet:
  weights: "weights/sketchcleannet.pth"
  device: "cuda"
```

Stage 1 will automatically use the model on the next run.

#### Validation checklist

```bash
python src/pipeline/stage1_preprocess.py input/test.png \
    --output output --config src/pipeline/config.yaml

# Check output:
# - model_used: sketchcleannet  (not "classical")
# - Quality score >= 0.70
# - output/cleaned/test_cleaned.png  → crisp, clean strokes
# - output/cleaned/test_skeleton.png → 1px-wide skeleton
```

---

## 6. Puhachov Keypoint CNN weights (Stage 2)

### Step 1: Clone the repository

```bash
git clone https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow.git
cd line-drawing-vectorization-polyvector-flow
```

### Step 2: Download pretrained weights

The pretrained checkpoint is provided in the repository's GitHub Releases.
Download `best_model_checkpoint.pth` from the releases page:

```
https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow/releases
```

Or via command line:

```bash
# From the project root (ip-drawing-drafter/)
wget -O weights/puhachov_keypoints.pth \
  https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow/releases/download/v1.0/best_model_checkpoint.pth
```

If the direct URL has changed, check the releases page manually and update
the URL. The file is named `best_model_checkpoint.pth` in all releases.

### Step 3: Verify the checkpoint loads

```bash
python - <<'EOF'
import torch
ckpt = torch.load("weights/puhachov_keypoints.pth", map_location="cpu")
print("Checkpoint keys:", list(ckpt.keys())[:5])
print("Checkpoint type:", type(ckpt))
EOF
```

The checkpoint may be a raw `state_dict` or a dict with key `"state_dict"`
or `"model_state_dict"`. The wrapper in `stage2_stroke_extract.py` handles
all three formats automatically.

### Step 4: Activate in config

```yaml
# config.yaml
puhachov:
  weights: "weights/puhachov_keypoints.pth"
  device: "cuda"
```

### Step 5: Validate

```bash
python src/pipeline/stage2_stroke_extract.py output/cleaned/test_skeleton.png \
    --output output --config src/pipeline/config.yaml

# Check output:
# - keypoint_source: cnn  (not "classical")
# - output/graphs/test_graph.json exists
# - Isolation ratio < 0.05
```

---

## 7. Running the pipeline

### Quick test with a synthetic sketch

If you do not have a real patent sketch yet, generate the standard test image:

```bash
python - <<'EOF'
import numpy as np, cv2
from pathlib import Path
Path("input").mkdir(exist_ok=True)
img = np.ones((300, 400), dtype=np.uint8) * 255
cv2.line(img,   (20, 150), (380, 150), 0, 3)   # horizontal line
cv2.line(img,   (200, 20), (200, 280), 0, 3)   # vertical line → T-junction
cv2.circle(img, (100, 230), 40, 0, 2)          # closed loop (circle)
cv2.line(img,   (260, 60), (360, 260), 0, 3)   # diagonal
rng = np.random.default_rng(42)
img[rng.random(img.shape) < 0.02] = 0          # 2% noise
cv2.imwrite("input/test_sketch.png", img)
print("Written: input/test_sketch.png")
EOF
```

### Run Stage 1

```bash
cd ip-drawing-drafter

python src/pipeline/stage1_preprocess.py \
    input/test_sketch.png \
    --output output \
    --config src/pipeline/config.yaml \
    --id test_sketch

# Expected output:
# ──────────────────────────────────────────────────
#   Sketch ID       : test_sketch
#   Model used      : classical  (or "sketchcleannet" if weights exist)
#   Cleaned PNG     : output/cleaned/test_sketch_cleaned.png
#   Skeleton PNG    : output/cleaned/test_sketch_skeleton.png
#   Quality score   : ~1.000
#   Flagged         : no
#   Processing time : ~0.03s
# ──────────────────────────────────────────────────
```

### Run Stage 2

```bash
python src/pipeline/stage2_stroke_extract.py \
    output/cleaned/test_sketch_skeleton.png \
    --output output \
    --config src/pipeline/config.yaml \
    --id test_sketch

# Expected output:
# ────────────────────────────────────────────────────────
#   Sketch ID        : test_sketch
#   Keypoint source  : classical  (or "cnn" if weights exist)
#   Nodes            : 10
#   Edges            : 9
#   Isolation ratio  : ~0.009
#   Flagged          : no
#   Graph JSON       : output/graphs/test_sketch_graph.json
#   Processing time  : ~0.40s
# ────────────────────────────────────────────────────────
```

### Inspect the graph JSON

```bash
python - <<'EOF'
import json
with open("output/graphs/test_sketch_graph.json") as f:
    g = json.load(f)
print(f"Image shape : {g['image_shape']}")
print(f"Nodes ({len(g['nodes'])}):")
for n in g['nodes']:
    print(f"  kp{n['id']:2d}  ({n['x']:3d},{n['y']:3d})  {n['type']}")
print(f"Edges ({len(g['edges'])}):")
for e in g['edges']:
    print(f"  {e['source']} → {e['target']}  "
          f"pixels={len(e['pixels'])}  smooth={len(e['smooth_pts'])}  "
          f"closed={e['is_closed']}")
EOF
```

---

## 8. Expected output files

After running both stages on `test_sketch.png`:

```
output/
├── cleaned/
│   ├── test_sketch_cleaned.png    ← denoised raster (SketchCleanNet or Otsu)
│   └── test_sketch_skeleton.png   ← 1px-wide binary skeleton (Zhang-Suen)
└── graphs/
    └── test_sketch_graph.json     ← stroke graph (nodes + edges)
```

The graph JSON schema:

```json
{
  "sketch_id": "test_sketch",
  "image_shape": [300, 400],
  "nodes": [
    {"id": 0, "x": 200, "y": 21, "type": "endpoint", "confidence": 1.0},
    {"id": 6, "x": 201, "y": 151, "type": "junction", "confidence": 1.0},
    {"id": 9, "x": 100, "y": 230, "type": "loop_anchor", "confidence": 1.0}
  ],
  "edges": [
    {
      "id": 0, "source": 0, "target": 6,
      "pixels": [[200,21],[200,22],"..."],
      "smooth_pts": [[200.0,21.0],[200.1,45.3],"..."],
      "is_closed": false
    },
    {
      "id": 8, "source": 9, "target": 9,
      "pixels": [[60,190],[61,190],"..."],
      "smooth_pts": [[60.0,190.0],"..."],
      "is_closed": true
    }
  ]
}
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `model_used: classical` when you expect the CNN | Weights path wrong or file missing | Check the path in `config.yaml` matches the actual `.pth` file location |
| `CUDA OK: False` | PyTorch CPU build installed | Reinstall with the CUDA wheel URL (Section 3) |
| `isolation_ratio > 0.05` (flagged) | Noisy input or very complex sketch | Lower `stage2.isolation_threshold` or pre-clean input manually |
| `Quality score < 0.70` (Stage 1 flagged) | Very noisy skeleton from classical fallback | Train SketchCleanNet (Section 5) |
| `networkx` import error | Package not installed | `pip install networkx` |
| `rdp` import error | Package not installed | `pip install rdp` |
| `ModuleNotFoundError: stage1_preprocess` | Wrong working directory | Run from `ip-drawing-drafter/` root, not from inside `src/pipeline/` |
| Skeleton PNG is all white | Input image not grayscale or path wrong | Check input is uint8 grayscale; use `--id` flag correctly |
| `torch.load` warning about weights_only | PyTorch >= 2.0 security default | Add `weights_only=False` to `torch.load()` calls in the stage files |

---

## 10. Next stages (not yet built)

After Stage 1 and Stage 2 are running correctly, the next steps are:

1. **Stage 1b — Bezugszeichen** (`stage1b_bezugszeichen.py`):
   OCR detect reference numerals → inpaint from skeleton → store JSON
   sidecar for re-insertion in Stage 4.

2. **Stage 0 — Ingest** (`stage0_ingest.py`):
   PDF → per-page PNG at 300 DPI (using `pdf2image`).

3. **Stage 3 — Primitive Fitting** (`stage3_primitive_fit.py`):
   Free2CAD autoregressive Transformer → lines, arcs, circles.
   Input: `output/graphs/<id>_graph.json` and `smooth_pts` from each edge.

4. **Stage 4 — Export** (`stage4_export.py`):
   `ezdxf` + `svgwrite` → `.dxf` and `.svg`.
   Re-inserts Bezugszeichen as text entities.

5. **Batch runner** (`run_pipeline.py`):
   `python run_pipeline.py --input ./input --output ./output --resume`

---

## 11. Key design decisions (for future reference)

**Weights-optional architecture**: both stages run classical fallbacks until
DL weights are available. Set the path in `config.yaml` — no code changes.

**Stage 2 topology algorithm**: CN-cluster skeleton tracing with extended
keypoint map. Walk stops when entering any pixel within the 8-connected
neighbourhood of a keypoint cluster. This absorbs Zhang-Suen staircase
artefacts that cause exact-pixel matching to miss junctions by 1px.

**No Dijkstra / no Gurobi**: topology extraction is pure skeleton tracing,
O(foreground pixels). Gurobi (Steiner-tree solver used in the original
Puhachov C++ implementation) is not required.

**Closed loop detection**: CCs with ≥40 foreground pixels and zero keypoints
are classified as closed loops (circles, rectangles). Noise fragments < 40px
are discarded.

**Confidence signals**:
- Stage 1: `skeleton_quality` (fraction of thin pixels). Threshold: 0.70.
- Stage 2: `isolation_ratio` (fraction of uncaptured foreground pixels). Threshold: 0.05.
- Flagged sketches copied to `output/review/` by the batch runner.
