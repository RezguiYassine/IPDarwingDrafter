#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATA_ROOT=output/SketchGraphsStage3Full
STAGE3_DATA="$DATA_ROOT/stage3"
CHECKPOINTS="$DATA_ROOT/checkpoints/free2cad_full_phaseA"
LOG="$DATA_ROOT/stage3_full_queue.log"
PILOT=models/free2cad_sketchgraphs.pth
LATEST="$CHECKPOINTS/free2cad_v3_latest.pth"

mkdir -p "$DATA_ROOT" "$CHECKPOINTS"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] Stage 3 full SketchGraphs queue started"

for split in train validation test; do
  echo "[$(date --iso-8601=seconds)] Preparing full $split split"
  .venv/bin/python -m tools.sketchgraphs_dataset prepare \
    --output "$DATA_ROOT" \
    --splits "$split" \
    --limit 0 \
    --workers 16 \
    --stage3-only \
    --source-chunk-size 10000 \
    --resume
done

train_args=(
  --data_dir "$STAGE3_DATA"
  --output_dir "$CHECKPOINTS"
  --epochs 2
  --batch_size 512
  --max_pts 64
  --arc_encoding three_point
  --class_weight_power 0.5
  --lr 3e-5
  --device cuda:1
  --stream_shards
  --save_every_shards 10
)

if [[ -f "$LATEST" ]]; then
  echo "[$(date --iso-8601=seconds)] Resuming Free2CAD from $LATEST"
  train_args+=(--resume "$LATEST")
else
  echo "[$(date --iso-8601=seconds)] Warm-starting Free2CAD from $PILOT"
  train_args+=(--init_weights "$PILOT")
fi

.venv/bin/python stage3_primitivesfitting/research/train_free2cad_v3.py \
  "${train_args[@]}"

echo "[$(date --iso-8601=seconds)] Evaluating pilot and full checkpoints"
.venv/bin/python -m tools.evaluate_free2cad_v3 \
  --data-dir "$STAGE3_DATA" \
  --checkpoint "$PILOT" \
  --split test --batch-size 512 --device cuda:1 \
  --output "$DATA_ROOT/evaluation_pilot_full_test.json"

for selection in best best_f1; do
  checkpoint="$CHECKPOINTS/free2cad_v3_${selection}.pth"
  .venv/bin/python -m tools.evaluate_free2cad_v3 \
    --data-dir "$STAGE3_DATA" \
    --checkpoint "$checkpoint" \
    --split test --batch-size 512 --device cuda:1 \
    --output "$DATA_ROOT/evaluation_${selection}_full_test.json"
done

echo "[$(date --iso-8601=seconds)] Stage 3 full SketchGraphs queue complete"
