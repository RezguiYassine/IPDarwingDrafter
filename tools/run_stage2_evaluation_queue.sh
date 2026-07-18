#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ROLE CHECKPOINT DEVICE CONFIG D2C_OUTPUT" >&2
  exit 2
fi

ROLE=$1
CHECKPOINT=$2
DEVICE=$3
CONFIG=$4
D2C_OUTPUT=$5
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ROOT/.venv/bin/python"
EVAL_OUTPUT="$ROOT/output/Stage2Evaluation/$ROLE"

cd "$ROOT"
mkdir -p "$EVAL_OUTPUT" "$D2C_OUTPUT"

while systemctl --user is-active --quiet puhachov-sg-eval-phasea.service || \
      systemctl --user is-active --quiet puhachov-sg-eval-production.service; do
  sleep 30
done

"$PYTHON" -m tools.evaluate_puhachov_streaming \
  --labels output/Drawing2CAD/kp_labels --split validation \
  --checkpoint "$CHECKPOINT" --device "$DEVICE" \
  --batch 4 --workers 4 --log-every 5000 \
  --output "$EVAL_OUTPUT/drawing2cad_validation.json"

"$PYTHON" -m tools.evaluate_puhachov_streaming \
  --labels output/ArchCAD/kp_labels --split validation \
  --checkpoint "$CHECKPOINT" --device "$DEVICE" \
  --batch 4 --workers 4 --log-every 500 \
  --output "$EVAL_OUTPUT/archcad_validation.json"

"$PYTHON" -m tools.d2c_eval \
  --config "$CONFIG" --output "$D2C_OUTPUT" \
  --db "$D2C_OUTPUT/d2c_results.db" \
  --split test --views all --limit 7881 --seed 42 --workers 4
