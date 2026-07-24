#!/usr/bin/env bash
# Pilot v3: full pipeline over 100 v3-filtered patent drawings, 3 Stage-2 configs.
set -uo pipefail
cd /home/safe/Desktop/yassine/Vectorization
ROOT=output/PatentData/pilotv3_input
WORKERS=8

run() {
  local name=$1 cfg=$2 out=$3
  echo "===== $(date '+%H:%M:%S') START $name ($cfg) -> $out ====="
  python3 -m tools.batch_run \
      --patent-root "$ROOT" \
      --output "$out" \
      --db "$out/results.db" \
      --config "$cfg" \
      --workers "$WORKERS" \
      2>&1
  echo "===== $(date '+%H:%M:%S') END   $name (exit $?) ====="
}

run cnfusion config_pilotv3_cnfusion.yaml output/pilotv3_cnfusion
run phaseA   config_pilotv3_phaseA.yaml   output/pilotv3_phaseA
run phaseB   config_pilotv3_phaseB.yaml   output/pilotv3_phaseB
echo "===== ALL DONE $(date '+%H:%M:%S') ====="
