#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ROOT/.venv/bin/python"
PHASE_A="$ROOT/output/Drawing2CAD/full_test_phaseA"
PHASE_B="$ROOT/output/Drawing2CAD/full_test_phaseB"
REPORT="$ROOT/output/Drawing2CAD/full_test_phaseA_vs_phaseB.json"

cd "$ROOT"
mkdir -p "$PHASE_B"

"$PYTHON" - <<'PY'
import hashlib
import json
import sqlite3
from pathlib import Path

root = Path.cwd()
phase_a = root / "output/Drawing2CAD/full_test_phaseA"
manifest = json.loads((phase_a / "evaluation_manifest.json").read_text())
config = root / "config_d2c_eval_phaseA.yaml"
config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
assert manifest["config_sha256"] == config_hash, "Phase A config has changed"
assert manifest["selected_entries"] == 31_524, "Phase A selection is incomplete"
with sqlite3.connect(phase_a / "d2c_results.db") as connection:
    ok = connection.execute(
        "SELECT count(*) FROM d2c_results WHERE status='ok'"
    ).fetchone()[0]
assert ok == 31_524, f"Phase A has only {ok} successful rows"
print("verified frozen Phase A baseline: 31,524 successful rows")
PY

"$PYTHON" -m tools.d2c_eval \
  --config config_d2c_eval_phaseB.yaml \
  --output "$PHASE_B" --db "$PHASE_B/d2c_results.db" \
  --split test --views all --limit 7881 --seed 42 --workers 4

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path.cwd() / "output/Drawing2CAD"
phase_a = json.loads((root / "full_test_phaseA/evaluation_manifest.json").read_text())
phase_b = json.loads((root / "full_test_phaseB/evaluation_manifest.json").read_text())
assert phase_a["selection_sha256"] == phase_b["selection_sha256"], (
    "Phase A and Phase B selections differ"
)
assert phase_b["selected_entries"] == 31_524, "Phase B selection is incomplete"
print("verified paired selection: " + phase_b["selection_sha256"])
PY

"$PYTHON" -m tools.compare_d2c_runs \
  --baseline "$PHASE_A/d2c_results.db" \
  --candidate "$PHASE_B/d2c_results.db" \
  --output "$REPORT"
