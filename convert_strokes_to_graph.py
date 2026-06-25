#!/usr/bin/env python3
import json
from pathlib import Path

# === CONFIGURATION – CHANGE THESE PATHS ===
INPUT_DIR = Path("stage3_primitivesfitting/research/data/free2cad_training_v3/test")
OUTPUT_DIR = Path("output/graphs_from_test")
# =========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

json_files = list(INPUT_DIR.glob("*.json"))
print(f"Found {len(json_files)} JSON files in {INPUT_DIR}")

converted = 0
for json_path in json_files:
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if "stroke" not in data:
        print(f"Skipping {json_path.name}: no 'stroke' key")
        continue
    
    points = data["stroke"]
    
    # Determine closed: approximate
    closed = False
    if len(points) >= 3:
        first = points[0]
        last = points[-1]
        dist = ((first[0]-last[0])**2 + (first[1]-last[1])**2)**0.5
        # bounding box diagonal
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        diag = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)**0.5
        if diag > 0 and dist/diag < 0.05:
            closed = True
    
    graph = {
        "edges": [
            {
                "id": 0,
                "pixels": points,
                "smooth_pts": points,
                "is_closed": closed
            }
        ]
    }
    
    # Output filename: original stem + "_graph.json"
    out_name = json_path.stem + "_graph.json"
    out_path = OUTPUT_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(graph, f, indent=2)
    
    converted += 1
    if converted % 100 == 0:
        print(f"Converted {converted}...")

print(f"Done. Converted {converted} files to {OUTPUT_DIR}")
print(f"Example output file: {list(OUTPUT_DIR.glob('*_graph.json'))[:3]}")