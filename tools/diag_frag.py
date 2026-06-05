"""
Diagnostic: reproduce + quantify stroke fragmentation on D2C samples.

Runs S1 -> S2 -> S3 on a handful of D2C Front SVGs and reports, per sketch:
  - n nodes / edges / primitives
  - node-degree histogram
  - count of degree-2 JUNCTION nodes (phantom-junction candidates)
  - count of tiny edges (< N px chain) — halo spurs
  - the through-angle at each degree-2 junction
"""
from __future__ import annotations
import io, sys, json, math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
for sub in ("stage1_preprocessing", "stage2_strokeextraction",
            "stage3_primitivesfitting", "stage4_export"):
    sys.path.insert(0, str(ROOT / sub))
import stage1_preprocess as s1
import stage2_stroke_extract as s2
import stage3_primitive_fit as s3
import yaml

CFG = yaml.safe_load(open(ROOT / "config_d2c_eval.yaml"))
WORK = ROOT / "output" / "diag"
WORK.mkdir(parents=True, exist_ok=True)
RES = 1024


def rasterize(svg, out):
    png = cairosvg.svg2png(url=str(svg), output_width=RES, output_height=RES,
                           background_color="white")
    img = np.array(Image.open(io.BytesIO(png)).convert("L"))
    _, b = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(out), b)
    return b


def analyze_graph(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    deg = Counter()
    incident = defaultdict(list)
    for e in edges:
        if e.get("is_closed"):
            continue
        deg[e["source"]] += 1
        deg[e["target"]] += 1
        incident[e["source"]].append(e)
        incident[e["target"]].append(e)
    # degree histogram over junction nodes
    junc_deg = Counter()
    deg2_junc = []
    for nid, n in nodes.items():
        if n["type"] == "junction":
            d = deg.get(nid, 0)
            junc_deg[d] += 1
            if d == 2:
                deg2_junc.append(nid)
    # through-angle at degree-2 junctions
    def edge_dir_at(e, nid):
        # direction of the edge chain leaving node nid
        px = e["pixels"]
        if e["source"] == nid:
            seq = px
        else:
            seq = px[::-1]
        if len(seq) < 2:
            return None
        p0 = np.array(seq[0], float)
        # take a point a few px along
        k = min(len(seq) - 1, 5)
        p1 = np.array(seq[k], float)
        v = p1 - p0
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None
    angles = []
    for nid in deg2_junc:
        es = incident[nid]
        if len(es) != 2:
            continue
        d1 = edge_dir_at(es[0], nid)
        d2 = edge_dir_at(es[1], nid)
        if d1 is None or d2 is None:
            continue
        # turning angle: 180 deg = straight through
        cosang = np.clip(np.dot(d1, d2), -1, 1)
        ang = math.degrees(math.acos(cosang))
        angles.append(ang)
    tiny = sum(1 for e in edges if not e.get("is_closed") and len(e["pixels"]) < 6)
    short = sum(1 for e in edges if not e.get("is_closed") and len(e["pixels"]) < 15)
    return {
        "n_nodes": len(nodes), "n_edges": len(edges),
        "junc_deg": dict(sorted(junc_deg.items())),
        "n_deg2_junc": len(deg2_junc),
        "deg2_through_angles": angles,
        "tiny_edges_lt6": tiny, "short_edges_lt15": short,
    }


def run_one(sid, view, svg):
    sk = f"{sid.replace('/','_')}_{view}"
    rp = WORK / f"{sk}_input.png"
    rasterize(svg, rp)
    r1 = s1.run(input_path=rp, output_dir=WORK, sketch_id=sk, config=CFG,
                model=None)
    r2 = s2.run(skeleton_path=r1.skeleton_path, output_dir=WORK,
                sketch_id=sk, config=CFG, model=None)
    graph = json.load(open(r2.graph_path))
    r3 = s3.run(graph_path=r2.graph_path, output_dir=WORK, sketch_id=sk,
                config=CFG, stroke_width=r1.mean_stroke_width)
    a = analyze_graph(graph)
    gt_paths = Path(svg).read_text().count("<path")
    # count primitive types
    prims = json.load(open(r3.primitives_path))["primitives"]
    ptypes = Counter(p["type"] for p in prims)
    return sk, gt_paths, r3.n_primitives, a, dict(ptypes)


if __name__ == "__main__":
    split = json.load(open(ROOT / "data/Drawing2CAD/train_val_test_split.json"))
    import random
    rng = random.Random(42)
    ids = list(split["test"]); rng.shuffle(ids)
    svg_root = ROOT / "data/Drawing2CAD/svg_raw"
    picked = []
    for sid in ids:
        outer, inner = sid.split("/")
        svg = svg_root / outer / inner / f"{inner}_Front.svg"
        if svg.exists():
            picked.append((sid, svg))
        if len(picked) >= int(sys.argv[1]) if len(sys.argv) > 1 else 15:
            break
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    picked = picked[:N]
    print(f"{'sketch':<22} {'gt':>3} {'prim':>4} {'edges':>5} {'d2j':>4} "
          f"{'tiny':>4} {'straight_d2j':>11}")
    tot_frag = 0
    for sid, svg in picked:
        sk, gt, nprim, a, ptypes = run_one(sid, "Front", svg)
        straight = sum(1 for ang in a["deg2_through_angles"] if ang > 150)
        frag = nprim > gt
        tot_frag += frag
        print(f"{sk:<22} {gt:>3} {nprim:>4} {a['n_edges']:>5} "
              f"{a['n_deg2_junc']:>4} {a['tiny_edges_lt6']:>4} "
              f"{straight:>11}  {ptypes if frag else ''}")
    print(f"\nfragmented: {tot_frag}/{len(picked)}")
