"""Run S1-S3 on a patent TIF (production config) and report fragmentation + visualize."""
import sys, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, cv2, yaml

ROOT = Path(__file__).resolve().parent.parent
for sub in ("stage1_preprocessing","stage2_strokeextraction","stage3_primitivesfitting","stage4_export"):
    sys.path.insert(0, str(ROOT/sub))
import stage1_preprocess as s1, stage2_stroke_extract as s2, stage3_primitive_fit as s3
CFG = yaml.safe_load(open(ROOT/"config.yaml"))
CFG.setdefault("puhachov",{})["weights"]=""   # CN path
OUT = ROOT/"output"/"diag_patent"
OUT.mkdir(parents=True, exist_ok=True)

def analyze(graph):
    nodes={n["id"]:n for n in graph["nodes"]}
    deg=Counter()
    for e in graph["edges"]:
        if e.get("is_closed"): continue
        deg[e["source"]]+=1; deg[e["target"]]+=1
    open_e=[e for e in graph["edges"] if not e.get("is_closed")]
    closed_e=[e for e in graph["edges"] if e.get("is_closed")]
    lens=sorted(len(e["pixels"]) for e in open_e)
    return dict(nodes=len(nodes), edges=len(graph["edges"]),
                open=len(open_e), closed=len(closed_e),
                tiny=sum(1 for L in lens if L<6),
                short=sum(1 for L in lens if L<15),
                med_len=lens[len(lens)//2] if lens else 0,
                deg2_junc=sum(1 for nid,n in nodes.items() if n["type"]=="junction" and deg.get(nid)==2))

def run(tif):
    sk = Path(tif).stem
    r1=s1.run(input_path=Path(tif), output_dir=OUT, sketch_id=sk, config=CFG, model=None)
    r2=s2.run(skeleton_path=r1.skeleton_path, output_dir=OUT, sketch_id=sk, config=CFG, model=None)
    g=json.load(open(r2.graph_path))
    r3=s3.run(graph_path=r2.graph_path, output_dir=OUT, sketch_id=sk, config=CFG, stroke_width=r1.mean_stroke_width)
    a=analyze(g)
    prims=json.load(open(r3.primitives_path))["primitives"]
    pt=Counter(p["type"] for p in prims)
    print(f"{sk:<28} prims={r3.n_primitives:<5} {a} {dict(pt)}")
    return sk, g

if __name__=="__main__":
    for tif in sys.argv[1:]:
        run(tif)
