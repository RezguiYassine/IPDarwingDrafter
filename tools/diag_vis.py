"""Render GT raster, skeleton+nodes, and output SVG side by side for given sketches."""
import io, sys, json
from pathlib import Path
import numpy as np, cv2, cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "diag"

def load_png(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

def render(sk):
    inp = OUT / f"{sk}_input.png"
    skel = OUT / "cleaned" / f"{sk}_skeleton.png"
    graph = json.load(open(OUT / "graphs" / f"{sk}_graph.json"))
    H, W = graph["image_shape"]
    # crop to bbox of skeleton
    s = load_png(skel)
    ys, xs = np.where(s > 0)
    if len(xs) == 0:
        print("empty skel", sk); return
    pad = 15
    x0, x1 = max(0, xs.min()-pad), min(W, xs.max()+pad)
    y0, y1 = max(0, ys.min()-pad), min(H, ys.max()+pad)
    scale = max(1, int(400 / max(x1-x0, y1-y0)))
    def crop_scale(img, color=False):
        c = img[y0:y1, x0:x1]
        c = cv2.resize(c, ((x1-x0)*scale, (y1-y0)*scale), interpolation=cv2.INTER_NEAREST)
        return c
    inp_img = load_png(inp)
    canvas = cv2.cvtColor(crop_scale(255 - inp_img), cv2.COLOR_GRAY2BGR)
    canvas = 255 - canvas  # ink black
    skcanvas = cv2.cvtColor(crop_scale(s), cv2.COLOR_GRAY2BGR)
    # draw edges in random colors, nodes as circles
    import random
    rng = random.Random(0)
    ecanvas = np.full(((y1-y0)*scale, (x1-x0)*scale, 3), 255, np.uint8)
    for e in graph["edges"]:
        col = tuple(rng.randint(0,200) for _ in range(3))
        pts = [( (px[0]-x0)*scale, (px[1]-y0)*scale ) for px in e["pixels"]]
        for i in range(len(pts)-1):
            cv2.line(ecanvas, pts[i], pts[i+1], col, max(1,scale//2))
    for n in graph["nodes"]:
        cx, cy = (n["x"]-x0)*scale, (n["y"]-y0)*scale
        col = (0,0,255) if n["type"]=="junction" else (0,150,0) if n["type"]=="endpoint" else (255,0,0)
        cv2.circle(ecanvas, (cx,cy), max(2,scale), col, -1)
    h = max(canvas.shape[0], skcanvas.shape[0], ecanvas.shape[0])
    def padto(img):
        out = np.full((h, img.shape[1], 3), 255, np.uint8)
        out[:img.shape[0]] = img
        return out
    strip = np.hstack([padto(canvas), np.full((h,4,3),0,np.uint8),
                       padto(skcanvas), np.full((h,4,3),0,np.uint8), padto(ecanvas)])
    outp = OUT / f"VIS_{sk}.png"
    cv2.imwrite(str(outp), strip)
    print("wrote", outp, "| edges", len(graph["edges"]), "nodes", len(graph["nodes"]))

if __name__ == "__main__":
    for sk in sys.argv[1:]:
        render(sk)
