"""
Error-attribution harness  (accuracy roadmap — step #1)
=======================================================

Decomposes the headline Drawing2CAD Chamfer into per-stage contributions, so we
know WHICH stage to fix instead of guessing. For each (sample, view) it
rasterizes the GT SVG, runs the pipeline, and measures the symmetric skeleton
Chamfer between the GT skeleton and each stage's geometry, in the SAME 1024-px
frame and against the SAME reference d2c_eval uses (skeletonize(GT raster)):

    gt_skel  → Stage-1 skeleton   (cleaning + skeletonization error)
    gt_skel  → Stage-2 graph      (topology tracing / simplification)
    gt_skel  → Stage-3 primitives (RANSAC fitting — can *improve* on the trace)

The per-stage deltas attribute the headline Chamfer. A negative delta means a
stage reduces error (e.g. fitting a clean line to a noisy chain).

Usage (from project root):

    python -m tools.error_attribution --limit 30 --views all \
        --config config_d2c_eval_puhachov_fusion.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("stage1_preprocessing", "stage2_strokeextraction",
            "stage3_primitivesfitting", "stage4_export"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))
import io  # noqa: E402

import cairosvg  # noqa: E402
from PIL import Image  # noqa: E402

import stage1_preprocess        # noqa: E402
import stage2_stroke_extract    # noqa: E402
import stage3_primitive_fit     # noqa: E402
import stage4_export            # noqa: E402
from tools.d2c_eval import _rasterize_svg_to_binary, RENDER_RES  # noqa: E402

D2C_ROOT = PROJECT_ROOT / "data" / "Drawing2CAD"
ALL_VIEWS = ("Front", "Top", "Right", "FrontTopRight")


def chamfer_sym(a: np.ndarray, b: np.ndarray):
    """Symmetric mean skeleton Chamfer between two boolean masks (px)."""
    if not a.any() or not b.any():
        return None
    dta = distance_transform_edt(~a)
    dtb = distance_transform_edt(~b)
    return float(0.5 * (dta[b].mean() + dtb[a].mean()))


def _gt_skeleton(svg: Path, work: Path) -> np.ndarray:
    raster = _rasterize_svg_to_binary(svg, work / "gt.png")     # ink=0, bg=255
    return skeletonize(raster < 128)


def _graph_mask(graph_path: Path, shape) -> np.ndarray:
    g = json.loads(Path(graph_path).read_text())
    scale = g.get("stage2_scale", 1.0) or 1.0
    m = np.zeros(shape, bool)
    H, W = shape
    for e in g.get("edges", []):
        for x, y in e.get("pixels", []):
            xi, yi = int(round(x / scale)), int(round(y / scale))
            if 0 <= yi < H and 0 <= xi < W:
                m[yi, xi] = True
    return m


def _svg_skeleton(svg_path: Path) -> np.ndarray:
    """Rasterize the real Stage-4 output SVG via cairosvg (the actual eval
    renderer) and skeletonize — this is the faithful Stage-3 geometry. Drawing
    primitives by hand mis-renders arc sweep direction and inflates the error."""
    png = cairosvg.svg2png(url=str(svg_path), output_width=RENDER_RES,
                           output_height=RENDER_RES, background_color="white")
    g = np.array(Image.open(io.BytesIO(png)).convert("L"))
    return skeletonize(g < 200)


def _prim_mask(prim_path: Path, shape) -> np.ndarray:  # legacy / unused
    d = json.loads(Path(prim_path).read_text())
    canvas = np.zeros(shape, np.uint8)

    def pt(p):
        return (int(round(p[0])), int(round(p[1])))

    for p in d.get("primitives", []):
        t = p.get("type")
        if t == "line":
            cv2.line(canvas, pt(p["p1"]), pt(p["p2"]), 255, 1)
        elif t in ("polyline", "polygon"):
            pts = np.array([pt(q) for q in p["points"]], np.int32)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], t == "polygon", 255, 1)
        elif t == "arc":
            c = pt(p["center"]); r = int(round(p["radius"]))
            a0, a1 = p["start_angle"], p["end_angle"]
            ang = np.linspace(a0, a1, max(8, int(abs(a1 - a0) * r)))
            xs = c[0] + r * np.cos(ang); ys = c[1] + r * np.sin(ang)
            for i in range(len(ang) - 1):
                cv2.line(canvas, (int(xs[i]), int(ys[i])),
                         (int(xs[i + 1]), int(ys[i + 1])), 255, 1)
        elif t == "ellipse":
            c = pt(p["center"])
            cv2.ellipse(canvas, c, (int(round(p["a"])), int(round(p["b"]))),
                        float(np.degrees(p["angle"])), 0, 360, 255, 1)
    return canvas > 0


def _fit(mask: np.ndarray, shape) -> np.ndarray:
    """Resize a boolean mask to `shape` (nearest) if needed."""
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0


def attribute_one(svg: Path, cfg, s1m, s2m, work: Path):
    sid = svg.stem
    gt = _gt_skeleton(svg, work)
    shape = gt.shape

    s1 = stage1_preprocess.run(input_path=work / "gt.png", output_dir=work,
                               sketch_id=sid, config=cfg, model=s1m)
    s1_skel = _fit(cv2.imread(str(s1.skeleton_path), 0) > 0, shape)

    s2 = stage2_stroke_extract.run(skeleton_path=s1.skeleton_path, output_dir=work,
                                   sketch_id=sid, config=cfg, model=s2m)
    s2_mask = _fit(_graph_mask(s2.graph_path, shape), shape)

    s3 = stage3_primitive_fit.run(graph_path=s2.graph_path, output_dir=work,
                                  sketch_id=sid, config=cfg,
                                  stroke_width=s1.mean_stroke_width)
    s4 = stage4_export.run(input_json=s3.primitives_path, output_dir=work,
                           sketch_id=sid, formats=("svg",), dxf_mode="basic")
    s3_mask = _fit(_svg_skeleton(s4.svg_path), shape)   # real renderer

    return {
        "stage1": chamfer_sym(gt, s1_skel),
        "stage2": chamfer_sym(gt, s2_mask),
        "stage3": chamfer_sym(gt, s3_mask),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config_d2c_eval_puhachov_fusion.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--views", default="all")
    ap.add_argument("--limit", type=int, default=30, help="samples per view")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg_dir = Path(args.config).resolve().parent
    for sec in ("sketchcleannet", "puhachov"):
        b = cfg.get(sec)
        if isinstance(b, dict) and b.get("weights") and not Path(b["weights"]).is_absolute():
            b["weights"] = str(cfg_dir / b["weights"])
    s1m = stage1_preprocess.load_model(cfg)
    s2m = stage2_stroke_extract.load_model(cfg)
    print("CNN keypoint model:", "loaded" if s2m else "None (CN path)")

    views = list(ALL_VIEWS) if args.views == "all" else args.views.split(",")
    split = json.loads((D2C_ROOT / "train_val_test_split.json").read_text())[args.split]
    import random
    random.Random(args.seed).shuffle(split)

    rows = defaultdict(list)   # view -> list of {stage1,stage2,stage3}
    work = Path(tempfile.mkdtemp(prefix="errattr_"))
    for view in views:
        n = 0
        for sid in split:
            grp, num = sid.split("/")
            svg = D2C_ROOT / "svg_raw" / grp / num / f"{num}_{view}.svg"
            if not svg.exists():
                continue
            try:
                rows[view].append(attribute_one(svg, cfg, s1m, s2m, work))
            except Exception as exc:
                print(f"  [skip] {num}_{view}: {type(exc).__name__}: {exc}")
                continue
            n += 1
            if n >= args.limit:
                break
        print(f"  {view}: {n} samples")

    def mean(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")

    print("\n── Chamfer-to-GT-skeleton by stage (px, lower = better) ──")
    print(f"{'view':16s}{'Stage1':>9s}{'Stage2':>9s}{'Stage3':>9s}   (Δ2  Δ3)")
    allr = [r for rs in rows.values() for r in rs]
    for view in views + ["__ALL__"]:
        rs = allr if view == "__ALL__" else rows[view]
        if not rs:
            continue
        c1, c2, c3 = mean(rs, "stage1"), mean(rs, "stage2"), mean(rs, "stage3")
        print(f"{view:16s}{c1:9.3f}{c2:9.3f}{c3:9.3f}   ({c2-c1:+.2f} {c3-c2:+.2f})  n={len(rs)}")
    print("\nReading: Stage1 = skeleton error (vs ideal skeletonization); "
          "Δ2 = topology adds/removes; Δ3 = fitting adds/removes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
