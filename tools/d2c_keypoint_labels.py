"""
Drawing2CAD keypoint label generator  (Puhachov roadmap — Phase 1)
==================================================================

Produces ground-truth keypoint labels for training the Stage 2 keypoint CNN.
For each Drawing2CAD (sample, view):

  1. Rasterize the ground-truth `svg_raw` SVG to a binary PNG and run the real
     Stage 1 (the exact preprocessing the pipeline uses at inference) to get the
     1-px skeleton the CNN will see.
  2. Parse the SVG's polyline geometry (D2C paths are absolute M/L polylines —
     no curves) and derive keypoints from vertex topology:
         degree 1                      -> endpoint   (channel 0)
         degree >= 3                   -> junction   (channel 1)
         degree 2 with a sharp turn    -> corner     (channel 2)
         degree 2 near-collinear       -> not a keypoint
  3. Snap each keypoint to the nearest skeleton pixel (absorbs the ~1 px
     thin-stroke skeleton offset); drop keypoints with no skeleton nearby.
  4. Cache `{skeleton, kps}` per sample as .npz. Dense Gaussian heatmap targets
     are built on the fly by the trainer (Phase 2) — caching them would be ~TBs.

Channel order (endpoint, junction, corner) matches PuhachovKeypointDetector.detect.

npz schema per sample:
    skeleton : uint8 (H, W)   0/255 1-px skeleton
    kps      : int32 (N, 3)   columns = x, y, type_idx   (0/1/2)
    meta     : json string    {sample_id, view, render_res, n_paths, n_dropped}

Usage (from project root):

    # 60-sample Front-view pilot on the train split + 24-tile audit sheet
    python -m tools.d2c_keypoint_labels --split train --views Front \
        --limit 60 --workers 4 --audit 24 \
        --output output/Drawing2CAD/kp_labels

    # full train split, all four views
    python -m tools.d2c_keypoint_labels --split train --views all --workers 8 \
        --output output/Drawing2CAD/kp_labels
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("stage1_preprocessing",):
    sys.path.insert(0, str(PROJECT_ROOT / _sub))
import stage1_preprocess  # noqa: E402
from tools.d2c_eval import _rasterize_svg_to_binary  # noqa: E402

logger = logging.getLogger("d2c_kp_labels")

D2C_ROOT_DEFAULT = PROJECT_ROOT / "data" / "Drawing2CAD"
ALL_VIEWS = ("Front", "Top", "Right", "FrontTopRight")

KP_ENDPOINT, KP_JUNCTION, KP_CORNER = 0, 1, 2
KP_NAMES = {KP_ENDPOINT: "endpoint", KP_JUNCTION: "junction", KP_CORNER: "corner"}
KP_COLORS = {  # BGR for cv2 audit overlay
    KP_ENDPOINT: (0, 170, 0),     # green
    KP_JUNCTION: (0, 0, 230),     # red
    KP_CORNER:   (230, 90, 0),    # blue
}

_NUM = r"-?\d+\.?\d*(?:[eE][-+]?\d+)?"
_TOKEN_RE = re.compile(rf"[MLCZ]|{_NUM}")
_DATTR_RE = re.compile(r'd="([^"]*)"')
_VIEWBOX_RE = re.compile(r'viewBox="\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*"')

CURVE_SAMPLES = 16   # polyline segments per cubic Bézier when flattening


# ─── SVG polyline parsing ────────────────────────────────────────────────────

def _flatten_cubic(p0, p1, p2, p3, k: int = CURVE_SAMPLES) -> list[tuple[float, float]]:
    """Flatten a cubic Bézier into k segments. Excludes p0, includes p3."""
    pts = []
    for i in range(1, k + 1):
        t = i / k
        mt = 1.0 - t
        a, b, c, d = mt * mt * mt, 3 * mt * mt * t, 3 * mt * t * t, t * t * t
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def parse_svg_polylines(svg_path: Path) -> tuple[list[list[tuple[float, float]]], float, float]:
    """Parse a D2C svg_raw file into flattened polyline subpaths + viewBox size.

    D2C paths use absolute M / L / C (cubic Bézier) only. Cubics are flattened
    into fine polylines so smooth curves yield smooth point chains (no phantom
    corners). Returns (subpaths, vb_w, vb_h) in viewBox coordinates.
    """
    text = svg_path.read_text()
    m = _VIEWBOX_RE.search(text)
    vb_w, vb_h = (float(m.group(3)), float(m.group(4))) if m else (200.0, 200.0)

    subpaths: list[list[tuple[float, float]]] = []
    for d in _DATTR_RE.findall(text):
        toks = _TOKEN_RE.findall(d)
        cur: list[tuple[float, float]] = []
        cmd = None
        cp = (0.0, 0.0)   # current point
        i, n = 0, len(toks)
        while i < n:
            if toks[i] in ("M", "L", "C", "Z"):
                cmd = toks[i]
                i += 1
            if cmd == "Z":
                continue
            try:
                if cmd == "M":
                    x, y = float(toks[i]), float(toks[i + 1]); i += 2
                    if len(cur) >= 2:
                        subpaths.append(cur)
                    cur = [(x, y)]; cp = (x, y)
                    cmd = "L"          # implicit lineto for extra M pairs (SVG spec)
                elif cmd == "L":
                    x, y = float(toks[i]), float(toks[i + 1]); i += 2
                    cur.append((x, y)); cp = (x, y)
                elif cmd == "C":
                    c1 = (float(toks[i]), float(toks[i + 1]))
                    c2 = (float(toks[i + 2]), float(toks[i + 3]))
                    end = (float(toks[i + 4]), float(toks[i + 5])); i += 6
                    cur.extend(_flatten_cubic(cp, c1, c2, end))
                    cp = end
                else:
                    i += 1
            except (IndexError, ValueError):
                break
        if len(cur) >= 2:
            subpaths.append(cur)
    return subpaths, vb_w, vb_h


# ─── Keypoint derivation from topology ───────────────────────────────────────

def derive_keypoints(
    subpaths: list[list[tuple[float, float]]],
    scale: float,
    corner_angle_deg: float = 35.0,
    merge_tol: float = 1e-3,
) -> list[tuple[int, int, int]]:
    """Classify vertices into endpoint / junction / corner keypoints.

    Topology and turn angles are computed in *float* viewBox space (vertices
    merged on a `merge_tol` grid), so finely-tessellated curves do not produce
    pixel-rounding-noise corners — a smooth circle correctly yields no
    keypoints. Coordinates are rounded to the render pixel grid only at output.

    `corner_angle_deg` is the minimum turn (deviation from straight) for a
    degree-2 vertex to count as a corner.
    """
    def _key(p: tuple[float, float]) -> tuple[int, int]:
        return (round(p[0] / merge_tol), round(p[1] / merge_tol))

    # merged-vertex key -> list of unit direction vectors (one per segment)
    incident: dict[tuple[int, int], list[tuple[float, float]]] = {}
    coords: dict[tuple[int, int], tuple[float, float]] = {}

    def _add(a: tuple[float, float], b: tuple[float, float]) -> None:
        ka, kb = _key(a), _key(b)
        if ka == kb:
            return
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy)
        incident.setdefault(ka, []).append((dx / n, dy / n))
        coords[ka] = a

    for sp in subpaths:
        for p, q in zip(sp[:-1], sp[1:]):
            _add(p, q)
            _add(q, p)

    cos_thresh = math.cos(math.radians(180.0 - corner_angle_deg))
    keypoints: list[tuple[int, int, int]] = []
    for k, dirs in incident.items():
        fx, fy = coords[k]
        x, y = int(round(fx * scale)), int(round(fy * scale))
        deg = len(dirs)
        if deg == 1:
            keypoints.append((x, y, KP_ENDPOINT))
        elif deg >= 3:
            keypoints.append((x, y, KP_JUNCTION))
        else:  # deg == 2 → corner if the two segments are not near-collinear
            d1, d2 = dirs[0], dirs[1]
            dot = d1[0] * d2[0] + d1[1] * d2[1]   # = cos(angle between dirs)
            # straight pass-through ⇒ dirs opposite ⇒ dot ≈ -1.
            # corner ⇒ angle between dirs < 180-thresh ⇒ dot > cos(180-thresh).
            if dot > cos_thresh:
                keypoints.append((x, y, KP_CORNER))
    return keypoints


def snap_keypoints(
    keypoints: list[tuple[int, int, int]],
    skeleton: np.ndarray,
    radius: int = 5,
) -> tuple[list[tuple[int, int, int]], int]:
    """Snap each keypoint to the nearest skeleton pixel within `radius`.

    Drops keypoints with no skeleton pixel nearby; de-duplicates identical
    (x, y, type) results.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return [], len(keypoints)
    tree = cKDTree(np.stack([xs, ys], axis=1))
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    dropped = 0
    for x, y, t in keypoints:
        dist, idx = tree.query([x, y], distance_upper_bound=radius + 1e-6)
        if not np.isfinite(dist):
            dropped += 1
            continue
        sx, sy = int(xs[idx]), int(ys[idx])
        key = (sx, sy, t)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out, dropped


# ─── Per-sample worker ───────────────────────────────────────────────────────

_WCFG: dict | None = None
_WS1 = None


def _worker_init(config_path: str, s1_device: str) -> None:
    global _WCFG, _WS1
    import yaml
    with open(config_path) as f:
        _WCFG = yaml.safe_load(f) or {}
    cfg_dir = Path(config_path).resolve().parent
    sc = _WCFG.get("sketchcleannet")
    if isinstance(sc, dict):
        w = sc.get("weights", "")
        if w and not Path(w).is_absolute():
            sc["weights"] = str(cfg_dir / w)
        if s1_device:
            sc["device"] = s1_device
    logging.basicConfig(level=logging.ERROR, force=True)
    _WS1 = stage1_preprocess.load_model(_WCFG)


def _process_one(job: tuple) -> dict:
    sample_id, view, svg_str, out_str, corner_angle, snap_radius = job
    svg_path = Path(svg_str)
    sketch_id = f"{sample_id.replace('/', '_')}_{view}"
    row = {"sample_id": sample_id, "view": view, "status": "ok", "error": None}

    tmp = Path(tempfile.mkdtemp(prefix="kpgen_"))
    try:
        raster = tmp / f"{sketch_id}_input.png"
        _rasterize_svg_to_binary(svg_path, raster)
        s1 = stage1_preprocess.run(
            input_path=raster, output_dir=tmp, sketch_id=sketch_id,
            config=_WCFG, model=_WS1,
        )
        skeleton = cv2.imread(str(s1.skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise RuntimeError("Stage 1 produced no skeleton")

        subpaths, vb_w, _vb_h = parse_svg_polylines(svg_path)
        scale = skeleton.shape[1] / vb_w
        kps = derive_keypoints(subpaths, scale, corner_angle)
        kps_snapped, n_dropped = snap_keypoints(kps, skeleton, snap_radius)

        kp_arr = (np.array(kps_snapped, dtype=np.int32)
                  if kps_snapped else np.zeros((0, 3), dtype=np.int32))
        meta = {
            "sample_id": sample_id, "view": view,
            "render_res": int(skeleton.shape[1]),
            "n_paths": len(subpaths), "n_dropped": n_dropped,
        }

        group = sample_id.split("/")[0]
        out_dir = Path(out_str) / group
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{sketch_id}.npz"
        np.savez_compressed(
            npz_path,
            skeleton=skeleton.astype(np.uint8),
            kps=kp_arr,
            meta=json.dumps(meta),
        )

        n_by_type = np.bincount(kp_arr[:, 2], minlength=3) if len(kp_arr) else [0, 0, 0]
        row.update({
            "npz_path": str(npz_path),
            "n_skeleton_px": int((skeleton > 0).sum()),
            "n_endpoint": int(n_by_type[KP_ENDPOINT]),
            "n_junction": int(n_by_type[KP_JUNCTION]),
            "n_corner": int(n_by_type[KP_CORNER]),
            "n_dropped": n_dropped,
            "n_paths": len(subpaths),
        })
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return row


# ─── Audit contact sheet ─────────────────────────────────────────────────────

def build_audit_sheet(npz_paths: list[Path], out_png: Path, cols: int = 6,
                      tile: int = 320) -> None:
    """Render skeleton + colour-coded keypoints for each npz into one sheet."""
    rows = max(1, math.ceil(len(npz_paths) / cols))
    sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for i, p in enumerate(npz_paths):
        d = np.load(p, allow_pickle=True)
        sk = d["skeleton"]
        kps = d["kps"]
        canvas = np.full((*sk.shape, 3), 255, np.uint8)
        canvas[sk > 0] = (190, 190, 190)          # skeleton = light grey
        for x, y, t in kps:
            cv2.circle(canvas, (int(x), int(y)), max(2, sk.shape[0] // 200),
                       KP_COLORS.get(int(t), (0, 0, 0)), -1)
        thumb = cv2.resize(canvas, (tile, tile), interpolation=cv2.INTER_AREA)
        meta = json.loads(str(d["meta"]))
        cv2.putText(thumb, meta["sample_id"].split("/")[-1], (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = thumb
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _resolve_views(spec: str) -> list[str]:
    if spec == "all":
        return list(ALL_VIEWS)
    return [v for v in spec.split(",") if v]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate D2C keypoint labels.")
    ap.add_argument("--split", default="train",
                    choices=["train", "validation", "test"])
    ap.add_argument("--views", default="Front", help="'all', or comma list")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config_d2c_eval.yaml"))
    ap.add_argument("--d2c-root", default=str(D2C_ROOT_DEFAULT))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "output/Drawing2CAD/kp_labels"))
    ap.add_argument("--s1-device", default="", help="override Stage 1 device (e.g. cuda)")
    ap.add_argument("--corner-angle", type=float, default=35.0)
    ap.add_argument("--snap-radius", type=int, default=5)
    ap.add_argument("--audit", type=int, default=0, help="render an N-tile audit sheet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d2c_root = Path(args.d2c_root)
    out_root = Path(args.output) / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    split = json.loads((d2c_root / "train_val_test_split.json").read_text())
    sample_ids = split[args.split]
    views = _resolve_views(args.views)

    jobs = []
    for sid in sample_ids:
        grp, num = sid.split("/")
        for view in views:
            svg = d2c_root / "svg_raw" / grp / num / f"{num}_{view}.svg"
            if svg.exists():
                jobs.append((sid, view, str(svg), str(out_root),
                             args.corner_angle, args.snap_radius))
    if args.limit:
        jobs = jobs[:args.limit]
    logger.info(f"Generating labels for {len(jobs)} (sample, view) pairs "
                f"→ {out_root}")

    manifest_path = out_root / "manifest.csv"
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.config, args.s1_device),
    ) as ex:
        futs = [ex.submit(_process_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), disable=False):
            rows.append(fut.result())

    ok = [r for r in rows if r["status"] == "ok"]
    err = [r for r in rows if r["status"] != "ok"]
    fields = ["sample_id", "view", "status", "error", "npz_path", "n_skeleton_px",
              "n_endpoint", "n_junction", "n_corner", "n_dropped", "n_paths"]
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    if ok:
        def _mean(k):
            return sum(r.get(k, 0) for r in ok) / len(ok)
        logger.info(
            f"\n  ok={len(ok)}  err={len(err)}\n"
            f"  mean keypoints/sample: "
            f"endpoint={_mean('n_endpoint'):.1f} "
            f"junction={_mean('n_junction'):.1f} "
            f"corner={_mean('n_corner'):.1f}\n"
            f"  mean dropped (no skeleton near GT): {_mean('n_dropped'):.2f}\n"
            f"  mean skeleton px: {_mean('n_skeleton_px'):.0f}\n"
            f"  manifest: {manifest_path}"
        )
    if err:
        logger.info(f"  first errors: "
                    f"{[(r['sample_id'], r['error']) for r in err[:3]]}")

    if args.audit and ok:
        rng = random.Random(args.seed)
        picks = rng.sample([Path(r["npz_path"]) for r in ok],
                           k=min(args.audit, len(ok)))
        sheet = out_root / f"audit_{args.split}_{'-'.join(views)}.png"
        build_audit_sheet(picks, sheet)
        logger.info(f"  audit sheet ({len(picks)} tiles): {sheet}\n"
                    f"  legend: green=endpoint  red=junction  blue=corner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
