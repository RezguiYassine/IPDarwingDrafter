#!/usr/bin/env python3
"""Interactive correction tool for patent keypoint GT (endpoint/junction/corner).

Patents are raster scans with no vector GT, so keypoints can't be auto-derived.
But the pipeline has already produced stroke graphs for tens of thousands of
patent figures — their nodes are endpoint/junction pseudo-labels and their edges
give corner candidates. This tool seeds each figure with those pseudo-labels and
lets a human CORRECT (add missed / delete wrong / reclassify) rather than label
from scratch, then saves the trainer-ready npz {skeleton, kps(x,y,type)} — the
same schema as the D2C and ArchCAD keypoint sets.

Skeleton is reconstructed from the graph's edge pixels, so it is always in the
exact frame the keypoints live in (the post-Stage-2 frame the CNN operates on).

Controls
  scroll        zoom to cursor          left-drag     pan
  left-click    add keypoint (active type, snapped to skeleton)
  right-click   delete nearest keypoint
  right-drag    box-delete (wipe all keypoints in the rectangle — fast cleanup
                of over-produced junction seeds along curves)
  e / j / c     set active type = endpoint / junction / corner
  r             reclassify nearest keypoint → active type
  u             undo last edit
  a  or  Enter  accept: save npz + next
  s             skip this figure (no save) + next
  q             save current + quit

    python -m tools.patent_kp_label \
        --graphs-root output --sample 60 \
        --out output/PatentData/kp_gt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from tools.d2c_keypoint_labels import derive_keypoints, snap_keypoints  # noqa: E402

KP_ENDPOINT, KP_JUNCTION, KP_CORNER = 0, 1, 2
NAMES = {KP_ENDPOINT: "endpoint", KP_JUNCTION: "junction", KP_CORNER: "corner"}
COLORS = {KP_ENDPOINT: (0.0, 0.8, 0.0), KP_JUNCTION: (1.0, 0.0, 0.0),
          KP_CORNER: (0.1, 0.45, 1.0)}
KEY_TYPE = {"e": KP_ENDPOINT, "j": KP_JUNCTION, "c": KP_CORNER}
_SNAP = 5


# ─── Pseudo-labels + skeleton from a graph ────────────────────────────────────

def build_from_graph(graph: dict, corner_angle: float = 35.0):
    """Return (skeleton uint8 HxW, kps list[(x,y,type)]) seeded from the graph."""
    H, W = int(graph["image_shape"][0]), int(graph["image_shape"][1])
    skel = np.zeros((H, W), np.uint8)
    subpaths = []
    for e in graph.get("edges", []):
        pix = e.get("pixels") or []
        for x, y in pix:
            if 0 <= y < H and 0 <= x < W:
                skel[int(y), int(x)] = 255
        pts = e.get("smooth_pts") or pix
        if len(pts) >= 2:
            subpaths.append([(float(p[0]), float(p[1])) for p in pts])

    # Endpoints + junctions from the graph's own typed nodes — the pipeline's
    # de-fragmented, hachure-removed estimate. It over-produces (some fragment-
    # boundary junctions along curves), but on noisy scanned patents it is the
    # cleanest available automatic seed; raw skeleton pixel-degree gives 1000s of
    # spurious endpoints from spurs/hatching. The human trims the excess with
    # box-delete (shift-drag) and adds what's missing.
    kps: list[tuple[int, int, int]] = []
    for n in graph.get("nodes", []):
        t = n.get("type")
        if t == "endpoint":
            kps.append((int(n["x"]), int(n["y"]), KP_ENDPOINT))
        elif t == "junction":
            kps.append((int(n["x"]), int(n["y"]), KP_JUNCTION))
        elif t == "corner":
            kps.append((int(n["x"]), int(n["y"]), KP_CORNER))

    # Corner CANDIDATES from edge geometry (degree-2 sharp turns).
    vec = derive_keypoints(subpaths, scale=1.0, corner_angle_deg=corner_angle,
                           merge_tol=0.5)
    corners = [(x, y, KP_CORNER) for (x, y, t) in vec if t == KP_CORNER]
    corners, _ = snap_keypoints(corners, skel, _SNAP)
    occ = {(x, y) for x, y, _ in kps}          # drop corners on an end/junction
    for x, y, t in corners:
        if not any(abs(x - ox) <= _SNAP and abs(y - oy) <= _SNAP for ox, oy in occ):
            kps.append((x, y, t))
    return skel, [list(k) for k in kps]


# ─── Interactive session ──────────────────────────────────────────────────────

class _Session:
    def __init__(self, ax, skel, kps):
        self.ax = ax
        self.H, self.W = skel.shape
        self.skel = skel
        self.disp = 255 - skel                      # black strokes on white
        self.kps = kps                              # list of [x, y, type]
        self.active = KP_CORNER                     # start on the weak channel
        self.undo_stack: list = []
        self.done = False
        self.quit_all = False
        self.skipped = False
        self._press = None
        self._panning = False
        self._pan_xlim = self._pan_ylim = None

    # -- geometry helpers --
    def _snap(self, x, y):
        xi, yi = int(round(x)), int(round(y))
        y0, y1 = max(0, yi - _SNAP), min(self.H, yi + _SNAP + 1)
        x0, x1 = max(0, xi - _SNAP), min(self.W, xi + _SNAP + 1)
        ys, xs = np.nonzero(self.skel[y0:y1, x0:x1])
        if len(xs) == 0:
            return xi, yi
        d2 = (xs - (xi - x0)) ** 2 + (ys - (yi - y0)) ** 2
        k = int(np.argmin(d2))
        return int(x0 + xs[k]), int(y0 + ys[k])

    def _nearest(self, x, y, max_d=12):
        if not self.kps:
            return None
        d2 = [(i, (k[0] - x) ** 2 + (k[1] - y) ** 2) for i, k in enumerate(self.kps)]
        i, dd = min(d2, key=lambda t: t[1])
        return i if dd <= max_d * max_d else None

    # -- draw --
    def redraw(self, reset=False):
        if not reset:
            xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self.disp, cmap="gray", vmin=0, vmax=255)
        by = {0: [], 1: [], 2: []}
        for x, y, t in self.kps:
            by[t].append((x, y))
        for t, pts in by.items():
            if pts:
                xs, ys = zip(*pts)
                self.ax.scatter(xs, ys, s=26, c=[COLORS[t]], edgecolors="k",
                                linewidths=0.4, zorder=3)
        n = {0: len(by[0]), 1: len(by[1]), 2: len(by[2])}
        self.ax.set_title(
            f"ADD={NAMES[self.active].upper()}  |  end={n[0]} junc={n[1]} corner={n[2]}"
            f"   [e/j/c type · L-click add · R-click del · r reclass · u undo · "
            f"a save+next · s skip · q quit]", fontsize=9)
        if reset:
            self.ax.set_xlim(0, self.W); self.ax.set_ylim(self.H, 0)
        else:
            self.ax.set_xlim(xl); self.ax.set_ylim(yl)
        self.ax.axis("off")
        self.ax.figure.canvas.draw_idle()

    # -- events --
    def on_press(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        self._press = (e.xdata, e.ydata, e.button)
        if e.button == 1:
            self._panning = False
            self._pan_xlim, self._pan_ylim = self.ax.get_xlim(), self.ax.get_ylim()
            self._pan_start = (e.xdata, e.ydata)

    def on_motion(self, e):
        if self._press is None or e.inaxes != self.ax or e.xdata is None:
            return
        if self._press[2] == 1:
            dx = e.xdata - self._pan_start[0]; dy = e.ydata - self._pan_start[1]
            if not self._panning and (dx * dx + dy * dy) > 9:
                self._panning = True
            if self._panning:
                x0, x1 = self._pan_xlim; y0, y1 = self._pan_ylim
                self.ax.set_xlim(x0 - dx, x1 - dx); self.ax.set_ylim(y0 - dy, y1 - dy)
                self.ax.figure.canvas.draw_idle()

    def on_release(self, e):
        if self._press is None:
            return
        px, py, btn = self._press
        self._press = None
        if e.inaxes != self.ax or e.xdata is None:
            self._panning = False
            return
        if btn == 1 and not self._panning:
            x, y = self._snap(e.xdata, e.ydata)
            self.undo_stack.append(list(self.kps))
            self.kps.append([x, y, self.active])
            self.redraw()
        elif btn == 3:
            moved = (abs(e.xdata - px) > 4 or abs(e.ydata - py) > 4)
            if moved:                                   # box-delete
                x0, x1 = sorted((px, e.xdata)); y0, y1 = sorted((py, e.ydata))
                keep = [k for k in self.kps
                        if not (x0 <= k[0] <= x1 and y0 <= k[1] <= y1)]
                if len(keep) != len(self.kps):
                    self.undo_stack.append(list(self.kps)); self.kps = keep
                self.redraw()
            else:                                       # delete nearest
                i = self._nearest(e.xdata, e.ydata)
                if i is not None:
                    self.undo_stack.append(list(self.kps))
                    self.kps.pop(i); self.redraw()
        self._panning = False

    def on_scroll(self, e):
        if e.inaxes != self.ax or e.xdata is None:
            return
        f = 0.8 if e.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(e.xdata - (e.xdata - x0) * f, e.xdata + (x1 - e.xdata) * f)
        self.ax.set_ylim(e.ydata - (e.ydata - y0) * f, e.ydata + (y1 - e.ydata) * f)
        self.ax.figure.canvas.draw_idle()

    def on_key(self, e):
        k = (e.key or "").lower()
        if k in KEY_TYPE:
            self.active = KEY_TYPE[k]; self.redraw()
        elif k == "r":
            i = self._nearest(e.xdata, e.ydata) if e.xdata is not None else None
            if i is not None:
                self.undo_stack.append(list(self.kps))
                self.kps[i][2] = self.active; self.redraw()
        elif k == "u" and self.undo_stack:
            self.kps = self.undo_stack.pop(); self.redraw()
        elif k in ("a", "enter"):
            self.done = True; plt.close(self.ax.figure)
        elif k == "s":
            self.done = True; self.skipped = True; plt.close(self.ax.figure)
        elif k == "q":
            self.done = True; self.quit_all = True; plt.close(self.ax.figure)

    def on_close(self, e):
        self.done = True; self.quit_all = True

    def run(self):
        fig = self.ax.figure
        cids = [fig.canvas.mpl_connect(ev, fn) for ev, fn in [
            ("button_press_event", self.on_press),
            ("button_release_event", self.on_release),
            ("motion_notify_event", self.on_motion),
            ("scroll_event", self.on_scroll),
            ("key_press_event", self.on_key),
            ("close_event", self.on_close)]]
        self.redraw(reset=True)
        plt.show(block=True)
        for c in cids:
            try: fig.canvas.mpl_disconnect(c)
            except Exception: pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def _discover(root: str, min_edges: int, max_edges: int, seed: int, n: int):
    files = glob.glob(os.path.join(root, "**", "graphs", "*_graph.json"), recursive=True)
    random.Random(seed).shuffle(files)
    picked = []
    for f in files:
        try:
            ne = len(json.load(open(f)).get("edges", []))
        except Exception:
            continue
        if min_edges <= ne <= max_edges:
            picked.append(f)
        if len(picked) >= n:
            break
    return sorted(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-root", default="output",
                    help="root to discover **/graphs/*_graph.json")
    ap.add_argument("--graphs-list", default="",
                    help="optional text file of explicit graph paths (one per line)")
    ap.add_argument("--out", default="output/PatentData/kp_gt")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--min-edges", type=int, default=40)
    ap.add_argument("--max-edges", type=int, default=1500)
    ap.add_argument("--corner-angle", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--redo", action="store_true", help="re-review already-saved figures")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.graphs_list:
        graphs = [ln.strip() for ln in open(a.graphs_list) if ln.strip()]
    else:
        graphs = _discover(a.graphs_root, a.min_edges, a.max_edges, a.seed, a.sample)
    if not graphs:
        raise SystemExit("no graphs found — check --graphs-root / filters")

    print(f"{len(graphs)} figures queued → {a.out}")
    n_done = n_skip = 0
    for gi, gp in enumerate(graphs, 1):
        graph = json.load(open(gp))
        sid = graph.get("sketch_id") or os.path.basename(gp).replace("_graph.json", "")
        # patent id = the run/EP dir two levels above graphs/
        parts = gp.split(os.sep)
        patent = parts[-3] if len(parts) >= 3 else "unknown"
        stem = f"{patent}__{sid}"
        out_path = os.path.join(a.out, f"{stem}.npz")
        if os.path.exists(out_path) and not a.redo:
            n_done += 1; continue

        skel, kps = build_from_graph(graph, a.corner_angle)
        if not skel.any():
            print(f"  [{gi}/{len(graphs)}] {stem}: empty skeleton, skipping"); continue

        fig, ax = plt.subplots(figsize=(13, 11))
        fig.canvas.manager.set_window_title(f"[{gi}/{len(graphs)}] {stem}")
        sess = _Session(ax, skel, kps)
        sess.run()

        if sess.skipped:
            n_skip += 1
            print(f"  [{gi}/{len(graphs)}] {stem}: SKIPPED")
        else:
            arr = np.array(sess.kps, dtype=np.int32) if sess.kps else np.zeros((0, 3), np.int32)
            np.savez_compressed(out_path, skeleton=skel.astype(np.uint8), kps=arr,
                                patent=patent, sketch=sid, source_graph=gp, reviewed=True)
            by = np.bincount(arr[:, 2], minlength=3) if len(arr) else [0, 0, 0]
            n_done += 1
            print(f"  [{gi}/{len(graphs)}] {stem}: saved "
                  f"end={by[0]} junc={by[1]} corner={by[2]} → {out_path}")

        if sess.quit_all:
            print("quit requested."); break

    print(f"\ndone: {n_done} saved, {n_skip} skipped → {a.out}")


if __name__ == "__main__":
    main()
