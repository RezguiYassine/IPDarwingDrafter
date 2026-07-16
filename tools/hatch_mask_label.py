#!/usr/bin/env python3
"""Ground-truth hatch-area labeller (ad-hoc, independent of the detector).

Unlike tools/hatch_label.py (which judges the upstream detector's *candidate*
regions yes/no), this tool has you draw the true hatch areas directly on the
original figure -- so it also captures hatching the detector currently misses.
Output is a separate ground-truth set; it does not touch output/PatentData/hatch_pilot.

If a sheet contains several separate drawings (e.g. multiple FIG.X panels on
one page), it auto-splits them by ink-blob/whitespace-gap detection and walks
you through them one at a time.

    python -m tools.hatch_mask_label --pack-source output/PatentData/hatch_pilot \
        --out output/PatentData/hatch_gt

To fix specific mislabeled drawings found via tools/hatch_review.py (read-only --
it cannot edit) without re-reviewing everything, target them directly:

    python -m tools.hatch_mask_label --out output/PatentData/hatch_gt \
        --only EP123B1__F0002__d1 EP456A1__F0003

(a bare patent__sketch, with no __dN, redoes all of that sheet's sub-drawings.)

Two drawing modes, toggled with spacebar:
    polygon mode (default)  click vertices, enter to close -- for any shape
    circle mode              for circular hatching (e.g. tube cross-sections):
                              click the center, then click the boundary to set
                              the radius (that 2nd click snaps to the nearest
                              edge). Click-drag inside any existing circle to
                              move it (works any time, not just right after
                              drawing it). Circles are saved as many-sided
                              polygons, so the output schema is unchanged.

Mouse:
    left-click          polygon mode: add a vertex (snapped to the nearest
                         strong edge within a small radius, so a few-px-off
                         click still lands on the true boundary)
                         circle mode: 1st click = center, 2nd click = radius
                         (snapped); click-drag inside an existing circle moves it
    scroll wheel        zoom in/out, centered on the cursor (clamped to the
                         drawing -- you can never scroll/zoom off it)
    middle/right-drag   pan while zoomed in
Keys (in the image window):
    space            tab between polygon mode and circle mode
    enter            (polygon mode) finish current polygon (needs >=3 points)
    z / backspace    undo last point/circle (or last finished shape if none pending)
    r                reset all shapes on this drawing
    f / home         fit-to-view (reset zoom/pan only, shapes untouched)
    g                toggle the faint guide overlay of the detector's candidate regions
    n / d            done with this drawing -- save + next (0 shapes = no hatch here)
    q                save + quit entirely
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Several of our keys collide with matplotlib's built-in keymap (e.g. 's' pops
# up a save-figure dialog, 'r'/'f'/'g'/backspace are bound to home/fullscreen/
# grid/back-history) -- disable those defaults so our handlers are the only ones.
for _k in ("keymap.save", "keymap.quit", "keymap.home", "keymap.fullscreen",
           "keymap.grid", "keymap.grid_minor", "keymap.back", "keymap.forward",
           "keymap.pan", "keymap.zoom", "keymap.xscale", "keymap.yscale"):
    mpl.rcParams[_k] = []

_DEFAULT_TIF_ROOT = "data/PatentData/ReorganisedData"
_DEFAULT_SELECTED_TIF_ROOT = None   # when --selected-file is used the tif path is inside the JSON
_INSTRUCTIONS = (
    "click: vertex/circle | scroll: zoom | mid/right-drag: pan | space: mode | f: fit | "
    "enter: close | z: undo | r: reset | g: guides | n/d: done | q: save+quit"
)
_SNAP_RADIUS = 10     # px, in original image coords
_SCROLL_ZOOM = 1.2
_CIRCLE_SIDES = 64
_MIN_CIRCLE_RADIUS = 3   # px, ignore degenerate clicks


def _circle_to_polygon(center, radius, n=_CIRCLE_SIDES):
    cx, cy = center
    return [(cx + radius * np.cos(t), cy + radius * np.sin(t))
            for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]


def _hachure_overlay(tif_rgb: np.ndarray, graph_path: str) -> np.ndarray:
    """Bake removed_hachures edge pixels (orange tint) onto a copy of tif_rgb.
    Returns tif_rgb unchanged if the graph has no hachure edges."""
    if not graph_path or not os.path.exists(graph_path):
        return tif_rgb
    d = json.load(open(graph_path))
    edges = d.get("removed_hachures", [])
    if not edges:
        return tif_rgb
    scale = 1.0 / d.get("stage2_scale", 1.0)
    H, W = tif_rgb.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for e in edges:
        for px, py in e.get("pixels", []):
            x, y = int(round(px * scale)), int(round(py * scale))
            if 0 <= x < W and 0 <= y < H:
                mask[y, x] = 1
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    overlay = tif_rgb.copy()
    where = mask > 0
    overlay[where] = (overlay[where] * 0.35 + np.array([255, 120, 0]) * 0.65).clip(0, 255).astype(np.uint8)
    return overlay


def _figure_list(pack_source):
    out = []
    for path in sorted(glob.glob(os.path.join(pack_source, "*_regions.json"))):
        doc = json.load(open(path))
        out.append((doc["patent"], doc["sketch"]))
    return out


def _candidate_guides(pack_source, patent, sketch, crop_box):
    """Detector's candidate region boundaries (hatch_pilot), offset into crop-local
    coords and filtered to those overlapping the crop. Best-effort: returns [] if
    the pack/figure isn't there."""
    path = os.path.join(pack_source, f"{patent}__{sketch}_regions.json")
    if not os.path.exists(path):
        return []
    x0, y0, x1, y1 = crop_box
    out = []
    for r in json.load(open(path)).get("regions", []):
        pts = r.get("boundary") or []
        if not pts:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            out.append([[p[0] - x0, p[1] - y0] for p in pts])
    return out


def _merge_boxes(boxes, gap):
    """Iteratively union any boxes that overlap once expanded by `gap`."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            ax0, ay0, ax1, ay1 = a
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx0, by0, bx1, by1 = boxes[j]
                if (ax0 - gap < bx1 and bx0 - gap < ax1 and
                        ay0 - gap < by1 and by0 - gap < ay1):
                    ax0, ay0 = min(ax0, bx0), min(ay0, by0)
                    ax1, ay1 = max(ax1, bx1), max(ay1, by1)
                    used[j] = True
                    changed = True
            out.append((ax0, ay0, ax1, ay1))
        boxes = out
    return boxes


def split_subdrawings(gray, min_area=4000, dilate_px=35, gap_px=40, pad=15):
    """Auto-split a sheet into separate drawings by ink-blob/whitespace gaps.

    Returns a sorted (reading-order) list of (x0, y0, x1, y1) boxes, clipped
    to the image. Falls back to the whole image if nothing distinct is found.
    """
    H, W = gray.shape[:2]
    ink = (gray < 250).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    dilated = cv2.dilate(ink, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    if not boxes:
        return [(0, 0, W, H)]
    boxes = _merge_boxes(boxes, gap_px)
    padded = []
    for x0, y0, x1, y1 in boxes:
        padded.append((max(0, x0 - pad), max(0, y0 - pad),
                        min(W, x1 + pad), min(H, y1 + pad)))
    if len(padded) == 1:
        return padded
    return sorted(padded, key=lambda b: (b[1], b[0]))


class _Session:
    def __init__(self, ax, img_shape, edge_mask, guides=None, img_overlay=None):
        self.ax = ax
        self.H, self.W = img_shape[:2]
        self.edge_mask = edge_mask
        self.guides = guides or []
        self.img_overlay = img_overlay   # pre-baked image with hachure tint (full-figure mode)
        # guides on by default if polygon guides exist; overlay off by default (toggle with g)
        self.show_guides = bool(self.guides)
        self.mode = "polygon"  # or "circle"
        self.polygons = []   # list of list[(x,y)]
        self.current = []    # list[(x,y)]
        self.circles = []    # list of {"center": (x,y), "radius": r}
        self._circle_pending_center = None
        self._dragging_circle_idx = None
        self._drag_offset = None
        self.done = False
        self.quit_all = False
        self._panning = False
        self._pan_start_display = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

    def _circle_at(self, x, y):
        """Index of the topmost (most recently drawn) circle containing (x,y), if any."""
        for i in range(len(self.circles) - 1, -1, -1):
            cx, cy = self.circles[i]["center"]
            if (x - cx) ** 2 + (y - cy) ** 2 <= self.circles[i]["radius"] ** 2:
                return i
        return None

    @staticmethod
    def _clamp_range(lo, hi, full):
        """Clamp [lo, hi] to fit within [0, full] without exceeding its span."""
        span = hi - lo
        if span >= full:
            return 0.0, float(full)
        if lo < 0:
            hi -= lo; lo = 0.0
        if hi > full:
            lo -= (hi - full); hi = float(full)
        return max(0.0, lo), min(float(full), hi)

    def _set_view(self, x0, x1, ybottom, ytop):
        """Set xlim/ylim, clamped to the image bounds (ylim is inverted: ytop < ybottom)."""
        x0, x1 = self._clamp_range(x0, x1, self.W)
        ytop, ybottom = self._clamp_range(ytop, ybottom, self.H)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(ybottom, ytop)

    def _snap(self, x, y):
        if self.edge_mask is None or x is None or y is None:
            return x, y
        xi, yi = int(round(x)), int(round(y))
        r = _SNAP_RADIUS
        y0, y1 = max(0, yi - r), min(self.H, yi + r + 1)
        x0, x1 = max(0, xi - r), min(self.W, xi + r + 1)
        patch = self.edge_mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(patch)
        if len(xs) == 0:
            return x, y
        cy, cx = yi - y0, xi - x0
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        k = int(np.argmin(d2))
        return float(x0 + xs[k]), float(y0 + ys[k])

    def redraw(self, img, title, reset_view=False):
        if not reset_view:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        # In full-figure mode img_overlay holds the hachure-tinted image; the g
        # key switches between raw and tinted.  In pack-source mode the overlay
        # is None and show_guides controls the cyan polygon guide lines instead.
        display = (self.img_overlay if (self.show_guides and self.img_overlay is not None) else img)
        self.ax.imshow(display)
        if self.show_guides and self.img_overlay is None:
            for poly in self.guides:
                xs = [p[0] for p in poly] + [poly[0][0]]
                ys = [p[1] for p in poly] + [poly[0][1]]
                self.ax.plot(xs, ys, "--", color="cyan", linewidth=1.5, alpha=0.6)
        for poly in self.polygons:
            xs = [p[0] for p in poly] + [poly[0][0]]
            ys = [p[1] for p in poly] + [poly[0][1]]
            self.ax.plot(xs, ys, "-o", color="lime", linewidth=2, markersize=3)
        for c in self.circles:
            self.ax.add_patch(plt.Circle(c["center"], c["radius"], fill=False,
                                          color="lime", linewidth=2))
        if self.current:
            xs = [p[0] for p in self.current]
            ys = [p[1] for p in self.current]
            self.ax.plot(xs, ys, "-o", color="red", linewidth=2, markersize=4)
        if self._circle_pending_center is not None:
            self.ax.plot(*self._circle_pending_center, "x", color="red", markersize=10,
                          markeredgewidth=2)
        if reset_view:
            self._set_view(0, self.W, self.H, 0)
        else:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.ax.axis("off")
        n_lab = len(self.polygons) + len(self.circles)
        if self.img_overlay is not None:
            g_tag = f"  hachures:{'on' if self.show_guides else 'off'}"
        elif self.guides:
            g_tag = f"  guides:{'on' if self.show_guides else 'off'}"
        else:
            g_tag = ""
        self.ax.set_title(f"{title}\nmode:{self.mode}  {n_lab} hatch area(s) marked{g_tag}  "
                           f"|  {_INSTRUCTIONS}", fontsize=9)
        self.ax.figure.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            x, y = event.xdata, event.ydata
            if self.mode == "circle":
                if self._circle_pending_center is not None:
                    sx, sy = self._snap(x, y)
                    cx, cy = self._circle_pending_center
                    r = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
                    self._circle_pending_center = None
                    if r >= _MIN_CIRCLE_RADIUS:
                        self.circles.append({"center": (cx, cy), "radius": r})
                    else:
                        print("  circle too small, discarded")
                    self.redraw(self._img, self._title)
                    return
                hit = self._circle_at(x, y)
                if hit is not None:
                    self._dragging_circle_idx = hit
                    cx, cy = self.circles[hit]["center"]
                    self._drag_offset = (x - cx, y - cy)
                    return
                self._circle_pending_center = (x, y)
                self.redraw(self._img, self._title)
                return
            x, y = self._snap(x, y)
            self.current.append((x, y))
            self.redraw(self._img, self._title)
        elif event.button in (2, 3):
            self._panning = True
            self._pan_start_display = (event.x, event.y)
            self._pan_start_xlim = self.ax.get_xlim()
            self._pan_start_ylim = self.ax.get_ylim()

    def on_release(self, event):
        if event.button == 1 and self._dragging_circle_idx is not None:
            self._dragging_circle_idx = None
            self._drag_offset = None
        elif event.button in (2, 3):
            self._panning = False

    def on_motion(self, event):
        if self._dragging_circle_idx is not None:
            if event.xdata is None or event.ydata is None:
                return
            idx = self._dragging_circle_idx
            dx, dy = self._drag_offset
            self.circles[idx]["center"] = (event.xdata - dx, event.ydata - dy)
            self.redraw(self._img, self._title)
            return
        if not self._panning or event.x is None or event.y is None:
            return
        inv = self.ax.transData.inverted()
        x0d, y0d = inv.transform(self._pan_start_display)
        x1d, y1d = inv.transform((event.x, event.y))
        dx, dy = x1d - x0d, y1d - y0d
        x0, x1 = self._pan_start_xlim
        ybottom, ytop = self._pan_start_ylim
        self._set_view(x0 - dx, x1 - dx, ybottom - dy, ytop - dy)
        self.ax.figure.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        factor = 1 / _SCROLL_ZOOM if event.button == "up" else _SCROLL_ZOOM
        x0, x1 = self.ax.get_xlim()
        ybottom, ytop = self.ax.get_ylim()
        relx = (event.xdata - x0) / (x1 - x0)
        rely = (event.ydata - ytop) / (ybottom - ytop)
        new_w, new_h = (x1 - x0) * factor, (ybottom - ytop) * factor
        nx0, nx1 = event.xdata - new_w * relx, event.xdata + new_w * (1 - relx)
        nytop, nybottom = event.ydata - new_h * rely, event.ydata + new_h * (1 - rely)
        self._set_view(nx0, nx1, nybottom, nytop)
        self.ax.figure.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "enter":
            if self.mode != "polygon":
                pass
            elif len(self.current) >= 3:
                self.polygons.append(self.current)
                self.current = []
            else:
                print("  need >=3 points to close a polygon")
        elif event.key in (" ", "space"):
            self.mode = "circle" if self.mode == "polygon" else "polygon"
            if self.current:
                print("  switching mode: discarding in-progress polygon")
                self.current = []
            if self._circle_pending_center is not None:
                print("  switching mode: discarding pending circle center")
                self._circle_pending_center = None
        elif event.key in ("z", "backspace"):
            if self._circle_pending_center is not None:
                self._circle_pending_center = None
            elif self.current:
                self.current.pop()
            elif self.circles:
                self.circles.pop()
            elif self.polygons:
                self.polygons.pop()
        elif event.key == "r":
            self.current = []
            self.polygons = []
            self.circles = []
            self._circle_pending_center = None
        elif event.key in ("f", "home"):
            self.redraw(self._img, self._title, reset_view=True)
            return
        elif event.key == "g":
            self.show_guides = not self.show_guides
        elif event.key in ("n", "d"):
            if len(self.current) >= 3:
                self.polygons.append(self.current)
            elif self.current:
                print("  discarding unfinished polygon (<3 points)")
            self.current = []
            self._circle_pending_center = None
            self.done = True
        elif event.key == "q":
            if len(self.current) >= 3:
                self.polygons.append(self.current)
            self.current = []
            self._circle_pending_center = None
            self.done = True
            self.quit_all = True
        else:
            return
        self.redraw(self._img, self._title)

    def on_close(self, event):
        self.done = True
        self.quit_all = True

    def run(self, fig, img, title):
        self._img, self._title = img, title
        self.redraw(img, title, reset_view=True)

        cids = [
            fig.canvas.mpl_connect("button_press_event", self.on_press),
            fig.canvas.mpl_connect("button_release_event", self.on_release),
            fig.canvas.mpl_connect("motion_notify_event", self.on_motion),
            fig.canvas.mpl_connect("key_press_event", self.on_key),
            fig.canvas.mpl_connect("close_event", self.on_close),
            fig.canvas.mpl_connect("scroll_event", self.on_scroll),
        ]
        while not self.done:
            plt.pause(0.05)
        for cid in cids:
            fig.canvas.mpl_disconnect(cid)


def _edge_mask(gray_crop):
    edges = cv2.Canny(gray_crop, 60, 150)
    return cv2.dilate(edges, np.ones((3, 3), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-source", default="output/PatentData/hatch_pilot",
                     help="existing candidate pack, used only for the figure list "
                          "(ignored when --selected-file is given)")
    ap.add_argument("--tif-root", default=_DEFAULT_TIF_ROOT,
                     help="TIF root for --pack-source mode (ignored with --selected-file)")
    ap.add_argument("--selected-file", default=None,
                     help="JSON produced by tools/hatch_select.py -- label selected "
                          "figures as full images (no sub-drawing split). The TIF and "
                          "graph paths stored in the JSON are used directly.")
    ap.add_argument("--out", default="output/PatentData/hatch_gt")
    ap.add_argument("--redo", action="store_true", help="revisit ALL already-reviewed drawings")
    ap.add_argument("--only", nargs="+", default=None,
                     help="only (re)label these specific drawings, e.g. "
                          "--only EP123B1__F0002__d1 EP456A1__F0003 "
                          "(a bare patent__sketch redoes all its sub-drawings). "
                          "Everything else is skipped, even if unreviewed.")
    ap.add_argument("--only-file", default=None,
                     help="newline-separated file of stems (same format as --only), "
                          "e.g. the to_fix.txt written by tools/hatch_review.py's "
                          "flag key. Merged with --only if both are given.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    full_figure_mode = bool(a.selected_file)

    only_list = list(a.only or [])
    if a.only_file:
        only_list += [ln.strip() for ln in open(a.only_file) if ln.strip()]

    only_whole, only_stems = set(), set()
    for o in only_list:
        tail = o.rsplit("__", 1)[-1]
        if tail.startswith("d") and tail[1:].isdigit():
            only_stems.add(o)
        else:
            only_whole.add(o)

    # ------------------------------------------------------------------ figure list
    if full_figure_mode:
        selected = json.load(open(a.selected_file))
        # each entry: {patent, sketch, tif, graph}
        figures_full = [
            (e["patent"], e["sketch"], e["tif"], e.get("graph"))
            for e in selected
        ]
        if only_list:
            figures_full = [
                (p, s, t, g) for p, s, t, g in figures_full
                if f"{p}__{s}" in only_whole
                or any(st.startswith(f"{p}__{s}__d") for st in only_stems)
            ]
        if not figures_full:
            print("no figures to process"); return
    else:
        figures = _figure_list(a.pack_source)
        if not figures:
            print("no figures found via", a.pack_source); return
        if only_list:
            figures = [(p, s) for p, s in figures if f"{p}__{s}" in only_whole
                       or any(st.startswith(f"{p}__{s}__d") for st in only_stems)]
            if not figures:
                print("--only/--only-file matched no figures in", a.pack_source); return

    parts = _INSTRUCTIONS.split(" | ")
    print(__doc__.split("Mouse:")[0])
    print("Mouse: " + ", ".join(parts[:3]))
    print("Keys: " + " | ".join(parts[3:]) + "\n")

    fig_win, ax = plt.subplots(figsize=(9, 9))
    plt.ion()
    fig_win.show()

    remaining_flags = set(only_list) if a.only_file else None

    def _persist_flags():
        if remaining_flags is None:
            return
        with open(a.only_file, "w") as fh:
            fh.write("\n".join(sorted(remaining_flags)) + ("\n" if remaining_flags else ""))

    n_done, quit_all = 0, False

    # ------------------------------------------------------------------ full-figure loop
    if full_figure_mode:
        total = len(figures_full)
        for patent, sketch, tif_path, graph_path in figures_full:
            if quit_all:
                break
            stem = f"{patent}__{sketch}__d0"
            out_path = os.path.join(a.out, f"{stem}_mask.json")
            force = bool(only_list)
            if os.path.exists(out_path) and not a.redo and not force:
                if json.load(open(out_path)).get("reviewed"):
                    n_done += 1
                    continue

            gray = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"skip {patent}__{sketch}: tif not found at {tif_path}")
                continue
            H, W = gray.shape
            crop_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            edges = _edge_mask(gray)
            overlay = _hachure_overlay(crop_rgb, graph_path)
            has_overlay = overlay is not crop_rgb

            sess = _Session(ax, gray.shape, edges, img_overlay=(overlay if has_overlay else None))
            sess.show_guides = False   # overlay off by default (toggle with g)
            title = (f"[{n_done+1}/{total}] {patent}__{sketch}  "
                     f"(full figure  {W}x{H}px)"
                     + ("  |  g: toggle hachure overlay" if has_overlay else ""))
            sess.run(fig_win, crop_rgb, title)

            all_shapes = sess.polygons + [_circle_to_polygon(c["center"], c["radius"])
                                          for c in sess.circles]
            polys = [[[round(float(x), 1), round(float(y), 1)] for x, y in poly]
                     for poly in all_shapes]
            json.dump({"patent": patent, "sketch": sketch, "subdrawing_index": 0,
                       "crop_box": [0, 0, int(W), int(H)],
                       "reviewed": True, "polygons": polys},
                      open(out_path, "w"), indent=2)
            n_done += 1
            print(f"  saved {stem}: {len(polys)} hatch area(s) -> {out_path}")
            if remaining_flags is not None and stem in remaining_flags:
                remaining_flags.discard(stem)
                _persist_flags()
            if sess.quit_all:
                quit_all = True

    # ------------------------------------------------------------------ pack-source loop (original)
    else:
        for patent, sketch in figures:
            if quit_all:
                break
            tif = os.path.join(a.tif_root, patent, f"{patent}_{sketch}.tif")
            gray = cv2.imread(tif, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"skip {patent}__{sketch}: tif not found at {tif}")
                continue

            boxes = split_subdrawings(gray)
            for di, (x0, y0, x1, y1) in enumerate(boxes):
                stem = f"{patent}__{sketch}__d{di}"
                wanted = (not only_list or f"{patent}__{sketch}" in only_whole
                          or stem in only_stems)
                out_path = os.path.join(a.out, f"{stem}_mask.json")
                if not wanted:
                    continue
                force = bool(only_list)
                if os.path.exists(out_path) and not a.redo and not force:
                    if json.load(open(out_path)).get("reviewed"):
                        n_done += 1
                        continue

                crop_gray = gray[y0:y1, x0:x1]
                crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
                edges = _edge_mask(crop_gray)
                guides = _candidate_guides(a.pack_source, patent, sketch, (x0, y0, x1, y1))

                sess = _Session(ax, crop_gray.shape, edges, guides=guides)
                title = (f"[{n_done+1}] {patent}__{sketch}  "
                         f"(drawing {di+1}/{len(boxes)} on this sheet)")
                sess.run(fig_win, crop_rgb, title)

                all_shapes = sess.polygons + [_circle_to_polygon(c["center"], c["radius"])
                                              for c in sess.circles]
                polys = [[[round(x + x0, 1), round(y + y0, 1)] for x, y in poly]
                         for poly in all_shapes]
                json.dump({"patent": patent, "sketch": sketch, "subdrawing_index": di,
                           "crop_box": [int(x0), int(y0), int(x1), int(y1)],
                           "reviewed": True, "polygons": polys},
                          open(out_path, "w"), indent=2)
                n_done += 1
                print(f"  saved {stem}: {len(polys)} hatch area(s) -> {out_path}")
                if remaining_flags is not None and stem in remaining_flags:
                    remaining_flags.discard(stem)
                    _persist_flags()
                if sess.quit_all:
                    quit_all = True
                    break
            else:
                fig_key = f"{patent}__{sketch}"
                if remaining_flags is not None and fig_key in remaining_flags:
                    remaining_flags.discard(fig_key)
                    _persist_flags()

    plt.close(fig_win)
    print(f"\n{n_done} drawing(s) reviewed -> {a.out}")
    if remaining_flags is not None:
        if remaining_flags:
            print(f"{len(remaining_flags)} flagged drawing(s) still unresolved in {a.only_file} "
                  f"(re-run the same command to continue)")
        else:
            print(f"all flagged drawings resolved -> {a.only_file} is now empty")


if __name__ == "__main__":
    main()
