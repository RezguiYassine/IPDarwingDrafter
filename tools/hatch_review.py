#!/usr/bin/env python3
"""Review the ground-truth hatch masks (output of tools/hatch_mask_label.py).

Read-only viewer: steps through every labeled drawing, showing the same crop
that was labeled, with two overlays --
    cyan dashed   the detector's candidate regions (from hatch_pilot)
    green solid   what the student actually drew as hatch (the ground truth)
-- so you can quickly spot anything the student over-marked, under-marked, or
got wrong relative to the detector's proposals.

    python -m tools.hatch_review --gt output/PatentData/hatch_gt \
        --pack-source output/PatentData/hatch_pilot

Keys:
    right / n / space    next drawing
    left / p / backspace previous drawing
    scroll wheel         zoom in/out, centered on the cursor (clamped)
    left-drag            pan while zoomed in
    f / home             fit-to-view
    c                    toggle the candidate-region overlay (cyan)
    l                    toggle the labeled ground-truth overlay (green)
    b                    flag/unflag this drawing as needing a re-label (saved
                          immediately to <gt>/to_fix.txt, survives across runs)
    q                    quit

After a pass, fix everything you flagged in one go:

    python -m tools.hatch_mask_label --out output/PatentData/hatch_gt \
        --only-file output/PatentData/hatch_gt/to_fix.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

from tools.hatch_mask_label import _candidate_guides

for _k in ("keymap.save", "keymap.quit", "keymap.home", "keymap.fullscreen",
           "keymap.grid", "keymap.grid_minor", "keymap.back", "keymap.forward",
           "keymap.pan", "keymap.zoom", "keymap.xscale", "keymap.yscale"):
    mpl.rcParams[_k] = []

_DEFAULT_TIF_ROOT = "data/PatentData/ReorganisedData"
_SCROLL_ZOOM = 1.2


def _stem(rec):
    return f"{rec['patent']}__{rec['sketch']}__d{rec['subdrawing_index']}"


class _Viewer:
    def __init__(self, ax, records, flagged=None, flag_path=None):
        self.ax = ax
        self.records = records
        self.i = 0
        self.show_guides = True
        self.show_labels = True
        self.flagged = flagged if flagged is not None else set()
        self.flag_path = flag_path
        self.quit = False
        self._panning = False
        self._pan_start_display = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

    def _save_flags(self):
        if not self.flag_path:
            return
        with open(self.flag_path, "w") as f:
            f.write("\n".join(sorted(self.flagged)) + ("\n" if self.flagged else ""))

    def toggle_flag(self):
        stem = _stem(self.records[self.i])
        if stem in self.flagged:
            self.flagged.discard(stem)
            print(f"  unflagged {stem}")
        else:
            self.flagged.add(stem)
            print(f"  flagged {stem} for re-label")
        self._save_flags()

    @staticmethod
    def _clamp_range(lo, hi, full):
        span = hi - lo
        if span >= full:
            return 0.0, float(full)
        if lo < 0:
            hi -= lo; lo = 0.0
        if hi > full:
            lo -= (hi - full); hi = float(full)
        return max(0.0, lo), min(float(full), hi)

    def _set_view(self, x0, x1, ybottom, ytop):
        H, W = self.img.shape[:2]
        x0, x1 = self._clamp_range(x0, x1, W)
        ytop, ybottom = self._clamp_range(ytop, ybottom, H)
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(ybottom, ytop)

    def _load(self, i):
        rec = self.records[i]
        gray = cv2.imread(rec["tif"], cv2.IMREAD_GRAYSCALE)
        x0, y0, x1, y1 = rec["crop_box"]
        crop = gray[y0:y1, x0:x1]
        self.img = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        # labels are saved in full-TIF coords (offset by the crop box) -- bring
        # them back to crop-local coords to match the displayed crop.
        self.labels = [[[p[0] - x0, p[1] - y0] for p in poly] for poly in rec["polygons"]]
        self.guides = _candidate_guides(rec["pack_source"], rec["patent"], rec["sketch"],
                                         (x0, y0, x1, y1))

    def redraw(self, reset_view=False):
        if not reset_view:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self.img)
        if self.show_guides:
            for poly in self.guides:
                xs = [p[0] for p in poly] + [poly[0][0]]
                ys = [p[1] for p in poly] + [poly[0][1]]
                self.ax.plot(xs, ys, "--", color="cyan", linewidth=1.5, alpha=0.7)
        if self.show_labels:
            for poly in self.labels:
                xs = [p[0] for p in poly] + [poly[0][0]]
                ys = [p[1] for p in poly] + [poly[0][1]]
                self.ax.plot(xs, ys, "-", color="lime", linewidth=2)
        if reset_view:
            H, W = self.img.shape[:2]
            self._set_view(0, W, H, 0)
        else:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.ax.axis("off")
        rec = self.records[self.i]
        stem = _stem(rec)
        flag_tag = "  [FLAGGED for re-label]" if stem in self.flagged else ""
        title_color = "#B03A2E" if stem in self.flagged else "black"
        self.ax.set_title(
            f"[{self.i+1}/{len(self.records)}] {stem}{flag_tag}\n"
            f"cyan: {len(self.guides)} candidate region(s)  |  "
            f"green: {len(self.labels)} labeled hatch area(s)  |  "
            f"{len(self.flagged)} flagged so far\n"
            f"(c: toggle cyan, l: toggle green, b: flag, scroll: zoom, drag: pan, f: fit, "
            f"←/→: prev/next, q: quit)", fontsize=9, color=title_color)
        self.ax.figure.canvas.draw_idle()

    def goto(self, i):
        self.i = max(0, min(len(self.records) - 1, i))
        self._load(self.i)
        self.redraw(reset_view=True)

    def on_key(self, event):
        if event.key in ("right", "n", " ", "space"):
            self.goto(self.i + 1)
        elif event.key in ("left", "p", "backspace"):
            self.goto(self.i - 1)
        elif event.key in ("f", "home"):
            self.redraw(reset_view=True)
        elif event.key == "c":
            self.show_guides = not self.show_guides
            self.redraw()
        elif event.key == "l":
            self.show_labels = not self.show_labels
            self.redraw()
        elif event.key == "b":
            self.toggle_flag()
            self.redraw()
        elif event.key == "q":
            self.quit = True

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._panning = True
        self._pan_start_display = (event.x, event.y)
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()

    def on_release(self, event):
        if event.button == 1:
            self._panning = False

    def on_motion(self, event):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="output/PatentData/hatch_gt")
    ap.add_argument("--pack-source", default="output/PatentData/hatch_pilot")
    ap.add_argument("--tif-root", default=_DEFAULT_TIF_ROOT)
    ap.add_argument("--flag-file", default=None,
                     help="default: <gt>/to_fix.txt")
    a = ap.parse_args()
    flag_path = a.flag_file or os.path.join(a.gt, "to_fix.txt")
    flagged = set()
    if os.path.exists(flag_path):
        flagged = {ln.strip() for ln in open(flag_path) if ln.strip()}

    records = []
    n_skipped = 0
    for path in sorted(glob.glob(os.path.join(a.gt, "*_mask.json"))):
        d = json.load(open(path))
        if "crop_box" not in d:
            n_skipped += 1
            continue
        tif = os.path.join(a.tif_root, d["patent"], f"{d['patent']}_{d['sketch']}.tif")
        records.append({**d, "tif": tif, "pack_source": a.pack_source})
    if n_skipped:
        print(f"skipped {n_skipped} legacy file(s) without crop_box (pre-split format)")
    if not records:
        print("no *_mask.json in", a.gt); return

    print(__doc__)
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.ion()
    fig.show()
    viewer = _Viewer(ax, records, flagged=flagged, flag_path=flag_path)
    viewer.goto(0)

    cids = [
        fig.canvas.mpl_connect("key_press_event", viewer.on_key),
        fig.canvas.mpl_connect("button_press_event", viewer.on_press),
        fig.canvas.mpl_connect("button_release_event", viewer.on_release),
        fig.canvas.mpl_connect("motion_notify_event", viewer.on_motion),
        fig.canvas.mpl_connect("scroll_event", viewer.on_scroll),
        fig.canvas.mpl_connect("close_event", lambda e: setattr(viewer, "quit", True)),
    ]
    while not viewer.quit:
        plt.pause(0.05)
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    if viewer.flagged:
        print(f"\n{len(viewer.flagged)} drawing(s) flagged -> {flag_path}")
        print("Fix them with:\n"
              f"  python -m tools.hatch_mask_label --out output/PatentData/hatch_gt "
              f"--only-file {flag_path}")
    else:
        print("\nno drawings flagged")


if __name__ == "__main__":
    main()
