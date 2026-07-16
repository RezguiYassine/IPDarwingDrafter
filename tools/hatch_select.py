#!/usr/bin/env python3
"""Browse candidate figures and classify as positive, negative, or skip.

Three-way classification:
    p        POSITIVE — has visible hatching; will go to hatch_mask_label for
             manual polygon drawing
    n        NEGATIVE — definitely no hatching (or hard-negative with confusing
             strokes); an empty mask is auto-saved without any manual drawing
    s / →    SKIP — not useful for the dataset, don't include in either set
    ←        go back and change a previous decision
    h        toggle removed_hachures edge overlay (orange — helps spot hatching)
    f        fit to view
    scroll   zoom in / out
    drag     pan while zoomed
    q        quit (all decisions are saved after every key press)

Outputs
-------
    hatch_selected.json            positives → feed to hatch_mask_label
    hatch_selected_negatives.json  negatives → feed to hatch_finalize_negatives
    hatch_selected_decisions.json  full record (pos/neg/skip) for resume

After this tool:

    # 1 — draw polygons on the positives:
    python -m tools.hatch_mask_label \\
        --selected-file output/PatentData/hatch_selected.json \\
        --out output/PatentData/hatch_gt

    # 2 — auto-save empty masks for the negatives (no manual drawing needed):
    python -m tools.hatch_finalize_negatives \\
        --negatives output/PatentData/hatch_selected_negatives.json \\
        --out output/PatentData/hatch_gt

Usage:
    python -m tools.hatch_select \\
        --candidate-list output/PatentData/hatch_candidates_processed.json
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

for _k in ("keymap.save", "keymap.quit", "keymap.home", "keymap.fullscreen",
           "keymap.grid", "keymap.grid_minor", "keymap.back", "keymap.forward",
           "keymap.pan", "keymap.zoom", "keymap.xscale", "keymap.yscale"):
    mpl.rcParams[_k] = []

_SCROLL_ZOOM = 1.2
_POS  = "pos"
_NEG  = "neg"
_SKIP = "skip"


# ─── Figure discovery ────────────────────────────────────────────────────────

def _discover_figures(tif_source, graph_source, exclude_sets):
    excluded: set[tuple[str, str]] = set()
    for directory in exclude_sets:
        for f in glob.glob(os.path.join(directory, "*_regions.json")):
            d = json.load(open(f)); excluded.add((d["patent"], d["sketch"]))
        for f in glob.glob(os.path.join(directory, "*_mask.json")):
            d = json.load(open(f)); excluded.add((d["patent"], d["sketch"]))
    figures = []
    for tif_path in sorted(glob.glob(os.path.join(tif_source, "*", "*_original.tif"))):
        parts = tif_path.replace("\\", "/").split("/")
        patent, sketch = parts[-2], parts[-1].replace("_original.tif", "")
        if (patent, sketch) in excluded:
            continue
        graph_path = os.path.join(graph_source, patent, "graphs", f"{sketch}_graph.json")
        figures.append({"patent": patent, "sketch": sketch, "tif": tif_path,
                         "graph": graph_path if os.path.exists(graph_path) else None})
    return figures


# ─── Hachure overlay ─────────────────────────────────────────────────────────

def _build_hachure_overlay(tif_rgb: np.ndarray, graph_path: str | None) -> np.ndarray:
    if not graph_path:
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
    overlay[where] = (overlay[where] * 0.35 + np.array([255, 120, 0]) * 0.65
                      ).clip(0, 255).astype(np.uint8)
    return overlay


# ─── Selector ────────────────────────────────────────────────────────────────

class _Selector:
    def __init__(self, ax, figures, decisions: dict[str, str],
                 out_pos: str, out_neg: str, decisions_path: str):
        self.ax             = ax
        self.figures        = figures
        self.decisions      = decisions   # stem -> "pos" | "neg" | "skip"
        self.out_pos        = out_pos
        self.out_neg        = out_neg
        self.decisions_path = decisions_path
        self.i              = 0
        self.show_overlay   = False
        self.quit           = False
        self._img           = None
        self._overlay_img   = None
        self._has_hachures  = False
        self._panning       = False
        self._pan_start_display = None
        self._pan_start_xlim    = None
        self._pan_start_ylim    = None

    def _stem(self):
        f = self.figures[self.i]
        return f"{f['patent']}__{f['sketch']}"

    # ── View helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_range(lo, hi, full):
        span = hi - lo
        if span >= full: return 0.0, float(full)
        if lo < 0:  hi -= lo; lo = 0.0
        if hi > full: lo -= (hi - full); hi = float(full)
        return max(0.0, lo), min(float(full), hi)

    def _set_view(self, x0, x1, ybottom, ytop):
        H, W = self._img.shape[:2]
        x0, x1 = self._clamp_range(x0, x1, W)
        ytop, ybottom = self._clamp_range(ytop, ybottom, H)
        self.ax.set_xlim(x0, x1); self.ax.set_ylim(ybottom, ytop)

    # ── Image loading ────────────────────────────────────────────────────────

    def _load(self, i):
        f = self.figures[i]
        gray = cv2.imread(f["tif"], cv2.IMREAD_GRAYSCALE)
        if gray is None:
            self._img = np.full((200, 300, 3), 180, np.uint8)
            self._overlay_img = self._img
            self._has_hachures = False
            return
        self._img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        self._overlay_img = _build_hachure_overlay(self._img, f["graph"])
        self._has_hachures = (self._overlay_img is not self._img)

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.out_pos)), exist_ok=True)
        # Current session's decisions
        cur_stems = {f"{f['patent']}__{f['sketch']}" for f in self.figures}
        new_pos = [f for f in self.figures
                   if self.decisions.get(f"{f['patent']}__{f['sketch']}") == _POS]
        new_neg = [f for f in self.figures
                   if self.decisions.get(f"{f['patent']}__{f['sketch']}") == _NEG]
        # Preserve entries from previous sessions not in the current candidate list
        prev_pos = json.load(open(self.out_pos)) if os.path.exists(self.out_pos) else []
        prev_neg = json.load(open(self.out_neg)) if os.path.exists(self.out_neg) else []
        kept_pos = [f for f in prev_pos if f"{f['patent']}__{f['sketch']}" not in cur_stems]
        kept_neg = [f for f in prev_neg if f"{f['patent']}__{f['sketch']}" not in cur_stems]
        json.dump(kept_pos + new_pos, open(self.out_pos, "w"), indent=2)
        json.dump(kept_neg + new_neg, open(self.out_neg, "w"), indent=2)
        json.dump(self.decisions, open(self.decisions_path, "w"), indent=2)

    # ── Display ──────────────────────────────────────────────────────────────

    def redraw(self, reset_view=False):
        if not reset_view:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        self.ax.imshow(self._overlay_img if self.show_overlay else self._img)
        if reset_view:
            H, W = self._img.shape[:2]
            self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0)
        else:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        self.ax.axis("off")

        stem     = self._stem()
        decision = self.decisions.get(stem)
        n_pos  = sum(1 for v in self.decisions.values() if v == _POS)
        n_neg  = sum(1 for v in self.decisions.values() if v == _NEG)
        n_skip = sum(1 for v in self.decisions.values() if v == _SKIP)
        n_rem  = len(self.figures) - len(self.decisions)

        if decision == _POS:
            dec_tag, col = "  [POSITIVE]", "#1a7a1a"
        elif decision == _NEG:
            dec_tag, col = "  [NEGATIVE]", "#B03A2E"
        elif decision == _SKIP:
            dec_tag, col = "  [skipped]", "#888888"
        else:
            dec_tag, col = "", "black"

        h_tag = (f"h:{'ON' if self.show_overlay else 'off'}  "
                 f"({'hachures detected' if self._has_hachures else 'no hachures'})")

        self.ax.set_title(
            f"[{self.i+1}/{len(self.figures)}] {stem}{dec_tag}\n"
            f"positives:{n_pos}  negatives:{n_neg}  skipped:{n_skip}  remaining:{n_rem}  "
            f"|  {h_tag}\n"
            f"p=positive  n=negative  s/→=skip  ←=back  h=overlay  f=fit  "
            f"scroll=zoom  drag=pan  q=quit",
            fontsize=8, color=col)
        self.ax.figure.canvas.draw_idle()

    def goto(self, i):
        self.i = max(0, min(len(self.figures) - 1, i))
        self._load(self.i)
        self.redraw(reset_view=True)

    # ── Actions ──────────────────────────────────────────────────────────────

    def mark_positive(self):
        self.decisions[self._stem()] = _POS
        self._save()
        print(f"  [POSITIVE] {self._stem()}")
        self.goto(self.i + 1) if self.i < len(self.figures) - 1 else self.redraw()

    def mark_negative(self):
        self.decisions[self._stem()] = _NEG
        self._save()
        print(f"  [NEGATIVE] {self._stem()}")
        self.goto(self.i + 1) if self.i < len(self.figures) - 1 else self.redraw()

    def skip(self):
        self.decisions[self._stem()] = _SKIP
        self._save()
        self.goto(self.i + 1) if self.i < len(self.figures) - 1 else self.redraw()

    # ── Event handlers ───────────────────────────────────────────────────────

    def on_key(self, event):
        if event.key == "p":
            self.mark_positive()
        elif event.key == "n":
            self.mark_negative()
        elif event.key in ("s", "right"):
            self.skip()
        elif event.key == "left":
            self.goto(self.i - 1)
        elif event.key == "h":
            self.show_overlay = not self.show_overlay
            self.redraw()
        elif event.key in ("f", "home"):
            self.redraw(reset_view=True)
        elif event.key == "q":
            self.quit = True

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1: return
        self._panning = True
        self._pan_start_display = (event.x, event.y)
        self._pan_start_xlim    = self.ax.get_xlim()
        self._pan_start_ylim    = self.ax.get_ylim()

    def on_release(self, event):
        if event.button == 1: self._panning = False

    def on_motion(self, event):
        if not self._panning or event.x is None: return
        inv = self.ax.transData.inverted()
        x0d, y0d = inv.transform(self._pan_start_display)
        x1d, y1d = inv.transform((event.x, event.y))
        dx, dy = x1d - x0d, y1d - y0d
        x0, x1 = self._pan_start_xlim
        ybottom, ytop = self._pan_start_ylim
        self._set_view(x0 - dx, x1 - dx, ybottom - dy, ytop - dy)
        self.ax.figure.canvas.draw_idle()

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        factor = 1 / _SCROLL_ZOOM if event.button == "up" else _SCROLL_ZOOM
        x0, x1 = self.ax.get_xlim()
        ybottom, ytop = self.ax.get_ylim()
        relx = (event.xdata - x0) / (x1 - x0)
        rely = (event.ydata - ytop) / (ybottom - ytop)
        new_w, new_h = (x1 - x0) * factor, (ybottom - ytop) * factor
        self._set_view(event.xdata - new_w * relx, event.xdata + new_w * (1 - relx),
                       event.ydata + new_h * (1 - rely), event.ydata - new_h * rely)
        self.ax.figure.canvas.draw_idle()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-list", default=None,
                     help="JSON from hatch_sample_candidates.py or "
                          "hatch_candidates_processed.json")
    ap.add_argument("--tif-source",   default="latest_run")
    ap.add_argument("--graph-source", default="latest_run_puhachovft")
    ap.add_argument("--exclude-pilot", default="output/PatentData/hatch_pilot")
    ap.add_argument("--exclude-gt",    default="output/PatentData/hatch_gt")
    ap.add_argument("--out", default="output/PatentData/hatch_selected.json",
                     help="output path for POSITIVES list (fed to hatch_mask_label)")
    a = ap.parse_args()

    out_pos        = a.out
    out_neg        = a.out.replace(".json", "_negatives.json")
    decisions_path = a.out.replace(".json", "_decisions.json")
    exclude_sets   = [a.exclude_pilot, a.exclude_gt]

    # ── Load previous decisions (backward compat: True→pos, False→skip) ─────
    decisions: dict[str, str] = {}
    if os.path.exists(decisions_path):
        raw = json.load(open(decisions_path))
        for k, v in raw.items():
            if v is True or v == _POS:      decisions[k] = _POS
            elif v is False or v == _SKIP:  decisions[k] = _SKIP
            elif v == _NEG:                 decisions[k] = _NEG
        n_pos  = sum(1 for v in decisions.values() if v == _POS)
        n_neg  = sum(1 for v in decisions.values() if v == _NEG)
        n_skip = sum(1 for v in decisions.values() if v == _SKIP)
        print(f"resuming: {n_pos} positive, {n_neg} negative, {n_skip} skipped")
    elif os.path.exists(out_pos):
        for f in json.load(open(out_pos)):
            decisions[f"{f['patent']}__{f['sketch']}"] = _POS
        print(f"resuming: {len(decisions)} previously accepted → treated as positive")

    # ── Build candidate list ─────────────────────────────────────────────────
    if a.candidate_list:
        raw_list = json.load(open(a.candidate_list))
        excluded: set[tuple[str, str]] = set()
        for directory in exclude_sets:
            for f in glob.glob(os.path.join(directory, "*_regions.json")):
                d = json.load(open(f)); excluded.add((d["patent"], d["sketch"]))
            for f in glob.glob(os.path.join(directory, "*_mask.json")):
                d = json.load(open(f)); excluded.add((d["patent"], d["sketch"]))
        figures = [f for f in raw_list
                   if (f["patent"], f["sketch"]) not in excluded
                   and f"{f['patent']}__{f['sketch']}" not in decisions]
        n_total = len(raw_list)
        print(f"candidate list: {n_total} total, "
              f"{n_total - len(figures)} already decided/excluded, "
              f"{len(figures)} unseen")
    else:
        figures = _discover_figures(a.tif_source, a.graph_source, exclude_sets)
        figures = [f for f in figures if f"{f['patent']}__{f['sketch']}" not in decisions]
        print(f"{len(figures)} unseen candidate figures")

    if not figures:
        n_pos = sum(1 for v in decisions.values() if v == _POS)
        n_neg = sum(1 for v in decisions.values() if v == _NEG)
        print(f"all candidates reviewed — {n_pos} positive, {n_neg} negative")
        _print_next_steps(out_pos, out_neg)
        return

    print("Keys: p=positive  n=negative  s/→=skip  ←=back  "
          "h=overlay  f=fit  scroll=zoom  drag=pan  q=quit\n")

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion(); fig.show()

    selector = _Selector(ax, figures, decisions, out_pos, out_neg, decisions_path)
    selector.goto(0)

    cids = [
        fig.canvas.mpl_connect("key_press_event",    selector.on_key),
        fig.canvas.mpl_connect("button_press_event", selector.on_press),
        fig.canvas.mpl_connect("button_release_event", selector.on_release),
        fig.canvas.mpl_connect("motion_notify_event", selector.on_motion),
        fig.canvas.mpl_connect("scroll_event",       selector.on_scroll),
        fig.canvas.mpl_connect("close_event", lambda e: setattr(selector, "quit", True)),
    ]
    while not selector.quit:
        plt.pause(0.05)
    for cid in cids:
        fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    n_pos  = sum(1 for v in selector.decisions.values() if v == _POS)
    n_neg  = sum(1 for v in selector.decisions.values() if v == _NEG)
    n_skip = sum(1 for v in selector.decisions.values() if v == _SKIP)
    print(f"\nSession done — positives:{n_pos}  negatives:{n_neg}  skipped:{n_skip}")
    _print_next_steps(out_pos, out_neg)


def _print_next_steps(out_pos: str, out_neg: str) -> None:
    print(f"\nNext steps:")
    print(f"  # 1 — draw polygons on positives:")
    print(f"  python -m tools.hatch_mask_label \\")
    print(f"      --selected-file {out_pos} --out output/PatentData/hatch_gt")
    print(f"\n  # 2 — auto-save empty masks for negatives:")
    print(f"  python -m tools.hatch_finalize_negatives \\")
    print(f"      --negatives {out_neg} --out output/PatentData/hatch_gt")


if __name__ == "__main__":
    main()
