#!/usr/bin/env python3
"""Interactive labeller for the hatch-region pilot pack.

For each figure it prints the overlay path (open it in your editor) and the
candidate regions; you type the IDs that are TRUE section hatching. Everything
else on that figure is marked not-hatch. Progress is saved after every figure,
so you can quit ('q') and resume any time.

    python -m tools.hatch_label --pack output/PatentData/hatch_pilot

Commands at the prompt:
    "0 2 3"  region ids that ARE hatching        (rest on this figure = not-hatch)
    a        all regions are hatch
    n        none are hatch
    s        skip (leave unlabelled)
    b        go back one figure
    q        save + quit
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _summarize(region):
    f = region.get("features", {})
    tag = "cross" if region.get("double") else "single"
    return (f"  [{region['id']}] {tag} angles={region['angles']} "
            f"lines={region['n_lines']} spacing={region['spacing']:.1f} "
            f"fill={f.get('fill_uniformity', 0):.2f} encl={f.get('boundary_on_main', 0):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--redo", action="store_true", help="re-label already-labelled figures")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.pack, "*_regions.json")))
    if not files:
        print("no *_regions.json in", a.pack); return
    i = 0
    while 0 <= i < len(files):
        path = files[i]
        doc = json.load(open(path))
        regs = doc["regions"]
        done = all(r["label"] is not None for r in regs)
        if done and not a.redo:
            i += 1; continue
        stem = os.path.basename(path).replace("_regions.json", "")
        overlay = os.path.join(a.pack, f"{stem}_overlay.png")
        n_lab = sum(1 for f in files if all(r["label"] is not None
                                            for r in json.load(open(f))["regions"]))
        print("\n" + "=" * 70)
        print(f"[{i+1}/{len(files)}]  {stem}   (labelled so far: {n_lab})")
        print(f"  overlay: {overlay}")
        for r in regs:
            print(_summarize(r))
        ans = input("hatch ids (space-sep) / a / n / s / b / q : ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            i = max(0, i - 1); continue
        if ans == "s":
            i += 1; continue
        if ans == "a":
            hatch = {r["id"] for r in regs}
        elif ans == "n":
            hatch = set()
        else:
            try:
                hatch = {int(t) for t in ans.split()}
            except ValueError:
                print("  ? not understood, skipping"); continue
        for r in regs:
            r["label"] = bool(r["id"] in hatch)
        json.dump(doc, open(path, "w"), indent=2)
        i += 1
    # final tally
    pos = neg = unl = 0
    for f in files:
        for r in json.load(open(f))["regions"]:
            if r["label"] is True: pos += 1
            elif r["label"] is False: neg += 1
            else: unl += 1
    print(f"\nlabelled: {pos} hatch, {neg} not-hatch, {unl} unlabelled. "
          f"Train de-risk AUC with:  python -m tools.hatch_train --pack {a.pack}")


if __name__ == "__main__":
    main()
