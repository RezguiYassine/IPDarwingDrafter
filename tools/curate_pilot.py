#!/usr/bin/env python3
"""Interactive accept/reject curation for a patent drawing cohort, with automatic
replacement of rejected figures.

Same working model as tools/hatch_label.py: one figure at a time in a reused
matplotlib window, a single keypress per decision, progress saved after every
answer so you can quit and resume. The difference is that rejecting a drawing
here does not just drop it -- the tool pulls a fresh candidate from the filtered
corpus and appends it to the queue, so the session ends with a FULL cohort of
accepted drawings rather than a shrunken one.

    ACCEPT  a real engineering drawing: mechanical views, cross-sections,
            assemblies, circuits-as-apparatus, optics, any figure whose content
            is geometry you would want vectorized.
    REJECT  content that is not an engineering drawing: chemical structures,
            flowcharts, block/network diagrams, tables, plots and charts, UI
            mock-ups, photographs, shaded renders, pages that are mostly text.

Replacements are sampled one-per-patent from the filtered corpus (label
'drawing' in the filter manifest), excluding every patent already seen in this
session, so the cohort keeps the stratification the original pilot had.

    python -m tools.curate_pilot --session output/PatentData/curation_v2
    python -m tools.curate_pilot --session output/PatentData/curation_v2 --no-display

Press the key WITH THE IMAGE WINDOW FOCUSED -- one keystroke, no Enter. The
prompt is echoed along the bottom of the window as well as in the terminal.
Under --no-display (or after you close the window) you type the letter in the
terminal and press Enter instead.

    y    accept
    n    reject  (then pick a reason; a replacement is queued automatically)
    s    skip    (decide later; does not count toward the target)
    b    back    (undo the previous decision and re-show it)
    q    save + quit   (Esc and Ctrl-C also save and exit)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# Patent TIFs carry a tag libtiff does not know, and cv2 logs a warning for
# every single read. Must be set before cv2 is first imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

REASONS = {
    "c": "chemistry",
    "f": "flowchart",
    "b": "block_diagram",
    "t": "table",
    "p": "plot_or_chart",
    "u": "ui_mockup",
    "s": "shaded_render_or_photo",
    "x": "text_page",
    "o": "other",
}

STATE = "session.json"
MANIFEST = "curated_manifest.csv"
DECISIONS = "decisions.csv"


# ── corpus access ────────────────────────────────────────────────────────────

def load_seed(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({"patent": r["patent"], "filename": r["filename"],
                     "path": r["path"], "origin": "pilot"})
    return rows


def load_prior_audit(path: Path) -> dict:
    """Optional: a previous hand audit, shown as context. Never auto-decides."""
    if not path or not Path(path).exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        key = (r.get("patent_id", ""), r.get("sketch_id", ""))
        out[key] = {"content_class": r.get("content_class", ""),
                    "bucket": r.get("bucket", ""),
                    "is_valid": r.get("is_valid_drawing", "")}
    return out


def build_pool(manifest: Path, exclude_patents: set[str], seed: int) -> list[dict]:
    """One 'drawing'-labelled figure per patent, shuffled, excluding seen patents."""
    by_patent: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(manifest)):
        if r.get("label") != "drawing":
            continue
        patent = r["patent"]
        if patent in exclude_patents:
            continue
        by_patent.setdefault(patent, []).append(r)
    rng = random.Random(seed)
    patents = sorted(by_patent)
    rng.shuffle(patents)
    pool = []
    for patent in patents:
        pick = sorted(by_patent[patent], key=lambda r: r["filename"])[0]
        pool.append({"patent": patent, "filename": pick["filename"],
                     "path": pick["path"], "origin": "replacement"})
    return pool


# ── session state ────────────────────────────────────────────────────────────

def load_state(session: Path) -> dict | None:
    path = session / STATE
    return json.loads(path.read_text()) if path.exists() else None


def save_state(session: Path, state: dict) -> None:
    session.mkdir(parents=True, exist_ok=True)
    tmp = session / (STATE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(session / STATE)


def key_of(item: dict) -> str:
    return f"{item['patent']}/{Path(item['filename']).stem}"


def sketch_id(item: dict) -> str:
    stem = Path(item["filename"]).stem
    prefix = item["patent"] + "_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


# ── display ──────────────────────────────────────────────────────────────────

class Viewer:
    """The figure window is the input device.

    Reading answers with input() would block the GUI event loop, so the window
    stops repainting and every key you press over the image goes nowhere -- it
    looks like a crash. Instead keys are captured from the canvas and the wait
    is a plt.pause() loop, which keeps the window live.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.fig = None
        self.pending: list[str] = []
        self.closed = False
        self.finishing = False
        if not enabled:
            return
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            # Matplotlib binds q/s/p/f/o/... to quit, save, pan, fullscreen.
            # Those collide with the decision and reason keys, so drop them all.
            for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
                plt.rcParams[key] = []
            self.fig, self.ax = plt.subplots(figsize=(9, 11))
            self.fig.canvas.manager.set_window_title("curate_pilot")
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            self.fig.canvas.mpl_connect("close_event", self._on_close)
            self.prompt_text = self.fig.text(0.5, 0.012, "", ha="center",
                                             fontsize=11, family="monospace")
            plt.ion()
            plt.show(block=False)
        except Exception as exc:                       # headless or no backend
            print(f"(display unavailable: {exc}; continuing without it)")
            self.enabled = False

    def _on_key(self, event) -> None:
        if event.key:
            self.pending.append(event.key)

    def _on_close(self, event) -> None:
        self.closed = True
        self.enabled = False
        if not self.finishing:
            print("\n(window closed; falling back to typed answers + Enter)")

    def finish(self) -> None:
        self.finishing = True
        if self.fig is not None:
            try:
                self.plt.close(self.fig)
            except Exception:
                pass

    def show(self, image_path: str, title: str) -> None:
        if not self.enabled:
            return
        try:
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"(could not read {image_path})")
                return
            self.ax.clear()
            self.ax.imshow(image, cmap="gray")
            self.ax.set_title(title, fontsize=10)
            self.ax.axis("off")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception as exc:
            print(f"(display error: {exc})")

    def ask(self, prompt: str) -> str:
        """One keypress in the figure window, no Enter. Typed input if headless."""
        if not self.enabled:
            return input(prompt).strip().lower()
        print(prompt, end="", flush=True)
        try:
            self.prompt_text.set_text(prompt.strip())
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        self.pending.clear()
        while not self.pending:
            if self.closed:                            # window went away mid-wait
                print()
                return input(prompt).strip().lower()
            try:
                self.plt.pause(0.05)
            except Exception:
                self.closed = True
        key = self.pending.pop(0)
        key = {"escape": "q", "enter": "", "backspace": "b"}.get(key, key)
        print(key)
        return key.strip().lower()


# ── reporting ────────────────────────────────────────────────────────────────

def write_outputs(session: Path, state: dict) -> tuple[int, int]:
    decided = state["decisions"]
    queue = state["queue"]
    by_key = {key_of(item): item for item in queue}
    accepted = [k for k, d in decided.items() if d["status"] == "accept"]
    accepted.sort()
    with open(session / MANIFEST, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patent", "sketch_id", "filename", "path", "origin"])
        for key in accepted:
            item = by_key[key]
            writer.writerow([item["patent"], sketch_id(item), item["filename"],
                             item["path"], item["origin"]])
    with open(session / DECISIONS, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patent", "sketch_id", "status", "reason", "origin", "prior_class"])
        for key, decision in sorted(decided.items()):
            item = by_key.get(key, {})
            writer.writerow([item.get("patent", ""), sketch_id(item) if item else "",
                             decision["status"], decision.get("reason", ""),
                             item.get("origin", ""), decision.get("prior_class", "")])
    return len(accepted), len(decided)


# ── main loop ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True,
                    help="session directory; resumes if it already exists")
    ap.add_argument("--seed-manifest", type=Path,
                    default=Path("benchmarks/pilotv3/frozen_sample_100.csv"),
                    help="cohort to review first")
    ap.add_argument("--filter-manifest", type=Path,
                    default=Path("output/PatentData/filter_manifest_v3.csv"),
                    help="replacement pool source; rows labelled 'drawing'")
    ap.add_argument("--prior-audit", type=Path,
                    default=Path("benchmarks/pilotv3/pilotv3_labeled_audit.csv"),
                    help="optional earlier audit, shown as context only")
    ap.add_argument("--target", type=int, default=100,
                    help="how many accepted drawings the cohort needs")
    ap.add_argument("--seed", type=int, default=170926)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    state = load_state(args.session)
    if state is None:
        if not args.seed_manifest.exists():
            print(f"missing seed manifest: {args.seed_manifest}")
            return 2
        queue = load_seed(args.seed_manifest)
        # Frozen at session creation so the shuffled pool -- and therefore
        # pool_cursor -- means the same thing on every resume. Excluding the
        # replacements drawn so far instead would reshuffle a shorter list and
        # silently skip candidates.
        state = {"queue": queue, "decisions": {}, "cursor": 0,
                 "pool_cursor": 0, "target": args.target, "seed": args.seed,
                 "pool_exclude": sorted({item["patent"] for item in queue})}
        print(f"new session: {len(queue)} seed drawings, target {args.target} accepted")
    else:
        print(f"resuming: {len(state['decisions'])} decided, "
              f"{sum(1 for d in state['decisions'].values() if d['status']=='accept')} accepted")

    prior = load_prior_audit(args.prior_audit)
    exclude = set(state.get("pool_exclude") or
                  {item["patent"] for item in state["queue"]})
    pool = build_pool(args.filter_manifest, exclude, state["seed"]) \
        if args.filter_manifest.exists() else []
    if not pool:
        print("warning: no replacement pool available; rejections will not be replaced")

    viewer = Viewer(not args.no_display)
    target = state["target"]

    while True:
        accepted = sum(1 for d in state["decisions"].values() if d["status"] == "accept")
        if accepted >= target:
            print(f"\ntarget reached: {accepted} accepted")
            break
        if state["cursor"] >= len(state["queue"]):
            print(f"\nqueue exhausted with {accepted}/{target} accepted")
            break

        item = state["queue"][state["cursor"]]
        key = key_of(item)
        if key in state["decisions"] and state["decisions"][key]["status"] != "skip":
            state["cursor"] += 1
            continue

        hint = prior.get((item["patent"], sketch_id(item)), {})
        hint_text = ""
        if hint:
            valid = {"1": "valid", "0": "INVALID"}.get(hint.get("is_valid", ""), "?")
            hint_text = f"  |  prior audit: {hint.get('content_class') or '?'} ({valid})"
        title = (f"[{accepted}/{target} accepted]  {key}  ({item['origin']}){hint_text}")
        print(f"\n{title}")
        if not os.path.exists(item["path"]):
            print(f"  missing file, auto-skipping: {item['path']}")
            state["decisions"][key] = {"status": "skip", "reason": "missing_file"}
            state["cursor"] += 1
            save_state(args.session, state)
            continue
        viewer.show(item["path"], title)

        try:
            answer = viewer.ask("  [y]accept [n]reject [s]kip [b]ack [q]uit > ")
        except (KeyboardInterrupt, EOFError):
            print("\ninterrupted")
            break
        if answer == "q":
            break
        if answer == "b":
            back = state["cursor"] - 1
            while back >= 0 and key_of(state["queue"][back]) not in state["decisions"]:
                back -= 1
            if back < 0:
                print("  nothing to go back to")
                continue
            removed = state["decisions"].pop(key_of(state["queue"][back]))
            # Undoing a rejection must also withdraw the replacement it queued,
            # or the cohort accumulates spare candidates and the provenance of
            # "why is this drawing here" stops being true.
            spare = removed.get("replacement")
            if spare:
                for index in range(len(state["queue"]) - 1, back, -1):
                    if key_of(state["queue"][index]) == spare \
                            and spare not in state["decisions"]:
                        state["queue"].pop(index)
                        state["pool_cursor"] = max(0, state["pool_cursor"] - 1)
                        print(f"  withdrew replacement: {spare}")
                        break
            print(f"  undid: {removed['status']}")
            state["cursor"] = back
            save_state(args.session, state)
            continue
        if answer == "s":
            state["decisions"][key] = {"status": "skip", "reason": "",
                                       "prior_class": hint.get("content_class", "")}
        elif answer == "y":
            state["decisions"][key] = {"status": "accept", "reason": "",
                                       "prior_class": hint.get("content_class", "")}
        elif answer == "n":
            menu = " ".join(f"[{k}]{v}" for k, v in REASONS.items())
            try:
                choice = viewer.ask(f"    reason {menu} > ")
            except (KeyboardInterrupt, EOFError):
                print("\ninterrupted")
                break
            reason = REASONS.get(choice, choice or "other")
            decision = {"status": "reject", "reason": reason,
                        "prior_class": hint.get("content_class", "")}
            queued = {item["patent"] for item in state["queue"]}
            while state["pool_cursor"] < len(pool) \
                    and pool[state["pool_cursor"]]["patent"] in queued:
                state["pool_cursor"] += 1
            if state["pool_cursor"] < len(pool):
                replacement = pool[state["pool_cursor"]]
                state["pool_cursor"] += 1
                state["queue"].append(replacement)
                decision["replacement"] = key_of(replacement)
                print(f"    queued replacement: {key_of(replacement)}")
            else:
                print("    no replacement available (pool exhausted)")
            state["decisions"][key] = decision
        else:
            print("  unrecognised; use y / n / s / b / q")
            continue

        state["cursor"] += 1
        save_state(args.session, state)

    viewer.finish()
    save_state(args.session, state)
    accepted, decided = write_outputs(args.session, state)
    print(f"\nsaved {decided} decisions, {accepted} accepted")
    print(f"  {args.session / MANIFEST}")
    print(f"  {args.session / DECISIONS}")
    if accepted < target:
        print(f"  {target - accepted} more needed; rerun to continue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
