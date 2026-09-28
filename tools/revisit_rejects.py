#!/usr/bin/env python3
"""Re-examine decisions already made, and keep or overturn each one.

A curation session records a reason with every rejection, and a reason can
turn out to cover two different things. The mechanical session rejected 475
figures as `shaded_render_or_photo`; a sample of those showed clean
perspective and isometric line art -- motorcycles, aircraft, exploded
assemblies -- at an ink fraction indistinguishable from the accepted
drawings. Whether such a view belongs in the corpus is a policy call, and
this tool is how that call gets applied to figures already judged rather
than by re-labelling from scratch.

It never edits the original session. Overturned decisions are written to a
separate overlay, so the first pass stays on the record and can be replayed.

    python -m tools.revisit_rejects --session output/PatentData/mechanical_labels \
        --reason shaded_render_or_photo

Keys, pressed with the image window focused (one keystroke, no Enter):

    k    keep the rejection      (it really is a photo or render)
    a    accept after all        (it is a drawing; overturns the rejection)
    s    skip, decide later
    b    back, undo the previous answer
    q    save and quit

Progress is saved after every answer, so quitting and rerunning resumes. On
exit it writes an overlay and, when anything was overturned, an updated
content manifest ready for a pipeline run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from tools.curate_pilot import Viewer, key_of, save_state, sketch_id   # noqa: E402

STATE = "revisit_state.json"
OVERLAY = "revisit_overlay.json"
REPORT = "revisit_report.csv"


def load_targets(session: Path, reason: str, status: str) -> tuple[list[dict], dict]:
    """The decisions to revisit, each paired with the figure it was made about."""
    state = json.loads((session / "session.json").read_text())
    queue = {key_of(item): item for item in state["queue"]}
    targets = []
    for key, decision in sorted(state["decisions"].items()):
        if decision.get("status") != status:
            continue
        if reason and decision.get("reason") != reason:
            continue
        item = queue.get(key)
        if item is None or not Path(item["path"]).is_file():
            continue
        targets.append({**item, "key": key, "original_reason": decision.get("reason", "")})
    return targets, state


def write_outputs(session: Path, out_dir: Path, state: dict, answers: dict) -> tuple[int, int]:
    overturned = [k for k, v in answers.items() if v == "accept"]
    kept = [k for k, v in answers.items() if v == "keep"]
    (out_dir / OVERLAY).write_text(json.dumps({
        "schema": "ap3-revisit-overlay-v1",
        "session": str(session.resolve()),
        "reviewed": len(answers), "overturned_to_accept": sorted(overturned),
        "rejection_confirmed": sorted(kept),
    }, indent=2) + "\n")
    with open(out_dir / REPORT, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patent", "sketch_id", "original_reason", "revisit", "final_status"])
        for key, verdict in sorted(answers.items()):
            item = state["queue_index"].get(key, {})
            writer.writerow([item.get("patent", ""), item.get("sketch_id", ""),
                             item.get("original_reason", ""), verdict,
                             "accept" if verdict == "accept" else "reject"])
    return len(overturned), len(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True,
                    help="the curate_pilot session whose decisions are being revisited")
    ap.add_argument("--reason", default="shaded_render_or_photo",
                    help="only revisit rejections carrying this reason; empty for all")
    ap.add_argument("--status", default="reject")
    ap.add_argument("--output", type=Path,
                    help="where the overlay is written; default: alongside the session")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    out_dir = args.output or (args.session / "revisit")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets, session_state = load_targets(args.session, args.reason, args.status)
    if not targets:
        print(f"nothing to revisit: no '{args.status}' decisions with reason "
              f"{args.reason!r} in {args.session}")
        return 1

    index = {t["key"]: {"patent": t["patent"], "sketch_id": sketch_id(t),
                        "original_reason": t["original_reason"]} for t in targets}
    session_state = {"queue_index": index}

    path = out_dir / STATE
    state = json.loads(path.read_text()) if path.exists() else {"answers": {}, "cursor": 0}
    print(f"revisiting {len(targets)} '{args.reason}' rejections from {args.session.name}")
    if state["answers"]:
        print(f"resuming: {len(state['answers'])} already reviewed")

    viewer = Viewer(not args.no_display)
    try:
        while state["cursor"] < len(targets):
            target = targets[state["cursor"]]
            if target["key"] in state["answers"]:
                state["cursor"] += 1
                continue
            overturned = sum(1 for v in state["answers"].values() if v == "accept")
            title = (f"[{state['cursor']+1}/{len(targets)}  overturned {overturned}]  "
                     f"{target['key']}  (rejected as {target['original_reason']})")
            print(f"\n{title}")
            viewer.show(target["path"], title)
            try:
                answer = viewer.ask("  [k]keep rejection  [a]accept after all  "
                                    "[s]kip  [b]ack  [q]uit > ")
            except (KeyboardInterrupt, EOFError):
                print("\ninterrupted")
                break
            if answer == "q":
                break
            if answer == "b":
                back = state["cursor"] - 1
                while back >= 0 and targets[back]["key"] not in state["answers"]:
                    back -= 1
                if back < 0:
                    print("  nothing to go back to")
                    continue
                removed = state["answers"].pop(targets[back]["key"])
                print(f"  undid: {removed}")
                state["cursor"] = back
                save_state(out_dir, state)
                continue
            if answer == "a":
                state["answers"][target["key"]] = "accept"
                print("  -> overturned: this is a drawing")
            elif answer == "k":
                state["answers"][target["key"]] = "keep"
            elif answer == "s":
                state["cursor"] += 1
                continue
            else:
                print("  unrecognised; use k / a / s / b / q")
                continue
            state["cursor"] += 1
            save_state(out_dir, state)
    finally:
        viewer.finish()

    save_state(out_dir, state)
    overturned, kept = write_outputs(args.session, out_dir, session_state, state["answers"])
    print(f"\nreviewed {len(state['answers'])} of {len(targets)}")
    print(f"  overturned to accept : {overturned}")
    print(f"  rejection confirmed  : {kept}")
    print(f"  {out_dir / OVERLAY}")
    print(f"  {out_dir / REPORT}")
    if len(state["answers"]) < len(targets):
        print(f"  {len(targets) - len(state['answers'])} left; rerun to continue")
    elif overturned:
        print("\nnext: rebuild the content manifest so the overturned figures enter the corpus")
        print("  python -m tools.build_content_manifest \\")
        print(f"      --session {args.session} --overlay {out_dir / OVERLAY} \\")
        print("      --output benchmarks/mechanical/content_decisions.json --decided-by \"<name>\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
