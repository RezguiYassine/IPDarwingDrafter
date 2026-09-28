#!/usr/bin/env python3
"""Turn a curation session into a content decision manifest for acceptance.

The curation tool records which figures a human judged to be engineering
drawings; acceptance needs those judgements in a form it can verify, which
means every decision bound to the SHA-256 of the exact image it was made
about. If the TIF is later replaced, the hash stops matching and the decision
reverts to `review` rather than silently applying to a different drawing.

    python -m tools.build_content_manifest \
        --session output/PatentData/curation_v2 \
        --output output/PatentData/content_decisions.json \
        --decided-by "<name>"

Rejections are carried over as well as acceptances: a figure the curator
rejected is an explicit `out_of_scope` decision, which fails the content check
rather than leaving it pending. Skips are omitted -- an undecided figure must
stay undecided.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path

from tools import content_routing


def build(session: Path, decided_by: str, authority: str,
          overlay: Path | None = None) -> dict:
    state = json.loads((session / "session.json").read_text())
    # A revisit overlay overturns decisions from the first pass without
    # editing it, so the original judgement stays on the record.
    overturned = set()
    if overlay:
        document = json.loads(Path(overlay).read_text())
        if document.get("schema") != "ap3-revisit-overlay-v1":
            raise SystemExit(f"not a revisit overlay: {overlay}")
        overturned = set(document.get("overturned_to_accept") or ())
    queue = {f"{item['patent']}/{Path(item['filename']).stem}": item
             for item in state["queue"]}
    decided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decisions, missing = [], []
    for key, decision in sorted(state["decisions"].items()):
        if decision["status"] not in {"accept", "reject"}:
            continue
        if key in overturned:
            decision = {**decision, "status": "accept",
                        "revisit": "overturned_from_" + (decision.get("reason") or "reject")}
        item = queue.get(key)
        if item is None:
            missing.append(key)
            continue
        source = Path(item["path"])
        if not source.is_file():
            missing.append(key)
            continue
        stem = Path(item["filename"]).stem
        prefix = item["patent"] + "_"
        decisions.append({
            "patent_id": item["patent"],
            "sketch_id": stem[len(prefix):] if stem.startswith(prefix) else stem,
            "source_path": str(source.resolve()),
            "source_sha256": content_routing.file_digest(source),
            "routing": "in_scope" if decision["status"] == "accept" else "out_of_scope",
            "content_class": decision.get("prior_class") or None,
            "reason": decision.get("reason") or None,
            "revisit": decision.get("revisit"),
            "decided_by": decided_by,
            "decided_at": decided_at,
        })
    if missing:
        raise SystemExit(f"cannot bind {len(missing)} decisions to a source file: "
                         f"{', '.join(missing[:5])}")
    return {
        "schema": content_routing.MANIFEST_SCHEMA,
        "version": "1",
        "authority": authority,
        "created_at": decided_at,
        "session": str(session.resolve()),
        "decisions": decisions,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, required=True,
                    help="curation session directory written by tools.curate_pilot")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--decided-by", default="curator",
                    help="who made the decisions; recorded on every entry")
    ap.add_argument("--authority", default="human_curation",
                    help="deciding authority recorded in the manifest header")
    ap.add_argument("--overlay", type=Path,
                    help="a revisit overlay whose overturned decisions are applied")
    args = ap.parse_args()

    document = build(args.session, args.decided_by, args.authority, args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(args.output)

    counts = collections.Counter(d["routing"] for d in document["decisions"])
    revisited = sum(1 for d in document["decisions"] if d.get("revisit"))
    reasons = collections.Counter(d["reason"] for d in document["decisions"]
                                  if d["routing"] == "out_of_scope")
    print(f"{args.output}")
    print(f"  decisions   : {len(document['decisions'])}")
    print(f"  in_scope    : {counts['in_scope']}")
    print(f"  out_of_scope: {counts['out_of_scope']}  {dict(reasons)}")
    if revisited:
        print(f"  overturned  : {revisited} (from a revisit overlay)")
    print(f"  sha256      : {content_routing.file_digest(args.output)}")
    # Fail loudly here rather than at acceptance time.
    content_routing.load_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
