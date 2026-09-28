#!/usr/bin/env python3
"""Select the mechanical subset of the corpus by IPC classification.

Every patent names its own subject in the XML, and that is a prior on whether
its drawings are engineering drawings at all. Measured on the 158 curated
figures, restricting to IPC sections B (performing operations, transporting)
and F (mechanical engineering, lighting, heating, weapons) raises the drawing
rate of the pool from about 71% to 81.8%.

It is a prior, not an authority. 81.8% is far from the 98% precision an
automated content decision needs, and the rule discards real drawings in
sections G and H where electrical and instrumentation patents live. What it
buys is a smaller, richer pool to spend hand labels on.

    python -m tools.ipc_filter --output benchmarks/mechanical
    python -m tools.ipc_filter --sections ABEF --output benchmarks/wider

Writes:
    <output>/pool.csv              patent_id, sketch_id, input_path, ipc
    <output>/ipc_index.json        every patent's codes, including the excluded
    <output>/pool_summary.json     counts and the curated-set check
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

SCHEMA = "ap3-ipc-pool-v1"
VERSION = "1"

# <classification-ipcr sequence="1"><text>F16K 31/06 20060101AFI...</text>
_IPCR = re.compile(r"<classification-ipcr[^>]*?sequence=\"(\d+)\"[^>]*>\s*<text>\s*"
                   r"([A-H]\d\d[A-Z])", re.S)
_IPCR_ANY = re.compile(r"<classification-ipcr[^>]*>\s*<text>\s*([A-H]\d\d[A-Z])", re.S)
_MAIN = re.compile(r"<main-classification>\s*([A-H]\d\d[A-Z])")


def codes_for(patent_dir: Path) -> list[str]:
    """IPC subclasses in the office's own sequence order; primary first."""
    xmls = sorted(patent_dir.glob("*.xml"))
    if not xmls:
        return []
    text = xmls[0].read_text(errors="ignore")
    sequenced = _IPCR.findall(text)
    if sequenced:
        ordered = sorted(sequenced, key=lambda pair: int(pair[0]))
        seen, out = set(), []
        for _, code in ordered:
            if code not in seen:
                seen.add(code)
                out.append(code)
        return out
    found = _IPCR_ANY.findall(text) or _MAIN.findall(text)
    return list(dict.fromkeys(found))


def build(manifest: Path, corpus: Path, sections: str) -> tuple[list[dict], dict]:
    by_patent: dict[str, list[dict]] = {}
    for row in csv.DictReader(open(manifest)):
        if row.get("label") == "drawing":
            by_patent.setdefault(row["patent"], []).append(row)
    index, pool = {}, []
    counts = collections.Counter()
    for patent in sorted(by_patent):
        codes = codes_for(corpus / patent)
        primary = codes[0] if codes else None
        index[patent] = {"codes": codes, "primary": primary}
        counts[primary[0] if primary else "none"] += 1
        if not primary or primary[0] not in sections:
            continue
        # One figure per patent, matching how the curated cohort was sampled.
        pick = sorted(by_patent[patent], key=lambda r: r["filename"])[0]
        stem = Path(pick["filename"]).stem
        prefix = patent + "_"
        pool.append({
            "patent_id": patent,
            "sketch_id": stem[len(prefix):] if stem.startswith(prefix) else stem,
            "input_path": str(Path(pick["path"]).resolve()),
            "ipc": primary,
        })
    return pool, {"sections_kept": sections, "patents_considered": len(by_patent),
                  "first_section_counts": dict(sorted(counts.items())),
                  "pool_size": len(pool), "index": index}


def check_against_curated(index: dict, sections: str, decisions: Path) -> dict | None:
    """What the rule would have done to figures a human already judged."""
    if not decisions.is_file():
        return None
    kept = collections.Counter()
    dropped = collections.Counter()
    for row in csv.DictReader(open(decisions)):
        if row["status"] not in {"accept", "reject"}:
            continue
        entry = index.get(row["patent"])
        primary = (entry or {}).get("primary")
        target = kept if primary and primary[0] in sections else dropped
        target[row["status"]] += 1
    total_kept = kept["accept"] + kept["reject"]
    return {
        "kept": dict(kept), "dropped": dict(dropped),
        "precision_of_pool": kept["accept"] / total_kept if total_kept else None,
        "recall_of_drawings": (kept["accept"] / (kept["accept"] + dropped["accept"])
                               if kept["accept"] + dropped["accept"] else None),
        "note": "precision is the drawing rate a labeller would meet inside the pool; "
                "recall is the share of real drawings the rule keeps",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=Path("output/PatentData/filter_manifest_v3.csv"))
    ap.add_argument("--corpus", type=Path,
                    default=Path("data/PatentData/ReorganisedData"))
    ap.add_argument("--sections", default="BF",
                    help="IPC sections to keep, e.g. BF or ABEF")
    ap.add_argument("--decisions", type=Path,
                    default=Path("benchmarks/curatedv1/curation_decisions.csv"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sections = args.sections.upper()
    pool, summary = build(args.manifest, args.corpus, sections)
    summary.update(schema=SCHEMA, version=VERSION,
                   curated_check=check_against_curated(summary["index"], sections, args.decisions))
    index = summary.pop("index")

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "pool.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patent_id", "sketch_id", "input_path", "ipc"])
        writer.writeheader()
        writer.writerows(pool)
    (args.output / "ipc_index.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.output / "pool_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"patents with a 'drawing' figure : {summary['patents_considered']}")
    print(f"first IPC section               : {summary['first_section_counts']}")
    print(f"pool (sections {sections})            : {summary['pool_size']} "
          f"({summary['pool_size']/max(1, summary['patents_considered']):.1%})")
    check = summary["curated_check"]
    if check:
        print(f"\non the {sum(check['kept'].values()) + sum(check['dropped'].values())} curated figures:")
        print(f"  inside the pool : {check['kept']}  -> drawing rate {check['precision_of_pool']:.1%}")
        print(f"  outside         : {check['dropped']}")
        print(f"  keeps {check['recall_of_drawings']:.1%} of the real drawings")
    print(f"\nwritten: {args.output}/pool.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
