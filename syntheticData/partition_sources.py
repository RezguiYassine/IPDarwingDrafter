"""Freeze source components and split by original document/CAD-model identity."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from syntheticData.patentvec.generator import SourcePool


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assigned_split(identity, seed):
    number = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()[:16], 16) % 100
    return "train" if number < 80 else "validation" if number < 90 else "test"


def freeze(component, split):
    payload = asdict(component)
    payload["source_to_component"] = component.source_to_component.tolist()
    payload["source"]["split"] = split
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=160926)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.index.read_text())
    pool = SourcePool(Path(source["sources"]["sketchgraphs"]), Path(source["sources"]["cadvg_root"]), split=source["split"])
    raw = pool.sketchgraphs._open()
    raw_sha = file_digest(pool.sketchgraphs.raw_path)
    entries = defaultdict(list)
    identities = defaultdict(set)
    files = {str(pool.sketchgraphs.raw_path): raw_sha}
    grouped = defaultdict(list)
    for entry in source["entries"]:
        grouped[(entry["dataset"], entry["source_id"], entry["view"])].append(entry)
    for position, ((dataset, source_id, view), items) in enumerate(sorted(grouped.items()), 1):
        if dataset == "SketchGraphs":
            document = raw["sketch_ids"][int(source_id)][0].decode("ascii")
            identity = f"SketchGraphs:document:{document}"
            components = pool.sketchgraphs.load_source_sample(int(source_id))
        else:
            identity = f"CAD-VGDrawing:model:{source_id}"
            path = pool.cadvg.svg_path(source_id, view)
            before = file_digest(path)
            components = pool.cadvg.load_source_sample(source_id, view)
            if before != file_digest(path):
                raise ValueError(f"Source changed during snapshot: {path}")
            files[str(path)] = before
        split = assigned_split(identity, args.seed)
        identities[split].add(identity)
        by_key = {c.component_key: c for c in components}
        for item in items:
            component = by_key[item["component_key"]]
            if set(item["profiles"]) != set(pool.component_profiles(component)):
                raise ValueError(f"Stale capability index: {item['component_key']}")
            snapshot = freeze(component, split)
            entries[split].append({**item, "source_identity": identity,
                                   "snapshot": snapshot, "snapshot_sha256": content_digest(snapshot)})
        if position % 500 == 0:
            print(f"Frozen source views {position}/{len(grouped)}", flush=True)
    if file_digest(pool.sketchgraphs.raw_path) != raw_sha:
        raise ValueError("SketchGraphs changed during snapshot")
    required = {"SketchGraphs:rich_balanced_region_line", "CAD-VGDrawing:rich_balanced_region_line",
                "SketchGraphs:endpoint_rich", "CAD-VGDrawing:endpoint_rich", "SketchGraphs:circle",
                "CAD-VGDrawing:endpoint_bezier", "SketchGraphs:single_line", "CAD-VGDrawing:single_line"}
    reports = {}
    for split in ("train", "validation", "test"):
        counts = Counter(f"{e['dataset']}:{p}" for e in entries[split] for p in e["profiles"])
        if any(counts[key] < 2 for key in required):
            raise ValueError(f"Insufficient profile supply in {split}: {counts}")
        payload = {"schema_version": "patentvec-frozen-source-index-2.0", "split": split,
                   "origin_split": source["split"], "seed": args.seed, "strict_index": True,
                   "sources": source["sources"], "source_index_sha256": file_digest(args.index),
                   "entry_count": len(entries[split]), "profile_counts": dict(counts),
                   "source_identity_count": len(identities[split]), "entries": entries[split]}
        path = args.output / f"{split}.json"
        path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        reports[split] = {"sha256": file_digest(path), "entry_count": len(entries[split]),
                          "profile_counts": dict(counts), "source_identity_count": len(identities[split])}
    overlaps = {f"{a}:{b}": len(identities[a] & identities[b]) for a, b in (
        ("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlaps.values()):
        raise ValueError("Source identity leakage")
    report = {"schema": "patentvec-source-partitions-v1", "partitions": reports, "overlaps": overlaps,
              "source_files_sha256": files, "source_index_sha256": file_digest(args.index),
              "implementation_sha256": file_digest(Path(__file__)),
              "scope": "Release partitions are document/model disjoint; these sources may have been seen by historical pretraining."}
    (args.output / "partitions.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"partitions": reports, "overlaps": overlaps}, indent=2))


if __name__ == "__main__":
    main()
