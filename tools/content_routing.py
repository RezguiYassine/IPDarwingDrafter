"""Explicit content routing decisions, independent of execution status.

Two questions the rest of the pipeline cannot answer about itself:

1. **Is this figure's content the kind of content the pipeline represents?**
   A chemical structure, a flowchart, a bar chart and a table all vectorize
   without error -- Stage 2 finds strokes, Stage 3 fits primitives, Stage 4
   exports them -- and the result is meaningless as a training target. Nothing
   in `status: ok`, RANSAC confidence or the quality gates distinguishes them
   from a cross-section. The decision is therefore taken by an authority
   outside the pipeline and supplied as a manifest, bound to the source file's
   hash so that swapping the image invalidates the decision.

2. **Was every region of source ink routed to a channel?** Ink leaves the
   source by exactly three declared routes: removed as a reference numeral at
   Stage 0, removed as a hachure at Stage 2, or retained as drawing ink in the
   Stage-1 output. Ink that reaches none of them was dropped silently, and the
   drawing is not a faithful record of the figure.

Both are fail-closed: a missing manifest, a missing entry, a stale hash or an
uncomputable routing all resolve to `pending`, never to `pass`. There is no
argument that turns an absent decision into an accepted one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA = "ap3-content-routing-v1"
VALIDATOR = "content_routing"
VERSION = "1"

MANIFEST_SCHEMA = "ap3-content-decisions-v1"
ROUTINGS = {"in_scope", "out_of_scope"}

# Ink-routing thresholds. Measured across the 100-figure patent cohort
# (docs/audits/2026-09-17/): unrouted ink is identically zero on every figure,
# because Stage 1 currently routes to `passthrough_binary` and preserves the
# source raster exactly. These bounds are therefore a guard against a Stage-1
# route that deletes content, not a discriminator among today's figures -- a
# component above `max_unrouted_component` is larger than any speckle and
# corresponds to real content.
@dataclass(frozen=True)
class Policy:
    ink_threshold: int = 128
    dilation: int = 2
    max_unrouted_fraction: float = 0.001
    max_unrouted_component: int = 100
    max_reported_components: int = 5


POLICY = Policy()

# Worst-first; a definite out-of-scope decision outranks an uncomputed routing.
_PRECEDENCE = ("error", "fail", "review", "pending", "pass")


def file_digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hashed(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def _worst(*statuses: str) -> str:
    return next(s for s in _PRECEDENCE if s in statuses)


# ── decision manifest ────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    """Read and validate a content decision manifest. Raises on anything malformed.

    A manifest that cannot be trusted must not degrade into a permissive one,
    so every structural problem is an exception rather than a skipped entry.
    """
    path = Path(path)
    document = json.loads(path.read_text())
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unexpected manifest schema: {document.get('schema')!r}")
    if not document.get("authority"):
        raise ValueError("Manifest declares no deciding authority")
    index: dict[tuple[str, str], dict] = {}
    for entry in document.get("decisions") or []:
        key = (entry.get("patent_id"), entry.get("sketch_id"))
        if not all(key):
            raise ValueError(f"Decision without patent/sketch id: {entry}")
        if key in index:
            raise ValueError(f"Duplicate decision for {key[0]}/{key[1]}")
        if entry.get("routing") not in ROUTINGS:
            raise ValueError(f"Invalid routing {entry.get('routing')!r} for {key[0]}/{key[1]}")
        if not entry.get("source_sha256"):
            raise ValueError(f"Decision not bound to a source hash: {key[0]}/{key[1]}")
        index[key] = entry
    if not index:
        raise ValueError("Manifest contains no decisions")
    return {
        "path": str(path.resolve()), "sha256": file_digest(path),
        "schema": document["schema"], "version": document.get("version"),
        "authority": document["authority"], "created_at": document.get("created_at"),
        "index": index,
    }


def manifest_identity(manifest: dict | None) -> dict | None:
    return None if not manifest else {k: manifest[k] for k in
                                      ("path", "sha256", "schema", "version", "authority")}


# ── figure class routing ─────────────────────────────────────────────────────

def _class_decision(manifest: dict | None, patent_id: str, sketch_id: str,
                    source_sha256: str) -> dict:
    if manifest is None:
        return {"status": "pending", "reason_codes": ["content_manifest_absent"]}
    entry = manifest["index"].get((patent_id, sketch_id))
    if entry is None:
        return {"status": "pending", "reason_codes": ["content_decision_missing"]}
    result = {"content_class": entry.get("content_class"), "routing": entry["routing"],
              "reason": entry.get("reason"), "decided_by": entry.get("decided_by"),
              "decided_at": entry.get("decided_at"),
              "decision_source_sha256": entry["source_sha256"]}
    if entry["source_sha256"] != source_sha256:
        # The decision describes a different image than the one processed.
        return {**result, "status": "review", "reason_codes": ["content_source_changed"]}
    if entry["routing"] == "out_of_scope":
        reason = entry.get("reason") or entry.get("content_class") or "unspecified"
        return {**result, "status": "fail",
                "reason_codes": ["content_out_of_scope", f"content_class_{reason}"]}
    return {**result, "status": "pass", "reason_codes": []}


# ── source ink routing ───────────────────────────────────────────────────────

def _channel_masks(shape: tuple[int, int], channels: dict[str, Path],
                   policy: Policy) -> tuple[dict[str, np.ndarray], list[str]]:
    masks, problems = {}, []
    if "reference_mask" in channels:
        mask = cv2.imread(str(channels["reference_mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            problems.append("content_reference_mask_unreadable")
        else:
            masks["stage0_references"] = mask > 0
    if "cleaned" in channels:
        cleaned = cv2.imread(str(channels["cleaned"]), cv2.IMREAD_GRAYSCALE)
        if cleaned is None or cleaned.shape != shape:
            problems.append("content_cleaned_unreadable")
        else:
            masks["stage1_retained"] = cleaned > 127
    if "graph" in channels:
        graph = json.loads(Path(channels["graph"]).read_text())
        hachures = np.zeros(shape, np.uint8)
        for edge in graph.get("removed_hachures") or []:
            points = np.asarray(edge.get("pixels") or [], np.int32)
            if points.size:
                cv2.polylines(hachures, [points.reshape(-1, 1, 2)], False, 255, 1)
        masks["stage2_hachures"] = hachures > 0
    return masks, problems


def _ink_routing(source_path: Path, channels: dict[str, Path] | None,
                 policy: Policy) -> dict:
    if not channels:
        return {"status": "pending", "reason_codes": ["content_ink_routing_pending"]}
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        return {"status": "error", "reason_codes": ["content_source_unreadable"]}
    ink = source < policy.ink_threshold
    masks, problems = _channel_masks(ink.shape, channels, policy)
    if problems or not masks:
        return {"status": "pending",
                "reason_codes": sorted(set(problems)) or ["content_ink_routing_pending"]}

    routed = np.zeros(ink.shape, bool)
    for mask in masks.values():
        routed |= mask
    # Dilate before subtracting: the channels are rendered at slightly
    # different stroke widths than the source, so an exact pixel match would
    # report every stroke edge as unrouted content.
    kernel = np.ones((2 * policy.dilation + 1,) * 2, np.uint8)
    unrouted = ink & ~(cv2.dilate(routed.astype(np.uint8), kernel) > 0)

    ink_pixels = int(ink.sum())
    count, _, stats, _ = cv2.connectedComponentsWithStats(unrouted.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.empty(0, int)
    order = np.argsort(-areas)[:policy.max_reported_components] if areas.size else []
    components = [{"area": int(areas[i]),
                   "bbox": [int(v) for v in stats[i + 1, :4]]} for i in order]
    fraction = float(unrouted.sum() / ink_pixels) if ink_pixels else 0.0
    largest = int(areas.max()) if areas.size else 0

    result = {
        "source_ink_pixels": ink_pixels, "unrouted_pixels": int(unrouted.sum()),
        "unrouted_fraction": fraction, "largest_unrouted_component": largest,
        "unrouted_components": components,
        "channels": {name: int(mask.sum()) for name, mask in sorted(masks.items())},
        "policy": {"dilation": policy.dilation,
                   "max_unrouted_fraction": policy.max_unrouted_fraction,
                   "max_unrouted_component": policy.max_unrouted_component},
    }
    if fraction > policy.max_unrouted_fraction or largest > policy.max_unrouted_component:
        return {**result, "status": "review", "reason_codes": ["content_unrouted_ink"]}
    return {**result, "status": "pass", "reason_codes": []}


# ── entry points ─────────────────────────────────────────────────────────────

def run(patent_id: str, sketch_id: str, source_path: Path, output_path: Path, *,
        manifest: dict | None = None, channels: dict[str, Path] | None = None,
        policy: Policy = POLICY) -> dict:
    inputs = {"source": _hashed(source_path)}
    for name, path in sorted((channels or {}).items()):
        inputs[name] = _hashed(path)
    class_decision = _class_decision(manifest, patent_id, sketch_id,
                                     inputs["source"]["sha256"])
    ink_routing = _ink_routing(Path(source_path), channels, policy)
    report = {
        "schema": SCHEMA, "validator": VALIDATOR, "version": VERSION,
        "patent_id": patent_id, "sketch_id": sketch_id,
        "status": _worst(class_decision["status"], ink_routing["status"]),
        "reason_codes": sorted(set(class_decision["reason_codes"]
                                   + ink_routing["reason_codes"])),
        "manifest": manifest_identity(manifest),
        "class_decision": class_decision, "ink_routing": ink_routing,
        "inputs": inputs,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output_path)
    return report


def acceptance_check(report: dict) -> dict:
    return {key: report[key] for key in ("status", "reason_codes", "validator", "version")}
