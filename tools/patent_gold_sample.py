"""
Gold patent keypoint eval set — sampler  (accuracy roadmap — step #1)
====================================================================

PatentData has no keypoint ground truth, so fusion's effect on patents can't be
measured. This builds a small labelling pack so you can hand-label a gold set:

  for N curated patent sketches (stratified by complexity):
    <id>_skeleton.png   the production-resolution skeleton (what the detector
                        sees at inference — resolution-capped like Stage 2)
    <id>.json           pre-labels {x,y,type} from the current fusion seeding,
                        as a CORRECTION starting point (NOT ground truth)
    <id>_overlay.png    skeleton + colour-coded pre-labels for review

YOU then correct each <id>.json: remove false positives, add missed keypoints,
fix types (endpoint / junction / corner). `tools.patent_gold_eval` scores
CN vs CNN detectors against the corrected labels.

Usage:
    python -m tools.patent_gold_sample --n 60 \
        --manifest output/PatentData_clean12_gated/training_manifest_clip.csv \
        --output output/PatentData/gold_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "stage2_strokeextraction"))
import stage2_stroke_extract as s2  # noqa: E402

KP_COLORS = {"endpoint": (0, 170, 0), "junction": (0, 0, 230), "corner": (230, 90, 0)}


def cap_skeleton(sk: np.ndarray, max_res: int) -> np.ndarray:
    """Replicate Stage 2's resolution cap so the gold skeleton matches what the
    detector sees at inference (dilate → resize → re-skeletonize)."""
    H, W = sk.shape
    if not max_res or max(H, W) <= max_res:
        return (sk > 0).astype(np.uint8) * 255
    scale = max_res / max(H, W)
    nW, nH = max(1, round(W * scale)), max(1, round(H * scale))
    k = max(3, int(1.0 / scale) * 2 + 1)
    out = cv2.dilate((sk > 0).astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    out = cv2.resize(out, (nW, nH), interpolation=cv2.INTER_AREA)
    out = (out > 30).astype(np.uint8)
    return s2._skeletonize(out).astype(np.uint8) * 255


def prelabels(sk: np.ndarray, model, conf: float) -> list[dict]:
    """Corner candidates only. Endpoints/junctions are NOT the research question
    — CN handles them on patents; the contested, fine-tuned class is corners.
    So the gold set is corner-only (tractable to label exhaustively), and the
    pre-labels are the CNN's corner detections for the user to prune/extend."""
    if model is None:
        return []
    return [{"x": int(k["x"]), "y": int(k["y"]), "type": "corner"}
            for k in model.detect(sk, conf, 5) if k["type"] == s2.KP_CORNER]


def overlay(sk: np.ndarray, kps: list[dict]) -> np.ndarray:
    c = np.full((*sk.shape, 3), 255, np.uint8)
    c[sk > 0] = (190, 190, 190)
    for k in kps:
        cv2.circle(c, (k["x"], k["y"]), 3, KP_COLORS.get(k["type"], (0, 0, 0)), -1)
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(
        PROJECT_ROOT / "output/PatentData_clean12_gated/training_manifest_clip.csv"))
    ap.add_argument("--output", default=str(PROJECT_ROOT / "output/PatentData/gold_eval"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-prims", type=float, default=150,
                    help="skip sketches with more fitted primitives (labelability)")
    ap.add_argument("--max-res", type=int, default=1000)
    ap.add_argument("--weights", default=str(PROJECT_ROOT / "models/puhachov_d2c.pth"))
    ap.add_argument("--conf", type=float, default=0.45, help="corner pre-label confidence")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest))
            if r.get("cleaned_skeleton_path") and Path(r["cleaned_skeleton_path"]).exists()]
    # Cap complexity so each sketch is feasible to hand-label exhaustively
    # (a 600-primitive figure has hundreds of keypoints — not labelable).
    rows = [r for r in rows if float(r.get("s3_n_primitives") or 1e9) <= args.max_prims]
    print(f"{len(rows)} sketches with <= {args.max_prims} primitives (labelable)")
    # stratify by primitive-count into 3 complexity bins, sample evenly
    rows.sort(key=lambda r: float(r.get("s3_n_primitives") or 0))
    rng = np.random.default_rng(args.seed)
    bins = np.array_split(rows, 3)
    per = max(1, args.n // 3)
    picks = []
    for b in bins:
        idx = rng.choice(len(b), size=min(per, len(b)), replace=False)
        picks.extend(b[i] for i in idx)
    picks = picks[:args.n]

    try:
        model = s2.PuhachovKeypointDetector(args.weights, device=args.device)
    except Exception as e:
        print(f"CNN unavailable ({e}); pre-labels will lack corners")
        model = None

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in picks:
        # sketch_id (e.g. F0069) is unique only within a patent; the patent id
        # is the dir above cleaned/. Build a globally-unique file id.
        skel_path = Path(r["cleaned_skeleton_path"])
        patent = skel_path.parent.parent.name
        sid = f"{patent}_{r['sketch_id'].replace('/', '_')}"
        sk = cv2.imread(r["cleaned_skeleton_path"], 0)
        if sk is None or not r.get("graph_path") or not Path(r["graph_path"]).exists():
            continue
        graph = json.loads(Path(r["graph_path"]).read_text())
        sk = cap_skeleton(sk, args.max_res)
        # align to the graph's (Stage-2) frame so its nodes overlay correctly
        ishape = graph.get("image_shape")
        if ishape and list(sk.shape) != list(ishape):
            sk = (cv2.resize(sk, (ishape[1], ishape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
        kps = prelabels(sk, model, args.conf)
        cv2.imwrite(str(out / f"{sid}_skeleton.png"), sk)
        cv2.imwrite(str(out / f"{sid}_overlay.png"), overlay(sk, kps))
        json.dump({"sketch_id": r["sketch_id"], "shape": list(sk.shape),
                   "labeled_classes": ["corner"], "keypoints": kps,
                   "reviewed": False},
                  open(out / f"{sid}.json", "w"), indent=1)
        manifest.append({"sketch_id": r["sketch_id"], "file": sid,
                         "n_prelabels": len(kps),
                         "s3_n_primitives": r.get("s3_n_primitives")})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sketch_id", "file", "n_prelabels", "s3_n_primitives"])
        w.writeheader(); w.writerows(manifest)

    (out / "LABELING.md").write_text(
        "# Gold patent CORNER labelling\n\n"
        f"{len(manifest)} sketches. This gold set is **corner-only** — endpoints "
        "and junctions are handled well by the crossing-number path on patents; "
        "the contested, fine-tuned class is corners, so that's all we label.\n\n"
        "A **corner** = a point where a *single* stroke bends sharply (a polygon "
        "vertex, a rectangle corner). NOT where strokes cross (that's a junction) "
        "and NOT a free stroke end (endpoint).\n\n"
        "For each `<id>.json`, open `<id>_overlay.png` (blue dots = CNN corner "
        "guesses) next to `<id>_skeleton.png` and correct the `keypoints` list:\n"
        "- remove dots that are not real corners (crossings, ends, curve noise)\n"
        "- add missed corners (x,y on the skeleton; type = corner)\n"
        "- set `\"reviewed\": true` when done\n\n"
        f"Then: `python -m tools.patent_gold_eval --gold {out} "
        "--weights-ft models/puhachov_d2c_patentft.pth`\n")
    print(f"wrote {len(manifest)} sketches → {out}\n"
          f"  next: label the *.json (see {out}/LABELING.md), then run "
          f"tools.patent_gold_eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
