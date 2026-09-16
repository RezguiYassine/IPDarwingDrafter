"""Measure retained synthetic rasters, exact C2 labels and real Stage 1 behavior."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import yaml

from stage1_preprocessing import stage1_preprocess as stage1
from syntheticData.audit_stage2_topology_contract import _cn_map_vectorized
from syntheticData.patentvec.stage2_targets import REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _components(binary):
    n, labels, stats, centres = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return n-1, stats[1:, cv2.CC_STAT_AREA], centres[1:]


def metrics(skeleton: np.ndarray) -> dict:
    foreground = skeleton > 0
    padded = np.pad(foreground.astype(np.uint8), 1)
    height, width = foreground.shape
    neighbours = sum(padded[dy:dy+height, dx:dx+width]
                     for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1))
    cn = _cn_map_vectorized(foreground.astype(np.uint8))
    pixels = int(foreground.sum())
    components, sizes, _ = _components(foreground)
    _, branch_sizes, _ = _components(foreground & (neighbours < 3))
    endpoints, _, _ = _components(foreground & (cn == 1))
    junctions, _, _ = _components(foreground & (cn >= 3))
    return {
        "skeleton_pixels": pixels, "components": components,
        "junction_pixels_per_1k": float(np.count_nonzero(foreground & (neighbours >= 3))*1000/max(pixels, 1)),
        "endpoint_pixels_per_1k": float(np.count_nonzero(foreground & (neighbours == 1))*1000/max(pixels, 1)),
        "tiny_component_fraction": float(np.mean(sizes <= 4)) if len(sizes) else 0.0,
        "short_branch_fraction": float(np.mean(branch_sizes <= 5)) if len(branch_sizes) else 0.0,
        "median_branch_length_px": float(np.median(branch_sizes)) if len(branch_sizes) else 0.0,
        "cn_endpoint_clusters": endpoints, "cn_junction_clusters": junctions,
        "cn_endpoint_clusters_per_1k": endpoints*1000/max(pixels, 1),
        "cn_junction_clusters_per_1k": junctions*1000/max(pixels, 1),
        "height": height, "width": width,
        "tile_count_512_stride256": int((1+np.ceil(max(0, height-512)/256)) *
                                        (1+np.ceil(max(0, width-512)/256))),
    }


def audit_c2(skeleton, keypoints, masks) -> dict:
    """Compare exact sets to independent production-CN cluster representatives."""
    keypoints = np.asarray(keypoints)
    if (keypoints.ndim != 2 or keypoints.shape[1] != 3 or not np.isfinite(keypoints).all()
            or not np.equal(keypoints, np.floor(keypoints)).all()
            or not np.isin(keypoints[:, 2], [0, 1, 2]).all()):
        raise ValueError("Invalid C2 keypoint array")
    binary = skeleton > 0
    cn = _cn_map_vectorized(binary.astype(np.uint8))
    expected = set()
    for kind, candidate in ((0, cn == 1), (1, cn >= 3)):
        _, _, centres = _components(binary & candidate)
        expected.update((int(x), int(y), kind) for x, y in centres)
    topology = [tuple(map(int, p)) for p in keypoints if int(p[2]) in (0, 1)]
    actual = set(topology)
    support = np.logical_or.reduce([masks[name] > 0 for name in ("object", "hidden_center", "hatch")])
    return {"missing_topology_labels": len(expected-actual), "extra_topology_labels": len(actual-expected),
            "duplicate_topology_labels": len(topology)-len(actual),
            "unsupported_skeleton_pixels": int(np.count_nonzero(binary & ~support)),
            "endpoints": sum(p[2] == 0 for p in expected), "junctions": sum(p[2] == 1 for p in expected)}


def skeleton_fidelity(actual, target, tolerance=2.0):
    actual, target = actual > 0, target > 0
    distance_to_actual = cv2.distanceTransform((~actual).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_to_target = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    recall = float(np.mean(distance_to_actual[target] <= tolerance)) if target.any() else 0.0
    precision = float(np.mean(distance_to_target[actual] <= tolerance)) if actual.any() else 0.0
    return {"c2_recall_at2px": recall, "c2_precision_at2px": precision,
            "c2_f1_at2px": 2*precision*recall/max(precision+recall, 1e-12)}


def raster_metrics(gray, *, foreground_white=False, longside=None):
    foreground = gray > 0 if foreground_white else gray < 210
    if longside is not None:
        height, width = foreground.shape
        scale = longside / max(height, width)
        foreground = cv2.resize(foreground.astype(np.float32),
                                (max(1, round(width*scale)), max(1, round(height*scale))),
                                interpolation=cv2.INTER_AREA) >= 0.5
    return metrics(skeletonize(foreground))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for view, values in row["metrics"].items():
            groups[(row["domain"], row["difficulty"], view)].append(values)
    summaries = []
    for (domain, difficulty, view), values in sorted(groups.items()):
        summaries.append({"domain": domain, "difficulty": difficulty, "view": view, "count": len(values),
                          "metrics": {name: {"p10": float(np.quantile([r[name] for r in values], .1)),
                                             "median": float(np.median([r[name] for r in values])),
                                             "p90": float(np.quantile([r[name] for r in values], .9))}
                                      for name in values[0]}})
    return summaries


def _png(payload):
    with Image.open(io.BytesIO(payload)) as image:
        return np.asarray(image.convert("L"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--patent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path)
    parser.add_argument("--stage1-device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    assert len(rows) == manifest["count"] == manifest["requested_count"]
    assert len({row["sample_id"] for row in rows}) == len(rows)
    for shard in manifest["shards"]:
        assert digest(args.dataset / shard["archive"]) == shard["sha256"]
    by_archive = defaultdict(list)
    for row in rows:
        by_archive[row["archive"]].append(row)
    model = None
    config = None
    model_record = None
    if args.stage1_config:
        config = yaml.safe_load(args.stage1_config.read_text())
        weights = Path(config["sketchcleannet"]["weights"]).resolve()
        config["sketchcleannet"]["weights"] = str(weights)
        config["sketchcleannet"]["device"] = args.stage1_device
        model = stage1.load_model(config)
        if model is None or model._model is None:
            raise RuntimeError("Requested Stage 1 model is unavailable")
        model_record = {"config_sha256": digest(args.stage1_config), "effective_config": config,
                        "stage1_implementation_sha256": digest(Path(stage1.__file__)),
                        "weights_sha256": digest(weights)}
    results = []
    audits = []
    routes = Counter()
    for archive_name, archive_rows in sorted(by_archive.items()):
        with tarfile.open(args.dataset / archive_name) as archive:
            for row in archive_rows:
                def read(name):
                    stream = archive.extractfile(row["member_prefix"]+"/"+name)
                    if stream is None:
                        raise ValueError(f"Missing member {row['sample_id']}/{name}")
                    return stream.read()
                masks = {}
                with np.load(io.BytesIO(read("masks.npz")), allow_pickle=False) as data:
                    masks = {name: np.asarray(data[name]) for name in data.files}
                with np.load(io.BytesIO(read("puhachov.npz")), allow_pickle=False) as data:
                    skeleton, keypoints = data["skeleton"], data["kps"]
                    assert json.loads(str(data["meta"]))["label_contract"] == REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT
                audited = audit_c2(skeleton, keypoints, masks)
                audits.append({"sample_id": row["sample_id"], **audited})
                views = {"c2_training_input": metrics(skeleton)}
                for name in ("clean", "degraded", "reference_free_clean", "reference_free_degraded"):
                    gray = _png(read(name+".png"))
                    assert tuple(gray.shape) == tuple(skeleton.shape)
                    views[name+"_threshold210"] = raster_metrics(gray)
                if config is not None:
                    sample_dir = args.output / "stage1" / row["sample_id"]
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    path = sample_dir / "reference_free_degraded.png"
                    path.write_bytes(read("reference_free_degraded.png"))
                    result = stage1.run(path, sample_dir, row["sample_id"], config, model=model)
                    actual = cv2.imread(str(result.skeleton_path), cv2.IMREAD_GRAYSCALE)
                    if actual is None:
                        raise ValueError("Stage 1 did not retain a readable skeleton")
                    views["deployed_stage1_oracle_reference_free"] = metrics(actual)
                    views["deployed_stage1_oracle_reference_free"].update(skeleton_fidelity(actual, skeleton))
                    routes[result.model_used] += 1
                results.append({"sample_id": row["sample_id"], "domain": "synthetic",
                                "difficulty": row["difficulty"], "metrics": views})
                if len(results) % 10 == 0:
                    print(f"Measured synthetic {len(results)}/{len(rows)}", flush=True)
    for path in sorted(args.patent_run.glob("*/cleaned/*_skeleton.png")):
        skeleton = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if skeleton is None:
            raise ValueError(f"Unreadable patent skeleton {path}")
        clean_path = path.with_name(path.name.replace("_skeleton.png", "_cleaned.png"))
        clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
        if clean is None:
            raise ValueError(f"Unreadable patent cleaned raster {clean_path}")
        if not np.isin(clean, [0, 255]).all():
            clean = stage1._binarize(clean)
        views = {"deployed_stage1_native": metrics(skeleton),
                 "deployed_stage1_matched_longside1024": raster_metrics(clean, foreground_white=True, longside=1024)}
        results.append({"sample_id": path.parent.parent.name+"/"+path.stem,
                        "domain": "patent", "difficulty": "real", "skeleton_sha256": digest(path), "metrics": views})
    totals = {name: sum(row[name] for row in audits) for name in audits[0] if name != "sample_id"}
    errors = sum(totals[name] for name in ("missing_topology_labels", "extra_topology_labels",
                                          "duplicate_topology_labels", "unsupported_skeleton_pixels"))
    report = {"schema": "patent-like-measurements-v1", "dataset": str(args.dataset.resolve()),
              "dataset_manifest_sha256": digest(args.dataset / "manifest.json"),
              "implementation_sha256": digest(Path(__file__)), "stage1": model_record,
              "stage1_routes": dict(routes), "contract_pass": errors == 0, "contract_totals": totals,
              "contract_samples": audits, "summaries": summarize(results), "rows": results,
              "release_accepted": False,
              "definitions": {"junction_pixels": "8-neighbour degree >=3 foreground pixels, not CN graph nodes",
                              "cn_clusters": "production crossing number: 8-connected endpoint CN=1 or junction CN>=3 clusters",
                              "tiny_component": "8-connected foreground component with <=4 pixels",
                              "branch": "8-connected component after removing degree>=3 foreground pixels; legacy diagnostic, not a traced stroke",
                              "short_branch": "branch component with <=5 pixels",
                              "raster_proxy": "gray<210 then skimage skeletonize, without Stage 1 cleanup",
                              "matched_frame": "binary ink area resampling to longside 1024, threshold >=0.5, then skeletonize; resampling may change topology",
                              "stage1_input": "oracle reference-free degraded raster; no OCR approximation",
                              "acceptance": "historic numeric bands are provisional; no synthetic deployment-gate calibration"}}
    (args.output / "measurements.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"contract_pass": errors == 0, "contract_totals": totals,
                      "rows": len(results), "stage1_routes": dict(routes)}))
    return int(errors != 0)


if __name__ == "__main__":
    raise SystemExit(main())
