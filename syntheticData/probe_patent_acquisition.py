"""Paired acquisition/scale diagnostics on frozen PatentVec source geometry."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import html
import io
import json
import logging
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image
import yaml

from stage1_preprocessing import stage1_preprocess as stage1
from syntheticData.build_dataset_audit import _stratified_rows
from syntheticData.measure_patent_like import audit_c2, digest, metrics, skeleton_fidelity, summarize
from syntheticData.patentvec.render import (
    DEFAULT_STROKE_WIDTH, MASK_GROUPS, degradation_parameters, degrade_patent_scan,
    render_clean, render_masks,
)
from syntheticData.patentvec.schema import CanonicalDrawing
from syntheticData.patentvec.stage2_targets import (
    DEFAULT_TOPOLOGY_MASKS, REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT,
    relabel_puhachov_payload,
)
from syntheticData.patentvec.training import puhachov_arrays


def scaled_drawing(drawing, canvas, width_mode):
    if width_mode not in ("relative", "fixed_pixel"):
        raise ValueError(f"Unknown width mode: {width_mode}")
    if drawing.canvas[0] != drawing.canvas[1] or canvas < 128:
        raise ValueError("Probe requires square source canvases and canvas >=128")
    factor = drawing.canvas[0] / canvas if width_mode == "fixed_pixel" else 1.0

    def scale(primitive):
        width = primitive.style.get("stroke_width", DEFAULT_STROKE_WIDTH.get(primitive.semantic, .0015))
        return replace(primitive, style={**primitive.style, "stroke_width": width * factor})

    return replace(drawing, canvas=[canvas, canvas],
                   primitives_visible=[scale(p) for p in drawing.primitives_visible],
                   primitives_amodal=[scale(p) for p in drawing.primitives_amodal])


def acquisition_variants(clean, support, seed, base_degradation):
    low = degradation_parameters({**base_degradation, "dark_speckle_probability": .00004,
                                  "light_speckle_probability": .00003})
    normal = degrade_patent_scan(clean, seed, base_degradation)
    reduced = degrade_patent_scan(clean, seed, low)
    return {
        "binary_c2_control": np.where(support, 0, 255).astype(np.uint8),
        "gray_default": normal,
        "gray_low_speckle": reduced,
        "binary_default": np.where(normal < 210, 0, 255).astype(np.uint8),
        "binary_low_speckle": np.where(reduced < 210, 0, 255).astype(np.uint8),
    }


def _npz(arrays):
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def _save_png(path, gray):
    Image.fromarray(gray).save(path)


def gallery(output, groups):
    sections = []
    for group in groups:
        columns = []
        for variant, row in group["variants"].items():
            folder = f"samples/{group['sample_id']}/{group['render_case']}/{variant}"
            score = row["metrics"]
            columns.append(
                f'<div class="variant"><h3>{html.escape(variant)}</h3>'
                f'<p>{html.escape(row["route"])}<br>F1 {score["c2_f1_at2px"]:.3f}; '
                f'CN/1k {score["cn_junction_clusters_per_1k"]:.2f}</p>'
                f'<a href="{folder}/reference_free.png"><img src="{folder}/reference_free.png" alt="Reference-free input"></a>'
                f'<a href="{folder}/cleaned/probe_skeleton.png"><img class="skeleton" '
                f'src="{folder}/cleaned/probe_skeleton.png" alt="Stage 1 skeleton"></a></div>'
            )
        target = f"samples/{group['sample_id']}/{group['render_case']}/clean_target.png"
        sections.append(f'<section><h2>{html.escape(group["sample_id"])} / {group["render_case"]}</h2>'
                        f'<div class="comparison"><div class="variant"><h3>Clean C2 target</h3>'
                        f'<a href="{target}"><img class="skeleton" src="{target}" alt="Clean target skeleton"></a></div>'
                        + "".join(columns) + '</div></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Patent acquisition comparison</title><style>
body { margin:0; color:#202020; background:#fff; font:14px system-ui,sans-serif; }
header { padding:20px; border-bottom:1px solid #bbb; } h1 { font-size:22px; margin:0 0 8px; }
main { padding:0 20px 24px; } section { border-bottom:1px solid #bbb; padding:20px 0; }
h2 { font-size:16px; overflow-wrap:anywhere; } h3 { font-size:13px; margin:0; overflow-wrap:anywhere; }
a { color:#075985; } p { min-height:42px; margin:8px 0; font-size:12px; }
.comparison { display:grid; grid-template-columns:repeat(6,minmax(180px,1fr)); overflow-x:auto; gap:12px; }
.variant { min-width:0; } img { display:block; width:100%; aspect-ratio:1; object-fit:contain; }
.skeleton { filter:invert(1); } img:hover { outline:1px solid #777; }
</style><header><h1>Patent acquisition comparison</h1><a href="report.json">Measurement report</a></header><main>'''
    (output / "index.html").write_text(page + "".join(sections) + "</main></html>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-difficulty", type=int, default=10)
    parser.add_argument("--stage1-config", type=Path, default=Path("config_deploy.yaml"))
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.per_difficulty < 1:
        parser.error("per-difficulty must be positive")
    logging.basicConfig(level=logging.ERROR)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    selected = _stratified_rows(rows, args.per_difficulty)
    counts = Counter(row["difficulty"] for row in selected)
    if counts != Counter({tier: args.per_difficulty for tier in ("medium", "hard", "very_hard")}):
        raise ValueError("Insufficient balanced source samples")
    for shard in manifest["shards"]:
        if digest(args.dataset / shard["archive"]) != shard["sha256"]:
            raise ValueError(f"Source shard checksum mismatch: {shard['archive']}")
    config = yaml.safe_load(args.stage1_config.read_text())
    weights = Path(config["sketchcleannet"]["weights"]).resolve()
    config["sketchcleannet"].update(weights=str(weights), device=args.device)
    model = stage1.load_model(config)
    if model is None or model._model is None:
        raise RuntimeError("Requested Stage 1 model is unavailable")
    render_cases = ((1024, "relative"), (2048, "relative"), (2048, "fixed_pixel"))
    base = manifest["settings"]["degradation"]
    plan = {
        "schema": "patent-acquisition-probe-v1", "dataset": str(args.dataset.resolve()),
        "source_manifest_sha256": digest(args.dataset / "manifest.json"),
        "sample_ids": [row["sample_id"] for row in selected], "render_cases": render_cases,
        "base_degradation": base, "low_speckle_overrides": {"dark_speckle_probability": .00004, "light_speckle_probability": .00003},
        "binary_scan_threshold": 210, "effective_config": config,
        "weights_sha256": digest(weights), "config_sha256": digest(args.stage1_config),
        "implementation_sha256": {str(path): digest(path) for path in (
            Path(__file__), Path(stage1.__file__), Path("syntheticData/measure_patent_like.py"),
            Path("syntheticData/patentvec/render.py"), Path("syntheticData/patentvec/training.py"),
            Path("syntheticData/patentvec/stage2_targets.py"))},
        "label_contract": "C2 clean target only; no clean coordinates attached to changed observations",
        "binary_c2_control": "positive control: exact semantic-mask union, not a learned recovery result",
        "scale_contract": "source geometry, dash lengths and text geometry fixed in normalized coordinates; fixed_pixel adjusts stroke widths only",
        "release_accepted": False,
    }
    # Each probe has a fresh immutable directory; a failed run is not silently resumed.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    measurements, groups, gallery_groups = [], [], []
    shown = set()
    for position, row in enumerate(selected, 1):
        with tarfile.open(args.dataset / row["archive"]) as archive:
            def read(name):
                stream = archive.extractfile(row["member_prefix"] + "/" + name)
                if stream is None:
                    raise ValueError(f"Missing source artifact: {name}")
                return stream.read()
            source = read("sample.json")
            original = CanonicalDrawing.from_dict(json.loads(source))
            if original.canvas != [1024, 1024]:
                raise ValueError("This scale probe requires a 1024-square baseline")
            source_dir = args.output / "samples" / row["sample_id"]
            source_dir.mkdir(parents=True)
            (source_dir / "source_sample.json").write_bytes(source)
            for canvas, width_mode in render_cases:
                name = f"{canvas}_{width_mode}"
                folder = source_dir / name
                folder.mkdir()
                drawing = scaled_drawing(original, canvas, width_mode)
                masks = render_masks(drawing)
                support = np.logical_or.reduce([masks[k] > 0 for k in DEFAULT_TOPOLOGY_MASKS])
                semantics = set().union(*(MASK_GROUPS[k] for k in DEFAULT_TOPOLOGY_MASKS))
                clean = render_clean(drawing)
                reference_free = render_clean(replace(drawing, primitives_visible=[p for p in drawing.primitives_visible if p.semantic in semantics]))
                np.savez_compressed(folder / "masks.npz", **masks)
                target_payload = relabel_puhachov_payload(_npz(puhachov_arrays(drawing, clean)), _npz(masks),
                                                        label_contract=REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT)
                (folder / "clean_target.npz").write_bytes(target_payload)
                with np.load(io.BytesIO(target_payload), allow_pickle=False) as arrays:
                    target = arrays["skeleton"]
                    audit = audit_c2(target, arrays["kps"], masks)
                if any(audit[k] for k in ("missing_topology_labels", "extra_topology_labels", "duplicate_topology_labels", "unsupported_skeleton_pixels")):
                    raise ValueError(f"Clean target contract failed: {row['sample_id']}/{name}")
                _save_png(folder / "clean_target.png", target)
                _save_png(folder / "clean.png", clean)
                _save_png(folder / "reference_free_clean.png", reference_free)
                inputs = acquisition_variants(reference_free, support, drawing.seed + 701, base)
                full = acquisition_variants(clean, np.logical_or.reduce([m > 0 for m in masks.values()]), drawing.seed + 701, base)
                if canvas == 1024:
                    with Image.open(io.BytesIO(read("reference_free_degraded.png"))) as image:
                        if not np.array_equal(inputs["gray_default"], np.asarray(image.convert("L"))):
                            raise ValueError("Rerender differs from frozen baseline")
                group = {"sample_id": row["sample_id"], "difficulty": row["difficulty"],
                         "render_case": name, "target_metrics": metrics(target), "target_audit": audit, "variants": {}}
                for variant, raster in inputs.items():
                    destination = folder / variant
                    destination.mkdir()
                    _save_png(destination / "reference_free.png", raster)
                    _save_png(destination / "full.png", full[variant])
                    result = stage1.run(destination / "reference_free.png", destination, "probe", config, model=model)
                    expected_route = "passthrough_binary" if variant.startswith("binary_") else "sketchcleannet"
                    if result.model_used != expected_route:
                        raise ValueError(f"Unexpected Stage 1 route: {variant}/{result.model_used}")
                    actual = cv2.imread(str(result.skeleton_path), cv2.IMREAD_GRAYSCALE)
                    if actual is None:
                        raise ValueError("Unreadable Stage 1 output")
                    if variant == "binary_c2_control" and not np.array_equal(actual, target):
                        raise ValueError("Binary positive control changed the clean C2 skeleton")
                    values = {**metrics(actual), **skeleton_fidelity(actual, target)}
                    if canvas != 1024:
                        values.update({key.replace("at2px", "at4px"): value
                                       for key, value in skeleton_fidelity(actual, target, tolerance=4).items()})
                    group["variants"][variant] = {"route": result.model_used, "metrics": values,
                                                  "input_sha256": digest(destination / "reference_free.png"),
                                                  "skeleton_sha256": digest(result.skeleton_path)}
                    measurements.append({"domain": name, "difficulty": row["difficulty"],
                                         "sample_id": row["sample_id"], "metrics": {variant: values}})
                groups.append(group)
                if row["difficulty"] not in shown:
                    gallery_groups.append(group)
            shown.add(row["difficulty"])
        (args.output / "progress.json").write_text(json.dumps({"completed_samples": position, "total_samples": len(selected)}) + "\n")
        print(f"Acquisition probe {position}/{len(selected)}: {row['sample_id']}", flush=True)
    report = {**plan, "completed_samples": len(selected), "stage1_passes": len(measurements),
              "clean_target_audits": len(groups), "summaries": summarize(measurements), "groups": groups}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    gallery(args.output, gallery_groups)
    print(json.dumps({"completed_samples": len(selected), "stage1_passes": len(measurements),
                      "report": str(args.output / "report.json"), "gallery": str(args.output / "index.html")}))


if __name__ == "__main__":
    main()
