"""Independent retained-data checks for a PatentVec release partition."""

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image

from stage1_preprocessing.stage1_preprocess import _thin_to_skeleton
from syntheticData.measure_patent_like import audit_c2, digest, metrics, skeleton_fidelity, summarize
from syntheticData.patentvec.annotations import annotation_failures
from syntheticData.patentvec.schema import CanonicalDrawing, validate_drawing
from syntheticData.patentvec.stage2_targets import reference_free_skeleton, REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT


POLICY = {"version": "patent-like-dataset-engineering-v1",
          "median_fidelity_min": .99, "p10_fidelity_min": .98,
          "tiny_component_median_range": [.15, .50],
          "native_cn_median_range": [23.8, 55.0],
          "scope": "Dataset integrity and coarse domain screens, not model transfer or deployment acceptance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    index = json.loads(args.source_index.read_text())
    if digest(args.source_index) != manifest["settings"]["source_index_sha256"]:
        raise ValueError("Source index differs from generation")
    allowed = {(e["dataset"], e["source_id"], e["component_key"]): e["source_identity"] for e in index["entries"]}
    rows, totals, class_counts, identities, ids = [], Counter(), Counter(), set(), set()
    for shard in manifest["shards"]:
        path = args.dataset / shard["archive"]
        if digest(path) != shard["sha256"]:
            raise ValueError(f"Corrupt shard {path}")
        with tarfile.open(path) as archive:
            prefixes = [m.name.rsplit("/", 1)[0] for m in archive.getmembers() if m.name.endswith("/sample.json")]
            for prefix in prefixes:
                def read(name):
                    stream = archive.extractfile(prefix + "/" + name)
                    if stream is None:
                        raise ValueError(f"Missing {prefix}/{name}")
                    return stream.read()
                drawing = CanonicalDrawing.from_dict(json.loads(read("sample.json")))
                if drawing.sample_id in ids or drawing.split != index["split"]:
                    raise ValueError("Duplicate sample or wrong partition")
                ids.add(drawing.sample_id)
                failures = validate_drawing(drawing) + annotation_failures(drawing)
                if failures:
                    raise ValueError(f"{drawing.sample_id}: {failures}")
                for component in drawing.components:
                    if component.source_dataset in {"generated", "PatentVec-Procedural"}:
                        continue
                    key = (component.source_dataset, component.source_sample_id, component.source_component_id)
                    if key not in allowed or component.source_split != index["split"]:
                        raise ValueError(f"Source partition escape: {key}")
                    identities.add(allowed[key])
                with np.load(io.BytesIO(read("masks.npz")), allow_pickle=False) as values:
                    masks = {key: values[key] for key in values.files}
                with np.load(io.BytesIO(read("puhachov.npz")), allow_pickle=False) as values:
                    target, points = values["skeleton"], values["kps"]
                    if json.loads(str(values["meta"]))["label_contract"] != REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT:
                        raise ValueError("Wrong Stage 2 label contract")
                checked = audit_c2(target, points, masks)
                if any(checked[k] for k in ("missing_topology_labels", "extra_topology_labels", "duplicate_topology_labels", "unsupported_skeleton_pixels")):
                    raise ValueError(f"Invalid C2 target: {checked}")
                totals.update(checked)
                with np.load(io.BytesIO(read("free2cad_edges.npz")), allow_pickle=False) as values:
                    class_counts.update(map(int, values["types"]))
                    if not np.all(values["topology_projected"]):
                        raise ValueError("Stage 3 targets were not topology projected")
                rasters = {}
                for name in ("clean", "degraded", "reference_free_clean", "reference_free_degraded"):
                    with Image.open(io.BytesIO(read(name + ".png"))) as image:
                        raster = np.asarray(image.convert("L"))
                    if raster.shape != target.shape:
                        raise ValueError("Raster/target dimensions differ")
                    rasters[name] = raster
                observed = rasters["reference_free_degraded"]
                if not np.isin(observed, [0, 255]).all() or np.mean(observed == 0) >= .5:
                    raise ValueError("Expected black-ink binary acquisition")
                actual = _thin_to_skeleton(255-observed)
                views = {"clean_c2": metrics(target),
                         "binary_stage1_equivalent": {**metrics(actual), **skeleton_fidelity(actual, target)}}
                for name, semantics in (("structural", ("object", "hidden_center")), ("hatch", ("hatch",))):
                    semantic = reference_free_skeleton(masks, target.shape, support_masks=semantics)
                    views[name] = {**metrics(semantic), "observed_recall_at2px": skeleton_fidelity(actual, semantic)["c2_recall_at2px"]}
                rows.append({"sample_id": drawing.sample_id, "domain": index["split"],
                             "difficulty": drawing.difficulty, "metrics": views})
        print(f"Audited {len(rows)}/{manifest['count']}", flush=True)
    if len(rows) != manifest["count"] or set(class_counts) != set(range(5)):
        raise ValueError("Incomplete sample/class coverage")
    observed = [r["metrics"]["binary_stage1_equivalent"] for r in rows]
    medians = {key: float(np.median([r[key] for r in observed])) for key in observed[0]}
    p10 = float(np.quantile([r["c2_f1_at2px"] for r in observed], .1))
    screens = {"fidelity_median": medians["c2_f1_at2px"] >= POLICY["median_fidelity_min"],
               "fidelity_p10": p10 >= POLICY["p10_fidelity_min"],
               "tiny_components": POLICY["tiny_component_median_range"][0] <= medians["tiny_component_fraction"] <= POLICY["tiny_component_median_range"][1],
               "native_cn": POLICY["native_cn_median_range"][0] <= medians["cn_junction_clusters_per_1k"] <= POLICY["native_cn_median_range"][1]}
    report = {"count": len(rows), "dataset": str(args.dataset.resolve()), "manifest_sha256": digest(args.dataset / "manifest.json"),
              "source_index_sha256": digest(args.source_index), "contract_totals": dict(totals),
              "source_identities": sorted(identities), "class_counts": dict(class_counts),
              "policy": POLICY, "screens": screens, "engineering_screens_pass": all(screens.values()),
              "integrity_pass": True, "stage1_equivalence": "binary minority-ink polarity normalization and production thinning; no neural inference",
              "stage1_medians": medians, "fidelity_p10": p10,
              "summaries": summarize(rows), "rows": rows, "model_transfer_accepted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"count": len(rows), "medians": medians, "screens": screens, "totals": dict(totals)}, indent=2))


if __name__ == "__main__":
    main()
