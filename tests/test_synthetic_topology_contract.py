import io
import json

import numpy as np

from syntheticData.audit_stage2_topology_contract import audit_sample
from syntheticData.patentvec.stage2_targets import (
    HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
    REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT,
    relabel_puhachov_payload,
    supported_raster_topology_keypoints,
)
from stage2_strokeextraction.stage2_stroke_extract import _cn_keypoint_clusters


def test_hatch_endpoints_are_reported_when_exact_labels_omit_them():
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    skeleton[10, 3:17] = 255
    hatch = skeleton.copy()
    keypoints = np.empty((0, 3), dtype=np.int32)

    report = audit_sample(skeleton, keypoints, {"hatch": hatch})

    assert report["total"]["endpoint"] == 2
    assert report["semantic"]["hatch"]["endpoint_total"] == 2
    assert report["semantic"]["hatch"]["endpoint_unmatched"] == 2
    assert report["skeleton"]["outside_topology_support_pixels"] == 0


def test_nearby_exact_endpoint_marks_hatch_support_as_matched():
    skeleton = np.zeros((20, 20), dtype=np.uint8)
    skeleton[10, 3:17] = 255
    hatch = skeleton.copy()
    keypoints = np.asarray([[3, 10, 0], [16, 10, 0]], dtype=np.int32)

    report = audit_sample(
        skeleton, keypoints, {"hatch": hatch}, match_radius=1.0,
    )

    assert report["semantic"]["hatch"]["endpoint_matched"] == 2
    assert report["semantic"]["hatch"].get("endpoint_unmatched", 0) == 0


def test_supported_raster_topology_matches_production_cn_clusters():
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    skeleton[16, 4:28] = 255
    skeleton[5:17, 16] = 255
    masks = {
        "object": skeleton.copy(),
        "hidden_center": np.zeros_like(skeleton),
        "hatch": np.zeros_like(skeleton),
    }

    keypoints, _stats = supported_raster_topology_keypoints(
        skeleton, np.empty((0, 3), dtype=np.int32), masks, support_radius=0
    )
    production = {
        (item["x"], item["y"], 0 if item["type"] == "endpoint" else 1)
        for item in _cn_keypoint_clusters(skeleton)
    }

    assert set(map(tuple, keypoints.tolist())) == production


def test_supported_raster_topology_ignores_annotation_only_components():
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    skeleton[8, 4:16] = 255
    skeleton[24, 16:28] = 255
    object_mask = np.zeros_like(skeleton)
    object_mask[8, 4:16] = 255
    masks = {
        "object": object_mask,
        "hidden_center": np.zeros_like(skeleton),
        "hatch": np.zeros_like(skeleton),
        "text_numeral": skeleton - object_mask,
    }

    keypoints, stats = supported_raster_topology_keypoints(
        skeleton, np.empty((0, 3), dtype=np.int32), masks, support_radius=0
    )

    assert set(map(tuple, keypoints.tolist())) == {(4, 8, 0), (15, 8, 0)}
    assert stats["endpoint"] == 2
    assert stats["junction"] == 0


def test_supported_raster_topology_adds_hatches_and_preserves_remote_corners():
    skeleton = np.zeros((40, 40), dtype=np.uint8)
    skeleton[10, 4:18] = 255
    skeleton[30, 20:36] = 255
    masks = {
        "object": np.zeros_like(skeleton),
        "hidden_center": np.zeros_like(skeleton),
        "hatch": skeleton.copy(),
    }
    source = np.asarray([[28, 30, 2], [5, 10, 2]], dtype=np.int32)

    keypoints, stats = supported_raster_topology_keypoints(
        skeleton, source, masks, support_radius=0, corner_min_distance=5.0
    )

    assert (28, 30, 2) in set(map(tuple, keypoints.tolist()))
    assert (5, 10, 2) not in set(map(tuple, keypoints.tolist()))
    assert stats == {
        "endpoint": 4,
        "junction": 0,
        "corner": 1,
        "total": 5,
        "source_corner": 2,
        "dropped_corner": 0,
        "suppressed_corner": 1,
    }


def test_relabel_payload_records_contract_and_is_deterministic():
    skeleton = np.zeros((24, 24), dtype=np.uint8)
    skeleton[12, 3:21] = 255
    masks = {
        "object": skeleton,
        "hidden_center": np.zeros_like(skeleton),
        "hatch": np.zeros_like(skeleton),
    }

    def npz_bytes(**arrays):
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()

    source = npz_bytes(
        skeleton=skeleton,
        kps=np.empty((0, 3), dtype=np.int32),
        meta=np.asarray("{}"),
    )
    mask_payload = npz_bytes(**masks)
    first = relabel_puhachov_payload(source, mask_payload)
    second = relabel_puhachov_payload(source, mask_payload)

    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as data:
        metadata = json.loads(str(data["meta"]))
        assert data["kps"].tolist() == [[3, 12, 0], [20, 12, 0]]
    assert metadata["label_contract"] == "supported-raster-topology-v1"
    assert metadata["topology_support_masks"] == [
        "object", "hidden_center", "hatch",
    ]


def test_reference_free_payload_removes_annotation_branch_before_cn_labels():
    source_skeleton = np.zeros((32, 32), dtype=np.uint8)
    source_skeleton[16, 4:28] = 255
    source_skeleton[5:17, 16] = 255
    object_mask = np.zeros_like(source_skeleton)
    object_mask[16, 4:28] = 255
    masks = {
        "object": object_mask,
        "hidden_center": np.zeros_like(source_skeleton),
        "hatch": np.zeros_like(source_skeleton),
        "leader_dimension": source_skeleton - object_mask,
    }

    def npz_bytes(**arrays):
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()

    relabelled = relabel_puhachov_payload(
        npz_bytes(
            skeleton=source_skeleton,
            kps=np.empty((0, 3), dtype=np.int32),
            meta=np.asarray(json.dumps({
                "distractor_semantics_in_skeleton": ["leader"],
            })),
        ),
        npz_bytes(**masks),
        label_contract=REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT,
    )

    with np.load(io.BytesIO(relabelled), allow_pickle=False) as data:
        assert np.count_nonzero(data["skeleton"][:16]) == 0
        assert data["kps"].tolist() == [[4, 16, 0], [27, 16, 0]]
        metadata = json.loads(str(data["meta"]))
    assert metadata["reference_free_input"] is True
    assert metadata["distractor_semantics_in_skeleton"] == []
    assert metadata["removed_distractor_semantics"] == ["leader"]


def test_hatch_suppressed_payload_keeps_hatch_as_unlabelled_distractor():
    source_skeleton = np.zeros((32, 32), dtype=np.uint8)
    object_mask = np.zeros_like(source_skeleton)
    object_mask[16, 4:28] = 255
    hatch_mask = np.zeros_like(source_skeleton)
    hatch_mask[5:27, 16] = 255
    source_skeleton |= object_mask
    source_skeleton |= hatch_mask
    masks = {
        "object": object_mask,
        "hidden_center": np.zeros_like(source_skeleton),
        "hatch": hatch_mask,
    }

    def npz_bytes(**arrays):
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()

    relabelled = relabel_puhachov_payload(
        npz_bytes(
            skeleton=source_skeleton,
            kps=np.empty((0, 3), dtype=np.int32),
            meta=np.asarray("{}"),
        ),
        npz_bytes(**masks),
        label_contract=HATCH_SUPPRESSED_RASTER_TOPOLOGY_CONTRACT,
    )

    with np.load(io.BytesIO(relabelled), allow_pickle=False) as data:
        assert np.count_nonzero(data["skeleton"][:, 16]) == 22
        assert np.count_nonzero(data["topology_skeleton"][:, 16]) == 1
        assert data["kps"].tolist() == [[4, 16, 0], [27, 16, 0]]
        metadata = json.loads(str(data["meta"]))
    assert metadata["input_support_masks"] == [
        "object", "hidden_center", "hatch",
    ]
    assert metadata["topology_support_masks"] == ["object", "hidden_center"]
    assert metadata["negative_topology_semantics"] == ["hatch"]
    assert metadata["hachure_side_layer_only"] is True
