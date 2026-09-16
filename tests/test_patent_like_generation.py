import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from syntheticData.generate_dataset import _record_generation, _sample_specs
from syntheticData.measure_patent_like import audit_c2, metrics, skeleton_fidelity
from syntheticData.probe_patent_acquisition import acquisition_variants, scaled_drawing
from syntheticData import build_dataset_audit
from syntheticData.patentvec.composition import CompositionBuilder
from syntheticData.patentvec.generator import PreparedDrawing, compact_sample_payload
from syntheticData.patentvec.quality import evaluate_quality
from syntheticData.patentvec.render import degradation_parameters, degrade_patent_scan, render_clean, render_masks
from syntheticData.patentvec.sources import procedural_polyline_component
from syntheticData.patentvec.stage2_targets import REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT


def test_balanced_probe_and_audits_cover_all_tiers():
    args = SimpleNamespace(curriculum="balanced", count=300, split="train", seed=123,
                           audit_every=100, audit_strategy="stratified")
    specs = _sample_specs(args)
    for difficulty in ("medium", "hard", "very_hard"):
        tier = [s for s in specs if s["difficulty"] == difficulty]
        assert len(tier) == 100
        assert sum(s["audit"] for s in tier) == 1
    assert len({s["seed"] for s in specs}) == 300


def test_generation_identity_refuses_overwrite_and_preserves_existing_plan(tmp_path):
    _record_generation(tmp_path, {"run_fingerprint": "a", "settings": {"retain": True}})
    before = (tmp_path / "generation.json").read_bytes()
    _record_generation(tmp_path, {"run_fingerprint": "a"})
    with pytest.raises(ValueError, match="identity changed"):
        _record_generation(tmp_path, {"run_fingerprint": "b"})
    assert (tmp_path / "generation.json").read_bytes() == before


def test_unidentified_existing_shards_are_not_adopted(tmp_path):
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards/shard_000000.tar").write_bytes(b"previous data")
    with pytest.raises(ValueError, match="no generation identity"):
        _record_generation(tmp_path, {"run_fingerprint": "a"})


@pytest.mark.parametrize("overrides", [{"dark_speckle_probability": -0.1},
                                       {"light_speckle_probability": 2},
                                       {"noise_std": float("nan")},
                                       {"blur_sigma_min": 0.9}, {"typo": 1}])
def test_invalid_degradation_rejected(overrides):
    with pytest.raises(ValueError):
        degradation_parameters(overrides)


def test_degradation_defaults_are_explicit_and_deterministic():
    clean = np.full((128, 128), 255, np.uint8)
    clean[64, 16:112] = 0
    assert np.array_equal(degrade_patent_scan(clean, 7),
                          degrade_patent_scan(clean, 7, degradation_parameters()))
    assert np.array_equal(degrade_patent_scan(clean, 7), degrade_patent_scan(clean, 7))


def test_non_audit_retains_full_and_reference_free_rasters_and_c2_labels(tmp_path, monkeypatch):
    component = procedural_polyline_component("retained", "train", seed=61, primitive_count=3)
    builder = CompositionBuilder("retained", seed=61, canvas=(256, 256))
    builder.add_base(component, scale=0.5)
    drawing = builder.finalize()
    clean, masks = render_clean(drawing), render_masks(drawing)
    quality = evaluate_quality(drawing, clean=clean, masks=masks)
    payload, row = compact_sample_payload(PreparedDrawing(drawing, clean, masks, quality),
                                          0, retain_rasters=True,
                                          label_contract=REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT)
    assert not row["audit"]
    assert "preview.png" not in payload
    for name in ("clean.png", "degraded.png", "reference_free_clean.png", "reference_free_degraded.png"):
        assert name in payload and name in row["raster_files"]
    sample = json.loads(payload["sample.json"])
    assert sample["images"]["reference_free_clean"] == "reference_free_clean.png"
    assert sample["processing"]["rendering"]["degradation_parameters"] == degradation_parameters()
    with np.load(io.BytesIO(payload["puhachov.npz"]), allow_pickle=False) as labels:
        meta = json.loads(str(labels["meta"]))
        assert meta["label_contract"] == REFERENCE_FREE_RASTER_TOPOLOGY_CONTRACT
        assert meta["reference_free_input"]
        support = masks["object"] | masks["hidden_center"] | masks["hatch"]
        assert not np.any((labels["skeleton"] > 0) & (support == 0))
    with tarfile.open(tmp_path / "sample.tar", "w") as archive:
        for name, value in payload.items():
            member = tarfile.TarInfo("sample/" + name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    def fail_rerender(*args, **kwargs):
        raise AssertionError("Retained rasters must take precedence")
    monkeypatch.setattr(build_dataset_audit, "render_clean", fail_rerender)
    monkeypatch.setattr(build_dataset_audit, "degrade_patent_scan", fail_rerender)
    previews = build_dataset_audit._extract_previews(tmp_path, tmp_path, [
        {**row, "archive": "sample.tar", "member_prefix": "sample"}])
    assert previews[0]["preview_source"] == "retained_rasters"


def test_cn_clusters_are_not_raw_degree_junction_pixels():
    ink = np.zeros((25, 25), np.uint8)
    ink[12, 5:20] = 255
    ink[5:20, 12] = 255
    measured = metrics(ink)
    assert measured["cn_junction_clusters"] == 1
    assert measured["cn_endpoint_clusters"] == 4
    assert measured["junction_pixels_per_1k"] > measured["cn_junction_clusters_per_1k"]


def test_exact_topology_audit_catches_nearby_missing_and_duplicate_labels():
    ink = np.zeros((25, 25), np.uint8)
    ink[12, 8:17] = 255
    ink[8:17, 12] = 255
    points = np.array([[8, 12, 0], [16, 12, 0], [12, 8, 0], [12, 16, 0], [12, 12, 1]])
    masks = {name: ink for name in ("object", "hidden_center", "hatch")}
    assert audit_c2(ink, points, masks)["missing_topology_labels"] == 0
    assert audit_c2(ink, points[1:], masks)["missing_topology_labels"] == 1
    duplicate = np.concatenate([points, points[:1]])
    assert audit_c2(ink, duplicate, masks)["duplicate_topology_labels"] == 1
    with pytest.raises(ValueError, match="Invalid C2"):
        audit_c2(ink, np.array([[2, 2, 99]]), masks)


def test_preprocessing_fidelity_detects_lost_source_ink():
    target = np.zeros((25, 25), np.uint8)
    target[12, 2:23] = 255
    actual = target.copy()
    assert skeleton_fidelity(actual, target)["c2_f1_at2px"] == 1.0
    actual[:, 12:] = 0
    result = skeleton_fidelity(actual, target)
    assert result["c2_precision_at2px"] == 1.0
    assert result["c2_recall_at2px"] < .7


def test_acquisition_width_probe_does_not_mutate_geometry_or_source():
    component = procedural_polyline_component("scale", "train", seed=62, primitive_count=3)
    builder = CompositionBuilder("scale", seed=62, canvas=(1024, 1024))
    builder.add_base(component, scale=.5)
    drawing = builder.finalize()
    before = drawing.to_dict()
    relative = scaled_drawing(drawing, 2048, "relative")
    fixed = scaled_drawing(drawing, 2048, "fixed_pixel")
    assert relative.canvas == fixed.canvas == [2048, 2048]
    assert drawing.to_dict() == before
    for original, a, b in zip(drawing.primitives_visible, relative.primitives_visible, fixed.primitives_visible):
        assert a.geometry == b.geometry == original.geometry
        assert a.source_sample_id == b.source_sample_id == original.source_sample_id
        assert a.style["stroke_width"] == 2 * b.style["stroke_width"]


def test_binary_acquisition_is_paired_with_its_gray_observation():
    clean = np.full((256, 256), 255, np.uint8)
    clean[128, 8:248] = 0
    support = clean == 0
    variants = acquisition_variants(clean, support, 63, degradation_parameters())
    assert len(variants) == 5
    assert np.array_equal(variants["binary_c2_control"], clean)
    for noise in ("default", "low_speckle"):
        binary, gray = variants["binary_" + noise], variants["gray_" + noise]
        assert set(np.unique(binary)) == {0, 255}
        assert np.array_equal(binary == 0, gray < 210)
