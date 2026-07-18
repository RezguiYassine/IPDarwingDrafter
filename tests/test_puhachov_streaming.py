import numpy as np

from stage2_strokeextraction.research.train_puhachov import (
    ConstantMemoryDistributedSampler,
    ExactMixedDataset,
    KPDataset,
    SOURCE_CACHED,
    SOURCE_SKETCHGRAPHS,
)
from tools.sketchgraphs_coverage import export_rejections, summarize
from tools.evaluate_puhachov_streaming import ExactEvalSampler, _match_counts


class _IndexDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return key


def test_distributed_sampler_covers_dataset_with_bounded_padding():
    dataset = _IndexDataset(11)
    samplers = [
        ConstantMemoryDistributedSampler(dataset, num_replicas=2, rank=rank, seed=7)
        for rank in range(2)
    ]
    partitions = [list(sampler) for sampler in samplers]
    indices = [index for partition in partitions for _epoch, index in partition]

    assert [len(partition) for partition in partitions] == [6, 6]
    assert set(indices) == set(range(11))
    assert len(indices) - len(set(indices)) == 1


def test_distributed_sampler_resume_offset_is_per_rank():
    dataset = _IndexDataset(12)
    sampler = ConstantMemoryDistributedSampler(
        dataset, num_replicas=2, rank=1, seed=13,
    )
    complete = list(sampler)
    sampler.set_start_index(3)

    assert list(sampler) == complete[3:]


def test_exact_mix_contains_every_sketchgraphs_index_once():
    mixed = ExactMixedDataset(
        _IndexDataset(5), _IndexDataset(11), primary_fraction=0.3, seed=42,
    )
    sources = [mixed.source_for_index(i) for i in range(len(mixed))]
    sketch_indices = [index for source, index in sources
                      if source == SOURCE_SKETCHGRAPHS]
    primary_indices = [index for source, index in sources
                       if source == SOURCE_CACHED]

    assert len(mixed) == 16
    assert sketch_indices == list(range(11))
    assert len(primary_indices) == 5
    assert set(primary_indices) == set(range(5))


def test_cached_augmentation_is_deterministic_for_epoch_and_index(tmp_path):
    path = tmp_path / "sample.npz"
    skeleton = np.zeros((32, 32), dtype=np.uint8)
    skeleton[8:24, 16] = 255
    kps = np.array([[16, 8, 0], [16, 23, 0]], dtype=np.int32)
    np.savez(path, skeleton=skeleton, kps=kps)
    dataset = KPDataset([path], crop=32, sigma=2.0, augment=True, seed=17)

    first = dataset[(3, 0)]
    repeated = dataset[(3, 0)]

    assert np.array_equal(first[0].numpy(), repeated[0].numpy())
    assert np.array_equal(first[1].numpy(), repeated[1].numpy())


def test_coverage_summary_and_rejection_export(tmp_path):
    path = tmp_path / "coverage.i8"
    np.array([0, 1, -1, 0, 3], dtype=np.int8).tofile(path)
    path.with_suffix(".i8.json").write_text(
        '{"source_total": 5, "source_path": "train.npy"}\n'
    )

    report = summarize(path)
    rejects = tmp_path / "rejects.csv"
    count = export_rejections(path, rejects)

    assert report["attempted"] == 4
    assert report["accepted"] == 2
    assert report["rejected"] == 2
    assert not report["complete"]
    assert count == 2
    assert "no_supported_geometry" in rejects.read_text()


def test_exact_eval_sampler_has_no_padding_or_duplicates():
    partitions = [list(ExactEvalSampler(11, rank, 3)) for rank in range(3)]
    flattened = [index for partition in partitions for index in partition]

    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def test_streaming_match_counts_uses_per_class_greedy_matching():
    heatmaps = np.zeros((3, 32, 32), dtype=np.float32)
    heatmaps[0, 5, 4] = 0.9
    heatmaps[0, 20, 20] = 0.8
    heatmaps[1, 10, 9] = 0.95
    keypoints = np.array([[4, 5, 0], [9, 10, 1], [12, 12, 2]], dtype=np.int32)

    tp, fp, fn = _match_counts(
        heatmaps, keypoints, conf=0.3, nms_radius=1, match_radius=2.0,
    )

    np.testing.assert_array_equal(tp, [1, 1, 0])
    np.testing.assert_array_equal(fp, [1, 0, 0])
    np.testing.assert_array_equal(fn, [0, 0, 1])
