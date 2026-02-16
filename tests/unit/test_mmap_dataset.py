"""Tests for MMapSequenceDataset."""

import numpy as np
import torch

from wavecast.data.mmap_dataset import MMapSequenceDataset


def _make_mmap_data(rng, n=50, ctx_len=8, vocab=10, n_levels=5, n_ac=4):
    """Generate random data for mmap dataset tests."""
    contexts = rng.integers(0, vocab, size=(n, ctx_len))
    targets = rng.integers(0, vocab, size=n)
    levels = rng.integers(0, n_levels, size=n)
    asset_classes = rng.integers(0, n_ac, size=n)
    return contexts, targets, levels, asset_classes


def test_save_load_roundtrip(tmp_path):
    """Save arrays, load MMapSequenceDataset, verify shapes."""
    rng = np.random.default_rng(42)
    contexts, targets, levels, asset_classes = _make_mmap_data(rng)

    ds_path = tmp_path / "ds"
    MMapSequenceDataset.save(ds_path, contexts, targets, levels, asset_classes)

    ds = MMapSequenceDataset(ds_path)
    assert len(ds) == 50

    # Verify first item matches source arrays
    item = ds[0]
    assert len(item) == 4  # ctx, lvl, ac, tgt
    np.testing.assert_array_equal(item[0].numpy(), contexts[0])
    assert item[1].item() == levels[0]
    assert item[2].item() == asset_classes[0]
    assert item[3].item() == targets[0]


def test_getitem_returns_tensors(tmp_path):
    """Each element is a torch.Tensor."""
    rng = np.random.default_rng(42)
    contexts, targets, levels, asset_classes = _make_mmap_data(rng)

    ds_path = tmp_path / "ds"
    MMapSequenceDataset.save(ds_path, contexts, targets, levels, asset_classes)
    ds = MMapSequenceDataset(ds_path)

    item = ds[0]
    for elem in item:
        assert isinstance(elem, torch.Tensor)


def test_len_matches_data(tmp_path):
    """__len__ returns correct count."""
    rng = np.random.default_rng(42)
    for n in [10, 100, 1]:
        contexts, targets, levels, asset_classes = _make_mmap_data(rng, n=n)
        ds_path = tmp_path / f"ds_{n}"
        MMapSequenceDataset.save(ds_path, contexts, targets, levels, asset_classes)
        ds = MMapSequenceDataset(ds_path)
        assert len(ds) == n


def test_multi_horizon_targets(tmp_path):
    """2D target array returns multiple target tensors per item."""
    rng = np.random.default_rng(42)
    n, ctx_len, vocab = 50, 8, 10
    n_horizons = 3

    contexts = rng.integers(0, vocab, size=(n, ctx_len))
    targets = rng.integers(0, vocab, size=(n, n_horizons))
    levels = rng.integers(0, 5, size=n)
    asset_classes = rng.integers(0, 4, size=n)

    ds_path = tmp_path / "ds_multi"
    MMapSequenceDataset.save(ds_path, contexts, targets, levels, asset_classes)
    ds = MMapSequenceDataset(ds_path)

    item = ds[0]
    # ctx, lvl, ac, tgt0, tgt1, tgt2
    assert len(item) == 3 + n_horizons
    for i in range(n_horizons):
        assert item[3 + i].item() == targets[0, i]
