"""Tests for sequence dataset builder."""

import numpy as np

from wavecast.core.types import MultiLevelTokenSequence, TokenSequence
from wavecast.tokenizer.dataset import SequenceSample, build_sequence_dataset
from wavecast.tokenizer.vocabulary import PAD_ID, SAXVocabulary


def _make_token_sequence(n_tokens=50, n_levels=3, ticker="TEST"):
    """Create a deterministic multi-level token sequence."""
    rng = np.random.default_rng(42)
    level_sequences = {}
    for lvl in range(1, n_levels + 1):
        token_ids = rng.integers(2, 20, size=n_tokens).tolist()
        words = [f"w{t}" for t in token_ids]
        level_sequences[lvl] = TokenSequence(
            token_ids=token_ids,
            words=words,
            ticker=ticker,
            interval="1d",
            wavelet_level=lvl,
        )
    return MultiLevelTokenSequence(
        ticker=ticker, interval="1d", level_sequences=level_sequences
    )


def test_dataset_to_arrays_shape():
    seq = _make_token_sequence(n_tokens=50, n_levels=2)
    vocab = SAXVocabulary()
    dataset = build_sequence_dataset([seq], vocab, context_length=10)
    contexts, targets, levels, asset_classes = dataset.to_arrays()

    # Each level has 50 tokens, context=10, so 40 samples per level, 2 levels = 80
    assert contexts.shape[0] == 80
    assert contexts.shape[1] == 10
    assert len(targets) == 80
    assert len(levels) == 80
    assert len(asset_classes) == 80


def test_dataset_context_length():
    seq = _make_token_sequence(n_tokens=50, n_levels=1)
    vocab = SAXVocabulary()
    dataset = build_sequence_dataset([seq], vocab, context_length=16)
    contexts, _, _, _ = dataset.to_arrays()
    assert contexts.shape[1] == 16


def test_dataset_short_sequence_gets_padded():
    level_sequences = {
        1: TokenSequence(
            token_ids=[5, 6, 7],
            words=["w5", "w6", "w7"],
            ticker="T",
            interval="1d",
            wavelet_level=1,
        ),
    }
    mlt = MultiLevelTokenSequence(ticker="T", interval="1d", level_sequences=level_sequences)
    vocab = SAXVocabulary()
    dataset = build_sequence_dataset([mlt], vocab, context_length=4)
    contexts, targets, _, _ = dataset.to_arrays()

    assert contexts.shape[0] >= 1
    # First context should contain PAD tokens
    assert PAD_ID in contexts[0]


def test_dataset_asset_class_map():
    seq = _make_token_sequence(n_tokens=50, n_levels=1, ticker="AAPL")
    vocab = SAXVocabulary()
    asset_map = {"AAPL": 2}
    dataset = build_sequence_dataset([seq], vocab, context_length=10, asset_class_map=asset_map)
    _, _, _, asset_classes = dataset.to_arrays()
    assert np.all(asset_classes == 2)


def test_dataset_empty_sequence():
    level_sequences = {
        1: TokenSequence(
            token_ids=[], words=[], ticker="T", interval="1d", wavelet_level=1
        ),
    }
    mlt = MultiLevelTokenSequence(ticker="T", interval="1d", level_sequences=level_sequences)
    vocab = SAXVocabulary()
    dataset = build_sequence_dataset([mlt], vocab, context_length=10)
    contexts, targets, levels, asset_classes = dataset.to_arrays()
    assert contexts.shape[0] == 0


def test_sequence_sample_fields():
    sample = SequenceSample(
        context_tokens=[1, 2, 3], target_token=4, level=1, asset_class_id=0
    )
    assert sample.context_tokens == [1, 2, 3]
    assert sample.target_token == 4


def test_multiple_sequences():
    seq1 = _make_token_sequence(n_tokens=30, n_levels=1, ticker="A")
    seq2 = _make_token_sequence(n_tokens=30, n_levels=1, ticker="B")
    vocab = SAXVocabulary()
    dataset = build_sequence_dataset([seq1, seq2], vocab, context_length=10)
    contexts, _, _, _ = dataset.to_arrays()
    # Each sequence: 30 tokens, 1 level, ctx=10 => 20 samples each => 40 total
    assert contexts.shape[0] == 40


# --- Multi-horizon dataset tests ---


def test_dataset_max_horizon_basic():
    """max_horizon=1 produces same results as default."""
    seq = _make_token_sequence(n_tokens=50, n_levels=1)
    vocab = SAXVocabulary()
    ds_default = build_sequence_dataset([seq], vocab, context_length=10)
    ds_h1 = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=1)
    assert len(ds_default.samples) == len(ds_h1.samples)


def test_dataset_max_horizon_reduces_samples():
    """Higher max_horizon produces fewer samples (more tokens needed per window)."""
    seq = _make_token_sequence(n_tokens=50, n_levels=1)
    vocab = SAXVocabulary()
    ds_h1 = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=1)
    ds_h4 = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=4)
    ds_h8 = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=8)
    # More horizons need more trailing tokens -> fewer windows
    assert len(ds_h1.samples) > len(ds_h4.samples)
    assert len(ds_h4.samples) > len(ds_h8.samples)


def test_dataset_max_horizon_targets_populated():
    """Each sample has targets for all horizons 1..max_horizon."""
    seq = _make_token_sequence(n_tokens=50, n_levels=1)
    vocab = SAXVocabulary()
    max_h = 4
    ds = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=max_h)
    for sample in ds.samples:
        # All horizons 1..max_h should have targets
        for h in range(1, max_h + 1):
            assert h in sample.targets
        # target_token should match horizon=1 target
        assert sample.target_token == sample.targets[1]


def test_dataset_max_horizon_no_out_of_bounds():
    """Targets never reference out-of-bounds token indices."""
    seq = _make_token_sequence(n_tokens=20, n_levels=1)
    vocab = SAXVocabulary()
    ds = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=8)
    # With 20 tokens and ctx=10, max index is 19.
    # Each sample's highest horizon target should be within bounds.
    for sample in ds.samples:
        for _h, target_id in sample.targets.items():
            assert 0 <= target_id < 20  # token IDs are 2-19 from fixture


def test_dataset_to_multi_horizon_arrays():
    """to_multi_horizon_arrays produces correct 2-D target shape."""
    seq = _make_token_sequence(n_tokens=50, n_levels=1)
    vocab = SAXVocabulary()
    horizons = [1, 2, 4, 8]
    ds = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=8)
    contexts, targets_2d, levels, asset_classes = ds.to_multi_horizon_arrays(horizons)
    assert targets_2d.shape == (len(ds.samples), len(horizons))
    # Column 0 should match target_token (horizon=1)
    for i, sample in enumerate(ds.samples):
        assert targets_2d[i, 0] == sample.target_token


def test_dataset_max_horizon_exact_count():
    """Verify exact sample count with max_horizon."""
    # 50 tokens, ctx=10, max_horizon=4
    # end_idx = 50 - 10 - (4-1) = 37
    # Samples = 37
    seq = _make_token_sequence(n_tokens=50, n_levels=1)
    vocab = SAXVocabulary()
    ds = build_sequence_dataset([seq], vocab, context_length=10, max_horizon=4)
    assert len(ds.samples) == 37
