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
