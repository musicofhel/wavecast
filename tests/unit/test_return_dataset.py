"""Tests for build_return_target_dataset."""

from wavecast.core.types import MultiLevelTokenSequence, TokenSequence
from wavecast.tokenizer.dataset import (
    build_return_target_dataset,
    build_sequence_dataset,
)
from wavecast.tokenizer.vocabulary import SAXVocabulary


def _make_vocab(n_words=10) -> SAXVocabulary:
    """Create a small vocabulary for testing."""
    words = [f"w{i}" for i in range(n_words)]
    return SAXVocabulary.from_corpus([words], min_freq=1, max_size=n_words + 10)


def _make_mlt(ticker="AAPL", n_tokens=30, level=1, vocab=None) -> MultiLevelTokenSequence:
    """Create a simple MLT for testing."""
    if vocab is None:
        vocab = _make_vocab()
    # Use vocab token IDs (skip PAD=0 and UNK=1)
    token_ids = list(range(2, min(2 + n_tokens, vocab.size)))
    while len(token_ids) < n_tokens:
        token_ids.extend(token_ids[:n_tokens - len(token_ids)])
    token_ids = token_ids[:n_tokens]

    return MultiLevelTokenSequence(
        ticker=ticker,
        interval="1h",
        level_sequences={
            level: TokenSequence(
                token_ids=token_ids,
                words=[f"w{i}" for i in range(n_tokens)],
                ticker=ticker,
                interval="1h",
                wavelet_level=level,
            )
        },
    )


def test_build_return_target_dataset_metadata():
    """Return target dataset records token_position, n_coeffs, n_symbols."""
    vocab = _make_vocab()
    mlt = _make_mlt(n_tokens=30, level=2, vocab=vocab)

    n_coeffs_map = {("AAPL", 2): 100}
    n_symbols_map = {("AAPL", 2): 80}

    ds = build_return_target_dataset(
        [mlt], vocab, context_length=8,
        n_coeffs_map=n_coeffs_map, n_symbols_map=n_symbols_map,
    )
    assert len(ds.samples) > 0
    # Check metadata is populated
    for s in ds.samples:
        assert s.n_coeffs == 100
        assert s.n_symbols == 80
        assert s.token_position >= 0


def test_build_return_target_dataset_positions_increase():
    """Token positions increase within a level's samples."""
    vocab = _make_vocab()
    mlt = _make_mlt(n_tokens=30, level=1, vocab=vocab)

    ds = build_return_target_dataset(
        [mlt], vocab, context_length=8,
        n_coeffs_map={("AAPL", 1): 50},
        n_symbols_map={("AAPL", 1): 50},
    )
    positions = [s.token_position for s in ds.samples]
    assert positions == sorted(positions)


def test_build_return_target_dataset_same_contexts():
    """Return target dataset produces same context windows as regular dataset."""
    vocab = _make_vocab()
    mlt = _make_mlt(n_tokens=30, level=1, vocab=vocab)

    regular = build_sequence_dataset([mlt], vocab, context_length=8)
    return_ds = build_return_target_dataset(
        [mlt], vocab, context_length=8,
        n_coeffs_map={("AAPL", 1): 50},
        n_symbols_map={("AAPL", 1): 50},
    )

    # Same number of samples (max_horizon=1 for both)
    assert len(regular.samples) == len(return_ds.samples)

    # Same context tokens
    for r, rt in zip(regular.samples, return_ds.samples, strict=True):
        assert r.context_tokens == rt.context_tokens


def test_build_return_target_dataset_backward_compat():
    """Original build_sequence_dataset still works unchanged."""
    vocab = _make_vocab()
    mlt = _make_mlt(n_tokens=30, level=1, vocab=vocab)

    ds = build_sequence_dataset([mlt], vocab, context_length=8)
    # Metadata defaults to 0
    for s in ds.samples:
        assert s.token_position == 0
        assert s.n_coeffs == 0
        assert s.n_symbols == 0


def test_build_return_target_dataset_empty():
    """Empty input produces empty dataset."""
    vocab = _make_vocab()
    ds = build_return_target_dataset([], vocab, context_length=8)
    assert len(ds.samples) == 0
