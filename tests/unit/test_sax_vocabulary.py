"""Tests for SAX vocabulary."""

import pytest

from wavecast.core.exceptions import TokenizerError
from wavecast.tokenizer.vocabulary import PAD_ID, UNK_ID, SAXVocabulary


def test_empty_vocabulary():
    vocab = SAXVocabulary()
    assert vocab.size == 2  # PAD + UNK
    assert vocab.words == []


def test_from_corpus():
    sequences = [
        ["ab", "cd", "ab", "ef"],
        ["cd", "ab", "gh", "cd"],
    ]
    vocab = SAXVocabulary.from_corpus(sequences, min_freq=2, max_size=500)
    # ab(3), cd(3) pass min_freq=2; ef(1), gh(1) don't
    assert vocab.size == 4  # PAD + UNK + ab + cd


def test_encode_decode_roundtrip():
    sequences = [["ab", "cd", "ab", "cd"]]
    vocab = SAXVocabulary.from_corpus(sequences, min_freq=1)
    token_id = vocab.encode("ab")
    assert token_id >= 2  # Not PAD or UNK
    assert vocab.decode(token_id) == "ab"


def test_encode_unknown_word():
    vocab = SAXVocabulary()
    assert vocab.encode("nonexistent") == UNK_ID


def test_decode_invalid_id():
    vocab = SAXVocabulary()
    with pytest.raises(TokenizerError):
        vocab.decode(999)


def test_encode_sequence():
    sequences = [["ab", "cd", "ab"]]
    vocab = SAXVocabulary.from_corpus(sequences, min_freq=1)
    encoded = vocab.encode_sequence(["ab", "cd", "unknown"])
    assert len(encoded) == 3
    assert encoded[2] == UNK_ID


def test_pad_id_is_zero():
    assert PAD_ID == 0


def test_words_property():
    sequences = [["ab", "cd", "ef", "ab", "cd", "ef"]]
    vocab = SAXVocabulary.from_corpus(sequences, min_freq=1)
    words = vocab.words
    assert len(words) == 3
    assert set(words) == {"ab", "cd", "ef"}


def test_save_load_roundtrip(tmp_path):
    sequences = [["ab", "cd", "ef", "ab", "cd"]]
    vocab = SAXVocabulary.from_corpus(sequences, min_freq=1)

    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = SAXVocabulary.load(path)

    assert loaded.size == vocab.size
    for word in vocab.words:
        assert loaded.encode(word) == vocab.encode(word)


def test_max_size():
    sequences = [["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]]
    vocab = SAXVocabulary.from_corpus(sequences, min_freq=1, max_size=3)
    # PAD + UNK + 3 words = 5
    assert vocab.size == 5
