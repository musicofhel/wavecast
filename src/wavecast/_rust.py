"""Rust acceleration module with Python fallback."""
from __future__ import annotations

try:
    from wavecast._wavecast_rs import (
        build_bow_rs,
        build_corpus_tfidf_rs,
        build_sliding_windows_rs,
        encode_batch_rs,
        extract_words_rs,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    # Stubs that will never be called (callers check HAS_RUST first)
    def extract_words_rs(*args, **kwargs): raise NotImplementedError  # type: ignore[misc]
    def build_bow_rs(*args, **kwargs): raise NotImplementedError  # type: ignore[misc]
    def build_corpus_tfidf_rs(*args, **kwargs): raise NotImplementedError  # type: ignore[misc]
    def encode_batch_rs(*args, **kwargs): raise NotImplementedError  # type: ignore[misc]
    def build_sliding_windows_rs(*args, **kwargs): raise NotImplementedError  # type: ignore[misc]
