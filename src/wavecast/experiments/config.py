"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Controls data selection, SAX parameters, model architecture,
    and training hyperparameters.
    """

    name: str
    tickers: list[str]
    interval: str = "1h"
    train_end: str = "2023-12-31"
    test_start: str = "2024-01-01"

    # SAX parameters
    alphabet_size: int = 7
    n_segments: int = 256
    word_length: int = 4
    word_stride: int = 1

    # Vocabulary parameters
    min_word_freq: int = 2
    max_vocab_size: int = 300

    # Model architecture
    context_length: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.0005
    patience: int = 15

    # DWT
    dwt_levels: list[int] | None = None

    # Split mode: "single" (walk-forward), "expanding", "rolling"
    split_mode: str = "single"
    initial_train_size: int | None = None  # expanding: initial train window (index)
    train_window_size: int | None = None  # rolling: fixed train size
    test_window_size: int | None = None  # expanding/rolling: test window size
    step_size: int | None = None  # expanding/rolling: step between splits

    # Multi-asset grouping
    sectors: list[str] | None = None
    cross_sector_training: bool = True
