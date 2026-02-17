"""Sequence dataset builder for training sequence models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from wavecast.core.types import MultiLevelTokenSequence
from wavecast.tokenizer.vocabulary import PAD_ID, SAXVocabulary


@dataclass
class SequenceSample:
    """A single training sample for the sequence model."""

    context_tokens: list[int]
    target_token: int
    level: int
    asset_class_id: int
    targets: dict[int, int] = field(default_factory=dict)
    # Metadata for return target computation (populated by build_return_target_dataset)
    token_position: int = 0
    n_coeffs: int = 0
    n_symbols: int = 0


@dataclass
class SequenceDataset:
    """Collection of sequence samples."""

    samples: list[SequenceSample] = field(default_factory=list)

    def to_arrays(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Convert to numpy arrays for training.

        Returns:
            Tuple of (contexts, targets, levels, asset_classes) arrays.
        """
        if not self.samples:
            return (
                np.empty((0, 0), dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        contexts = np.array(
            [s.context_tokens for s in self.samples], dtype=np.int64
        )
        targets = np.array([s.target_token for s in self.samples], dtype=np.int64)
        levels = np.array([s.level for s in self.samples], dtype=np.int64)
        asset_classes = np.array(
            [s.asset_class_id for s in self.samples], dtype=np.int64
        )

        return contexts, targets, levels, asset_classes

    def to_multi_horizon_arrays(
        self, horizons: list[int]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Convert to numpy arrays with multi-horizon targets.

        Args:
            horizons: List of prediction horizons (e.g. [1, 2, 4, 8]).

        Returns:
            Tuple of (contexts, targets_2d, levels, asset_classes) where
            targets_2d has shape (n_samples, len(horizons)) with one column
            per horizon.
        """
        if not self.samples:
            return (
                np.empty((0, 0), dtype=np.int64),
                np.empty((0, len(horizons)), dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        contexts = np.array(
            [s.context_tokens for s in self.samples], dtype=np.int64
        )
        # Build 2-D target array: one column per horizon
        targets_2d = np.zeros(
            (len(self.samples), len(horizons)), dtype=np.int64
        )
        for row_idx, sample in enumerate(self.samples):
            for col_idx, h in enumerate(horizons):
                targets_2d[row_idx, col_idx] = sample.targets.get(
                    h, sample.target_token if h == 1 else PAD_ID
                )

        levels = np.array([s.level for s in self.samples], dtype=np.int64)
        asset_classes = np.array(
            [s.asset_class_id for s in self.samples], dtype=np.int64
        )

        return contexts, targets_2d, levels, asset_classes


def build_sequence_dataset(
    token_sequences: list[MultiLevelTokenSequence],
    vocabulary: SAXVocabulary,
    context_length: int = 32,
    asset_class_map: dict[str, int] | None = None,
    max_horizon: int = 1,
) -> SequenceDataset:
    """Build a sequence dataset from tokenized wavelet decompositions.

    Creates sliding-window (context, target) pairs from each level's token IDs.
    Short sequences are padded with PAD token on the left.

    Args:
        token_sequences: List of multi-level token sequences.
        vocabulary: SAX vocabulary (used for PAD token ID).
        context_length: Number of tokens in each context window.
        asset_class_map: Optional mapping from ticker to asset class ID.
        max_horizon: Maximum prediction horizon (default=1 for backward compat).
            When >1, each sample includes targets for horizons 1..max_horizon.

    Returns:
        SequenceDataset with all samples.
    """
    dataset = SequenceDataset()

    for mlt in token_sequences:
        asset_class_id = (
            asset_class_map.get(mlt.ticker, 0) if asset_class_map else 0
        )

        for level, seq in mlt.level_sequences.items():
            tokens = seq.token_ids
            if len(tokens) == 0:
                continue

            # Pad short sequences on the left
            if len(tokens) <= context_length:
                padded = [PAD_ID] * (context_length + 1 - len(tokens)) + tokens
                tokens = padded

            # Sliding window: context[i:i+ctx_len] -> target[i+ctx_len]
            # With max_horizon, ensure we have room for all horizon targets
            end_idx = len(tokens) - context_length - (max_horizon - 1)
            for i in range(max(0, end_idx)):
                context = tokens[i : i + context_length]
                target = tokens[i + context_length]

                # Build targets dict for all horizons
                targets: dict[int, int] = {}
                for h in range(1, max_horizon + 1):
                    target_idx = i + context_length + (h - 1)
                    if target_idx < len(tokens):
                        targets[h] = tokens[target_idx]

                dataset.samples.append(
                    SequenceSample(
                        context_tokens=context,
                        target_token=target,
                        level=level,
                        asset_class_id=asset_class_id,
                        targets=targets,
                    )
                )

    return dataset


def build_return_target_dataset(
    token_sequences: list[MultiLevelTokenSequence],
    vocabulary: SAXVocabulary,
    context_length: int = 32,
    asset_class_map: dict[str, int] | None = None,
    n_coeffs_map: dict[tuple[str, int], int] | None = None,
    n_symbols_map: dict[tuple[str, int], int] | None = None,
) -> SequenceDataset:
    """Build a sequence dataset with metadata for return target computation.

    Same sliding-window logic as build_sequence_dataset(), but records
    token_position, n_coeffs, and n_symbols per sample so that return
    targets can be computed post-hoc via targets.returns.

    Args:
        token_sequences: List of multi-level token sequences.
        vocabulary: SAX vocabulary (used for PAD token ID).
        context_length: Number of tokens in each context window.
        asset_class_map: Optional mapping from ticker to asset class ID.
        n_coeffs_map: Mapping from (ticker, level) to number of DWT coefficients.
        n_symbols_map: Mapping from (ticker, level) to number of SAX symbols.

    Returns:
        SequenceDataset with metadata fields populated.
    """
    dataset = SequenceDataset()

    for mlt in token_sequences:
        asset_class_id = (
            asset_class_map.get(mlt.ticker, 0) if asset_class_map else 0
        )

        for level, seq in mlt.level_sequences.items():
            tokens = seq.token_ids
            if len(tokens) == 0:
                continue

            nc = (
                n_coeffs_map.get((mlt.ticker, level), 0)
                if n_coeffs_map
                else 0
            )
            ns = (
                n_symbols_map.get((mlt.ticker, level), 0)
                if n_symbols_map
                else 0
            )

            # Pad short sequences on the left
            pad_offset = 0
            if len(tokens) <= context_length:
                pad_count = context_length + 1 - len(tokens)
                padded = [PAD_ID] * pad_count + tokens
                pad_offset = pad_count
                tokens = padded

            # Sliding window: context[i:i+ctx_len] -> target[i+ctx_len]
            end_idx = len(tokens) - context_length
            for i in range(max(0, end_idx)):
                context = tokens[i : i + context_length]
                target = tokens[i + context_length]
                # token_position in the original (unpadded) symbol space
                token_pos = i + context_length - pad_offset

                dataset.samples.append(
                    SequenceSample(
                        context_tokens=context,
                        target_token=target,
                        level=level,
                        asset_class_id=asset_class_id,
                        token_position=token_pos,
                        n_coeffs=nc,
                        n_symbols=ns,
                    )
                )

    return dataset
