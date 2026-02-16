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


def build_sequence_dataset(
    token_sequences: list[MultiLevelTokenSequence],
    vocabulary: SAXVocabulary,
    context_length: int = 32,
    asset_class_map: dict[str, int] | None = None,
) -> SequenceDataset:
    """Build a sequence dataset from tokenized wavelet decompositions.

    Creates sliding-window (context, target) pairs from each level's token IDs.
    Short sequences are padded with PAD token on the left.

    Args:
        token_sequences: List of multi-level token sequences.
        vocabulary: SAX vocabulary (used for PAD token ID).
        context_length: Number of tokens in each context window.
        asset_class_map: Optional mapping from ticker to asset class ID.

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
            for i in range(len(tokens) - context_length):
                context = tokens[i : i + context_length]
                target = tokens[i + context_length]
                dataset.samples.append(
                    SequenceSample(
                        context_tokens=context,
                        target_token=target,
                        level=level,
                        asset_class_id=asset_class_id,
                    )
                )

    return dataset
