"""SAX vocabulary for token mapping."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from wavecast.core.exceptions import TokenizerError

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


@dataclass
class SAXVocabulary:
    """Maps SAX words to integer token IDs.

    Token 0 = PAD, token 1 = UNK, 2+ = SAX words sorted by frequency descending.
    """

    _word_to_id: dict[str, int] = field(default_factory=dict)
    _id_to_word: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._word_to_id:
            self._word_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
            self._id_to_word = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    @classmethod
    def from_corpus(
        cls,
        word_sequences: list[list[str]],
        min_freq: int = 2,
        max_size: int = 500,
    ) -> SAXVocabulary:
        """Build vocabulary from a corpus of word sequences.

        Args:
            word_sequences: List of word lists (one per document/level).
            min_freq: Minimum frequency for a word to be included.
            max_size: Maximum vocabulary size (excluding PAD/UNK).

        Returns:
            SAXVocabulary instance.
        """
        counter: Counter[str] = Counter()
        for seq in word_sequences:
            counter.update(seq)

        # Filter by min_freq, sort by frequency descending, limit to max_size
        filtered = [
            (word, count)
            for word, count in counter.most_common()
            if count >= min_freq
        ][:max_size]

        vocab = cls()
        for word, _ in filtered:
            token_id = len(vocab._word_to_id)
            vocab._word_to_id[word] = token_id
            vocab._id_to_word[token_id] = word

        return vocab

    def encode(self, word: str) -> int:
        """Encode a word to its token ID. Returns UNK_ID for unknown words."""
        return self._word_to_id.get(word, UNK_ID)

    def decode(self, token_id: int) -> str:
        """Decode a token ID back to its word string."""
        if token_id not in self._id_to_word:
            raise TokenizerError(f"Unknown token ID: {token_id}")
        return self._id_to_word[token_id]

    def encode_sequence(self, words: list[str]) -> list[int]:
        """Encode a list of words to token IDs."""
        return [self.encode(w) for w in words]

    def __len__(self) -> int:
        """Total vocabulary size including PAD and UNK."""
        return len(self._word_to_id)

    @property
    def size(self) -> int:
        """Total vocabulary size including PAD and UNK."""
        return len(self)

    @property
    def words(self) -> list[str]:
        """All words in the vocabulary (excluding special tokens)."""
        return [self._id_to_word[i] for i in range(2, len(self._id_to_word))]

    def save(self, path: Path) -> None:
        """Save vocabulary to a JSON file."""
        data = {
            "word_to_id": self._word_to_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> SAXVocabulary:
        """Load vocabulary from a JSON file."""
        data = json.loads(path.read_text())
        vocab = cls()
        vocab._word_to_id = data["word_to_id"]
        vocab._id_to_word = {int(v): k for k, v in vocab._word_to_id.items()}
        return vocab
