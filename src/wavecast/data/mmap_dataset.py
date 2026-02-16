"""Memory-mapped sequence dataset for zero-copy loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class MMapSequenceDataset(Dataset):
    """Memory-mapped dataset backed by .npy files.

    Stores contexts, targets, levels, and asset_classes as separate .npy files.
    Uses numpy memory-mapping for zero-copy loading.
    """

    def __init__(self, path: Path) -> None:
        path = Path(path)
        self._contexts = np.load(path / "contexts.npy", mmap_mode="r")
        self._targets = np.load(path / "targets.npy", mmap_mode="r")
        self._levels = np.load(path / "levels.npy", mmap_mode="r")
        self._asset_classes = np.load(path / "asset_classes.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self._contexts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        ctx = torch.tensor(self._contexts[idx], dtype=torch.long)
        lvl = torch.tensor(self._levels[idx], dtype=torch.long)
        ac = torch.tensor(self._asset_classes[idx], dtype=torch.long)

        target = self._targets[idx]
        if target.ndim == 0:
            # Single-horizon: scalar target
            tgt = torch.tensor(int(target), dtype=torch.long)
            return ctx, lvl, ac, tgt
        else:
            # Multi-horizon: vector target (one per horizon)
            tgts = tuple(
                torch.tensor(int(target[i]), dtype=torch.long)
                for i in range(len(target))
            )
            return (ctx, lvl, ac, *tgts)

    @staticmethod
    def save(
        path: Path,
        contexts: NDArray,
        targets: NDArray,
        levels: NDArray,
        asset_classes: NDArray,
    ) -> None:
        """Save arrays as .npy files for memory-mapped loading."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "contexts.npy", contexts.astype(np.int64))
        np.save(path / "targets.npy", targets.astype(np.int64))
        np.save(path / "levels.npy", levels.astype(np.int64))
        np.save(path / "asset_classes.npy", asset_classes.astype(np.int64))
