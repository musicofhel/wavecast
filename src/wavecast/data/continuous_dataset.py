"""Continuous coefficient dataset builder for bypass-SAX input mode."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ContinuousWindow:
    """A single sliding window of normalized wavelet coefficients."""

    context_values: list[float]
    level: int
    asset_class_id: int
    token_position: int
    n_coeffs: int
    ticker: str


@dataclass
class ContinuousDataset:
    """Collection of continuous coefficient windows."""

    windows: list[ContinuousWindow] = field(default_factory=list)

    def to_arrays(self) -> tuple[NDArray, NDArray, NDArray]:
        """Convert to numpy arrays (contexts, levels, asset_classes)."""
        if not self.windows:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        contexts = np.array(
            [w.context_values for w in self.windows], dtype=np.float64
        )
        levels = np.array([w.level for w in self.windows], dtype=np.int64)
        asset_classes = np.array(
            [w.asset_class_id for w in self.windows], dtype=np.int64
        )
        return contexts, levels, asset_classes


def build_continuous_dataset(
    coefficient_series: dict[tuple[str, int], NDArray],
    context_length: int = 16,
    asset_class_map: dict[str, int] | None = None,
    normalize: bool = True,
) -> ContinuousDataset:
    """Build sliding windows from continuous wavelet coefficients.

    Args:
        coefficient_series: Mapping from (ticker, level) to 1-D coefficient array.
            Can be raw detail/approximation coefficients, deltas, or any 1-D float series.
        context_length: Number of values per window.
        asset_class_map: Mapping from ticker to asset class ID.
        normalize: If True, z-normalize each series before windowing.

    Returns:
        ContinuousDataset with sliding windows.
    """
    dataset = ContinuousDataset()

    for (ticker, level), coeffs in sorted(coefficient_series.items()):
        if len(coeffs) <= context_length:
            continue

        ac_id = asset_class_map.get(ticker, 0) if asset_class_map else 0
        n_coeffs = len(coeffs)

        # Z-normalize per series
        if normalize:
            std = np.std(coeffs)
            mean = np.mean(coeffs)
            coeffs = (coeffs - mean) / std if std > 1e-10 else coeffs - mean

        # Sliding window
        for i in range(len(coeffs) - context_length):
            window = coeffs[i : i + context_length].tolist()
            dataset.windows.append(
                ContinuousWindow(
                    context_values=window,
                    level=level,
                    asset_class_id=ac_id,
                    token_position=i + context_length,
                    n_coeffs=n_coeffs,
                    ticker=ticker,
                )
            )

    return dataset
