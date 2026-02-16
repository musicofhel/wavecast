"""Continuous Wavelet Transform scalogram computation."""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray

from wavecast.core.exceptions import DecompositionError
from wavecast.core.types import TimeSeries


def compute_scalogram(
    data: NDArray | TimeSeries,
    wavelet: str = "morl",
    scales: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute the CWT scalogram (time-frequency representation).

    Args:
        data: Input 1D array or TimeSeries.
        wavelet: CWT wavelet name (e.g. 'morl', 'cmor1.5-1.0', 'mexh').
        scales: Array of scales to use. Defaults to 1..min(128, len/2).

    Returns:
        Tuple of (coefficients, frequencies, scales) where:
        - coefficients: 2D complex/real array of shape (n_scales, n_samples)
        - frequencies: pseudo-frequencies corresponding to each scale
        - scales: the scales used

    Raises:
        DecompositionError: If data is too short or wavelet is invalid.
    """
    if isinstance(data, TimeSeries):
        values = data.values.astype(np.float64)
    else:
        values = np.asarray(data, dtype=np.float64)

    if values.ndim != 1:
        raise DecompositionError(
            f"Expected 1D data, got shape {values.shape}"
        )

    if len(values) < 4:
        raise DecompositionError(
            "Data must have at least 4 samples for CWT"
        )

    if scales is None:
        max_scale = min(128, len(values) // 2)
        if max_scale < 1:
            max_scale = 1
        scales = np.arange(1, max_scale + 1, dtype=np.float64)
    else:
        scales = np.asarray(scales, dtype=np.float64)
        if len(scales) == 0:
            raise DecompositionError("Scales array must not be empty")
        if np.any(scales <= 0):
            raise DecompositionError("All scales must be positive")

    try:
        coefficients, frequencies = pywt.cwt(values, scales, wavelet)
    except Exception as e:
        raise DecompositionError(f"CWT failed with wavelet '{wavelet}': {e}") from e

    return coefficients, frequencies, scales
