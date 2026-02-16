"""Wavelet reconstruction and denoising."""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import NDArray

from wavecast.core.exceptions import DecompositionError
from wavecast.core.types import WaveletDecomposition


def reconstruct(decomp: WaveletDecomposition) -> NDArray:
    """Reconstruct the original signal from all wavelet coefficients.

    Args:
        decomp: A WaveletDecomposition result.

    Returns:
        1D array of reconstructed values, trimmed to original length.
    """
    reconstructed = pywt.waverec(decomp.coefficients, decomp.wavelet)
    return reconstructed[: decomp.original_length].astype(np.float64)


def reconstruct_level(decomp: WaveletDecomposition, level: int) -> NDArray:
    """Reconstruct the signal component from a single detail level.

    Sets all coefficients except the specified level to zero and reconstructs.

    Args:
        decomp: A WaveletDecomposition result.
        level: Detail level to reconstruct (1=finest, N=coarsest).
            Use 0 for the approximation component.

    Returns:
        1D array of the single-level reconstructed component.

    Raises:
        DecompositionError: If the level is out of range.
    """
    if level < 0 or level > decomp.level:
        raise DecompositionError(
            f"Level must be 0-{decomp.level}, got {level}. "
            f"0=approximation, 1..{decomp.level}=detail levels."
        )

    # Create zeroed coefficient list
    zeroed = [np.zeros_like(c) for c in decomp.coefficients]

    if level == 0:
        # Approximation component
        zeroed[0] = decomp.coefficients[0].copy()
    else:
        # Detail level: index mapping — level 1 (finest) is last in list
        coeff_idx = decomp.level - level + 1
        zeroed[coeff_idx] = decomp.coefficients[coeff_idx].copy()

    reconstructed = pywt.waverec(zeroed, decomp.wavelet)
    return reconstructed[: decomp.original_length].astype(np.float64)


def denoise(
    data: NDArray,
    wavelet: str = "db4",
    level: int = 5,
    threshold_method: str = "soft",
) -> NDArray:
    """Denoise a signal using wavelet thresholding.

    Applies universal (VisuShrink) threshold to detail coefficients.

    Args:
        data: Input 1D signal.
        wavelet: PyWavelets wavelet name.
        level: DWT decomposition depth.
        threshold_method: 'soft' or 'hard' thresholding.

    Returns:
        Denoised 1D array.

    Raises:
        DecompositionError: On invalid parameters.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 1:
        raise DecompositionError(f"Expected 1D data, got shape {data.shape}")
    if len(data) < 2:
        raise DecompositionError("Data must have at least 2 samples")

    if threshold_method not in ("soft", "hard"):
        raise DecompositionError(
            f"threshold_method must be 'soft' or 'hard', got '{threshold_method}'"
        )

    try:
        w = pywt.Wavelet(wavelet)
    except ValueError as e:
        raise DecompositionError(f"Invalid wavelet '{wavelet}': {e}") from e

    max_level = pywt.dwt_max_level(len(data), w.dec_len)
    if level > max_level:
        raise DecompositionError(
            f"Requested level {level} exceeds max level {max_level} "
            f"for data length {len(data)} with wavelet '{wavelet}'"
        )

    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Universal threshold (VisuShrink): sigma * sqrt(2 * ln(N))
    # Estimate noise sigma from finest detail coefficients (MAD estimator)
    finest_detail = coeffs[-1]
    sigma = float(np.median(np.abs(finest_detail)) / 0.6745)
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # Threshold all detail coefficients (keep approximation untouched)
    denoised_coeffs = [coeffs[0]]  # approximation
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode=threshold_method))

    result = pywt.waverec(denoised_coeffs, wavelet)
    return result[: len(data)].astype(np.float64)
