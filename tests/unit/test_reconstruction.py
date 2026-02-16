"""Tests for wavelet reconstruction."""

import numpy as np

from wavecast.wavelets.dwt import decompose
from wavecast.wavelets.reconstruction import denoise, reconstruct, reconstruct_level


def test_reconstruct_roundtrip(sine_series):
    decomp = decompose(sine_series, level=3)
    reconstructed = reconstruct(decomp)
    np.testing.assert_array_almost_equal(
        reconstructed[:sine_series.length],
        sine_series.values,
        decimal=10,
    )


def test_reconstruct_level(sine_series):
    decomp = decompose(sine_series, level=3)
    # Reconstruct only level 1 detail
    result = reconstruct_level(decomp, level=1)
    assert len(result) >= sine_series.length


def test_denoise():
    rng = np.random.default_rng(42)
    clean = np.sin(np.linspace(0, 4 * np.pi, 256))
    noisy = clean + 0.5 * rng.standard_normal(256)
    denoised = denoise(noisy, wavelet="db4", level=3)
    # Denoised should be closer to clean than noisy
    noise_error = np.mean((noisy - clean) ** 2)
    denoised_error = np.mean((denoised[:len(clean)] - clean) ** 2)
    assert denoised_error < noise_error
