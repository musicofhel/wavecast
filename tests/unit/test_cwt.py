"""Tests for CWT scalogram."""

import numpy as np

from wavecast.wavelets.cwt import compute_scalogram


def test_compute_scalogram_basic():
    data = np.sin(np.linspace(0, 4 * np.pi, 256))
    coeffs, freqs, scales = compute_scalogram(data)
    assert coeffs.ndim == 2
    assert len(freqs) == len(scales)
    assert coeffs.shape[0] == len(scales)
    assert coeffs.shape[1] == len(data)


def test_compute_scalogram_timeseries(sine_series):
    coeffs, freqs, scales = compute_scalogram(sine_series)
    assert coeffs.shape[1] == sine_series.length
