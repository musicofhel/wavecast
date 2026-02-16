"""Integration test: full pipeline from synthetic data through features."""

import numpy as np

from tests.fixtures.generators import make_sine_series
from wavecast.core.config import ShapeletConfig
from wavecast.data.preprocessing import label_returns, log_returns
from wavecast.features.pipeline import FeaturePipeline
from wavecast.fractal.hurst import wavelet_hurst
from wavecast.fractal.self_similarity import cross_scale_similarity
from wavecast.shapelets.discovery import discover_shapelets
from wavecast.shapelets.library import ShapeletLibrary
from wavecast.wavelets.dwt import compute_level_stats, decompose


def test_full_pipeline():
    """Test the complete pipeline: data → decompose → discover → fractal → features."""
    # Stage 1: Synthetic data
    ts = make_sine_series(n=500)
    assert ts.length == 500

    # Stage 2: DWT decomposition
    decomp = decompose(ts, wavelet="db4", level=4)
    assert decomp.level == 4
    stats = compute_level_stats(decomp)
    assert len(stats) == 5  # approx + 4 details

    # Stage 3: Shapelet discovery
    returns_ts = log_returns(ts)
    labels = label_returns(returns_ts)
    config = ShapeletConfig(z_threshold=0.5, min_length=3, ig_min=0.001)
    shapelets = discover_shapelets(decomp, labels, config)
    assert isinstance(shapelets, list)

    # Build library
    _library = ShapeletLibrary(shapelets)

    # Stage 4: Fractal analysis
    hurst = wavelet_hurst(ts.values)
    assert np.isfinite(hurst.hurst_exponent)
    self_sim = cross_scale_similarity(decomp)
    assert self_sim.mean_similarity >= 0

    # Stage 5: Features
    pipeline = FeaturePipeline()
    X, y = pipeline.build_feature_matrix(ts=ts, decomp=decomp, window=50)
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
    # Feature vector should have consistent width
    assert X.shape[1] > 10  # At least some features

    # Verify no NaN/Inf in features
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(y))
