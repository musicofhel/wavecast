"""Tests for shapelet discovery."""


from tests.fixtures.generators import make_labels
from wavecast.core.config import ShapeletConfig
from wavecast.shapelets.discovery import discover_shapelets
from wavecast.wavelets.dwt import decompose


def test_discover_on_synthetic(sine_series):
    decomp = decompose(sine_series, level=3)
    labels = make_labels(sine_series.length)
    config = ShapeletConfig(z_threshold=0.5, min_length=3, ig_min=0.001)
    shapelets = discover_shapelets(decomp, labels, config)
    # Should find at least some shapelets from the sine wave's energy patterns
    assert isinstance(shapelets, list)
    for s in shapelets:
        assert s.id != ""
        assert s.wavelet_level >= 1
        assert s.information_gain >= 0


def test_discover_sorted_by_ig(sine_series):
    decomp = decompose(sine_series, level=3)
    labels = make_labels(sine_series.length)
    config = ShapeletConfig(z_threshold=0.5, min_length=3, ig_min=0.001)
    shapelets = discover_shapelets(decomp, labels, config)
    if len(shapelets) >= 2:
        for i in range(len(shapelets) - 1):
            assert shapelets[i].information_gain >= shapelets[i + 1].information_gain
