"""Tests for configuration."""

from wavecast.core.config import WaveCastConfig


def test_default_config():
    config = WaveCastConfig()
    assert config.wavelet.wavelet == "db4"
    assert config.wavelet.level == 5
    assert config.shapelet.z_threshold == 0.5
    assert config.dtw.window == 10
    assert config.fractal.hurst_method == "wavelet"


def test_ensure_dirs(tmp_path):
    config = WaveCastConfig(
        data_dir=tmp_path / "data",
        library_dir=tmp_path / "lib",
        model_dir=tmp_path / "models",
        cache_dir=tmp_path / "cache",
    )
    config.ensure_dirs()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "lib").exists()
    assert (tmp_path / "models").exists()
    assert (tmp_path / "cache").exists()
