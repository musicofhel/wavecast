"""Tests for forward testing configuration."""

from pathlib import Path

from wavecast.forward.config import ForwardTestConfig


def test_defaults():
    """ForwardTestConfig() has correct defaults."""
    config = ForwardTestConfig()
    assert config.test_name == "default"
    assert config.model_path == ""
    assert config.vocab_path == ""
    assert config.tickers == []
    assert config.intervals == ["1h"]
    assert config.horizons == [1]
    assert config.lookback_bars == 300
    assert config.dwt_levels == [1, 2, 5]
    assert config.sax.n_segments == 256
    assert config.signal.confidence_threshold == 0.0


def test_validation():
    """ForwardTestConfig with custom values works."""
    config = ForwardTestConfig(
        tickers=["AAPL", "MSFT"],
        model_path="/tmp/model.pt",
        vocab_path="/tmp/vocab.json",
        intervals=["1h", "4h"],
        horizons=[1, 2],
    )
    assert config.tickers == ["AAPL", "MSFT"]
    assert config.model_path == "/tmp/model.pt"
    assert config.intervals == ["1h", "4h"]
    assert config.horizons == [1, 2]


def test_path_expansion():
    """log_dir default uses Path.home()."""
    config = ForwardTestConfig()
    assert config.log_dir == Path.home() / ".wavecast" / "forward_tests"
    assert isinstance(config.log_dir, Path)
