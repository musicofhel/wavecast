"""Tests for ExperimentConfig."""

import yaml

from wavecast.experiments.config import ExperimentConfig


def test_default_values():
    config = ExperimentConfig(name="test", tickers=["AAPL"])
    assert config.interval == "1h"
    assert config.train_end == "2023-12-31"
    assert config.test_start == "2024-01-01"
    assert config.alphabet_size == 7
    assert config.n_segments == 256
    assert config.word_length == 4
    assert config.word_stride == 1
    assert config.min_word_freq == 2
    assert config.max_vocab_size == 300
    assert config.context_length == 16
    assert config.embed_dim == 64
    assert config.num_heads == 4
    assert config.num_layers == 3
    assert config.dropout == 0.1
    assert config.epochs == 80
    assert config.batch_size == 64
    assert config.learning_rate == 0.0005
    assert config.patience == 15
    assert config.dwt_levels is None
    assert config.sectors is None
    assert config.cross_sector_training is True


def test_custom_values():
    config = ExperimentConfig(
        name="custom",
        tickers=["AAPL", "MSFT"],
        interval="1d",
        alphabet_size=5,
        dwt_levels=[1, 2, 3],
        sectors=["tech"],
        cross_sector_training=False,
    )
    assert config.name == "custom"
    assert config.tickers == ["AAPL", "MSFT"]
    assert config.interval == "1d"
    assert config.alphabet_size == 5
    assert config.dwt_levels == [1, 2, 3]
    assert config.sectors == ["tech"]
    assert config.cross_sector_training is False


def test_yaml_roundtrip(tmp_path):
    config = ExperimentConfig(
        name="yaml_test",
        tickers=["SPY", "QQQ"],
        alphabet_size=9,
        dwt_levels=[2, 3, 4],
    )
    yaml_path = tmp_path / "config.yaml"

    # Serialize to YAML via dataclass dict
    from dataclasses import asdict

    yaml_path.write_text(yaml.dump(asdict(config)))

    # Deserialize back
    raw = yaml.safe_load(yaml_path.read_text())
    restored = ExperimentConfig(**raw)

    assert restored.name == config.name
    assert restored.tickers == config.tickers
    assert restored.alphabet_size == config.alphabet_size
    assert restored.dwt_levels == config.dwt_levels
    assert restored.interval == config.interval


def test_yaml_roundtrip_with_none_fields(tmp_path):
    config = ExperimentConfig(
        name="none_fields",
        tickers=["AAPL"],
        dwt_levels=None,
        sectors=None,
    )
    yaml_path = tmp_path / "config.yaml"

    from dataclasses import asdict

    yaml_path.write_text(yaml.dump(asdict(config)))
    raw = yaml.safe_load(yaml_path.read_text())
    restored = ExperimentConfig(**raw)

    assert restored.dwt_levels is None
    assert restored.sectors is None
