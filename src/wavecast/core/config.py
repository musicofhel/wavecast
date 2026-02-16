"""Configuration management for WaveCast."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class WaveletConfig(BaseSettings):
    """Wavelet decomposition settings."""

    wavelet: str = "db4"
    level: int = 5
    mode: str = "symmetric"


class ShapeletConfig(BaseSettings):
    """Shapelet discovery settings."""

    z_threshold: float = 0.5
    min_length: int = 3
    min_variance: float = 0.01
    top_k: int = 20
    ig_min: float = 0.01
    cluster_threshold: float = 0.3


class DTWConfig(BaseSettings):
    """DTW matching settings."""

    window: int = 10
    use_pruning: bool = True
    top_k: int = 5
    normalize: bool = True


class FractalConfig(BaseSettings):
    """Fractal analysis settings."""

    hurst_method: str = "wavelet"
    hurst_window: int = 252
    mfdfa_q_range: tuple[float, float] = (-5.0, 5.0)
    mfdfa_q_steps: int = 21
    trending_threshold: float = 0.55
    mean_revert_threshold: float = 0.45


class ModelConfig(BaseSettings):
    """Model training settings."""

    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    ensemble_weights: list[float] = Field(default_factory=lambda: [0.5, 0.5])


class BacktestConfig(BaseSettings):
    """Backtesting settings."""

    initial_capital: float = 100000.0
    position_size: float = 1.0
    commission: float = 0.001
    walk_forward_train: int = 252
    walk_forward_test: int = 21


class SAXConfig(BaseSettings):
    """SAX transformation settings."""

    n_segments: int = 256
    alphabet_size: int = 7
    word_length: int = 4
    word_stride: int = 1


class TokenizerConfig(BaseSettings):
    """Tokenizer settings."""

    context_length: int = 16
    min_word_freq: int = 1
    max_vocab_size: int = 100


class SequenceModelConfig(BaseSettings):
    """Sequence model settings.

    Optimal dwt_levels=[1,2,5] (levels 3&4 are noise).
    """

    context_length: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    epochs: int = 80
    batch_size: int = 64
    lr: float = 0.0005
    patience: int = 15


class WaveCastConfig(BaseSettings):
    """Top-level configuration."""

    model_config = {"env_prefix": "WAVECAST_"}

    data_dir: Path = Path.home() / ".wavecast" / "data"
    library_dir: Path = Path.home() / ".wavecast" / "library"
    model_dir: Path = Path.home() / ".wavecast" / "models"
    cache_dir: Path = Path.home() / ".wavecast" / "cache"

    wavelet: WaveletConfig = Field(default_factory=WaveletConfig)
    shapelet: ShapeletConfig = Field(default_factory=ShapeletConfig)
    dtw: DTWConfig = Field(default_factory=DTWConfig)
    fractal: FractalConfig = Field(default_factory=FractalConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    sax: SAXConfig = Field(default_factory=SAXConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    sequence_model: SequenceModelConfig = Field(default_factory=SequenceModelConfig)

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [self.data_dir, self.library_dir, self.model_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
