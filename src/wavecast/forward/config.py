"""Forward testing configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

from wavecast.core.config import SAXConfig
from wavecast.signals.config import SignalConfig


class ForwardTestConfig(BaseSettings):
    """Configuration for forward testing."""

    test_name: str = "default"
    model_path: str = ""
    vocab_path: str = ""
    tickers: list[str] = Field(default_factory=list)
    intervals: list[str] = Field(default_factory=lambda: ["1h"])
    horizons: list[int] = Field(default_factory=lambda: [1])
    lookback_bars: int = 300
    dwt_levels: list[int] = Field(default_factory=lambda: [1, 2, 5])
    sax: SAXConfig = Field(default_factory=SAXConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    log_dir: Path = Field(default_factory=lambda: Path.home() / ".wavecast" / "forward_tests")
    retrain_after_days: int | None = None
