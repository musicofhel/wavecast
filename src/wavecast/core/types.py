"""Core data types for WaveCast."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class MarketLabel(str, Enum):
    """Direction labels for classification."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class RegimeType(str, Enum):
    """Market regime classification."""

    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM_WALK = "random_walk"


@dataclass
class TimeSeries:
    """A financial time series with metadata."""

    values: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]
    ticker: str
    interval: str = "1d"
    column: str = "close"

    @property
    def length(self) -> int:
        return len(self.values)

    def slice(self, start: int, end: int) -> TimeSeries:
        return TimeSeries(
            values=self.values[start:end],
            timestamps=self.timestamps[start:end],
            ticker=self.ticker,
            interval=self.interval,
            column=self.column,
        )


@dataclass
class WaveletDecomposition:
    """Result of DWT decomposition."""

    coefficients: list[NDArray[np.float64]]  # [cA_n, cD_n, ..., cD_1]
    wavelet: str
    level: int
    original_length: int
    ticker: str = ""

    @property
    def approximation(self) -> NDArray[np.float64]:
        return self.coefficients[0]

    @property
    def details(self) -> list[NDArray[np.float64]]:
        return self.coefficients[1:]

    def detail_at_level(self, level: int) -> NDArray[np.float64]:
        """Get detail coefficients at a specific level (1=finest)."""
        if level < 1 or level > self.level:
            raise ValueError(f"Level must be 1-{self.level}, got {level}")
        return self.coefficients[self.level - level + 1]


@dataclass
class LevelStats:
    """Statistics for a single DWT level."""

    level: int
    energy: float
    variance: float
    entropy: float
    mean: float
    std: float
    num_coefficients: int
    approximate_period_days: float


@dataclass
class Shapelet:
    """A discriminative pattern discovered in wavelet domain."""

    id: str
    coefficients: NDArray[np.float64]
    wavelet_level: int
    ticker: str
    label: MarketLabel
    information_gain: float
    start_index: int
    end_index: int
    threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.coefficients)


@dataclass
class ShapeletMatch:
    """Result of DTW matching a query against a library shapelet."""

    shapelet_id: str
    distance: float
    normalized_distance: float
    warping_path: list[tuple[int, int]]
    query_start: int
    query_end: int
    shapelet: Shapelet | None = None


@dataclass
class MatchResult:
    """Collection of shapelet matches for a query."""

    ticker: str
    wavelet_level: int
    matches: list[ShapeletMatch]
    query_length: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def best_match(self) -> ShapeletMatch | None:
        return self.matches[0] if self.matches else None


@dataclass
class HurstResult:
    """Hurst exponent estimation result."""

    hurst_exponent: float
    intercept: float
    r_squared: float
    level_variances: NDArray[np.float64]
    regime: RegimeType

    @property
    def is_trending(self) -> bool:
        return self.hurst_exponent > 0.55

    @property
    def is_mean_reverting(self) -> bool:
        return self.hurst_exponent < 0.45


@dataclass
class MFDFAResult:
    """Multifractal DFA result."""

    q_values: NDArray[np.float64]
    hurst_q: NDArray[np.float64]  # Generalized Hurst H(q)
    tau_q: NDArray[np.float64]  # Scaling exponents
    alpha: NDArray[np.float64]  # Singularity strengths
    f_alpha: NDArray[np.float64]  # Singularity spectrum
    spectrum_width: float


@dataclass
class SelfSimilarityResult:
    """Cross-scale self-similarity analysis."""

    level_pairs: list[tuple[int, int]]
    dtw_distances: list[float]
    similarity_scores: list[float]
    mean_similarity: float


@dataclass
class RegimeDetection:
    """Fractal-based regime detection result."""

    regime: RegimeType
    hurst: HurstResult
    mfdfa: MFDFAResult | None
    confidence: float
    window_size: int


@dataclass
class FeatureVector:
    """Engineered feature vector for a time point."""

    wavelet_features: NDArray[np.float64]
    shapelet_features: NDArray[np.float64]
    fractal_features: NDArray[np.float64]
    market_features: NDArray[np.float64]
    sax_features: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))

    @property
    def combined(self) -> NDArray[np.float64]:
        parts = [
            self.wavelet_features,
            self.shapelet_features,
            self.fractal_features,
            self.market_features,
        ]
        if len(self.sax_features) > 0:
            parts.append(self.sax_features)
        return np.concatenate(parts)


@dataclass
class ForecastResult:
    """Forecasting result."""

    predictions: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]
    model_name: str
    ticker: str
    horizon: int
    confidence_lower: NDArray[np.float64] | None = None
    confidence_upper: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Walk-forward backtest result."""

    returns: NDArray[np.float64]
    positions: NDArray[np.float64]
    equity_curve: NDArray[np.float64]
    timestamps: NDArray[np.datetime64]
    metrics: dict[str, float]
    model_name: str
    ticker: str


class AssetClass(str, Enum):
    """Asset class classification."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"


class Sector(str, Enum):
    """Sector-based classification for Phase 3 universe."""

    TECH = "tech"
    FINANCE = "finance"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"
    BROAD_ETF = "broad_etf"
    COMMODITY_ETF = "commodity_etf"


@dataclass
class SAXRepresentation:
    """Result of SAX transformation."""

    symbols: str
    alphabet_size: int
    breakpoints: NDArray[np.float64]
    n_segments: int
    original_length: int


@dataclass
class SAXWord:
    """A SAX word with token mapping."""

    word: str
    token_id: int
    level: int


@dataclass
class TokenSequence:
    """Token sequence for a single wavelet level."""

    token_ids: list[int]
    words: list[str]
    ticker: str
    interval: str
    wavelet_level: int


@dataclass
class MultiLevelTokenSequence:
    """Token sequences across all wavelet levels."""

    ticker: str
    interval: str
    level_sequences: dict[int, TokenSequence]


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""

    rmse: float
    mae: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float | None = None
