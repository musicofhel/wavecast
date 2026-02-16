"""Traditional market microstructure features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# log_return, rolling_vol, rolling_mean_return, momentum, mean_reversion = 5
MARKET_FEATURE_SIZE = 5


def extract(values: NDArray, window: int = 20) -> NDArray:
    """Extract market features from raw price values at the last time point.

    Features:
        - log return (last period)
        - rolling volatility (std of log returns over window)
        - rolling mean log return over window
        - momentum (cumulative return over window)
        - mean reversion indicator (z-score of price relative to rolling mean)
    """
    values = np.asarray(values, dtype=np.float64)
    features = np.zeros(MARKET_FEATURE_SIZE, dtype=np.float64)

    if len(values) < 2:
        return features

    # Log returns
    log_returns = np.diff(np.log(np.maximum(values, 1e-10)))

    # Latest log return
    features[0] = log_returns[-1]

    # Use available window (may be smaller than requested)
    effective_window = min(window, len(log_returns))
    recent_returns = log_returns[-effective_window:]

    # Rolling volatility
    features[1] = float(np.std(recent_returns))

    # Rolling mean return
    features[2] = float(np.mean(recent_returns))

    # Momentum: cumulative return over window
    features[3] = float(np.sum(recent_returns))

    # Mean reversion indicator: z-score of current price vs rolling mean
    effective_price_window = min(window, len(values))
    rolling_mean = np.mean(values[-effective_price_window:])
    rolling_std = np.std(values[-effective_price_window:])
    if rolling_std > 0:
        features[4] = (values[-1] - rolling_mean) / rolling_std

    return features
