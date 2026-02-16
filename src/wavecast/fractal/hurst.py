"""Hurst exponent estimation via wavelet variance method."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pywt
from numpy.typing import NDArray

from wavecast.core.exceptions import FractalError
from wavecast.core.types import HurstResult, RegimeType

logger = logging.getLogger(__name__)


def wavelet_hurst(
    values: NDArray,
    wavelet: str = "db4",
    max_level: int | None = None,
) -> HurstResult:
    """Estimate the Hurst exponent using wavelet variance scaling.

    Performs DWT decomposition, computes variance at each detail level,
    then fits log2(var_j) vs log2(scale_j) to extract H.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    if len(values) < 16:
        raise FractalError("Need at least 16 data points for wavelet Hurst estimation")

    if max_level is None:
        max_level = pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet).dec_len)
    max_level = max(min(max_level, pywt.dwt_max_level(len(values), pywt.Wavelet(wavelet).dec_len)), 1)

    coeffs = pywt.wavedec(values, wavelet, level=max_level)
    # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    details = coeffs[1:]  # detail coefficients from coarsest to finest

    if len(details) < 2:
        raise FractalError("Need at least 2 detail levels for regression")

    level_variances = np.array([np.var(d) for d in details], dtype=np.float64)

    # Filter out zero-variance levels to avoid log(0)
    valid = level_variances > 0
    if valid.sum() < 2:
        raise FractalError("Insufficient non-zero variance levels for regression")

    # Levels go from n (coarsest) down to 1 (finest) in the details list
    num_levels = len(details)
    j_values = np.arange(num_levels, 0, -1, dtype=np.float64)  # [n, n-1, ..., 1]

    log_scale = np.log2(2.0 ** j_values[valid])
    log_var = np.log2(level_variances[valid])

    slope, intercept = np.polyfit(log_scale, log_var, 1)
    hurst_exponent = (slope + 1.0) / 2.0

    # Compute R-squared
    predicted = slope * log_scale + intercept
    ss_res = np.sum((log_var - predicted) ** 2)
    ss_tot = np.sum((log_var - np.mean(log_var)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Classify regime
    if hurst_exponent > 0.55:
        regime = RegimeType.TRENDING
    elif hurst_exponent < 0.45:
        regime = RegimeType.MEAN_REVERTING
    else:
        regime = RegimeType.RANDOM_WALK

    return HurstResult(
        hurst_exponent=float(hurst_exponent),
        intercept=float(intercept),
        r_squared=float(r_squared),
        level_variances=level_variances,
        regime=regime,
    )


def rolling_hurst(
    values: NDArray,
    window: int = 252,
    step: int = 1,
    wavelet: str = "db4",
) -> tuple[NDArray, NDArray]:
    """Compute rolling Hurst exponent over a sliding window.

    Returns:
        Tuple of (hurst_values, center_indices) arrays.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    n = len(values)
    if n < window:
        raise FractalError(f"Series length {n} is shorter than window {window}")

    hurst_vals = []
    indices = []

    for start in range(0, n - window + 1, step):
        segment = values[start : start + window]
        try:
            result = wavelet_hurst(segment, wavelet=wavelet)
            hurst_vals.append(result.hurst_exponent)
        except FractalError:
            hurst_vals.append(np.nan)
        indices.append(start + window // 2)

    return np.array(hurst_vals, dtype=np.float64), np.array(indices, dtype=np.int64)


def classify_regime(
    hurst_value: float,
    trending_threshold: float = 0.6,
    mean_revert_threshold: float = 0.4,
) -> RegimeType:
    """Classify a Hurst exponent value into a market regime.

    Uses Phase 3 thresholds (0.6/0.4) which are wider than the
    per-window thresholds (0.55/0.45) in wavelet_hurst().
    """
    if np.isnan(hurst_value):
        return RegimeType.RANDOM_WALK
    if hurst_value > trending_threshold:
        return RegimeType.TRENDING
    if hurst_value < mean_revert_threshold:
        return RegimeType.MEAN_REVERTING
    return RegimeType.RANDOM_WALK


def rolling_hurst_with_regimes(
    values: NDArray,
    window: int = 1638,
    step: int = 1,
    wavelet: str = "db4",
    trending_threshold: float = 0.6,
    mean_revert_threshold: float = 0.4,
    use_returns: bool = False,
) -> tuple[NDArray, NDArray, list[RegimeType]]:
    """Compute rolling Hurst and classify each window into a regime.

    Args:
        values: Raw price series (or returns if use_returns=False).
        window: Rolling window size (default 1638 ~ 252 trading days in hourly).
        step: Step size between windows.
        wavelet: Wavelet family.
        trending_threshold: H above this -> TRENDING.
        mean_revert_threshold: H below this -> MEAN_REVERTING.
        use_returns: If True, compute Hurst on log returns instead of raw
            prices. Raw prices are integrated processes and will show H > 1.0
            (always TRENDING). Log returns give meaningful regime detection.

    Returns:
        Tuple of (hurst_values, center_indices, regime_labels).
    """
    if use_returns:
        prices = np.array(values, dtype=np.float64)
        log_returns = np.diff(np.log(prices))
        hurst_vals, indices = rolling_hurst(
            log_returns, window=window, step=step, wavelet=wavelet
        )
        # Shift indices by +1 to align back to the original price indices
        # since np.diff reduces length by 1
        indices = indices + 1
    else:
        hurst_vals, indices = rolling_hurst(
            values, window=window, step=step, wavelet=wavelet
        )

    regimes = [
        classify_regime(h, trending_threshold, mean_revert_threshold)
        for h in hurst_vals
    ]

    return hurst_vals, indices, regimes


class HurstCache:
    """Disk-backed cache for rolling Hurst + regime computations.

    Stores results as .npz files:
        {cache_dir}/{ticker}_{interval}_hurst_w{window}_s{step}.npz
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".wavecast" / "cache" / "hurst"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(
        self, ticker: str, interval: str, window: int, step: int
    ) -> Path:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_ticker}_{interval}_hurst_w{window}_s{step}.npz"

    def get(
        self, ticker: str, interval: str, window: int, step: int
    ) -> tuple[NDArray, NDArray, list[RegimeType]] | None:
        """Retrieve cached Hurst + regimes, or None if not cached."""
        path = self._key_path(ticker, interval, window, step)
        if not path.exists():
            return None

        try:
            data = np.load(path, allow_pickle=False)
            hurst_vals = data["hurst_values"]
            indices = data["center_indices"]
            regime_codes = data["regime_codes"]
            regimes = [_CODE_TO_REGIME[int(c)] for c in regime_codes]
            return hurst_vals, indices, regimes
        except Exception:
            logger.warning("Corrupt Hurst cache file: %s", path)
            return None

    def put(
        self,
        ticker: str,
        interval: str,
        window: int,
        step: int,
        hurst_vals: NDArray,
        indices: NDArray,
        regimes: list[RegimeType],
    ) -> Path:
        """Store Hurst + regime results to the cache."""
        path = self._key_path(ticker, interval, window, step)
        regime_codes = np.array(
            [_REGIME_TO_CODE[r] for r in regimes], dtype=np.int8
        )
        np.savez_compressed(
            path,
            hurst_values=hurst_vals,
            center_indices=indices,
            regime_codes=regime_codes,
        )
        return path

    def compute_and_cache(
        self,
        ticker: str,
        interval: str,
        values: NDArray,
        window: int = 1638,
        step: int = 1,
        wavelet: str = "db4",
        trending_threshold: float = 0.6,
        mean_revert_threshold: float = 0.4,
        use_returns: bool = False,
    ) -> tuple[NDArray, NDArray, list[RegimeType]]:
        """Compute rolling Hurst + regimes, using cache if available."""
        cached = self.get(ticker, interval, window, step)
        if cached is not None:
            return cached

        hurst_vals, indices, regimes = rolling_hurst_with_regimes(
            values,
            window=window,
            step=step,
            wavelet=wavelet,
            trending_threshold=trending_threshold,
            mean_revert_threshold=mean_revert_threshold,
            use_returns=use_returns,
        )

        self.put(ticker, interval, window, step, hurst_vals, indices, regimes)
        return hurst_vals, indices, regimes


# Regime <-> integer code mapping for storage
_REGIME_TO_CODE: dict[RegimeType, int] = {
    RegimeType.TRENDING: 0,
    RegimeType.MEAN_REVERTING: 1,
    RegimeType.RANDOM_WALK: 2,
}
_CODE_TO_REGIME: dict[int, RegimeType] = {v: k for k, v in _REGIME_TO_CODE.items()}
