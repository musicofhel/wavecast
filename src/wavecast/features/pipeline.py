"""Feature engineering pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import SAXConfig
from wavecast.core.types import (
    FeatureVector,
    HurstResult,
    MatchResult,
    MFDFAResult,
    SelfSimilarityResult,
    TimeSeries,
    WaveletDecomposition,
)

from . import fractal_features, market_features, sax_features, shapelet_features, wavelet_features


class FeaturePipeline:
    """Orchestrates feature extraction from all sources."""

    def extract_all(
        self,
        decomp: WaveletDecomposition,
        match_result: MatchResult | None = None,
        hurst: HurstResult | None = None,
        mfdfa: MFDFAResult | None = None,
        self_sim: SelfSimilarityResult | None = None,
        values: NDArray | None = None,
        market_window: int = 20,
        shapelet_top_k: int = 5,
        sax_config: SAXConfig | None = None,
    ) -> FeatureVector:
        """Extract all features and return a FeatureVector."""
        wf = wavelet_features.extract(decomp)
        sf = shapelet_features.extract(match_result, top_k=shapelet_top_k)
        ff = fractal_features.extract(hurst, mfdfa, self_sim)
        mf = (
            market_features.extract(values, window=market_window)
            if values is not None
            else np.zeros(market_features.MARKET_FEATURE_SIZE, dtype=np.float64)
        )
        sf_sax = (
            sax_features.extract(decomp, sax_config)
            if sax_config is not None
            else np.array([], dtype=np.float64)
        )

        return FeatureVector(
            wavelet_features=wf,
            shapelet_features=sf,
            fractal_features=ff,
            market_features=mf,
            sax_features=sf_sax,
        )

    def build_feature_matrix(
        self,
        ts: TimeSeries,
        decomp: WaveletDecomposition,
        hurst_results: list[HurstResult | None] | None = None,
        mfdfa_results: list[MFDFAResult | None] | None = None,
        self_sim_results: list[SelfSimilarityResult | None] | None = None,
        match_results: list[MatchResult | None] | None = None,
        window: int = 50,
        horizon: int = 1,
        market_window: int = 20,
        sax_config: SAXConfig | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Build a feature matrix X and target vector y using rolling windows.

        For each window position i, extracts features from ts.values[i:i+window]
        and the target is the log-return over the next `horizon` steps.

        Returns:
            (X, y) where X has shape (n_samples, n_features) and y has shape (n_samples,).
        """
        values = ts.values
        n = len(values)
        if n < window + horizon:
            return np.empty((0, 0)), np.empty(0)

        X_rows = []
        y_vals = []

        for i in range(n - window - horizon + 1):
            segment = values[i : i + window]

            # Get corresponding pre-computed results if available
            hurst_i = hurst_results[i] if hurst_results and i < len(hurst_results) else None
            mfdfa_i = mfdfa_results[i] if mfdfa_results and i < len(mfdfa_results) else None
            self_sim_i = (
                self_sim_results[i] if self_sim_results and i < len(self_sim_results) else None
            )
            match_i = match_results[i] if match_results and i < len(match_results) else None

            fv = self.extract_all(
                decomp=decomp,
                match_result=match_i,
                hurst=hurst_i,
                mfdfa=mfdfa_i,
                self_sim=self_sim_i,
                values=segment,
                market_window=market_window,
                sax_config=sax_config,
            )
            X_rows.append(fv.combined)

            # Target: log return over horizon
            future_val = values[i + window + horizon - 1]
            current_val = values[i + window - 1]
            if current_val > 0 and future_val > 0:
                y_vals.append(np.log(future_val / current_val))
            else:
                y_vals.append(0.0)

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_vals, dtype=np.float64)

        return X, y
