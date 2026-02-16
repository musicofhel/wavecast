"""Full pipeline orchestration."""

from __future__ import annotations

from typing import Any

from wavecast.core.config import WaveCastConfig
from wavecast.core.exceptions import PipelineError
from wavecast.core.types import (
    ForecastResult,
    Shapelet,
    TimeSeries,
    WaveletDecomposition,
)
from wavecast.pipeline.stages import (
    stage_data,
    stage_decompose,
    stage_discover,
    stage_fractal,
)


class PipelineRunner:
    """Orchestrates the full WaveCast pipeline."""

    def __init__(self, config: WaveCastConfig | None = None) -> None:
        self.config = config or WaveCastConfig()
        self.config.ensure_dirs()

    def run_discovery(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> list[Shapelet]:
        """Run data → decompose → discover pipeline."""
        from wavecast.data.preprocessing import label_returns, log_returns

        data_result = stage_data(ticker, start, end, config=self.config)
        ts: TimeSeries = data_result.data

        decomp_result = stage_decompose(ts, config=self.config)
        decomp: WaveletDecomposition = decomp_result.data

        returns_ts = log_returns(ts)
        labels = label_returns(returns_ts)

        discover_result = stage_discover(decomp, labels, config=self.config)
        return discover_result.data

    def run_analysis(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Run data → decompose → fractal analysis pipeline."""
        from wavecast.fractal.regime import detect_regime
        from wavecast.wavelets.dwt import compute_level_stats

        data_result = stage_data(ticker, start, end, config=self.config)
        ts: TimeSeries = data_result.data

        decomp_result = stage_decompose(ts, config=self.config)
        decomp: WaveletDecomposition = decomp_result.data

        level_stats = compute_level_stats(decomp)

        fractal_result = stage_fractal(ts, config=self.config)
        hurst, mfdfa, self_sim = fractal_result.data

        regime = detect_regime(ts.values, config=self.config.fractal)

        return {
            "ticker": ticker,
            "data_length": ts.length,
            "decomposition": decomp,
            "level_stats": level_stats,
            "hurst": hurst,
            "mfdfa": mfdfa,
            "self_similarity": self_sim,
            "regime": regime,
            "timings": {
                "data": data_result.duration_seconds,
                "decompose": decomp_result.duration_seconds,
                "fractal": fractal_result.duration_seconds,
            },
        }

    def run_full(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        horizon: int = 5,
        model_type: str = "ensemble",
    ) -> ForecastResult:
        """Run the complete pipeline: data → features → model → forecast."""
        from wavecast.data.preprocessing import label_returns, log_returns
        from wavecast.dtw.matching import match_against_library
        from wavecast.features.pipeline import FeaturePipeline
        from wavecast.models.ensemble import EnsembleModel
        from wavecast.models.gradient_boost import GradientBoostModel
        from wavecast.models.wavelet_lstm import WaveletLSTM
        from wavecast.shapelets.library import ShapeletLibrary

        # Stage 1: Data
        data_result = stage_data(ticker, start, end, config=self.config)
        ts: TimeSeries = data_result.data

        # Stage 2: Decompose
        decomp_result = stage_decompose(ts, config=self.config)
        decomp: WaveletDecomposition = decomp_result.data

        # Stage 3: Discover shapelets
        returns_ts = log_returns(ts)
        labels = label_returns(returns_ts)
        discover_result = stage_discover(decomp, labels, config=self.config)
        shapelets: list[Shapelet] = discover_result.data
        library = ShapeletLibrary(shapelets)

        # Stage 4: Match
        _match_result = None
        if len(shapelets) > 0:
            best_level = max(range(1, decomp.level + 1), key=lambda lv: sum(
                1 for s in shapelets if s.wavelet_level == lv
            ))
            _match_result = match_against_library(
                decomp.detail_at_level(best_level),
                library,
                best_level,
                self.config.dtw,
            )

        # Stage 5: Fractal
        fractal_result = stage_fractal(ts, config=self.config)
        hurst, mfdfa, self_sim = fractal_result.data

        # Stage 6: Features
        feature_pipeline = FeaturePipeline()
        X, y = feature_pipeline.build_feature_matrix(
            ts=ts,
            decomp=decomp,
        )

        if len(X) < 100:
            raise PipelineError(
                f"Not enough feature samples: {len(X)} (need at least 100)"
            )

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, _y_test = y[:split], y[split:]

        val_split = int(len(X_train) * 0.8)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train_fit = X_train[:val_split]
        y_train_fit = y_train[:val_split]

        # Stage 7: Train model
        if model_type == "xgboost":
            model = GradientBoostModel()
            model.fit(X_train_fit, y_train_fit, X_val, y_val)
        elif model_type == "lstm":
            n_features = X.shape[1]
            model = WaveletLSTM()
            model.fit(X_train_fit, y_train_fit, X_val, y_val)
        else:  # ensemble
            xgb = GradientBoostModel()
            xgb.fit(X_train_fit, y_train_fit, X_val, y_val)
            n_features = X.shape[1]
            lstm = WaveletLSTM(input_size=n_features)
            lstm.fit(X_train_fit, y_train_fit, X_val, y_val)
            model = EnsembleModel(models=[xgb, lstm])
            model.fit(X_train_fit, y_train_fit, X_val, y_val)

        # Predict
        predictions = model.predict(X_test)

        # Build forecast timestamps
        test_timestamps = ts.timestamps[-(len(predictions)):]

        return ForecastResult(
            predictions=predictions,
            timestamps=test_timestamps,
            model_name=model.name,
            ticker=ticker,
            horizon=horizon,
            metadata={
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": X.shape[1],
                "n_shapelets": len(shapelets),
            },
        )
