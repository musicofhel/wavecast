"""Integration test: end-to-end signal generation and backtesting pipeline."""

from __future__ import annotations

import numpy as np

from wavecast.signals.backtest import SignalBacktest
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.generator import SignalGenerator
from wavecast.signals.position import PositionSizer


class TestSignalPipelineEndToEnd:
    def test_synthetic_pipeline(self):
        """Full pipeline: synthetic probabilities -> signals -> backtest -> metrics."""
        rng = np.random.default_rng(42)
        vocab_size = 100
        n_samples = 200

        # Generate synthetic softmax probabilities with a directional bias
        probabilities = rng.dirichlet(np.ones(vocab_size), size=n_samples)
        predicted_tokens = np.argmax(probabilities, axis=1)
        start = np.datetime64("2024-01-01")
        timestamps = np.arange(
            start, start + np.timedelta64(n_samples, "h"), np.timedelta64(1, "h")
        )

        # Synthetic actual returns correlated with token predictions
        midpoint = vocab_size // 2
        actual_returns = (
            (predicted_tokens - midpoint).astype(np.float64) / midpoint * 0.01
        )
        actual_returns += rng.normal(0, 0.005, n_samples)

        # Step 1: Generate signals
        gen = SignalGenerator(
            vocab_size=vocab_size,
            confidence_threshold=0.3,
        )
        signals = gen.generate(probabilities, predicted_tokens, timestamps, ticker="SYN")
        assert len(signals) == n_samples

        # Step 2: Calibrate on first half, test on second half
        cal_end = n_samples // 2
        gen.calibrate(probabilities[:cal_end], predicted_tokens[:cal_end])
        assert gen.temperature > 0

        # Re-generate with calibrated model
        signals = gen.generate(
            probabilities[cal_end:],
            predicted_tokens[cal_end:],
            timestamps[cal_end:],
            ticker="SYN",
        )

        # Step 3: Backtest
        bt = SignalBacktest(
            cost_model=TransactionCostModel(
                commission_rate=0.001, spread_bps=2.0, slippage_bps=1.0
            ),
            position_sizer=PositionSizer(
                method="fractional_kelly", max_position=0.5, kelly_fraction=0.5
            ),
            initial_capital=100000.0,
        )
        result = bt.run(signals, actual_returns[cal_end:], timestamps[cal_end:])

        # Verify all components
        assert len(result.returns) == n_samples - cal_end
        assert len(result.equity_curve) == n_samples - cal_end
        assert result.equity_curve[0] > 0

        # All 16 metrics populated
        assert len(result.metrics) >= 16
        assert "sharpe_ratio" in result.metrics
        assert "sortino_ratio" in result.metrics
        assert "calmar_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "value_at_risk_95" in result.metrics
        assert "win_rate" in result.metrics
        assert "expectancy" in result.metrics
        assert "tail_ratio" in result.metrics

        # Verify higher costs -> lower net returns
        bt_free = SignalBacktest(
            cost_model=TransactionCostModel.zero(),
            position_sizer=PositionSizer(
                method="fractional_kelly", max_position=0.5, kelly_fraction=0.5
            ),
        )
        result_free = bt_free.run(signals, actual_returns[cal_end:], timestamps[cal_end:])
        # Net returns with costs should be <= without costs
        assert result.equity_curve[-1] <= result_free.equity_curve[-1]
