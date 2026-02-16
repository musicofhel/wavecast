"""Walk-forward backtesting engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.core.config import BacktestConfig
from wavecast.core.types import BacktestResult
from wavecast.evaluation import metrics as m
from wavecast.models.base import BaseModel


class WalkForwardBacktest:
    """Walk-forward backtesting with rolling train/test windows."""

    def __init__(
        self,
        model: BaseModel,
        config: BacktestConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or BacktestConfig()

    def run(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        timestamps: NDArray[np.datetime64] | None = None,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Rolls a train/test window through the data, retraining the model
        at each step and recording out-of-sample predictions.
        """
        train_size = self.config.walk_forward_train
        test_size = self.config.walk_forward_test
        n = len(X)

        if n < train_size + test_size:
            raise ValueError(
                f"Not enough data: {n} samples, need at least "
                f"{train_size + test_size} (train={train_size} + test={test_size})"
            )

        all_predictions: list[float] = []
        all_actuals: list[float] = []
        all_positions: list[float] = []
        all_returns: list[float] = []
        test_indices: list[int] = []

        start = 0
        while start + train_size + test_size <= n:
            train_end = start + train_size
            test_end = min(train_end + test_size, n)

            X_train = X[start:train_end]
            y_train = y[start:train_end]
            X_test = X[train_end:test_end]
            y_test = y[train_end:test_end]

            # Train
            val_split = max(1, len(X_train) // 5)
            self.model.fit(
                X_train[:-val_split],
                y_train[:-val_split],
                X_train[-val_split:],
                y_train[-val_split:],
            )

            # Predict
            preds = self.model.predict(X_test)

            for i in range(len(preds)):
                idx = train_end + i
                pred = float(preds[i])
                actual = float(y_test[i])

                # Position: long if predicted return > 0, short otherwise
                position = self.config.position_size if pred > 0 else -self.config.position_size

                # Return after commission
                trade_return = position * actual - abs(position) * self.config.commission

                all_predictions.append(pred)
                all_actuals.append(actual)
                all_positions.append(position)
                all_returns.append(trade_return)
                test_indices.append(idx)

            start += test_size  # Slide forward by test_size

        returns_arr = np.array(all_returns)
        positions_arr = np.array(all_positions)
        equity_curve = self.config.initial_capital * np.cumprod(1 + returns_arr)

        if timestamps is not None:
            bt_timestamps = timestamps[test_indices]
        else:
            bt_timestamps = np.array(test_indices, dtype=np.datetime64)

        actuals_arr = np.array(all_actuals)
        preds_arr = np.array(all_predictions)

        metrics_dict = {
            "rmse": m.rmse(actuals_arr, preds_arr),
            "mae": m.mae(actuals_arr, preds_arr),
            "directional_accuracy": m.directional_accuracy(actuals_arr, preds_arr),
            "sharpe_ratio": m.sharpe_ratio(returns_arr),
            "max_drawdown": m.max_drawdown(equity_curve),
            "profit_factor": m.profit_factor(returns_arr),
            "total_return": float(equity_curve[-1] / self.config.initial_capital - 1)
            if len(equity_curve) > 0
            else 0.0,
            "num_trades": len(returns_arr),
        }

        return BacktestResult(
            returns=returns_arr,
            positions=positions_arr,
            equity_curve=equity_curve,
            timestamps=bt_timestamps,
            metrics=metrics_dict,
            model_name=self.model.name,
            ticker="",
        )
