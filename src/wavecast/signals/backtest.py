"""Signal-based backtest engine."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from wavecast.evaluation import metrics as m
from wavecast.signals.costs import TransactionCostModel
from wavecast.signals.position import PositionSizer
from wavecast.signals.types import SignalBacktestResult, SignalSeries, TradeRecord


class SignalBacktest:
    """Backtests trading signals with transaction costs and position sizing.

    Args:
        cost_model: Transaction cost model
        position_sizer: Position sizing strategy
        initial_capital: Starting capital
    """

    def __init__(
        self,
        cost_model: TransactionCostModel | None = None,
        position_sizer: PositionSizer | None = None,
        initial_capital: float = 100000.0,
    ) -> None:
        self.cost_model = cost_model or TransactionCostModel()
        self.position_sizer = position_sizer or PositionSizer()
        self.initial_capital = initial_capital

    def run(
        self,
        signals: SignalSeries,
        actual_returns: NDArray[np.float64],
        timestamps: NDArray[np.datetime64] | None = None,
    ) -> SignalBacktestResult:
        """Run backtest on signal series.

        Args:
            signals: Trading signals with direction and confidence
            actual_returns: Actual period returns aligned with signals
            timestamps: Optional timestamps (uses signal timestamps if not provided)

        Returns:
            SignalBacktestResult with equity curve, trades, and metrics
        """
        n = len(signals)
        if n != len(actual_returns):
            raise ValueError(
                f"Signal length ({n}) != returns length ({len(actual_returns)})"
            )

        if timestamps is None:
            timestamps = signals.timestamps

        directions = signals.directions
        confidences = signals.confidences

        returns = np.zeros(n)
        positions = np.zeros(n)
        equity = np.zeros(n + 1)
        equity[0] = self.initial_capital
        trades: list[TradeRecord] = []

        prev_direction = 0
        trade_entry_idx = 0

        for i in range(n):
            direction = int(directions[i])
            confidence = float(confidences[i])

            # Position size based on confidence
            size = self.position_sizer.size(confidence) if direction != 0 else 0.0
            positions[i] = size * direction

            # Gross return for this period
            gross_ret = positions[i] * actual_returns[i]

            # Transaction costs on direction change
            direction_changed = (direction != prev_direction) and (
                direction != 0 or prev_direction != 0
            )
            if direction_changed:
                trade_value = equity[i] * abs(size)
                total_cost = self.cost_model.total(trade_value, direction_change=True)
                cost_as_return = total_cost / equity[i] if equity[i] > 0 else 0.0
            else:
                cost_as_return = 0.0

            net_ret = gross_ret - cost_as_return
            returns[i] = net_ret
            equity[i + 1] = equity[i] * (1 + net_ret)

            # Record trade on direction change
            if direction_changed and prev_direction != 0:
                cost_breakdown = self.cost_model.compute(
                    equity[trade_entry_idx] * abs(size), direction_change=True
                )
                trades.append(
                    TradeRecord(
                        entry_timestamp=timestamps[trade_entry_idx],
                        exit_timestamp=timestamps[i],
                        direction=prev_direction,
                        position_size=abs(float(positions[trade_entry_idx])),
                        gross_return=float(
                            np.sum(
                                directions[trade_entry_idx:i]
                                * actual_returns[trade_entry_idx:i]
                            )
                        ),
                        net_return=float(np.sum(returns[trade_entry_idx:i])),
                        commission=cost_breakdown[0],
                        spread_cost=cost_breakdown[1],
                        slippage_cost=cost_breakdown[2],
                        confidence=float(confidences[trade_entry_idx]),
                    )
                )

            if direction_changed:
                trade_entry_idx = i

            # Update position sizer with trade results
            if direction != 0:
                self.position_sizer.update(float(net_ret))

            prev_direction = direction

        # Close final trade
        if prev_direction != 0:
            cost_breakdown = self.cost_model.compute(0.0, direction_change=False)
            trades.append(
                TradeRecord(
                    entry_timestamp=timestamps[trade_entry_idx],
                    exit_timestamp=timestamps[-1],
                    direction=prev_direction,
                    position_size=abs(float(positions[trade_entry_idx])),
                    gross_return=float(
                        np.sum(
                            directions[trade_entry_idx:]
                            * actual_returns[trade_entry_idx:]
                        )
                    ),
                    net_return=float(np.sum(returns[trade_entry_idx:])),
                    commission=cost_breakdown[0],
                    spread_cost=cost_breakdown[1],
                    slippage_cost=cost_breakdown[2],
                    confidence=float(confidences[trade_entry_idx]),
                )
            )

        # Compute metrics
        equity_final = equity[1:]  # Remove initial capital entry
        metrics = self._compute_metrics(returns, equity_final, trades)

        return SignalBacktestResult(
            returns=returns,
            positions=positions,
            equity_curve=equity_final,
            timestamps=timestamps,
            trades=trades,
            metrics=metrics,
            config={
                "initial_capital": self.initial_capital,
                "cost_model": {
                    "commission_rate": self.cost_model.commission_rate,
                    "spread_bps": self.cost_model.spread_bps,
                    "slippage_bps": self.cost_model.slippage_bps,
                },
                "position_sizer": {
                    "method": self.position_sizer.method,
                    "max_position": self.position_sizer.max_position,
                },
            },
        )

    def _compute_metrics(
        self,
        returns: NDArray[np.float64],
        equity_curve: NDArray[np.float64],
        trades: list[TradeRecord],
    ) -> dict[str, float]:
        """Compute all 16 risk/return metrics."""
        periods = 252

        trade_returns = (
            np.array([t.net_return for t in trades]) if trades else returns
        )

        return {
            "sharpe_ratio": m.sharpe_ratio(returns, periods=periods),
            "sortino_ratio": m.sortino_ratio(returns, periods=periods),
            "calmar_ratio": m.calmar_ratio(returns, periods=periods),
            "max_drawdown": m.max_drawdown(equity_curve),
            "profit_factor": m.profit_factor(returns),
            "value_at_risk_95": m.value_at_risk(returns, confidence=0.95),
            "conditional_var_95": m.conditional_var(returns, confidence=0.95),
            "win_rate": m.win_rate(trade_returns),
            "avg_win_loss_ratio": m.avg_win_loss_ratio(trade_returns),
            "expectancy": m.expectancy(trade_returns),
            "tail_ratio": m.tail_ratio(returns),
            "total_return": float(equity_curve[-1] / equity_curve[0] - 1)
            if len(equity_curve) > 1
            else 0.0,
            "annualized_return": float(np.mean(returns) * periods),
            "annualized_volatility": float(np.std(returns) * np.sqrt(periods)),
            "num_trades": float(len(trades)),
            "avg_trade_return": float(np.mean(trade_returns))
            if len(trade_returns) > 0
            else 0.0,
        }
