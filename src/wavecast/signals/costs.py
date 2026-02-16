"""Transaction cost model for signal backtesting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    """Models commission, spread, and slippage costs.

    Args:
        commission_rate: Commission per trade as fraction of trade value (e.g., 0.001 = 10bps)
        spread_bps: Bid-ask spread in basis points
        slippage_bps: Market impact slippage in basis points
    """
    commission_rate: float = 0.001
    spread_bps: float = 2.0
    slippage_bps: float = 1.0

    def compute(
        self, trade_value: float, direction_change: bool
    ) -> tuple[float, float, float]:
        """Compute transaction costs for a trade.

        Args:
            trade_value: Absolute value of the trade
            direction_change: True if position direction changed (doubles costs
                because old position is closed and new one opened)

        Returns:
            Tuple of (commission, spread_cost, slippage_cost)
        """
        multiplier = 2.0 if direction_change else 1.0
        commission = trade_value * self.commission_rate * multiplier
        spread_cost = trade_value * (self.spread_bps / 10000) * multiplier
        slippage_cost = trade_value * (self.slippage_bps / 10000) * multiplier
        return commission, spread_cost, slippage_cost

    def total(self, trade_value: float, direction_change: bool) -> float:
        """Total cost for a trade."""
        c, s, sl = self.compute(trade_value, direction_change)
        return c + s + sl

    @classmethod
    def zero(cls) -> TransactionCostModel:
        """Create a zero-cost model (for testing)."""
        return cls(commission_rate=0.0, spread_bps=0.0, slippage_bps=0.0)
