"""Position sizing methods for signal-based trading."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PositionSizer:
    """Converts signal confidence to position size.

    Args:
        method: Sizing method — "fixed", "linear", "kelly", "fractional_kelly"
        max_position: Maximum position size (fraction of capital, 0-1)
        kelly_fraction: Fraction of Kelly criterion to use (for fractional_kelly)
        min_position: Minimum position size (below this, position = 0)
        lookback: Number of past trades for Kelly estimation
    """
    method: str = "fixed"
    max_position: float = 1.0
    kelly_fraction: float = 0.5
    min_position: float = 0.0
    lookback: int = 50
    _trade_returns: list[float] = field(default_factory=list, repr=False)

    def size(self, confidence: float) -> float:
        """Compute position size given signal confidence.

        Args:
            confidence: Signal confidence in [0, 1]

        Returns:
            Position size in [0, max_position]
        """
        if confidence <= self.min_position:
            return 0.0

        if self.method == "fixed":
            return self.max_position
        elif self.method == "linear":
            return min(confidence * self.max_position, self.max_position)
        elif self.method == "kelly":
            return self._kelly_size(confidence, fraction=1.0)
        elif self.method == "fractional_kelly":
            return self._kelly_size(confidence, fraction=self.kelly_fraction)
        else:
            raise ValueError(f"Unknown sizing method: {self.method}")

    def update(self, trade_return: float) -> None:
        """Record a completed trade return for Kelly estimation."""
        self._trade_returns.append(trade_return)

    def _kelly_size(self, confidence: float, fraction: float) -> float:
        """Compute Kelly-criterion-based position size.

        Kelly: f* = (p*b - q) / b
        where p=win_rate, b=avg_win/avg_loss, q=1-p

        Falls back to linear sizing if insufficient trade history.
        """
        returns = self._trade_returns[-self.lookback:]
        if len(returns) < 10:
            # Not enough data for Kelly — fallback to linear
            return min(confidence * self.max_position, self.max_position)

        arr = np.array(returns)
        wins = arr[arr > 0]
        losses = arr[arr < 0]

        if len(wins) == 0:
            return 0.0
        if len(losses) == 0:
            return min(confidence * self.max_position, self.max_position)

        p = len(wins) / len(arr)
        q = 1 - p
        b = float(np.mean(wins) / np.mean(np.abs(losses)))

        if b == 0:
            return 0.0

        kelly = (p * b - q) / b
        # Scale by confidence and fraction, clamp to [0, max_position]
        size = max(0.0, kelly * fraction * confidence)
        return min(size, self.max_position)

    def reset(self) -> None:
        """Clear trade history."""
        self._trade_returns.clear()
