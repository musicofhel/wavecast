"""Forward testing data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ForwardPrediction:
    """A single forward test prediction with optional resolution data."""

    id: str  # uuid4 string
    timestamp: str  # ISO8601, when prediction was made
    target_timestamp: str  # ISO8601, when the predicted bar closes
    ticker: str
    interval: str
    horizon: int
    predicted_direction: int  # +1, -1, 0
    predicted_confidence: float
    predicted_token: int
    # Filled on resolution:
    actual_return: float | None = None
    actual_direction: int | None = None
    correct: bool | None = None
    resolved_at: str | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "target_timestamp": self.target_timestamp,
            "ticker": self.ticker,
            "interval": self.interval,
            "horizon": self.horizon,
            "predicted_direction": self.predicted_direction,
            "predicted_confidence": self.predicted_confidence,
            "predicted_token": self.predicted_token,
            "actual_return": self.actual_return,
            "actual_direction": self.actual_direction,
            "correct": self.correct,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ForwardPrediction:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            target_timestamp=data["target_timestamp"],
            ticker=data["ticker"],
            interval=data["interval"],
            horizon=data["horizon"],
            predicted_direction=data["predicted_direction"],
            predicted_confidence=data["predicted_confidence"],
            predicted_token=data["predicted_token"],
            actual_return=data.get("actual_return"),
            actual_direction=data.get("actual_direction"),
            correct=data.get("correct"),
            resolved_at=data.get("resolved_at"),
        )


@dataclass
class ForwardTestSummary:
    """Summary metrics for a forward test."""

    test_name: str
    start_time: str
    tickers: list[str] = field(default_factory=list)
    intervals: list[str] = field(default_factory=list)
    total_predictions: int = 0
    resolved_predictions: int = 0
    pending_predictions: int = 0
    accuracy: float = 0.0
    directional_accuracy: float = 0.0
    cumulative_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    per_ticker: dict[str, dict[str, float]] = field(default_factory=dict)
    per_interval: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "start_time": self.start_time,
            "tickers": self.tickers,
            "intervals": self.intervals,
            "total_predictions": self.total_predictions,
            "resolved_predictions": self.resolved_predictions,
            "pending_predictions": self.pending_predictions,
            "accuracy": self.accuracy,
            "directional_accuracy": self.directional_accuracy,
            "cumulative_pnl": self.cumulative_pnl,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "per_ticker": self.per_ticker,
            "per_interval": self.per_interval,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ForwardTestSummary:
        return cls(**data)
