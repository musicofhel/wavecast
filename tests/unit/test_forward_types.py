"""Tests for forward testing data types."""

import uuid

from wavecast.forward.types import ForwardPrediction, ForwardTestSummary


def test_prediction_serialization():
    """Create a ForwardPrediction, to_dict, from_dict, verify roundtrip equality."""
    pred = ForwardPrediction(
        id=str(uuid.uuid4()),
        timestamp="2025-01-15T10:00:00+00:00",
        target_timestamp="2025-01-15T11:00:00+00:00",
        ticker="AAPL",
        interval="1h",
        horizon=1,
        predicted_direction=1,
        predicted_confidence=0.85,
        predicted_token=42,
        actual_return=0.02,
        actual_direction=1,
        correct=True,
        resolved_at="2025-01-15T11:05:00+00:00",
    )
    d = pred.to_dict()
    restored = ForwardPrediction.from_dict(d)
    assert restored.id == pred.id
    assert restored.timestamp == pred.timestamp
    assert restored.target_timestamp == pred.target_timestamp
    assert restored.ticker == pred.ticker
    assert restored.interval == pred.interval
    assert restored.horizon == pred.horizon
    assert restored.predicted_direction == pred.predicted_direction
    assert restored.predicted_confidence == pred.predicted_confidence
    assert restored.predicted_token == pred.predicted_token
    assert restored.actual_return == pred.actual_return
    assert restored.actual_direction == pred.actual_direction
    assert restored.correct == pred.correct
    assert restored.resolved_at == pred.resolved_at


def test_prediction_defaults():
    """Unresolved prediction has None for resolution fields."""
    pred = ForwardPrediction(
        id=str(uuid.uuid4()),
        timestamp="2025-01-15T10:00:00+00:00",
        target_timestamp="2025-01-15T11:00:00+00:00",
        ticker="MSFT",
        interval="1h",
        horizon=1,
        predicted_direction=-1,
        predicted_confidence=0.6,
        predicted_token=10,
    )
    assert pred.actual_return is None
    assert pred.actual_direction is None
    assert pred.correct is None
    assert pred.resolved_at is None


def test_summary_serialization():
    """Create ForwardTestSummary with known values, verify to_dict/from_dict roundtrip."""
    summary = ForwardTestSummary(
        test_name="test_run_1",
        start_time="2025-01-15T10:00:00+00:00",
        tickers=["AAPL", "MSFT"],
        intervals=["1h"],
        total_predictions=50,
        resolved_predictions=30,
        pending_predictions=20,
        accuracy=0.75,
        directional_accuracy=0.80,
        cumulative_pnl=0.15,
        max_drawdown=0.05,
        win_rate=0.60,
        per_ticker={"AAPL": {"accuracy": 0.8, "cumulative_pnl": 0.1}},
        per_interval={"1h": {"accuracy": 0.75, "cumulative_pnl": 0.15}},
    )
    d = summary.to_dict()
    restored = ForwardTestSummary.from_dict(d)
    assert restored.test_name == summary.test_name
    assert restored.total_predictions == 50
    assert restored.accuracy == 0.75
    assert restored.per_ticker == summary.per_ticker
    assert restored.per_interval == summary.per_interval


def test_prediction_resolution():
    """Create prediction, set resolution fields, verify correct is not None."""
    pred = ForwardPrediction(
        id=str(uuid.uuid4()),
        timestamp="2025-01-15T10:00:00+00:00",
        target_timestamp="2025-01-15T11:00:00+00:00",
        ticker="AAPL",
        interval="1h",
        horizon=1,
        predicted_direction=1,
        predicted_confidence=0.9,
        predicted_token=50,
    )
    assert pred.correct is None

    pred.actual_return = 0.02
    pred.actual_direction = 1
    pred.correct = pred.predicted_direction == pred.actual_direction
    pred.resolved_at = "2025-01-15T11:05:00+00:00"

    assert pred.correct is True
    assert pred.actual_return == 0.02
    assert pred.resolved_at is not None
