"""Tests for forward test tracker."""

import datetime
import uuid

import pandas as pd
import pytest

from wavecast.forward.tracker import ForwardTestTracker
from wavecast.forward.types import ForwardPrediction


def _make_prediction(
    ticker: str = "AAPL",
    interval: str = "1h",
    direction: int = 1,
    confidence: float = 0.8,
    hours_ago: int = 3,
    target_hours_ago: int = 2,
    pred_id: str | None = None,
) -> ForwardPrediction:
    """Helper to create a prediction with timestamps relative to now."""
    now = datetime.datetime.now(datetime.UTC)
    ts = (now - datetime.timedelta(hours=hours_ago)).isoformat()
    target_ts = (now - datetime.timedelta(hours=target_hours_ago)).isoformat()
    return ForwardPrediction(
        id=pred_id or str(uuid.uuid4()),
        timestamp=ts,
        target_timestamp=target_ts,
        ticker=ticker,
        interval=interval,
        horizon=1,
        predicted_direction=direction,
        predicted_confidence=confidence,
        predicted_token=42,
    )


def _make_price_df(
    base_price: float = 100.0,
    target_price: float = 102.0,
    hours_ago_start: int = 4,
    hours_ago_end: int = 1,
    num_points: int = 4,
) -> pd.DataFrame:
    """Helper to create a price DataFrame spanning a time range."""
    now = datetime.datetime.now(datetime.UTC)
    timestamps = pd.date_range(
        start=now - datetime.timedelta(hours=hours_ago_start),
        end=now - datetime.timedelta(hours=hours_ago_end),
        periods=num_points,
        tz="UTC",
    )
    prices = [
        base_price + (target_price - base_price) * i / (num_points - 1)
        for i in range(num_points)
    ]
    return pd.DataFrame({"timestamp": timestamps, "close": prices})


def test_log_prediction(tmp_path):
    """log_prediction writes to file and memory."""
    tracker = ForwardTestTracker(tmp_path, "test1")
    pred = _make_prediction()
    tracker.log_prediction(pred)

    assert len(tracker._predictions) == 1
    # Verify file has 1 line
    lines = (tmp_path / "test1" / "predictions.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1


def test_resolve_pending(tmp_path):
    """Resolve a prediction with target_timestamp in the past."""
    tracker = ForwardTestTracker(tmp_path, "test2")

    # Use fixed timestamps to avoid race between prediction and price creation
    now = datetime.datetime.now(datetime.UTC)
    pred_ts = (now - datetime.timedelta(hours=3)).isoformat()
    target_ts = (now - datetime.timedelta(hours=2)).isoformat()
    pred = ForwardPrediction(
        id=str(uuid.uuid4()),
        timestamp=pred_ts,
        target_timestamp=target_ts,
        ticker="AAPL",
        interval="1h",
        horizon=1,
        predicted_direction=1,
        predicted_confidence=0.8,
        predicted_token=42,
    )
    tracker.log_prediction(pred)

    # Price data: 100 well before prediction, 102 well after target
    prices = pd.DataFrame({
        "timestamp": [
            now - datetime.timedelta(hours=4),
            now - datetime.timedelta(hours=3),
            now - datetime.timedelta(hours=2),
            now - datetime.timedelta(hours=1),
        ],
        "close": [100.0, 100.0, 102.0, 102.0],
    })
    resolved = tracker.resolve_pending("AAPL", "1h", prices)

    assert resolved == 1
    assert pred.actual_return is not None
    assert pred.actual_return == pytest.approx(0.02, abs=0.001)
    assert pred.actual_direction == 1
    assert pred.correct is True
    assert pred.resolved_at is not None


def test_get_pending(tmp_path):
    """Log 2 predictions, resolve 1, get_pending returns 1."""
    tracker = ForwardTestTracker(tmp_path, "test3")
    pred1 = _make_prediction(hours_ago=3, target_hours_ago=2)
    pred2 = _make_prediction(hours_ago=1, target_hours_ago=-1)  # Future target
    tracker.log_prediction(pred1)
    tracker.log_prediction(pred2)

    prices = _make_price_df()
    tracker.resolve_pending("AAPL", "1h", prices)

    pending = tracker.get_pending()
    assert len(pending) == 1
    assert pending[0].id == pred2.id


def test_get_resolved(tmp_path):
    """Log 2 predictions, resolve 1, get_resolved returns 1."""
    tracker = ForwardTestTracker(tmp_path, "test4")
    pred1 = _make_prediction(hours_ago=3, target_hours_ago=2)
    pred2 = _make_prediction(hours_ago=1, target_hours_ago=-1)  # Future target
    tracker.log_prediction(pred1)
    tracker.log_prediction(pred2)

    prices = _make_price_df()
    tracker.resolve_pending("AAPL", "1h", prices)

    resolved = tracker.get_resolved()
    assert len(resolved) == 1
    assert resolved[0].id == pred1.id


def test_get_summary(tmp_path):
    """Log and resolve 3 predictions with known returns, verify summary."""
    tracker = ForwardTestTracker(tmp_path, "test5")

    # 3 predictions: 2 correct, 1 wrong
    p1 = _make_prediction(direction=1, hours_ago=5, target_hours_ago=4)
    p2 = _make_prediction(direction=1, hours_ago=4, target_hours_ago=3)
    p3 = _make_prediction(direction=-1, hours_ago=3, target_hours_ago=2)
    tracker.log_prediction(p1)
    tracker.log_prediction(p2)
    tracker.log_prediction(p3)

    # Resolve all with 2% positive return (direction +1 correct, -1 wrong)
    prices = _make_price_df(base_price=100.0, target_price=102.0, hours_ago_start=6, hours_ago_end=1)
    tracker.resolve_pending("AAPL", "1h", prices)

    summary = tracker.get_summary()
    assert summary.test_name == "test5"
    assert summary.total_predictions == 3
    assert summary.resolved_predictions == 3
    assert summary.pending_predictions == 0
    # 2 out of 3 predicted direction=+1 matching actual direction=+1
    assert summary.accuracy == pytest.approx(2 / 3, abs=0.01)
    assert summary.tickers == ["AAPL"]
    assert summary.intervals == ["1h"]
    assert "AAPL" in summary.per_ticker


def test_empty_state(tmp_path):
    """New tracker with no predictions has zeroed summary."""
    tracker = ForwardTestTracker(tmp_path, "test_empty")
    summary = tracker.get_summary()
    assert summary.total_predictions == 0
    assert summary.resolved_predictions == 0
    assert summary.pending_predictions == 0
    assert summary.accuracy == 0.0
    assert summary.tickers == []


def test_multi_ticker(tmp_path):
    """Predictions for 2 tickers, resolve one, verify per_ticker breakdown."""
    tracker = ForwardTestTracker(tmp_path, "test_multi")
    p1 = _make_prediction(ticker="AAPL", direction=1, hours_ago=3, target_hours_ago=2)
    p2 = _make_prediction(ticker="MSFT", direction=1, hours_ago=3, target_hours_ago=2)
    tracker.log_prediction(p1)
    tracker.log_prediction(p2)

    prices = _make_price_df(base_price=100.0, target_price=102.0)

    # Only resolve AAPL
    tracker.resolve_pending("AAPL", "1h", prices)

    summary = tracker.get_summary()
    assert summary.total_predictions == 2
    assert summary.resolved_predictions == 1
    assert summary.pending_predictions == 1
    assert "AAPL" in summary.per_ticker
    assert "MSFT" not in summary.per_ticker


def test_duplicate_prevention(tmp_path):
    """Log prediction, try to log with same id, expect ValueError."""
    tracker = ForwardTestTracker(tmp_path, "test_dup")
    pred_id = str(uuid.uuid4())
    p1 = _make_prediction(pred_id=pred_id)
    tracker.log_prediction(p1)

    p2 = _make_prediction(pred_id=pred_id)
    with pytest.raises(ValueError, match="Duplicate prediction ID"):
        tracker.log_prediction(p2)
