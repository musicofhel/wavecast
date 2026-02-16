"""Forward test tracker: JSONL-based prediction logging and resolution."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wavecast.evaluation.metrics import max_drawdown as compute_max_drawdown
from wavecast.evaluation.metrics import win_rate as compute_win_rate
from wavecast.forward.types import ForwardPrediction, ForwardTestSummary

logger = logging.getLogger(__name__)


class ForwardTestTracker:
    """Tracks forward test predictions in JSONL format.

    Stores predictions in {log_dir}/{test_name}/predictions.jsonl.
    Loads existing predictions on init, appends new ones, rewrites on resolution.
    """

    def __init__(self, log_dir: Path, test_name: str) -> None:
        self.log_dir = log_dir
        self.test_name = test_name
        self._dir = log_dir / test_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pred_file = self._dir / "predictions.jsonl"
        self._predictions: list[ForwardPrediction] = []
        self._ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing predictions from JSONL file."""
        if not self._pred_file.exists():
            return
        with open(self._pred_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pred = ForwardPrediction.from_dict(json.loads(line))
                self._predictions.append(pred)
                self._ids.add(pred.id)
        logger.info("Loaded %d predictions from %s", len(self._predictions), self._pred_file)

    def _save_all(self) -> None:
        """Rewrite the full predictions file."""
        with open(self._pred_file, "w") as f:
            for pred in self._predictions:
                f.write(json.dumps(pred.to_dict()) + "\n")

    def log_prediction(self, pred: ForwardPrediction) -> None:
        """Append a new prediction.

        Raises ValueError if prediction ID already exists.
        """
        if pred.id in self._ids:
            raise ValueError(f"Duplicate prediction ID: {pred.id}")
        self._predictions.append(pred)
        self._ids.add(pred.id)
        # Append to file
        with open(self._pred_file, "a") as f:
            f.write(json.dumps(pred.to_dict()) + "\n")

    def resolve_pending(
        self, ticker: str, interval: str, actual_prices: pd.DataFrame
    ) -> int:
        """Resolve pending predictions whose target timestamp is in the past.

        Args:
            ticker: Filter predictions by ticker.
            interval: Filter predictions by interval.
            actual_prices: DataFrame with 'timestamp' and 'close' columns.

        Returns:
            Number of predictions resolved.
        """
        if actual_prices.empty:
            return 0

        now = datetime.datetime.now(datetime.UTC)
        resolved_count = 0

        # Build a timestamp -> close price lookup
        price_ts = pd.to_datetime(actual_prices["timestamp"])
        if price_ts.dt.tz is None:
            price_ts = price_ts.dt.tz_localize("UTC")
        prices_indexed = pd.Series(
            actual_prices["close"].values, index=price_ts
        ).sort_index()

        for pred in self._predictions:
            if pred.actual_return is not None:
                continue  # Already resolved
            if pred.ticker != ticker or pred.interval != interval:
                continue

            target_dt = pd.Timestamp(pred.target_timestamp)
            if target_dt.tzinfo is None:
                target_dt = target_dt.tz_localize("UTC")

            if target_dt > now:
                continue  # Not yet due

            # Find the prediction's base timestamp in prices
            pred_dt = pd.Timestamp(pred.timestamp)
            if pred_dt.tzinfo is None:
                pred_dt = pred_dt.tz_localize("UTC")

            # Find closest price at or before prediction time
            base_prices = prices_indexed[prices_indexed.index <= pred_dt]
            if base_prices.empty:
                continue
            base_price = float(base_prices.iloc[-1])

            # Find closest price at or after target time
            target_prices = prices_indexed[prices_indexed.index >= target_dt]
            if target_prices.empty:
                # Try closest before target
                target_prices = prices_indexed[prices_indexed.index <= target_dt]
                if target_prices.empty:
                    continue
                target_price = float(target_prices.iloc[-1])
            else:
                target_price = float(target_prices.iloc[0])

            if base_price == 0:
                continue

            actual_return = (target_price - base_price) / base_price
            actual_direction = int(np.sign(actual_return)) if actual_return != 0 else 0

            pred.actual_return = actual_return
            pred.actual_direction = actual_direction
            pred.correct = pred.predicted_direction == actual_direction
            pred.resolved_at = now.isoformat()
            resolved_count += 1

        if resolved_count > 0:
            self._save_all()

        return resolved_count

    def get_pending(self) -> list[ForwardPrediction]:
        """Return unresolved predictions."""
        return [p for p in self._predictions if p.actual_return is None]

    def get_resolved(self) -> list[ForwardPrediction]:
        """Return resolved predictions."""
        return [p for p in self._predictions if p.actual_return is not None]

    def get_summary(self) -> ForwardTestSummary:
        """Compute summary metrics from all predictions."""
        resolved = self.get_resolved()
        pending = self.get_pending()
        all_preds = self._predictions

        tickers = sorted(set(p.ticker for p in all_preds)) if all_preds else []
        intervals = sorted(set(p.interval for p in all_preds)) if all_preds else []

        start_time = min(p.timestamp for p in all_preds) if all_preds else ""

        if not resolved:
            return ForwardTestSummary(
                test_name=self.test_name,
                start_time=start_time,
                tickers=tickers,
                intervals=intervals,
                total_predictions=len(all_preds),
                resolved_predictions=0,
                pending_predictions=len(pending),
            )

        # Compute metrics from resolved predictions
        correct_count = sum(1 for p in resolved if p.correct)
        accuracy = correct_count / len(resolved)

        # Directional accuracy: only where actual_direction != 0
        dir_preds = [p for p in resolved if p.actual_direction != 0]
        if dir_preds:
            dir_correct = sum(
                1 for p in dir_preds if p.predicted_direction == p.actual_direction
            )
            dir_accuracy = dir_correct / len(dir_preds)
        else:
            dir_accuracy = 0.0

        # PnL: sum of actual_return * predicted_direction
        pnl_values = np.array(
            [p.actual_return * p.predicted_direction for p in resolved],
            dtype=np.float64,
        )
        cumulative_pnl = float(np.sum(pnl_values))

        # Max drawdown from cumulative PnL curve
        equity_curve = np.cumsum(pnl_values) + 1.0  # Start at 1.0
        mdd = float(compute_max_drawdown(equity_curve))

        # Win rate from PnL values
        wr = float(compute_win_rate(pnl_values))

        # Per-ticker metrics
        per_ticker: dict[str, dict[str, float]] = {}
        for t in tickers:
            t_resolved = [p for p in resolved if p.ticker == t]
            if t_resolved:
                t_correct = sum(1 for p in t_resolved if p.correct)
                t_dir = [p for p in t_resolved if p.actual_direction != 0]
                t_dir_correct = (
                    sum(1 for p in t_dir if p.predicted_direction == p.actual_direction)
                    if t_dir
                    else 0
                )
                t_pnl = sum(p.actual_return * p.predicted_direction for p in t_resolved)
                per_ticker[t] = {
                    "accuracy": t_correct / len(t_resolved),
                    "directional_accuracy": t_dir_correct / len(t_dir) if t_dir else 0.0,
                    "cumulative_pnl": t_pnl,
                }

        # Per-interval metrics
        per_interval: dict[str, dict[str, float]] = {}
        for iv in intervals:
            iv_resolved = [p for p in resolved if p.interval == iv]
            if iv_resolved:
                iv_correct = sum(1 for p in iv_resolved if p.correct)
                iv_dir = [p for p in iv_resolved if p.actual_direction != 0]
                iv_dir_correct = (
                    sum(1 for p in iv_dir if p.predicted_direction == p.actual_direction)
                    if iv_dir
                    else 0
                )
                iv_pnl = sum(p.actual_return * p.predicted_direction for p in iv_resolved)
                per_interval[iv] = {
                    "accuracy": iv_correct / len(iv_resolved),
                    "directional_accuracy": iv_dir_correct / len(iv_dir) if iv_dir else 0.0,
                    "cumulative_pnl": iv_pnl,
                }

        return ForwardTestSummary(
            test_name=self.test_name,
            start_time=start_time,
            tickers=tickers,
            intervals=intervals,
            total_predictions=len(all_preds),
            resolved_predictions=len(resolved),
            pending_predictions=len(pending),
            accuracy=accuracy,
            directional_accuracy=dir_accuracy,
            cumulative_pnl=cumulative_pnl,
            max_drawdown=mdd,
            win_rate=wr,
            per_ticker=per_ticker,
            per_interval=per_interval,
        )
