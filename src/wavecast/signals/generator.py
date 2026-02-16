"""Signal generator: converts token predictions to trading signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from wavecast.signals.types import SignalSeries, TradingSignal


class SignalGenerator:
    """Converts WaveletGPT softmax probabilities into trading signals.

    Maps token probabilities to directional signals with confidence scores
    by aggregating probability mass over quartile-based direction buckets.

    Args:
        vocab_size: Size of the token vocabulary
        alphabet_size: SAX alphabet size (for quartile boundary computation)
        confidence_threshold: Minimum confidence to emit a signal (below → direction=0)
        calibration_method: "none" or "temperature"
        temperature: Temperature for calibration (1.0 = no effect). Lower values
            sharpen distributions, higher values soften them.
    """

    def __init__(
        self,
        vocab_size: int,
        alphabet_size: int = 7,
        confidence_threshold: float = 0.0,
        calibration_method: str = "none",
        temperature: float = 1.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.alphabet_size = alphabet_size
        self.confidence_threshold = confidence_threshold
        self.calibration_method = calibration_method
        self.temperature = temperature

        # Compute quartile boundaries (same as token_eval.py)
        self._midpoint = vocab_size // 2
        self._quarter = vocab_size // 4

        # Token ranges for each direction
        self._up_start = self._midpoint + self._quarter  # UP: tokens >= this
        self._down_end = self._midpoint - self._quarter   # DOWN: tokens <= this

    def generate(
        self,
        probabilities: NDArray[np.float64],
        predicted_tokens: NDArray[np.int64],
        timestamps: NDArray[np.datetime64],
        ticker: str = "",
        horizon: int = 1,
    ) -> SignalSeries:
        """Generate trading signals from model output.

        Args:
            probabilities: Softmax probabilities (n_samples, vocab_size)
            predicted_tokens: Predicted token IDs (n_samples,)
            timestamps: Timestamps (n_samples,)
            ticker: Asset ticker
            horizon: Prediction horizon

        Returns:
            SignalSeries with one TradingSignal per sample
        """
        n = len(predicted_tokens)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)

        # Apply temperature calibration if configured
        if self.calibration_method == "temperature" and self.temperature != 1.0:
            probabilities = self._apply_temperature(probabilities)

        signals = []
        for i in range(n):
            proba = probabilities[i]
            direction, confidence = self._aggregate_probs(proba)

            # Apply threshold
            if confidence < self.confidence_threshold:
                direction = 0

            signals.append(
                TradingSignal(
                    timestamp=timestamps[i],
                    direction=direction,
                    confidence=confidence,
                    raw_probability=float(proba[predicted_tokens[i]]),
                    token_id=int(predicted_tokens[i]),
                    horizon=horizon,
                    ticker=ticker,
                )
            )

        return SignalSeries(signals=signals, ticker=ticker, horizon=horizon)

    def calibrate(
        self,
        probabilities: NDArray[np.float64],
        actual_tokens: NDArray[np.int64],
    ) -> float:
        """Fit temperature scaling on validation data.

        Finds optimal temperature T that minimizes negative log-likelihood
        of the actual tokens under the calibrated distribution.

        Args:
            probabilities: Raw softmax probabilities (n_samples, vocab_size)
            actual_tokens: Actual token IDs (n_samples,)

        Returns:
            Optimal temperature value
        """
        # Recover logits from probabilities
        logits = np.log(np.clip(probabilities, 1e-10, None))

        def nll(T: float) -> float:
            """Negative log-likelihood at temperature T."""
            scaled = logits / T
            # Numerically stable softmax
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exp_scaled = np.exp(shifted)
            log_probs = shifted - np.log(exp_scaled.sum(axis=1, keepdims=True))
            # NLL for actual tokens
            return float(-np.mean(log_probs[np.arange(len(actual_tokens)), actual_tokens]))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        self.calibration_method = "temperature"
        return self.temperature

    def _aggregate_probs(self, proba: NDArray[np.float64]) -> tuple[int, float]:
        """Aggregate softmax probabilities into direction and confidence.

        Sums probability mass over three quartile buckets:
        - UP: tokens >= midpoint + quarter
        - DOWN: tokens <= midpoint - quarter
        - FLAT: tokens in between

        Direction = argmax of {P(up), P(down), P(flat)}
        Confidence = max of {P(up), P(down), P(flat)}
        """
        p_up = float(np.sum(proba[self._up_start:]))
        p_down = float(np.sum(proba[:self._down_end + 1]))
        p_flat = 1.0 - p_up - p_down

        probs = {"up": p_up, "down": p_down, "flat": p_flat}
        best = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence = probs[best]

        direction_map = {"up": 1, "down": -1, "flat": 0}
        return direction_map[best], confidence

    def _apply_temperature(self, probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply temperature scaling to probabilities.

        Recovers logits via log(proba), divides by T, then re-applies softmax.
        """
        logits = np.log(np.clip(probabilities, 1e-10, None))
        scaled = logits / self.temperature
        # Numerically stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_scaled = np.exp(shifted)
        return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
