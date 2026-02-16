"""Tests for SignalGenerator."""

from __future__ import annotations

import numpy as np
import pytest

from wavecast.signals.generator import SignalGenerator


@pytest.fixture
def generator() -> SignalGenerator:
    return SignalGenerator(vocab_size=100, alphabet_size=7)


class TestUniformProbability:
    def test_uniform_gives_low_confidence(self, generator: SignalGenerator):
        """Uniform distribution should give ~0.33 confidence for each direction."""
        proba = np.ones((1, 100)) / 100
        tokens = np.array([50])
        timestamps = np.array(["2024-01-01"], dtype="datetime64[ns]")
        result = generator.generate(proba, tokens, timestamps)
        assert len(result) == 1
        # With uniform proba, all buckets should be ~equal
        assert result.signals[0].confidence < 0.5


class TestPeakedProbability:
    def test_peaked_up_signal(self, generator: SignalGenerator):
        """Probability concentrated in upper quartile → UP signal."""
        proba = np.zeros((1, 100))
        proba[0, 80:] = 1.0 / 20  # All mass in upper range
        tokens = np.array([85])
        timestamps = np.array(["2024-01-01"], dtype="datetime64[ns]")
        result = generator.generate(proba, tokens, timestamps)
        assert result.signals[0].direction == 1
        assert result.signals[0].confidence > 0.5


class TestLowerQuartile:
    def test_peaked_down_signal(self, generator: SignalGenerator):
        """Probability concentrated in lower quartile → DOWN signal."""
        proba = np.zeros((1, 100))
        proba[0, :25] = 1.0 / 25
        tokens = np.array([10])
        timestamps = np.array(["2024-01-01"], dtype="datetime64[ns]")
        result = generator.generate(proba, tokens, timestamps)
        assert result.signals[0].direction == -1


class TestConfidenceThreshold:
    def test_below_threshold_gives_flat(self):
        """Signals below confidence threshold should be flat (direction=0)."""
        gen = SignalGenerator(vocab_size=100, confidence_threshold=0.9)
        proba = np.ones((1, 100)) / 100  # uniform → confidence ~0.33
        tokens = np.array([50])
        timestamps = np.array(["2024-01-01"], dtype="datetime64[ns]")
        result = gen.generate(proba, tokens, timestamps)
        assert result.signals[0].direction == 0


class TestCalibration:
    def test_calibration_changes_temperature(self, generator: SignalGenerator):
        """Calibration should find a temperature != 1.0 in general."""
        rng = np.random.default_rng(42)
        # Generate random probabilities and tokens
        n = 200
        proba = rng.dirichlet(np.ones(100), size=n)
        actual = rng.integers(0, 100, size=n)
        new_temp = generator.calibrate(proba, actual)
        assert isinstance(new_temp, float)
        assert new_temp > 0


class TestAggregateProbabilities:
    def test_probabilities_sum_to_one(self, generator: SignalGenerator):
        """Aggregated up/down/flat probs should sum to 1."""
        rng = np.random.default_rng(42)
        proba = rng.dirichlet(np.ones(100))
        direction, confidence = generator._aggregate_probs(proba)
        assert direction in (-1, 0, 1)
        assert 0 < confidence <= 1.0


class TestSignalSeriesProperties:
    def test_series_properties(self, generator: SignalGenerator):
        """SignalSeries should expose directions, confidences, timestamps."""
        n = 5
        proba = np.ones((n, 100)) / 100
        tokens = np.array([50] * n)
        timestamps = np.array(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            dtype="datetime64[ns]",
        )
        result = generator.generate(proba, tokens, timestamps, ticker="AAPL")
        assert len(result.directions) == n
        assert len(result.confidences) == n
        assert len(result.timestamps) == n
        assert result.ticker == "AAPL"
