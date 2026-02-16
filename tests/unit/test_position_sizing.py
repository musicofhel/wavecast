"""Tests for position sizing methods."""

from __future__ import annotations

import pytest

from wavecast.signals.position import PositionSizer


class TestFixedSizing:
    def test_fixed_returns_max(self):
        sizer = PositionSizer(method="fixed", max_position=0.5)
        assert sizer.size(0.9) == 0.5
        assert sizer.size(0.1) == 0.5


class TestLinearSizing:
    def test_scales_with_confidence(self):
        sizer = PositionSizer(method="linear", max_position=1.0)
        assert sizer.size(0.5) == pytest.approx(0.5)
        assert sizer.size(1.0) == pytest.approx(1.0)

    def test_clamped_to_max(self):
        sizer = PositionSizer(method="linear", max_position=0.5)
        assert sizer.size(1.0) == pytest.approx(0.5)


class TestKellySizing:
    def test_kelly_with_positive_edge(self):
        """Kelly should produce positive size with positive edge."""
        sizer = PositionSizer(method="kelly", max_position=1.0)
        # Simulate 60% win rate, 1:1 payoff
        for _ in range(30):
            sizer.update(0.01)
        for _ in range(20):
            sizer.update(-0.01)
        size = sizer.size(0.8)
        assert size > 0

    def test_fractional_kelly_smaller_than_full(self):
        """Fractional Kelly should produce smaller positions."""
        full = PositionSizer(method="kelly", max_position=1.0)
        frac = PositionSizer(method="fractional_kelly", max_position=1.0, kelly_fraction=0.5)
        for _ in range(30):
            full.update(0.01)
            frac.update(0.01)
        for _ in range(20):
            full.update(-0.01)
            frac.update(-0.01)
        assert frac.size(0.8) <= full.size(0.8)


class TestNegativeEdge:
    def test_no_edge_zero_size(self):
        """All losses → Kelly should return 0."""
        sizer = PositionSizer(method="kelly", max_position=1.0)
        for _ in range(20):
            sizer.update(-0.01)
        assert sizer.size(0.8) == 0.0


class TestMaxClamp:
    def test_kelly_clamped(self):
        sizer = PositionSizer(method="kelly", max_position=0.3)
        for _ in range(45):
            sizer.update(0.02)
        for _ in range(5):
            sizer.update(-0.002)
        size = sizer.size(1.0)
        assert size <= 0.3


class TestMinThreshold:
    def test_below_minimum_returns_zero(self):
        sizer = PositionSizer(method="linear", max_position=1.0, min_position=0.3)
        assert sizer.size(0.2) == 0.0
        assert sizer.size(0.3) == 0.0  # Below threshold since 0.3 is not > 0.3
        assert sizer.size(0.5) > 0.0
