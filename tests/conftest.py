"""Shared pytest fixtures."""

import pytest

from tests.fixtures.generators import (
    make_feature_matrix,
    make_labels,
    make_mean_reverting_series,
    make_random_walk,
    make_sine_series,
    make_trending_series,
)
from wavecast.core.types import TimeSeries


@pytest.fixture
def sine_series() -> TimeSeries:
    return make_sine_series()


@pytest.fixture
def random_walk() -> TimeSeries:
    return make_random_walk()


@pytest.fixture
def trending_series() -> TimeSeries:
    return make_trending_series()


@pytest.fixture
def mean_reverting_series() -> TimeSeries:
    return make_mean_reverting_series()


@pytest.fixture
def labels():
    return make_labels()


@pytest.fixture
def feature_matrix():
    return make_feature_matrix()
