"""Shapelet discovery and library management."""

from wavecast.shapelets.clustering import deduplicate
from wavecast.shapelets.discovery import discover_shapelets
from wavecast.shapelets.library import ShapeletLibrary
from wavecast.shapelets.quality import entropy, f_statistic, information_gain, score_shapelet

__all__ = [
    "deduplicate",
    "discover_shapelets",
    "entropy",
    "f_statistic",
    "information_gain",
    "score_shapelet",
    "ShapeletLibrary",
]
