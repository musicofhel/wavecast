"""DTW matching and similarity computation."""

from wavecast.dtw.matching import match_against_library, match_single
from wavecast.dtw.shape_dtw import shape_descriptor, shape_dtw_distance
from wavecast.dtw.similarity import pairwise_dtw_matrix, similarity_matrix
from wavecast.dtw.subsequence import subsequence_search

__all__ = [
    "match_against_library",
    "match_single",
    "pairwise_dtw_matrix",
    "shape_descriptor",
    "shape_dtw_distance",
    "similarity_matrix",
    "subsequence_search",
]
