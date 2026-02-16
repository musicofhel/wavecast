"""Shapelet quality scoring — information gain, F-statistic, entropy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def entropy(labels: NDArray) -> float:
    """Shannon entropy of a label array.

    Parameters
    ----------
    labels : array of class labels (any hashable dtype)

    Returns
    -------
    Entropy in nats (natural log).
    """
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    # filter out zero-probability entries to avoid log(0)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def information_gain(
    values: NDArray[np.float64],
    labels: NDArray,
    split_point: float,
) -> float:
    """Information gain of a binary split on *values* at *split_point*.

    IG = H(parent) - weighted-avg H(children)
    """
    n = len(labels)
    if n == 0:
        return 0.0

    left_mask = values <= split_point
    right_mask = ~left_mask
    n_left = int(left_mask.sum())
    n_right = n - n_left

    if n_left == 0 or n_right == 0:
        return 0.0

    h_parent = entropy(labels)
    h_left = entropy(labels[left_mask])
    h_right = entropy(labels[right_mask])

    return h_parent - (n_left / n) * h_left - (n_right / n) * h_right


def f_statistic(values: NDArray[np.float64], labels: NDArray) -> float:
    """One-way ANOVA F-statistic across groups defined by *labels*.

    Returns 0.0 when fewer than 2 groups or zero within-group variance.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n = len(values)
    if k < 2 or n <= k:
        return 0.0

    grand_mean = float(np.mean(values))

    ss_between = 0.0
    ss_within = 0.0
    for lbl in unique_labels:
        group = values[labels == lbl]
        group_mean = float(np.mean(group))
        ss_between += len(group) * (group_mean - grand_mean) ** 2
        ss_within += float(np.sum((group - group_mean) ** 2))

    if ss_within == 0.0:
        return 0.0

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)

    if ms_within == 0.0:
        return 0.0

    return ms_between / ms_within


def score_shapelet(
    shapelet_values: NDArray[np.float64],
    series: NDArray[np.float64],
    labels: NDArray,
) -> float:
    """Score a candidate shapelet against a set of labelled series.

    For each position in *series* that can contain the shapelet, compute the
    Euclidean distance between the shapelet and the subsequence.  The minimum
    distance for the whole series is the "shapelet distance".  Then find the
    optimal split point on these distances and return the information gain.

    Parameters
    ----------
    shapelet_values : 1-D array, the candidate shapelet.
    series : 2-D array of shape (n_series, length) **or** 1-D coefficient
             array (treated as a single series whose sliding-window distances
             are computed).
    labels : 1-D array of class labels, one per series row (or per
             sliding-window position when *series* is 1-D).
    """
    m = len(shapelet_values)

    if series.ndim == 1:
        # Sliding-window distances within a single coefficient array.
        n = len(series)
        if n < m:
            return 0.0
        num_windows = n - m + 1
        # Truncate labels to match windows if necessary
        lab = labels[:num_windows] if len(labels) > num_windows else labels
        if len(lab) != num_windows:
            return 0.0
        dists = np.array([
            np.sqrt(np.sum((series[i : i + m] - shapelet_values) ** 2))
            for i in range(num_windows)
        ])
    else:
        # Multiple series — one distance per series (minimum over positions).
        n_series, length = series.shape
        if length < m:
            return 0.0
        dists = np.empty(n_series)
        for idx in range(n_series):
            row = series[idx]
            min_d = np.inf
            for j in range(length - m + 1):
                d = np.sqrt(np.sum((row[j : j + m] - shapelet_values) ** 2))
                if d < min_d:
                    min_d = d
            dists[idx] = min_d
        lab = labels

    if len(dists) < 2:
        return 0.0

    # Optimal split: try midpoints between sorted unique distances.
    sorted_dists = np.sort(np.unique(dists))
    if len(sorted_dists) < 2:
        return 0.0

    best_ig = 0.0
    for i in range(len(sorted_dists) - 1):
        sp = (sorted_dists[i] + sorted_dists[i + 1]) / 2.0
        ig = information_gain(dists, lab, sp)
        if ig > best_ig:
            best_ig = ig

    return best_ig
