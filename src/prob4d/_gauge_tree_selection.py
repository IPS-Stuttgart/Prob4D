"""Selected covariance access for sparse gauge-tree priors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._gauge_tree_algebra import covariance_action
from ._gauge_tree_common import GAUGE_DIMENSION, FloatArray


def positions(
    all_gauge_ids: tuple[str, ...],
    requested: Sequence[str],
    *,
    name: str,
) -> tuple[int, ...]:
    values = tuple(str(value) for value in requested)
    if not values or any(not value for value in values):
        raise ValueError(f"{name} must contain nonempty gauge IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate gauge IDs")
    lookup = {gauge_id: index for index, gauge_id in enumerate(all_gauge_ids)}
    unknown = [gauge_id for gauge_id in values if gauge_id not in lookup]
    if unknown:
        raise ValueError(f"{name} reference unknown gauges: {unknown}")
    return tuple(lookup[gauge_id] for gauge_id in values)


def cross_covariance(
    all_gauge_ids: tuple[str, ...],
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    left_gauge_ids: Sequence[str],
    right_gauge_ids: Sequence[str],
) -> FloatArray:
    left = positions(all_gauge_ids, left_gauge_ids, name="left_gauge_ids")
    right = positions(all_gauge_ids, right_gauge_ids, name="right_gauge_ids")
    column_count = GAUGE_DIMENSION * len(right)
    basis = np.zeros((len(parents), GAUGE_DIMENSION, column_count), dtype=np.float64)
    identity = np.eye(GAUGE_DIMENSION, dtype=np.float64)
    for offset, position in enumerate(right):
        start = GAUGE_DIMENSION * offset
        basis[position, :, start : start + GAUGE_DIMENSION] = identity
    columns = np.asarray(covariance_action(parents, transitions, scales, basis))
    return np.concatenate([columns[position] for position in left], axis=0)


def selected_covariance(
    all_gauge_ids: tuple[str, ...],
    parents: np.ndarray,
    transitions: np.ndarray,
    scales: np.ndarray,
    gauge_ids: Sequence[str],
) -> FloatArray:
    result = cross_covariance(
        all_gauge_ids,
        parents,
        transitions,
        scales,
        gauge_ids,
        gauge_ids,
    )
    return 0.5 * (result + result.T)
