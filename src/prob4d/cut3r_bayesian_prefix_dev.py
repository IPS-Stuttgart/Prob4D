"""Fixed, exploratory sparse-prefix residual conditioning for CUT3R.

All numerical coordinates are normalized by a prefix-only span. The generic
Gaussian operator is used algebraically; no metre-valued observation artifact
is exported from this experiment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .api.v2 import StackedObservationFactors, build_observation_gaussian_operator
from .dot_rope_cut3r_study import robust_fit_sim3
from .query_posterior import augment_observation_gaussian_operator, condition_gaussian_query

ARMS = (
    "cut3r_initial_alignment",
    "cut3r_full_prefix_alignment",
    "last_residual",
    "bayesian_iid",
    "bayesian_shared",
)
PREFIX_STOP = 5
PRIOR_STD = 0.10
NOISE_STD = 0.02
LENGTH_SCALE = 0.25
SHARED_CORRELATION = 0.80
CHI2_3_90 = 6.251388631170325


@dataclass(frozen=True)
class Rows:
    frame: np.ndarray
    identity: np.ndarray
    points: np.ndarray

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame)
        identity = np.asarray(self.identity)
        points = np.asarray(self.points, dtype=np.float64)
        if (
            frame.ndim != 1
            or identity.shape != frame.shape
            or points.shape != (len(frame), 3)
            or frame.dtype.kind not in "iu"
            or identity.dtype.kind not in "iu"
            or np.any(frame < 1)
            or np.any(identity < 0)
            or not np.isfinite(points).all()
        ):
            raise ValueError("invalid finite frame/identity/point rows")
        for name, value in (("frame", frame), ("identity", identity), ("points", points)):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def kernel(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    squared = np.sum((first[:, None] - second[None, :]) ** 2, axis=-1)
    return PRIOR_STD**2 * (0.5 + 0.5 * np.exp(-squared / (2 * LENGTH_SCALE**2)))


def _deduplicate(rows: Rows, values: np.ndarray) -> tuple[Rows, np.ndarray]:
    indices: dict[tuple[int, int], int] = {}
    for index, key in enumerate(zip(rows.frame.tolist(), rows.identity.tolist(), strict=True)):
        if key in indices:
            old = indices[key]
            if not np.array_equal(rows.points[old], rows.points[index]) or not np.array_equal(
                values[old], values[index]
            ):
                raise ValueError("conflicting duplicate observation")
        else:
            indices[key] = index
    selected = np.asarray([indices[key] for key in sorted(indices)], dtype=np.int64)
    return Rows(rows.frame[selected], rows.identity[selected], rows.points[selected]), values[
        selected
    ]


def bayesian_residual(
    observations: Rows,
    residual: np.ndarray,
    queries: np.ndarray,
    *,
    shared_correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Condition a fixed spatial residual GP through the existing Gaussian API."""
    values = np.asarray(residual, dtype=np.float64)
    query = np.asarray(queries, dtype=np.float64)
    if values.shape != observations.points.shape or not np.isfinite(values).all():
        raise ValueError("residual shape/values invalid")
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError("query shape/values invalid")
    if np.any(observations.frame > PREFIX_STOP):
        raise ValueError("future observation in prefix update")
    if not 0 <= shared_correlation < 1:
        raise ValueError("correlation must be in [0, 1)")
    observations, values = _deduplicate(observations, values)
    count = len(values)
    if count == 0:
        return np.zeros_like(query), np.broadcast_to(
            np.eye(3) * (PRIOR_STD**2 + NOISE_STD**2), (len(query), 3, 3)
        ).copy()

    conditional = np.broadcast_to(
        np.eye(3) * NOISE_STD**2 * (1 - shared_correlation), (count, 3, 3)
    ).copy()
    zero_gauge = np.zeros((count, 3, 7))
    stack = StackedObservationFactors(
        world_mean_m=np.zeros((count, 3)),
        conditional_world_covariance_m2=conditional,
        marginal_world_covariance_m2=conditional,
        gauge_jacobian=zero_gauge,
        gauge_prior_covariance=np.zeros((7, 7)),
        association_probability=np.ones(count),
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.ones(count),
        composite_weight=np.ones(count),
        point_ids=observations.identity,
        frame_indices=observations.frame,
        view_ids=("cam001",) * count,
        factor_ids=tuple(f"prefix-{index}" for index in range(count)),
        correlation_group_ids=("shared-camera",) * count,
        gauge_ids=("zero-gauge",),
        causal_frame_stop=PREFIX_STOP + 1,
    )
    base = build_observation_gaussian_operator(stack)
    prior = kernel(observations.points, observations.points)
    eigenvalues, vectors = np.linalg.eigh(prior)
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    # Keep a shared camera-bias nuisance in R, separate from the residual GP.
    shared = np.full((count, 1), NOISE_STD * np.sqrt(shared_correlation))
    factor = np.kron(np.concatenate((root, shared), axis=1), np.eye(3))
    operator = augment_observation_gaussian_operator(base, factor.reshape(count, 3, -1))
    cross = kernel(query, observations.points)
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    for row in cross:
        posterior = condition_gaussian_query(
            prior_mean=np.zeros(3),
            prior_covariance=np.eye(3) * PRIOR_STD**2,
            innovation=values,
            query_observation_cross_covariance=np.kron(row[None], np.eye(3)),
            innovation_operator=operator,
        )
        means.append(posterior.posterior_mean)
        covariances.append(posterior.posterior_covariance + np.eye(3) * NOISE_STD**2)
    return np.asarray(means).reshape(-1, 3), np.asarray(covariances).reshape(-1, 3, 3)


def last_residual_shift(prefix: Rows, residual: np.ndarray, identities: np.ndarray) -> np.ndarray:
    """Retain each identity's last valid residual, not just the final frame."""
    rows, values = _deduplicate(prefix, np.asarray(residual, dtype=np.float64))
    if np.any(rows.frame > PREFIX_STOP):
        raise ValueError("future observation in last-residual update")
    latest: dict[int, np.ndarray] = {}
    for identity, value in zip(rows.identity, values, strict=True):
        latest[int(identity)] = value
    return np.asarray([latest.get(int(identity), np.zeros(3)) for identity in identities])


def predict_arms(prefix: Rows, truth: np.ndarray, query: Rows) -> dict[str, Any]:
    values = np.asarray(truth, dtype=np.float64)
    if values.shape != prefix.points.shape or not np.isfinite(values).all():
        raise ValueError("invalid prefix truth")
    if np.any(prefix.frame > PREFIX_STOP) or np.any(query.frame <= PREFIX_STOP):
        raise ValueError("prefix/query time boundary violated")
    prefix, values = _deduplicate(prefix, values)
    initial = prefix.frame <= 2
    update = (prefix.frame >= 3) & (prefix.frame <= PREFIX_STOP)
    if np.count_nonzero(initial) < 6:
        raise ValueError("fewer than six initial alignment correspondences")
    if len(np.unique(prefix.frame[initial])) < 2:
        raise ValueError("initial alignment requires two nonempty frame clusters")
    initial_fit, _ = robust_fit_sim3(prefix.points[initial], values[initial])
    full_fit, _ = robust_fit_sim3(prefix.points, values)
    center = np.mean(values[initial], axis=0)
    span = float(np.linalg.norm(np.ptp(values[initial], axis=0)))
    if not np.isfinite(span) or span <= 1e-12:
        raise ValueError("invalid prefix-only normalization span")
    initial_query = (initial_fit.apply(query.points) - center) / span
    initial_prefix = (initial_fit.apply(prefix.points) - center) / span
    normalized_truth = (values - center) / span
    residual = normalized_truth - initial_prefix
    baseline_covariance = np.broadcast_to(
        np.eye(3) * (PRIOR_STD**2 + NOISE_STD**2), (len(query.points), 3, 3)
    ).copy()
    means = {
        "cut3r_initial_alignment": initial_query,
        "cut3r_full_prefix_alignment": (full_fit.apply(query.points) - center) / span,
        "last_residual": initial_query + last_residual_shift(prefix, residual, query.identity),
    }
    covariances = {name: baseline_covariance.copy() for name in means}
    update_rows = Rows(prefix.frame[update], prefix.identity[update], initial_prefix[update])
    for name, correlation in (("bayesian_iid", 0.0), ("bayesian_shared", SHARED_CORRELATION)):
        if np.count_nonzero(update) < 3 or len(np.unique(prefix.frame[update])) < 2:
            means[name] = initial_query.copy()
            covariances[name] = baseline_covariance.copy()
        else:
            shift, covariance = bayesian_residual(
                update_rows, residual[update], initial_query, shared_correlation=correlation
            )
            means[name] = initial_query + shift
            covariances[name] = covariance
    return {
        "means": means,
        "covariances": covariances,
        "normalization_center": center,
        "normalization_span": span,
        "prefix_count": len(values),
        "update_count": int(np.count_nonzero(update)),
        "bayesian_fallback": bool(
            np.count_nonzero(update) < 3 or len(np.unique(prefix.frame[update])) < 2
        ),
    }


def score_arms(
    predictions: Mapping[str, Any], query: Rows, truth: np.ndarray
) -> dict[str, dict[str, float]]:
    values = np.asarray(truth, dtype=np.float64)
    if values.shape != query.points.shape:
        raise ValueError("score truth shape differs from sealed query rows")
    normalized = (values - predictions["normalization_center"]) / predictions["normalization_span"]
    valid = np.isfinite(normalized).all(axis=1)
    if np.count_nonzero(valid) < 2 or len(np.unique(query.frame[valid])) != 2:
        raise ValueError("score requires two supported later frames and two rows")
    result: dict[str, dict[str, float]] = {}
    for name in ARMS:
        mean = np.asarray(predictions["means"][name])[valid]
        covariance = np.asarray(predictions["covariances"][name])[valid]
        error = normalized[valid] - mean
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("nonfinite prediction")
        np.linalg.cholesky(covariance)
        solved = np.linalg.solve(covariance, error[..., None])[..., 0]
        nees = np.sum(error * solved, axis=1)
        nll = 0.5 * (3 * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1] + nees) / 3
        frame_metrics = []
        for frame in sorted(np.unique(query.frame[valid])):
            selected = query.frame[valid] == frame
            frame_metrics.append(
                {
                    "rmse_prefix_span": float(
                        np.sqrt(np.mean(np.sum(error[selected] ** 2, axis=1)))
                    ),
                    "nll_per_coordinate": float(np.mean(nll[selected])),
                    "normalized_nees": float(np.mean(nees[selected]) / 3),
                    "coverage90": float(np.mean(nees[selected] <= CHI2_3_90)),
                    "full_coordinate_width90_prefix_span": float(
                        2
                        * 1.6448536269514722
                        * np.mean(np.sqrt(np.diagonal(covariance[selected], axis1=-2, axis2=-1)))
                    ),
                }
            )
        result[name] = {
            key: float(np.mean([frame[key] for frame in frame_metrics])) for key in frame_metrics[0]
        }
        result[name]["scored_rows"] = float(np.count_nonzero(valid))
    return result
