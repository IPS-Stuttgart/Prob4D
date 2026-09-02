#!/usr/bin/env python3
"""Recording-disjoint posterior rank--distortion on real cloth trajectories.

This experiment uses the public Tracking Cloth Deformation motion-capture CSVs.
It fits one local Gaussian query/observation model per cloth size and
recording-disjoint fold, decomposes the observation covariance as S=A+UU.T, and
compares equal-rank generalized-eigen, response-SVD, and covariance-PCA factors.

The script never exports raw trajectories and makes no learned-provider claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.posterior_rank_distortion import posterior_rank_distortion_frontier

SCHEMA = "prob4d.tracking-cloth-rank-distortion-real.v1"
CHI_SQUARE_3_90 = 6.251388631170325
_SIZE_PATTERN = re.compile(r"(?:^|[_-])(A[23])(?:[_\-.]|$)", re.IGNORECASE)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: str) -> float:
    stripped = value.strip()
    if not stripped:
        return float("nan")
    try:
        result = float(stripped)
    except ValueError:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


@dataclass(frozen=True, slots=True)
class Recording:
    relative_path: str
    size: str
    marker_count: int
    frames: np.ndarray
    timestamps_s: np.ndarray
    positions_m: np.ndarray
    source_sha256: str
    source_bytes: int
    original_coordinate_scale_to_m: float


@dataclass(frozen=True, slots=True)
class RecordingSamples:
    relative_path: str
    size: str
    observations_m: np.ndarray
    queries_m: np.ndarray
    horizon_seconds: np.ndarray
    candidate_window_count: int
    retained_window_count: int


class DenseInnovation:
    """Small dense reference implementing Prob4D's innovation-solve protocol."""

    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.dimension = int(self.covariance.shape[0])
        if self.dimension % 3:
            raise ValueError("observation dimension must be divisible by three")
        self.observation_count = self.dimension // 3

    def solve(self, value: object) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        return np.linalg.solve(self.covariance, raw.reshape(self.dimension, -1)).reshape(raw.shape)


def _find_header_and_data(rows: list[list[str]]) -> tuple[int, int]:
    for index, row in enumerate(rows[:20]):
        lowered = [field.strip().lower() for field in row]
        has_frame = any(field == "frame" or field.startswith("frame") for field in lowered)
        has_time = any("time" in field for field in lowered)
        if has_frame and has_time:
            return index, index + 1
    if len(rows) >= 6:
        return 4, 5
    raise ValueError("CSV does not contain the expected Motive header")


def _infer_coordinate_count(rows: list[list[str]], data_start: int) -> int:
    candidate_lengths = []
    for row in rows[data_start : data_start + 50]:
        if len(row) < 5:
            continue
        if not math.isfinite(_finite_float(row[0])) or not math.isfinite(_finite_float(row[1])):
            continue
        candidate_lengths.append(len(row) - 2)
    if not candidate_lengths:
        raise ValueError("CSV has no numeric trajectory rows")
    coordinate_count = max(candidate_lengths)
    coordinate_count -= coordinate_count % 3
    if coordinate_count < 3:
        raise ValueError("CSV has fewer than one 3-D marker")
    return coordinate_count


def _detect_scale_to_m(positions: np.ndarray, size: str) -> float:
    finite_rows = positions[np.all(np.isfinite(positions), axis=(1, 2))]
    if not len(finite_rows):
        raise ValueError("recording has no complete frame for unit detection")
    sample = finite_rows[np.linspace(0, len(finite_rows) - 1, min(64, len(finite_rows)), dtype=int)]
    extents = np.ptp(sample, axis=1)
    diagonals = np.linalg.norm(extents, axis=1)
    typical = float(np.quantile(diagonals, 0.90))
    expected = 0.50 if size == "A2" else 0.36
    if 0.25 * expected <= typical <= 3.0 * expected:
        return 1.0
    if 250.0 * expected <= typical <= 3000.0 * expected:
        return 0.001
    raise ValueError(f"implausible cloth extent {typical:g} for {size}")


def parse_recording(path: Path, dataset_root: Path) -> Recording:
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        rows = list(csv.reader(handle))
    _, data_start = _find_header_and_data(rows)
    coordinate_count = _infer_coordinate_count(rows, data_start)
    marker_count = coordinate_count // 3
    match = _SIZE_PATTERN.search(path.name)
    if not match:
        raise ValueError("filename does not encode A2/A3 size")
    size = match.group(1).upper()
    expected_markers = 20 if size == "A2" else 12
    if marker_count != expected_markers:
        raise ValueError(
            f"{size} recording has {marker_count} markers, expected {expected_markers}; "
            "rod/stick recordings are intentionally excluded"
        )

    frames: list[int] = []
    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    for row in rows[data_start:]:
        if len(row) < 2:
            continue
        frame_value = _finite_float(row[0])
        time_value = _finite_float(row[1])
        if not math.isfinite(frame_value) or not math.isfinite(time_value):
            continue
        coordinates = [_finite_float(value) for value in row[2 : 2 + coordinate_count]]
        if len(coordinates) < coordinate_count:
            coordinates.extend([float("nan")] * (coordinate_count - len(coordinates)))
        frames.append(int(round(frame_value)))
        timestamps.append(time_value)
        positions.append(np.asarray(coordinates, dtype=np.float64).reshape(marker_count, 3))
    if len(frames) < 32:
        raise ValueError("recording is too short")
    frame_array = np.asarray(frames, dtype=np.int64)
    time_array = np.asarray(timestamps, dtype=np.float64)
    position_array = np.stack(positions)
    if np.any(np.diff(frame_array) <= 0):
        raise ValueError("frame identifiers are not strictly increasing")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("timestamps are not strictly increasing")
    scale = _detect_scale_to_m(position_array, size)
    position_array = position_array * scale
    return Recording(
        relative_path=path.relative_to(dataset_root).as_posix(),
        size=size,
        marker_count=marker_count,
        frames=frame_array,
        timestamps_s=time_array,
        positions_m=position_array,
        source_sha256=_sha256_file(path),
        source_bytes=path.stat().st_size,
        original_coordinate_scale_to_m=scale,
    )


def make_samples(recording: Recording, protocol: dict[str, Any]) -> RecordingSamples:
    lag = int(protocol["lag_frames"])
    horizon = int(protocol["horizon_frames"])
    stride = int(protocol["stride_frames"])
    maximum = int(protocol["maximum_windows_per_recording"])
    positions = recording.positions_m
    observations: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    horizons: list[float] = []
    candidates = 0
    for current in range(lag, len(positions) - horizon, stride):
        candidates += 1
        selected = positions[[current - lag, current, current + horizon]]
        if not np.all(np.isfinite(selected)):
            continue
        dt_lag = float(recording.timestamps_s[current] - recording.timestamps_s[current - lag])
        dt_horizon = float(
            recording.timestamps_s[current + horizon] - recording.timestamps_s[current]
        )
        if not (dt_lag > 0.0 and dt_horizon > 0.0):
            continue
        observation = (positions[current] - positions[current - lag]) * (dt_horizon / dt_lag)
        query = positions[current + horizon].mean(axis=0) - positions[current].mean(axis=0)
        observations.append(observation)
        queries.append(query)
        horizons.append(dt_horizon)
    if not observations:
        raise ValueError("recording has no complete causal/query window")
    if len(observations) > maximum:
        indices = np.linspace(0, len(observations) - 1, maximum, dtype=int)
        observations = [observations[int(index)] for index in indices]
        queries = [queries[int(index)] for index in indices]
        horizons = [horizons[int(index)] for index in indices]
    return RecordingSamples(
        relative_path=recording.relative_path,
        size=recording.size,
        observations_m=np.stack(observations),
        queries_m=np.stack(queries),
        horizon_seconds=np.asarray(horizons, dtype=np.float64),
        candidate_window_count=candidates,
        retained_window_count=len(observations),
    )


def _fold_assignments(records: Iterable[RecordingSamples], folds: int) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(record.relative_path.encode("utf-8")).hexdigest(),
            record.relative_path,
        ),
    )
    return {record.relative_path: index % folds for index, record in enumerate(ordered)}


def _joint_covariance(
    queries: np.ndarray,
    observations: np.ndarray,
    shrinkage: float,
    ridge_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must have shape (samples, 3)")
    flat_observations = observations.reshape(len(observations), -1)
    data = np.concatenate((queries, flat_observations), axis=1)
    mean = data.mean(axis=0)
    centered = data - mean
    covariance = centered.T @ centered / max(len(data) - 1, 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    positive_diagonal = np.diag(covariance)
    scale = float(np.median(positive_diagonal[positive_diagonal > 0.0]))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("joint covariance has no positive scale")
    covariance += ridge_fraction * scale * np.eye(covariance.shape[0])
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    return mean, covariance


def _block_diagonal(covariance: np.ndarray) -> np.ndarray:
    if covariance.shape[0] % 3:
        raise ValueError("observation covariance is not composed of 3-D blocks")
    result = np.zeros_like(covariance)
    for start in range(0, covariance.shape[0], 3):
        result[start : start + 3, start : start + 3] = covariance[
            start : start + 3, start : start + 3
        ]
    return result


def _whiten_symmetric(root: np.ndarray, value: np.ndarray) -> np.ndarray:
    left = np.linalg.solve(root, value)
    return np.linalg.solve(root, left.T).T


def decompose_shared_covariance(
    covariance: np.ndarray,
    conditional_fraction: float,
    eigenvalue_relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    block = _block_diagonal(covariance)
    block_root = np.linalg.cholesky(block)
    normalized = _whiten_symmetric(block_root, covariance)
    minimum_generalized = float(np.linalg.eigvalsh(normalized)[0])
    if minimum_generalized <= 0.0:
        raise ValueError("observation covariance is not positive definite")
    beta = min(conditional_fraction, 0.5 * minimum_generalized)
    if beta <= 0.0:
        raise ValueError("conditional covariance fraction is nonpositive")
    conditional = beta * block
    remainder = 0.5 * ((covariance - conditional) + (covariance - conditional).T)
    eigenvalues, eigenvectors = np.linalg.eigh(remainder)
    largest = float(eigenvalues[-1])
    keep = eigenvalues > eigenvalue_relative_tolerance * largest
    if not np.any(keep):
        raise ValueError("shared covariance factor is empty")
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    np.testing.assert_allclose(
        conditional + factor @ factor.T,
        covariance,
        atol=1e-10 * max(1.0, float(np.max(np.diag(covariance)))),
        rtol=1e-9,
    )
    return conditional, factor, beta


def _posterior(
    prior: np.ndarray, cross: np.ndarray, innovation_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.linalg.solve(innovation_covariance, cross.T).T
    covariance = prior - gain @ cross.T
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    return gain, covariance


def _prediction_errors(
    queries: np.ndarray,
    observations: np.ndarray,
    query_mean: np.ndarray,
    observation_mean: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    innovation = observations.reshape(len(observations), -1) - observation_mean
    predictions = query_mean + innovation @ gain.T
    return queries - predictions


def _metrics(
    queries: np.ndarray,
    observations: np.ndarray,
    query_mean: np.ndarray,
    observation_mean: np.ndarray,
    gain: np.ndarray,
    posterior_covariance: np.ndarray,
) -> dict[str, Any]:
    errors = _prediction_errors(queries, observations, query_mean, observation_mean, gain)
    eigenvalues = np.linalg.eigvalsh(0.5 * (posterior_covariance + posterior_covariance.T))
    result: dict[str, Any] = {
        "sample_count": int(len(queries)),
        "query_rmse_m": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
        "maximum_query_error_m": float(np.max(np.linalg.norm(errors, axis=1))),
        "minimum_posterior_eigenvalue_m2": float(eigenvalues[0]),
        "posterior_valid": bool(eigenvalues[0] > 0.0),
        "mean_query_nll_nats": None,
        "mean_normalized_nees": None,
        "coverage_90": None,
    }
    if not result["posterior_valid"]:
        return result
    root = np.linalg.cholesky(posterior_covariance)
    whitened = np.linalg.solve(root, errors.T)
    nees = np.sum(whitened**2, axis=0)
    logdet = 2.0 * float(np.log(np.diag(root)).sum())
    nll = 0.5 * (nees + logdet + 3.0 * math.log(2.0 * math.pi))
    result.update(
        {
            "mean_query_nll_nats": float(np.mean(nll)),
            "mean_normalized_nees": float(np.mean(nees) / 3.0),
            "coverage_90": float(np.mean(nees <= CHI_SQUARE_3_90)),
        }
    )
    return result


def _method_from_observation_covariance(
    prior: np.ndarray,
    cross: np.ndarray,
    innovation_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gain = np.linalg.solve(innovation_covariance, cross.T).T
    covariance = prior - gain @ cross.T
    return gain, 0.5 * (covariance + covariance.T)


def _response_svd_basis(
    shared: np.ndarray,
    prior: np.ndarray,
    cross: np.ndarray,
    observation_covariance: np.ndarray,
) -> np.ndarray:
    rank = shared.shape[1]
    solved = np.linalg.solve(
        observation_covariance,
        np.concatenate((shared, cross.T), axis=1),
    )
    solved_cross = solved[:, rank:]
    posterior = prior - cross @ solved_cross
    posterior = 0.5 * (posterior + posterior.T)
    root = np.linalg.cholesky(posterior)
    response = shared.T @ solved_cross
    normalized_response = np.linalg.solve(root, response.T).T
    left, _, _ = np.linalg.svd(normalized_response, full_matrices=True)
    return left


def _candidate_evaluation(
    *,
    retained_rank: int,
    factor: np.ndarray,
    conditional: np.ndarray,
    prior: np.ndarray,
    cross: np.ndarray,
    test_q: np.ndarray,
    test_y: np.ndarray,
    query_mean: np.ndarray,
    observation_mean: np.ndarray,
    full_gain: np.ndarray,
    full_posterior: np.ndarray,
    full_means: np.ndarray,
) -> dict[str, Any]:
    observation_covariance = conditional + factor @ factor.T
    gain, posterior = _method_from_observation_covariance(
        prior,
        cross,
        observation_covariance,
    )
    record = _metrics(
        test_q,
        test_y,
        query_mean,
        observation_mean,
        gain,
        posterior,
    )
    contraction = 0.5 * ((full_posterior - posterior) + (full_posterior - posterior).T)
    root = np.linalg.cholesky(full_posterior)
    normalized = _whiten_symmetric(root, contraction)
    normalized = 0.5 * (normalized + normalized.T)
    eigenvalues = np.linalg.eigvalsh(normalized)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues[0]) < -1e-9 * scale:
        raise ValueError("factor projection produced non-PSD posterior contraction")
    innovation = test_y.reshape(len(test_y), -1) - observation_mean
    means = query_mean + innovation @ gain.T
    mean_difference = means - full_means
    whitened_mean_difference = np.linalg.solve(root, mean_difference.T)
    gain_denominator = max(float(np.linalg.norm(full_gain, ord="fro")), 1e-30)
    covariance_denominator = max(
        float(np.linalg.norm(full_posterior, ord="fro")),
        1e-30,
    )
    record.update(
        {
            "retained_rank": retained_rank,
            "payload_bytes": int(factor.nbytes),
            "normalized_covariance_trace_loss": max(
                float(np.trace(normalized)),
                0.0,
            ),
            "maximum_normalized_covariance_contraction": max(
                float(eigenvalues[-1]),
                0.0,
            ),
            "relative_gain_error": float(
                np.linalg.norm(gain - full_gain, ord="fro") / gain_denominator
            ),
            "relative_posterior_covariance_difference": float(
                np.linalg.norm(posterior - full_posterior, ord="fro") / covariance_denominator
            ),
            "maximum_realized_mean_difference_m": float(
                np.max(np.linalg.norm(mean_difference, axis=1))
            ),
            "heldout_normalized_mean_shift_risk": float(
                np.mean(np.sum(whitened_mean_difference**2, axis=0)) / prior.shape[0]
            ),
        }
    )
    return record


def evaluate_fold(
    train: list[RecordingSamples],
    test: list[RecordingSamples],
    protocol: dict[str, Any],
    fold_index: int,
    size: str,
) -> dict[str, Any]:
    train_y = np.concatenate([record.observations_m for record in train])
    train_q = np.concatenate([record.queries_m for record in train])
    test_y = np.concatenate([record.observations_m for record in test])
    test_q = np.concatenate([record.queries_m for record in test])
    mean, joint = _joint_covariance(
        train_q,
        train_y,
        float(protocol["joint_covariance_shrinkage"]),
        float(protocol["joint_covariance_ridge_fraction"]),
    )
    qdim = 3
    prior = joint[:qdim, :qdim]
    cross = joint[:qdim, qdim:]
    observation_covariance = joint[qdim:, qdim:]
    query_mean = mean[:qdim]
    observation_mean = mean[qdim:]
    conditional, shared, beta = decompose_shared_covariance(
        observation_covariance,
        float(protocol["maximum_conditional_block_fraction"]),
        float(protocol["factor_eigenvalue_relative_tolerance"]),
    )
    count = observation_covariance.shape[0] // 3
    original_rank = shared.shape[1]
    frontier = posterior_rank_distortion_frontier(
        shared.reshape(count, 3, original_rank),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=DenseInnovation(observation_covariance),
        numerical_relative_tolerance=float(protocol["rank_relative_tolerance"]),
    )
    retained_ranks = [int(value) for value in protocol["retained_ranks"]]
    if retained_ranks != sorted(set(retained_ranks)):
        raise ValueError("retained_ranks must be strictly increasing and unique")
    if not retained_ranks or retained_ranks[0] < 0:
        raise ValueError("retained_ranks must be nonempty and nonnegative")
    if retained_ranks[-1] > original_rank:
        raise ValueError("retained_ranks exceed the shared-factor rank")

    full_gain, full_posterior = _posterior(
        prior,
        cross,
        observation_covariance,
    )
    full_metrics = _metrics(
        test_q,
        test_y,
        query_mean,
        observation_mean,
        full_gain,
        full_posterior,
    )
    full_metrics.update(
        {
            "retained_rank": original_rank,
            "payload_bytes": int(shared.nbytes),
            "normalized_covariance_trace_loss": 0.0,
            "maximum_normalized_covariance_contraction": 0.0,
            "relative_gain_error": 0.0,
            "relative_posterior_covariance_difference": 0.0,
            "maximum_realized_mean_difference_m": 0.0,
            "heldout_normalized_mean_shift_risk": 0.0,
        }
    )
    test_innovation = test_y.reshape(len(test_y), -1) - observation_mean
    full_means = query_mean + test_innovation @ full_gain.T
    response_svd_basis = _response_svd_basis(
        shared,
        prior,
        cross,
        observation_covariance,
    )
    covariance_pca_basis = np.linalg.svd(
        shared,
        full_matrices=False,
    )[2].T

    rank_results: list[dict[str, Any]] = []
    optimality_tolerance = float(protocol["optimality_relative_tolerance"])
    for retained_rank in retained_ranks:
        point = frontier.point(retained_rank)
        optimal_factor = point.compressed_factor_m.reshape(
            observation_covariance.shape[0],
            retained_rank,
        )
        response_svd_factor = shared @ response_svd_basis[:, :retained_rank]
        covariance_pca_factor = shared @ covariance_pca_basis[:, :retained_rank]
        methods = {
            "optimal_generalized_eigen": _candidate_evaluation(
                retained_rank=retained_rank,
                factor=optimal_factor,
                conditional=conditional,
                prior=prior,
                cross=cross,
                test_q=test_q,
                test_y=test_y,
                query_mean=query_mean,
                observation_mean=observation_mean,
                full_gain=full_gain,
                full_posterior=full_posterior,
                full_means=full_means,
            ),
            "response_svd": _candidate_evaluation(
                retained_rank=retained_rank,
                factor=response_svd_factor,
                conditional=conditional,
                prior=prior,
                cross=cross,
                test_q=test_q,
                test_y=test_y,
                query_mean=query_mean,
                observation_mean=observation_mean,
                full_gain=full_gain,
                full_posterior=full_posterior,
                full_means=full_means,
            ),
            "covariance_pca": _candidate_evaluation(
                retained_rank=retained_rank,
                factor=covariance_pca_factor,
                conditional=conditional,
                prior=prior,
                cross=cross,
                test_q=test_q,
                test_y=test_y,
                query_mean=query_mean,
                observation_mean=observation_mean,
                full_gain=full_gain,
                full_posterior=full_posterior,
                full_means=full_means,
            ),
        }
        optimal_trace = float(
            methods["optimal_generalized_eigen"]["normalized_covariance_trace_loss"]
        )
        audit_scale = max(
            float(point.audited_normalized_covariance_trace_loss),
            optimal_trace,
            1.0,
        )
        if (
            abs(optimal_trace - float(point.audited_normalized_covariance_trace_loss))
            > optimality_tolerance * audit_scale
        ):
            raise ValueError("frontier point failed direct posterior-distortion audit")
        for baseline in ("response_svd", "covariance_pca"):
            baseline_trace = float(methods[baseline]["normalized_covariance_trace_loss"])
            comparison_scale = max(optimal_trace, baseline_trace, 1.0)
            if optimal_trace > (baseline_trace + optimality_tolerance * comparison_scale):
                raise ValueError(
                    f"generalized-eigen optimum lost to {baseline} at rank {retained_rank}"
                )
        rank_results.append(
            {
                "retained_rank": retained_rank,
                "frontier": point.summary(),
                "methods": methods,
            }
        )

    exact_rank = int(frontier.numerical_exact_rank)
    exact_matches = [record for record in rank_results if record["retained_rank"] == exact_rank]
    if len(exact_matches) != 1:
        raise ValueError("retained_ranks must contain the numerical exact rank exactly once")
    exact_method = exact_matches[0]["methods"]["optimal_generalized_eigen"]
    return {
        "size": size,
        "fold": fold_index,
        "train_recording_count": len(train),
        "test_recording_count": len(test),
        "train_window_count": int(len(train_q)),
        "test_window_count": int(len(test_q)),
        "test_recordings": sorted(record.relative_path for record in test),
        "median_horizon_seconds": float(
            np.median(np.concatenate([record.horizon_seconds for record in test]))
        ),
        "observation_count": count,
        "original_shared_rank": original_rank,
        "conditional_block_fraction": beta,
        "frontier": {
            "query_dimension": frontier.query_dimension,
            "numerical_exact_rank": exact_rank,
            "shared_precision_max_eigenvalue": (frontier.shared_precision_max_eigenvalue),
            "generalized_eigenvalues": frontier.generalized_eigenvalues.tolist(),
        },
        "full": full_metrics,
        "rank_results": rank_results,
        "exact_rank_full_parity": {
            "retained_rank": exact_rank,
            "relative_gain_error": exact_method["relative_gain_error"],
            "relative_posterior_covariance_error": exact_method[
                "relative_posterior_covariance_difference"
            ],
            "maximum_realized_mean_difference_m": exact_method[
                "maximum_realized_mean_difference_m"
            ],
        },
        "payload_bytes": {
            "full_shared_factor": int(shared.nbytes),
            "exact_frontier_factor": int(exact_method["payload_bytes"]),
            "cached_full_query_message": int(full_gain.nbytes + full_posterior.nbytes),
        },
    }


def _aggregate_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.asarray(
        [record["sample_count"] for record in records],
        dtype=np.float64,
    )
    values: dict[str, Any] = {
        "sample_count": int(weights.sum()),
        "posterior_valid_fold_count": int(
            sum(bool(record["posterior_valid"]) for record in records)
        ),
        "fold_count": len(records),
        "minimum_posterior_eigenvalue_m2": float(
            min(record["minimum_posterior_eigenvalue_m2"] for record in records)
        ),
        "maximum_query_error_m": float(max(record["maximum_query_error_m"] for record in records)),
        "payload_bytes_sum": int(sum(record["payload_bytes"] for record in records)),
        "normalized_covariance_trace_loss_mean": float(
            np.mean([record["normalized_covariance_trace_loss"] for record in records])
        ),
        "normalized_covariance_trace_loss_median": float(
            np.median([record["normalized_covariance_trace_loss"] for record in records])
        ),
        "normalized_covariance_trace_loss_maximum": float(
            max(record["normalized_covariance_trace_loss"] for record in records)
        ),
        "maximum_normalized_covariance_contraction": float(
            max(record["maximum_normalized_covariance_contraction"] for record in records)
        ),
        "maximum_relative_gain_error": float(
            max(record["relative_gain_error"] for record in records)
        ),
        "maximum_relative_posterior_covariance_difference": float(
            max(record["relative_posterior_covariance_difference"] for record in records)
        ),
        "maximum_realized_mean_difference_m": float(
            max(record["maximum_realized_mean_difference_m"] for record in records)
        ),
        "heldout_normalized_mean_shift_risk": float(
            np.sum(
                weights
                * np.asarray([record["heldout_normalized_mean_shift_risk"] for record in records])
            )
            / np.sum(weights)
        ),
    }
    rmse = np.asarray([record["query_rmse_m"] for record in records])
    values["query_rmse_m"] = float(math.sqrt(np.sum(weights * rmse**2) / np.sum(weights)))
    for metric in (
        "mean_query_nll_nats",
        "mean_normalized_nees",
        "coverage_90",
    ):
        valid_indices = [
            index for index, record in enumerate(records) if record[metric] is not None
        ]
        if not valid_indices:
            values[metric] = None
            continue
        valid_weights = weights[valid_indices]
        items = np.asarray(
            [records[index][metric] for index in valid_indices],
            dtype=np.float64,
        )
        values[metric] = float(np.sum(valid_weights * items) / np.sum(valid_weights))
    return values


def _aggregate(
    folds: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    total_test = sum(fold["test_window_count"] for fold in folds)
    retained_ranks = [int(value) for value in protocol["retained_ranks"]]
    aggregate_ranks: list[dict[str, Any]] = []
    strict_tolerance = float(protocol["strict_improvement_relative_tolerance"])
    maximum_optimality_violation = 0.0

    for retained_rank in retained_ranks:
        fold_rank_records = []
        for fold in folds:
            matches = [
                record
                for record in fold["rank_results"]
                if record["retained_rank"] == retained_rank
            ]
            if len(matches) != 1:
                raise ValueError("fold does not contain exactly one requested rank")
            fold_rank_records.append(matches[0])
        methods = {
            name: _aggregate_metric_records(
                [record["methods"][name] for record in fold_rank_records]
            )
            for name in (
                "optimal_generalized_eigen",
                "response_svd",
                "covariance_pca",
            )
        }
        response_improvements: list[float] = []
        response_relative_improvements: list[float] = []
        covariance_improvements: list[float] = []
        strict_response_count = 0
        strict_covariance_count = 0
        unique_count = 0
        exact_count = 0
        boundary_gaps: list[float] = []
        for record in fold_rank_records:
            point = record["frontier"]
            optimum = float(
                record["methods"]["optimal_generalized_eigen"]["normalized_covariance_trace_loss"]
            )
            response = float(record["methods"]["response_svd"]["normalized_covariance_trace_loss"])
            covariance = float(
                record["methods"]["covariance_pca"]["normalized_covariance_trace_loss"]
            )
            response_gap = response - optimum
            covariance_gap = covariance - optimum
            response_improvements.append(response_gap)
            covariance_improvements.append(covariance_gap)
            response_relative_improvements.append(response_gap / max(abs(response), 1e-30))
            response_scale = max(abs(response), abs(optimum), 1.0)
            covariance_scale = max(abs(covariance), abs(optimum), 1.0)
            if response_gap > strict_tolerance * response_scale:
                strict_response_count += 1
            if covariance_gap > strict_tolerance * covariance_scale:
                strict_covariance_count += 1
            maximum_optimality_violation = max(
                maximum_optimality_violation,
                optimum - response,
                optimum - covariance,
            )
            unique_count += int(bool(point["optimal_subspace_unique"]))
            exact_count += int(bool(point["exact_posterior"]))
            if point["boundary_generalized_eigengap"] is not None:
                boundary_gaps.append(float(point["boundary_generalized_eigengap"]))

        aggregate_ranks.append(
            {
                "retained_rank": retained_rank,
                "frontier": {
                    "optimal_subspace_unique_fold_count": unique_count,
                    "exact_posterior_fold_count": exact_count,
                    "minimum_boundary_generalized_eigengap": (
                        min(boundary_gaps) if boundary_gaps else None
                    ),
                },
                "methods": methods,
                "comparisons": {
                    "response_svd_strict_improvement_fold_count": (strict_response_count),
                    "response_svd_trace_improvement_mean": float(np.mean(response_improvements)),
                    "response_svd_trace_improvement_median": float(
                        np.median(response_improvements)
                    ),
                    "response_svd_relative_trace_improvement_mean": float(
                        np.mean(response_relative_improvements)
                    ),
                    "covariance_pca_strict_improvement_fold_count": (strict_covariance_count),
                    "covariance_pca_trace_improvement_mean": float(
                        np.mean(covariance_improvements)
                    ),
                    "covariance_pca_trace_improvement_median": float(
                        np.median(covariance_improvements)
                    ),
                },
            }
        )

    full_records = [fold["full"] for fold in folds]
    full = _aggregate_metric_records(full_records)
    parity = [fold["exact_rank_full_parity"] for fold in folds]
    full_bytes = sum(fold["payload_bytes"]["full_shared_factor"] for fold in folds)
    exact_bytes = sum(fold["payload_bytes"]["exact_frontier_factor"] for fold in folds)
    return {
        "fold_count": len(folds),
        "test_window_count": total_test,
        "retained_ranks": retained_ranks,
        "numerical_exact_ranks": sorted(
            {int(fold["frontier"]["numerical_exact_rank"]) for fold in folds}
        ),
        "original_shared_rank_min": min(int(fold["original_shared_rank"]) for fold in folds),
        "original_shared_rank_max": max(int(fold["original_shared_rank"]) for fold in folds),
        "maximum_optimality_violation": max(
            maximum_optimality_violation,
            0.0,
        ),
        "exact_rank_full_parity": {
            "maximum_relative_gain_error": max(
                float(value["relative_gain_error"]) for value in parity
            ),
            "maximum_relative_posterior_covariance_error": max(
                float(value["relative_posterior_covariance_error"]) for value in parity
            ),
            "maximum_realized_mean_difference_m": max(
                float(value["maximum_realized_mean_difference_m"]) for value in parity
            ),
        },
        "summed_full_shared_factor_bytes": full_bytes,
        "summed_exact_frontier_factor_bytes": exact_bytes,
        "exact_rank_shared_factor_payload_reduction_ratio": (
            float(full_bytes / exact_bytes) if exact_bytes else None
        ),
        "full": full,
        "ranks": aggregate_ranks,
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported protocol schema")
    required = {
        "fold_count",
        "lag_frames",
        "horizon_frames",
        "stride_frames",
        "maximum_windows_per_recording",
        "minimum_recordings_per_size",
        "joint_covariance_shrinkage",
        "joint_covariance_ridge_fraction",
        "maximum_conditional_block_fraction",
        "factor_eigenvalue_relative_tolerance",
        "rank_relative_tolerance",
        "retained_ranks",
        "optimality_relative_tolerance",
        "strict_improvement_relative_tolerance",
        "required_maximum_relative_parity_error",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing fields: {missing}")
    ranks = protocol["retained_ranks"]
    if (
        not isinstance(ranks, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ranks)
        or ranks != sorted(set(ranks))
        or not ranks
        or ranks[0] < 0
    ):
        raise ValueError("retained_ranks must be a nonempty increasing integer list")
    return protocol


def _summary_markdown(result: dict[str, Any]) -> str:
    if result["status"] != "evaluated-real-rank-distortion":
        return (
            "# Tracking Cloth rank--distortion evaluation\n\n"
            f"Status: **{result['status']}**\n\n"
            f"Reason: {result.get('reason', 'unspecified')}\n"
        )
    aggregate = result["aggregate"]
    accepted = result["inventory"]["accepted_recording_count"]
    csv_count = result["inventory"]["csv_file_count"]
    parity = aggregate["exact_rank_full_parity"]
    reduction = aggregate["exact_rank_shared_factor_payload_reduction_ratio"]
    lines = [
        "# Tracking Cloth posterior rank--distortion evaluation",
        "",
        "Status: **evaluated real trajectories**",
        "",
        f"- Parsed cloth-only recordings: {accepted} / {csv_count}",
        f"- Recording-disjoint folds: {aggregate['fold_count']}",
        f"- Held-out windows: {aggregate['test_window_count']}",
        (
            "- Original shared rank: "
            f"{aggregate['original_shared_rank_min']}–"
            f"{aggregate['original_shared_rank_max']}"
        ),
        f"- Evaluated retained ranks: {aggregate['retained_ranks']}",
        f"- Numerical exact ranks: {aggregate['numerical_exact_ranks']}",
        (
            "- Maximum theorem-optimality violation: "
            f"{aggregate['maximum_optimality_violation']:.3e}"
        ),
        (
            "- Exact-rank maximum relative gain / covariance error: "
            f"{parity['maximum_relative_gain_error']:.3e} / "
            f"{parity['maximum_relative_posterior_covariance_error']:.3e}"
        ),
        (
            "- Exact-rank maximum realized posterior-mean difference: "
            f"{1000.0 * parity['maximum_realized_mean_difference_m']:.3e} mm"
        ),
        f"- Exact-rank shared-factor payload reduction: {reduction:.2f}x",
        "",
        "## Rank frontier",
        "",
        (
            "| rank | optimal D | response-SVD D | mean trace improvement | "
            "strict folds | valid folds | optimal RMSE [mm] | SVD RMSE [mm] |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank_record in aggregate["ranks"]:
        optimal = rank_record["methods"]["optimal_generalized_eigen"]
        response = rank_record["methods"]["response_svd"]
        comparisons = rank_record["comparisons"]
        lines.append(
            "| "
            f"{rank_record['retained_rank']} | "
            f"{optimal['normalized_covariance_trace_loss_mean']:.6g} | "
            f"{response['normalized_covariance_trace_loss_mean']:.6g} | "
            f"{comparisons['response_svd_trace_improvement_mean']:.6g} | "
            f"{comparisons['response_svd_strict_improvement_fold_count']}/"
            f"{aggregate['fold_count']} | "
            f"{optimal['posterior_valid_fold_count']}/"
            f"{aggregate['fold_count']} | "
            f"{1000.0 * optimal['query_rmse_m']:.3f} | "
            f"{1000.0 * response['query_rmse_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The generalized-eigen method is globally optimal only for "
                "the registered normalized posterior-covariance trace "
                "contraction within each fitted `U -> U V` family. Held-out "
                "RMSE, NLL, NEES, coverage, posterior validity, eigengaps, "
                "and non-unique boundaries are reported rather than inferred "
                "from that theorem."
            ),
            "",
            (
                "The experiment is a recording-disjoint local-Gaussian "
                "mechanism study on real motion-capture trajectories. It does "
                "not establish a learned 4-D provider, deployment calibration, "
                "or BayesianPhysTwin/Causal4D physical benefit."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(
    dataset_root: Path,
    protocol_path: Path,
    output_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol = _load_protocol(protocol_path)
    csv_paths = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"
    )
    inventory: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "csv_file_count": len(csv_paths),
        "reported_verified_recording_count": 120,
        "accepted_recording_count": 0,
        "accepted_by_size": {},
        "excluded": [],
        "files": [],
    }
    recordings: list[Recording] = []
    samples: list[RecordingSamples] = []
    for path in csv_paths:
        relative = path.relative_to(dataset_root).as_posix()
        file_record = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        inventory["files"].append(file_record)
        try:
            recording = parse_recording(path, dataset_root)
            recording_samples = make_samples(recording, protocol)
        except (OSError, UnicodeError, ValueError) as exc:
            inventory["excluded"].append({"relative_path": relative, "reason": str(exc)})
            continue
        recordings.append(recording)
        samples.append(recording_samples)
        inventory["accepted_by_size"][recording.size] = (
            int(inventory["accepted_by_size"].get(recording.size, 0)) + 1
        )
    inventory["accepted_recording_count"] = len(recordings)
    inventory["dataset_manifest_sha256"] = _sha256_bytes(_canonical_bytes(inventory["files"]))
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "status": "not-evaluated",
        "inventory": {
            key: value for key, value in inventory.items() if key not in {"files", "excluded"}
        },
        "claim_boundary": {
            "real_motion_capture_trajectories": True,
            "training_only_joint_covariance_fit": True,
            "recording_disjoint_heldout_evaluation": True,
            "rank_distortion_theorem_evaluated": True,
            "learned_4d_provider_evaluated": False,
            "deployment_covariance_calibration_claimed": False,
            "arbitrary_task_loss_optimality_claimed": False,
            "recursive_exactness_claimed": False,
            "bayesian_phystwin_benefit_claimed": False,
            "causal4d_benefit_claimed": False,
        },
    }
    folds = int(protocol["fold_count"])
    minimum = int(protocol["minimum_recordings_per_size"])
    evaluations: list[dict[str, Any]] = []
    try:
        for size in ("A2", "A3"):
            group = [record for record in samples if record.size == size]
            if len(group) < minimum:
                raise ValueError(f"{size} has {len(group)} accepted recordings, needs {minimum}")
            assignments = _fold_assignments(group, folds)
            for fold_index in range(folds):
                train = [
                    record for record in group if assignments[record.relative_path] != fold_index
                ]
                test = [
                    record for record in group if assignments[record.relative_path] == fold_index
                ]
                evaluations.append(evaluate_fold(train, test, protocol, fold_index, size))
        aggregate = _aggregate(evaluations, protocol)
        parity = aggregate["exact_rank_full_parity"]
        if parity["maximum_relative_gain_error"] > float(
            protocol["required_maximum_relative_parity_error"]
        ):
            raise ValueError("exact-rank posterior gain parity exceeded the protocol limit")
        if parity["maximum_relative_posterior_covariance_error"] > float(
            protocol["required_maximum_relative_parity_error"]
        ):
            raise ValueError("exact-rank posterior covariance parity exceeded the protocol limit")
        if aggregate["maximum_optimality_violation"] > float(
            protocol["optimality_relative_tolerance"]
        ):
            raise ValueError("the registered global-optimality inequality was violated")
        result.update(
            {
                "status": "evaluated-real-rank-distortion",
                "evaluations": evaluations,
                "aggregate": aggregate,
            }
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.update(
            {
                "status": "technical-or-support-negative",
                "reason": str(exc),
                "evaluations": evaluations,
            }
        )

    result_bytes = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    manifest = {
        "schema": "prob4d.tracking-cloth-rank-distortion-manifest.v1",
        "source_revision": source_revision,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "result_sha256": _sha256_bytes(result_bytes),
        "inventory_sha256": _sha256_file(output_dir / "inventory.json"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dataset_manifest_sha256": inventory["dataset_manifest_sha256"],
        "raw_data_copied_to_output": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not args.dataset_root.is_dir():
        raise SystemExit(f"dataset root is unavailable: {args.dataset_root}")
    result = run(
        args.dataset_root.resolve(),
        args.protocol,
        args.output_dir,
        args.source_revision,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(args.output_dir),
                "source_revision": args.source_revision,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "evaluated-real-rank-distortion" else 3


if __name__ == "__main__":
    raise SystemExit(main())
