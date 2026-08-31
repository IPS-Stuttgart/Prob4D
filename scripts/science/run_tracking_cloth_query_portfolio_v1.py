#!/usr/bin/env python3
"""Measure exact query-portfolio compression and cache break-even on real cloth data.

The experiment uses recording-disjoint folds from the public Tracking Cloth
Deformation motion-capture dataset. It compares the exact full shared factor,
the posterior-preserving factor, a direct cached Gaussian query message, and a
PSD-consistent generic spectral approximation of the complete joint covariance.
Raw trajectories are never copied to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)

SCHEMA = "prob4d.tracking-cloth-query-portfolio.v1"
_SIZE_PATTERN = re.compile(r"(?:^|[_-])(A[23])(?:[_\-.]|$)", re.IGNORECASE)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: str) -> float:
    stripped = value.strip().replace(",", ".")
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
    timestamps_s: np.ndarray
    positions_m: np.ndarray
    source_sha256: str
    source_bytes: int


@dataclass(frozen=True, slots=True)
class Samples:
    relative_path: str
    size: str
    observations_m: np.ndarray
    future_displacements_m: np.ndarray
    horizon_seconds: np.ndarray


def _find_header_and_data(rows: list[list[str]]) -> tuple[int, int]:
    for index, row in enumerate(rows[:20]):
        lowered = [field.strip().lower() for field in row]
        has_frame = any(
            field == "frame" or field.startswith("frame") for field in lowered
        )
        has_time = any("time" in field for field in lowered)
        if has_frame and has_time:
            return index, index + 1
    if len(rows) >= 6:
        return 4, 5
    raise ValueError("CSV does not contain the expected Motive header")


def _infer_coordinate_count(rows: list[list[str]], data_start: int) -> int:
    candidate_lengths: list[int] = []
    for row in rows[data_start : data_start + 50]:
        if len(row) < 5:
            continue
        if not math.isfinite(_finite_float(row[0])) or not math.isfinite(
            _finite_float(row[1])
        ):
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
    finite = positions[np.all(np.isfinite(positions), axis=(1, 2))]
    if not len(finite):
        raise ValueError("recording has no complete frame for unit detection")
    sample = finite[
        np.linspace(0, len(finite) - 1, min(64, len(finite)), dtype=int)
    ]
    typical = float(
        np.quantile(np.linalg.norm(np.ptp(sample, axis=1), axis=1), 0.90)
    )
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
            "rod/stick recordings are excluded before modeling"
        )

    frames: list[int] = []
    times: list[float] = []
    positions: list[np.ndarray] = []
    for row in rows[data_start:]:
        if len(row) < 2:
            continue
        frame = _finite_float(row[0])
        stamp = _finite_float(row[1])
        if not math.isfinite(frame) or not math.isfinite(stamp):
            continue
        coordinates = [
            _finite_float(value) for value in row[2 : 2 + coordinate_count]
        ]
        if len(coordinates) < coordinate_count:
            coordinates.extend(
                [float("nan")] * (coordinate_count - len(coordinates))
            )
        frames.append(int(round(frame)))
        times.append(stamp)
        positions.append(np.asarray(coordinates).reshape(marker_count, 3))
    if len(frames) < 32:
        raise ValueError("recording is too short")
    frame_array = np.asarray(frames, dtype=np.int64)
    time_array = np.asarray(times, dtype=np.float64)
    position_array = np.stack(positions).astype(np.float64)
    if np.any(np.diff(frame_array) <= 0):
        raise ValueError("frame identifiers are not strictly increasing")
    if np.any(np.diff(time_array) <= 0):
        raise ValueError("timestamps are not strictly increasing")
    position_array *= _detect_scale_to_m(position_array, size)
    return Recording(
        relative_path=path.relative_to(dataset_root).as_posix(),
        size=size,
        marker_count=marker_count,
        timestamps_s=time_array,
        positions_m=position_array,
        source_sha256=_sha256_file(path),
        source_bytes=path.stat().st_size,
    )


def make_samples(recording: Recording, protocol: dict[str, Any]) -> Samples:
    lag = int(protocol["lag_frames"])
    horizon = int(protocol["horizon_frames"])
    stride = int(protocol["stride_frames"])
    maximum = int(protocol["maximum_windows_per_recording"])
    observations: list[np.ndarray] = []
    future: list[np.ndarray] = []
    horizons: list[float] = []
    for current in range(lag, len(recording.positions_m) - horizon, stride):
        selected = recording.positions_m[[current - lag, current, current + horizon]]
        if not np.all(np.isfinite(selected)):
            continue
        dt_lag = float(
            recording.timestamps_s[current]
            - recording.timestamps_s[current - lag]
        )
        dt_horizon = float(
            recording.timestamps_s[current + horizon]
            - recording.timestamps_s[current]
        )
        if not (dt_lag > 0.0 and dt_horizon > 0.0):
            continue
        observations.append(
            (recording.positions_m[current] - recording.positions_m[current - lag])
            * (dt_horizon / dt_lag)
        )
        future.append(
            recording.positions_m[current + horizon]
            - recording.positions_m[current]
        )
        horizons.append(dt_horizon)
    if not observations:
        raise ValueError("recording has no complete causal/query window")
    if len(observations) > maximum:
        indices = np.linspace(0, len(observations) - 1, maximum, dtype=int)
        observations = [observations[int(index)] for index in indices]
        future = [future[int(index)] for index in indices]
        horizons = [horizons[int(index)] for index in indices]
    return Samples(
        relative_path=recording.relative_path,
        size=recording.size,
        observations_m=np.stack(observations),
        future_displacements_m=np.stack(future),
        horizon_seconds=np.asarray(horizons),
    )


def _fold_assignments(records: list[Samples], folds: int) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(record.relative_path.encode()).hexdigest(),
            record.relative_path,
        ),
    )
    return {
        record.relative_path: index % folds
        for index, record in enumerate(ordered)
    }


def select_query_indices(marker_count: int, query_count: int) -> np.ndarray:
    if not 1 <= query_count <= marker_count:
        raise ValueError("query_count must lie between one and marker_count")
    if query_count == marker_count:
        return np.arange(marker_count, dtype=np.int64)
    indices = np.rint(
        np.linspace(0, marker_count - 1, query_count)
    ).astype(np.int64)
    if len(np.unique(indices)) != query_count:
        raise RuntimeError("deterministic marker selection produced duplicates")
    return indices


def _joint_covariance(
    future: np.ndarray,
    observations: np.ndarray,
    shrinkage: float,
    ridge_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    flat_q = future.reshape(len(future), -1)
    flat_y = observations.reshape(len(observations), -1)
    data = np.concatenate((flat_q, flat_y), axis=1)
    mean = data.mean(axis=0)
    centered = data - mean
    covariance = centered.T @ centered / max(len(data) - 1, 1)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(
        np.diag(covariance)
    )
    diagonal = np.diag(covariance)
    positive = diagonal[diagonal > 0.0]
    if not len(positive):
        raise ValueError("joint covariance has no positive scale")
    covariance += (
        ridge_fraction
        * float(np.median(positive))
        * np.eye(len(covariance))
    )
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
    root = np.linalg.cholesky(block)
    normalized = _whiten_symmetric(root, covariance)
    minimum_generalized = float(np.linalg.eigvalsh(normalized)[0])
    beta = min(conditional_fraction, 0.5 * minimum_generalized)
    if beta <= 0.0:
        raise ValueError("conditional covariance fraction is nonpositive")
    conditional = beta * block
    remainder = covariance - conditional
    remainder = 0.5 * (remainder + remainder.T)
    eigenvalues, eigenvectors = np.linalg.eigh(remainder)
    largest = float(eigenvalues[-1])
    keep = eigenvalues > eigenvalue_relative_tolerance * largest
    if not np.any(keep):
        raise ValueError("shared covariance factor is empty")
    factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
    np.testing.assert_allclose(
        conditional + factor @ factor.T,
        covariance,
        atol=1e-12,
        rtol=1e-9,
    )
    return conditional, factor, beta


class BlockLowRankSolver:
    """Woodbury solver for 3-D block diagonal plus a shared low-rank factor."""

    def __init__(self, conditional: np.ndarray, factor: np.ndarray) -> None:
        dimension = conditional.shape[0]
        if conditional.shape != (dimension, dimension) or dimension % 3:
            raise ValueError("conditional covariance has an invalid shape")
        if factor.shape[0] != dimension:
            raise ValueError("factor dimension mismatch")
        self.dimension = dimension
        self.observation_count = dimension // 3
        self.factor = np.asarray(factor, dtype=np.float64)
        blocks = np.stack(
            [
                conditional[start : start + 3, start : start + 3]
                for start in range(0, dimension, 3)
            ]
        )
        self.inverse_blocks = np.linalg.inv(blocks)
        self.ainv_factor = self._apply_ainv(self.factor)
        core = np.eye(self.factor.shape[1]) + self.factor.T @ self.ainv_factor
        self.core_root = np.linalg.cholesky(0.5 * (core + core.T))

    def _apply_ainv(self, value: np.ndarray) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        matrix = raw.reshape(self.dimension, -1)
        blocked = matrix.reshape(self.observation_count, 3, -1)
        solved = np.einsum("nij,njk->nik", self.inverse_blocks, blocked)
        return solved.reshape(self.dimension, -1)

    def solve(self, value: object) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        matrix = raw.reshape(self.dimension, -1)
        ainv_value = self._apply_ainv(matrix)
        if not self.factor.shape[1]:
            result = ainv_value
        else:
            rhs = self.factor.T @ ainv_value
            correction = np.linalg.solve(
                self.core_root.T,
                np.linalg.solve(self.core_root, rhs),
            )
            result = ainv_value - self.ainv_factor @ correction
        return result.reshape(raw.shape)


def _posterior(
    prior: np.ndarray,
    cross: np.ndarray,
    solver: BlockLowRankSolver,
) -> tuple[np.ndarray, np.ndarray]:
    solved = solver.solve(cross.T)
    gain = solved.T
    covariance = prior - cross @ solved
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    return gain, covariance


def _metrics(
    queries: np.ndarray,
    observations: np.ndarray,
    query_mean: np.ndarray,
    observation_mean: np.ndarray,
    gain: np.ndarray,
    posterior: np.ndarray,
) -> dict[str, float | int | bool | None]:
    innovations = observations.reshape(len(observations), -1) - observation_mean
    errors = queries.reshape(len(queries), -1) - (
        query_mean + innovations @ gain.T
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (posterior + posterior.T))
    result: dict[str, float | int | bool | None] = {
        "sample_count": int(len(errors)),
        "query_dimension": int(errors.shape[1]),
        "rmse_per_coordinate_m": float(np.sqrt(np.mean(errors**2))),
        "vector_rmse_m": float(
            np.sqrt(np.mean(np.sum(errors**2, axis=1)))
        ),
        "minimum_posterior_eigenvalue_m2": float(eigenvalues[0]),
        "posterior_valid": bool(eigenvalues[0] > 0.0),
        "mean_normalized_nees": None,
        "joint_coverage_90": None,
        "mean_nll_nats": None,
    }
    if eigenvalues[0] <= 0.0:
        return result
    root = np.linalg.cholesky(posterior)
    whitened = np.linalg.solve(root, errors.T)
    nees = np.sum(whitened**2, axis=0)
    dimension = errors.shape[1]
    z90 = 1.2815515655446004
    threshold = dimension * (
        1.0
        - 2.0 / (9.0 * dimension)
        + z90 * math.sqrt(2.0 / (9.0 * dimension))
    ) ** 3
    logdet = 2.0 * float(np.log(np.diag(root)).sum())
    nll = 0.5 * (
        nees + logdet + dimension * math.log(2.0 * math.pi)
    )
    result.update(
        {
            "mean_normalized_nees": float(np.mean(nees) / dimension),
            "joint_coverage_90": float(np.mean(nees <= threshold)),
            "mean_nll_nats": float(np.mean(nll)),
        }
    )
    return result


def _spectral_joint_baseline(
    prior: np.ndarray,
    cross: np.ndarray,
    observation: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    joint = np.block([[prior, cross], [cross.T, observation]])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (joint + joint.T))
    floor = max(
        float(eigenvalues[0]),
        np.finfo(np.float64).eps * float(eigenvalues[-1]),
    )
    count = min(max(int(rank), 1), len(eigenvalues))
    selected_values = eigenvalues[-count:]
    selected_vectors = eigenvectors[:, -count:]
    low_rank_values = np.maximum(selected_values - floor, 0.0)
    approximation = floor * np.eye(len(joint)) + (
        selected_vectors * low_rank_values[None, :]
    ) @ selected_vectors.T
    qdim = prior.shape[0]
    q_approx = approximation[:qdim, :qdim]
    c_approx = approximation[:qdim, qdim:]
    s_approx = approximation[qdim:, qdim:]
    gain = np.linalg.solve(s_approx, c_approx.T).T
    posterior = q_approx - gain @ c_approx.T
    posterior = 0.5 * (posterior + posterior.T)
    np.linalg.cholesky(posterior)
    return gain, posterior, {
        "low_rank_dimension": count,
        "isotropic_floor": floor,
        "representation_bytes": int(
            selected_vectors.nbytes
            + low_rank_values.nbytes
            + np.dtype(np.float64).itemsize
        ),
    }


def _median_seconds(call: Callable[[], Any], repeats: int) -> float:
    values: list[float] = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        output = call()
        values.append((time.perf_counter_ns() - start) * 1e-9)
        checksum += (
            float(np.asarray(output).reshape(-1)[0]) if np.size(output) else 0.0
        )
    if not math.isfinite(checksum):
        raise ValueError("benchmark produced a nonfinite result")
    return float(np.median(values))


def evaluate_fold(
    train: list[Samples],
    test: list[Samples],
    protocol: dict[str, Any],
    size: str,
    fold_index: int,
) -> list[dict[str, Any]]:
    train_y = np.concatenate([record.observations_m for record in train])
    train_q_all = np.concatenate(
        [record.future_displacements_m for record in train]
    )
    test_y = np.concatenate([record.observations_m for record in test])
    test_q_all = np.concatenate(
        [record.future_displacements_m for record in test]
    )
    marker_count = train_y.shape[1]
    mean, joint = _joint_covariance(
        train_q_all,
        train_y,
        float(protocol["joint_covariance_shrinkage"]),
        float(protocol["joint_covariance_ridge_fraction"]),
    )
    full_qdim = 3 * marker_count
    observation_mean = mean[full_qdim:]
    observation_covariance = joint[full_qdim:, full_qdim:]
    conditional, shared, beta = decompose_shared_covariance(
        observation_covariance,
        float(protocol["maximum_conditional_block_fraction"]),
        float(protocol["factor_eigenvalue_relative_tolerance"]),
    )
    full_solver = BlockLowRankSolver(conditional, shared)
    full_rank = shared.shape[1]
    benchmark_count = min(
        int(protocol["benchmark_batch_windows"]), len(test_y)
    )
    benchmark_innovations = (
        test_y[:benchmark_count].reshape(benchmark_count, -1)
        - observation_mean
    )
    timing_repeats = int(protocol["timing_repeats"])
    rows: list[dict[str, Any]] = []

    for query_count in [int(value) for value in protocol["query_counts"]]:
        if query_count > marker_count:
            continue
        marker_indices = select_query_indices(marker_count, query_count)
        coordinate_indices = np.concatenate(
            [
                np.arange(3 * index, 3 * index + 3)
                for index in marker_indices
            ]
        )
        prior = joint[np.ix_(coordinate_indices, coordinate_indices)]
        cross = joint[
            np.ix_(
                coordinate_indices,
                full_qdim + np.arange(3 * marker_count),
            )
        ]
        query_mean = mean[coordinate_indices]
        test_queries = test_q_all[:, marker_indices, :]

        cache_start = time.perf_counter_ns()
        full_gain, full_posterior = _posterior(prior, cross, full_solver)
        cache_build_seconds = (time.perf_counter_ns() - cache_start) * 1e-9

        compression_start = time.perf_counter_ns()
        compression = compress_shared_factor_for_posterior(
            shared.reshape(marker_count, 3, full_rank),
            prior_query_covariance=prior,
            query_observation_cross_covariance=cross,
            innovation_operator=full_solver,
            maximum_rank=min(prior.shape[0], full_rank),
            rank_relative_tolerance=float(protocol["rank_relative_tolerance"]),
            parity_relative_tolerance=float(protocol["parity_relative_tolerance"]),
        )
        compression_seconds = (
            time.perf_counter_ns() - compression_start
        ) * 1e-9
        reduced_factor = compression.compressed_factor_m.reshape(
            3 * marker_count, -1
        )
        reduced_solver = BlockLowRankSolver(conditional, reduced_factor)
        reduced_gain, reduced_posterior = _posterior(
            prior, cross, reduced_solver
        )

        full_denominator = max(
            float(np.linalg.norm(full_gain, ord="fro")), 1e-30
        )
        covariance_denominator = max(
            float(np.linalg.norm(full_posterior, ord="fro")), 1e-30
        )
        full_means = query_mean + benchmark_innovations @ full_gain.T
        reduced_means = query_mean + benchmark_innovations @ reduced_gain.T

        spectral_gain, spectral_posterior, spectral_details = (
            _spectral_joint_baseline(
                prior,
                cross,
                observation_covariance,
                rank=compression.retained_rank,
            )
        )
        methods = {
            "full": _metrics(
                test_queries,
                test_y,
                query_mean,
                observation_mean,
                full_gain,
                full_posterior,
            ),
            "posterior_preserving": _metrics(
                test_queries,
                test_y,
                query_mean,
                observation_mean,
                reduced_gain,
                reduced_posterior,
            ),
            "joint_psd_spectral": _metrics(
                test_queries,
                test_y,
                query_mean,
                observation_mean,
                spectral_gain,
                spectral_posterior,
            ),
        }

        def full_update() -> np.ndarray:
            return full_solver.solve(benchmark_innovations.T).T @ cross.T

        def reduced_update() -> np.ndarray:
            return reduced_solver.solve(benchmark_innovations.T).T @ cross.T

        def cached_update() -> np.ndarray:
            return benchmark_innovations @ full_gain.T

        full_seconds = _median_seconds(full_update, timing_repeats)
        reduced_seconds = _median_seconds(reduced_update, timing_repeats)
        cached_seconds = _median_seconds(cached_update, timing_repeats)
        query_dimension = 3 * query_count
        factor_bytes = int(reduced_factor.nbytes)
        projector_bytes = int(compression.latent_projection.nbytes)
        joint_cache_bytes = int(full_gain.nbytes + full_posterior.nbytes)
        independent_cache_bytes = int(
            full_gain.nbytes
            + query_count * 3 * 3 * np.dtype(np.float64).itemsize
        )
        rows.append(
            {
                "size": size,
                "fold": fold_index,
                "query_marker_count": query_count,
                "query_dimension": query_dimension,
                "query_marker_indices": marker_indices.tolist(),
                "train_recording_count": len(train),
                "test_recording_count": len(test),
                "train_window_count": int(len(train_y)),
                "test_window_count": int(len(test_y)),
                "median_horizon_seconds": float(
                    np.median(
                        np.concatenate(
                            [record.horizon_seconds for record in test]
                        )
                    )
                ),
                "observation_dimension": int(
                    observation_covariance.shape[0]
                ),
                "original_shared_rank": full_rank,
                "conditional_block_fraction": beta,
                "compression": compression.summary(),
                "parity": {
                    "relative_gain_error": float(
                        np.linalg.norm(
                            reduced_gain - full_gain, ord="fro"
                        )
                        / full_denominator
                    ),
                    "relative_posterior_covariance_error": float(
                        np.linalg.norm(
                            reduced_posterior - full_posterior,
                            ord="fro",
                        )
                        / covariance_denominator
                    ),
                    "maximum_realized_mean_difference_m": float(
                        np.max(
                            np.linalg.norm(
                                reduced_means - full_means, axis=1
                            )
                        )
                    ),
                },
                "payload_bytes": {
                    "full_shared_factor": int(shared.nbytes),
                    "compressed_factor": factor_bytes,
                    "compressed_factor_plus_projection": (
                        factor_bytes + projector_bytes
                    ),
                    "joint_cached_gain_and_posterior": joint_cache_bytes,
                    "independent_query_caches_without_cross_query_covariance": independent_cache_bytes,
                    "joint_psd_spectral": spectral_details[
                        "representation_bytes"
                    ],
                },
                "resident_model_payload_comparison": {
                    "factor_over_joint_cache": factor_bytes
                    / joint_cache_bytes,
                    "factor_plus_projection_over_joint_cache": (
                        factor_bytes + projector_bytes
                    )
                    / joint_cache_bytes,
                    "factor_is_smaller_than_joint_cache": (
                        factor_bytes < joint_cache_bytes
                    ),
                    "factor_plus_projection_is_smaller_than_joint_cache": (
                        factor_bytes + projector_bytes
                    )
                    < joint_cache_bytes,
                    "assumption": "Q, C, conditional covariance, and means are already resident at the consumer",
                },
                "timing": {
                    "cache_construction_ms": 1000.0
                    * cache_build_seconds,
                    "compression_construction_ms": 1000.0
                    * compression_seconds,
                    "benchmark_windows": benchmark_count,
                    "full_factor_update_us_per_window": 1e6
                    * full_seconds
                    / benchmark_count,
                    "compressed_factor_update_us_per_window": 1e6
                    * reduced_seconds
                    / benchmark_count,
                    "cached_message_update_us_per_window": 1e6
                    * cached_seconds
                    / benchmark_count,
                },
                "joint_psd_spectral": spectral_details,
                "methods": methods,
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parity_limit = max(
        row["parity"]["relative_gain_error"] for row in rows
    )
    covariance_limit = max(
        row["parity"]["relative_posterior_covariance_error"]
        for row in rows
    )
    mean_limit = max(
        row["parity"]["maximum_realized_mean_difference_m"]
        for row in rows
    )
    by_portfolio: list[dict[str, Any]] = []
    keys = sorted(
        {(row["size"], row["query_marker_count"]) for row in rows}
    )
    for size, query_count in keys:
        selected = [
            row
            for row in rows
            if row["size"] == size
            and row["query_marker_count"] == query_count
        ]
        weights = np.asarray(
            [row["test_window_count"] for row in selected], dtype=float
        )
        total = float(weights.sum())
        full_rmse = np.asarray(
            [
                row["methods"]["full"]["rmse_per_coordinate_m"]
                for row in selected
            ]
        )
        spectral_rmse = np.asarray(
            [
                row["methods"]["joint_psd_spectral"][
                    "rmse_per_coordinate_m"
                ]
                for row in selected
            ]
        )
        by_portfolio.append(
            {
                "size": size,
                "query_marker_count": query_count,
                "query_dimension": selected[0]["query_dimension"],
                "original_shared_rank": selected[0][
                    "original_shared_rank"
                ],
                "retained_rank_min": min(
                    row["compression"]["retained_rank"]
                    for row in selected
                ),
                "retained_rank_max": max(
                    row["compression"]["retained_rank"]
                    for row in selected
                ),
                "exact_fallback_folds": sum(
                    bool(row["compression"]["exact_fallback"])
                    for row in selected
                ),
                "test_windows": int(total),
                "full_rmse_per_coordinate_m": float(
                    math.sqrt(np.sum(weights * full_rmse**2) / total)
                ),
                "spectral_rmse_per_coordinate_m": float(
                    math.sqrt(
                        np.sum(weights * spectral_rmse**2) / total
                    )
                ),
                "mean_factor_over_joint_cache": float(
                    np.average(
                        [
                            row[
                                "resident_model_payload_comparison"
                            ]["factor_over_joint_cache"]
                            for row in selected
                        ],
                        weights=weights,
                    )
                ),
                "factor_smaller_fold_count": sum(
                    row["resident_model_payload_comparison"][
                        "factor_is_smaller_than_joint_cache"
                    ]
                    for row in selected
                ),
                "mean_compression_construction_ms": float(
                    np.average(
                        [
                            row["timing"][
                                "compression_construction_ms"
                            ]
                            for row in selected
                        ],
                        weights=weights,
                    )
                ),
                "mean_cache_construction_ms": float(
                    np.average(
                        [
                            row["timing"]["cache_construction_ms"]
                            for row in selected
                        ],
                        weights=weights,
                    )
                ),
                "mean_full_update_us_per_window": float(
                    np.average(
                        [
                            row["timing"][
                                "full_factor_update_us_per_window"
                            ]
                            for row in selected
                        ],
                        weights=weights,
                    )
                ),
                "mean_compressed_update_us_per_window": float(
                    np.average(
                        [
                            row["timing"][
                                "compressed_factor_update_us_per_window"
                            ]
                            for row in selected
                        ],
                        weights=weights,
                    )
                ),
                "mean_cached_update_us_per_window": float(
                    np.average(
                        [
                            row["timing"][
                                "cached_message_update_us_per_window"
                            ]
                            for row in selected
                        ],
                        weights=weights,
                    )
                ),
            }
        )
    return {
        "fold_portfolio_count": len(rows),
        "maximum_relative_gain_error": float(parity_limit),
        "maximum_relative_posterior_covariance_error": float(
            covariance_limit
        ),
        "maximum_realized_mean_difference_m": float(mean_limit),
        "all_exact_posteriors_valid": all(
            row["methods"]["posterior_preserving"]["posterior_valid"]
            for row in rows
        ),
        "any_factor_payload_advantage": any(
            row["resident_model_payload_comparison"][
                "factor_is_smaller_than_joint_cache"
            ]
            for row in rows
        ),
        "all_spectral_posteriors_valid": all(
            row["methods"]["joint_psd_spectral"]["posterior_valid"]
            for row in rows
        ),
        "portfolio_rows": by_portfolio,
    }


def _summary(result: dict[str, Any]) -> str:
    if result["status"] != "evaluated-real-query-portfolios":
        return (
            "# Tracking Cloth query-portfolio study\n\n"
            f"Status: **{result['status']}**\n\n"
            f"{result.get('reason', '')}\n"
        )
    aggregate = result["aggregate"]
    lines = [
        "# Tracking Cloth query-portfolio study",
        "",
        "Status: **evaluated real query portfolios**",
        "",
        f"- Accepted recordings: {result['inventory']['accepted_recording_count']} / {result['inventory']['csv_file_count']}",
        f"- Fold/portfolio evaluations: {aggregate['fold_portfolio_count']}",
        f"- Maximum relative gain error: {aggregate['maximum_relative_gain_error']:.3e}",
        f"- Maximum posterior-covariance error: {aggregate['maximum_relative_posterior_covariance_error']:.3e}",
        f"- Maximum realized mean difference: {1e3 * aggregate['maximum_realized_mean_difference_m']:.3e} mm",
        f"- At least one resident-model factor/cache payload advantage: {aggregate['any_factor_payload_advantage']}",
        "",
        "| Size | Query markers | Dimension | Rank | Factor/cache | Full RMSE/coord (mm) | PSD spectral RMSE/coord (mm) | Compressed/cache update (us) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["portfolio_rows"]:
        rank = f"{row['retained_rank_min']}–{row['retained_rank_max']}"
        lines.append(
            "| {size} | {count} | {dimension} | {rank} | {ratio:.3f} | {full:.3f} | {spectral:.3f} | {compressed:.3f}/{cached:.3f} |".format(
                size=row["size"],
                count=row["query_marker_count"],
                dimension=row["query_dimension"],
                rank=rank,
                ratio=row["mean_factor_over_joint_cache"],
                full=1e3 * row["full_rmse_per_coordinate_m"],
                spectral=1e3
                * row["spectral_rmse_per_coordinate_m"],
                compressed=row[
                    "mean_compressed_update_us_per_window"
                ],
                cached=row["mean_cached_update_us_per_window"],
            )
        )
    lines.extend(
        [
            "",
            "Payload ratios assume Q, C, conditional covariance, and means are already resident. A direct cached query message is the preferred implementation for a single immutable query when that assumption is false or when minimal online latency dominates.",
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
    protocol = json.loads(protocol_bytes)
    if protocol.get("schema") != SCHEMA:
        raise ValueError("unsupported protocol schema")
    csv_paths = sorted(
        path for path in dataset_root.rglob("*.csv") if path.is_file()
    )
    inventory: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "csv_file_count": len(csv_paths),
        "accepted_recording_count": 0,
        "accepted_by_size": {},
        "excluded": [],
        "files": [],
    }
    samples: list[Samples] = []
    for path in csv_paths:
        relative = path.relative_to(dataset_root).as_posix()
        file_record = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        inventory["files"].append(file_record)
        try:
            record = parse_recording(path, dataset_root)
            sample = make_samples(record, protocol)
        except (OSError, UnicodeError, ValueError) as exc:
            inventory["excluded"].append(
                {"relative_path": relative, "reason": str(exc)}
            )
            continue
        samples.append(sample)
        inventory["accepted_by_size"][record.size] = int(
            inventory["accepted_by_size"].get(record.size, 0)
        ) + 1
    inventory["accepted_recording_count"] = len(samples)
    inventory["dataset_manifest_sha256"] = _sha256_bytes(
        _canonical_bytes(inventory["files"])
    )
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source_revision": source_revision,
        "status": "not-evaluated",
        "inventory": {
            key: value
            for key, value in inventory.items()
            if key not in {"files", "excluded"}
        },
        "claim_boundary": {
            "real_motion_capture_geometry": True,
            "recording_disjoint": True,
            "fixed_query_portfolios": True,
            "learned_4d_provider": False,
            "bayesian_phystwin_benefit": False,
            "causal4d_benefit": False,
            "deployment_calibration": False,
        },
    }
    evaluations: list[dict[str, Any]] = []
    try:
        folds = int(protocol["fold_count"])
        minimum = int(protocol["minimum_recordings_per_size"])
        for size in ("A2", "A3"):
            group = [record for record in samples if record.size == size]
            if len(group) < minimum:
                raise ValueError(
                    f"{size} has {len(group)} accepted recordings, needs {minimum}"
                )
            assignments = _fold_assignments(group, folds)
            for fold in range(folds):
                train = [
                    record
                    for record in group
                    if assignments[record.relative_path] != fold
                ]
                test = [
                    record
                    for record in group
                    if assignments[record.relative_path] == fold
                ]
                evaluations.extend(
                    evaluate_fold(train, test, protocol, size, fold)
                )
        aggregate = _aggregate(evaluations)
        limit = float(protocol["required_maximum_relative_parity_error"])
        if aggregate["maximum_relative_gain_error"] > limit:
            raise ValueError("gain parity exceeded the registered limit")
        if (
            aggregate["maximum_relative_posterior_covariance_error"]
            > limit
        ):
            raise ValueError(
                "posterior covariance parity exceeded the registered limit"
            )
        result.update(
            {
                "status": "evaluated-real-query-portfolios",
                "evaluations": evaluations,
                "aggregate": aggregate,
            }
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        result.update(
            {
                "status": "technical-or-support-negative",
                "reason": str(exc),
                "evaluations": evaluations,
            }
        )
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    manifest = {
        "schema": "prob4d.tracking-cloth-query-portfolio-manifest.v1",
        "source_revision": source_revision,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "result_sha256": _sha256_bytes(result_bytes),
        "inventory_sha256": _sha256_file(output_dir / "inventory.json"),
        "dataset_manifest_sha256": inventory[
            "dataset_manifest_sha256"
        ],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "raw_data_copied_to_output": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _summary(result), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
            {"status": result["status"], "output_dir": str(args.output_dir)}
        )
    )
    return 0 if result["status"] == "evaluated-real-query-portfolios" else 3


if __name__ == "__main__":
    raise SystemExit(main())
