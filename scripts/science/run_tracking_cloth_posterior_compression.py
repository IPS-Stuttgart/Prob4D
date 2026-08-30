#!/usr/bin/env python3
"""Evaluate posterior-preserving shared-noise compression on real cloth trajectories.

The script reads only CSV trajectory files from the verified Tracking Cloth
Deformation dataset. It performs recording-disjoint cross-validation, writes
compact derived evidence, and never copies raw trajectories into its outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)

SCHEMA = "prob4d.tracking-cloth-posterior-compression-real.v1"
CHI_SQUARE_3_90 = 6.251388631170325
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
        return np.linalg.solve(
            self.covariance, raw.reshape(self.dimension, -1)
        ).reshape(raw.shape)


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
    candidate_lengths = []
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
    finite_rows = positions[np.all(np.isfinite(positions), axis=(1, 2))]
    if not len(finite_rows):
        raise ValueError("recording has no complete frame for unit detection")
    sample = finite_rows[
        np.linspace(0, len(finite_rows) - 1, min(64, len(finite_rows)), dtype=int)
    ]
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
    with path.open(
        "r", encoding="utf-8-sig", errors="strict", newline=""
    ) as handle:
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
        coordinates = [
            _finite_float(value) for value in row[2 : 2 + coordinate_count]
        ]
        if len(coordinates) < coordinate_count:
            coordinates.extend(
                [float("nan")] * (coordinate_count - len(coordinates))
            )
        frames.append(int(round(frame_value)))
        timestamps.append(time_value)
        positions.append(
            np.asarray(coordinates, dtype=np.float64).reshape(marker_count, 3)
        )
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
        observation = (
            positions[current] - positions[current - lag]
        ) * (dt_horizon / dt_lag)
        query = (
            positions[current + horizon].mean(axis=0)
            - positions[current].mean(axis=0)
        )
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


def _fold_assignments(
    records: Iterable[RecordingSamples], folds: int
) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(record.relative_path.encode("utf-8")).hexdigest(),
            record.relative_path,
        ),
    )
    return {
        record.relative_path: index % folds
        for index, record in enumerate(ordered)
    }


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
    errors = _prediction_errors(
        queries, observations, query_mean, observation_mean, gain
    )
    eigenvalues = np.linalg.eigvalsh(
        0.5 * (posterior_covariance + posterior_covariance.T)
    )
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
    rank = shared.shape[1]
    compression = compress_shared_factor_for_posterior(
        shared.reshape(count, 3, rank),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=DenseInnovation(observation_covariance),
        maximum_rank=qdim,
        rank_relative_tolerance=float(protocol["rank_relative_tolerance"]),
        parity_relative_tolerance=float(protocol["parity_relative_tolerance"]),
    )
    compressed = compression.compressed_factor_m.reshape(
        observation_covariance.shape[0], -1
    )
    full_gain, full_covariance = _posterior(prior, cross, observation_covariance)
    reduced_covariance = conditional + compressed @ compressed.T
    compressed_gain, compressed_posterior = _posterior(
        prior, cross, reduced_covariance
    )

    pca_vectors = np.linalg.svd(shared, full_matrices=False)[2].T[
        :, : compression.retained_rank
    ]
    pca_factor = shared @ pca_vectors
    pca_gain, pca_posterior = _method_from_observation_covariance(
        prior, cross, conditional + pca_factor @ pca_factor.T
    )
    conditional_gain, conditional_posterior = _method_from_observation_covariance(
        prior, cross, conditional
    )
    prior_gain = np.zeros_like(full_gain)
    methods = {
        "full": (full_gain, full_covariance),
        "posterior_preserving": (compressed_gain, compressed_posterior),
        "equal_rank_covariance_pca": (pca_gain, pca_posterior),
        "conditional_only": (conditional_gain, conditional_posterior),
        "prior_only": (prior_gain, prior),
        "cached_full_query_message": (full_gain, full_covariance),
    }
    method_metrics = {
        name: _metrics(
            test_q,
            test_y,
            query_mean,
            observation_mean,
            gain,
            covariance,
        )
        for name, (gain, covariance) in methods.items()
    }
    innovation = test_y.reshape(len(test_y), -1) - observation_mean
    full_means = query_mean + innovation @ full_gain.T
    compressed_means = query_mean + innovation @ compressed_gain.T
    gain_denominator = max(float(np.linalg.norm(full_gain, ord="fro")), 1e-30)
    covariance_denominator = max(
        float(np.linalg.norm(full_covariance, ord="fro")), 1e-30
    )
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
        "original_shared_rank": rank,
        "conditional_block_fraction": beta,
        "compression": compression.summary(),
        "full_vs_compressed": {
            "relative_gain_error": float(
                np.linalg.norm(compressed_gain - full_gain, ord="fro")
                / gain_denominator
            ),
            "relative_posterior_covariance_error": float(
                np.linalg.norm(compressed_posterior - full_covariance, ord="fro")
                / covariance_denominator
            ),
            "maximum_realized_mean_difference_m": float(
                np.max(np.linalg.norm(compressed_means - full_means, axis=1))
            ),
        },
        "payload_bytes": {
            "full_shared_factor": int(shared.nbytes),
            "posterior_preserving_factor": int(compressed.nbytes),
            "equal_rank_pca_factor": int(pca_factor.nbytes),
            "cached_full_query_message": int(
                full_gain.nbytes + full_covariance.nbytes
            ),
        },
        "methods": method_metrics,
    }


def _aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    parity = [fold["full_vs_compressed"] for fold in folds]
    total_test = sum(fold["test_window_count"] for fold in folds)
    methods = sorted(folds[0]["methods"])
    aggregate_methods: dict[str, dict[str, Any]] = {}
    for method in methods:
        records = [fold["methods"][method] for fold in folds]
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
                min(
                    record["minimum_posterior_eigenvalue_m2"]
                    for record in records
                )
            ),
        }
        rmse = np.asarray([record["query_rmse_m"] for record in records])
        values["query_rmse_m"] = float(
            math.sqrt(np.sum(weights * rmse**2) / np.sum(weights))
        )
        values["maximum_query_error_m"] = float(
            max(record["maximum_query_error_m"] for record in records)
        )
        for metric in (
            "mean_query_nll_nats",
            "mean_normalized_nees",
            "coverage_90",
        ):
            valid_indices = [
                index
                for index, record in enumerate(records)
                if record[metric] is not None
            ]
            if not valid_indices:
                values[metric] = None
                continue
            valid_weights = weights[valid_indices]
            items = np.asarray(
                [records[index][metric] for index in valid_indices],
                dtype=np.float64,
            )
            values[metric] = float(
                np.sum(valid_weights * items) / np.sum(valid_weights)
            )
        aggregate_methods[method] = values
    full_bytes = sum(fold["payload_bytes"]["full_shared_factor"] for fold in folds)
    reduced_bytes = sum(
        fold["payload_bytes"]["posterior_preserving_factor"] for fold in folds
    )
    return {
        "fold_count": len(folds),
        "test_window_count": total_test,
        "retained_ranks": sorted(
            {int(fold["compression"]["retained_rank"]) for fold in folds}
        ),
        "original_shared_rank_min": min(
            int(fold["original_shared_rank"]) for fold in folds
        ),
        "original_shared_rank_max": max(
            int(fold["original_shared_rank"]) for fold in folds
        ),
        "maximum_relative_gain_error": max(
            float(value["relative_gain_error"]) for value in parity
        ),
        "maximum_relative_posterior_covariance_error": max(
            float(value["relative_posterior_covariance_error"])
            for value in parity
        ),
        "maximum_realized_mean_difference_m": max(
            float(value["maximum_realized_mean_difference_m"])
            for value in parity
        ),
        "summed_full_shared_factor_bytes": full_bytes,
        "summed_posterior_preserving_factor_bytes": reduced_bytes,
        "shared_factor_payload_reduction_ratio": (
            float(full_bytes / reduced_bytes) if reduced_bytes else None
        ),
        "methods": aggregate_methods,
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
        "parity_relative_tolerance",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol is missing fields: {missing}")
    return protocol


def _summary_markdown(result: dict[str, Any]) -> str:
    if result["status"] != "evaluated-real-trajectories":
        return (
            "# Tracking Cloth posterior-compression evaluation\n\n"
            f"Status: **{result['status']}**\n\n"
            f"Reason: {result.get('reason', 'unspecified')}\n"
        )
    aggregate = result["aggregate"]
    full = aggregate["methods"]["full"]
    compressed = aggregate["methods"]["posterior_preserving"]
    pca = aggregate["methods"]["equal_rank_covariance_pca"]
    accepted = result["inventory"]["accepted_recording_count"]
    csv_count = result["inventory"]["csv_file_count"]
    rank_min = aggregate["original_shared_rank_min"]
    rank_max = aggregate["original_shared_rank_max"]
    gain_error = aggregate["maximum_relative_gain_error"]
    covariance_error = aggregate["maximum_relative_posterior_covariance_error"]
    mean_difference_mm = 1000.0 * aggregate["maximum_realized_mean_difference_m"]
    reduction = aggregate["shared_factor_payload_reduction_ratio"]
    full_rmse_mm = 1000.0 * full["query_rmse_m"]
    compressed_rmse_mm = 1000.0 * compressed["query_rmse_m"]
    pca_rmse_mm = 1000.0 * pca["query_rmse_m"]
    full_coverage = 100.0 * full["coverage_90"]
    compressed_coverage = 100.0 * compressed["coverage_90"]
    return f"""# Tracking Cloth posterior-compression evaluation

Status: **evaluated real trajectories**

- Parsed recordings: {accepted} / {csv_count}
- Recording-disjoint folds: {aggregate['fold_count']}
- Held-out windows: {aggregate['test_window_count']}
- Original shared rank: {rank_min}–{rank_max}
- Retained ranks: {aggregate['retained_ranks']}
- Maximum relative gain error: {gain_error:.3e}
- Maximum relative posterior-covariance error: {covariance_error:.3e}
- Maximum realized posterior-mean difference: {mean_difference_mm:.3e} mm
- Summed shared-factor payload reduction: {reduction:.2f}x
- Full / compressed RMSE: {full_rmse_mm:.3f} / {compressed_rmse_mm:.3f} mm
- Full / compressed 90% coverage: {full_coverage:.2f}% / {compressed_coverage:.2f}%
- Equal-rank PCA RMSE: {pca_rmse_mm:.3f} mm

The experiment is a recording-disjoint local-Gaussian mechanism study. It does
not establish a learned 4-D provider, calibrated deployment uncertainty, or
BayesianPhysTwin/Causal4D physical benefit.
"""


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
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".csv"
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
            inventory["excluded"].append(
                {"relative_path": relative, "reason": str(exc)}
            )
            continue
        recordings.append(recording)
        samples.append(recording_samples)
        inventory["accepted_by_size"][recording.size] = (
            int(inventory["accepted_by_size"].get(recording.size, 0)) + 1
        )
    inventory["accepted_recording_count"] = len(recordings)
    inventory["dataset_manifest_sha256"] = _sha256_bytes(
        _canonical_bytes(inventory["files"])
    )
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
            key: value
            for key, value in inventory.items()
            if key not in {"files", "excluded"}
        },
        "claim_boundary": {
            "real_motion_capture_trajectories": True,
            "recording_disjoint_evaluation": True,
            "learned_4d_provider_evaluated": False,
            "real_covariance_calibration_claimed": False,
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
                raise ValueError(
                    f"{size} has {len(group)} accepted recordings, needs {minimum}"
                )
            assignments = _fold_assignments(group, folds)
            for fold_index in range(folds):
                train = [
                    record
                    for record in group
                    if assignments[record.relative_path] != fold_index
                ]
                test = [
                    record
                    for record in group
                    if assignments[record.relative_path] == fold_index
                ]
                evaluations.append(
                    evaluate_fold(train, test, protocol, fold_index, size)
                )
        aggregate = _aggregate(evaluations)
        if aggregate["maximum_relative_gain_error"] > float(
            protocol["required_maximum_relative_parity_error"]
        ):
            raise ValueError("posterior gain parity exceeded the protocol limit")
        if aggregate["maximum_relative_posterior_covariance_error"] > float(
            protocol["required_maximum_relative_parity_error"]
        ):
            raise ValueError("posterior covariance parity exceeded the protocol limit")
        result.update(
            {
                "status": "evaluated-real-trajectories",
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

    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    manifest = {
        "schema": "prob4d.tracking-cloth-posterior-compression-manifest.v1",
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
    (output_dir / "summary.md").write_text(
        _summary_markdown(result), encoding="utf-8"
    )
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
    return 0 if result["status"] == "evaluated-real-trajectories" else 3


if __name__ == "__main__":
    raise SystemExit(main())
