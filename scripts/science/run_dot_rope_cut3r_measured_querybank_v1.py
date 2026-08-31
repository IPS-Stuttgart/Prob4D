#!/usr/bin/env python3
"""Measure query-sufficient compression on measured DOT/CUT3R Sim(3) factors.

The experiment reuses the immutable marker-free CUT3R provider bundle from the
registered DOT source study.  It opens only the already-authorized R01--R03
marker payloads, reconstructs the clustered-bootstrap relative-Sim(3)
covariance, propagates it through the real provider geometry, applies the frozen
source dependence tempering strength, and compares:

* the full measured shared factor;
* the exact posterior-preserving query factor;
* a direct cached joint-query Gaussian message; and
* a positive-definite complete-joint spectral approximation under the same
  incremental byte budget.

No provider inference is rerun and no R04--R70 payload is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import platform
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from prob4d.dot_rope_cut3r_study import (
    clustered_bootstrap_sim3,
    content_id,
    finite_difference_derivatives,
    make_off_axis_probes,
    robust_fit_sim3,
    sim3_to_vector,
    transform_probe_vector,
)
from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)

SCHEMA = "prob4d.dot-rope-cut3r-measured-querybank.v1"
RESULT_SCHEMA = "prob4d.dot-rope-cut3r-measured-querybank-result.v1"
POOLED_EVALUATOR_PATH = Path("scripts/science/evaluate_dot_rope_cut3r_pooled.py")
POOLED_EVALUATOR_GIT_BLOB_SHA1 = "6195e70997f0e9582251c08772b1e423a3062ad6"
SOURCE_PROTOCOL_PATH = Path("protocols/dot-rope-cut3r-native-provider-v1.json")
SOURCE_PROTOCOL_GIT_BLOB_SHA1 = "eaf84956189015c35e53a521cf1b152ca813e680"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate_hex(value: object, *, label: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must contain {length} hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("schema") != SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unsupported measured-querybank protocol")
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id", None)
    _validate_hex(protocol_id, label="protocol_id", length=64)
    if content_id(unsigned) != protocol_id:
        raise ValueError("measured-querybank protocol identity mismatch")
    expected = {
        "archive": "R01-10.zip",
        "source_sequences": ["R01", "R02", "R03"],
        "reserved_sequences": "R04-R70",
        "provider_run_id": 33329701704,
        "provider_artifact_name": "dot-rope-cut3r-sealed-provider-33329701704-1",
        "provider_bundle_id": "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7",
        "provider_request_id": "83cc26be92364fc7715d692b3bb966cf914fb9f911e0763823f2789480a00cf2",
        "provider_revision": "7eb4867e36742d819c514fad21436d4f475b4bed",
        "source_protocol_git_blob_sha1": SOURCE_PROTOCOL_GIT_BLOB_SHA1,
        "pooled_evaluator_git_blob_sha1": POOLED_EVALUATOR_GIT_BLOB_SHA1,
        "query_counts": [1, 2, 4, 8],
        "probe_count": 8,
        "coordinate_columns": [0, 1],
        "coordinate_mode": "pixel-zero-based",
        "dependence_strength": 0.85,
        "dependence_calibration_id": (
            "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
        ),
        "observation_noise_fraction": 0.02,
    }
    for name, expected_value in expected.items():
        if protocol.get(name) != expected_value:
            raise ValueError(f"registered {name} changed")
    return protocol


def _load_module(path: Path, *, name: str, expected_blob: str) -> ModuleType:
    source = path.read_bytes()
    if _git_blob_sha1(source) != expected_blob:
        raise ValueError(f"{path} source bytes changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_registered_evaluator(protocol: Mapping[str, Any]) -> tuple[ModuleType, ModuleType]:
    pooled = _load_module(
        POOLED_EVALUATOR_PATH,
        name="dot_cut3r_pooled_querybank_adapter",
        expected_blob=str(protocol["pooled_evaluator_git_blob_sha1"]),
    )
    base = pooled._load_base_module()
    source_protocol_bytes = SOURCE_PROTOCOL_PATH.read_bytes()
    if _git_blob_sha1(source_protocol_bytes) != protocol["source_protocol_git_blob_sha1"]:
        raise ValueError("registered source protocol bytes changed")
    pooled._ACTIVE_COORDINATE_COLUMNS = tuple(protocol["coordinate_columns"])
    pooled._ACTIVE_COORDINATE_MODE = str(protocol["coordinate_mode"])
    pooled._MARKER_DIAGNOSTICS.clear()
    pooled._COLLECTION_DIAGNOSTICS.clear()
    base._ORIGINAL_LOAD_RUN = base._load_run

    def load_run(bundle: Path, record: Mapping[str, Any]) -> dict[str, Any]:
        return pooled._load_run_with_metadata(base, bundle, record)

    base._load_run = load_run
    base.parse_coordinate_text = pooled._parse_coordinate_text
    base._sample_markers = pooled._sample_markers
    base._collect_pair = pooled._collect_pair
    base._collect_provider_truth = pooled._collect_provider_truth
    return pooled, base


class DiagonalLowRankSolver:
    """Woodbury solver for ``diag(a) + U U.T``."""

    def __init__(self, diagonal: np.ndarray, factor: np.ndarray) -> None:
        diagonal_array = np.asarray(diagonal, dtype=np.float64)
        factor_array = np.asarray(factor, dtype=np.float64)
        if diagonal_array.ndim != 1 or np.any(diagonal_array <= 0.0):
            raise ValueError("diagonal covariance must be a positive vector")
        if factor_array.ndim != 2 or factor_array.shape[0] != diagonal_array.size:
            raise ValueError("factor shape does not match the diagonal covariance")
        if diagonal_array.size % 3:
            raise ValueError("observation dimension must consist of 3-D rows")
        if not np.isfinite(diagonal_array).all() or not np.isfinite(factor_array).all():
            raise ValueError("innovation model must be finite")
        self.diagonal = diagonal_array
        self.inverse_diagonal = 1.0 / diagonal_array
        self.factor = factor_array
        self.dimension = int(diagonal_array.size)
        self.observation_count = self.dimension // 3
        weighted_factor = self.inverse_diagonal[:, None] * factor_array
        core = np.eye(factor_array.shape[1]) + factor_array.T @ weighted_factor
        self.core_root = np.linalg.cholesky(0.5 * (core + core.T))
        self.weighted_factor = weighted_factor

    def solve(self, value: object) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        matrix = raw.reshape(self.dimension, -1)
        direct = self.inverse_diagonal[:, None] * matrix
        if self.factor.shape[1]:
            rhs = self.factor.T @ direct
            correction = np.linalg.solve(
                self.core_root.T,
                np.linalg.solve(self.core_root, rhs),
            )
            direct = direct - self.weighted_factor @ correction
        return direct.reshape(raw.shape)


def select_query_indices(total: int, count: int) -> np.ndarray:
    if not 1 <= count <= total:
        raise ValueError("query count must lie in [1, total]")
    if count == total:
        return np.arange(total, dtype=np.int64)
    result = np.rint(np.linspace(0, total - 1, count)).astype(np.int64)
    if np.unique(result).size != count:
        raise RuntimeError("deterministic query selection produced duplicate indices")
    return result


def coordinate_indices(point_indices: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.arange(3 * int(index), 3 * int(index) + 3) for index in point_indices]
    )


def measured_factor(
    covariance: np.ndarray,
    jacobian: np.ndarray,
    *,
    relative_tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sigma = np.asarray(covariance, dtype=np.float64)
    values, vectors = np.linalg.eigh(0.5 * (sigma + sigma.T))
    values = np.maximum(values, 0.0)
    largest = float(values[-1]) if values.size else 0.0
    if largest <= 0.0:
        raise ValueError("measured Sim(3) covariance is degenerate")
    keep = values > relative_tolerance * largest
    root = vectors[:, keep] * np.sqrt(values[keep])[None, :]
    factor = np.asarray(jacobian, dtype=np.float64) @ root
    return factor, {
        "parameter_rank": int(np.count_nonzero(keep)),
        "parameter_eigenvalues": [float(value) for value in values],
        "factor_singular_values": [
            float(value) for value in np.linalg.svd(factor, compute_uv=False)
        ],
    }


def tempered_joint_model(
    raw_factor: np.ndarray,
    query_rows: np.ndarray,
    *,
    dependence_strength: float,
    noise_standard_deviation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_factor, dtype=np.float64)
    rows = np.asarray(query_rows, dtype=np.int64)
    alpha = float(dependence_strength)
    if raw.ndim != 2 or raw.shape[0] % 3:
        raise ValueError("raw factor must have shape (3N, R)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("dependence strength must lie in [0, 1]")
    if noise_standard_deviation <= 0.0:
        raise ValueError("noise standard deviation must be positive")
    query_raw = raw[rows]
    factor = math.sqrt(alpha) * raw
    observation_diagonal = (
        noise_standard_deviation**2
        + (1.0 - alpha) * np.sum(raw * raw, axis=1)
    )
    query_diagonal = (
        noise_standard_deviation**2
        + (1.0 - alpha) * np.sum(query_raw * query_raw, axis=1)
    )
    prior = np.diag(query_diagonal) + alpha * query_raw @ query_raw.T
    cross = alpha * query_raw @ raw.T
    return prior, cross, observation_diagonal, factor


def posterior(
    prior: np.ndarray,
    cross: np.ndarray,
    solver: DiagonalLowRankSolver,
) -> tuple[np.ndarray, np.ndarray]:
    solved = solver.solve(cross.T)
    gain = solved.T
    covariance = prior - cross @ solved
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    return gain, covariance


def spectral_joint_baseline(
    prior: np.ndarray,
    cross: np.ndarray,
    observation_covariance: np.ndarray,
    *,
    byte_budget: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    joint = np.block([[prior, cross], [cross.T, observation_covariance]])
    joint = 0.5 * (joint + joint.T)
    values, vectors = np.linalg.eigh(joint)
    maximum = max(float(values[-1]), np.finfo(np.float64).tiny)
    floor = max(float(values[0]), np.finfo(np.float64).eps * maximum)
    dimension = joint.shape[0]
    available_scalars = max(byte_budget // np.dtype(np.float64).itemsize - 1, 0)
    rank = min(dimension, available_scalars // (dimension + 1))
    if rank:
        selected_values = values[-rank:]
        selected_vectors = vectors[:, -rank:]
        increments = np.maximum(selected_values - floor, 0.0)
        approximation = floor * np.eye(dimension) + (
            selected_vectors * increments[None, :]
        ) @ selected_vectors.T
    else:
        increments = np.empty(0, dtype=np.float64)
        selected_vectors = np.empty((dimension, 0), dtype=np.float64)
        approximation = floor * np.eye(dimension)
    query_dimension = prior.shape[0]
    q_approx = approximation[:query_dimension, :query_dimension]
    c_approx = approximation[:query_dimension, query_dimension:]
    s_approx = approximation[query_dimension:, query_dimension:]
    root = np.linalg.cholesky(0.5 * (s_approx + s_approx.T))
    solved = np.linalg.solve(root.T, np.linalg.solve(root, c_approx.T))
    gain = solved.T
    post = q_approx - c_approx @ solved
    post = 0.5 * (post + post.T)
    np.linalg.cholesky(post)
    representation_bytes = int(
        selected_vectors.nbytes
        + increments.nbytes
        + np.dtype(np.float64).itemsize
    )
    return gain, post, {
        "rank": int(rank),
        "isotropic_floor": floor,
        "raw_representation_bytes": representation_bytes,
        "byte_budget": int(byte_budget),
    }


def gaussian_metrics(
    query_samples: np.ndarray,
    observation_samples: np.ndarray,
    query_mean: np.ndarray,
    observation_mean: np.ndarray,
    gain: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, Any]:
    innovations = observation_samples - observation_mean[None, :]
    predictions = query_mean[None, :] + innovations @ gain.T
    errors = query_samples - predictions
    root = np.linalg.cholesky(0.5 * (covariance + covariance.T))
    whitened = np.linalg.solve(root, errors.T)
    squared = np.sum(whitened * whitened, axis=0)
    dimension = query_samples.shape[1]
    logdet = 2.0 * float(np.log(np.diag(root)).sum())
    nll = 0.5 * (
        squared + logdet + dimension * math.log(2.0 * math.pi)
    )
    threshold = dimension * (
        1.0
        - 2.0 / (9.0 * dimension)
        + 1.6448536269514722 * math.sqrt(2.0 / (9.0 * dimension))
    ) ** 3
    return {
        "sample_count": int(query_samples.shape[0]),
        "rmse_per_coordinate": float(math.sqrt(float(np.mean(errors * errors)))),
        "vector_rmse": float(
            math.sqrt(float(np.mean(np.sum(errors * errors, axis=1))))
        ),
        "mean_normalized_nees": float(np.mean(squared) / dimension),
        "joint_coverage_95": float(np.mean(squared <= threshold)),
        "mean_nll_per_dimension": float(np.mean(nll) / dimension),
    }


def npz_payload(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def median_seconds(function: Callable[[], object], repeats: int) -> float:
    values: list[float] = []
    checksum = 0.0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = function()
        values.append((time.perf_counter_ns() - start) * 1.0e-9)
        if isinstance(result, bytes):
            checksum += float(len(result))
        elif result is not None:
            array = np.asarray(result)
            if array.size:
                checksum += float(array.reshape(-1)[0])
    if not math.isfinite(checksum):
        raise RuntimeError("benchmark checksum became nonfinite")
    return float(np.median(values))


def deserialize_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def evaluate_bank(
    *,
    sequence: str,
    query_count: int,
    fixed_mean: np.ndarray,
    raw_factor: np.ndarray,
    parameter_jacobian: np.ndarray,
    bootstrap_vectors: np.ndarray,
    provider_span: float,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    point_indices = select_query_indices(int(protocol["probe_count"]), query_count)
    query_rows = coordinate_indices(point_indices)
    alpha = float(protocol["dependence_strength"])
    noise_standard_deviation = (
        float(protocol["observation_noise_fraction"]) * provider_span
    )
    prior, cross, diagonal, full_factor = tempered_joint_model(
        raw_factor,
        query_rows,
        dependence_strength=alpha,
        noise_standard_deviation=noise_standard_deviation,
    )
    full_solver = DiagonalLowRankSolver(diagonal, full_factor)
    cache_start = time.perf_counter_ns()
    full_gain, full_posterior = posterior(prior, cross, full_solver)
    cache_construction_seconds = (time.perf_counter_ns() - cache_start) * 1.0e-9

    compression_start = time.perf_counter_ns()
    compression = compress_shared_factor_for_posterior(
        full_factor.reshape(int(protocol["probe_count"]), 3, -1),
        prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=full_solver,
        maximum_rank=min(prior.shape[0], full_factor.shape[1]),
        rank_relative_tolerance=float(protocol["rank_relative_tolerance"]),
        parity_relative_tolerance=float(protocol["parity_relative_tolerance"]),
    )
    compression_seconds = (time.perf_counter_ns() - compression_start) * 1.0e-9
    compressed_factor = compression.compressed_factor_m.reshape(
        full_factor.shape[0], -1
    )
    reduced_solver = DiagonalLowRankSolver(diagonal, compressed_factor)
    reduced_gain, reduced_posterior = posterior(prior, cross, reduced_solver)

    gain_scale = max(float(np.linalg.norm(full_gain, ord="fro")), 1.0e-30)
    posterior_scale = max(
        float(np.linalg.norm(full_posterior, ord="fro")), 1.0e-30
    )
    relative_gain_error = float(
        np.linalg.norm(reduced_gain - full_gain, ord="fro") / gain_scale
    )
    relative_posterior_error = float(
        np.linalg.norm(reduced_posterior - full_posterior, ord="fro")
        / posterior_scale
    )

    query_mean = fixed_mean[query_rows]
    delta = bootstrap_vectors - np.mean(bootstrap_vectors, axis=0, keepdims=True)
    # Preserve the measured, potentially non-Gaussian bootstrap parameter
    # draws while using the same registered local query Jacobian as the model.
    common_observation = (
        math.sqrt(alpha) * delta @ np.asarray(parameter_jacobian).T
    )
    rng = np.random.default_rng(
        int(protocol["monte_carlo_seed"]) + int(sequence[1:]) * 100 + query_count
    )
    query_raw = raw_factor[query_rows]
    query_diagonal = (
        noise_standard_deviation**2
        + (1.0 - alpha) * np.sum(query_raw * query_raw, axis=1)
    )
    observation_samples = fixed_mean[None, :] + common_observation + rng.normal(
        scale=np.sqrt(diagonal), size=common_observation.shape
    )
    query_samples = query_mean[None, :] + common_observation[:, query_rows] + rng.normal(
        scale=np.sqrt(query_diagonal),
        size=(common_observation.shape[0], query_rows.size),
    )

    compressed_budget = int(
        compressed_factor.nbytes + compression.latent_projection.nbytes
    )
    observation_covariance = np.diag(diagonal) + full_factor @ full_factor.T
    spectral_gain, spectral_posterior, spectral_details = spectral_joint_baseline(
        prior,
        cross,
        observation_covariance,
        byte_budget=compressed_budget,
    )

    full_metrics = gaussian_metrics(
        query_samples,
        observation_samples,
        query_mean,
        fixed_mean,
        full_gain,
        full_posterior,
    )
    compressed_metrics = gaussian_metrics(
        query_samples,
        observation_samples,
        query_mean,
        fixed_mean,
        reduced_gain,
        reduced_posterior,
    )
    spectral_metrics = gaussian_metrics(
        query_samples,
        observation_samples,
        query_mean,
        fixed_mean,
        spectral_gain,
        spectral_posterior,
    )

    cache_payload = npz_payload(gain=full_gain, posterior=full_posterior)
    factor_payload = npz_payload(factor=compressed_factor)
    projection_payload = npz_payload(projection=compression.latent_projection)
    factor_projection_payload = npz_payload(
        factor=compressed_factor,
        projection=compression.latent_projection,
    )
    self_contained_payload = npz_payload(
        factor=compressed_factor,
        diagonal=diagonal,
        prior=prior,
        cross=cross,
    )

    repeats = int(protocol["timing_repeats"])
    batch_count = int(protocol["benchmark_batch_size"])
    innovations = observation_samples - fixed_mean[None, :]
    if innovations.shape[0] < batch_count:
        innovations = np.tile(
            innovations,
            (math.ceil(batch_count / innovations.shape[0]), 1),
        )
    innovations = innovations[:batch_count]

    cache_update_seconds = median_seconds(
        lambda: innovations @ full_gain.T,
        repeats,
    )
    factor_direct_seconds = median_seconds(
        lambda: reduced_solver.solve(innovations.T).T @ cross.T,
        repeats,
    )
    cache_serialization_seconds = median_seconds(
        lambda: npz_payload(gain=full_gain, posterior=full_posterior),
        repeats,
    )
    factor_serialization_seconds = median_seconds(
        lambda: npz_payload(factor=compressed_factor),
        repeats,
    )
    self_serialization_seconds = median_seconds(
        lambda: npz_payload(
            factor=compressed_factor,
            diagonal=diagonal,
            prior=prior,
            cross=cross,
        ),
        repeats,
    )
    cache_deserialization_seconds = median_seconds(
        lambda: deserialize_npz(cache_payload)["gain"],
        repeats,
    )

    def materialize_resident_factor() -> np.ndarray:
        payload = deserialize_npz(factor_payload)
        solver = DiagonalLowRankSolver(diagonal, payload["factor"])
        gain, _ = posterior(prior, cross, solver)
        return gain

    def materialize_self_contained_factor() -> np.ndarray:
        payload = deserialize_npz(self_contained_payload)
        solver = DiagonalLowRankSolver(payload["diagonal"], payload["factor"])
        gain, _ = posterior(payload["prior"], payload["cross"], solver)
        return gain

    resident_materialization_seconds = median_seconds(
        materialize_resident_factor,
        repeats,
    )
    self_materialization_seconds = median_seconds(
        materialize_self_contained_factor,
        repeats,
    )

    maximum_mean_difference = float(
        np.max(
            np.linalg.norm(
                (observation_samples - fixed_mean[None, :])
                @ (reduced_gain - full_gain).T,
                axis=1,
            )
        )
    )
    repeated_break_even = math.inf
    per_sample_factor = factor_direct_seconds / batch_count
    per_sample_cache = cache_update_seconds / batch_count
    if per_sample_factor > per_sample_cache:
        repeated_break_even = math.ceil(
            resident_materialization_seconds
            / (per_sample_factor - per_sample_cache)
        )

    bandwidth_rows: list[dict[str, Any]] = []
    for bandwidth_mbps in protocol["bandwidth_mbps"]:
        bytes_per_second = float(bandwidth_mbps) * 1.0e6 / 8.0
        cache_one_time = (
            cache_construction_seconds
            + cache_serialization_seconds
            + len(cache_payload) / bytes_per_second
            + cache_deserialization_seconds
        )
        factor_one_time = (
            compression_seconds
            + factor_serialization_seconds
            + len(factor_payload) / bytes_per_second
            + resident_materialization_seconds
        )
        self_one_time = (
            compression_seconds
            + self_serialization_seconds
            + len(self_contained_payload) / bytes_per_second
            + self_materialization_seconds
        )
        bandwidth_rows.append(
            {
                "bandwidth_mbps": float(bandwidth_mbps),
                "cache_one_time_ms": 1000.0 * cache_one_time,
                "resident_factor_one_time_ms": 1000.0 * factor_one_time,
                "self_contained_factor_one_time_ms": 1000.0 * self_one_time,
                "resident_factor_wins": factor_one_time < cache_one_time,
                "self_contained_factor_wins": self_one_time < cache_one_time,
            }
        )

    return {
        "sequence": sequence,
        "query_point_count": query_count,
        "query_dimension": int(query_rows.size),
        "query_point_indices": point_indices.tolist(),
        "observation_dimension": int(full_factor.shape[0]),
        "dependence_strength": alpha,
        "noise_standard_deviation_provider_units": noise_standard_deviation,
        "original_shared_rank": int(full_factor.shape[1]),
        "compression": compression.summary(),
        "parity": {
            "relative_gain_error": relative_gain_error,
            "relative_posterior_covariance_error": relative_posterior_error,
            "maximum_realized_mean_difference_provider_units": maximum_mean_difference,
        },
        "payload": {
            "raw_bytes": {
                "full_factor": int(full_factor.nbytes),
                "compressed_factor": int(compressed_factor.nbytes),
                "latent_projection": int(compression.latent_projection.nbytes),
                "compressed_factor_plus_projection": compressed_budget,
                "joint_query_cache": int(full_gain.nbytes + full_posterior.nbytes),
                "self_contained_factor": int(
                    compressed_factor.nbytes
                    + diagonal.nbytes
                    + prior.nbytes
                    + cross.nbytes
                ),
                "spectral_baseline": spectral_details["raw_representation_bytes"],
            },
            "npz_bytes": {
                "compressed_factor": len(factor_payload),
                "latent_projection": len(projection_payload),
                "compressed_factor_plus_projection": len(factor_projection_payload),
                "joint_query_cache": len(cache_payload),
                "self_contained_factor": len(self_contained_payload),
            },
        },
        "timing": {
            "cache_construction_ms": 1000.0 * cache_construction_seconds,
            "compression_construction_ms": 1000.0 * compression_seconds,
            "batch_size": batch_count,
            "cache_update_us_per_query": 1.0e6 * cache_update_seconds / batch_count,
            "factor_direct_update_us_per_query": (
                1.0e6 * factor_direct_seconds / batch_count
            ),
            "cache_serialization_ms": 1000.0 * cache_serialization_seconds,
            "factor_serialization_ms": 1000.0 * factor_serialization_seconds,
            "self_contained_serialization_ms": 1000.0 * self_serialization_seconds,
            "cache_deserialization_ms": 1000.0 * cache_deserialization_seconds,
            "resident_factor_materialization_ms": (
                1000.0 * resident_materialization_seconds
            ),
            "self_contained_factor_materialization_ms": (
                1000.0 * self_materialization_seconds
            ),
            "factor_direct_to_materialized_cache_break_even_queries": (
                None if not math.isfinite(repeated_break_even) else int(repeated_break_even)
            ),
            "bandwidth_scenarios": bandwidth_rows,
        },
        "metrics": {
            "full": full_metrics,
            "posterior_preserving": compressed_metrics,
            "joint_query_cache": full_metrics,
            "complete_joint_spectral_same_budget": spectral_metrics,
        },
        "spectral_baseline": spectral_details,
    }


def evaluate_sequence(
    *,
    sequence: str,
    runs: Mapping[str, Mapping[str, np.ndarray]],
    frame_payloads: Mapping[int, tuple[np.ndarray, np.ndarray]],
    base: ModuleType,
    source_protocol: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    overlap_frames = [
        int(value) for value in source_protocol["evaluation"]["overlap_frames"]
    ]
    source, target, groups = base._collect_pair(
        runs["window_a"],
        runs["window_b"],
        frame_payloads,
        overlap_frames,
    )
    estimated, residuals = robust_fit_sim3(source, target)
    covariance, bootstrap = clustered_bootstrap_sim3(
        source,
        target,
        groups,
        replicates=int(source_protocol["uncertainty"]["bootstrap_replicates"]),
        seed=int(source_protocol["uncertainty"]["bootstrap_seed"])
        + int(sequence[1:]),
    )
    probes, provider_span = make_off_axis_probes(
        source,
        count=int(protocol["probe_count"]),
    )
    center = sim3_to_vector(estimated)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    minimum_steps = np.asarray(
        [
            1.0e-4 * provider_span,
            1.0e-4 * provider_span,
            1.0e-4 * provider_span,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-4,
        ]
    )
    steps = np.maximum(0.1 * standard_deviation, minimum_steps)
    fixed_mean = transform_probe_vector(center, probes)
    jacobian, _ = finite_difference_derivatives(
        lambda value: transform_probe_vector(value, probes),
        center,
        steps,
    )
    raw_factor, factor_details = measured_factor(
        covariance,
        jacobian,
        relative_tolerance=float(protocol["factor_eigenvalue_relative_tolerance"]),
    )
    bootstrap_vectors = np.stack([sim3_to_vector(item) for item in bootstrap])
    common_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    banks = [
        evaluate_bank(
            sequence=sequence,
            query_count=int(query_count),
            fixed_mean=fixed_mean,
            raw_factor=raw_factor,
            parameter_jacobian=jacobian,
            bootstrap_vectors=bootstrap_vectors,
            provider_span=provider_span,
            protocol=protocol,
        )
        for query_count in protocol["query_counts"]
    ]
    return {
        "sequence": sequence,
        "overlap_correspondence_count": int(source.shape[0]),
        "overlap_rmse_provider_units": float(
            math.sqrt(float(np.mean(residuals * residuals)))
        ),
        "provider_span": provider_span,
        "common_covariance_acquisition_seconds": common_seconds,
        "bootstrap_transform_count": int(len(bootstrap)),
        "factor": factor_details,
        "banks": banks,
    }


def aggregate(sequence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    banks = [bank for sequence in sequence_rows for bank in sequence["banks"]]
    groups: dict[str, Any] = {}
    for query_count in sorted({int(bank["query_point_count"]) for bank in banks}):
        selected = [
            bank for bank in banks if int(bank["query_point_count"]) == query_count
        ]
        raw_ratios = [
            bank["payload"]["raw_bytes"]["compressed_factor_plus_projection"]
            / bank["payload"]["raw_bytes"]["joint_query_cache"]
            for bank in selected
        ]
        npz_ratios = [
            bank["payload"]["npz_bytes"]["compressed_factor_plus_projection"]
            / bank["payload"]["npz_bytes"]["joint_query_cache"]
            for bank in selected
        ]
        groups[str(query_count)] = {
            "query_dimension": selected[0]["query_dimension"],
            "sequence_count": len(selected),
            "original_rank_min": min(bank["original_shared_rank"] for bank in selected),
            "original_rank_max": max(bank["original_shared_rank"] for bank in selected),
            "retained_rank_min": min(
                int(bank["compression"]["retained_rank"]) for bank in selected
            ),
            "retained_rank_max": max(
                int(bank["compression"]["retained_rank"]) for bank in selected
            ),
            "mean_raw_factor_plus_projection_to_cache": float(np.mean(raw_ratios)),
            "mean_npz_factor_plus_projection_to_cache": float(np.mean(npz_ratios)),
            "raw_factor_plus_projection_smaller_count": int(
                sum(ratio < 1.0 for ratio in raw_ratios)
            ),
            "npz_factor_plus_projection_smaller_count": int(
                sum(ratio < 1.0 for ratio in npz_ratios)
            ),
            "maximum_relative_gain_error": max(
                bank["parity"]["relative_gain_error"] for bank in selected
            ),
            "maximum_relative_posterior_covariance_error": max(
                bank["parity"]["relative_posterior_covariance_error"]
                for bank in selected
            ),
            "maximum_realized_mean_difference_provider_units": max(
                bank["parity"]["maximum_realized_mean_difference_provider_units"]
                for bank in selected
            ),
            "mean_cache_update_us_per_query": float(
                np.mean(
                    [bank["timing"]["cache_update_us_per_query"] for bank in selected]
                )
            ),
            "mean_factor_direct_update_us_per_query": float(
                np.mean(
                    [
                        bank["timing"]["factor_direct_update_us_per_query"]
                        for bank in selected
                    ]
                )
            ),
            "mean_cache_construction_ms": float(
                np.mean(
                    [bank["timing"]["cache_construction_ms"] for bank in selected]
                )
            ),
            "mean_compression_construction_ms": float(
                np.mean(
                    [
                        bank["timing"]["compression_construction_ms"]
                        for bank in selected
                    ]
                )
            ),
            "mean_full_nll_per_dimension": float(
                np.mean(
                    [
                        bank["metrics"]["full"]["mean_nll_per_dimension"]
                        for bank in selected
                    ]
                )
            ),
            "mean_spectral_nll_per_dimension": float(
                np.mean(
                    [
                        bank["metrics"]["complete_joint_spectral_same_budget"][
                            "mean_nll_per_dimension"
                        ]
                        for bank in selected
                    ]
                )
            ),
        }
    break_even_raw = next(
        (
            int(key)
            for key in sorted(groups, key=int)
            if groups[key]["raw_factor_plus_projection_smaller_count"]
            == groups[key]["sequence_count"]
        ),
        None,
    )
    break_even_npz = next(
        (
            int(key)
            for key in sorted(groups, key=int)
            if groups[key]["npz_factor_plus_projection_smaller_count"]
            == groups[key]["sequence_count"]
        ),
        None,
    )
    return {
        "sequence_count": len(sequence_rows),
        "bank_evaluation_count": len(banks),
        "groups": groups,
        "raw_all_sequence_break_even_query_points": break_even_raw,
        "npz_all_sequence_break_even_query_points": break_even_npz,
        "maximum_relative_gain_error": max(
            bank["parity"]["relative_gain_error"] for bank in banks
        ),
        "maximum_relative_posterior_covariance_error": max(
            bank["parity"]["relative_posterior_covariance_error"] for bank in banks
        ),
        "maximum_realized_mean_difference_provider_units": max(
            bank["parity"]["maximum_realized_mean_difference_provider_units"]
            for bank in banks
        ),
        "all_posteriors_valid": True,
    }


def summary_markdown(result: Mapping[str, Any]) -> str:
    if result["status"] != "evaluated-measured-dot-cut3r-querybank":
        return (
            "# DOT/CUT3R measured query-bank experiment\n\n"
            f"Status: **{result['status']}**\n\n"
            f"{result.get('reason', '')}\n"
        )
    aggregate_value = result["aggregate"]
    lines = [
        "# DOT/CUT3R measured query-bank experiment",
        "",
        "Status: **evaluated measured provider covariance**",
        "",
        f"- Source sequences: {aggregate_value['sequence_count']}",
        f"- Bank evaluations: {aggregate_value['bank_evaluation_count']}",
        f"- Maximum relative gain error: {aggregate_value['maximum_relative_gain_error']:.3e}",
        "- Maximum relative posterior covariance error: "
        f"{aggregate_value['maximum_relative_posterior_covariance_error']:.3e}",
        "- Raw all-sequence factor/cache break-even: "
        f"{aggregate_value['raw_all_sequence_break_even_query_points']}",
        "- NPZ all-sequence factor/cache break-even: "
        f"{aggregate_value['npz_all_sequence_break_even_query_points']}",
        "",
        "| Query points | Dimension | Measured rank | Retained rank | "
        "Raw factor/cache | NPZ factor/cache | Cache update (us) | "
        "Factor-direct update (us) | Full NLL/dim | Spectral NLL/dim |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(aggregate_value["groups"], key=int):
        row = aggregate_value["groups"][key]
        original = (
            str(row["original_rank_min"])
            if row["original_rank_min"] == row["original_rank_max"]
            else f"{row['original_rank_min']}--{row['original_rank_max']}"
        )
        retained = (
            str(row["retained_rank_min"])
            if row["retained_rank_min"] == row["retained_rank_max"]
            else f"{row['retained_rank_min']}--{row['retained_rank_max']}"
        )
        lines.append(
            "| {query} | {dimension} | {original} | {retained} | {raw:.3f} | "
            "{npz:.3f} | {cache:.3f} | {factor:.3f} | {full:.4f} | {spectral:.4f} |".format(
                query=key,
                dimension=row["query_dimension"],
                original=original,
                retained=retained,
                raw=row["mean_raw_factor_plus_projection_to_cache"],
                npz=row["mean_npz_factor_plus_projection_to_cache"],
                cache=row["mean_cache_update_us_per_query"],
                factor=row["mean_factor_direct_update_us_per_query"],
                full=row["mean_full_nll_per_dimension"],
                spectral=row["mean_spectral_nll_per_dimension"],
            )
        )
    lines.extend(
        [
            "",
            "The factor/cache payload ratio assumes the observation diagonal, "
            "query prior, cross covariance, and means are already resident. The "
            "self-contained payload is reported separately in result.json. A direct "
            "cache remains the reference implementation for one immutable query or "
            "when those model blocks are not resident.",
            "",
            "The measured covariance is reconstructed from the immutable R01--R03 "
            "CUT3R provider bundle and the already-authorized clustered marker "
            "bootstrap. The frozen dependence strength is 0.85. R04--R70 remain "
            "unopened.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    dataset_root: Path,
    provider_bundle: Path,
    protocol_path: Path,
    output_dir: Path,
    source_revision: str,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.mkdir(parents=True)
    protocol = load_protocol(protocol_path)
    protocol_bytes = protocol_path.read_bytes()
    pooled, base = _configure_registered_evaluator(protocol)
    source_protocol = base._load_protocol(SOURCE_PROTOCOL_PATH)
    manifest = base._verify_provider_bundle(provider_bundle, source_protocol)
    if manifest["provider_bundle_id"] != protocol["provider_bundle_id"]:
        raise ValueError("provider bundle identity differs from the registered artifact")
    if manifest["request_id"] != protocol["provider_request_id"]:
        raise ValueError("provider request identity changed")
    if manifest["prob4d_revision"] != protocol["provider_revision"]:
        raise ValueError("provider source revision changed")

    archive_path = dataset_root / str(protocol["archive"])
    if not archive_path.is_file():
        raise FileNotFoundError(f"official DOT archive is unavailable: {archive_path}")
    records_by_sequence: dict[str, dict[str, Mapping[str, Any]]] = {
        sequence: {} for sequence in protocol["source_sequences"]
    }
    for record in manifest["outputs"]:
        sequence = str(record["sequence"])
        if sequence in records_by_sequence:
            records_by_sequence[sequence][str(record["run"])] = record

    opened_members: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = set(archive.namelist())
        for sequence in protocol["source_sequences"]:
            frame_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for frame in source_protocol["frames"]:
                member_2d = base._coordinate_member(
                    sequence,
                    2,
                    int(frame),
                    str(source_protocol["camera"]),
                )
                member_3d = base._coordinate_member(
                    sequence,
                    3,
                    int(frame),
                    str(source_protocol["camera"]),
                )
                if member_2d not in names or member_3d not in names:
                    raise ValueError("registered DOT marker payload is missing")
                raw_2d = archive.read(member_2d)
                raw_3d = archive.read(member_3d)
                opened_members.extend(
                    [
                        {
                            "sequence": sequence,
                            "frame": int(frame),
                            "kind": "2d",
                            "member": member_2d,
                            "byte_count": len(raw_2d),
                            "sha256": _sha256_bytes(raw_2d),
                        },
                        {
                            "sequence": sequence,
                            "frame": int(frame),
                            "kind": "3d",
                            "member": member_3d,
                            "byte_count": len(raw_3d),
                            "sha256": _sha256_bytes(raw_3d),
                        },
                    ]
                )
                frame_payloads[int(frame)] = (
                    pooled._parse_coordinate_text(raw_2d.decode("utf-8"), 2),
                    pooled._parse_coordinate_text(raw_3d.decode("utf-8"), 3),
                )
            runs = {
                run_name: base._load_run(
                    provider_bundle,
                    records_by_sequence[sequence][run_name],
                )
                for run_name in ("continuous", "window_a", "window_b")
            }
            sequence_rows.append(
                evaluate_sequence(
                    sequence=sequence,
                    runs=runs,
                    frame_payloads=frame_payloads,
                    base=base,
                    source_protocol=source_protocol,
                    protocol=protocol,
                )
            )

    aggregate_value = aggregate(sequence_rows)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "evaluated-measured-dot-cut3r-querybank",
        "source_revision": source_revision,
        "protocol_id": protocol["protocol_id"],
        "provider_bundle_id": manifest["provider_bundle_id"],
        "provider_run_id": protocol["provider_run_id"],
        "provider_artifact_name": protocol["provider_artifact_name"],
        "dependence_calibration_id": protocol["dependence_calibration_id"],
        "dependence_strength": protocol["dependence_strength"],
        "sequence_results": sequence_rows,
        "aggregate": aggregate_value,
        "opened_marker_members": opened_members,
        "marker_support": {
            "marker_frames": sorted(
                pooled._MARKER_DIAGNOSTICS.values(),
                key=lambda row: (row["sequence"], row["run"], row["frame"]),
            ),
            "collections": pooled._COLLECTION_DIAGNOSTICS,
        },
        "runtime": {
            "total_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "information_boundary": {
            "sealed_provider_predictions_reused": True,
            "provider_inference_rerun": False,
            "opened_sequences": protocol["source_sequences"],
            "reserved_sequences": protocol["reserved_sequences"],
            "r04_r10_confirmation_consumed": False,
            "target_payloads_opened": False,
            "raw_marker_payloads_written_to_evidence": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_id"] = content_id(result)
    result_bytes = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    (output_dir / "result.json").write_bytes(result_bytes)
    (output_dir / "summary.md").write_text(
        summary_markdown(result),
        encoding="utf-8",
    )
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    manifest_value = {
        "schema": "prob4d.dot-rope-cut3r-measured-querybank-manifest.v1",
        "result_id": result["result_id"],
        "source_revision": source_revision,
        "protocol_sha256": _sha256_bytes(protocol_bytes),
        "result_sha256": _sha256_bytes(result_bytes),
        "provider_bundle_id": manifest["provider_bundle_id"],
        "official_archive_sha256": _sha256_file(archive_path),
        "opened_marker_manifest_sha256": _sha256_bytes(
            _canonical_json(opened_members)
        ),
        "raw_data_copied_to_evidence": False,
    }
    _write_json(output_dir / "manifest.json", manifest_value)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--provider-bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(
        dataset_root=args.dataset_root.resolve(strict=True),
        provider_bundle=args.provider_bundle.resolve(strict=True),
        protocol_path=args.protocol,
        output_dir=args.output_dir,
        source_revision=_validate_hex(
            args.source_revision,
            label="source_revision",
            length=40,
        ),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "result_id": result["result_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
