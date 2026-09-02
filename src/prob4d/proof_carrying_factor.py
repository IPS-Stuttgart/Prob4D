"""Build portable proof-carrying certificates for partial physical factors.

Version 1 covers one deliberately small claim: a linearized downstream query is
supported by the observable subspace of an :class:`ObservableGaugeFactor`.
The resulting JSON object is content-addressed and can be checked by
``prob4d_independent_verifier`` without importing the producing Prob4D module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text
from ._version import __version__
from .observable_gauge import ObservableGaugeFactor

FloatArray: TypeAlias = NDArray[np.floating[Any]]

PROOF_CARRYING_FACTOR_SCHEMA = "prob4d.proof-carrying-linear-query-factor"
PROOF_CARRYING_FACTOR_VERSION = 1
PROOF_CARRYING_FACTOR_KIND = "observable-gauge-linear-query-v1"
PROOF_CARRYING_FACTOR_CLAIM_SCOPE = "local-first-order-query-identifiability-only"
EXACT_FALLBACK_POLICY = "exact-caller-owned-fallback"
DEFAULT_ASSUMPTION_IDS = (
    "declared-centroid-normalized-sim3-chart-v1",
    "declared-local-query-jacobian-v1",
    "local-linearization-only-v1",
)
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def compute_proof_carrying_factor_id(certificate: Mapping[str, object]) -> str:
    """Return the content ID after excluding the ``certificate_id`` field."""

    payload = dict(certificate)
    payload.pop("certificate_id", None)
    return _sha256(payload)


def render_proof_carrying_factor(certificate: Mapping[str, object]) -> str:
    """Render one certificate in the canonical human-readable JSON form."""

    expected = compute_proof_carrying_factor_id(certificate)
    if certificate.get("certificate_id") != expected:
        raise ValueError("certificate_id does not match the certificate contents")
    return (
        json.dumps(
            certificate,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def write_proof_carrying_factor(
    path: str | Path,
    certificate: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one complete certificate."""

    atomic_write_text(
        path,
        render_proof_carrying_factor(certificate),
        overwrite=overwrite,
    )


def _nonempty_text(value: object, *, name: str, maximum_length: int = 512) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty and have no surrounding whitespace")
    if len(value) > maximum_length:
        raise ValueError(f"{name} is too long")
    return value


def _digest(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name, maximum_length=71)
    if _DIGEST_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase sha256: digest")
    return text


def _unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return numeric


def _matrix(
    value: object,
    *,
    name: str,
    columns: int | None = None,
    square: bool = False,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if columns is not None and (array.shape[0] < 1 or array.shape[1] != columns):
        raise ValueError(f"{name} must have shape (Q, {columns}) with Q positive")
    if square and (array.shape[0] < 1 or array.shape[0] != array.shape[1]):
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _spectral_norm(value: FloatArray, *, name: str) -> float:
    singular_values = np.linalg.svd(value, compute_uv=False)
    result = 0.0 if singular_values.size == 0 else float(singular_values[0])
    if not math.isfinite(result):
        raise ValueError(f"{name} spectral norm must be finite")
    return result


def _positive_definite(value: object, *, name: str) -> FloatArray:
    matrix = _matrix(value, name=name, square=True)
    symmetric = 0.5 * (matrix + matrix.T)
    if not np.allclose(matrix, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(f"{name} must be positive definite")
    return symmetric


def _assumption_ids(values: Iterable[str]) -> list[str]:
    result = sorted({_nonempty_text(value, name="assumption_id") for value in values})
    if not result:
        raise ValueError("assumption_ids must not be empty")
    return result


def _source_factor_digest(factor: ObservableGaugeFactor) -> str:
    context: dict[str, object] = {
        "chart": {
            "kind": "centroid-normalized-sim3-v1",
            "linearization_vector": factor.chart.linearization.as_vector().tolist(),
            "source_centroid": factor.chart.source_centroid.tolist(),
            "cloud_scale": factor.chart.cloud_scale,
        },
        "diagnostics": {
            "normalized_geometry_spectrum": factor.normalized_geometry_spectrum.tolist(),
            "rank_threshold": factor.rank_threshold,
            "residual_rms": factor.residual_rms,
            "residual_variance": factor.residual_variance,
            "inlier_fraction": factor.inlier_fraction,
            "num_correspondences": factor.num_correspondences,
            "covariance_method": factor.covariance_method,
            "num_covariance_clusters": factor.num_covariance_clusters,
        },
    }
    return _sha256(context)


def build_observable_gauge_query_certificate(
    factor: ObservableGaugeFactor,
    *,
    query_jacobian_local: object,
    query_id: str,
    query_program_digest: str,
    fallback_id: str,
    input_digest: str,
    query_metric: object | None = None,
    maximum_relative_nullspace_sensitivity: float = 1e-8,
    producer: str = "prob4d",
    producer_version: str = __version__,
    assumption_ids: Iterable[str] = DEFAULT_ASSUMPTION_IDS,
    calibration_receipt_digest: str | None = None,
) -> dict[str, object]:
    """Build a content-addressed local query-support certificate.

    The certificate does not assert that the provider inputs are truthful or
    that a nonlinear query is globally invariant. It binds those assumptions
    explicitly and proves only the declared local linear-algebra statement.
    """

    jacobian = _matrix(
        query_jacobian_local,
        name="query_jacobian_local",
        columns=7,
    )
    metric = (
        np.eye(jacobian.shape[0], dtype=np.float64)
        if query_metric is None
        else _positive_definite(query_metric, name="query_metric")
    )
    if metric.shape != (jacobian.shape[0], jacobian.shape[0]):
        raise ValueError("query_metric must match the query dimension")
    threshold = _unit_interval(
        maximum_relative_nullspace_sensitivity,
        name="maximum_relative_nullspace_sensitivity",
    )
    declared_query_id = _nonempty_text(query_id, name="query_id")
    declared_query_digest = _digest(
        query_program_digest,
        name="query_program_digest",
    )
    fallback = _nonempty_text(fallback_id, name="fallback_id")
    provider = _nonempty_text(producer, name="producer")
    provider_version = _nonempty_text(producer_version, name="producer_version")
    source_digest = _digest(input_digest, name="input_digest")
    calibration_digest = (
        None
        if calibration_receipt_digest is None
        else _digest(
            calibration_receipt_digest,
            name="calibration_receipt_digest",
        )
    )
    assumptions = _assumption_ids(assumption_ids)

    metric_sqrt = np.linalg.cholesky(metric).T
    weighted_jacobian = metric_sqrt @ jacobian
    observable_coordinates = weighted_jacobian @ factor.observable_basis
    denominator = max(
        _spectral_norm(weighted_jacobian, name="weighted query Jacobian"),
        float(np.finfo(np.float64).tiny),
    )
    nullspace_sensitivity = (
        0.0
        if factor.nullspace_basis.shape[1] == 0
        else _spectral_norm(
            weighted_jacobian @ factor.nullspace_basis,
            name="weighted query nullspace component",
        )
        / denominator
    )
    reconstruction_residual = (
        _spectral_norm(
            weighted_jacobian - observable_coordinates @ factor.observable_basis.T,
            name="weighted query reconstruction residual",
        )
        / denominator
    )
    admitted = nullspace_sensitivity <= threshold

    certificate: dict[str, object] = {
        "schema": PROOF_CARRYING_FACTOR_SCHEMA,
        "schema_version": PROOF_CARRYING_FACTOR_VERSION,
        "certificate_kind": PROOF_CARRYING_FACTOR_KIND,
        "claim_scope": PROOF_CARRYING_FACTOR_CLAIM_SCOPE,
        "factor": {
            "coordinate_dimension": 7,
            "rank": factor.rank,
            "coordinate_system": {
                "kind": "centroid-normalized-sim3-v1",
                "linearization_vector": factor.chart.linearization.as_vector().tolist(),
                "source_centroid": factor.chart.source_centroid.tolist(),
                "cloud_scale": factor.chart.cloud_scale,
            },
            "observable_basis": factor.observable_basis.tolist(),
            "nullspace_basis": factor.nullspace_basis.tolist(),
            "observable_information": factor.observable_information.tolist(),
            "information_matrix": factor.information_matrix.tolist(),
        },
        "query": {
            "query_id": declared_query_id,
            "query_program_digest": declared_query_digest,
            "query_dimension": int(jacobian.shape[0]),
            "jacobian_local": jacobian.tolist(),
            "metric": metric.tolist(),
            "observable_coordinates": observable_coordinates.tolist(),
            "maximum_relative_nullspace_sensitivity": threshold,
            "reported_relative_nullspace_sensitivity": nullspace_sensitivity,
            "reported_relative_reconstruction_residual": reconstruction_residual,
        },
        "decision": {
            "admitted": admitted,
            "fallback_policy": EXACT_FALLBACK_POLICY,
            "fallback_id": fallback,
        },
        "provenance": {
            "producer": provider,
            "producer_version": provider_version,
            "input_digest": source_digest,
            "source_factor_digest": _source_factor_digest(factor),
            "assumption_ids": assumptions,
            "calibration_receipt_digest": calibration_digest,
        },
    }
    certificate["certificate_id"] = compute_proof_carrying_factor_id(certificate)
    return certificate


__all__ = [
    "DEFAULT_ASSUMPTION_IDS",
    "EXACT_FALLBACK_POLICY",
    "PROOF_CARRYING_FACTOR_CLAIM_SCOPE",
    "PROOF_CARRYING_FACTOR_KIND",
    "PROOF_CARRYING_FACTOR_SCHEMA",
    "PROOF_CARRYING_FACTOR_VERSION",
    "build_observable_gauge_query_certificate",
    "compute_proof_carrying_factor_id",
    "render_proof_carrying_factor",
    "write_proof_carrying_factor",
]
