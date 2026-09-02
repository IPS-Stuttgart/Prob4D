"""Independent verification of Proof4D linear-query factor certificates.

This module intentionally does not import :mod:`prob4d`. It validates the JSON
shape, content address, subspace decomposition, information witness, query
witness, producer decision, and exact-fallback declaration from first
principles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]

PROOF_CARRYING_FACTOR_SCHEMA = "prob4d.proof-carrying-linear-query-factor"
PROOF_CARRYING_FACTOR_VERSION = 1
PROOF_CARRYING_FACTOR_KIND = "observable-gauge-linear-query-v1"
PROOF_CARRYING_FACTOR_CLAIM_SCOPE = "local-first-order-query-identifiability-only"
EXACT_FALLBACK_POLICY = "exact-caller-owned-fallback"

VERIFICATION_SCHEMA = "prob4d.proof-carrying-factor-verification"
VERIFICATION_VERSION = 1
VERIFIER_IMPLEMENTATION = "prob4d-independent-proof-carrying-v1"
VERIFICATION_CLAIM_BOUNDARY = (
    "The verifier checks internal algebraic consistency and local first-order "
    "query support in the declared chart.",
    "It does not verify sensor truth, provider competence, assumption truth, "
    "global nonlinear invariance, statistical calibration, or deployment safety.",
    "A rejected or invalid certificate is never admitted; exact fallback remains "
    "caller-owned and is referenced by its declared identifier.",
)

_MAX_JSON_BYTES = 16 * 1024**2
_MAX_COORDINATE_DIMENSION = 256
_MAX_QUERY_DIMENSION = 256
_MATRIX_RELATIVE_TOLERANCE = 1e-8
_REPORTED_VALUE_ATOL = 1e-12
_REPORTED_VALUE_RTOL = 1e-9
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {
    "schema",
    "schema_version",
    "certificate_kind",
    "claim_scope",
    "factor",
    "query",
    "decision",
    "provenance",
    "certificate_id",
}
_FACTOR_KEYS = {
    "coordinate_dimension",
    "rank",
    "coordinate_system",
    "observable_basis",
    "nullspace_basis",
    "observable_information",
    "information_matrix",
}
_COORDINATE_SYSTEM_KEYS = {
    "kind",
    "linearization_vector",
    "source_centroid",
    "cloud_scale",
}
_QUERY_KEYS = {
    "query_id",
    "query_program_digest",
    "query_dimension",
    "jacobian_local",
    "metric",
    "observable_coordinates",
    "maximum_relative_nullspace_sensitivity",
    "reported_relative_nullspace_sensitivity",
    "reported_relative_reconstruction_residual",
}
_DECISION_KEYS = {"admitted", "fallback_policy", "fallback_id"}
_PROVENANCE_KEYS = {
    "producer",
    "producer_version",
    "input_digest",
    "source_factor_digest",
    "assumption_ids",
    "calibration_receipt_digest",
}


class _InvalidCertificate(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ProofCarryingFactorVerification:
    """Fail-closed result of independent certificate verification."""

    decision: str
    valid: bool
    admitted: bool
    certificate_id: str | None
    reason_codes: tuple[str, ...]
    detail: str
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": VERIFICATION_SCHEMA,
            "schema_version": VERIFICATION_VERSION,
            "verifier_implementation": VERIFIER_IMPLEMENTATION,
            "decision": self.decision,
            "valid": self.valid,
            "admitted": self.admitted,
            "certificate_id": self.certificate_id,
            "reason_codes": list(self.reason_codes),
            "detail": self.detail,
            "metrics": dict(self.metrics),
            "claim_boundary": list(VERIFICATION_CLAIM_BOUNDARY),
        }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _certificate_id(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("certificate_id", None)
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _reject_constant(value: str) -> None:
    raise _InvalidCertificate(
        "non-finite-json-number",
        f"non-finite JSON number {value!r} is forbidden",
    )


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidCertificate("duplicate-json-key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _InvalidCertificate("artifact-unreadable", str(error)) from error
    if size > _MAX_JSON_BYTES:
        raise _InvalidCertificate(
            "artifact-too-large",
            f"certificate exceeds {_MAX_JSON_BYTES} bytes",
        )
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_object,
        )
    except _InvalidCertificate:
        raise
    except json.JSONDecodeError as error:
        raise _InvalidCertificate("invalid-json", str(error)) from error
    if type(value) is not dict:
        raise _InvalidCertificate(
            "top-level-not-object",
            "certificate must contain one JSON object",
        )
    return value


def _require_keys(value: object, expected: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise _InvalidCertificate("invalid-object", f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise _InvalidCertificate(
            "unexpected-object-keys",
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )
    return value


def _integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _InvalidCertificate("invalid-integer", f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise _InvalidCertificate(
            "integer-out-of-range",
            f"{name} must lie in [{minimum}, {maximum}]",
        )
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _InvalidCertificate("invalid-number", f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _InvalidCertificate("non-finite-number", f"{name} must be finite")
    return numeric


def _unit_interval(value: object, *, name: str) -> float:
    numeric = _finite_float(value, name=name)
    if not 0.0 <= numeric <= 1.0:
        raise _InvalidCertificate(
            "number-out-of-range",
            f"{name} must lie in [0, 1]",
        )
    return numeric


def _nonempty_text(value: object, *, name: str, maximum_length: int = 512) -> str:
    if type(value) is not str:
        raise _InvalidCertificate("invalid-string", f"{name} must be a string")
    if not value or value.strip() != value or len(value) > maximum_length:
        raise _InvalidCertificate(
            "invalid-string",
            f"{name} must be nonempty, trimmed, and at most {maximum_length} characters",
        )
    return value


def _digest(value: object, *, name: str) -> str:
    text = _nonempty_text(value, name=name, maximum_length=71)
    if _DIGEST_PATTERN.fullmatch(text) is None:
        raise _InvalidCertificate(
            "invalid-digest",
            f"{name} must be a lowercase sha256: digest",
        )
    return text


def _matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> FloatArray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _InvalidCertificate(
            "invalid-matrix",
            f"{name} cannot be converted to float64",
        ) from error
    if matrix.shape != shape:
        raise _InvalidCertificate(
            "invalid-matrix-shape",
            f"{name} must have shape {shape}, got {matrix.shape}",
        )
    if not np.all(np.isfinite(matrix)):
        raise _InvalidCertificate("non-finite-matrix", f"{name} must be finite")
    return matrix


def _spectral_norm(value: FloatArray, *, name: str) -> float:
    singular_values = np.linalg.svd(value, compute_uv=False)
    result = 0.0 if singular_values.size == 0 else float(singular_values[0])
    if not math.isfinite(result):
        raise _InvalidCertificate("numeric-overflow", f"{name} spectral norm overflowed")
    return result


def _relative_norm(residual: FloatArray, reference: FloatArray) -> float:
    denominator = max(
        _spectral_norm(reference, name="reference"),
        float(np.finfo(np.float64).tiny),
    )
    return _spectral_norm(residual, name="residual") / denominator


def _require_small(value: float, *, code: str, name: str) -> None:
    if value > _MATRIX_RELATIVE_TOLERANCE:
        raise _InvalidCertificate(
            code,
            f"{name}={value:.6g} exceeds {_MATRIX_RELATIVE_TOLERANCE:.6g}",
        )


def _verify_factor(
    value: object,
) -> tuple[int, int, FloatArray, FloatArray, dict[str, float]]:
    factor = _require_keys(value, _FACTOR_KEYS, name="factor")
    dimension = _integer(
        factor["coordinate_dimension"],
        name="factor.coordinate_dimension",
        minimum=1,
        maximum=_MAX_COORDINATE_DIMENSION,
    )
    rank = _integer(
        factor["rank"],
        name="factor.rank",
        minimum=1,
        maximum=dimension,
    )
    nullity = dimension - rank
    coordinate_system = _require_keys(
        factor["coordinate_system"],
        _COORDINATE_SYSTEM_KEYS,
        name="factor.coordinate_system",
    )
    if coordinate_system["kind"] != "centroid-normalized-sim3-v1" or dimension != 7:
        raise _InvalidCertificate(
            "unsupported-coordinate-system",
            "version 1 supports only the seven-dimensional centroid-normalized Sim(3) chart",
        )
    _matrix(
        [coordinate_system["linearization_vector"]],
        name="factor.coordinate_system.linearization_vector",
        shape=(1, 7),
    )
    _matrix(
        [coordinate_system["source_centroid"]],
        name="factor.coordinate_system.source_centroid",
        shape=(1, 3),
    )
    if (
        _finite_float(
            coordinate_system["cloud_scale"],
            name="factor.coordinate_system.cloud_scale",
        )
        <= 0.0
    ):
        raise _InvalidCertificate(
            "invalid-coordinate-scale",
            "factor.coordinate_system.cloud_scale must be positive",
        )

    observable = _matrix(
        factor["observable_basis"],
        name="factor.observable_basis",
        shape=(dimension, rank),
    )
    nullspace = _matrix(
        factor["nullspace_basis"],
        name="factor.nullspace_basis",
        shape=(dimension, nullity),
    )
    observable_information = _matrix(
        factor["observable_information"],
        name="factor.observable_information",
        shape=(rank, rank),
    )
    information = _matrix(
        factor["information_matrix"],
        name="factor.information_matrix",
        shape=(dimension, dimension),
    )

    combined = np.concatenate((observable, nullspace), axis=1)
    basis_residual = _spectral_norm(
        combined.T @ combined - np.eye(dimension),
        name="basis orthonormality residual",
    )
    _require_small(
        basis_residual,
        code="subspace-basis-not-orthonormal",
        name="basis_orthonormality_residual",
    )

    information_symmetry = _relative_norm(information - information.T, information)
    observable_symmetry = _relative_norm(
        observable_information - observable_information.T,
        observable_information,
    )
    _require_small(
        information_symmetry,
        code="information-not-symmetric",
        name="information_symmetry_residual",
    )
    _require_small(
        observable_symmetry,
        code="observable-information-not-symmetric",
        name="observable_information_symmetry_residual",
    )
    observable_symmetric = 0.5 * (observable_information + observable_information.T)
    observable_eigenvalues = np.linalg.eigvalsh(observable_symmetric)
    if float(observable_eigenvalues[0]) <= 0.0:
        raise _InvalidCertificate(
            "observable-information-not-positive-definite",
            "factor.observable_information must be positive definite",
        )
    information_symmetric = 0.5 * (information + information.T)
    information_scale = max(
        _spectral_norm(information_symmetric, name="information matrix"),
        float(np.finfo(np.float64).tiny),
    )
    information_eigenvalues = np.linalg.eigvalsh(information_symmetric)
    if float(information_eigenvalues[0]) < (-_MATRIX_RELATIVE_TOLERANCE * information_scale):
        raise _InvalidCertificate(
            "information-not-positive-semidefinite",
            "factor.information_matrix must be positive semidefinite",
        )

    reconstructed = observable @ observable_information @ observable.T
    reconstruction_residual = _relative_norm(information - reconstructed, information)
    nullspace_residual = (
        0.0
        if nullity == 0
        else _spectral_norm(
            information @ nullspace,
            name="information nullspace component",
        )
        / information_scale
    )
    _require_small(
        reconstruction_residual,
        code="information-witness-mismatch",
        name="information_reconstruction_residual",
    )
    _require_small(
        nullspace_residual,
        code="information-leaks-into-nullspace",
        name="information_nullspace_residual",
    )
    metrics = {
        "basis_orthonormality_residual": basis_residual,
        "information_symmetry_residual": information_symmetry,
        "observable_information_symmetry_residual": observable_symmetry,
        "information_reconstruction_residual": reconstruction_residual,
        "information_nullspace_residual": nullspace_residual,
    }
    return dimension, rank, observable, nullspace, metrics


def _verify_query(
    value: object,
    *,
    dimension: int,
    rank: int,
    observable: FloatArray,
    nullspace: FloatArray,
) -> tuple[bool, dict[str, float]]:
    query = _require_keys(value, _QUERY_KEYS, name="query")
    _nonempty_text(query["query_id"], name="query.query_id")
    _digest(query["query_program_digest"], name="query.query_program_digest")
    query_dimension = _integer(
        query["query_dimension"],
        name="query.query_dimension",
        minimum=1,
        maximum=_MAX_QUERY_DIMENSION,
    )
    jacobian = _matrix(
        query["jacobian_local"],
        name="query.jacobian_local",
        shape=(query_dimension, dimension),
    )
    metric = _matrix(
        query["metric"],
        name="query.metric",
        shape=(query_dimension, query_dimension),
    )
    metric_symmetry = _relative_norm(metric - metric.T, metric)
    _require_small(
        metric_symmetry,
        code="query-metric-not-symmetric",
        name="query_metric_symmetry_residual",
    )
    metric_symmetric = 0.5 * (metric + metric.T)
    metric_eigenvalues = np.linalg.eigvalsh(metric_symmetric)
    if float(metric_eigenvalues[0]) <= 0.0:
        raise _InvalidCertificate(
            "query-metric-not-positive-definite",
            "query.metric must be positive definite",
        )
    coordinates = _matrix(
        query["observable_coordinates"],
        name="query.observable_coordinates",
        shape=(query_dimension, rank),
    )
    threshold = _unit_interval(
        query["maximum_relative_nullspace_sensitivity"],
        name="query.maximum_relative_nullspace_sensitivity",
    )
    reported_nullspace = _unit_interval(
        query["reported_relative_nullspace_sensitivity"],
        name="query.reported_relative_nullspace_sensitivity",
    )
    reported_reconstruction = _unit_interval(
        query["reported_relative_reconstruction_residual"],
        name="query.reported_relative_reconstruction_residual",
    )

    weighted = np.linalg.cholesky(metric_symmetric).T @ jacobian
    expected_coordinates = weighted @ observable
    witness_residual = _relative_norm(
        coordinates - expected_coordinates,
        expected_coordinates,
    )
    _require_small(
        witness_residual,
        code="query-observable-witness-mismatch",
        name="query_observable_witness_residual",
    )
    denominator = max(
        _spectral_norm(weighted, name="weighted query Jacobian"),
        float(np.finfo(np.float64).tiny),
    )
    nullspace_sensitivity = (
        0.0
        if nullspace.shape[1] == 0
        else _spectral_norm(
            weighted @ nullspace,
            name="weighted query nullspace component",
        )
        / denominator
    )
    reconstruction_residual = (
        _spectral_norm(
            weighted - coordinates @ observable.T,
            name="weighted query reconstruction residual",
        )
        / denominator
    )
    if not np.isclose(
        reported_nullspace,
        nullspace_sensitivity,
        atol=_REPORTED_VALUE_ATOL,
        rtol=_REPORTED_VALUE_RTOL,
    ):
        raise _InvalidCertificate(
            "reported-nullspace-sensitivity-mismatch",
            "reported query nullspace sensitivity does not match recomputation",
        )
    if not np.isclose(
        reported_reconstruction,
        reconstruction_residual,
        atol=_REPORTED_VALUE_ATOL,
        rtol=_REPORTED_VALUE_RTOL,
    ):
        raise _InvalidCertificate(
            "reported-reconstruction-residual-mismatch",
            "reported query reconstruction residual does not match recomputation",
        )
    if not np.isclose(
        nullspace_sensitivity,
        reconstruction_residual,
        atol=_REPORTED_VALUE_ATOL,
        rtol=_REPORTED_VALUE_RTOL,
    ):
        raise _InvalidCertificate(
            "subspace-decomposition-inconsistent",
            "nullspace and reconstruction residuals disagree",
        )
    return nullspace_sensitivity <= threshold, {
        "query_metric_symmetry_residual": metric_symmetry,
        "query_observable_witness_residual": witness_residual,
        "relative_query_nullspace_sensitivity": nullspace_sensitivity,
        "relative_query_reconstruction_residual": reconstruction_residual,
        "admission_threshold": threshold,
    }


def _verify_decision_and_provenance(
    certificate: dict[str, object],
    *,
    recomputed_admitted: bool,
) -> None:
    decision = _require_keys(
        certificate["decision"],
        _DECISION_KEYS,
        name="decision",
    )
    if type(decision["admitted"]) is not bool:
        raise _InvalidCertificate(
            "invalid-producer-decision",
            "decision.admitted must be a bool",
        )
    if decision["admitted"] is not recomputed_admitted:
        raise _InvalidCertificate(
            "producer-decision-mismatch",
            "producer admission does not match the verified query certificate",
        )
    if decision["fallback_policy"] != EXACT_FALLBACK_POLICY:
        raise _InvalidCertificate(
            "unsupported-fallback-policy",
            "decision.fallback_policy must require exact caller-owned fallback",
        )
    _nonempty_text(decision["fallback_id"], name="decision.fallback_id")

    provenance = _require_keys(
        certificate["provenance"],
        _PROVENANCE_KEYS,
        name="provenance",
    )
    _nonempty_text(provenance["producer"], name="provenance.producer")
    _nonempty_text(
        provenance["producer_version"],
        name="provenance.producer_version",
    )
    _digest(provenance["input_digest"], name="provenance.input_digest")
    _digest(
        provenance["source_factor_digest"],
        name="provenance.source_factor_digest",
    )
    assumptions = provenance["assumption_ids"]
    if type(assumptions) is not list or not assumptions:
        raise _InvalidCertificate(
            "invalid-assumption-list",
            "provenance.assumption_ids must be a nonempty list",
        )
    normalized = [_nonempty_text(item, name="provenance.assumption_id") for item in assumptions]
    if normalized != sorted(set(normalized)):
        raise _InvalidCertificate(
            "invalid-assumption-list",
            "assumption IDs must be sorted and unique",
        )
    calibration = provenance["calibration_receipt_digest"]
    if calibration is not None:
        _digest(calibration, name="provenance.calibration_receipt_digest")


def _semantic_verify(
    certificate: dict[str, object],
) -> ProofCarryingFactorVerification:
    root = _require_keys(certificate, _TOP_LEVEL_KEYS, name="certificate")
    certificate_id = _digest(root["certificate_id"], name="certificate_id")
    expected_id = _certificate_id(root)
    if certificate_id != expected_id:
        raise _InvalidCertificate(
            "certificate-id-mismatch",
            f"expected {expected_id}, got {certificate_id}",
        )
    if root["schema"] != PROOF_CARRYING_FACTOR_SCHEMA:
        raise _InvalidCertificate("unsupported-schema", "unsupported certificate schema")
    if root["schema_version"] != PROOF_CARRYING_FACTOR_VERSION:
        raise _InvalidCertificate(
            "unsupported-schema-version",
            "unsupported certificate schema version",
        )
    if root["certificate_kind"] != PROOF_CARRYING_FACTOR_KIND:
        raise _InvalidCertificate("unsupported-kind", "unsupported certificate kind")
    if root["claim_scope"] != PROOF_CARRYING_FACTOR_CLAIM_SCOPE:
        raise _InvalidCertificate("invalid-claim-scope", "claim scope was widened")

    dimension, rank, observable, nullspace, factor_metrics = _verify_factor(root["factor"])
    admitted, query_metrics = _verify_query(
        root["query"],
        dimension=dimension,
        rank=rank,
        observable=observable,
        nullspace=nullspace,
    )
    _verify_decision_and_provenance(root, recomputed_admitted=admitted)
    metrics = {**factor_metrics, **query_metrics}
    if admitted:
        return ProofCarryingFactorVerification(
            decision="verified-admit",
            valid=True,
            admitted=True,
            certificate_id=certificate_id,
            reason_codes=(),
            detail="certificate is internally valid and the declared local query is supported",
            metrics=metrics,
        )
    return ProofCarryingFactorVerification(
        decision="verified-reject",
        valid=True,
        admitted=False,
        certificate_id=certificate_id,
        reason_codes=("query-nullspace-sensitivity-exceeds-threshold",),
        detail="certificate is internally valid but the declared local query is unsupported",
        metrics=metrics,
    )


def verify_proof_carrying_factor(
    artifact: str | Path | Mapping[str, object],
) -> ProofCarryingFactorVerification:
    """Verify one certificate and never admit malformed input."""

    certificate_id: str | None = None
    try:
        if isinstance(artifact, Mapping):
            value = json.loads(
                _canonical_bytes(dict(artifact)).decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_no_duplicate_object,
            )
            if type(value) is not dict:
                raise _InvalidCertificate(
                    "top-level-not-object",
                    "certificate must contain one JSON object",
                )
            certificate = value
        else:
            certificate = _load_json(Path(artifact))
        raw_id = certificate.get("certificate_id")
        if type(raw_id) is str:
            certificate_id = raw_id
        return _semantic_verify(certificate)
    except _InvalidCertificate as error:
        return ProofCarryingFactorVerification(
            decision="invalid-fail-closed",
            valid=False,
            admitted=False,
            certificate_id=certificate_id,
            reason_codes=(error.code,),
            detail=error.message,
            metrics={},
        )
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        return ProofCarryingFactorVerification(
            decision="invalid-fail-closed",
            valid=False,
            admitted=False,
            certificate_id=certificate_id,
            reason_codes=("unexpected-verification-error",),
            detail=str(error),
            metrics={},
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Verify a certificate as an execution gate."""

    parser = argparse.ArgumentParser(
        prog="python -m prob4d_independent_verifier.proof_carrying",
        description=(
            "Independently verify a Proof4D linear-query factor certificate. "
            "Exit 0 admits, 2 rejects a valid unsupported query, and 3 marks "
            "the certificate invalid."
        ),
    )
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    report = verify_proof_carrying_factor(arguments.certificate)
    print(
        json.dumps(
            report.to_dict(),
            sort_keys=True,
            indent=None if arguments.compact else 2,
            separators=(",", ":") if arguments.compact else None,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if report.admitted:
        return 0
    if report.valid:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXACT_FALLBACK_POLICY",
    "PROOF_CARRYING_FACTOR_CLAIM_SCOPE",
    "PROOF_CARRYING_FACTOR_KIND",
    "PROOF_CARRYING_FACTOR_SCHEMA",
    "PROOF_CARRYING_FACTOR_VERSION",
    "ProofCarryingFactorVerification",
    "VERIFICATION_CLAIM_BOUNDARY",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_VERSION",
    "VERIFIER_IMPLEMENTATION",
    "main",
    "verify_proof_carrying_factor",
]
