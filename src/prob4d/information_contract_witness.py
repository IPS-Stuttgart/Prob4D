"""Source-selected falsification witnesses for probabilistic 4-D contracts.

A witness is selected only from source residuals and reported covariance inside a
caller-registered linear physical-query span.  Held evaluation consumes the
frozen witness without re-optimizing it.  Complete objects, sessions, or
trajectories are represented by ``group_index`` and receive equal weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

SOURCE_SCHEMA: Final = "prob4d.information-contract-witness-source"
SOURCE_VERSION: Final = 1
HELD_SCHEMA: Final = "prob4d.information-contract-witness-held"
HELD_VERSION: Final = 1
WITNESS_SCHEMA: Final = "prob4d.information-contract-falsification-witness"
WITNESS_VERSION: Final = 1
RESULT_SCHEMA: Final = "prob4d.information-contract-witness-result"
RESULT_VERSION: Final = 1

SOURCE_ARRAYS: Final = frozenset(
    {"residual_vectors", "reported_covariance", "group_index", "query_basis"}
)
HELD_ARRAYS: Final = frozenset(
    {"residual_vectors", "reported_covariance", "group_index"}
)
INFORMATION_ORDERS: Final = frozenset(
    {"prospective-sealed-target", "retrospective-open-target"}
)
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CLAIM_BOUNDARY: Final = (
    "The witness is optimal only inside the registered linear query span and for "
    "the equal-source-group second-moment ratio. Held validity requires independent "
    "target groups and a witness frozen before their outcomes are opened. The result "
    "does not prove physical meaning of the query basis, covariance calibration "
    "outside the evaluated groups, causal identification, deployment safety, or "
    "state of the art."
)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"cannot read payload {path}") from error
    return digest.hexdigest()


def _write_bytes(path: Path, value: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _write_json(path: Path, value: object, *, overwrite: bool) -> None:
    _write_bytes(path, _canonical_bytes(value), overwrite=overwrite)


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            relation = "at most" if maximum_inclusive else "less than"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return result


def _hex_sha256(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _relative_payload(root: Path, value: object, *, name: str) -> Path:
    relative = Path(_string(value, name=name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{name} must be relative and must not contain '..'")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes its manifest directory") from error
    if not resolved.is_file():
        raise ValueError(f"{name} does not identify a file")
    return resolved


def _load_npz(
    path: Path,
    *,
    allowed: frozenset[str],
    exact: frozenset[str],
) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = set(archive.files)
            if names != set(exact):
                missing = sorted(set(exact) - names)
                extra = sorted(names - set(allowed))
                raise ValueError(f"payload array mismatch: missing={missing}, extra={extra}")
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("payload array mismatch"):
            raise
        raise ValueError(f"cannot load NPZ payload {path}") from error


def _float_array(value: object, *, name: str, ndim: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return result


def _group_index(value: object, *, case_count: int, group_count: int) -> IntArray:
    array = np.asarray(value)
    if array.shape != (case_count,) or array.dtype.kind not in {"i", "u"}:
        raise ValueError("group_index must be an integer vector with one entry per case")
    result = array.astype(np.int64, copy=False)
    if group_count < 1 or set(result.tolist()) != set(range(group_count)):
        raise ValueError("group_index must use every contiguous label 0..G-1")
    return result


def _covariances(value: object, *, case_count: int, dimension: int) -> FloatArray:
    covariance = _float_array(value, name="reported_covariance", ndim=3)
    if covariance.shape != (case_count, dimension, dimension):
        raise ValueError(
            "reported_covariance must have shape "
            f"({case_count}, {dimension}, {dimension})"
        )
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(-2, -1)), 1.0)
    asymmetry = np.max(
        np.abs(covariance - np.swapaxes(covariance, -1, -2)), axis=(-2, -1)
    )
    if np.any(asymmetry > 1e-12 + 1e-10 * scale):
        raise ValueError("reported_covariance must be symmetric")
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise ValueError("reported_covariance must be positive definite") from error
    return symmetric


def _validate_groups(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    groups = tuple(_string(item, name=f"{name} entry") for item in value)
    if len(groups) != len(set(groups)):
        raise ValueError(f"{name} entries must be unique")
    return groups


def _equal_group_moments(
    residual: FloatArray,
    covariance: FloatArray,
    group_index: IntArray,
    group_count: int,
) -> tuple[FloatArray, FloatArray]:
    dimension = residual.shape[1]
    empirical = np.zeros((dimension, dimension), dtype=np.float64)
    reported = np.zeros_like(empirical)
    for group in range(group_count):
        selected = group_index == group
        if not np.any(selected):
            raise ValueError("empty independent group")
        values = residual[selected]
        empirical += values.T @ values / len(values)
        reported += np.mean(covariance[selected], axis=0)
    empirical /= group_count
    reported /= group_count
    return 0.5 * (empirical + empirical.T), 0.5 * (reported + reported.T)


def _basis(value: object, *, dimension: int) -> FloatArray:
    basis = _float_array(value, name="query_basis", ndim=2)
    if basis.shape[0] != dimension or basis.shape[1] < 1:
        raise ValueError(f"query_basis must have shape ({dimension}, K), K >= 1")
    singular_values = np.linalg.svd(basis, compute_uv=False)
    threshold = np.finfo(np.float64).eps * max(basis.shape) * singular_values[0]
    if int(np.count_nonzero(singular_values > threshold)) != basis.shape[1]:
        raise ValueError("query_basis must have full column rank")
    return basis


def _whitened_generalized_eigenpair(
    empirical: FloatArray,
    reported: FloatArray,
) -> tuple[float, FloatArray, float]:
    try:
        cholesky = np.linalg.cholesky(reported)
    except np.linalg.LinAlgError as error:
        raise ValueError("reported query covariance must be positive definite") from error
    left = np.linalg.solve(cholesky, empirical)
    whitened = np.linalg.solve(cholesky, left.T).T
    whitened = 0.5 * (whitened + whitened.T)
    values, vectors = np.linalg.eigh(whitened)
    index = int(np.argmax(values))
    eigenvalue = float(values[index])
    if not math.isfinite(eigenvalue) or eigenvalue < -1e-10:
        raise ValueError("generalized eigenvalue is invalid")
    coefficients = np.linalg.solve(cholesky.T, vectors[:, index])
    denominator = float(coefficients @ reported @ coefficients)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("selected query has invalid reported variance")
    coefficients /= math.sqrt(denominator)
    residual = empirical @ coefficients - eigenvalue * (reported @ coefficients)
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(empirical @ coefficients), abs(eigenvalue), 1e-15)
    )
    return max(eigenvalue, 0.0), coefficients, relative_residual


def _deterministic_sign(
    query: FloatArray, coefficients: FloatArray
) -> tuple[FloatArray, FloatArray]:
    pivot = int(np.argmax(np.abs(query)))
    if query[pivot] < 0.0:
        return -query, -coefficients
    return query, coefficients


def _source_manifest(path: Path) -> tuple[dict[str, Any], Path, tuple[str, ...]]:
    manifest = _json_object(path)
    allowed = {
        "schema_name",
        "schema_version",
        "source_id",
        "audited_submission_id",
        "aggregation_unit",
        "group_ids",
        "payload",
        "payload_sha256",
        "query_family",
        "coverage_probability",
        "information_order",
        "claim_boundary",
    }
    unknown = set(manifest) - allowed
    if unknown:
        raise ValueError(f"source manifest contains unregistered fields: {sorted(unknown)}")
    if (
        manifest.get("schema_name") != SOURCE_SCHEMA
        or manifest.get("schema_version") != SOURCE_VERSION
    ):
        raise ValueError("unsupported source-witness schema")
    _string(manifest.get("source_id"), name="source_id")
    _string(manifest.get("audited_submission_id"), name="audited_submission_id")
    if manifest.get("aggregation_unit") != "group_index":
        raise ValueError("aggregation_unit must be 'group_index'")
    groups = _validate_groups(manifest.get("group_ids"), name="group_ids")
    family = manifest.get("query_family")
    if not isinstance(family, dict) or set(family) != {
        "query_family_id",
        "semantic_label",
        "units",
        "basis_frozen_before_source_outcomes",
    }:
        raise ValueError("query_family must contain exactly the registered fields")
    _string(family.get("query_family_id"), name="query_family_id")
    _string(family.get("semantic_label"), name="query_family semantic_label")
    _string(family.get("units"), name="query_family units")
    if family.get("basis_frozen_before_source_outcomes") is not True:
        raise ValueError("query basis must be registered before source outcomes")
    _real(
        manifest.get("coverage_probability"),
        name="coverage_probability",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if manifest.get("information_order") != "source-only":
        raise ValueError("source information_order must be 'source-only'")
    _string(manifest.get("claim_boundary"), name="claim_boundary")
    payload = _relative_payload(path.parent, manifest.get("payload"), name="payload")
    expected = _hex_sha256(manifest.get("payload_sha256"), name="payload_sha256")
    actual = _sha256_file(payload)
    if actual != expected:
        raise ValueError(f"source payload SHA-256 mismatch: {actual} != {expected}")
    return manifest, payload, groups


def select_falsification_witness(source_manifest: str | Path) -> dict[str, Any]:
    """Select the worst standardized linear query using source groups only."""

    path = Path(source_manifest)
    manifest, payload_path, groups = _source_manifest(path)
    arrays = _load_npz(payload_path, allowed=SOURCE_ARRAYS, exact=SOURCE_ARRAYS)
    residual = _float_array(arrays["residual_vectors"], name="residual_vectors", ndim=2)
    if residual.shape[0] < 1 or residual.shape[1] < 1:
        raise ValueError("residual_vectors must have nonempty shape (C, D)")
    covariance = _covariances(
        arrays["reported_covariance"],
        case_count=residual.shape[0],
        dimension=residual.shape[1],
    )
    index = _group_index(
        arrays["group_index"], case_count=residual.shape[0], group_count=len(groups)
    )
    basis = _basis(arrays["query_basis"], dimension=residual.shape[1])
    empirical, reported = _equal_group_moments(residual, covariance, index, len(groups))
    reduced_empirical = basis.T @ empirical @ basis
    reduced_reported = basis.T @ reported @ basis
    eigenvalue, coefficients, eigen_residual = _whitened_generalized_eigenpair(
        reduced_empirical, reduced_reported
    )
    query = basis @ coefficients
    query_norm = float(np.linalg.norm(query))
    if not math.isfinite(query_norm) or query_norm <= 0.0:
        raise ValueError("selected query has invalid Euclidean norm")
    query /= query_norm
    coefficients /= query_norm
    query, coefficients = _deterministic_sign(query, coefficients)
    reported_variance = float(query @ reported @ query)
    empirical_energy = float(query @ empirical @ query)
    if not math.isclose(float(np.linalg.norm(query)), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("selected query normalization failed")
    group_ratios: list[float] = []
    for group in range(len(groups)):
        selected = index == group
        error = residual[selected] @ query
        variance = np.einsum("i,nij,j->n", query, covariance[selected], query)
        group_ratios.append(float(np.mean(np.square(error)) / np.mean(variance)))
    record: dict[str, Any] = {
        "schema_name": WITNESS_SCHEMA,
        "schema_version": WITNESS_VERSION,
        "source_id": manifest["source_id"],
        "source_manifest_sha256": _sha256_file(path),
        "source_payload_sha256": manifest["payload_sha256"],
        "audited_submission_id": manifest["audited_submission_id"],
        "query_family": manifest["query_family"],
        "aggregation_unit": "equal independent group",
        "source_case_count": int(residual.shape[0]),
        "source_group_ids": list(groups),
        "source_group_count": len(groups),
        "state_dimension": int(residual.shape[1]),
        "query_basis_dimension": int(basis.shape[1]),
        "query_coefficients": coefficients.tolist(),
        "query_vector": query.tolist(),
        "normalization": "Euclidean norm of the state-space query vector equals one",
        "source_equal_group_empirical_query_energy": empirical_energy,
        "source_equal_group_reported_query_variance": reported_variance,
        "source_max_normalized_error_ratio": empirical_energy / reported_variance,
        "source_generalized_eigenvalue": eigenvalue,
        "generalized_eigen_relative_residual": eigen_residual,
        "source_per_group_normalized_error_ratio": group_ratios,
        "coverage_probability": manifest["coverage_probability"],
        "source_information_order": "source-only",
        "held_outcomes_opened_during_selection": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    record["witness_id"] = _sha256_bytes(_canonical_bytes(record))
    return record


def _witness(path: Path) -> dict[str, Any]:
    value = _json_object(path)
    allowed = {
        "schema_name",
        "schema_version",
        "source_id",
        "source_manifest_sha256",
        "source_payload_sha256",
        "audited_submission_id",
        "query_family",
        "aggregation_unit",
        "source_case_count",
        "source_group_ids",
        "source_group_count",
        "state_dimension",
        "query_basis_dimension",
        "query_coefficients",
        "query_vector",
        "normalization",
        "source_equal_group_empirical_query_energy",
        "source_equal_group_reported_query_variance",
        "source_max_normalized_error_ratio",
        "source_generalized_eigenvalue",
        "generalized_eigen_relative_residual",
        "source_per_group_normalized_error_ratio",
        "coverage_probability",
        "source_information_order",
        "held_outcomes_opened_during_selection",
        "claim_boundary",
        "witness_id",
    }
    if set(value) != allowed:
        raise ValueError("witness fields do not match version 1")
    if value.get("schema_name") != WITNESS_SCHEMA or value.get("schema_version") != WITNESS_VERSION:
        raise ValueError("unsupported falsification witness")
    identifier = _hex_sha256(value.get("witness_id"), name="witness_id")
    unsigned = dict(value)
    del unsigned["witness_id"]
    if _sha256_bytes(_canonical_bytes(unsigned)) != identifier:
        raise ValueError("witness content ID does not verify")
    if value.get("source_information_order") != "source-only":
        raise ValueError("witness is not source-selected")
    if value.get("held_outcomes_opened_during_selection") is not False:
        raise ValueError("witness selection accessed held outcomes")
    return value


def _held_manifest(
    path: Path, witness: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...], list[dict[str, Any]]]:
    manifest = _json_object(path)
    allowed = {
        "schema_name",
        "schema_version",
        "held_id",
        "source_witness_id",
        "aggregation_unit",
        "group_ids",
        "information_order",
        "submissions",
        "claim_boundary",
    }
    unknown = set(manifest) - allowed
    if unknown:
        raise ValueError(f"held manifest contains unregistered fields: {sorted(unknown)}")
    if manifest.get("schema_name") != HELD_SCHEMA or manifest.get("schema_version") != HELD_VERSION:
        raise ValueError("unsupported held-witness schema")
    _string(manifest.get("held_id"), name="held_id")
    if manifest.get("source_witness_id") != witness["witness_id"]:
        raise ValueError("held manifest is not bound to the supplied witness")
    if manifest.get("aggregation_unit") != "group_index":
        raise ValueError("aggregation_unit must be 'group_index'")
    groups = _validate_groups(manifest.get("group_ids"), name="group_ids")
    information_order = manifest.get("information_order")
    if information_order not in INFORMATION_ORDERS:
        raise ValueError("unsupported held information_order")
    _string(manifest.get("claim_boundary"), name="claim_boundary")
    submissions = manifest.get("submissions")
    if not isinstance(submissions, list) or len(submissions) < 2:
        raise ValueError("held manifest requires at least two submissions")
    validated: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for ordinal, raw in enumerate(submissions):
        if not isinstance(raw, dict) or set(raw) != {
            "submission_id",
            "payload",
            "payload_sha256",
        }:
            raise ValueError(f"submissions[{ordinal}] has invalid fields")
        identifier = _string(raw.get("submission_id"), name="submission_id")
        if identifier in identifiers:
            raise ValueError("submission_id values must be unique")
        identifiers.add(identifier)
        payload = _relative_payload(
            path.parent, raw.get("payload"), name=f"{identifier}.payload"
        )
        expected = _hex_sha256(raw.get("payload_sha256"), name=f"{identifier}.payload_sha256")
        actual = _sha256_file(payload)
        if actual != expected:
            raise ValueError(f"held payload SHA-256 mismatch for {identifier}")
        validated.append({**raw, "resolved_payload": payload})
    return manifest, groups, validated


def _submission_metrics(
    residual: FloatArray,
    covariance: FloatArray,
    group_index: IntArray,
    group_count: int,
    query: FloatArray,
    *,
    coverage_probability: float,
) -> dict[str, Any]:
    z = NormalDist().inv_cdf(0.5 + 0.5 * coverage_probability)
    group_coordinate_mse: list[float] = []
    group_query_mse: list[float] = []
    group_nees: list[float] = []
    group_nll: list[float] = []
    group_coverage: list[float] = []
    for group in range(group_count):
        selected = group_index == group
        values = residual[selected]
        matrices = covariance[selected]
        error = values @ query
        variance = np.einsum("i,nij,j->n", query, matrices, query)
        if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
            raise ValueError("held query variance is invalid")
        group_coordinate_mse.append(float(np.mean(np.square(values))))
        group_query_mse.append(float(np.mean(np.square(error))))
        group_nees.append(float(np.mean(np.square(error) / variance)))
        group_nll.append(
            float(np.mean(0.5 * (np.log(2.0 * math.pi * variance) + np.square(error) / variance)))
        )
        group_coverage.append(float(np.mean(np.abs(error) <= z * np.sqrt(variance))))
    nees = float(np.mean(group_nees))
    return {
        "case_count": int(residual.shape[0]),
        "independent_group_count": group_count,
        "equal_group_coordinate_rmse": math.sqrt(float(np.mean(group_coordinate_mse))),
        "equal_group_query_rmse": math.sqrt(float(np.mean(group_query_mse))),
        "equal_group_query_normalized_error_ratio": nees,
        "equal_group_query_absolute_log_calibration_error": abs(math.log(max(nees, 1e-300))),
        "equal_group_query_gaussian_nll": float(np.mean(group_nll)),
        "coverage_probability": coverage_probability,
        "equal_group_query_coverage": float(np.mean(group_coverage)),
        "per_group_query_normalized_error_ratio": group_nees,
        "per_group_query_gaussian_nll": group_nll,
    }


def evaluate_frozen_witness(
    held_manifest: str | Path,
    witness_path: str | Path,
) -> dict[str, Any]:
    """Evaluate a source-frozen witness on held submissions without reselection."""

    witness_file = Path(witness_path)
    witness = _witness(witness_file)
    held_file = Path(held_manifest)
    held, groups, submissions = _held_manifest(held_file, witness)
    query = _float_array(witness["query_vector"], name="query_vector", ndim=1)
    dimension = int(witness["state_dimension"])
    if query.shape != (dimension,):
        raise ValueError("witness query dimension is inconsistent")
    reference_index: IntArray | None = None
    reference_case_count: int | None = None
    metrics: dict[str, dict[str, Any]] = {}
    for item in submissions:
        identifier = str(item["submission_id"])
        arrays = _load_npz(
            Path(item["resolved_payload"]), allowed=HELD_ARRAYS, exact=HELD_ARRAYS
        )
        residual = _float_array(
            arrays["residual_vectors"], name=f"{identifier}.residual_vectors", ndim=2
        )
        if residual.shape[0] < 1 or residual.shape[1] != dimension:
            raise ValueError(f"{identifier} residual dimension does not match witness")
        covariance = _covariances(
            arrays["reported_covariance"],
            case_count=residual.shape[0],
            dimension=dimension,
        )
        index = _group_index(
            arrays["group_index"], case_count=residual.shape[0], group_count=len(groups)
        )
        if reference_index is None:
            reference_index = index.copy()
            reference_case_count = residual.shape[0]
        elif reference_case_count != residual.shape[0] or not np.array_equal(
            index, reference_index
        ):
            raise ValueError("held submissions do not share an exact case/group roster")
        metrics[identifier] = _submission_metrics(
            residual,
            covariance,
            index,
            len(groups),
            query,
            coverage_probability=float(witness["coverage_probability"]),
        )
    point_order = sorted(
        metrics,
        key=lambda key: (metrics[key]["equal_group_coordinate_rmse"], key),
    )
    calibration_order = sorted(
        metrics,
        key=lambda key: (
            metrics[key]["equal_group_query_absolute_log_calibration_error"],
            key,
        ),
    )
    nll_order = sorted(
        metrics,
        key=lambda key: (metrics[key]["equal_group_query_gaussian_nll"], key),
    )
    point_winner = point_order[0]
    calibration_winner = calibration_order[0]
    result = {
        "schema_name": RESULT_SCHEMA,
        "schema_version": RESULT_VERSION,
        "held_id": held["held_id"],
        "held_manifest_sha256": _sha256_file(held_file),
        "witness_id": witness["witness_id"],
        "witness_sha256": _sha256_file(witness_file),
        "audited_submission_id": witness["audited_submission_id"],
        "query_family": witness["query_family"],
        "query_vector": witness["query_vector"],
        "source_max_normalized_error_ratio": witness["source_max_normalized_error_ratio"],
        "source_query_selection_reused_without_target_optimization": True,
        "target_query_reselection": False,
        "information_order": {
            "mode": held["information_order"],
            "claim_class": (
                "prospective-held-confirmation"
                if held["information_order"] == "prospective-sealed-target"
                else "retrospective-diagnostic"
            ),
            "prospective_claim_eligible": held["information_order"]
            == "prospective-sealed-target",
        },
        "group_ids": list(groups),
        "submissions": metrics,
        "rankings": {
            "point_accuracy_order": point_order,
            "selected_query_calibration_order": calibration_order,
            "selected_query_nll_order": nll_order,
            "point_accuracy_winner": point_winner,
            "selected_query_calibration_winner": calibration_winner,
            "point_vs_query_calibration_ranking_reversal": point_winner
            != calibration_winner,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["result_id"] = _sha256_bytes(_canonical_bytes(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="select a source-only witness")
    select.add_argument("source_manifest", type=Path)
    select.add_argument("output", type=Path)
    select.add_argument("--overwrite", action="store_true")
    evaluate = subparsers.add_parser("evaluate", help="evaluate a frozen witness")
    evaluate.add_argument("held_manifest", type=Path)
    evaluate.add_argument("witness", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    smoke = subparsers.add_parser("smoke", help="generate a deterministic rank-reversal control")
    smoke.add_argument("directory", type=Path)
    smoke.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "select":
        _write_json(
            args.output,
            select_falsification_witness(args.source_manifest),
            overwrite=args.overwrite,
        )
        return 0
    if args.command == "evaluate":
        _write_json(
            args.output,
            evaluate_frozen_witness(args.held_manifest, args.witness),
            overwrite=args.overwrite,
        )
        return 0
    from .information_contract_witness_smoke import generate_witness_smoke

    generate_witness_smoke(args.directory, overwrite=args.overwrite)
    return 0


__all__ = [
    "CLAIM_BOUNDARY",
    "HELD_SCHEMA",
    "RESULT_SCHEMA",
    "SOURCE_SCHEMA",
    "WITNESS_SCHEMA",
    "evaluate_frozen_witness",
    "main",
    "select_falsification_witness",
]


if __name__ == "__main__":
    raise SystemExit(main())
