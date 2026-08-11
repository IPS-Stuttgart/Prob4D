"""Certify query-space preservation under covariance approximations.

Prob4D can represent the same observation uncertainty with full joint, sparse,
rank-capped, block-diagonal, or marginal-only treatments. This module compares
those treatments only after projection into a downstream consumer's frozen physical
query. The consumer still owns the query, tolerances, computational selection, and
update admission.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_text
from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._strict_json import (
    load_json_object,
    require_exact_fields,
    require_exact_integer,
    require_exact_string,
    require_finite_json_mapping,
    require_json_number,
    require_mapping,
    require_sha256,
)

FloatArray: TypeAlias = NDArray[np.floating[Any]]

QUERY_COVARIANCE_PRESERVATION_SCHEMA = "prob4d.query-covariance-preservation"
QUERY_COVARIANCE_PRESERVATION_VERSION = 1
QUERY_COVARIANCE_PRESERVATION_CLAIM_BOUNDARY = (
    "This certificate compares supplied covariance treatments in one exact "
    "consumer-defined query space. It does not define the query, select a treatment, "
    "authorize a BayesianPhysTwin update, prove provider competence or calibration, "
    "establish Causal4D benefit, deployment safety, or state of the art."
)

_POLICY_FIELDS = frozenset(
    {
        "relative_rank_tolerance",
        "maximum_relative_trace_distortion",
        "maximum_relative_frobenius_distortion",
        "minimum_directional_variance_ratio",
        "maximum_directional_variance_ratio",
        "maximum_unsupported_trace_fraction",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "representation",
        "covariance",
        "estimated_memory_bytes",
        "estimated_runtime_seconds",
        "metadata",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "candidate_id",
        "representation",
        "relative_trace_distortion",
        "relative_frobenius_distortion",
        "minimum_directional_variance_ratio",
        "mean_directional_variance_ratio",
        "maximum_directional_variance_ratio",
        "maximum_directional_variance_ratio_error",
        "unsupported_trace_fraction",
        "preserved",
        "failure_reasons",
        "estimated_memory_bytes",
        "estimated_runtime_seconds",
    }
)
_CERTIFICATE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "query_definition_id",
        "observation_artifact_id",
        "reference_representation",
        "reference_covariance",
        "query_dimension",
        "reference_effective_rank",
        "policy",
        "candidates",
        "results",
        "preserved_candidate_ids",
        "any_preserved",
        "all_preserved",
        "metadata",
        "claim_boundary",
        "query_covariance_preservation_id",
    }
)


def _readonly(value: object) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(value: object, *, name: str) -> float:
    result = require_json_number(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _optional_nonnegative_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return require_exact_integer(value, name=name, minimum=0)


def _optional_nonnegative_number(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative(value, name=name)


def _validated_covariance(
    value: object,
    *,
    name: str,
    expected_dimension: int | None = None,
    require_positive_trace: bool = False,
) -> FloatArray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if covariance.shape[0] < 1:
        raise ValueError(f"{name} must have positive dimension")
    if expected_dimension is not None and covariance.shape != (
        expected_dimension,
        expected_dimension,
    ):
        raise ValueError(f"{name} has an invalid shape")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (covariance + covariance.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0)) < -(
        1e-12 + 1e-10 * scale
    ):
        raise ValueError(f"{name} must be positive semidefinite")
    if require_positive_trace and float(np.trace(symmetric)) <= 0.0:
        raise ValueError(f"{name} must have positive trace")
    return _readonly(symmetric)


def _relative_rank_tolerance(value: object) -> float:
    result = require_json_number(value, name="relative_rank_tolerance")
    if not 0.0 <= result < 1.0:
        raise ValueError("relative_rank_tolerance must lie in [0, 1)")
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


@dataclass(frozen=True, slots=True)
class QueryCovariancePreservationPolicyV1:
    """Consumer-frozen tolerances for one query-space comparison."""

    relative_rank_tolerance: float
    maximum_relative_trace_distortion: float
    maximum_relative_frobenius_distortion: float
    minimum_directional_variance_ratio: float
    maximum_directional_variance_ratio: float
    maximum_unsupported_trace_fraction: float

    def __post_init__(self) -> None:
        tolerance = _relative_rank_tolerance(self.relative_rank_tolerance)
        trace = _nonnegative(
            self.maximum_relative_trace_distortion,
            name="maximum_relative_trace_distortion",
        )
        frobenius = _nonnegative(
            self.maximum_relative_frobenius_distortion,
            name="maximum_relative_frobenius_distortion",
        )
        minimum_ratio = _nonnegative(
            self.minimum_directional_variance_ratio,
            name="minimum_directional_variance_ratio",
        )
        maximum_ratio = _nonnegative(
            self.maximum_directional_variance_ratio,
            name="maximum_directional_variance_ratio",
        )
        if maximum_ratio < minimum_ratio:
            raise ValueError(
                "maximum_directional_variance_ratio must not be below the minimum"
            )
        unsupported = _probability(
            self.maximum_unsupported_trace_fraction,
            name="maximum_unsupported_trace_fraction",
        )
        object.__setattr__(self, "relative_rank_tolerance", tolerance)
        object.__setattr__(self, "maximum_relative_trace_distortion", trace)
        object.__setattr__(self, "maximum_relative_frobenius_distortion", frobenius)
        object.__setattr__(self, "minimum_directional_variance_ratio", minimum_ratio)
        object.__setattr__(self, "maximum_directional_variance_ratio", maximum_ratio)
        object.__setattr__(self, "maximum_unsupported_trace_fraction", unsupported)

    def to_dict(self) -> dict[str, float]:
        return {
            "relative_rank_tolerance": self.relative_rank_tolerance,
            "maximum_relative_trace_distortion": (
                self.maximum_relative_trace_distortion
            ),
            "maximum_relative_frobenius_distortion": (
                self.maximum_relative_frobenius_distortion
            ),
            "minimum_directional_variance_ratio": (
                self.minimum_directional_variance_ratio
            ),
            "maximum_directional_variance_ratio": (
                self.maximum_directional_variance_ratio
            ),
            "maximum_unsupported_trace_fraction": (
                self.maximum_unsupported_trace_fraction
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> QueryCovariancePreservationPolicyV1:
        mapping = require_mapping(value, name="query covariance preservation policy")
        require_exact_fields(
            mapping,
            _POLICY_FIELDS,
            name="query covariance preservation policy",
        )
        return cls(
            relative_rank_tolerance=mapping["relative_rank_tolerance"],
            maximum_relative_trace_distortion=mapping[
                "maximum_relative_trace_distortion"
            ],
            maximum_relative_frobenius_distortion=mapping[
                "maximum_relative_frobenius_distortion"
            ],
            minimum_directional_variance_ratio=mapping[
                "minimum_directional_variance_ratio"
            ],
            maximum_directional_variance_ratio=mapping[
                "maximum_directional_variance_ratio"
            ],
            maximum_unsupported_trace_fraction=mapping[
                "maximum_unsupported_trace_fraction"
            ],
        )


@dataclass(frozen=True, slots=True)
class QueryCovarianceCandidateV1:
    """One query-space covariance representation and optional cost evidence."""

    candidate_id: str
    representation: str
    covariance: FloatArray
    estimated_memory_bytes: int | None = None
    estimated_runtime_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            require_exact_string(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self,
            "representation",
            require_exact_string(self.representation, name="representation"),
        )
        object.__setattr__(
            self,
            "covariance",
            _validated_covariance(self.covariance, name="candidate covariance"),
        )
        object.__setattr__(
            self,
            "estimated_memory_bytes",
            _optional_nonnegative_integer(
                self.estimated_memory_bytes,
                name="estimated_memory_bytes",
            ),
        )
        object.__setattr__(
            self,
            "estimated_runtime_seconds",
            _optional_nonnegative_number(
                self.estimated_runtime_seconds,
                name="estimated_runtime_seconds",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="candidate metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "representation": self.representation,
            "covariance": self.covariance.tolist(),
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "metadata": plain_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> QueryCovarianceCandidateV1:
        mapping = require_mapping(value, name="query covariance candidate")
        require_exact_fields(
            mapping,
            _CANDIDATE_FIELDS,
            name="query covariance candidate",
        )
        return cls(
            candidate_id=mapping["candidate_id"],
            representation=mapping["representation"],
            covariance=mapping["covariance"],
            estimated_memory_bytes=mapping["estimated_memory_bytes"],
            estimated_runtime_seconds=mapping["estimated_runtime_seconds"],
            metadata=require_finite_json_mapping(
                mapping["metadata"], name="candidate metadata"
            ),
        )


@dataclass(frozen=True, slots=True)
class QueryCovarianceApproximationResultV1:
    """Derived directional and aggregate distortion for one candidate."""

    candidate_id: str
    representation: str
    relative_trace_distortion: float
    relative_frobenius_distortion: float
    minimum_directional_variance_ratio: float
    mean_directional_variance_ratio: float
    maximum_directional_variance_ratio: float
    maximum_directional_variance_ratio_error: float
    unsupported_trace_fraction: float
    preserved: bool
    failure_reasons: tuple[str, ...]
    estimated_memory_bytes: int | None
    estimated_runtime_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "representation": self.representation,
            "relative_trace_distortion": self.relative_trace_distortion,
            "relative_frobenius_distortion": self.relative_frobenius_distortion,
            "minimum_directional_variance_ratio": (
                self.minimum_directional_variance_ratio
            ),
            "mean_directional_variance_ratio": self.mean_directional_variance_ratio,
            "maximum_directional_variance_ratio": (
                self.maximum_directional_variance_ratio
            ),
            "maximum_directional_variance_ratio_error": (
                self.maximum_directional_variance_ratio_error
            ),
            "unsupported_trace_fraction": self.unsupported_trace_fraction,
            "preserved": self.preserved,
            "failure_reasons": list(self.failure_reasons),
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> QueryCovarianceApproximationResultV1:
        mapping = require_mapping(value, name="query covariance approximation result")
        require_exact_fields(
            mapping,
            _RESULT_FIELDS,
            name="query covariance approximation result",
        )
        reasons = mapping["failure_reasons"]
        if not isinstance(reasons, list):
            raise ValueError("failure_reasons must be a JSON array")
        normalized_reasons = tuple(
            require_exact_string(item, name=f"failure_reasons[{index}]")
            for index, item in enumerate(reasons)
        )
        if normalized_reasons != tuple(sorted(set(normalized_reasons))):
            raise ValueError("failure_reasons must be sorted and unique")
        preserved = mapping["preserved"]
        if type(preserved) is not bool:
            raise ValueError("preserved must be a Boolean")
        return cls(
            candidate_id=require_exact_string(
                mapping["candidate_id"], name="candidate_id"
            ),
            representation=require_exact_string(
                mapping["representation"], name="representation"
            ),
            relative_trace_distortion=_nonnegative(
                mapping["relative_trace_distortion"],
                name="relative_trace_distortion",
            ),
            relative_frobenius_distortion=_nonnegative(
                mapping["relative_frobenius_distortion"],
                name="relative_frobenius_distortion",
            ),
            minimum_directional_variance_ratio=_nonnegative(
                mapping["minimum_directional_variance_ratio"],
                name="minimum_directional_variance_ratio",
            ),
            mean_directional_variance_ratio=_nonnegative(
                mapping["mean_directional_variance_ratio"],
                name="mean_directional_variance_ratio",
            ),
            maximum_directional_variance_ratio=_nonnegative(
                mapping["maximum_directional_variance_ratio"],
                name="maximum_directional_variance_ratio",
            ),
            maximum_directional_variance_ratio_error=_nonnegative(
                mapping["maximum_directional_variance_ratio_error"],
                name="maximum_directional_variance_ratio_error",
            ),
            unsupported_trace_fraction=_probability(
                mapping["unsupported_trace_fraction"],
                name="unsupported_trace_fraction",
            ),
            preserved=preserved,
            failure_reasons=normalized_reasons,
            estimated_memory_bytes=_optional_nonnegative_integer(
                mapping["estimated_memory_bytes"],
                name="estimated_memory_bytes",
            ),
            estimated_runtime_seconds=_optional_nonnegative_number(
                mapping["estimated_runtime_seconds"],
                name="estimated_runtime_seconds",
            ),
        )


@dataclass(frozen=True, slots=True)
class QueryCovariancePreservationCertificateV1:
    """Replay-complete query-space covariance approximation certificate."""

    query_definition_id: str
    observation_artifact_id: str
    reference_representation: str
    reference_covariance: FloatArray
    policy: QueryCovariancePreservationPolicyV1
    candidates: tuple[QueryCovarianceCandidateV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    query_dimension: int = field(init=False)
    reference_effective_rank: int = field(init=False)
    results: tuple[QueryCovarianceApproximationResultV1, ...] = field(init=False)
    preserved_candidate_ids: tuple[str, ...] = field(init=False)
    any_preserved: bool = field(init=False)
    all_preserved: bool = field(init=False)
    query_covariance_preservation_id: str = field(init=False)

    def __post_init__(self) -> None:
        query_id = require_sha256(self.query_definition_id, name="query_definition_id")
        observation_id = require_sha256(
            self.observation_artifact_id,
            name="observation_artifact_id",
        )
        reference_representation = require_exact_string(
            self.reference_representation,
            name="reference_representation",
        )
        reference = _validated_covariance(
            self.reference_covariance,
            name="reference_covariance",
            require_positive_trace=True,
        )
        if not isinstance(self.policy, QueryCovariancePreservationPolicyV1):
            raise TypeError("policy must be QueryCovariancePreservationPolicyV1")
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be a nonempty canonical tuple")
        if any(not isinstance(item, QueryCovarianceCandidateV1) for item in self.candidates):
            raise TypeError("candidates must contain QueryCovarianceCandidateV1 values")
        candidates = tuple(sorted(self.candidates, key=lambda item: item.candidate_id))
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ValueError("candidate IDs must be unique")
        dimension = int(reference.shape[0])
        if any(item.covariance.shape != reference.shape for item in candidates):
            raise ValueError("candidate covariance dimensions must match the reference")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="query preservation metadata",
        )
        eigenvalues = np.linalg.eigvalsh(reference)
        maximum = float(np.max(eigenvalues, initial=0.0))
        active_rank = int(
            np.count_nonzero(
                eigenvalues > self.policy.relative_rank_tolerance * maximum
            )
        )
        if active_rank < 1:
            raise ValueError("reference covariance has no numerically supported query rank")
        results = tuple(self._evaluate_candidate(reference, item) for item in candidates)
        preserved_ids = tuple(item.candidate_id for item in results if item.preserved)

        object.__setattr__(self, "query_definition_id", query_id)
        object.__setattr__(self, "observation_artifact_id", observation_id)
        object.__setattr__(self, "reference_representation", reference_representation)
        object.__setattr__(self, "reference_covariance", reference)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "query_dimension", dimension)
        object.__setattr__(self, "reference_effective_rank", active_rank)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "preserved_candidate_ids", preserved_ids)
        object.__setattr__(self, "any_preserved", bool(preserved_ids))
        object.__setattr__(self, "all_preserved", len(preserved_ids) == len(candidates))
        object.__setattr__(
            self,
            "query_covariance_preservation_id",
            _sha256_json(self._content_dict()),
        )

    def _evaluate_candidate(
        self,
        reference: FloatArray,
        candidate: QueryCovarianceCandidateV1,
    ) -> QueryCovarianceApproximationResultV1:
        candidate_covariance = candidate.covariance
        reference_trace = float(np.trace(reference))
        reference_norm = float(np.linalg.norm(reference, ord="fro"))
        trace_distortion = abs(float(np.trace(candidate_covariance)) - reference_trace)
        trace_distortion /= reference_trace
        frobenius_distortion = float(
            np.linalg.norm(candidate_covariance - reference, ord="fro")
        ) / reference_norm

        eigenvalues, eigenvectors = np.linalg.eigh(reference)
        maximum = float(np.max(eigenvalues, initial=0.0))
        active = eigenvalues > self.policy.relative_rank_tolerance * maximum
        basis = eigenvectors[:, active]
        active_values = eigenvalues[active]
        candidate_active = basis.T @ candidate_covariance @ basis
        inverse_root = 1.0 / np.sqrt(active_values)
        whitened = inverse_root[:, None] * candidate_active * inverse_root[None, :]
        whitened = 0.5 * (whitened + whitened.T)
        ratios = np.linalg.eigvalsh(whitened)
        scale = max(float(np.max(np.abs(ratios), initial=0.0)), 1.0)
        if float(np.min(ratios, initial=0.0)) < -(1e-12 + 1e-10 * scale):
            raise ValueError("candidate covariance became indefinite in query whitening")
        ratios = np.maximum(ratios, 0.0)
        minimum_ratio = float(np.min(ratios))
        mean_ratio = float(np.mean(ratios))
        maximum_ratio = float(np.max(ratios))
        maximum_ratio_error = max(abs(minimum_ratio - 1.0), abs(maximum_ratio - 1.0))

        candidate_trace = float(np.trace(candidate_covariance))
        active_trace = float(np.trace(candidate_active))
        unsupported_trace = max(candidate_trace - active_trace, 0.0)
        unsupported_fraction = (
            0.0
            if candidate_trace <= np.finfo(np.float64).eps
            else float(np.clip(unsupported_trace / candidate_trace, 0.0, 1.0))
        )

        reasons: list[str] = []
        if trace_distortion > self.policy.maximum_relative_trace_distortion:
            reasons.append("relative-trace-distortion-exceeded")
        if frobenius_distortion > self.policy.maximum_relative_frobenius_distortion:
            reasons.append("relative-frobenius-distortion-exceeded")
        if minimum_ratio < self.policy.minimum_directional_variance_ratio:
            reasons.append("directional-understatement-exceeded")
        if maximum_ratio > self.policy.maximum_directional_variance_ratio:
            reasons.append("directional-overstatement-exceeded")
        if unsupported_fraction > self.policy.maximum_unsupported_trace_fraction:
            reasons.append("unsupported-query-trace-exceeded")
        return QueryCovarianceApproximationResultV1(
            candidate_id=candidate.candidate_id,
            representation=candidate.representation,
            relative_trace_distortion=trace_distortion,
            relative_frobenius_distortion=frobenius_distortion,
            minimum_directional_variance_ratio=minimum_ratio,
            mean_directional_variance_ratio=mean_ratio,
            maximum_directional_variance_ratio=maximum_ratio,
            maximum_directional_variance_ratio_error=maximum_ratio_error,
            unsupported_trace_fraction=unsupported_fraction,
            preserved=not reasons,
            failure_reasons=tuple(sorted(reasons)),
            estimated_memory_bytes=candidate.estimated_memory_bytes,
            estimated_runtime_seconds=candidate.estimated_runtime_seconds,
        )

    @classmethod
    def from_projections(
        cls,
        *,
        query_definition_id: str,
        observation_artifact_id: str,
        reference_representation: str,
        reference_projection: object,
        candidate_projections: Mapping[str, object],
        policy: QueryCovariancePreservationPolicyV1,
        candidate_representations: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> QueryCovariancePreservationCertificateV1:
        """Build from QueryCovarianceProjectionV1-like objects."""

        reference = getattr(reference_projection, "total_covariance", None)
        representations = {} if candidate_representations is None else candidate_representations
        candidates = tuple(
            QueryCovarianceCandidateV1(
                candidate_id=candidate_id,
                representation=representations.get(candidate_id, candidate_id),
                covariance=getattr(projection, "total_covariance", None),
            )
            for candidate_id, projection in sorted(candidate_projections.items())
        )
        return cls(
            query_definition_id=query_definition_id,
            observation_artifact_id=observation_artifact_id,
            reference_representation=reference_representation,
            reference_covariance=reference,
            policy=policy,
            candidates=candidates,
            metadata={} if metadata is None else metadata,
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": QUERY_COVARIANCE_PRESERVATION_SCHEMA,
            "schema_version": QUERY_COVARIANCE_PRESERVATION_VERSION,
            "query_definition_id": self.query_definition_id,
            "observation_artifact_id": self.observation_artifact_id,
            "reference_representation": self.reference_representation,
            "reference_covariance": self.reference_covariance.tolist(),
            "query_dimension": self.query_dimension,
            "reference_effective_rank": self.reference_effective_rank,
            "policy": self.policy.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "results": [item.to_dict() for item in self.results],
            "preserved_candidate_ids": list(self.preserved_candidate_ids),
            "any_preserved": self.any_preserved,
            "all_preserved": self.all_preserved,
            "metadata": plain_json(self.metadata),
            "claim_boundary": QUERY_COVARIANCE_PRESERVATION_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["query_covariance_preservation_id"] = (
            self.query_covariance_preservation_id
        )
        return result

    @classmethod
    def from_dict(cls, value: object) -> QueryCovariancePreservationCertificateV1:
        mapping = require_mapping(value, name="query covariance preservation certificate")
        require_exact_fields(
            mapping,
            _CERTIFICATE_FIELDS,
            name="query covariance preservation certificate",
        )
        if mapping["schema"] != QUERY_COVARIANCE_PRESERVATION_SCHEMA:
            raise ValueError("query covariance preservation schema changed")
        if mapping["schema_version"] != QUERY_COVARIANCE_PRESERVATION_VERSION:
            raise ValueError("query covariance preservation version changed")
        if mapping["claim_boundary"] != QUERY_COVARIANCE_PRESERVATION_CLAIM_BOUNDARY:
            raise ValueError("query covariance preservation claim boundary changed")
        raw_candidates = mapping["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates must be a JSON array")
        result = cls(
            query_definition_id=mapping["query_definition_id"],
            observation_artifact_id=mapping["observation_artifact_id"],
            reference_representation=mapping["reference_representation"],
            reference_covariance=mapping["reference_covariance"],
            policy=QueryCovariancePreservationPolicyV1.from_dict(mapping["policy"]),
            candidates=tuple(
                QueryCovarianceCandidateV1.from_dict(item) for item in raw_candidates
            ),
            metadata=require_finite_json_mapping(
                mapping["metadata"], name="query preservation metadata"
            ),
        )
        raw_results = mapping["results"]
        if not isinstance(raw_results, list):
            raise ValueError("results must be a JSON array")
        tuple(QueryCovarianceApproximationResultV1.from_dict(item) for item in raw_results)
        if plain_json(result.to_dict()) != plain_json(mapping):
            raise ValueError("query covariance preservation derived fields changed")
        return result

    def summary(self) -> dict[str, object]:
        return {
            "query_covariance_preservation_id": self.query_covariance_preservation_id,
            "query_dimension": self.query_dimension,
            "reference_effective_rank": self.reference_effective_rank,
            "preserved_candidate_ids": list(self.preserved_candidate_ids),
            "any_preserved": self.any_preserved,
            "all_preserved": self.all_preserved,
            "claim_boundary": QUERY_COVARIANCE_PRESERVATION_CLAIM_BOUNDARY,
        }


def write_query_covariance_preservation(
    path: str | Path,
    certificate: QueryCovariancePreservationCertificateV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(certificate, QueryCovariancePreservationCertificateV1):
        raise TypeError("certificate must be QueryCovariancePreservationCertificateV1")
    _atomic_write_json(path, certificate.to_dict(), overwrite=overwrite)


def load_query_covariance_preservation(
    path: str | Path,
) -> QueryCovariancePreservationCertificateV1:
    return QueryCovariancePreservationCertificateV1.from_dict(
        load_json_object(path, name="query covariance preservation certificate")
    )


__all__ = [
    "QUERY_COVARIANCE_PRESERVATION_CLAIM_BOUNDARY",
    "QUERY_COVARIANCE_PRESERVATION_SCHEMA",
    "QUERY_COVARIANCE_PRESERVATION_VERSION",
    "QueryCovarianceApproximationResultV1",
    "QueryCovarianceCandidateV1",
    "QueryCovariancePreservationCertificateV1",
    "QueryCovariancePreservationPolicyV1",
    "load_query_covariance_preservation",
    "write_query_covariance_preservation",
]
