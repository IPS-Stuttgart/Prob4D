"""Experimental probabilistic mixtures over cross-window material identities.

The contracts in this module preserve every endpoint as a window-local
``(window_id, track_id)`` identity.  They do not construct global point IDs or
connected components.  A mandatory null hypothesis represents the unchanged
newest-window reference, so an empty linked candidate set reproduces the exact
fallback behavior.

Prob4D owns the portable source-side identity mixture.  A downstream consumer,
such as BayesianPhysTwin, may marginalize its own likelihood over the retained
hypotheses.  This module does not decide whether a physical-state update is
accepted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._atomic_file import atomic_write_bytes
from ._immutable_json import frozen_finite_json_mapping, plain_json

FloatArray: TypeAlias = NDArray[np.floating[Any]]

MATERIAL_IDENTITY_MIXTURE_SCHEMA = "prob4d.material-identity-mixture"
MATERIAL_IDENTITY_MIXTURE_VERSION = 1
MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA = "prob4d.material-identity-hypothesis"
MATERIAL_IDENTITY_HYPOTHESIS_VERSION = 1
WEIGHT_SEMANTICS: Literal["source-calibrated-log-weight-v1"] = (
    "source-calibrated-log-weight-v1"
)
NULL_HYPOTHESIS_SEMANTICS: Literal[
    "newest-window-local-reference-v1"
] = "newest-window-local-reference-v1"
LIKELIHOOD_MARGINALIZATION_SEMANTICS: Literal[
    "logsumexp-discrete-identity-v1"
] = "logsumexp-discrete-identity-v1"
MOMENT_MATCH_SEMANTICS: Literal[
    "law-of-total-covariance-v1"
] = "law-of-total-covariance-v1"
CLAIM_BOUNDARY = (
    "Source-calibrated cross-window material-identity hypotheses only. Endpoints "
    "remain window-local, the null hypothesis preserves the newest-window "
    "reference exactly, and no physical-state update or Causal4D benefit is "
    "established by this artifact."
)

_MIXTURE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "mixture_id",
        "target_endpoint",
        "window_order",
        "causal_frame_stop",
        "association_rule_id",
        "calibration_id",
        "tracklet_producer_revision",
        "association_revision",
        "weight_semantics",
        "null_hypothesis_semantics",
        "candidates",
        "metadata",
        "claim_boundary",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "kind",
        "source_endpoint",
        "association_result_id",
        "source_score",
        "calibrated_log_weight",
        "metadata",
    }
)
_ENDPOINT_FIELDS = frozenset({"window_id", "track_id"})


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not permitted: {value}")


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read material-identity mixture {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("material-identity mixture root must be a JSON object")
    return payload


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed: missing={missing}, extra={extra}")


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return str(value)


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int and not isinstance(value, np.integer):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in {int, float} and not isinstance(
        value, (np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(cast(Any, value))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _sha256(value: object, *, name: str) -> str:
    digest = _string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _revision(value: object, *, name: str) -> str:
    revision = _string(value, name=name)
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must be an exact lowercase Git revision")
    return revision


def _readonly(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _logsumexp(values: FloatArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("logsumexp values must be a non-empty vector")
    if np.any(np.isnan(array)) or np.any(np.isposinf(array)):
        raise ValueError("logsumexp values may not contain NaN or positive infinity")
    maximum = float(np.max(array))
    if np.isneginf(maximum):
        return float("-inf")
    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


@dataclass(frozen=True, order=True)
class LocalTrackEndpoint:
    """A window-local track identity that is never rewritten globally."""

    window_id: str
    track_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_id", _string(self.window_id, name="window_id"))
        object.__setattr__(
            self,
            "track_id",
            _integer(self.track_id, name="track_id", minimum=0),
        )

    def to_dict(self) -> dict[str, object]:
        return {"window_id": self.window_id, "track_id": self.track_id}

    @classmethod
    def from_mapping(cls, value: object, *, name: str) -> LocalTrackEndpoint:
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        _require_exact_fields(value, expected=_ENDPOINT_FIELDS, name=name)
        return cls(window_id=value["window_id"], track_id=value["track_id"])


@dataclass(frozen=True)
class MaterialIdentityCandidateV1:
    """One null or linked source hypothesis for a target-local track."""

    source_endpoint: LocalTrackEndpoint | None
    association_result_id: str | None
    source_score: float | None
    calibrated_log_weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        log_weight = _finite_real(
            self.calibrated_log_weight,
            name="calibrated_log_weight",
        )
        if self.source_endpoint is None:
            if self.association_result_id is not None or self.source_score is not None:
                raise ValueError(
                    "the null hypothesis must not carry source association evidence"
                )
        else:
            if not isinstance(self.source_endpoint, LocalTrackEndpoint):
                raise ValueError("source_endpoint must be LocalTrackEndpoint or None")
            object.__setattr__(
                self,
                "association_result_id",
                _sha256(
                    self.association_result_id,
                    name="association_result_id",
                ),
            )
            object.__setattr__(
                self,
                "source_score",
                _finite_real(
                    self.source_score,
                    name="source_score",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(self, "calibrated_log_weight", log_weight)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity candidate metadata",
            ),
        )

    @property
    def kind(self) -> Literal["null", "linked"]:
        return "null" if self.source_endpoint is None else "linked"

    def ordering_key(self) -> tuple[object, ...]:
        if self.source_endpoint is None:
            return (0, "", -1, "")
        return (
            1,
            self.source_endpoint.window_id,
            self.source_endpoint.track_id,
            self.association_result_id,
        )

    def identity_record(self, *, target_endpoint: LocalTrackEndpoint) -> dict[str, object]:
        return {
            "schema": MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA,
            "schema_version": MATERIAL_IDENTITY_HYPOTHESIS_VERSION,
            "target_endpoint": target_endpoint.to_dict(),
            "kind": self.kind,
            "source_endpoint": (
                None if self.source_endpoint is None else self.source_endpoint.to_dict()
            ),
            "association_result_id": self.association_result_id,
        }

    def candidate_id(self, *, target_endpoint: LocalTrackEndpoint) -> str:
        return _sha256_json(self.identity_record(target_endpoint=target_endpoint))

    def to_record(self, *, target_endpoint: LocalTrackEndpoint) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id(target_endpoint=target_endpoint),
            "kind": self.kind,
            "source_endpoint": (
                None if self.source_endpoint is None else self.source_endpoint.to_dict()
            ),
            "association_result_id": self.association_result_id,
            "source_score": self.source_score,
            "calibrated_log_weight": self.calibrated_log_weight,
            "metadata": plain_json(self.metadata),
        }


@dataclass(frozen=True)
class MaterialIdentityMixtureV1:
    """A calibrated discrete identity mixture for one target-local track."""

    target_endpoint: LocalTrackEndpoint
    window_order: tuple[str, ...]
    causal_frame_stop: int
    association_rule_id: str
    calibration_id: str
    tracklet_producer_revision: str
    association_revision: str
    candidates: tuple[MaterialIdentityCandidateV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    weight_semantics: Literal[
        "source-calibrated-log-weight-v1"
    ] = WEIGHT_SEMANTICS
    null_hypothesis_semantics: Literal[
        "newest-window-local-reference-v1"
    ] = NULL_HYPOTHESIS_SEMANTICS
    mixture_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_endpoint, LocalTrackEndpoint):
            raise ValueError("target_endpoint must be LocalTrackEndpoint")
        if type(self.window_order) is not tuple or not self.window_order:
            raise ValueError("window_order must be a non-empty tuple")
        window_order = tuple(
            _string(window_id, name=f"window_order[{index}]")
            for index, window_id in enumerate(self.window_order)
        )
        if len(set(window_order)) != len(window_order):
            raise ValueError("window_order must contain unique window IDs")
        if window_order[-1] != self.target_endpoint.window_id:
            raise ValueError("target_endpoint window must be last in window_order")
        source_windows = frozenset(window_order[:-1])
        causal_frame_stop = _integer(
            self.causal_frame_stop,
            name="causal_frame_stop",
            minimum=1,
        )
        if type(self.candidates) is not tuple or not self.candidates:
            raise ValueError("candidates must be a non-empty tuple")
        if any(
            not isinstance(candidate, MaterialIdentityCandidateV1)
            for candidate in self.candidates
        ):
            raise ValueError("candidates must contain MaterialIdentityCandidateV1 values")
        candidates = tuple(
            sorted(self.candidates, key=lambda candidate: candidate.ordering_key())
        )
        null_count = sum(candidate.source_endpoint is None for candidate in candidates)
        if null_count != 1:
            raise ValueError("exactly one null identity hypothesis is required")
        linked_endpoints = [
            candidate.source_endpoint
            for candidate in candidates
            if candidate.source_endpoint is not None
        ]
        if len(set(linked_endpoints)) != len(linked_endpoints):
            raise ValueError("linked source endpoints must be unique")
        if any(endpoint.window_id not in source_windows for endpoint in linked_endpoints):
            raise ValueError(
                "linked source endpoint windows must precede the target in window_order"
            )
        if self.weight_semantics != WEIGHT_SEMANTICS:
            raise ValueError("unsupported weight_semantics")
        if self.null_hypothesis_semantics != NULL_HYPOTHESIS_SEMANTICS:
            raise ValueError("unsupported null_hypothesis_semantics")

        object.__setattr__(self, "window_order", window_order)
        object.__setattr__(self, "causal_frame_stop", causal_frame_stop)
        object.__setattr__(
            self,
            "association_rule_id",
            _sha256(self.association_rule_id, name="association_rule_id"),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _sha256(self.calibration_id, name="calibration_id"),
        )
        object.__setattr__(
            self,
            "tracklet_producer_revision",
            _revision(
                self.tracklet_producer_revision,
                name="tracklet_producer_revision",
            ),
        )
        object.__setattr__(
            self,
            "association_revision",
            _revision(self.association_revision, name="association_revision"),
        )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="material-identity mixture metadata",
            ),
        )

        expected = _sha256_json(self.identity_record())
        supplied = self.mixture_id
        if supplied is not None and _sha256(supplied, name="mixture_id") != expected:
            raise ValueError("material-identity mixture ID mismatch")
        object.__setattr__(self, "mixture_id", expected)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            candidate.candidate_id(target_endpoint=self.target_endpoint)
            for candidate in self.candidates
        )

    @property
    def normalized_log_weights(self) -> FloatArray:
        log_weights = np.asarray(
            [candidate.calibrated_log_weight for candidate in self.candidates],
            dtype=np.float64,
        )
        normalizer = _logsumexp(log_weights)
        return _readonly(log_weights - normalizer, dtype=np.float64)

    @property
    def probabilities(self) -> FloatArray:
        return _readonly(np.exp(self.normalized_log_weights), dtype=np.float64)

    @property
    def null_probability(self) -> float:
        return float(self.probabilities[0])

    @property
    def identity_entropy_nats(self) -> float:
        probabilities = self.probabilities
        active = probabilities > 0.0
        return float(-np.sum(probabilities[active] * np.log(probabilities[active])))

    @property
    def effective_hypothesis_count(self) -> float:
        return float(np.exp(self.identity_entropy_nats))

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": MATERIAL_IDENTITY_MIXTURE_SCHEMA,
            "schema_version": MATERIAL_IDENTITY_MIXTURE_VERSION,
            "target_endpoint": self.target_endpoint.to_dict(),
            "window_order": list(self.window_order),
            "causal_frame_stop": self.causal_frame_stop,
            "association_rule_id": self.association_rule_id,
            "calibration_id": self.calibration_id,
            "tracklet_producer_revision": self.tracklet_producer_revision,
            "association_revision": self.association_revision,
            "weight_semantics": self.weight_semantics,
            "null_hypothesis_semantics": self.null_hypothesis_semantics,
            "candidates": [
                candidate.to_record(target_endpoint=self.target_endpoint)
                for candidate in self.candidates
            ],
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "mixture_id": self.mixture_id}


@dataclass(frozen=True)
class MarginalizedIdentityLikelihood:
    """Result of exact discrete identity likelihood marginalization."""

    candidate_ids: tuple[str, ...]
    log_marginal_likelihood: float
    posterior_probabilities: FloatArray
    identity_entropy_nats: float
    effective_hypothesis_count: float
    likelihood_power: float
    semantics: Literal[
        "logsumexp-discrete-identity-v1"
    ] = LIKELIHOOD_MARGINALIZATION_SEMANTICS

    def __post_init__(self) -> None:
        if type(self.candidate_ids) is not tuple or not self.candidate_ids:
            raise ValueError("candidate_ids must be a non-empty tuple")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if any(type(value) is not str or not value for value in self.candidate_ids):
            raise ValueError("candidate_ids must contain non-empty strings")
        probabilities = np.asarray(self.posterior_probabilities, dtype=np.float64)
        if probabilities.shape != (len(self.candidate_ids),):
            raise ValueError("posterior_probabilities do not match candidate_ids")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("posterior_probabilities must be finite and non-negative")
        if not np.isclose(
            float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("posterior_probabilities must sum to one")
        log_marginal = _finite_real(
            self.log_marginal_likelihood,
            name="log_marginal_likelihood",
        )
        entropy = _finite_real(
            self.identity_entropy_nats,
            name="identity_entropy_nats",
            minimum=0.0,
        )
        effective = _finite_real(
            self.effective_hypothesis_count,
            name="effective_hypothesis_count",
            minimum=1.0,
        )
        power = _finite_real(
            self.likelihood_power,
            name="likelihood_power",
            minimum=0.0,
        )
        if self.semantics != LIKELIHOOD_MARGINALIZATION_SEMANTICS:
            raise ValueError("unsupported likelihood marginalization semantics")
        object.__setattr__(
            self,
            "posterior_probabilities",
            _readonly(probabilities, dtype=np.float64),
        )
        object.__setattr__(self, "log_marginal_likelihood", log_marginal)
        object.__setattr__(self, "identity_entropy_nats", entropy)
        object.__setattr__(self, "effective_hypothesis_count", effective)
        object.__setattr__(self, "likelihood_power", power)


@dataclass(frozen=True)
class GaussianIdentityMomentMatch:
    """Moment-matched Gaussian under a discrete material-identity mixture."""

    candidate_ids: tuple[str, ...]
    probabilities: FloatArray
    mean: FloatArray
    covariance: FloatArray
    within_hypothesis_covariance: FloatArray
    between_hypothesis_covariance: FloatArray
    identity_entropy_nats: float
    effective_hypothesis_count: float
    semantics: Literal[
        "law-of-total-covariance-v1"
    ] = MOMENT_MATCH_SEMANTICS

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        mean = np.asarray(self.mean, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        within = np.asarray(self.within_hypothesis_covariance, dtype=np.float64)
        between = np.asarray(self.between_hypothesis_covariance, dtype=np.float64)
        if type(self.candidate_ids) is not tuple or not self.candidate_ids:
            raise ValueError("candidate_ids must be a non-empty tuple")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must be unique")
        if any(type(value) is not str or not value for value in self.candidate_ids):
            raise ValueError("candidate_ids must contain non-empty strings")
        if probabilities.shape != (len(self.candidate_ids),):
            raise ValueError("probabilities do not match candidate_ids")
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("mean must be a non-empty vector")
        expected = (mean.size, mean.size)
        if any(value.shape != expected for value in (covariance, within, between)):
            raise ValueError("moment covariances must be square and match the mean")
        for name, value in (
            ("probabilities", probabilities),
            ("mean", mean),
            ("covariance", covariance),
            ("within_hypothesis_covariance", within),
            ("between_hypothesis_covariance", between),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
        if np.any(probabilities < 0.0) or not np.isclose(
            float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("probabilities must be non-negative and sum to one")
        if not np.allclose(
            covariance, within + between, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("total covariance must equal within plus between covariance")
        for name, value in (
            ("covariance", covariance),
            ("within_hypothesis_covariance", within),
            ("between_hypothesis_covariance", between),
        ):
            if not np.allclose(value, value.T, atol=1e-12, rtol=1e-12):
                raise ValueError(f"{name} must be symmetric")
            scale = max(1.0, float(np.linalg.norm(value, ord=2)))
            if float(np.min(np.linalg.eigvalsh(value))) < -1e-10 * scale:
                raise ValueError(f"{name} must be positive semidefinite")
        entropy = _finite_real(
            self.identity_entropy_nats,
            name="identity_entropy_nats",
            minimum=0.0,
        )
        effective = _finite_real(
            self.effective_hypothesis_count,
            name="effective_hypothesis_count",
            minimum=1.0,
        )
        if self.semantics != MOMENT_MATCH_SEMANTICS:
            raise ValueError("unsupported moment-match semantics")
        object.__setattr__(self, "identity_entropy_nats", entropy)
        object.__setattr__(self, "effective_hypothesis_count", effective)
        object.__setattr__(
            self,
            "probabilities",
            _readonly(probabilities, dtype=np.float64),
        )
        object.__setattr__(self, "mean", _readonly(mean, dtype=np.float64))
        object.__setattr__(
            self,
            "covariance",
            _readonly(covariance, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "within_hypothesis_covariance",
            _readonly(within, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "between_hypothesis_covariance",
            _readonly(between, dtype=np.float64),
        )


def marginalize_identity_log_likelihoods(
    mixture: MaterialIdentityMixtureV1,
    candidate_ids: tuple[str, ...],
    log_likelihoods: FloatArray,
    *,
    likelihood_power: float = 1.0,
) -> MarginalizedIdentityLikelihood:
    """Marginalize a downstream log likelihood over local identity hypotheses.

    ``candidate_ids`` must exactly match the mixture order.  This prevents a
    consumer from silently assigning a likelihood to the wrong local endpoint.
    A power of zero returns the calibrated source prior without consulting the
    supplied likelihood values.
    """

    if not isinstance(mixture, MaterialIdentityMixtureV1):
        raise ValueError("mixture must be MaterialIdentityMixtureV1")
    if type(candidate_ids) is not tuple or candidate_ids != mixture.candidate_ids:
        raise ValueError("candidate_ids must exactly match mixture.candidate_ids")
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if values.shape != (len(candidate_ids),):
        raise ValueError("log_likelihoods must match the candidate count")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_likelihoods may not contain NaN or positive infinity")
    power = _finite_real(
        likelihood_power,
        name="likelihood_power",
        minimum=0.0,
    )
    prior_log_weights = np.asarray(
        mixture.normalized_log_weights,
        dtype=np.float64,
    )
    if power == 0.0:
        log_terms = prior_log_weights
    else:
        log_terms = prior_log_weights + power * values
    log_marginal = _logsumexp(log_terms)
    if np.isneginf(log_marginal):
        raise ValueError("every identity hypothesis has impossible likelihood")
    probabilities = np.exp(log_terms - log_marginal)
    active = probabilities > 0.0
    entropy = float(-np.sum(probabilities[active] * np.log(probabilities[active])))
    return MarginalizedIdentityLikelihood(
        candidate_ids=candidate_ids,
        log_marginal_likelihood=log_marginal,
        posterior_probabilities=probabilities,
        identity_entropy_nats=entropy,
        effective_hypothesis_count=float(np.exp(entropy)),
        likelihood_power=power,
    )


def moment_match_gaussian_identity_hypotheses(
    mixture: MaterialIdentityMixtureV1,
    candidate_ids: tuple[str, ...],
    means: FloatArray,
    covariances: FloatArray,
    *,
    probabilities: FloatArray | None = None,
) -> GaussianIdentityMomentMatch:
    """Apply the law of total covariance over local identity hypotheses."""

    if not isinstance(mixture, MaterialIdentityMixtureV1):
        raise ValueError("mixture must be MaterialIdentityMixtureV1")
    if type(candidate_ids) is not tuple or candidate_ids != mixture.candidate_ids:
        raise ValueError("candidate_ids must exactly match mixture.candidate_ids")
    mean_array = np.asarray(means, dtype=np.float64)
    covariance_array = np.asarray(covariances, dtype=np.float64)
    count = len(candidate_ids)
    if (
        mean_array.ndim != 2
        or mean_array.shape[0] != count
        or mean_array.shape[1] < 1
    ):
        raise ValueError("means must have shape (candidate_count, dimension)")
    dimension = mean_array.shape[1]
    if covariance_array.shape != (count, dimension, dimension):
        raise ValueError(
            "covariances must have shape (candidate_count, dimension, dimension)"
        )
    if not np.all(np.isfinite(mean_array)) or not np.all(
        np.isfinite(covariance_array)
    ):
        raise ValueError("means and covariances must be finite")
    for index, covariance in enumerate(covariance_array):
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError(f"covariances[{index}] must be symmetric")
        scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10 * scale:
            raise ValueError(f"covariances[{index}] must be positive semidefinite")

    if probabilities is None:
        weights = np.asarray(mixture.probabilities, dtype=np.float64)
    else:
        weights = np.asarray(probabilities, dtype=np.float64)
        if weights.shape != (count,):
            raise ValueError("probabilities must match the candidate count")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        total = float(np.sum(weights))
        if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("probabilities must sum to one")

    marginal_mean = np.sum(weights[:, None] * mean_array, axis=0)
    within = np.sum(weights[:, None, None] * covariance_array, axis=0)
    centered = mean_array - marginal_mean
    between = np.einsum("i,ij,ik->jk", weights, centered, centered)
    within = 0.5 * (within + within.T)
    between = 0.5 * (between + between.T)
    covariance = within + between
    active = weights > 0.0
    entropy = float(-np.sum(weights[active] * np.log(weights[active])))
    return GaussianIdentityMomentMatch(
        candidate_ids=candidate_ids,
        probabilities=weights,
        mean=marginal_mean,
        covariance=covariance,
        within_hypothesis_covariance=within,
        between_hypothesis_covariance=between,
        identity_entropy_nats=entropy,
        effective_hypothesis_count=float(np.exp(entropy)),
    )


def write_material_identity_mixture(
    path: str | Path,
    mixture: MaterialIdentityMixtureV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write one portable material-identity mixture."""

    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a Boolean")
    payload = _canonical_json(mixture.to_record()) + b"\n"
    atomic_write_bytes(path, payload, overwrite=overwrite)


def load_material_identity_mixture(path: str | Path) -> MaterialIdentityMixtureV1:
    """Load and independently validate one portable identity mixture."""

    payload = _load_strict_json(Path(path))
    _require_exact_fields(payload, expected=_MIXTURE_FIELDS, name="mixture")
    if payload["schema"] != MATERIAL_IDENTITY_MIXTURE_SCHEMA:
        raise ValueError("unsupported material-identity mixture schema")
    if payload["schema_version"] != MATERIAL_IDENTITY_MIXTURE_VERSION:
        raise ValueError("unsupported material-identity mixture schema version")
    if payload["weight_semantics"] != WEIGHT_SEMANTICS:
        raise ValueError("unsupported material-identity weight semantics")
    if payload["null_hypothesis_semantics"] != NULL_HYPOTHESIS_SEMANTICS:
        raise ValueError("unsupported null-hypothesis semantics")
    if payload["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValueError("material-identity mixture claim boundary changed")
    target = LocalTrackEndpoint.from_mapping(
        payload["target_endpoint"],
        name="target_endpoint",
    )
    raw_candidates = payload["candidates"]
    if type(raw_candidates) is not list or not raw_candidates:
        raise ValueError("candidates must be a non-empty JSON array")
    candidates: list[MaterialIdentityCandidateV1] = []
    supplied_ids: list[str] = []
    for index, raw_candidate in enumerate(raw_candidates):
        name = f"candidates[{index}]"
        if not isinstance(raw_candidate, Mapping):
            raise ValueError(f"{name} must be a JSON object")
        _require_exact_fields(raw_candidate, expected=_CANDIDATE_FIELDS, name=name)
        kind = raw_candidate["kind"]
        if kind not in {"null", "linked"}:
            raise ValueError(f"{name}.kind is unsupported")
        source_raw = raw_candidate["source_endpoint"]
        source = (
            None
            if source_raw is None
            else LocalTrackEndpoint.from_mapping(
                source_raw,
                name=f"{name}.source_endpoint",
            )
        )
        if (kind == "null") != (source is None):
            raise ValueError(f"{name}.kind does not match source_endpoint")
        candidates.append(
            MaterialIdentityCandidateV1(
                source_endpoint=source,
                association_result_id=raw_candidate["association_result_id"],
                source_score=raw_candidate["source_score"],
                calibrated_log_weight=raw_candidate["calibrated_log_weight"],
                metadata=(
                    raw_candidate["metadata"]
                    if isinstance(raw_candidate["metadata"], Mapping)
                    else _raise_candidate_metadata(name)
                ),
            )
        )
        supplied_ids.append(
            _sha256(raw_candidate["candidate_id"], name=f"{name}.candidate_id")
        )
    raw_window_order = payload["window_order"]
    if type(raw_window_order) is not list or not raw_window_order:
        raise ValueError("window_order must be a non-empty JSON array")
    window_order = tuple(
        _string(window_id, name=f"window_order[{index}]")
        for index, window_id in enumerate(raw_window_order)
    )
    mixture = MaterialIdentityMixtureV1(
        target_endpoint=target,
        window_order=window_order,
        causal_frame_stop=payload["causal_frame_stop"],
        association_rule_id=payload["association_rule_id"],
        calibration_id=payload["calibration_id"],
        tracklet_producer_revision=payload["tracklet_producer_revision"],
        association_revision=payload["association_revision"],
        candidates=tuple(candidates),
        metadata=(
            payload["metadata"]
            if isinstance(payload["metadata"], Mapping)
            else _raise_mixture_metadata()
        ),
        weight_semantics=payload["weight_semantics"],
        null_hypothesis_semantics=payload["null_hypothesis_semantics"],
        mixture_id=payload["mixture_id"],
    )
    if tuple(supplied_ids) != mixture.candidate_ids:
        raise ValueError("material-identity candidate ID mismatch")
    return mixture


def _raise_candidate_metadata(name: str) -> Mapping[str, Any]:
    raise ValueError(f"{name}.metadata must be a JSON object")


def _raise_mixture_metadata() -> Mapping[str, Any]:
    raise ValueError("metadata must be a JSON object")


__all__ = [
    "CLAIM_BOUNDARY",
    "GaussianIdentityMomentMatch",
    "LIKELIHOOD_MARGINALIZATION_SEMANTICS",
    "LocalTrackEndpoint",
    "MATERIAL_IDENTITY_HYPOTHESIS_SCHEMA",
    "MATERIAL_IDENTITY_HYPOTHESIS_VERSION",
    "MATERIAL_IDENTITY_MIXTURE_SCHEMA",
    "MATERIAL_IDENTITY_MIXTURE_VERSION",
    "MOMENT_MATCH_SEMANTICS",
    "MarginalizedIdentityLikelihood",
    "MaterialIdentityCandidateV1",
    "MaterialIdentityMixtureV1",
    "NULL_HYPOTHESIS_SEMANTICS",
    "WEIGHT_SEMANTICS",
    "load_material_identity_mixture",
    "marginalize_identity_log_likelihoods",
    "moment_match_gaussian_identity_hypotheses",
    "write_material_identity_mixture",
]
