"""Prior-anchored Gaussian query messages and dependence-safe fusion.

A high-dimensional correlated update may be reduced, after exact query
conditioning, to one Gaussian natural-parameter increment in query coordinates.
For anchor prior ``N(m0, P0)`` and posterior ``N(m1, P1)`` the message is

``Lambda = P1^{-1} - P0^{-1}``
``eta = P1^{-1} m1 - P0^{-1} m0``.

Applying the anchor prior once plus ``(Lambda, eta)`` reproduces the query
posterior.  The message is query-bound and prior-bound; it is not a reusable
observation factor for a different prior or query.

Messages with unknown cross-correlation may be combined by nonnegative
covariance-intersection weights whose sum is at most one.  The unused weight is
assigned to the anchor prior, so the shared prior is counted exactly once.
This is an algebraic dependence guard, not a calibration or safety guarantee.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .query_posterior import GaussianQueryPosterior

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identifier_tuple(value: Iterable[str], *, name: str) -> tuple[str, ...]:
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of strings") from error
    retained = tuple(sorted({_identifier(item, name=f"{name} item") for item in values}))
    if not retained:
        raise ValueError(f"{name} must contain at least one identifier")
    return retained


def _readonly_vector(value: object, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    copied = result.copy()
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _readonly_symmetric(
    value: object,
    *,
    name: str,
    positive_definite: bool,
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (result + result.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(result, symmetric, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if positive_definite:
        try:
            np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"{name} must be positive definite; reduce deterministic or "
                "duplicated query coordinates first"
            ) from error
    elif float(eigenvalues[0]) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    copied = symmetric.copy()
    copied.setflags(write=False)
    return cast(FloatArray, copied)


def _positive_definite(value: object, *, name: str) -> FloatArray:
    return _readonly_symmetric(value, name=name, positive_definite=True)


def _positive_semidefinite(value: object, *, name: str) -> FloatArray:
    return _readonly_symmetric(value, name=name, positive_definite=False)


def _precision(covariance: FloatArray) -> FloatArray:
    identity = np.eye(covariance.shape[0], dtype=np.float64)
    result = np.linalg.solve(covariance, identity)
    return cast(FloatArray, 0.5 * (result + result.T))


def _fraction(value: object, *, name: str, include_one: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    upper_valid = result <= 1.0 if include_one else result < 1.0
    if not math.isfinite(result) or result < 0.0 or not upper_valid:
        interval = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _array_digest(hasher: Any, value: FloatArray) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    shape = ",".join(str(item) for item in array.shape).encode("ascii")
    hasher.update(len(shape).to_bytes(4, "big"))
    hasher.update(shape)
    payload = array.tobytes(order="C")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _text_digest(hasher: Any, value: str) -> None:
    payload = value.encode("utf-8")
    hasher.update(len(payload).to_bytes(4, "big"))
    hasher.update(payload)


@dataclass(frozen=True, slots=True)
class GaussianQueryBelief:
    """One complete Gaussian belief for a registered query."""

    query_id: str
    prior_id: str
    mean: FloatArray
    covariance: FloatArray
    source_message_id: str

    def __post_init__(self) -> None:
        query_id = _identifier(self.query_id, name="query_id")
        prior_id = _identifier(self.prior_id, name="prior_id")
        source_message_id = _identifier(
            self.source_message_id,
            name="source_message_id",
        )
        mean = _readonly_vector(self.mean, name="mean")
        covariance = _positive_definite(self.covariance, name="covariance")
        if covariance.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("covariance shape must match mean")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "prior_id", prior_id)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source_message_id", source_message_id)

    @property
    def query_dimension(self) -> int:
        return int(self.mean.shape[0])


@dataclass(frozen=True, slots=True)
class GaussianQueryMessage:
    """Prior-relative Gaussian information retained in query coordinates.

    The message may be applied only to the byte-identical anchor prior.  A
    covariance-intersection message records the positive-weight component
    messages and the remaining anchor-prior weight.
    """

    query_id: str
    prior_id: str
    evidence_ids: tuple[str, ...]
    anchor_prior_mean: FloatArray
    anchor_prior_covariance: FloatArray
    information_increment: FloatArray
    natural_parameter_increment: FloatArray
    construction_id: str = "posterior-ratio-v1"
    component_message_ids: tuple[str, ...] = ()
    component_weights: tuple[float, ...] = ()
    prior_weight: float = 0.0
    message_id: str = field(init=False)

    def __post_init__(self) -> None:
        query_id = _identifier(self.query_id, name="query_id")
        prior_id = _identifier(self.prior_id, name="prior_id")
        construction_id = _identifier(self.construction_id, name="construction_id")
        evidence_ids = _identifier_tuple(self.evidence_ids, name="evidence_ids")
        prior_mean = _readonly_vector(
            self.anchor_prior_mean,
            name="anchor_prior_mean",
        )
        prior_covariance = _positive_definite(
            self.anchor_prior_covariance,
            name="anchor_prior_covariance",
        )
        information = _positive_semidefinite(
            self.information_increment,
            name="information_increment",
        )
        natural = _readonly_vector(
            self.natural_parameter_increment,
            name="natural_parameter_increment",
        )
        dimension = prior_mean.shape[0]
        expected_matrix_shape = (dimension, dimension)
        if prior_covariance.shape != expected_matrix_shape:
            raise ValueError("anchor_prior_covariance shape must match anchor_prior_mean")
        if information.shape != expected_matrix_shape:
            raise ValueError("information_increment shape must match anchor prior")
        if natural.shape != (dimension,):
            raise ValueError("natural_parameter_increment shape must match anchor prior")

        component_ids = tuple(
            _identifier(item, name="component_message_ids item")
            for item in self.component_message_ids
        )
        component_weights = tuple(
            _fraction(item, name="component_weights item")
            for item in self.component_weights
        )
        prior_weight = _fraction(self.prior_weight, name="prior_weight")
        if len(component_ids) != len(component_weights):
            raise ValueError("component_message_ids and component_weights must align")
        if component_ids:
            if any(weight <= 0.0 for weight in component_weights):
                raise ValueError("retained component_weights must be positive")
            if not math.isclose(
                sum(component_weights) + prior_weight,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "component_weights plus prior_weight must sum to one"
                )
        elif component_weights or prior_weight != 0.0:
            raise ValueError(
                "uncomposed messages must not declare component or prior weights"
            )

        hasher = hashlib.sha256()
        _text_digest(hasher, "prob4d.gaussian-query-message.v1")
        for item in (query_id, prior_id, construction_id):
            _text_digest(hasher, item)
        for item in evidence_ids:
            _text_digest(hasher, item)
        for item in component_ids:
            _text_digest(hasher, item)
        for item in component_weights:
            _array_digest(hasher, np.asarray([item], dtype=np.float64))
        _array_digest(hasher, np.asarray([prior_weight], dtype=np.float64))
        for item in (prior_mean, prior_covariance, information, natural):
            _array_digest(hasher, item)

        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "prior_id", prior_id)
        object.__setattr__(self, "construction_id", construction_id)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "anchor_prior_mean", prior_mean)
        object.__setattr__(self, "anchor_prior_covariance", prior_covariance)
        object.__setattr__(self, "information_increment", information)
        object.__setattr__(self, "natural_parameter_increment", natural)
        object.__setattr__(self, "component_message_ids", component_ids)
        object.__setattr__(self, "component_weights", component_weights)
        object.__setattr__(self, "prior_weight", prior_weight)
        object.__setattr__(self, "message_id", hasher.hexdigest())

    @property
    def query_dimension(self) -> int:
        return int(self.anchor_prior_mean.shape[0])

    @property
    def payload_nbytes(self) -> int:
        """Bytes needed after the anchor prior is already resident."""

        return int(
            self.information_increment.nbytes
            + self.natural_parameter_increment.nbytes
        )

    @property
    def anchored_storage_nbytes(self) -> int:
        """Bytes for the message plus its explicit anchor prior."""

        return int(
            self.payload_nbytes
            + self.anchor_prior_mean.nbytes
            + self.anchor_prior_covariance.nbytes
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "prob4d.gaussian-query-message.v1",
            "message_id": self.message_id,
            "query_id": self.query_id,
            "prior_id": self.prior_id,
            "query_dimension": self.query_dimension,
            "evidence_ids": list(self.evidence_ids),
            "construction_id": self.construction_id,
            "component_message_ids": list(self.component_message_ids),
            "component_weights": list(self.component_weights),
            "prior_weight": self.prior_weight,
            "payload_nbytes": self.payload_nbytes,
            "anchored_storage_nbytes": self.anchored_storage_nbytes,
            "claim_boundary": (
                "same-query same-anchor Gaussian posterior only; "
                "not observation evidence or a calibration guarantee"
            ),
        }


def apply_gaussian_query_message(
    message: GaussianQueryMessage,
    *,
    prior_id: str | None = None,
    prior_mean: object | None = None,
    prior_covariance: object | None = None,
) -> GaussianQueryBelief:
    """Apply a query message to its byte-identical anchor prior.

    Omitting all prior arguments uses the retained anchor.  Supplying a prior is
    an explicit compatibility audit: all three fields are required and the
    arrays must be byte-identical after float64 conversion.
    """

    if not isinstance(message, GaussianQueryMessage):
        raise TypeError("message must be a GaussianQueryMessage")
    supplied = (
        prior_id is not None,
        prior_mean is not None,
        prior_covariance is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "prior_id, prior_mean, and prior_covariance must be supplied together"
        )
    if all(supplied):
        retained_prior_id = _identifier(prior_id, name="prior_id")
        retained_mean = _readonly_vector(prior_mean, name="prior_mean")
        retained_covariance = _positive_definite(
            prior_covariance,
            name="prior_covariance",
        )
        if retained_prior_id != message.prior_id:
            raise ValueError("prior_id does not match the message anchor")
        if not np.array_equal(retained_mean, message.anchor_prior_mean):
            raise ValueError("prior_mean is not byte-identical to the message anchor")
        if not np.array_equal(
            retained_covariance,
            message.anchor_prior_covariance,
        ):
            raise ValueError(
                "prior_covariance is not byte-identical to the message anchor"
            )
    else:
        retained_mean = message.anchor_prior_mean
        retained_covariance = message.anchor_prior_covariance

    prior_precision = _precision(retained_covariance)
    posterior_precision = 0.5 * (
        prior_precision
        + message.information_increment
        + (prior_precision + message.information_increment).T
    )
    posterior_precision = _positive_definite(
        posterior_precision,
        name="posterior_precision",
    )
    posterior_natural = (
        prior_precision @ retained_mean
        + message.natural_parameter_increment
    )
    posterior_covariance = _precision(posterior_precision)
    posterior_mean = np.linalg.solve(posterior_precision, posterior_natural)
    return GaussianQueryBelief(
        query_id=message.query_id,
        prior_id=message.prior_id,
        mean=posterior_mean,
        covariance=posterior_covariance,
        source_message_id=message.message_id,
    )


def compress_gaussian_query_posterior(
    posterior: GaussianQueryPosterior,
    *,
    query_id: str,
    prior_id: str,
    evidence_ids: Iterable[str],
    parity_relative_tolerance: float = 1e-10,
) -> GaussianQueryMessage:
    """Compress one exact Gaussian query posterior to a prior-relative message."""

    if not isinstance(posterior, GaussianQueryPosterior):
        raise TypeError("posterior must be a GaussianQueryPosterior")
    tolerance = _fraction(
        parity_relative_tolerance,
        name="parity_relative_tolerance",
        include_one=False,
    )
    prior_covariance = _positive_definite(
        posterior.prior_covariance,
        name="posterior.prior_covariance",
    )
    posterior_covariance = _positive_definite(
        posterior.posterior_covariance,
        name="posterior.posterior_covariance",
    )
    prior_precision = _precision(prior_covariance)
    posterior_precision = _precision(posterior_covariance)
    information = 0.5 * (
        posterior_precision
        - prior_precision
        + (posterior_precision - prior_precision).T
    )
    natural = (
        posterior_precision @ posterior.posterior_mean
        - prior_precision @ posterior.prior_mean
    )
    message = GaussianQueryMessage(
        query_id=query_id,
        prior_id=prior_id,
        evidence_ids=tuple(evidence_ids),
        anchor_prior_mean=posterior.prior_mean,
        anchor_prior_covariance=prior_covariance,
        information_increment=information,
        natural_parameter_increment=natural,
    )
    reconstructed = apply_gaussian_query_message(message)
    mean_scale = max(
        float(np.linalg.norm(posterior.posterior_mean)),
        1.0,
    )
    covariance_scale = max(
        float(np.linalg.norm(posterior_covariance, ord="fro")),
        1.0,
    )
    mean_error = float(
        np.linalg.norm(reconstructed.mean - posterior.posterior_mean)
        / mean_scale
    )
    covariance_error = float(
        np.linalg.norm(
            reconstructed.covariance - posterior_covariance,
            ord="fro",
        )
        / covariance_scale
    )
    if mean_error > tolerance or covariance_error > tolerance:
        raise RuntimeError(
            "query-message reconstruction exceeded parity_relative_tolerance"
        )
    return message


def _matching_anchor(messages: tuple[GaussianQueryMessage, ...]) -> None:
    first = messages[0]
    for message in messages[1:]:
        if message.query_id != first.query_id:
            raise ValueError("all messages must have the same query_id")
        if message.prior_id != first.prior_id:
            raise ValueError("all messages must have the same prior_id")
        if not np.array_equal(
            message.anchor_prior_mean,
            first.anchor_prior_mean,
        ):
            raise ValueError("all messages must have the same anchor prior mean")
        if not np.array_equal(
            message.anchor_prior_covariance,
            first.anchor_prior_covariance,
        ):
            raise ValueError("all messages must have the same anchor prior covariance")


def fuse_gaussian_query_messages_covariance_intersection(
    messages: Iterable[GaussianQueryMessage],
    *,
    weights: object,
    construction_id: str = "query-ci-v1",
) -> GaussianQueryMessage:
    """Fuse prior-anchored messages without assuming cross-message independence.

    If component weights sum to less than one, the remaining weight belongs to
    the unchanged anchor prior.  Equivalently, the result is covariance
    intersection over the component posteriors and the anchor prior.
    """

    retained = tuple(messages)
    if not retained:
        raise ValueError("messages must contain at least one message")
    if any(not isinstance(item, GaussianQueryMessage) for item in retained):
        raise TypeError("messages must contain only GaussianQueryMessage values")
    _matching_anchor(retained)
    raw_weights = np.asarray(weights, dtype=np.float64)
    if raw_weights.shape != (len(retained),):
        raise ValueError("weights must have one entry per message")
    if not np.all(np.isfinite(raw_weights)):
        raise ValueError("weights must be finite")
    if np.any(raw_weights < 0.0):
        raise ValueError("weights must be nonnegative")
    total_weight = float(np.sum(raw_weights, dtype=np.float64))
    if total_weight <= 0.0:
        raise ValueError("at least one message weight must be positive")
    if total_weight > 1.0:
        raise ValueError("message weights must sum to at most one")

    aggregate: dict[str, tuple[GaussianQueryMessage, float]] = {}
    for message, weight in zip(retained, raw_weights, strict=True):
        if weight == 0.0:
            continue
        if message.message_id in aggregate:
            previous_message, previous_weight = aggregate[message.message_id]
            aggregate[message.message_id] = (
                previous_message,
                previous_weight + float(weight),
            )
        else:
            aggregate[message.message_id] = (message, float(weight))
    ordered = tuple(aggregate[key] for key in sorted(aggregate))
    first = retained[0]
    information = np.zeros_like(first.information_increment)
    natural = np.zeros_like(first.natural_parameter_increment)
    evidence: set[str] = set()
    component_ids: list[str] = []
    component_weights: list[float] = []
    for message, weight in ordered:
        information += weight * message.information_increment
        natural += weight * message.natural_parameter_increment
        evidence.update(message.evidence_ids)
        component_ids.append(message.message_id)
        component_weights.append(weight)

    prior_weight = 1.0 - total_weight
    return GaussianQueryMessage(
        query_id=first.query_id,
        prior_id=first.prior_id,
        evidence_ids=tuple(sorted(evidence)),
        anchor_prior_mean=first.anchor_prior_mean,
        anchor_prior_covariance=first.anchor_prior_covariance,
        information_increment=information,
        natural_parameter_increment=natural,
        construction_id=construction_id,
        component_message_ids=tuple(component_ids),
        component_weights=tuple(component_weights),
        prior_weight=prior_weight,
    )


def select_pairwise_covariance_intersection(
    first: GaussianQueryMessage,
    second: GaussianQueryMessage,
    *,
    grid_size: int = 1001,
    objective: Literal["logdet", "trace"] = "logdet",
) -> GaussianQueryMessage:
    """Select a deterministic pairwise covariance-intersection weight."""

    if not isinstance(first, GaussianQueryMessage) or not isinstance(
        second,
        GaussianQueryMessage,
    ):
        raise TypeError("first and second must be GaussianQueryMessage values")
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(
        grid_size,
        (int, np.integer),
    ):
        raise TypeError("grid_size must be an integer")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    if objective not in ("logdet", "trace"):
        raise ValueError("objective must be 'logdet' or 'trace'")
    ordered = tuple(sorted((first, second), key=lambda item: item.message_id))
    _matching_anchor(ordered)
    prior_precision = _precision(ordered[0].anchor_prior_covariance)
    best_index = 0
    best_value = math.inf
    for index, weight in enumerate(np.linspace(0.0, 1.0, int(grid_size))):
        information = (
            weight * ordered[0].information_increment
            + (1.0 - weight) * ordered[1].information_increment
        )
        covariance = _precision(
            _positive_definite(
                prior_precision + information,
                name="candidate posterior precision",
            )
        )
        if objective == "logdet":
            sign, value = np.linalg.slogdet(covariance)
            if sign <= 0.0:
                raise RuntimeError("candidate posterior covariance is not positive definite")
            score = float(value)
        else:
            score = float(np.trace(covariance))
        if score < best_value:
            best_index = index
            best_value = score
    selected = best_index / (int(grid_size) - 1)
    return fuse_gaussian_query_messages_covariance_intersection(
        ordered,
        weights=np.asarray([selected, 1.0 - selected], dtype=np.float64),
        construction_id=f"pairwise-{objective}-grid-{int(grid_size)}-v1",
    )
