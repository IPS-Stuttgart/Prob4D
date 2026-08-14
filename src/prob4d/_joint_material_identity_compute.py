"""Exact bounded enumeration and inference for joint material identities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from ._joint_material_identity_common import (
    FloatArray,
    _canonical_mixtures,
    _endpoint_key,
    _finite_real,
    _integer,
    _logsumexp,
    _selection_is_feasible,
    _sha256,
    _WindowUniqueForest,
)
from ._joint_material_identity_likelihood import MarginalizedJointIdentityLikelihood
from ._joint_material_identity_model import JointMaterialIdentityPosteriorV1
from ._joint_material_identity_records import (
    JointIdentityAssignmentV1,
    JointIdentityMarginalV1,
)
from .material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityMixtureV1,
)


def _marginals_from_assignments(
    mixtures: tuple[MaterialIdentityMixtureV1, ...],
    assignments: tuple[JointIdentityAssignmentV1, ...],
    probabilities: FloatArray,
) -> tuple[JointIdentityMarginalV1, ...]:
    results: list[JointIdentityMarginalV1] = []
    for mixture_index, mixture in enumerate(mixtures):
        values: FloatArray = np.zeros(len(mixture.candidates), dtype=np.float64)
        index_by_id = {
            candidate_id: index for index, candidate_id in enumerate(mixture.candidate_ids)
        }
        for assignment, probability in zip(assignments, probabilities, strict=True):
            values[index_by_id[assignment.candidate_ids[mixture_index]]] += probability
        results.append(
            JointIdentityMarginalV1(
                mixture_id=cast(str, mixture.mixture_id),
                target_endpoint=mixture.target_endpoint,
                candidate_ids=mixture.candidate_ids,
                probabilities=values,
            )
        )
    return tuple(results)


def build_joint_material_identity_posterior(
    mixtures: Sequence[MaterialIdentityMixtureV1],
    *,
    window_order: tuple[str, ...],
    maximum_joint_assignments: int = 100_000,
    metadata: Mapping[str, Any] | None = None,
) -> JointMaterialIdentityPosteriorV1:
    """Condition local mixtures on the exact window-unique forest rule."""

    canonical = _canonical_mixtures(mixtures, window_order=window_order)
    maximum = _integer(
        maximum_joint_assignments,
        name="maximum_joint_assignments",
        minimum=1,
    )
    unconstrained = math.prod(len(mixture.candidates) for mixture in canonical)
    if unconstrained > maximum:
        raise ValueError(
            "unconstrained assignment count "
            f"{unconstrained} exceeds maximum_joint_assignments {maximum}"
        )

    raw: list[tuple[tuple[str, ...], float]] = []
    candidate_ranges = tuple(range(len(mixture.candidates)) for mixture in canonical)
    for indices in itertools.product(*candidate_ranges):
        if not _selection_is_feasible(canonical, indices):
            continue
        candidate_ids = tuple(
            mixture.candidate_ids[index]
            for mixture, index in zip(canonical, indices, strict=True)
        )
        log_weight = float(
            sum(
                mixture.candidates[index].calibrated_log_weight
                for mixture, index in zip(canonical, indices, strict=True)
            )
        )
        raw.append((candidate_ids, log_weight))
    if not raw:
        raise ValueError("the joint identity constraint admitted no assignments")
    log_weights = np.asarray([value[1] for value in raw], dtype=np.float64)
    log_normalizer = _logsumexp(log_weights)
    probabilities = np.exp(log_weights - log_normalizer)
    mixture_ids = tuple(cast(str, mixture.mixture_id) for mixture in canonical)
    assignments = tuple(
        JointIdentityAssignmentV1(
            mixture_ids=mixture_ids,
            candidate_ids=candidate_ids,
            log_weight=log_weight,
            probability=float(probability),
        )
        for (candidate_ids, log_weight), probability in zip(
            raw,
            probabilities,
            strict=True,
        )
    )
    marginals = _marginals_from_assignments(canonical, assignments, probabilities)
    return JointMaterialIdentityPosteriorV1(
        window_order=window_order,
        mixtures=canonical,
        maximum_joint_assignments=maximum,
        unconstrained_assignment_count=unconstrained,
        assignments=assignments,
        marginals=marginals,
        log_normalizer=log_normalizer,
        metadata={} if metadata is None else metadata,
    )


def marginalize_joint_assignment_log_likelihoods(
    posterior: JointMaterialIdentityPosteriorV1,
    assignment_ids: tuple[str, ...],
    log_likelihoods: FloatArray,
    *,
    likelihood_power: float = 1.0,
) -> MarginalizedJointIdentityLikelihood:
    """Marginalize a downstream likelihood over feasible assignments."""

    if not isinstance(posterior, JointMaterialIdentityPosteriorV1):
        raise ValueError("posterior must be JointMaterialIdentityPosteriorV1")
    if type(assignment_ids) is not tuple or assignment_ids != posterior.assignment_ids:
        raise ValueError("assignment_ids must exactly match posterior.assignment_ids")
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if values.shape != (len(assignment_ids),):
        raise ValueError("log_likelihoods must match the assignment count")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_likelihoods may not contain NaN or positive infinity")
    power = _finite_real(likelihood_power, name="likelihood_power", minimum=0.0)
    prior_log = np.log(posterior.probabilities)
    terms = prior_log if power == 0.0 else prior_log + power * values
    log_marginal = _logsumexp(terms)
    if np.isneginf(log_marginal):
        raise ValueError("every joint assignment has impossible likelihood")
    return MarginalizedJointIdentityLikelihood(
        assignment_ids=assignment_ids,
        log_marginal_likelihood=log_marginal,
        posterior_probabilities=np.exp(terms - log_marginal),
        likelihood_power=power,
    )


def joint_candidate_marginals(
    posterior: JointMaterialIdentityPosteriorV1,
    *,
    assignment_probabilities: FloatArray | None = None,
) -> tuple[JointIdentityMarginalV1, ...]:
    """Project assignment probabilities to exact local candidate order."""

    if not isinstance(posterior, JointMaterialIdentityPosteriorV1):
        raise ValueError("posterior must be JointMaterialIdentityPosteriorV1")
    if assignment_probabilities is None:
        probabilities = posterior.probabilities
    else:
        probabilities = np.asarray(assignment_probabilities, dtype=np.float64)
        if probabilities.shape != (posterior.feasible_assignment_count,):
            raise ValueError("assignment_probabilities must match assignments")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("assignment_probabilities must be finite and non-negative")
        if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("assignment_probabilities must sum to one")
    return _marginals_from_assignments(
        posterior.mixtures,
        posterior.assignments,
        probabilities,
    )


def assignment_components(
    posterior: JointMaterialIdentityPosteriorV1,
    assignment_id: str,
) -> tuple[tuple[LocalTrackEndpoint, ...], ...]:
    """Return canonical local endpoint components for one assignment."""

    expected = _sha256(assignment_id, name="assignment_id")
    matches = [
        assignment
        for assignment in posterior.assignments
        if assignment.assignment_id == expected
    ]
    if not matches:
        raise ValueError("assignment_id is absent from the posterior")
    assignment = matches[0]
    lookup = {
        cast(str, mixture.mixture_id): dict(
            zip(mixture.candidate_ids, mixture.candidates, strict=True)
        )
        for mixture in posterior.mixtures
    }
    forest = _WindowUniqueForest()
    for mixture, candidate_id in zip(
        posterior.mixtures,
        assignment.candidate_ids,
        strict=True,
    ):
        forest.add(mixture.target_endpoint)
        candidate = lookup[cast(str, mixture.mixture_id)][candidate_id]
        if candidate.source_endpoint is not None and not forest.union(
            candidate.source_endpoint,
            mixture.target_endpoint,
        ):
            raise AssertionError("stored assignment violates the forest constraint")
    indices = {value: index for index, value in enumerate(posterior.window_order)}
    components = tuple(
        tuple(
            sorted(
                component,
                key=lambda endpoint: _endpoint_key(endpoint, window_indices=indices),
            )
        )
        for component in forest.components()
    )
    return tuple(
        sorted(
            components,
            key=lambda component: _endpoint_key(
                component[0],
                window_indices=indices,
            ),
        )
    )
