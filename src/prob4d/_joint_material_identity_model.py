"""Immutable bounded joint material-identity posterior record."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from ._immutable_json import frozen_finite_json_mapping, plain_json
from ._joint_material_identity_common import (
    CLAIM_BOUNDARY,
    JOINT_ASSIGNMENT_SEMANTICS,
    JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA,
    JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION,
    FloatArray,
    _canonical_mixtures,
    _finite_real,
    _integer,
    _readonly,
    _sha256,
    _sha256_json,
    _string,
)
from ._joint_material_identity_records import (
    JointIdentityAssignmentV1,
    JointIdentityMarginalV1,
)
from .material_identity_mixture import MaterialIdentityMixtureV1


@dataclass(frozen=True, eq=False)
class JointMaterialIdentityPosteriorV1:
    """Self-contained posterior over exact globally feasible assignments."""

    window_order: tuple[str, ...]
    mixtures: tuple[MaterialIdentityMixtureV1, ...]
    maximum_joint_assignments: int
    unconstrained_assignment_count: int
    assignments: tuple[JointIdentityAssignmentV1, ...]
    marginals: tuple[JointIdentityMarginalV1, ...]
    log_normalizer: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    constraint_semantics: Literal[
        "conditioned-local-mixtures-window-unique-forest-v1"
    ] = JOINT_ASSIGNMENT_SEMANTICS
    posterior_id: str | None = None

    def __post_init__(self) -> None:
        order = tuple(
            _string(value, name=f"window_order[{index}]")
            for index, value in enumerate(self.window_order)
        )
        canonical = _canonical_mixtures(self.mixtures, window_order=order)
        if canonical != self.mixtures:
            raise ValueError("mixtures must use canonical target-endpoint order")
        maximum = _integer(
            self.maximum_joint_assignments,
            name="maximum_joint_assignments",
            minimum=1,
        )
        unconstrained = _integer(
            self.unconstrained_assignment_count,
            name="unconstrained_assignment_count",
            minimum=1,
        )
        if type(self.assignments) is not tuple or not self.assignments:
            raise ValueError("assignments must be a non-empty tuple")
        if type(self.marginals) is not tuple or len(self.marginals) != len(canonical):
            raise ValueError("marginals must align with mixtures")
        probabilities = np.asarray(
            [assignment.probability for assignment in self.assignments],
            dtype=np.float64,
        )
        if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("assignment probabilities must sum to one")
        if self.constraint_semantics != JOINT_ASSIGNMENT_SEMANTICS:
            raise ValueError("unsupported joint assignment semantics")
        metadata = frozen_finite_json_mapping(
            self.metadata,
            name="joint material-identity posterior metadata",
        )
        log_normalizer = _finite_real(self.log_normalizer, name="log_normalizer")
        object.__setattr__(self, "window_order", order)
        object.__setattr__(self, "maximum_joint_assignments", maximum)
        object.__setattr__(self, "unconstrained_assignment_count", unconstrained)
        object.__setattr__(self, "log_normalizer", log_normalizer)
        object.__setattr__(self, "metadata", metadata)
        expected = _sha256_json(self.identity_record())
        if self.posterior_id is not None and _sha256(
            self.posterior_id,
            name="posterior_id",
        ) != expected:
            raise ValueError("joint material-identity posterior ID mismatch")
        object.__setattr__(self, "posterior_id", expected)

    @property
    def mixture_ids(self) -> tuple[str, ...]:
        return tuple(cast(str, mixture.mixture_id) for mixture in self.mixtures)

    @property
    def assignment_ids(self) -> tuple[str, ...]:
        return tuple(cast(str, assignment.assignment_id) for assignment in self.assignments)

    @property
    def probabilities(self) -> FloatArray:
        return _readonly(
            np.asarray(
                [assignment.probability for assignment in self.assignments],
                dtype=np.float64,
            ),
            dtype=np.float64,
        )

    @property
    def feasible_assignment_count(self) -> int:
        return len(self.assignments)

    @property
    def rejected_assignment_count(self) -> int:
        return self.unconstrained_assignment_count - self.feasible_assignment_count

    @property
    def constraint_rejection_fraction(self) -> float:
        return self.rejected_assignment_count / self.unconstrained_assignment_count

    @property
    def joint_entropy_nats(self) -> float:
        probabilities = self.probabilities
        active = probabilities > 0.0
        return float(-np.sum(probabilities[active] * np.log(probabilities[active])))

    @property
    def effective_assignment_count(self) -> float:
        return float(np.exp(self.joint_entropy_nats))

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": JOINT_MATERIAL_IDENTITY_POSTERIOR_SCHEMA,
            "schema_version": JOINT_MATERIAL_IDENTITY_POSTERIOR_VERSION,
            "window_order": list(self.window_order),
            "maximum_joint_assignments": self.maximum_joint_assignments,
            "unconstrained_assignment_count": self.unconstrained_assignment_count,
            "feasible_assignment_count": self.feasible_assignment_count,
            "rejected_assignment_count": self.rejected_assignment_count,
            "log_normalizer": self.log_normalizer,
            "joint_entropy_nats": self.joint_entropy_nats,
            "effective_assignment_count": self.effective_assignment_count,
            "constraint_semantics": self.constraint_semantics,
            "mixtures": [mixture.to_record() for mixture in self.mixtures],
            "assignments": [assignment.to_record() for assignment in self.assignments],
            "marginals": [marginal.to_record() for marginal in self.marginals],
            "metadata": plain_json(self.metadata),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "posterior_id": self.posterior_id}

    def __eq__(self, other: object) -> bool:
        return bool(
            isinstance(other, JointMaterialIdentityPosteriorV1)
            and self.posterior_id == other.posterior_id
            and self.window_order == other.window_order
            and self.mixture_ids == other.mixture_ids
            and self.assignment_ids == other.assignment_ids
            and np.allclose(self.probabilities, other.probabilities)
            and self.marginals == other.marginals
            and plain_json(self.metadata) == plain_json(other.metadata)
        )

