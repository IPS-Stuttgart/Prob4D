"""Immutable assignment and marginal records for joint material identity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._joint_material_identity_common import (
    FloatArray,
    _finite_real,
    _readonly,
    _sha256,
    _sha256_json,
)
from .material_identity_mixture import LocalTrackEndpoint


@dataclass(frozen=True)
class JointIdentityAssignmentV1:
    """One globally feasible choice of one candidate per local mixture."""

    mixture_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    log_weight: float
    probability: float
    assignment_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.mixture_ids) is not tuple or not self.mixture_ids:
            raise ValueError("mixture_ids must be a non-empty tuple")
        if type(self.candidate_ids) is not tuple:
            raise ValueError("candidate_ids must be a tuple")
        mixture_ids = tuple(
            _sha256(value, name=f"mixture_ids[{index}]")
            for index, value in enumerate(self.mixture_ids)
        )
        candidate_ids = tuple(
            _sha256(value, name=f"candidate_ids[{index}]")
            for index, value in enumerate(self.candidate_ids)
        )
        if len(candidate_ids) != len(mixture_ids):
            raise ValueError("candidate_ids must align with mixture_ids")
        log_weight = _finite_real(self.log_weight, name="log_weight")
        probability = _finite_real(
            self.probability,
            name="probability",
            minimum=0.0,
        )
        if probability > 1.0:
            raise ValueError("probability must be at most one")
        identity = {
            "mixture_ids": list(mixture_ids),
            "candidate_ids": list(candidate_ids),
        }
        expected = _sha256_json(identity)
        if self.assignment_id is not None and _sha256(
            self.assignment_id,
            name="assignment_id",
        ) != expected:
            raise ValueError("joint assignment ID mismatch")
        object.__setattr__(self, "mixture_ids", mixture_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "log_weight", log_weight)
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "assignment_id", expected)

    def to_record(self) -> dict[str, object]:
        return {
            "assignment_id": self.assignment_id,
            "mixture_ids": list(self.mixture_ids),
            "candidate_ids": list(self.candidate_ids),
            "log_weight": self.log_weight,
            "probability": self.probability,
        }


@dataclass(frozen=True, eq=False)
class JointIdentityMarginalV1:
    """Candidate-aligned marginal for one local mixture after conditioning."""

    mixture_id: str
    target_endpoint: LocalTrackEndpoint
    candidate_ids: tuple[str, ...]
    probabilities: FloatArray

    def __post_init__(self) -> None:
        mixture_id = _sha256(self.mixture_id, name="mixture_id")
        if not isinstance(self.target_endpoint, LocalTrackEndpoint):
            raise ValueError("target_endpoint must be LocalTrackEndpoint")
        candidate_ids = tuple(
            _sha256(value, name=f"candidate_ids[{index}]")
            for index, value in enumerate(self.candidate_ids)
        )
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.shape != (len(candidate_ids),):
            raise ValueError("probabilities must align with candidate_ids")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("probabilities must be finite and non-negative")
        if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("probabilities must sum to one")
        object.__setattr__(self, "mixture_id", mixture_id)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(
            self,
            "probabilities",
            _readonly(probabilities, dtype=np.float64),
        )

    @property
    def null_probability(self) -> float:
        return float(self.probabilities[0])

    @property
    def identity_entropy_nats(self) -> float:
        active = self.probabilities > 0.0
        return float(
            -np.sum(self.probabilities[active] * np.log(self.probabilities[active]))
        )

    @property
    def effective_hypothesis_count(self) -> float:
        return float(np.exp(self.identity_entropy_nats))

    def __eq__(self, other: object) -> bool:
        return bool(
            isinstance(other, JointIdentityMarginalV1)
            and self.mixture_id == other.mixture_id
            and self.target_endpoint == other.target_endpoint
            and self.candidate_ids == other.candidate_ids
            and np.allclose(
                self.probabilities,
                other.probabilities,
                atol=1e-12,
                rtol=1e-10,
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "mixture_id": self.mixture_id,
            "target_endpoint": self.target_endpoint.to_dict(),
            "candidate_ids": list(self.candidate_ids),
            "probabilities": self.probabilities.tolist(),
            "null_probability": self.null_probability,
            "identity_entropy_nats": self.identity_entropy_nats,
            "effective_hypothesis_count": self.effective_hypothesis_count,
        }

