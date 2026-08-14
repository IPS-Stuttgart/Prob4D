"""Downstream likelihood result for a joint identity posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._joint_material_identity_common import (
    JOINT_LIKELIHOOD_SEMANTICS,
    FloatArray,
    _finite_real,
    _readonly,
    _sha256,
)


@dataclass(frozen=True)
class MarginalizedJointIdentityLikelihood:
    """Downstream likelihood marginalized over feasible joint assignments."""

    assignment_ids: tuple[str, ...]
    log_marginal_likelihood: float
    posterior_probabilities: FloatArray
    likelihood_power: float
    semantics: Literal[
        "logsumexp-discrete-joint-identity-v1"
    ] = JOINT_LIKELIHOOD_SEMANTICS

    def __post_init__(self) -> None:
        assignment_ids = tuple(
            _sha256(value, name=f"assignment_ids[{index}]")
            for index, value in enumerate(self.assignment_ids)
        )
        probabilities = np.asarray(self.posterior_probabilities, dtype=np.float64)
        if probabilities.shape != (len(assignment_ids),):
            raise ValueError("posterior_probabilities must match assignment_ids")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("posterior_probabilities must be finite and non-negative")
        if not np.isclose(float(np.sum(probabilities)), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("posterior_probabilities must sum to one")
        if self.semantics != JOINT_LIKELIHOOD_SEMANTICS:
            raise ValueError("unsupported joint likelihood semantics")
        object.__setattr__(self, "assignment_ids", assignment_ids)
        object.__setattr__(
            self,
            "log_marginal_likelihood",
            _finite_real(self.log_marginal_likelihood, name="log_marginal_likelihood"),
        )
        object.__setattr__(
            self,
            "posterior_probabilities",
            _readonly(probabilities, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "likelihood_power",
            _finite_real(self.likelihood_power, name="likelihood_power", minimum=0.0),
        )

    @property
    def joint_entropy_nats(self) -> float:
        active = self.posterior_probabilities > 0.0
        return float(
            -np.sum(
                self.posterior_probabilities[active]
                * np.log(self.posterior_probabilities[active])
            )
        )

    @property
    def effective_assignment_count(self) -> float:
        return float(np.exp(self.joint_entropy_nats))
