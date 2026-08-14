"""Immutable numerical policies for Prob4D observation export.

The policy makes covariance-root and composition-Jacobian choices explicit at
provider boundaries. Compatibility contexts remain available for older internal
callers, but provider-v2 exports no longer depend on import order or private
function replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .composition_jacobian import (
    COMPOSITION_JACOBIAN_MODES,
    CompositionJacobianMode,
    composition_jacobian_function,
    current_composition_jacobian_mode,
)
from .covariance_root import (
    COVARIANCE_ROOT_MODES,
    CovarianceRootMode,
    covariance_root_function,
    current_covariance_root_mode,
)
from .sim3 import Sim3


class CompositionJacobianFunction(Protocol):
    """Callable contract for one ``Sim(3)`` composition derivative policy."""

    def __call__(
        self,
        parent: Sim3,
        relative: Sim3,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class CovarianceRootFunction(Protocol):
    """Callable contract for covariance factorization used by the exporter."""

    def __call__(
        self,
        covariance: np.ndarray,
        *,
        max_rank: int | None = None,
        relative_eigenvalue_floor: float = 1e-12,
        coordinate_normalizer: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]: ...


@dataclass(frozen=True)
class ExportNumericsPolicy:
    """Complete immutable numerical choice for one observation export."""

    policy_id: str
    composition_jacobian_mode: CompositionJacobianMode
    covariance_root_mode: CovarianceRootMode
    compose_jacobians: CompositionJacobianFunction
    covariance_root: CovarianceRootFunction

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("export numerical policy_id must be nonempty")
        if self.composition_jacobian_mode not in COMPOSITION_JACOBIAN_MODES:
            raise ValueError(
                "composition Jacobian mode must be one of "
                f"{COMPOSITION_JACOBIAN_MODES}"
            )
        if self.covariance_root_mode not in COVARIANCE_ROOT_MODES:
            raise ValueError(
                f"covariance root mode must be one of {COVARIANCE_ROOT_MODES}"
            )
        if not callable(self.compose_jacobians) or not callable(self.covariance_root):
            raise TypeError("export numerical policy implementations must be callable")


def export_numerics_policy(
    *,
    composition_jacobian_mode: CompositionJacobianMode,
    covariance_root_mode: CovarianceRootMode,
) -> ExportNumericsPolicy:
    """Build the policy associated with two declared numerical modes."""

    if composition_jacobian_mode not in COMPOSITION_JACOBIAN_MODES:
        raise ValueError(
            "composition Jacobian mode must be one of "
            f"{COMPOSITION_JACOBIAN_MODES}"
        )
    if covariance_root_mode not in COVARIANCE_ROOT_MODES:
        raise ValueError(
            f"covariance root mode must be one of {COVARIANCE_ROOT_MODES}"
        )
    policy_id = (
        "prob4d-export-numerics-v1:"
        f"{composition_jacobian_mode}:{covariance_root_mode}"
    )
    return ExportNumericsPolicy(
        policy_id=policy_id,
        composition_jacobian_mode=composition_jacobian_mode,
        covariance_root_mode=covariance_root_mode,
        compose_jacobians=composition_jacobian_function(
            composition_jacobian_mode
        ),
        covariance_root=covariance_root_function(covariance_root_mode),
    )


LEGACY_PROVIDER_V1_NUMERICS = export_numerics_policy(
    composition_jacobian_mode="legacy_finite_difference",
    covariance_root_mode="legacy_eigenvectors",
)
PROVIDER_V2_NUMERICS = export_numerics_policy(
    composition_jacobian_mode="analytic",
    covariance_root_mode="canonical_eigenspaces",
)


def current_export_numerics_policy() -> ExportNumericsPolicy:
    """Resolve the compatibility-local modes into one immutable policy."""

    return export_numerics_policy(
        composition_jacobian_mode=current_composition_jacobian_mode(),
        covariance_root_mode=current_covariance_root_mode(),
    )


def resolve_export_numerics_policy(
    policy: ExportNumericsPolicy | None,
) -> ExportNumericsPolicy:
    """Return an explicit policy or resolve the legacy compatibility context."""

    if policy is None:
        return current_export_numerics_policy()
    if not isinstance(policy, ExportNumericsPolicy):
        raise TypeError("numerics_policy must be an ExportNumericsPolicy")
    return policy


__all__ = [
    "CompositionJacobianFunction",
    "CovarianceRootFunction",
    "ExportNumericsPolicy",
    "LEGACY_PROVIDER_V1_NUMERICS",
    "PROVIDER_V2_NUMERICS",
    "current_export_numerics_policy",
    "export_numerics_policy",
    "resolve_export_numerics_policy",
]
