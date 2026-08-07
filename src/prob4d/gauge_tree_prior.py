"""Sparse square-root priors for causal trees of linearized ``Sim(3)`` gauges.

A causal spanning-tree posterior stores one parent, transition, and independent
innovation Cholesky factor per seven-dimensional window gauge. The representation
is additive and does not change existing provider-v2 or factor-bundle schemas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._gauge_tree_common import (
    GAUGE_DIMENSION,
    GAUGE_TREE_PRIOR_SCHEMA,
    GAUGE_TREE_PRIOR_SEMANTICS,
    GAUGE_TREE_PRIOR_VERSION,
    FloatArray,
    IntArray,
    canonical_array_descriptor,
    canonical_json_sha256,
    validate_factor_arrays,
)
from ._gauge_tree_factorization import dense_factors, transition_factors
from ._gauge_tree_methods import GaugeTreePriorMethods


@dataclass(frozen=True, slots=True)
class GaugeTreeSquareRootPriorV1(GaugeTreePriorMethods):
    """An immutable zero-mean Gaussian prior represented by one causal gauge tree."""

    gauge_ids: tuple[str, ...]
    parent_indices: IntArray
    transition_matrices: FloatArray
    innovation_scale_tril: FloatArray
    source_joint_covariance_sha256: str | None = None
    representation_semantics: str = GAUGE_TREE_PRIOR_SEMANTICS

    def __post_init__(self) -> None:
        ids, parents, transitions, scales, digest = validate_factor_arrays(
            self.gauge_ids,
            self.parent_indices,
            self.transition_matrices,
            self.innovation_scale_tril,
            self.source_joint_covariance_sha256,
            self.representation_semantics,
        )
        object.__setattr__(self, "gauge_ids", ids)
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "transition_matrices", transitions)
        object.__setattr__(self, "innovation_scale_tril", scales)
        object.__setattr__(self, "source_joint_covariance_sha256", digest)

    @classmethod
    def from_transition_covariances(
        cls,
        *,
        gauge_ids: Sequence[str],
        parent_indices: Any,
        transition_matrices: Any,
        innovation_covariances: Any,
    ) -> GaugeTreeSquareRootPriorV1:
        ids, parents, transitions, scales = transition_factors(
            gauge_ids=gauge_ids,
            parent_indices=parent_indices,
            transition_matrices=transition_matrices,
            innovation_covariances=innovation_covariances,
        )
        return cls(ids, parents, transitions, scales)

    @classmethod
    def from_dense_covariance(
        cls,
        *,
        gauge_ids: Sequence[str],
        parent_indices: Any,
        joint_covariance: Any,
        parity_atol: float = 1e-10,
        parity_rtol: float = 1e-8,
    ) -> GaugeTreeSquareRootPriorV1:
        ids, parents, transitions, scales, digest = dense_factors(
            gauge_ids=gauge_ids,
            parent_indices=parent_indices,
            joint_covariance=joint_covariance,
            parity_atol=parity_atol,
            parity_rtol=parity_rtol,
        )
        return cls(ids, parents, transitions, scales, digest)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": GAUGE_TREE_PRIOR_SCHEMA,
            "version": GAUGE_TREE_PRIOR_VERSION,
            "representation_semantics": self.representation_semantics,
            "gauge_dimension": GAUGE_DIMENSION,
            "gauge_ids": list(self.gauge_ids),
            "parent_indices": [int(value) for value in self.parent_indices],
            "transition_matrices": canonical_array_descriptor(self.transition_matrices),
            "innovation_scale_tril": canonical_array_descriptor(self.innovation_scale_tril),
            "source_joint_covariance_sha256": self.source_joint_covariance_sha256,
        }

    @property
    def prior_id(self) -> str:
        return canonical_json_sha256(self._identity_payload())

    def to_dict(self) -> dict[str, object]:
        result = self._identity_payload()
        result.update(
            prior_id=self.prior_id,
            gauge_count=self.gauge_count,
            factor_storage_nbytes=self.factor_storage_nbytes,
            dense_covariance_nbytes=self.dense_covariance_nbytes,
        )
        return result


def main(argv: Sequence[str] | None = None) -> int:
    """Verify or explicitly densify a portable sparse-prior artifact."""

    from .gauge_tree_prior_artifact import main as artifact_main

    return artifact_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GAUGE_DIMENSION",
    "GAUGE_TREE_PRIOR_SCHEMA",
    "GAUGE_TREE_PRIOR_SEMANTICS",
    "GAUGE_TREE_PRIOR_VERSION",
    "GaugeTreeSquareRootPriorV1",
    "main",
]
