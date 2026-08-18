"""Preview version-1 structured-covariance façade backed by :mod:`prob4d.api.v2`."""

from __future__ import annotations

from typing import Final

from .v2 import (
    CovarianceComponent,
    ObservationFactorStack,
    ProjectedObservationCovariance,
    SparseStackedObservationFactors,
    StackedObservationFactors,
    TreeSparseStackedObservationFactors,
    observation_covariance_action,
    observation_covariance_quadratic,
    project_observation_covariance,
    stack_observation_factors,
    stack_sparse_observation_factors,
    stack_tree_sparse_observation_factors,
)

FACADE_VERSION: Final = 1
LIFECYCLE: Final = "preview"

__all__ = [
    "CovarianceComponent",
    "FACADE_VERSION",
    "LIFECYCLE",
    "ObservationFactorStack",
    "ProjectedObservationCovariance",
    "SparseStackedObservationFactors",
    "StackedObservationFactors",
    "TreeSparseStackedObservationFactors",
    "observation_covariance_action",
    "observation_covariance_quadratic",
    "project_observation_covariance",
    "stack_observation_factors",
    "stack_sparse_observation_factors",
    "stack_tree_sparse_observation_factors",
]
