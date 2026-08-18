"""Preview version-1 structured-covariance façade backed by :mod:`prob4d.api.v2`."""

from __future__ import annotations

from typing import Final

from .v2 import (
    CovarianceComponent,
    ObservationFactorStack,
    ObservationGaussianOperator,
    ProjectedObservationCovariance,
    SparseStackedObservationFactors,
    StackedObservationFactors,
    TreeSparseStackedObservationFactors,
    build_observation_gaussian_operator,
    observation_covariance_action,
    observation_covariance_quadratic,
    observation_gaussian_nll,
    observation_log_determinant,
    observation_precision_quadratic,
    project_observation_covariance,
    solve_observation_covariance,
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
    "ObservationGaussianOperator",
    "ProjectedObservationCovariance",
    "SparseStackedObservationFactors",
    "StackedObservationFactors",
    "TreeSparseStackedObservationFactors",
    "build_observation_gaussian_operator",
    "observation_covariance_action",
    "observation_covariance_quadratic",
    "observation_gaussian_nll",
    "observation_log_determinant",
    "observation_precision_quadratic",
    "project_observation_covariance",
    "solve_observation_covariance",
    "stack_observation_factors",
    "stack_sparse_observation_factors",
    "stack_tree_sparse_observation_factors",
]
