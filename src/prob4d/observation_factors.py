"""Unfused observation-factor contracts for downstream Bayesian estimators.

The ordinary Prob4D fusion products intentionally collapse overlapping windows
into one trajectory. This module provides a lossless interface that preserves
window gauges, view provenance, correlation groups, reliability, and causal
timing.
"""

from ._observation_factor_bundle import (
    GAUGE_PARAMETERIZATION,
    JOINT_GAUGE_COVARIANCE_SEMANTICS,
    JOINT_OBSERVATION_FACTOR_SCHEMA_VERSION,
    LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    OBSERVATION_FACTOR_SOURCE_REPOSITORY,
    JointObservationFactorBundle,
    ObservationFactorBundle,
    sim3_point_jacobian,
    stack_observation_factors,
)
from ._observation_factor_io import (
    load_observation_factor_bundle,
    write_observation_factor_bundle,
)
from ._observation_factor_types import (
    LinearizedObservationFactor,
    ObservationFactor,
    StackedObservationFactors,
)

__all__ = [
    "GAUGE_PARAMETERIZATION",
    "JOINT_GAUGE_COVARIANCE_SEMANTICS",
    "JOINT_OBSERVATION_FACTOR_SCHEMA_VERSION",
    "LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "OBSERVATION_FACTOR_SOURCE_REPOSITORY",
    "JointObservationFactorBundle",
    "LinearizedObservationFactor",
    "ObservationFactor",
    "ObservationFactorBundle",
    "StackedObservationFactors",
    "load_observation_factor_bundle",
    "sim3_point_jacobian",
    "stack_observation_factors",
    "write_observation_factor_bundle",
]
