"""Stable provider-v2 surface for explicit-gauge observation factors.

This module collects the neutral schema-v4 factor contract, the strict
claim-bearing provider-v2 envelope, and sparse in-memory execution adapters in
one import boundary. It is additive to :mod:`prob4d.provider_v2`, whose frozen
observation-belief and factor-stream symbols remain unchanged.
"""

from .observation_factors import (
    GAUGE_COVARIANCE_SEMANTICS,
    GAUGE_PARAMETERIZATION,
    LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
    OBSERVATION_FACTOR_SOURCE_REPOSITORY,
    PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
    GaugeCovarianceSemantics,
    LinearizedObservationFactor,
    ObservationFactor,
    ObservationFactorBundle,
    StackedObservationFactors,
    load_observation_factor_bundle,
    sim3_point_jacobian,
    stack_observation_factors,
    write_observation_factor_bundle,
)
from .provider_v2_factor_bundle import (
    CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA,
    CLAIM_BEARING_FACTOR_BUNDLE_VERSION,
    ClaimBearingObservationFactorBundleEnvelopeV1,
    ValidatedClaimBearingObservationFactorBundle,
    load_claim_bearing_observation_factor_bundle,
    seal_claim_bearing_observation_factor_bundle,
    validate_claim_bearing_observation_factor_bundle,
    write_claim_bearing_observation_factor_bundle,
)
from .sparse_observation_factors import (
    SparseStackedObservationFactors,
    stack_sparse_observation_factors,
)
from .tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    bind_gauge_tree_prior,
    stack_tree_sparse_observation_factors,
)

PROVIDER_FACTOR_API_VERSION = 2
PROB4D_PROVIDER_FACTOR_API_VERSION = PROVIDER_FACTOR_API_VERSION

__all__ = [
    "CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA",
    "CLAIM_BEARING_FACTOR_BUNDLE_VERSION",
    "ClaimBearingObservationFactorBundleEnvelopeV1",
    "GAUGE_COVARIANCE_SEMANTICS",
    "GAUGE_PARAMETERIZATION",
    "GaugeCovarianceSemantics",
    "LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION",
    "LinearizedObservationFactor",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "OBSERVATION_FACTOR_SOURCE_REPOSITORY",
    "ObservationFactor",
    "ObservationFactorBundle",
    "PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION",
    "PROB4D_PROVIDER_FACTOR_API_VERSION",
    "PROVIDER_FACTOR_API_VERSION",
    "SparseStackedObservationFactors",
    "StackedObservationFactors",
    "TreeSparseStackedObservationFactors",
    "ValidatedClaimBearingObservationFactorBundle",
    "bind_gauge_tree_prior",
    "load_claim_bearing_observation_factor_bundle",
    "load_observation_factor_bundle",
    "seal_claim_bearing_observation_factor_bundle",
    "sim3_point_jacobian",
    "stack_observation_factors",
    "stack_sparse_observation_factors",
    "stack_tree_sparse_observation_factors",
    "validate_claim_bearing_observation_factor_bundle",
    "write_claim_bearing_observation_factor_bundle",
    "write_observation_factor_bundle",
]
