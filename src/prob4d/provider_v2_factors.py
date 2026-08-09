"""Stable provider-v2 surface for explicit-gauge observation factors.

This module collects the neutral schema-v4 factor contract, strict claim-bearing
provider-v2 envelopes, and sparse execution and persistence adapters in one
import boundary. It is additive to :mod:`prob4d.provider_v2`, whose frozen
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
from .provider_v2_tree_sparse_artifact import (
    CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA,
    CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION,
    ClaimBearingTreeSparseObservationEnvelopeV1,
    ValidatedClaimBearingTreeSparseObservation,
    load_claim_bearing_tree_sparse_observation,
    seal_claim_bearing_tree_sparse_observation,
    validate_claim_bearing_tree_sparse_observation,
    write_claim_bearing_tree_sparse_observation,
)
from .provider_v2_tree_sparse_manifest import (
    CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION,
    TREE_SPARSE_PROVIDER_CAPABILITIES,
    prob4d_tree_sparse_provider_manifest,
)
from .sparse_observation_factors import (
    SparseStackedObservationFactors,
    stack_sparse_observation_factors,
)
from .tree_sparse_observation_artifact import (
    TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA,
    TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION,
    TREE_SPARSE_OBSERVATION_CLAIM_BOUNDARY,
    TREE_SPARSE_OBSERVATION_STORAGE_SEMANTICS,
    LoadedTreeSparseObservationArtifactV1,
    TreeSparseObservationArrayMemberV1,
    TreeSparseObservationArtifactV1,
    load_tree_sparse_observation_artifact,
    write_tree_sparse_observation_artifact,
)
from .tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    bind_gauge_tree_prior,
    build_tree_sparse_observation_factors,
    stack_tree_sparse_observation_factors,
)

PROVIDER_FACTOR_API_VERSION = 2
PROB4D_PROVIDER_FACTOR_API_VERSION = PROVIDER_FACTOR_API_VERSION

__all__ = [
    "CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA",
    "CLAIM_BEARING_FACTOR_BUNDLE_VERSION",
    "CLAIM_BEARING_TREE_SPARSE_OBSERVATION_ENVELOPE_VERSION",
    "CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA",
    "CLAIM_BEARING_TREE_SPARSE_OBSERVATION_VERSION",
    "ClaimBearingObservationFactorBundleEnvelopeV1",
    "ClaimBearingTreeSparseObservationEnvelopeV1",
    "GAUGE_COVARIANCE_SEMANTICS",
    "GAUGE_PARAMETERIZATION",
    "GaugeCovarianceSemantics",
    "LEGACY_OBSERVATION_FACTOR_SCHEMA_VERSION",
    "LinearizedObservationFactor",
    "LoadedTreeSparseObservationArtifactV1",
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
    "TREE_SPARSE_OBSERVATION_ARTIFACT_SCHEMA",
    "TREE_SPARSE_OBSERVATION_ARTIFACT_VERSION",
    "TREE_SPARSE_OBSERVATION_CLAIM_BOUNDARY",
    "TREE_SPARSE_OBSERVATION_STORAGE_SEMANTICS",
    "TREE_SPARSE_PROVIDER_CAPABILITIES",
    "TreeSparseObservationArrayMemberV1",
    "TreeSparseObservationArtifactV1",
    "TreeSparseStackedObservationFactors",
    "ValidatedClaimBearingObservationFactorBundle",
    "ValidatedClaimBearingTreeSparseObservation",
    "bind_gauge_tree_prior",
    "build_tree_sparse_observation_factors",
    "load_claim_bearing_observation_factor_bundle",
    "load_claim_bearing_tree_sparse_observation",
    "load_observation_factor_bundle",
    "load_tree_sparse_observation_artifact",
    "prob4d_tree_sparse_provider_manifest",
    "seal_claim_bearing_observation_factor_bundle",
    "seal_claim_bearing_tree_sparse_observation",
    "sim3_point_jacobian",
    "stack_observation_factors",
    "stack_sparse_observation_factors",
    "stack_tree_sparse_observation_factors",
    "validate_claim_bearing_observation_factor_bundle",
    "validate_claim_bearing_tree_sparse_observation",
    "write_claim_bearing_observation_factor_bundle",
    "write_claim_bearing_tree_sparse_observation",
    "write_observation_factor_bundle",
    "write_tree_sparse_observation_artifact",
]
