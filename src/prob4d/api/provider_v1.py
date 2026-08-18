"""Preview version-1 provider-boundary façade backed by :mod:`prob4d.api.v2`."""

from __future__ import annotations

from typing import Final

from .v2 import (
    RuntimeRevisionAttestation,
    ValidatedClaimBearingObservation,
    ValidatedClaimBearingObservationFactorBundle,
    ValidatedClaimBearingTreeSparseObservation,
    assert_runtime_revision,
    build_provider_attestation,
    compute_provider_manifest_id,
    export_calibrated_observation_belief,
    export_exploratory_observation_belief,
    inspect_runtime_revision,
    load_claim_bearing_observation_belief,
    load_claim_bearing_observation_factor_bundle,
    load_claim_bearing_tree_sparse_observation,
    prob4d_project_identity,
    prob4d_provider_manifest,
    prob4d_tree_sparse_provider_manifest,
    validate_claim_bearing_observation_belief,
    validate_claim_bearing_observation_factor_bundle,
    validate_claim_bearing_tree_sparse_observation,
    validate_prob4d_project_identity,
    validate_provider_attestation,
    validate_provider_manifest,
)

FACADE_VERSION: Final = 1
LIFECYCLE: Final = "preview"

__all__ = [
    "FACADE_VERSION",
    "LIFECYCLE",
    "RuntimeRevisionAttestation",
    "ValidatedClaimBearingObservation",
    "ValidatedClaimBearingObservationFactorBundle",
    "ValidatedClaimBearingTreeSparseObservation",
    "assert_runtime_revision",
    "build_provider_attestation",
    "compute_provider_manifest_id",
    "export_calibrated_observation_belief",
    "export_exploratory_observation_belief",
    "inspect_runtime_revision",
    "load_claim_bearing_observation_belief",
    "load_claim_bearing_observation_factor_bundle",
    "load_claim_bearing_tree_sparse_observation",
    "prob4d_project_identity",
    "prob4d_provider_manifest",
    "prob4d_tree_sparse_provider_manifest",
    "validate_claim_bearing_observation_belief",
    "validate_claim_bearing_observation_factor_bundle",
    "validate_claim_bearing_tree_sparse_observation",
    "validate_prob4d_project_identity",
    "validate_provider_attestation",
    "validate_provider_manifest",
]
