"""Focused stable imports for Prob4D's portable cross-repository contracts.

This facade intentionally excludes estimator experiments, benchmark runners, and
provider-specific source adapters. New producer inputs belong in
:mod:`prob4d.source`; calibrated claim-bearing exports belong in
:mod:`prob4d.provider_v2`.
"""

from .observation_contract import (
    OBSERVATION_BELIEF_SCHEMA,
    OBSERVATION_BELIEF_VERSION,
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from .observation_factor_stream import (
    OBSERVATION_FACTOR_STREAM_SCHEMA,
    OBSERVATION_FACTOR_STREAM_VERSION,
    ObservationFactorStreamUpdateV1,
    ObservationFactorStreamV1,
    append_observation_factor_bundle,
    load_observation_factor_stream,
    write_observation_factor_stream,
)
from .observation_factors import (
    GAUGE_COVARIANCE_SEMANTICS,
    GAUGE_PARAMETERIZATION,
    OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION,
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
from .observation_validation import load_observation_belief_export
from .provider_attestation import (
    PROVIDER_ATTESTATION_SCHEMA,
    PROVIDER_ATTESTATION_VERSION,
    build_provider_attestation,
    compute_provider_manifest_id,
    validate_provider_attestation,
    validate_provider_manifest,
)

__all__ = [
    "GAUGE_COVARIANCE_SEMANTICS",
    "GAUGE_PARAMETERIZATION",
    "OBSERVATION_BELIEF_SCHEMA",
    "OBSERVATION_BELIEF_VERSION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "OBSERVATION_FACTOR_STREAM_SCHEMA",
    "OBSERVATION_FACTOR_STREAM_VERSION",
    "PROVIDER_ATTESTATION_SCHEMA",
    "PROVIDER_ATTESTATION_VERSION",
    "GaugeCovarianceSemantics",
    "LinearizedObservationFactor",
    "ObservationBeliefExportV1",
    "ObservationFactor",
    "ObservationFactorBundle",
    "ObservationFactorStreamUpdateV1",
    "ObservationFactorStreamV1",
    "StackedObservationFactors",
    "append_observation_factor_bundle",
    "build_provider_attestation",
    "compute_provider_manifest_id",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_observation_factor_stream",
    "save_observation_belief_export",
    "sim3_point_jacobian",
    "stack_observation_factors",
    "validate_provider_attestation",
    "validate_provider_manifest",
    "write_observation_factor_bundle",
    "write_observation_factor_stream",
]
