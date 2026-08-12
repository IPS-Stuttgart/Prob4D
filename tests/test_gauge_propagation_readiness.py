from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.diagnostics.sim3_linearization import (
    GaussianLinearizationAdequacyV1,
    LinearizationAdequacyThresholdsV1,
)
from prob4d.fresh_provider_readiness import FreshProviderCohortLockV1
from prob4d.gauge_propagation_readiness import (
    GAUGE_PROPAGATION_BINDING_METADATA_KEY,
    GaugePropagationReadinessPolicyV1,
    build_gauge_propagation_readiness,
    compose_source_covariance_readiness_gates,
    load_gauge_propagation_readiness,
    write_gauge_propagation_readiness,
)
from prob4d.source_covariance_localization import (
    SourceCovarianceLocalizationGroupV1,
    SourceCovarianceLocalizationPolicyV1,
    SourceCovarianceLocalizationV1,
)

PROVIDER_ID = "1" * 64
COHORT_ID = "2" * 64
QUERY_ID = "3" * 64
SOURCE_COMPETENCE_ID = "4" * 64
JOINT_DIAGNOSTIC_ID = "5" * 64
RESIDUAL_SOURCE_ID = "6" * 64


def _localization(
    *,
    shared: float = 1.0,
    conditional: float = 1.0,
) -> SourceCovarianceLocalizationV1:
    policy = SourceCovarianceLocalizationPolicyV1(
        minimum_group_count=1,
        normalized_nees_lower=0.8,
        normalized_nees_upper=1.2,
        minimum_joint_pass_fraction=1.0,
        shared_energy_lower=0.8,
        shared_energy_upper=1.2,
        minimum_shared_pass_fraction=1.0,
        conditional_energy_lower=0.8,
        conditional_energy_upper=1.2,
        minimum_conditional_pass_fraction=1.0,
        require_shared_subspace=True,
    )
    normalized_nees = max(shared, conditional)
    return SourceCovarianceLocalizationV1(
        provider_manifest_id=PROVIDER_ID,
        cohort_binding_id=COHORT_ID,
        source_provider_competence_id=SOURCE_COMPETENCE_ID,
        joint_diagnostic_sha256=JOINT_DIAGNOSTIC_ID,
        joint_residual_source_sha256=RESIDUAL_SOURCE_ID,
        policy=policy,
        groups=(
            SourceCovarianceLocalizationGroupV1(
                group_id="group-a",
                normalized_nees=normalized_nees,
                shared_subspace_normalized_energy=shared,
                conditional_subspace_normalized_energy=conditional,
                joint_in_band=0.8 <= normalized_nees <= 1.2,
                shared_in_band=0.8 <= shared <= 1.2,
                conditional_in_band=0.8 <= conditional <= 1.2,
            ),
        ),
        source_mean_status="pass",
        identity_reliability_status="pass",
    )


def _cohort_lock() -> FreshProviderCohortLockV1:
    return FreshProviderCohortLockV1(
        protocol_id="fresh-provider-test-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        provider_repository="IPS-Stuttgart/Prob4D",
        provider_revision="b" * 40,
        model_set_id="7" * 64,
        loader_id="8" * 64,
        cohort_binding_id=COHORT_ID,
        promotion_lock_id="9" * 64,
        query_definition_id=QUERY_ID,
        fallback_identity_id="a" * 64,
        development_group_ids=("group-a",),
        calibration_group_ids=("group-b",),
        target_group_ids=("group-c",),
    )


def _binding(localization: SourceCovarianceLocalizationV1) -> dict[str, object]:
    return {
        "provider_manifest_id": PROVIDER_ID,
        "cohort_binding_id": COHORT_ID,
        "query_definition_id": QUERY_ID,
        "source_covariance_localization_id": (
            localization.source_covariance_localization_id
        ),
        "source_group_ids": ["group-a"],
        "causal_prefix_only": True,
        "target_residuals_used": False,
        "target_outcomes_used": False,
    }


def _certificate(
    localization: SourceCovarianceLocalizationV1,
    *,
    adequate: bool = True,
    query: bool = True,
    jacobian_validated: bool = True,
    sample_count: int = 4096,
    binding_updates: dict[str, object] | None = None,
) -> GaussianLinearizationAdequacyV1:
    binding = _binding(localization)
    binding.update(binding_updates or {})
    return GaussianLinearizationAdequacyV1(
        parameterization="sim3-left-perturbation",
        parameter_order=("scale", "rotation", "translation"),
        parameter_dimension=7,
        output_shape=(1, 3),
        sample_count=sample_count,
        batch_size=256,
        seed=23,
        finite_difference_step=1.0e-6,
        jacobian_validated=jacobian_validated,
        thresholds=LinearizationAdequacyThresholdsV1(),
        point_diagnostics=(
            {
                "item_index": 0,
                "relative_trace_error": 0.01,
                "relative_frobenius_error": 0.02,
                "mean_shift_standard_deviations": 0.03,
                "nonlinear_trace": 1.0,
                "linearized_trace": 1.01,
                "principal_axis_anisotropy": 1.1,
                "principal_axis_angle_degrees": None,
            },
        ),
        query_diagnostics=(
            {
                "query_dimension": 1,
                "relative_trace_error": 0.01,
                "relative_frobenius_error": 0.02,
                "mean_shift_standard_deviations": 0.03,
                "nonlinear_trace": 1.0,
                "linearized_trace": 1.01,
            }
            if query
            else None
        ),
        adequate=adequate,
        failure_reasons=() if adequate else ("point-trace-distortion",),
        metadata={
            "mean_transform_vector": [0.0] * 7,
            "point_count": 1,
            "perturbation_side": "left",
            GAUGE_PROPAGATION_BINDING_METADATA_KEY: binding,
        },
    )


def _first_order_policy() -> GaugePropagationReadinessPolicyV1:
    return GaugePropagationReadinessPolicyV1(
        propagation_mode="first-order-marginalized",
        expected_perturbation_side="left",
        expected_parameter_order=("scale", "rotation", "translation"),
        minimum_sample_count=4096,
        require_query_projection=True,
        require_supplied_jacobian_validation=True,
    )


def test_adequate_first_order_readiness_round_trips(tmp_path: Path) -> None:
    localization = _localization()
    readiness = build_gauge_propagation_readiness(
        localization,
        _first_order_policy(),
        query_definition_id=QUERY_ID,
        certificate=_certificate(localization),
    )
    path = tmp_path / "readiness.json"
    write_gauge_propagation_readiness(path, readiness)

    loaded = load_gauge_propagation_readiness(path)
    gauge_gate, point_gate = compose_source_covariance_readiness_gates(
        _cohort_lock(),
        localization,
        loaded,
    )

    assert loaded.classification == "first-order-adequate"
    assert loaded.linearized_marginalization_authorized
    assert gauge_gate.status == "pass"
    assert point_gate.status == "pass"
    assert loaded.to_dict() == readiness.to_dict()


def test_inadequate_linearization_blocks_point_model_authorization() -> None:
    localization = _localization(conditional=2.0)
    readiness = build_gauge_propagation_readiness(
        localization,
        _first_order_policy(),
        query_definition_id=QUERY_ID,
        certificate=_certificate(localization, adequate=False),
    )

    gauge_gate, point_gate = compose_source_covariance_readiness_gates(
        _cohort_lock(),
        localization,
        readiness,
    )

    assert readiness.classification == "first-order-inadequate"
    assert gauge_gate.status == "fail"
    assert "linearization:point-trace-distortion" in gauge_gate.reason_codes
    assert point_gate.status == "not-evaluated"


def test_explicit_gauge_latent_preserves_conditional_covariance_localization() -> None:
    localization = _localization(conditional=2.0)
    readiness = build_gauge_propagation_readiness(
        localization,
        GaugePropagationReadinessPolicyV1.explicit_latent(),
        query_definition_id=QUERY_ID,
    )

    gauge_gate, point_gate = compose_source_covariance_readiness_gates(
        _cohort_lock(),
        localization,
        readiness,
    )

    assert readiness.classification == "explicit-gauge-latent-retained"
    assert gauge_gate.status == "pass"
    assert point_gate.status == "fail"
    assert point_gate.reason_codes == (
        "conditional-subspace-energy-outside-frozen-band",
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"query": False}, "missing-required-query-projection"),
        ({"jacobian_validated": False}, "supplied-jacobian-not-validated"),
        ({"sample_count": 1024}, "insufficient-linearization-sample-count"),
    ],
)
def test_incomplete_first_order_evidence_is_technical_failure(
    kwargs: dict[str, object],
    reason: str,
) -> None:
    localization = _localization(conditional=2.0)
    readiness = build_gauge_propagation_readiness(
        localization,
        _first_order_policy(),
        query_definition_id=QUERY_ID,
        certificate=_certificate(localization, **kwargs),  # type: ignore[arg-type]
    )
    gauge_gate, point_gate = compose_source_covariance_readiness_gates(
        _cohort_lock(),
        localization,
        readiness,
    )

    assert readiness.classification == "technical-failure"
    assert reason in readiness.reason_codes
    assert gauge_gate.status == "technical-failure"
    assert point_gate.status == "not-evaluated"


def test_certificate_binding_mismatch_is_invalid_evidence() -> None:
    localization = _localization()
    certificate = _certificate(
        localization,
        binding_updates={"query_definition_id": "f" * 64},
    )

    with pytest.raises(ValueError, match="binding changed"):
        build_gauge_propagation_readiness(
            localization,
            _first_order_policy(),
            query_definition_id=QUERY_ID,
            certificate=certificate,
        )


def test_target_outcome_use_is_rejected() -> None:
    localization = _localization()
    certificate = _certificate(
        localization,
        binding_updates={"target_outcomes_used": True},
    )

    with pytest.raises(ValueError, match="must not use target residuals or outcomes"):
        build_gauge_propagation_readiness(
            localization,
            _first_order_policy(),
            query_definition_id=QUERY_ID,
            certificate=certificate,
        )


def test_passing_source_gauge_requires_propagation_evidence() -> None:
    with pytest.raises(ValueError, match="requires propagation readiness"):
        compose_source_covariance_readiness_gates(
            _cohort_lock(),
            _localization(),
            None,
        )


def test_source_gauge_failure_has_priority() -> None:
    localization = _localization(shared=2.0)

    gauge_gate, point_gate = compose_source_covariance_readiness_gates(
        _cohort_lock(),
        localization,
        None,
    )

    assert gauge_gate.status == "fail"
    assert gauge_gate.reason_codes == (
        "shared-subspace-energy-outside-frozen-band",
    )
    assert point_gate.status == "not-evaluated"


def test_readiness_loader_rejects_tampered_derived_status(tmp_path: Path) -> None:
    localization = _localization()
    readiness = build_gauge_propagation_readiness(
        localization,
        _first_order_policy(),
        query_definition_id=QUERY_ID,
        certificate=_certificate(localization),
    )
    payload = readiness.to_dict()
    payload["gate_status"] = "fail"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="derived fields changed"):
        load_gauge_propagation_readiness(path)
