from __future__ import annotations

import numpy as np
import pytest

import prob4d.observation_export as observation_export
import prob4d.provider_v2 as provider
from prob4d.composition_jacobian import (
    composition_jacobian_mode,
    current_composition_jacobian_mode,
)
from prob4d.covariance_root import (
    covariance_root_mode,
    current_covariance_root_mode,
)
from prob4d.export_numerics import (
    LEGACY_PROVIDER_V1_NUMERICS,
    PROVIDER_V2_NUMERICS,
    ExportNumericsPolicy,
    export_numerics_policy,
    resolve_export_numerics_policy,
)
from prob4d.observation_contract import ObservationBeliefExportV1
from prob4d.runtime_revision import RuntimeRevisionAttestation


def _observation() -> ObservationBeliefExportV1:
    return ObservationBeliefExportV1(
        case_id="case-a",
        stream_id="prob4d:test",
        causal_frame_stop=2,
        view_names=("camera0",),
        window_names=("window0",),
        factor_names=(),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="d" * 64,
        declared_frame_ids=np.asarray([0]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]]),
        frame_ids=np.asarray([0]),
        entity_ids=np.asarray([0]),
        view_indices=np.asarray([0]),
        window_indices=np.asarray([0]),
        correlation_group_ids=np.asarray([0]),
        factor_group_ids=np.asarray([0]),
        prior_reliability=np.asarray([1.0]),
        association_probability=np.asarray([1.0]),
        local_covariance_m2=np.asarray([np.eye(3)]),
        low_rank_factor_m=np.empty((1, 3, 0)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([1.0]),
        group_composite_weight=np.asarray([1.0]),
        metadata={"existing": True},
    )


def _runtime_attestation() -> RuntimeRevisionAttestation:
    return RuntimeRevisionAttestation(
        expected_revision="a" * 40,
        observed_revision="a" * 40,
        source="deployment_environment",
        clean_checkout=None,
        matched=True,
        independently_verified=False,
    )


def test_importing_numerical_modules_does_not_replace_export_functions() -> None:
    assert observation_export._compose_jacobians.__module__ == (
        "prob4d.observation_export"
    )
    assert observation_export.deterministic_covariance_root.__module__ == (
        "prob4d.observation_export"
    )


def test_named_policies_are_immutable_and_select_declared_implementations() -> None:
    assert LEGACY_PROVIDER_V1_NUMERICS.composition_jacobian_mode == (
        "legacy_finite_difference"
    )
    assert LEGACY_PROVIDER_V1_NUMERICS.covariance_root_mode == "legacy_eigenvectors"
    assert PROVIDER_V2_NUMERICS.composition_jacobian_mode == "analytic"
    assert PROVIDER_V2_NUMERICS.covariance_root_mode == "canonical_eigenspaces"
    with pytest.raises((AttributeError, TypeError)):
        PROVIDER_V2_NUMERICS.policy_id = "changed"  # type: ignore[misc]


def test_explicit_policy_is_independent_of_compatibility_context() -> None:
    covariance = np.diag([4.0, 4.0, 1.0])
    with (
        covariance_root_mode("legacy_eigenvectors"),
        composition_jacobian_mode("legacy_finite_difference"),
    ):
        legacy_root, _ = observation_export.deterministic_covariance_root(
            covariance,
            max_rank=1,
        )
        assert legacy_root.shape == (3, 1)
        with pytest.raises(ValueError, match="max_rank cuts through"):
            PROVIDER_V2_NUMERICS.covariance_root(covariance, max_rank=1)


def test_policy_factory_and_resolver_fail_closed() -> None:
    policy = export_numerics_policy(
        composition_jacobian_mode="analytic",
        covariance_root_mode="legacy_eigenvectors",
    )
    assert isinstance(policy, ExportNumericsPolicy)
    assert resolve_export_numerics_policy(policy) is policy
    with pytest.raises(ValueError, match="composition Jacobian mode"):
        export_numerics_policy(
            composition_jacobian_mode="invalid",  # type: ignore[arg-type]
            covariance_root_mode="legacy_eigenvectors",
        )
    with pytest.raises(TypeError, match="ExportNumericsPolicy"):
        resolve_export_numerics_policy(object())  # type: ignore[arg-type]


def test_provider_v2_passes_explicit_policy_without_changing_attestation(
    monkeypatch,
) -> None:
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured.update(manifest_path=manifest_path, **kwargs)
        assert current_covariance_root_mode() == "canonical_eigenspaces"
        assert current_composition_jacobian_mode() == "analytic"
        return _observation()

    monkeypatch.setattr(provider._v1, "export_observation_belief", fake_export)
    monkeypatch.setattr(
        provider,
        "inspect_runtime_revision",
        lambda revision: _runtime_attestation(),
    )
    result = provider.export_exploratory_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        source_revision="a" * 40,
    )

    policy = captured["numerics_policy"]
    assert isinstance(policy, ExportNumericsPolicy)
    assert policy.composition_jacobian_mode == "analytic"
    assert policy.covariance_root_mode == "canonical_eigenspaces"
    assert result.metadata["prob4d_provider_attestation"][
        "composition_jacobian_mode"
    ] == "analytic"
    assert current_covariance_root_mode() == "legacy_eigenvectors"
    assert current_composition_jacobian_mode() == "legacy_finite_difference"


def test_private_export_core_forwards_explicit_policy(monkeypatch) -> None:
    import prob4d._provider_export_core as core

    captured = {}

    def fake_build(manifest_path, **kwargs):
        captured.update(manifest_path=manifest_path, **kwargs)
        return _observation()

    monkeypatch.setattr(core, "build_prob4d_observation_belief", fake_build)
    monkeypatch.setattr(
        core,
        "bind_causal_stream_contract_v2",
        lambda artifact, **kwargs: artifact,
    )
    result = core.export_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        source_revision="a" * 40,
        allow_pointwise_covariance_fallback=True,
        numerics_policy=PROVIDER_V2_NUMERICS,
    )

    assert captured["numerics_policy"] is PROVIDER_V2_NUMERICS
    assert result.metadata["covariance_calibration"]["status"] == (
        "uncalibrated_exploratory"
    )
