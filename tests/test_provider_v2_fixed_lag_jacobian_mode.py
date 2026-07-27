from __future__ import annotations

import numpy as np

import prob4d.provider_v2 as provider
from prob4d.composition_jacobian import current_composition_jacobian_mode
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
        metadata={},
    )


def test_exploratory_fixed_lag_records_legacy_composition_derivatives(
    monkeypatch,
) -> None:
    captured = {}

    def fake_export(manifest_path, **kwargs):
        captured["mode"] = current_composition_jacobian_mode()
        return _observation()

    monkeypatch.setattr(provider._v1, "export_observation_belief", fake_export)
    monkeypatch.setattr(
        provider,
        "inspect_runtime_revision",
        lambda revision: RuntimeRevisionAttestation(
            expected_revision="a" * 40,
            observed_revision="a" * 40,
            source="source_checkout",
            clean_checkout=True,
            matched=True,
            independently_verified=True,
        ),
    )

    result = provider.export_exploratory_observation_belief(
        "predictions.json",
        case_id="case-a",
        causal_frame_stop=10,
        metric_anchor=object(),
        gauge_mode="fixed_lag",
        allow_approximate_fixed_lag_covariance=True,
        source_revision="a" * 40,
    )

    assert captured["mode"] == "legacy_finite_difference"
    assert result.metadata["prob4d_provider_attestation"][
        "composition_jacobian_mode"
    ] == "legacy_finite_difference"
    assert current_composition_jacobian_mode() == "legacy_finite_difference"
