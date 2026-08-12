from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.calibration_transport import (
    CalibrationTransportPolicyV1,
    CalibrationTransportUnitV1,
    calibration_transport_feature_contract_id,
    evaluate_calibration_transport,
    fit_calibration_transport_model,
)
from prob4d.provider_prefix_admission import (
    PROVIDER_PREFIX_BINDING_METADATA_KEY,
    build_provider_prefix_admission,
    load_provider_prefix_admission,
    write_provider_prefix_admission,
)
from prob4d.provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
    evaluate_provider_support_feasibility,
)

MANIFEST_ID = "1" * 64
COHORT_ID = "2" * 64
PREFIX_ID = "3" * 64
FEATURE_NAMES = ("overlap", "relative_variance")
FEATURE_CONTRACT = calibration_transport_feature_contract_id(
    FEATURE_NAMES,
    semantics="provider-prefix-admission-test-v1",
    configuration={"causal_prefix_only": True},
)


def _support(*, supported: bool = True):
    stream = ProviderSupportStreamV1(
        group_id="target-object",
        stream_id="camera-0",
        causal_frame_start=0,
        causal_frame_stop_exclusive=4,
        required_frame_ids=(0, 1, 2, 3),
        available_frame_ids=(0, 1, 2, 3),
        geometry_supported_frame_ids=(0, 1, 2, 3) if supported else (0,),
        minimum_geometry_support_fraction=1.0,
        intrinsics_required=False,
        intrinsics_id=None,
        extrinsics_required=False,
        extrinsics_id=None,
        metric_anchor_required=False,
        metric_anchor_id=None,
    )
    request = ProviderSupportFeasibilityRequestV1(
        protocol_id="prefix-admission-test-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        provider_family="test-provider",
        provider_repository="example/provider",
        provider_revision="b" * 40,
        model_set_id="4" * 64,
        loader_id="5" * 64,
        cohort_binding_id=COHORT_ID,
        promotion_lock_id="6" * 64,
        coordinate_semantics="metric-world-frame",
        admission_rule="all-streams",
        minimum_supported_fraction=1.0,
        permitted_technical_exclusion_codes=(),
        maximum_technical_exclusions=0,
        prediction_payloads_opened=False,
        residuals_used=False,
        target_outcomes_used=False,
        streams=(stream,),
    )
    return evaluate_provider_support_feasibility(request)


def _transport(*, shifted: bool = False, manifest_id: str = MANIFEST_ID):
    policy = CalibrationTransportPolicyV1(
        quantile_levels=(0.1, 0.5, 0.9),
        miscoverage_rate=0.2,
        minimum_source_units=6,
        neighbor_count=1,
        maximum_unsupported_group_fraction=0.0,
        maximum_unsupported_row_fraction=0.0,
        absolute_scale_floor=1e-6,
        relative_scale_floor=1e-6,
    )
    sources = tuple(
        CalibrationTransportUnitV1(
            unit_id=f"source-{index}",
            feature_contract_id=FEATURE_CONTRACT,
            feature_names=FEATURE_NAMES,
            feature_values=np.column_stack(
                (
                    np.linspace(-0.2, 0.2, 40) + 0.01 * index,
                    np.linspace(0.1, -0.1, 40) - 0.01 * index,
                )
            ),
        )
        for index in range(6)
    )
    model = fit_calibration_transport_model(sources, policy=policy)
    offset = 5.0 if shifted else 0.0
    target = CalibrationTransportUnitV1(
        unit_id="target-object",
        feature_contract_id=FEATURE_CONTRACT,
        feature_names=FEATURE_NAMES,
        feature_values=np.column_stack(
            (
                np.linspace(-0.2, 0.2, 40) + offset,
                np.linspace(0.1, -0.1, 40),
            )
        ),
    )
    evidence = evaluate_calibration_transport(
        model,
        [target],
        metadata={
            PROVIDER_PREFIX_BINDING_METADATA_KEY: {
                "provider_manifest_id": manifest_id,
                "cohort_binding_id": COHORT_ID,
                "target_prefix_id": PREFIX_ID,
                "causal_prefix_only": True,
                "target_residuals_used": False,
                "target_outcomes_used": False,
            }
        },
    )
    return evidence


def test_both_upstream_gates_must_pass() -> None:
    admitted = build_provider_prefix_admission(
        _support(),
        _transport(),
        provider_manifest_id=MANIFEST_ID,
        target_prefix_id=PREFIX_ID,
    )
    assert admitted.admitted
    assert not admitted.exact_fallback_required
    assert admitted.decision_reasons == ()

    transport_negative = build_provider_prefix_admission(
        _support(),
        _transport(shifted=True),
        provider_manifest_id=MANIFEST_ID,
        target_prefix_id=PREFIX_ID,
    )
    assert not transport_negative.admitted
    assert transport_negative.exact_fallback_required
    assert transport_negative.decision_reasons == ("calibration-transport-negative",)

    support_negative = build_provider_prefix_admission(
        _support(supported=False),
        _transport(),
        provider_manifest_id=MANIFEST_ID,
        target_prefix_id=PREFIX_ID,
    )
    assert not support_negative.admitted
    assert support_negative.decision_reasons == ("support-feasibility-negative",)


def test_binding_mismatch_is_invalid_not_a_scientific_negative() -> None:
    with pytest.raises(ValueError, match="provider_manifest_id binding changed"):
        build_provider_prefix_admission(
            _support(),
            _transport(manifest_id="9" * 64),
            provider_manifest_id=MANIFEST_ID,
            target_prefix_id=PREFIX_ID,
        )


def test_admission_round_trip_replays_fallback_decision(tmp_path: Path) -> None:
    admission = build_provider_prefix_admission(
        _support(),
        _transport(shifted=True),
        provider_manifest_id=MANIFEST_ID,
        target_prefix_id=PREFIX_ID,
    )
    path = tmp_path / "admission.json"
    write_provider_prefix_admission(path, admission)
    assert load_provider_prefix_admission(path).to_dict() == admission.to_dict()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["admitted"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch|derived fields changed"):
        load_provider_prefix_admission(path)
