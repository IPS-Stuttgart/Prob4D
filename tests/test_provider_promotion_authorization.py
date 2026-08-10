from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prob4d.heldout_promotion import (
    HeldoutProviderPromotionLockV1,
    promotion_lock_from_config,
    write_promotion_lock,
)
from prob4d.provider_promotion_authorization import (
    ProviderPromotionAuthorizationV2,
    authorize_provider_promotion,
    load_provider_promotion_authorization,
    main,
    write_provider_promotion_authorization,
)
from prob4d.provider_support_feasibility import (
    ProviderSupportFeasibilityRequestV1,
    ProviderSupportStreamV1,
    evaluate_provider_support_feasibility,
    write_provider_support_feasibility,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _lock() -> HeldoutProviderPromotionLockV1:
    roles = (
        ("fallback", "physical_fallback", None, "bpt-fallback", False),
        ("visual", "visual_baseline", "provider-visual", "bpt-visual", False),
        (
            "rowwise",
            "rowwise_gauge_marginalized",
            "provider-rowwise",
            "bpt-rowwise",
            False,
        ),
        (
            "framewise",
            "framewise_explicit_joint_gauge",
            "provider-framewise",
            "bpt-framewise",
            False,
        ),
        (
            "persistent",
            "persistent_explicit_joint_gauge",
            "provider-persistent",
            "bpt-persistent",
            False,
        ),
        (
            "identity",
            "cross_window_identity_marginalized",
            "provider-identity",
            "bpt-identity",
            False,
        ),
        ("sensor", "sensor_assisted", "provider-sensor", "bpt-sensor", True),
    )
    return promotion_lock_from_config(
        {
            "experiment_id": "support-authorization-test-v2",
            "source_repository": "IPS-Stuttgart/Prob4D",
            "source_revision": "a" * 40,
            "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
            "bayesian_phystwin_revision": "b" * 40,
            "motioncrafter_revision": "c" * 40,
            "model_set_id": "d" * 64,
            "prediction_run_spec_id": "e" * 64,
            "provider_evaluation_manifest_sha256": "f" * 64,
            "frozen_artifact_ids": {
                "provider_configuration": "0" * 64,
                "gauge_calibration": "1" * 64,
                "point_calibration": "2" * 64,
                "source_reliability_calibration": "3" * 64,
                "material_identity_calibration": "4" * 64,
                "selection_lock": "5" * 64,
                "bayesian_guard_configuration": "6" * 64,
            },
            "development_group_ids": ["development-1"],
            "calibration_group_ids": ["calibration-1", "calibration-2"],
            "target_group_ids": ["target-1", "target-2", "target-3"],
            "arms": [
                {
                    "arm_id": arm_id,
                    "role": role,
                    "query_method_id": query_method,
                    "provider_method_id": provider_method,
                    "sensor_assisted": sensor_assisted,
                    "metadata": {},
                }
                for (
                    arm_id,
                    role,
                    provider_method,
                    query_method,
                    sensor_assisted,
                ) in roles
            ],
            "provider_reference_arm_id": "visual",
            "primary_query_arm_id": "identity",
            "bootstrap_resamples": 500,
            "bootstrap_seed": 17,
            "minimum_target_group_count": 3,
            "query_superiority_margin_mm": 0.25,
            "harmful_update_margin_mm": 0.0,
            "maximum_harmful_accepted_updates": 0,
            "maximum_worst_group_regression_mm": 0.0,
            "maximum_technical_failures": 0,
            "minimum_mean_accepted_coverage": 0.9,
            "metadata": {},
        }
    )


def _stream(group_id: str, *, supported: bool = True) -> ProviderSupportStreamV1:
    frames = (0, 1, 2, 3) if supported else (0,)
    return ProviderSupportStreamV1(
        group_id=group_id,
        stream_id="camera-0",
        causal_frame_start=0,
        causal_frame_stop_exclusive=4,
        required_frame_ids=(0, 1, 2, 3),
        available_frame_ids=(0, 1, 2, 3),
        geometry_supported_frame_ids=frames,
        minimum_geometry_support_fraction=1.0,
        intrinsics_required=True,
        intrinsics_id=_digest(f"intrinsics:{group_id}"),
        extrinsics_required=True,
        extrinsics_id=_digest(f"extrinsics:{group_id}"),
        metric_anchor_required=True,
        metric_anchor_id=_digest(f"anchor:{group_id}"),
    )


def _request(
    lock: HeldoutProviderPromotionLockV1,
    *,
    supported_groups: tuple[str, ...] | None = None,
    minimum_fraction: float = 1.0,
) -> ProviderSupportFeasibilityRequestV1:
    supported = set(supported_groups or lock.target_group_ids)
    return ProviderSupportFeasibilityRequestV1(
        protocol_id="support-authorization-test-v2",
        source_repository=lock.source_repository,
        source_revision=lock.source_revision,
        provider_family="external-4d-provider",
        provider_repository="example/provider",
        provider_revision="9" * 40,
        model_set_id=lock.model_set_id,
        loader_id="7" * 64,
        cohort_binding_id="8" * 64,
        promotion_lock_id=lock.promotion_lock_id,
        coordinate_semantics="metric-world-frame",
        admission_rule=(
            "all-streams" if minimum_fraction == 1.0 else "minimum-stream-fraction"
        ),
        minimum_supported_fraction=minimum_fraction,
        permitted_technical_exclusion_codes=(),
        maximum_technical_exclusions=0,
        prediction_payloads_opened=False,
        residuals_used=False,
        target_outcomes_used=False,
        streams=tuple(
            _stream(group, supported=group in supported)
            for group in lock.target_group_ids
        ),
    )


def test_exact_positive_support_creates_content_addressed_authorization() -> None:
    lock = _lock()
    support = evaluate_provider_support_feasibility(_request(lock))
    result = authorize_provider_promotion(lock, support)

    assert isinstance(result, ProviderPromotionAuthorizationV2)
    assert result.authorized
    assert result.target_group_ids == lock.target_group_ids
    assert result.supported_target_group_ids == lock.target_group_ids
    assert result.promotion_lock_id == lock.promotion_lock_id
    assert result.support_feasibility_id == support.provider_support_feasibility_id
    assert len(result.stream_roster_id) == 64
    assert len(result.technical_exclusion_policy_id) == 64
    assert len(result.authorization_id) == 64


def test_negative_or_group_incomplete_support_fails_closed() -> None:
    lock = _lock()
    negative = evaluate_provider_support_feasibility(
        _request(lock, supported_groups=("target-1", "target-2"))
    )
    with pytest.raises(PermissionError, match="not authorized"):
        authorize_provider_promotion(lock, negative)

    aggregate_pass = evaluate_provider_support_feasibility(
        _request(
            lock,
            supported_groups=("target-1", "target-2"),
            minimum_fraction=2.0 / 3.0,
        )
    )
    assert aggregate_pass.support_feasible
    with pytest.raises(PermissionError, match="target-3"):
        authorize_provider_promotion(lock, aggregate_pass)

    support = evaluate_provider_support_feasibility(_request(lock))
    with pytest.raises(ValueError, match="must be false"):
        authorize_provider_promotion(
            lock,
            support,
            target_payloads_opened=True,
        )


def test_round_trip_cli_no_clobber_and_tamper_detection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock = _lock()
    support = evaluate_provider_support_feasibility(_request(lock))
    lock_path = tmp_path / "lock.json"
    support_path = tmp_path / "support.json"
    output = tmp_path / "authorization.json"
    write_promotion_lock(lock, lock_path)
    write_provider_support_feasibility(support_path, support)

    assert main(
        [
            "authorize",
            "--promotion-lock",
            str(lock_path),
            "--support-feasibility",
            str(support_path),
            "--output",
            str(output),
            "--compact",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["authorized"] is True
    assert summary["promotion_lock_id"] == lock.promotion_lock_id

    loaded = load_provider_promotion_authorization(output)
    assert loaded.authorization_id == summary["authorization_id"]
    with pytest.raises(FileExistsError):
        write_provider_promotion_authorization(output, loaded)

    assert main(["verify", "--artifact", str(output), "--compact"]) == 0
    assert json.loads(capsys.readouterr().out) == summary

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["stream_roster_id"] = "f" * 64
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stream_roster_id changed"):
        load_provider_promotion_authorization(output)
