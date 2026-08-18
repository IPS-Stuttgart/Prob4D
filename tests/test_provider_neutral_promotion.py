from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from prob4d._heldout_promotion_lock import (
    HELDOUT_PROMOTION_LOCK_PROVIDER_NEUTRAL_VERSION,
    ProviderPromotionIdentityV1,
    promotion_lock_from_config,
    promotion_lock_from_dict,
)
from prob4d.heldout_promotion import (
    PromotionQueryRowV1,
    build_query_results,
    evaluate_heldout_promotion,
)
from prob4d.promotion_evidence import (
    PROMOTION_EVIDENCE_CARD_PROVIDER_NEUTRAL_VERSION,
    build_promotion_evidence_card,
    promotion_evidence_card_from_dict,
    render_promotion_evidence_markdown,
)
from prob4d.target_provider_admission import (
    AdmittedTargetPayloadV1,
    HeldoutTargetProviderAdmissionV1,
    TargetProviderManifestAdmissionV1,
    validate_target_provider_admission_against_lock,
)

_ROLES = (
    ("fallback", "physical_fallback", None, "bpt-fallback", False),
    (
        "framewise",
        "framewise_explicit_joint_gauge",
        "provider-framewise",
        "bpt-framewise",
        False,
    ),
    (
        "identity",
        "cross_window_identity_marginalized",
        "provider-identity",
        "bpt-identity",
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
        "rowwise",
        "rowwise_gauge_marginalized",
        "provider-rowwise",
        "bpt-rowwise",
        False,
    ),
    ("sensor", "sensor_assisted", "provider-sensor", "bpt-sensor", True),
    ("visual", "visual_baseline", "provider-visual", "bpt-visual", False),
)


def _arms() -> list[dict[str, object]]:
    return [
        {
            "arm_id": arm_id,
            "role": role,
            "provider_method_id": provider_method,
            "query_method_id": query_method,
            "sensor_assisted": sensor_assisted,
            "metadata": {},
        }
        for arm_id, role, provider_method, query_method, sensor_assisted in _ROLES
    ]


def _common_config() -> dict[str, Any]:
    return {
        "experiment_id": "provider-neutral-heldout-v2",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "a" * 40,
        "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "bayesian_phystwin_revision": "b" * 40,
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
            "cohort_binding": "7" * 64,
        },
        "development_group_ids": ["dev-1"],
        "calibration_group_ids": ["cal-1"],
        "target_group_ids": ["target-1", "target-2"],
        "arms": _arms(),
        "provider_reference_arm_id": "visual",
        "primary_query_arm_id": "identity",
        "bootstrap_resamples": 500,
        "bootstrap_seed": 11,
        "minimum_target_group_count": 2,
        "query_superiority_margin_mm": 0.1,
        "harmful_update_margin_mm": 0.0,
        "maximum_harmful_accepted_updates": 0,
        "maximum_worst_group_regression_mm": 0.0,
        "maximum_technical_failures": 0,
        "minimum_mean_accepted_coverage": 0.9,
        "metadata": {"split_semantics": "complete-object-session-v1"},
    }


def _provider_identity() -> dict[str, object]:
    return {
        "schema_name": "prob4d.provider-promotion-identity",
        "schema_version": 1,
        "provider_family": "cut3r",
        "provider_repository": "naver/CUT3R",
        "provider_revision": "c" * 40,
        "model_set_id": "d" * 64,
        "loader_id": "8" * 64,
        "coordinate_semantics": "sequence-local-sim3",
        "point_semantics": "dense-point-map",
        "flow_semantics": "absent",
        "ray_semantics": "absent",
        "source_dependency_semantics": (
            "per-output-exclusive-source-frame-interval-v1"
        ),
    }


def _v1_config() -> dict[str, Any]:
    config = _common_config()
    config["motioncrafter_revision"] = "c" * 40
    config["model_set_id"] = "d" * 64
    return config


def _v2_config() -> dict[str, Any]:
    config = _common_config()
    config["provider_identity"] = _provider_identity()
    return config


def _provider_report(lock: Any) -> dict[str, Any]:
    return {
        "schema_name": "prob4d.provider-evaluation-report",
        "schema_version": 3,
        "source_manifest_sha256": lock.provider_evaluation_manifest_sha256,
        "primary_mode": "metric",
        "primary_support": "common_across_registered_methods",
        "reference_method": lock.provider_reference_method_id,
        "bootstrap_resamples": lock.bootstrap_resamples,
        "bootstrap_seed": lock.bootstrap_seed,
        "legacy_artifacts_allowed": False,
        "cases": [
            {
                "case_id": f"{group_id}-case",
                "group_id": group_id,
                "method_id": method_id,
            }
            for group_id in lock.target_group_ids
            for method_id in lock.provider_method_ids
        ],
        "decision": {
            "policy_id": "provider-v2",
            "overall_passed": True,
            "rules": [],
        },
    }


def _query_rows(lock: Any) -> tuple[PromotionQueryRowV1, ...]:
    rows: list[PromotionQueryRowV1] = []
    for group_id in lock.target_group_ids:
        fallback_id = hashlib.sha256(f"fallback-{group_id}".encode()).hexdigest()
        rows.append(
            PromotionQueryRowV1(
                group_id=group_id,
                arm_id="fallback",
                query_rmse_mm=5.0,
                deployed_artifact_id=fallback_id,
                fallback_artifact_id=fallback_id,
                accepted=None,
                exact_fallback_reproduced=None,
                accepted_coverage=None,
                accepted_width_mm=None,
            )
        )
        for arm in lock.arms:
            if arm.role == "physical_fallback":
                continue
            rows.append(
                PromotionQueryRowV1(
                    group_id=group_id,
                    arm_id=arm.arm_id,
                    query_rmse_mm=4.0,
                    deployed_artifact_id=hashlib.sha256(
                        f"deployed-{group_id}-{arm.arm_id}".encode()
                    ).hexdigest(),
                    fallback_artifact_id=fallback_id,
                    accepted=True,
                    exact_fallback_reproduced=None,
                    accepted_coverage=0.94,
                    accepted_width_mm=1.2,
                )
            )
    return tuple(rows)


def _admission(lock: Any, *, loader_id: str = "8" * 64) -> HeldoutTargetProviderAdmissionV1:
    payload = AdmittedTargetPayloadV1(
        payload_id="9" * 64,
        window_id="window-0",
        output_frame_ids=(0,),
        source_frame_start=0,
        source_frame_stop_exclusive=1,
        dependence_group_ids=("shared-model",),
    )
    entries = tuple(
        TargetProviderManifestAdmissionV1(
            group_id=group_id,
            episode_id=index,
            stratum="sheet",
            sequence_id=f"{group_id}-sequence",
            manifest_sha256=f"{index + 10:064x}",
            manifest_artifact_id=f"{index + 20:064x}",
            provider_run_id=f"{index + 30:064x}",
            causal_frame_stop=2,
            admitted_payloads=(payload,),
        )
        for index, group_id in enumerate(lock.target_group_ids)
    )
    return HeldoutTargetProviderAdmissionV1(
        promotion_lock_id=lock.promotion_lock_id,
        cohort_binding_id="7" * 64,
        source_repository=lock.source_repository,
        source_revision=lock.source_revision,
        prediction_run_spec_id=lock.prediction_run_spec_id,
        provider_family="cut3r",
        provider_repository="naver/CUT3R",
        provider_revision="c" * 40,
        model_set_id="d" * 64,
        loader_id=loader_id,
        coordinate_semantics="sequence-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        source_dependency_semantics=(
            "per-output-exclusive-source-frame-interval-v1"
        ),
        target_outcomes_used=False,
        entries=entries,
        metadata={},
    )


def test_v1_lock_descriptor_and_identity_remain_legacy() -> None:
    lock = promotion_lock_from_config(_v1_config())
    encoded = lock.to_dict()

    assert encoded["schema_version"] == 1
    assert encoded["motioncrafter_revision"] == "c" * 40
    assert encoded["model_set_id"] == "d" * 64
    assert "provider_identity" not in encoded
    assert promotion_lock_from_dict(encoded) == lock


def test_provider_neutral_v2_lock_round_trip() -> None:
    lock = promotion_lock_from_config(_v2_config())
    encoded = lock.to_dict()

    assert lock.schema_version == HELDOUT_PROMOTION_LOCK_PROVIDER_NEUTRAL_VERSION
    assert isinstance(lock.provider_identity, ProviderPromotionIdentityV1)
    assert encoded["schema_version"] == 2
    assert encoded["provider_identity"] == _provider_identity()
    assert "motioncrafter_revision" not in encoded
    assert "model_set_id" not in encoded
    assert promotion_lock_from_dict(encoded) == lock


def test_v2_target_admission_binds_complete_provider_contract() -> None:
    lock = promotion_lock_from_config(_v2_config())
    validate_target_provider_admission_against_lock(_admission(lock), lock)

    with pytest.raises(ValueError, match="loader_id"):
        validate_target_provider_admission_against_lock(
            _admission(lock, loader_id="0" * 64),
            lock,
        )


def test_provider_neutral_evidence_card_uses_provider_identity() -> None:
    lock = promotion_lock_from_config(_v2_config())
    query_results = build_query_results(lock, rows=_query_rows(lock))
    provider_report = _provider_report(lock)
    provider_bytes = json.dumps(provider_report, sort_keys=True).encode()
    report = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256=hashlib.sha256(provider_bytes).hexdigest(),
    )

    card = build_promotion_evidence_card(lock.to_dict(), report.to_dict())

    assert card["schema_version"] == (
        PROMOTION_EVIDENCE_CARD_PROVIDER_NEUTRAL_VERSION
    )
    assert card["repositories"]["provider"] == _provider_identity()
    assert "motioncrafter" not in card["repositories"]
    assert promotion_evidence_card_from_dict(card) == card
    assert "Provider: `cut3r`" in render_promotion_evidence_markdown(card)


def test_v2_rejects_mixed_legacy_and_provider_neutral_fields() -> None:
    config = _v2_config()
    config["motioncrafter_revision"] = "c" * 40

    with pytest.raises(ValueError, match="fields"):
        promotion_lock_from_config(config)
