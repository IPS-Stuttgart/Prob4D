from __future__ import annotations

import copy

import pytest

from prob4d._heldout_promotion_lock import (
    HELDOUT_PROMOTION_LOCK_V2_VERSION,
    HeldoutProviderPromotionLockV1,
    HeldoutProviderPromotionLockV2,
    promotion_lock_from_config,
    promotion_lock_from_dict,
)


def _arms() -> list[dict[str, object]]:
    values = (
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
    return [
        {
            "arm_id": arm_id,
            "role": role,
            "provider_method_id": provider_method,
            "query_method_id": query_method,
            "sensor_assisted": sensor_assisted,
            "metadata": {},
        }
        for arm_id, role, provider_method, query_method, sensor_assisted in values
    ]


def _provider_identity() -> dict[str, object]:
    return {
        "schema_name": "prob4d.heldout-provider-promotion-identity",
        "schema_version": 1,
        "provider_family": "cut3r",
        "provider_repository": "naver/CUT3R",
        "provider_revision": "c" * 40,
        "model_set_id": "d" * 64,
        "loader_id": "7" * 64,
        "coordinate_semantics": "sequence-local-sim3",
        "point_semantics": "dense-point-map",
        "flow_semantics": "absent",
        "ray_semantics": "absent",
        "source_dependency_semantics": "per-output-exclusive-source-frame-interval-v1",
    }


def _config_v2() -> dict[str, object]:
    return {
        "experiment_id": "generic-provider-gate-v2",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "a" * 40,
        "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "bayesian_phystwin_revision": "b" * 40,
        "provider_identity": _provider_identity(),
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
        "calibration_group_ids": ["calibration-1"],
        "target_group_ids": ["target-1", "target-2", "target-3"],
        "arms": _arms(),
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
        "metadata": {"split_semantics": "complete-object-session-v1"},
    }


def _config_v1() -> dict[str, object]:
    config = _config_v2()
    identity = config.pop("provider_identity")
    assert isinstance(identity, dict)
    config["motioncrafter_revision"] = identity["provider_revision"]
    config["model_set_id"] = identity["model_set_id"]
    return config


def test_v2_lock_round_trips_without_provider_specific_aliases() -> None:
    lock = promotion_lock_from_config(_config_v2())

    assert isinstance(lock, HeldoutProviderPromotionLockV2)
    assert lock.provider_contract.provider_family == "cut3r"
    assert lock.provider_contract.provider_repository == "naver/CUT3R"
    record = lock.to_dict()
    assert record["schema_version"] == HELDOUT_PROMOTION_LOCK_V2_VERSION
    assert "provider_identity" in record
    assert "motioncrafter_revision" not in record
    assert "model_set_id" not in record
    assert promotion_lock_from_dict(record) == lock


def test_v1_motioncrafter_lock_remains_exactly_replayable() -> None:
    lock = promotion_lock_from_config(_config_v1())

    assert type(lock) is HeldoutProviderPromotionLockV1
    assert lock.to_dict()["schema_version"] == 1
    assert lock.to_dict()["motioncrafter_revision"] == "c" * 40
    assert promotion_lock_from_dict(lock.to_dict()) == lock


def test_v2_rejects_mixed_or_unknown_provider_identity_fields() -> None:
    mixed = _config_v2()
    mixed["motioncrafter_revision"] = "c" * 40
    mixed["model_set_id"] = "d" * 64
    with pytest.raises(ValueError, match="cannot be mixed"):
        promotion_lock_from_config(mixed)

    unknown = _config_v2()
    identity = unknown["provider_identity"]
    assert isinstance(identity, dict)
    identity["unexpected"] = "field"
    with pytest.raises(ValueError, match="provider promotion identity"):
        promotion_lock_from_config(unknown)


def test_every_variable_provider_contract_field_changes_the_lock_identity() -> None:
    baseline = promotion_lock_from_config(_config_v2())
    # Point and source-dependency semantics currently each have one registered value.
    fields = (
        ("provider_family", "vggt"),
        ("provider_repository", "facebookresearch/vggt"),
        ("provider_revision", "8" * 40),
        ("model_set_id", "8" * 64),
        ("loader_id", "8" * 64),
        ("coordinate_semantics", "metric-world"),
        ("flow_semantics", "forward-point-displacement"),
        ("ray_semantics", "camera-ray-unit-vector"),
    )
    for field_name, replacement in fields:
        changed = copy.deepcopy(_config_v2())
        identity = changed["provider_identity"]
        assert isinstance(identity, dict)
        identity[field_name] = replacement
        candidate = promotion_lock_from_config(changed)
        assert candidate.promotion_lock_id != baseline.promotion_lock_id
