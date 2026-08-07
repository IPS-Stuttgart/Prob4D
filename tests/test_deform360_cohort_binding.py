from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from prob4d._selection_evidence_common import _sha256_json
from prob4d.deform360_cohort_binding import (
    BAYESIAN_PHYSTWIN_REPOSITORY,
    DEFORM360_SELECTION_PATH,
    Deform360OfficialHubCohortBindingV1,
    build_deform360_official_hub_cohort_binding,
    deform360_cohort_binding_from_dict,
    load_deform360_cohort_binding,
    validate_deform360_cohort_binding_against_selection,
    validate_deform360_official_hub_selection,
    validate_promotion_config_against_deform360_binding,
    write_deform360_cohort_binding,
)
from prob4d.heldout_promotion import load_promotion_lock, main

BPT_REVISION = "a" * 40


def _unit(role: str, index: int, stratum: str) -> dict[str, object]:
    object_id = f"{role}-{index:02d}"
    return {
        "object_id": object_id,
        "stratum": stratum,
        "episode_id": index,
        "metadata_path": f"raw/{object_id}/metadata.json",
        "metadata_sha256": f"{index + 1:064x}",
    }


def _rehash_selection(value: dict[str, object]) -> None:
    selection = value["selection"]
    assert isinstance(selection, dict)
    value["selection_sha256"] = _sha256_json(selection)
    content = copy.deepcopy(value)
    content.pop("content_selection_sha256", None)
    content.pop("implementation_revision", None)
    content.pop("selection_artifact_sha256", None)
    value["content_selection_sha256"] = _sha256_json(content)
    artifact = copy.deepcopy(value)
    artifact.pop("selection_artifact_sha256", None)
    value["selection_artifact_sha256"] = _sha256_json(artifact)


def _selection() -> dict[str, object]:
    calibration = [
        _unit("calibration", index, "sheet" if index < 5 else "volumetric") for index in range(10)
    ]
    confirmation = [
        _unit("confirmation", index, "sheet" if index < 6 else "volumetric") for index in range(12)
    ]
    value: dict[str, object] = {
        "available_raw_object_count": 192,
        "cache_preflight": {
            "inventory_sha256": "1" * 64,
            "content_inventory_sha256": "2" * 64,
        },
        "dataset": {
            "repo_id": "brownu/deform360",
            "requested_revision": "main",
            "resolved_revision": "b" * 40,
            "raw_prefix": "raw",
        },
        "excluded_object_count": 92,
        "implementation_revision": "c" * 40,
        "information_boundary": {
            "object_directory_names_opened": True,
            "object_metadata_json_opened": True,
            "opened_metadata_paths": sorted(
                str(unit["metadata_path"]) for unit in (*calibration, *confirmation)
            ),
            "camera_media_opened": False,
            "tactile_arrays_opened": False,
            "robot_arrays_opened": False,
            "geometry_annotations_opened": False,
            "target_outcomes_opened": False,
        },
        "next_gate": "freeze the exact selection before payload access",
        "official_processing": {
            "future_processing_revision_change_requires_new_protocol": True,
            "repository": "lhy0807/deform360",
            "required_stages": ["download-selected-object", "depth", "tracking"],
            "revision": "d" * 40,
        },
        "prior_protocols": {
            "v1": {"config_sha256": "3" * 64},
            "v2": {"config_sha256": "4" * 64},
        },
        "protocol_id": "deform360-official-hub-visuotactile-v1",
        "protocol_sha256": "5" * 64,
        "replacement_allowed_after_payload_access": False,
        "schema": "bayesian-phystwin/deform360-official-hub-selection-v1",
        "schema_version": 1,
        "selection": {
            "calibration": calibration,
            "confirmation": confirmation,
        },
    }
    _rehash_selection(value)
    return value


def _binding() -> Deform360OfficialHubCohortBindingV1:
    return build_deform360_official_hub_cohort_binding(
        _selection(),
        source_repository=BAYESIAN_PHYSTWIN_REPOSITORY,
        source_revision=BPT_REVISION,
        source_path=DEFORM360_SELECTION_PATH,
    )


def _arms() -> list[dict[str, object]]:
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
    return [
        {
            "arm_id": arm_id,
            "role": role,
            "query_method_id": query_method,
            "provider_method_id": provider_method,
            "sensor_assisted": sensor_assisted,
            "metadata": {},
        }
        for arm_id, role, provider_method, query_method, sensor_assisted in roles
    ]


def _promotion_config(
    binding: Deform360OfficialHubCohortBindingV1,
) -> dict[str, object]:
    return {
        "experiment_id": "deform360-real-provider-v1",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "e" * 40,
        "bayesian_phystwin_repository": binding.source_repository,
        "bayesian_phystwin_revision": binding.source_revision,
        "motioncrafter_revision": "f" * 40,
        "model_set_id": "0" * 64,
        "prediction_run_spec_id": "1" * 64,
        "provider_evaluation_manifest_sha256": "2" * 64,
        "frozen_artifact_ids": {
            "provider_configuration": "3" * 64,
            "gauge_calibration": "4" * 64,
            "point_calibration": "5" * 64,
            "source_reliability_calibration": "6" * 64,
            "material_identity_calibration": "7" * 64,
            "selection_lock": "8" * 64,
            "bayesian_guard_configuration": "9" * 64,
            "cohort_binding": binding.cohort_binding_id,
        },
        "development_group_ids": ["development-object-01"],
        "calibration_group_ids": list(binding.calibration_group_ids),
        "target_group_ids": list(binding.target_group_ids),
        "arms": _arms(),
        "provider_reference_arm_id": "visual",
        "primary_query_arm_id": "identity",
        "bootstrap_resamples": 500,
        "bootstrap_seed": 17,
        "minimum_target_group_count": len(binding.target_group_ids),
        "query_superiority_margin_mm": 0.25,
        "harmful_update_margin_mm": 0.0,
        "maximum_harmful_accepted_updates": 0,
        "maximum_worst_group_regression_mm": 0.0,
        "maximum_technical_failures": 0,
        "minimum_mean_accepted_coverage": 0.9,
        "metadata": {"cohort_protocol_id": binding.protocol_id},
    }


def test_selection_and_binding_round_trip(tmp_path: Path) -> None:
    selection = _selection()
    validated = validate_deform360_official_hub_selection(selection)
    binding = _binding()
    path = tmp_path / "cohort-binding.json"

    write_deform360_cohort_binding(binding, path)
    loaded = load_deform360_cohort_binding(path)
    validate_deform360_cohort_binding_against_selection(loaded, selection)

    assert len(validated["calibration_units"]) == 10
    assert len(validated["target_units"]) == 12
    assert loaded == binding
    assert loaded.calibration_group_ids == tuple(sorted(loaded.calibration_group_ids))
    assert loaded.target_group_ids == tuple(sorted(loaded.target_group_ids))
    assert loaded.selection_artifact_sha256 == selection["selection_artifact_sha256"]


def test_selection_hashes_and_information_boundary_fail_closed() -> None:
    selection = _selection()
    split = selection["selection"]
    assert isinstance(split, dict)
    calibration = split["calibration"]
    assert isinstance(calibration, list)
    calibration[0]["episode_id"] = 99
    with pytest.raises(ValueError, match="selection_sha256"):
        validate_deform360_official_hub_selection(selection)

    boundary_tamper = _selection()
    boundary = boundary_tamper["information_boundary"]
    assert isinstance(boundary, dict)
    boundary["target_outcomes_opened"] = True
    _rehash_selection(boundary_tamper)
    with pytest.raises(ValueError, match="target_outcomes_opened"):
        validate_deform360_official_hub_selection(boundary_tamper)


def test_selection_rejects_changed_strata_counts_and_object_overlap() -> None:
    wrong_strata = _selection()
    split = wrong_strata["selection"]
    assert isinstance(split, dict)
    calibration = split["calibration"]
    assert isinstance(calibration, list)
    calibration[0]["stratum"] = "volumetric"
    _rehash_selection(wrong_strata)
    with pytest.raises(ValueError, match="exactly 5 sheet"):
        validate_deform360_official_hub_selection(wrong_strata)

    overlap = _selection()
    split = overlap["selection"]
    assert isinstance(split, dict)
    calibration = split["calibration"]
    confirmation = split["confirmation"]
    assert isinstance(calibration, list) and isinstance(confirmation, list)
    confirmation[0]["object_id"] = calibration[0]["object_id"]
    confirmation[0]["metadata_path"] = calibration[0]["metadata_path"]
    _rehash_selection(overlap)
    with pytest.raises(ValueError, match="disjoint"):
        validate_deform360_official_hub_selection(overlap)


def test_binding_rejects_tampering() -> None:
    binding = _binding()
    tampered = copy.deepcopy(binding.to_dict())
    tampered["target_group_ids"] = list(binding.calibration_group_ids)
    with pytest.raises(ValueError, match="target_group_ids"):
        deform360_cohort_binding_from_dict(tampered)

    source_tamper = copy.deepcopy(binding.to_dict())
    source_tamper["source_revision"] = "f" * 40
    with pytest.raises(ValueError, match="cohort_binding_id"):
        deform360_cohort_binding_from_dict(source_tamper)


def test_promotion_configuration_requires_exact_binding() -> None:
    binding = _binding()
    config = _promotion_config(binding)
    validate_promotion_config_against_deform360_binding(config, binding)

    changed_target = copy.deepcopy(config)
    changed_target["target_group_ids"] = list(reversed(binding.target_group_ids))
    with pytest.raises(ValueError, match="target_group_ids"):
        validate_promotion_config_against_deform360_binding(changed_target, binding)

    wrong_revision = copy.deepcopy(config)
    wrong_revision["bayesian_phystwin_revision"] = "0" * 40
    with pytest.raises(ValueError, match="bayesian_phystwin_revision"):
        validate_promotion_config_against_deform360_binding(wrong_revision, binding)

    weak_count = copy.deepcopy(config)
    weak_count["minimum_target_group_count"] = 3
    with pytest.raises(ValueError, match="complete bound confirmation"):
        validate_promotion_config_against_deform360_binding(weak_count, binding)


def test_grouped_bind_verify_and_promotion_freeze(tmp_path: Path) -> None:
    selection_path = tmp_path / "selection.json"
    binding_path = tmp_path / "cohort-binding.json"
    promotion_config_path = tmp_path / "promotion-config.json"
    promotion_lock_path = tmp_path / "promotion-lock.json"
    selection_path.write_text(json.dumps(_selection()), encoding="utf-8")

    assert (
        main(
            [
                "cohort-bind",
                str(selection_path),
                "--source-revision",
                BPT_REVISION,
                "--output",
                str(binding_path),
            ]
        )
        == 0
    )
    binding = load_deform360_cohort_binding(binding_path)
    assert (
        main(
            [
                "cohort-verify",
                str(binding_path),
                "--selection",
                str(selection_path),
            ]
        )
        == 0
    )
    promotion_config_path.write_text(
        json.dumps(_promotion_config(binding)),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "freeze",
                str(promotion_config_path),
                "--cohort-binding",
                str(binding_path),
                "--output",
                str(promotion_lock_path),
            ]
        )
        == 0
    )
    promotion_lock = load_promotion_lock(promotion_lock_path)
    assert promotion_lock.calibration_group_ids == binding.calibration_group_ids
    assert promotion_lock.target_group_ids == binding.target_group_ids
    assert promotion_lock.frozen_artifact_ids["cohort_binding"] == (binding.cohort_binding_id)
