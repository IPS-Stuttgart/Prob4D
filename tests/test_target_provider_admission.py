from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import prob4d.target_provider_admission as target_admission
from prob4d._deform360_cohort_schema import (
    BAYESIAN_PHYSTWIN_REPOSITORY,
    DEFORM360_SELECTION_PATH,
    Deform360CohortUnitV1,
)
from prob4d._heldout_promotion_common import PromotionArmV1
from prob4d._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    write_promotion_lock,
)
from prob4d.deform360_cohort_binding import (
    Deform360OfficialHubCohortBindingV1,
    write_deform360_cohort_binding,
)
from prob4d.prediction_provider_manifest import (
    PredictionFrameLineageV1,
    PredictionPayloadDescriptorV1,
    PredictionProviderManifestV1,
    save_prediction_provider_manifest,
)
from prob4d.target_provider_admission import (
    TARGET_PROVIDER_ADMISSION_CONFIG_SCHEMA,
    TARGET_PROVIDER_ADMISSION_CONFIG_VERSION,
    build_target_provider_admission,
    load_target_provider_admission,
    target_provider_admission_from_dict,
    validate_target_provider_admission_against_lock,
    verify_target_provider_admission,
    write_target_provider_admission,
)
from prob4d.target_provider_admission_cli import main_admit, main_verify

PROB4D_REVISION = "a" * 40
BPT_REVISION = "b" * 40
PROVIDER_REVISION = "c" * 40
MODEL_SET_ID = "d" * 64
LOADER_ID = "e" * 64
RUN_SPEC_ID = "f" * 64


def _unit(prefix: str, index: int, stratum: str) -> Deform360CohortUnitV1:
    object_id = f"{prefix}-{index:02d}"
    return Deform360CohortUnitV1(
        object_id=object_id,
        stratum=stratum,
        episode_id=index,
        metadata_path=f"raw/{object_id}/metadata.json",
        metadata_sha256=f"{index + 1:064x}",
    )


def _binding() -> Deform360OfficialHubCohortBindingV1:
    calibration = tuple(
        _unit("calibration", index, "sheet" if index < 5 else "volumetric") for index in range(10)
    )
    target = tuple(
        _unit("target", index, "sheet" if index < 6 else "volumetric") for index in range(12)
    )
    return Deform360OfficialHubCohortBindingV1(
        source_repository=BAYESIAN_PHYSTWIN_REPOSITORY,
        source_revision=BPT_REVISION,
        source_path=DEFORM360_SELECTION_PATH,
        selection_artifact_sha256="1" * 64,
        content_selection_sha256="2" * 64,
        selection_sha256="3" * 64,
        selection_implementation_revision="4" * 40,
        protocol_id="deform360-official-hub-visuotactile-v1",
        protocol_sha256="5" * 64,
        dataset_repository="brownu/deform360",
        dataset_requested_revision="main",
        dataset_resolved_revision="6" * 40,
        processing_repository="lhy0807/deform360",
        processing_revision="7" * 40,
        calibration_units=calibration,
        target_units=target,
    )


def _arms() -> tuple[PromotionArmV1, ...]:
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
    return tuple(
        PromotionArmV1(
            arm_id=arm_id,
            role=role,
            provider_method_id=provider_method,
            query_method_id=query_method,
            sensor_assisted=sensor_assisted,
            metadata={},
        )
        for arm_id, role, provider_method, query_method, sensor_assisted in values
    )


def _lock(binding: Deform360OfficialHubCohortBindingV1) -> HeldoutProviderPromotionLockV1:
    return HeldoutProviderPromotionLockV1(
        experiment_id="target-provider-admission-test-v1",
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision=PROB4D_REVISION,
        bayesian_phystwin_repository=binding.source_repository,
        bayesian_phystwin_revision=binding.source_revision,
        motioncrafter_revision=PROVIDER_REVISION,
        model_set_id=MODEL_SET_ID,
        prediction_run_spec_id=RUN_SPEC_ID,
        provider_evaluation_manifest_sha256="8" * 64,
        frozen_artifact_ids={
            "provider_configuration": "9" * 64,
            "gauge_calibration": "a" * 64,
            "point_calibration": "b" * 64,
            "source_reliability_calibration": "c" * 64,
            "material_identity_calibration": "d" * 64,
            "selection_lock": "e" * 64,
            "bayesian_guard_configuration": "f" * 64,
            "cohort_binding": binding.cohort_binding_id,
        },
        development_group_ids=("development-00",),
        calibration_group_ids=binding.calibration_group_ids,
        target_group_ids=binding.target_group_ids,
        arms=_arms(),
        provider_reference_arm_id="visual",
        primary_query_arm_id="identity",
        bootstrap_resamples=500,
        bootstrap_seed=17,
        minimum_target_group_count=12,
        query_superiority_margin_mm=0.25,
        harmful_update_margin_mm=0.0,
        maximum_harmful_accepted_updates=0,
        maximum_worst_group_regression_mm=0.0,
        maximum_technical_failures=0,
        minimum_mean_accepted_coverage=0.9,
        metadata={},
    )


def _payload(group_id: str, *, future: bool) -> PredictionPayloadDescriptorV1:
    start = 20 if future else 0
    stop = 30 if future else 8
    suffix = "future" if future else "causal"
    lineage = tuple(
        PredictionFrameLineageV1(
            output_frame_id=frame,
            source_frame_start=start,
            source_frame_stop_exclusive=stop,
            contributor_ids=(f"{group_id}-{suffix}-call",),
        )
        for frame in range(start, start + 2)
    )
    return PredictionPayloadDescriptorV1(
        product_role="independent-window",
        window_id=f"{group_id}-{suffix}",
        path=f"missing-{suffix}.npz",
        sha256=hashlib.sha256(f"{group_id}-{suffix}".encode()).hexdigest(),
        byte_count=123,
        view_id="camera-0",
        stochastic_member_id="member-0",
        dependence_group_ids=("shared-model",),
        dense_storage_dtype="float32",
        has_scene_flow=False,
        has_ray_directions=False,
        frame_lineage=lineage,
    )


def _write_manifest(
    root: Path,
    group_id: str,
    *,
    provider_revision: str = PROVIDER_REVISION,
    model_set_id: str = MODEL_SET_ID,
    loader_id: str = LOADER_ID,
    provider_repository: str = "example/provider",
) -> tuple[Path, str]:
    path = root / "manifests" / f"{group_id}.json"
    manifest = PredictionProviderManifestV1(
        sequence_id=f"{group_id}-episode",
        provider_family="external-4d-provider",
        provider_repository=provider_repository,
        provider_revision=provider_revision,
        provider_run_id=hashlib.sha256(f"run-{group_id}".encode()).hexdigest(),
        model_set_id=model_set_id,
        loader_id=loader_id,
        coordinate_semantics="sequence-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        payloads=(_payload(group_id, future=False), _payload(group_id, future=True)),
        metadata={},
    )
    save_prediction_provider_manifest(path, manifest)
    return path, manifest.sequence_id


def _config(root: Path, binding: Deform360OfficialHubCohortBindingV1) -> dict[str, object]:
    entries = []
    for group_id in binding.target_group_ids:
        path, sequence_id = _write_manifest(root, group_id)
        entries.append(
            {
                "group_id": group_id,
                "expected_sequence_id": sequence_id,
                "manifest_path": path.relative_to(root).as_posix(),
                "causal_frame_stop": 12,
            }
        )
    return {
        "schema_name": TARGET_PROVIDER_ADMISSION_CONFIG_SCHEMA,
        "schema_version": TARGET_PROVIDER_ADMISSION_CONFIG_VERSION,
        "prediction_run_spec_id": RUN_SPEC_ID,
        "target_outcomes_used": False,
        "entries": entries,
        "metadata": {"payload_access": "manifest-only"},
    }


def test_target_provider_admission_round_trip_and_replay(tmp_path: Path) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)

    admission = build_target_provider_admission(
        lock,
        binding,
        config,
        request_root=tmp_path,
    )
    path = tmp_path / "target-admission.json"
    write_target_provider_admission(admission, path)
    loaded = load_target_provider_admission(path)
    replayed = verify_target_provider_admission(
        loaded,
        lock,
        binding,
        config,
        request_root=tmp_path,
    )

    assert loaded == admission == replayed
    assert loaded.target_group_ids == binding.target_group_ids
    assert loaded.provider_revision == PROVIDER_REVISION
    assert loaded.model_set_id == MODEL_SET_ID
    assert loaded.target_outcomes_used is False
    assert all(len(entry.admitted_payloads) == 1 for entry in loaded.entries)
    assert all(
        entry.admitted_payloads[0].source_frame_stop_exclusive <= entry.causal_frame_stop
        for entry in loaded.entries
    )
    validate_target_provider_admission_against_lock(loaded, lock)


def test_admission_opens_no_dense_prediction_payload(tmp_path: Path) -> None:
    binding = _binding()
    admission = build_target_provider_admission(
        _lock(binding),
        binding,
        _config(tmp_path, binding),
        request_root=tmp_path,
    )

    assert len(admission.entries) == 12
    assert not any(tmp_path.rglob("*.npz"))


def test_manifest_is_parsed_from_a_private_exact_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)
    source_paths = {
        (tmp_path / "manifests" / f"{group_id}.json").resolve()
        for group_id in binding.target_group_ids
    }
    original = target_admission.load_prediction_provider_manifest
    parsed_paths: list[Path] = []

    def observing_loader(path: str | Path) -> PredictionProviderManifestV1:
        parsed_paths.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(target_admission, "load_prediction_provider_manifest", observing_loader)
    build_target_provider_admission(lock, binding, config, request_root=tmp_path)

    assert len(parsed_paths) == len(binding.target_group_ids)
    assert all(path not in source_paths for path in parsed_paths)
    assert all(path.name == "provider-manifest.json" for path in parsed_paths)


def test_manifest_mutation_during_admission_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)
    source = tmp_path / "manifests" / f"{binding.target_group_ids[0]}.json"
    original = target_admission.load_prediction_provider_manifest
    mutated = False

    def mutating_loader(path: str | Path) -> PredictionProviderManifestV1:
        nonlocal mutated
        manifest = original(path)
        if not mutated:
            source.write_bytes(source.read_bytes() + b"\n")
            mutated = True
        return manifest

    monkeypatch.setattr(target_admission, "load_prediction_provider_manifest", mutating_loader)
    with pytest.raises(ValueError, match="changed during target admission"):
        build_target_provider_admission(lock, binding, config, request_root=tmp_path)


def test_admission_rejects_target_use_missing_groups_and_sequence_drift(
    tmp_path: Path,
) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)

    target_use = copy.deepcopy(config)
    target_use["target_outcomes_used"] = True
    with pytest.raises(ValueError, match="cannot use target outcomes"):
        build_target_provider_admission(lock, binding, target_use, request_root=tmp_path)

    missing = copy.deepcopy(config)
    entries = missing["entries"]
    assert isinstance(entries, list)
    entries.pop()
    with pytest.raises(ValueError, match="exact frozen target groups"):
        build_target_provider_admission(lock, binding, missing, request_root=tmp_path)

    wrong_sequence = copy.deepcopy(config)
    entries = wrong_sequence["entries"]
    assert isinstance(entries, list)
    entries[0]["expected_sequence_id"] = "different-sequence"
    with pytest.raises(ValueError, match="provider sequence changed"):
        build_target_provider_admission(lock, binding, wrong_sequence, request_root=tmp_path)


def test_admission_rejects_provider_and_model_drift(tmp_path: Path) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)
    first_group = binding.target_group_ids[0]
    path = tmp_path / "manifests" / f"{first_group}.json"
    path.unlink()
    _write_manifest(tmp_path, first_group, provider_revision="0" * 40)
    with pytest.raises(ValueError, match="provider revision"):
        build_target_provider_admission(lock, binding, config, request_root=tmp_path)

    path.unlink()
    _write_manifest(tmp_path, first_group, model_set_id="0" * 64)
    with pytest.raises(ValueError, match="model set"):
        build_target_provider_admission(lock, binding, config, request_root=tmp_path)


def test_admission_rejects_cross_group_contract_drift(tmp_path: Path) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)
    last_group = binding.target_group_ids[-1]
    path = tmp_path / "manifests" / f"{last_group}.json"
    path.unlink()
    _write_manifest(tmp_path, last_group, loader_id="0" * 64)

    with pytest.raises(ValueError, match="contract drifts"):
        build_target_provider_admission(lock, binding, config, request_root=tmp_path)


def test_admission_tamper_and_no_clobber_fail_closed(tmp_path: Path) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)
    admission = build_target_provider_admission(
        lock,
        binding,
        config,
        request_root=tmp_path,
    )
    path = tmp_path / "target-admission.json"
    write_target_provider_admission(admission, path)
    write_target_provider_admission(admission, path)

    tampered = copy.deepcopy(admission.to_dict())
    tampered["provider_revision"] = "0" * 40
    with pytest.raises(ValueError, match="admission ID mismatch"):
        target_provider_admission_from_dict(tampered)

    other = copy.deepcopy(admission.to_dict())
    other["metadata"] = {"changed": True}
    other.pop("target_provider_admission_id")
    changed = target_provider_admission_from_dict(
        {
            **other,
            "target_provider_admission_id": hashlib.sha256(
                json.dumps(
                    other,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        }
    )
    with pytest.raises(FileExistsError):
        write_target_provider_admission(changed, path)


def test_installed_admit_and_verify_commands(tmp_path: Path) -> None:
    binding = _binding()
    lock = _lock(binding)
    config = _config(tmp_path, binding)
    lock_path = tmp_path / "promotion-lock.json"
    binding_path = tmp_path / "cohort-binding.json"
    config_path = tmp_path / "target-admission-config.json"
    admission_path = tmp_path / "target-admission.json"
    write_promotion_lock(lock, lock_path)
    write_deform360_cohort_binding(binding, binding_path)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert (
        main_admit(
            [
                str(lock_path),
                str(binding_path),
                str(config_path),
                "--output",
                str(admission_path),
            ]
        )
        == 0
    )
    assert (
        main_verify(
            [
                str(admission_path),
                str(lock_path),
                str(binding_path),
                str(config_path),
            ]
        )
        == 0
    )
