from __future__ import annotations

from pathlib import Path

import pytest

from prob4d._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    promotion_lock_from_config,
)
from prob4d.target_admission_enforcement import (
    TARGET_PROVIDER_ADMISSION_METADATA_KEY,
    load_target_admission_for_execution,
    validate_target_admission_execution_binding,
)
from prob4d.target_provider_admission import (
    AdmittedTargetPayloadV1,
    HeldoutTargetProviderAdmissionV1,
    TargetProviderManifestAdmissionV1,
    write_target_provider_admission,
)

SOURCE_REVISION = "a" * 40
BPT_REVISION = "b" * 40
PROVIDER_REVISION = "c" * 40
MODEL_SET_ID = "d" * 64
RUN_SPEC_ID = "e" * 64
COHORT_BINDING_ID = "f" * 64
LOADER_ID = "1" * 64


def _lock(
    *,
    cohort_bound: bool = True,
    source_revision: str = SOURCE_REVISION,
) -> HeldoutProviderPromotionLockV1:
    frozen_artifact_ids = {
        "provider_configuration": "0" * 64,
        "gauge_calibration": "1" * 64,
        "point_calibration": "2" * 64,
        "source_reliability_calibration": "3" * 64,
        "material_identity_calibration": "4" * 64,
        "selection_lock": "5" * 64,
        "bayesian_guard_configuration": "6" * 64,
    }
    if cohort_bound:
        frozen_artifact_ids["cohort_binding"] = COHORT_BINDING_ID
    roles = {
        "physical_fallback": (None, False),
        "visual_baseline": ("provider-visual", False),
        "rowwise_gauge_marginalized": ("provider-rowwise", False),
        "framewise_explicit_joint_gauge": ("provider-framewise", False),
        "persistent_explicit_joint_gauge": ("provider-persistent", False),
        "cross_window_identity_marginalized": ("provider-identity", False),
        "sensor_assisted": ("provider-sensor", True),
    }
    config = {
        "schema_name": "prob4d.heldout-provider-promotion-config",
        "schema_version": 1,
        "experiment_id": "target-admission-enforcement-test-v1",
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": source_revision,
        "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "bayesian_phystwin_revision": BPT_REVISION,
        "motioncrafter_revision": PROVIDER_REVISION,
        "model_set_id": MODEL_SET_ID,
        "prediction_run_spec_id": RUN_SPEC_ID,
        "provider_evaluation_manifest_sha256": "7" * 64,
        "frozen_artifact_ids": frozen_artifact_ids,
        "development_group_ids": ["development-a"],
        "calibration_group_ids": ["calibration-a", "calibration-b"],
        "target_group_ids": ["target-a", "target-b"],
        "arms": [
            {
                "arm_id": role,
                "role": role,
                "provider_method_id": provider_method,
                "query_method_id": f"query-{role}",
                "sensor_assisted": sensor_assisted,
                "metadata": {},
            }
            for role, (provider_method, sensor_assisted) in roles.items()
        ],
        "provider_reference_arm_id": "visual_baseline",
        "primary_query_arm_id": "cross_window_identity_marginalized",
        "bootstrap_resamples": 200,
        "bootstrap_seed": 17,
        "minimum_target_group_count": 2,
        "query_superiority_margin_mm": 0.25,
        "harmful_update_margin_mm": 0.0,
        "maximum_harmful_accepted_updates": 0,
        "maximum_worst_group_regression_mm": 0.0,
        "maximum_technical_failures": 0,
        "minimum_mean_accepted_coverage": 0.9,
        "metadata": {},
    }
    return promotion_lock_from_config(config)


def _admission(lock: HeldoutProviderPromotionLockV1) -> HeldoutTargetProviderAdmissionV1:
    cohort_binding_id = lock.frozen_artifact_ids.get("cohort_binding")
    if not isinstance(cohort_binding_id, str):
        raise AssertionError("test admission requires a cohort-bound promotion lock")
    entries = tuple(
        TargetProviderManifestAdmissionV1(
            group_id=group_id,
            episode_id=index,
            stratum="sheet" if index == 0 else "volumetric",
            sequence_id=f"{group_id}-sequence",
            manifest_sha256=f"{20 + index:064x}",
            manifest_artifact_id=f"{30 + index:064x}",
            provider_run_id=f"{40 + index:064x}",
            causal_frame_stop=12,
            admitted_payloads=(
                AdmittedTargetPayloadV1(
                    payload_id=f"{50 + index:064x}",
                    window_id=f"window-{index}",
                    output_frame_ids=(0, 1),
                    source_frame_start=0,
                    source_frame_stop_exclusive=8,
                    dependence_group_ids=("shared-model",),
                ),
            ),
        )
        for index, group_id in enumerate(lock.target_group_ids)
    )
    return HeldoutTargetProviderAdmissionV1(
        promotion_lock_id=lock.promotion_lock_id,
        cohort_binding_id=cohort_binding_id,
        source_repository=lock.source_repository,
        source_revision=lock.source_revision,
        prediction_run_spec_id=lock.prediction_run_spec_id,
        provider_family="external-4d-provider",
        provider_repository="example/provider",
        provider_revision=lock.motioncrafter_revision,
        model_set_id=lock.model_set_id,
        loader_id=LOADER_ID,
        coordinate_semantics="sequence-local-sim3",
        point_semantics="dense-point-map",
        flow_semantics="absent",
        ray_semantics="absent",
        source_dependency_semantics="per-output-exclusive-source-frame-interval-v1",
        target_outcomes_used=False,
        entries=entries,
        metadata={},
    )


def _stream_metadata(admission: HeldoutTargetProviderAdmissionV1) -> dict[str, str]:
    return {
        TARGET_PROVIDER_ADMISSION_METADATA_KEY: admission.target_provider_admission_id,
    }


def test_unbound_promotion_preserves_legacy_execution() -> None:
    lock = _lock(cohort_bound=False)
    assert (
        load_target_admission_for_execution(
            lock,
            None,
            provider_report={},
            query_metadata={},
        )
        is None
    )
    with pytest.raises(ValueError, match="without a frozen cohort binding"):
        load_target_admission_for_execution(
            lock,
            "unexpected.json",
            provider_report={},
            query_metadata={},
        )


def test_cohort_bound_promotion_requires_exact_admission(tmp_path: Path) -> None:
    lock = _lock()
    admission = _admission(lock)
    path = tmp_path / "target-admission.json"
    write_target_provider_admission(admission, path)
    metadata = _stream_metadata(admission)
    provider_report = {"manifest_metadata": metadata}

    with pytest.raises(ValueError, match="requires --target-provider-admission"):
        load_target_admission_for_execution(
            lock,
            None,
            provider_report=provider_report,
            query_metadata=metadata,
        )

    loaded = load_target_admission_for_execution(
        lock,
        path,
        provider_report=provider_report,
        query_metadata=metadata,
    )
    assert loaded == admission


def test_provider_and_query_streams_must_bind_the_same_admission(tmp_path: Path) -> None:
    lock = _lock()
    admission = _admission(lock)
    path = tmp_path / "target-admission.json"
    write_target_provider_admission(admission, path)
    metadata = _stream_metadata(admission)

    with pytest.raises(ValueError, match="manifest_metadata must bind"):
        load_target_admission_for_execution(
            lock,
            path,
            provider_report={"manifest_metadata": {}},
            query_metadata=metadata,
        )
    with pytest.raises(ValueError, match="query metadata must bind"):
        load_target_admission_for_execution(
            lock,
            path,
            provider_report={"manifest_metadata": metadata},
            query_metadata={},
        )
    with pytest.raises(ValueError, match="provider report uses another"):
        load_target_admission_for_execution(
            lock,
            path,
            provider_report={
                "manifest_metadata": {
                    TARGET_PROVIDER_ADMISSION_METADATA_KEY: "0" * 64,
                }
            },
            query_metadata=metadata,
        )
    with pytest.raises(ValueError, match="query results use another"):
        load_target_admission_for_execution(
            lock,
            path,
            provider_report={"manifest_metadata": metadata},
            query_metadata={
                TARGET_PROVIDER_ADMISSION_METADATA_KEY: "0" * 64,
            },
        )


def test_admission_must_match_the_exact_promotion_lock() -> None:
    lock = _lock()
    admission = _admission(lock)
    other_lock = _lock(source_revision="9" * 40)
    metadata = _stream_metadata(admission)

    with pytest.raises(ValueError, match="another promotion lock"):
        validate_target_admission_execution_binding(
            other_lock,
            admission,
            provider_report={"manifest_metadata": metadata},
            query_metadata=metadata,
        )


def test_symlinked_admission_is_rejected(tmp_path: Path) -> None:
    lock = _lock()
    admission = _admission(lock)
    target = tmp_path / "target-admission.json"
    link = tmp_path / "target-admission-link.json"
    write_target_provider_admission(admission, target)
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    metadata = _stream_metadata(admission)

    with pytest.raises(ValueError, match="symbolic link"):
        load_target_admission_for_execution(
            lock,
            link,
            provider_report={"manifest_metadata": metadata},
            query_metadata=metadata,
        )


def test_heldout_run_and_verify_route_through_admission_enforcement() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "prob4d" / "heldout_promotion.py"
    ).read_text(encoding="utf-8")

    assert source.count('"--target-provider-admission"') == 2
    assert source.count("load_target_admission_for_execution(") == 2
    assert "query_metadata=raw_query.get(\"metadata\")" in source
    assert "query_metadata=query_results.metadata" in source
