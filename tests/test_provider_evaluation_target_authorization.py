from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import prob4d.provider_evaluation as provider_evaluation
from prob4d._heldout_promotion_lock import (
    HeldoutProviderPromotionLockV1,
    promotion_lock_from_config,
    write_promotion_lock,
)
from prob4d.provider_evaluation_target_authorization import (
    PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD,
    TARGET_PROVIDER_ADMISSION_METADATA_KEY,
    build_provider_evaluation_target_authorization,
    load_provider_evaluation_manifest_snapshot,
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
TARGET_GROUP_IDS = ("target-a", "target-b")
METHOD_IDS = (
    "provider-framewise",
    "provider-identity",
    "provider-persistent",
    "provider-rowwise",
    "provider-sensor",
    "provider-visual",
)


def _manifest(root: Path, *, metadata: dict[str, object] | None = None) -> Path:
    cases = []
    for index, group_id in enumerate(TARGET_GROUP_IDS):
        truth = root / f"truth-{index}.npz"
        truth.write_bytes(b"truth-placeholder")
        predictions: dict[str, str] = {}
        for method_id in METHOD_IDS:
            path = root / f"prediction-{index}-{method_id}.npz"
            path.write_bytes(b"prediction-placeholder")
            predictions[method_id] = path.name
        cases.append(
            {
                "case_id": f"case-{index}",
                "group_id": group_id,
                "truth": truth.name,
                "predictions": predictions,
                "boundary_frames": [],
                "prefix_frame_stop_exclusive": None,
            }
        )
    value = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 2,
        "primary_mode": "metric",
        "reference_method": "provider-visual",
        "cases": cases,
        "metadata": {} if metadata is None else metadata,
        "decision_policy": {
            "policy_id": "provider-gate-v1",
            "minimum_group_count": len(TARGET_GROUP_IDS),
            "rules": [
                {
                    "rule_id": "identity-rmse-superiority",
                    "candidate_method": "provider-identity",
                    "metric": "metric_point_rmse",
                    "direction": "lower",
                    "criterion": "superiority",
                    "margin": 0.25,
                }
            ],
        },
    }
    path = root / "provider-evaluation.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _lock(manifest_sha256: str) -> HeldoutProviderPromotionLockV1:
    roles = (
        ("fallback", "physical_fallback", None, "query-fallback", False),
        (
            "framewise",
            "framewise_explicit_joint_gauge",
            "provider-framewise",
            "query-framewise",
            False,
        ),
        (
            "identity",
            "cross_window_identity_marginalized",
            "provider-identity",
            "query-identity",
            False,
        ),
        (
            "persistent",
            "persistent_explicit_joint_gauge",
            "provider-persistent",
            "query-persistent",
            False,
        ),
        (
            "rowwise",
            "rowwise_gauge_marginalized",
            "provider-rowwise",
            "query-rowwise",
            False,
        ),
        ("sensor", "sensor_assisted", "provider-sensor", "query-sensor", True),
        ("visual", "visual_baseline", "provider-visual", "query-visual", False),
    )
    return promotion_lock_from_config(
        {
            "experiment_id": "provider-target-authorization-test-v1",
            "source_repository": "IPS-Stuttgart/Prob4D",
            "source_revision": SOURCE_REVISION,
            "bayesian_phystwin_repository": "IPS-Stuttgart/BayesianPhysTwin",
            "bayesian_phystwin_revision": BPT_REVISION,
            "motioncrafter_revision": PROVIDER_REVISION,
            "model_set_id": MODEL_SET_ID,
            "prediction_run_spec_id": RUN_SPEC_ID,
            "provider_evaluation_manifest_sha256": manifest_sha256,
            "frozen_artifact_ids": {
                "provider_configuration": "0" * 64,
                "gauge_calibration": "1" * 64,
                "point_calibration": "2" * 64,
                "source_reliability_calibration": "3" * 64,
                "material_identity_calibration": "4" * 64,
                "selection_lock": "5" * 64,
                "bayesian_guard_configuration": "6" * 64,
                "cohort_binding": COHORT_BINDING_ID,
            },
            "development_group_ids": ["development-a"],
            "calibration_group_ids": ["calibration-a", "calibration-b"],
            "target_group_ids": list(TARGET_GROUP_IDS),
            "arms": [
                {
                    "arm_id": arm_id,
                    "role": role,
                    "provider_method_id": provider_method_id,
                    "query_method_id": query_method_id,
                    "sensor_assisted": sensor_assisted,
                    "metadata": {},
                }
                for (
                    arm_id,
                    role,
                    provider_method_id,
                    query_method_id,
                    sensor_assisted,
                ) in roles
            ],
            "provider_reference_arm_id": "visual",
            "primary_query_arm_id": "identity",
            "bootstrap_resamples": 200,
            "bootstrap_seed": 17,
            "minimum_target_group_count": len(TARGET_GROUP_IDS),
            "query_superiority_margin_mm": 0.25,
            "harmful_update_margin_mm": 0.0,
            "maximum_harmful_accepted_updates": 0,
            "maximum_worst_group_regression_mm": 0.0,
            "maximum_technical_failures": 0,
            "minimum_mean_accepted_coverage": 0.9,
            "metadata": {},
        }
    )


def _admission(lock: HeldoutProviderPromotionLockV1) -> HeldoutTargetProviderAdmissionV1:
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
        cohort_binding_id=COHORT_BINDING_ID,
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


def _bundle(tmp_path: Path) -> tuple[Path, HeldoutProviderPromotionLockV1, object]:
    manifest = _manifest(tmp_path)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    lock = _lock(digest)
    return manifest, lock, _admission(lock)


def test_authorization_breaks_the_manifest_lock_admission_identity_cycle(
    tmp_path: Path,
) -> None:
    manifest, lock, admission = _bundle(tmp_path)
    snapshot = load_provider_evaluation_manifest_snapshot(manifest)
    authorization = build_provider_evaluation_target_authorization(
        snapshot,
        lock,
        admission,
        bootstrap_resamples=200,
        bootstrap_seed=17,
        legacy_artifacts_allowed=False,
    )

    assert TARGET_PROVIDER_ADMISSION_METADATA_KEY not in snapshot.metadata
    assert authorization["promotion_lock_id"] == lock.promotion_lock_id
    assert (
        authorization["target_provider_admission_id"]
        == admission.target_provider_admission_id
    )
    assert authorization["source_manifest_sha256"] == lock.provider_evaluation_manifest_sha256
    assert authorization["target_outcomes_opened_during_authorization"] is False


def test_circular_manifest_metadata_is_rejected_before_target_path_resolution(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        metadata={TARGET_PROVIDER_ADMISSION_METADATA_KEY: "0" * 64},
    )

    with pytest.raises(ValueError, match="circular content identity"):
        load_provider_evaluation_manifest_snapshot(manifest)


def test_manifest_lock_mismatch_fails_before_evaluation_plan_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    lock = _lock("9" * 64)
    admission = _admission(lock)
    lock_path = tmp_path / "promotion-lock.json"
    admission_path = tmp_path / "target-admission.json"
    write_promotion_lock(lock, lock_path)
    write_target_provider_admission(admission, admission_path)

    def forbidden_loader(path: Path) -> object:
        raise AssertionError(f"target plan was resolved before authorization: {path}")

    monkeypatch.setattr(provider_evaluation, "load_provider_evaluation_plan", forbidden_loader)
    with pytest.raises(ValueError, match="manifest bytes differ"):
        provider_evaluation.run_provider_evaluation(
            manifest,
            tmp_path / "output",
            bootstrap_resamples=200,
            seed=17,
            promotion_lock_path=lock_path,
            target_provider_admission_path=admission_path,
        )


def test_authorized_provider_evaluation_emits_replayable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, lock, admission = _bundle(tmp_path)
    lock_path = tmp_path / "promotion-lock.json"
    admission_path = tmp_path / "target-admission.json"
    write_promotion_lock(lock, lock_path)
    write_target_provider_admission(admission, admission_path)

    def fake_evaluate(cases: object, **kwargs: object) -> tuple[list[dict[str, object]], dict]:
        del kwargs
        records = []
        for case in cases:
            for method_id in case.predictions:
                records.append(
                    {
                        "case_id": case.case_id,
                        "group_id": case.group_id,
                        "method_id": method_id,
                    }
                )
        return records, {}

    monkeypatch.setattr(provider_evaluation, "evaluate_provider_cases", fake_evaluate)
    monkeypatch.setattr(
        provider_evaluation,
        "aggregate_provider_records",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        provider_evaluation,
        "evaluate_provider_decision_policy",
        lambda *args, **kwargs: {"overall_passed": True, "policy_id": "provider-gate-v1"},
    )
    monkeypatch.setattr(
        provider_evaluation,
        "write_provider_evaluation_outputs",
        lambda *args, **kwargs: None,
    )

    report = provider_evaluation.run_provider_evaluation(
        manifest,
        tmp_path / "output",
        bootstrap_resamples=200,
        seed=17,
        promotion_lock_path=lock_path,
        target_provider_admission_path=admission_path,
    )

    assert report["schema_version"] == 4
    authorization = report[PROVIDER_EVALUATION_TARGET_AUTHORIZATION_FIELD]
    assert authorization["promotion_lock_id"] == lock.promotion_lock_id
    assert (
        authorization["target_provider_admission_id"]
        == admission.target_provider_admission_id
    )
    assert report["manifest_metadata"] == {}


def test_manifest_mutation_during_authorized_evaluation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, lock, admission = _bundle(tmp_path)
    lock_path = tmp_path / "promotion-lock.json"
    admission_path = tmp_path / "target-admission.json"
    write_promotion_lock(lock, lock_path)
    write_target_provider_admission(admission, admission_path)

    def mutating_evaluate(cases: object, **kwargs: object) -> tuple[list[dict], dict]:
        del cases, kwargs
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return [], {}

    monkeypatch.setattr(provider_evaluation, "evaluate_provider_cases", mutating_evaluate)
    monkeypatch.setattr(
        provider_evaluation,
        "aggregate_provider_records",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        provider_evaluation,
        "evaluate_provider_decision_policy",
        lambda *args, **kwargs: {"overall_passed": True},
    )

    with pytest.raises(ValueError, match="manifest changed during evaluation"):
        provider_evaluation.run_provider_evaluation(
            manifest,
            tmp_path / "output",
            bootstrap_resamples=200,
            seed=17,
            promotion_lock_path=lock_path,
            target_provider_admission_path=admission_path,
        )
