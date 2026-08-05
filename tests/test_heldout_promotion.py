from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prob4d.heldout_promotion import (
    HeldoutProviderPromotionLockV1,
    PromotionQueryRowV1,
    build_query_results,
    evaluate_heldout_promotion,
    load_promotion_lock,
    load_promotion_report,
    load_query_results,
    main,
    promotion_lock_from_config,
    write_promotion_lock,
    write_promotion_report,
    write_query_results,
)

ROLES = (
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
    (
        "sensor",
        "sensor_assisted",
        "provider-sensor",
        "bpt-sensor",
        True,
    ),
)


def _arm_values() -> list[dict[str, object]]:
    return [
        {
            "arm_id": arm_id,
            "role": role,
            "query_method_id": query_method,
            "provider_method_id": provider_method,
            "sensor_assisted": sensor_assisted,
            "metadata": {},
        }
        for arm_id, role, provider_method, query_method, sensor_assisted in ROLES
    ]


def _config() -> dict[str, object]:
    return {
        "experiment_id": "real-provider-gate-v1",
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
        "development_group_ids": ["development-1", "development-2"],
        "calibration_group_ids": ["calibration-1", "calibration-2"],
        "target_group_ids": ["target-1", "target-2", "target-3"],
        "arms": _arm_values(),
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


def _lock() -> HeldoutProviderPromotionLockV1:
    return promotion_lock_from_config(_config())


def _provider_report(lock: HeldoutProviderPromotionLockV1) -> dict[str, object]:
    records = []
    for group_id in lock.target_group_ids:
        for method_id in lock.provider_method_ids:
            records.append(
                {
                    "case_id": f"{group_id}-case",
                    "group_id": group_id,
                    "method_id": method_id,
                }
            )
    return {
        "schema_name": "prob4d.provider-evaluation-report",
        "schema_version": 3,
        "source_manifest": "/sealed/provider-manifest.json",
        "source_manifest_sha256": lock.provider_evaluation_manifest_sha256,
        "primary_mode": "metric",
        "primary_support": "common_across_registered_methods",
        "secondary_support": "native_per_method",
        "reference_method": lock.provider_reference_method_id,
        "bootstrap_resamples": lock.bootstrap_resamples,
        "bootstrap_seed": lock.bootstrap_seed,
        "legacy_artifacts_allowed": False,
        "evaluation_chunk_size": 4096,
        "manifest_metadata": {},
        "method_metadata": {},
        "cases": records,
        "aggregate": {},
        "comparisons": {},
        "decision_policy": {},
        "decision": {
            "policy_id": "provider-gate-v1",
            "overall_passed": True,
            "group_count_passed": True,
            "rules": [],
        },
        "support_semantics": "common support",
        "claim_boundary": "provider only",
    }


def _query_rows(
    lock: HeldoutProviderPromotionLockV1,
    *,
    harmful_primary: bool = False,
    break_fallback: bool = False,
) -> tuple[PromotionQueryRowV1, ...]:
    rows: list[PromotionQueryRowV1] = []
    for group_index, group_id in enumerate(lock.target_group_ids):
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
            rejected = arm.arm_id == "visual" and group_index == 2
            if rejected:
                deployed = fallback_id
                rmse = 5.0
                exact = not break_fallback
                if break_fallback:
                    deployed = hashlib.sha256(b"wrong-fallback").hexdigest()
            else:
                deployed = hashlib.sha256(f"deployed-{group_id}-{arm.arm_id}".encode()).hexdigest()
                rmse = 4.0
                exact = None
                if harmful_primary and arm.arm_id == lock.primary_query_arm_id:
                    rmse = 5.5
            rows.append(
                PromotionQueryRowV1(
                    group_id=group_id,
                    arm_id=arm.arm_id,
                    query_rmse_mm=rmse,
                    deployed_artifact_id=deployed,
                    fallback_artifact_id=fallback_id,
                    accepted=not rejected,
                    exact_fallback_reproduced=exact,
                    accepted_coverage=None if rejected else 0.92,
                    accepted_width_mm=None if rejected else 1.5,
                )
            )
    return tuple(rows)


def test_documented_promotion_configuration_is_valid() -> None:
    config = json.loads(
        Path("docs/examples/heldout-provider-promotion-config.json").read_text(encoding="utf-8")
    )

    lock = promotion_lock_from_config(config)

    assert lock.experiment_id == "heldout-real-provider-v1"
    assert lock.primary_query_arm_id == "identity"
    assert len(lock.target_group_ids) == 3


def test_promotion_lock_round_trip_and_target_separation(tmp_path: Path) -> None:
    lock = _lock()
    path = tmp_path / "promotion-lock.json"
    write_promotion_lock(lock, path)

    loaded = load_promotion_lock(path)
    with pytest.raises(FileExistsError):
        write_promotion_lock(lock, path)

    assert loaded == lock
    assert loaded.promotion_lock_id == lock.promotion_lock_id
    assert loaded.physical_fallback_arm_id == "fallback"
    assert loaded.provider_reference_method_id == "provider-visual"

    config = _config()
    config["target_group_ids"] = ["calibration-1", "target-2", "target-3"]
    with pytest.raises(ValueError, match="calibration and target groups"):
        promotion_lock_from_config(config)


def test_complete_promotion_gate_passes_and_replays(tmp_path: Path) -> None:
    lock = _lock()
    query_results = build_query_results(lock, rows=_query_rows(lock))
    provider_report = _provider_report(lock)
    provider_bytes = json.dumps(provider_report, sort_keys=True).encode()

    report = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256=hashlib.sha256(provider_bytes).hexdigest(),
    )

    assert report.overall_passed is True
    assert report.provider_decision["overall_passed"] is True
    assert report.query_decision["overall_passed"] is True
    assert report.query_decision["exact_fallback_failure_count"] == 0
    assert report.query_aggregate["identity"]["harmful_accepted_update_count"] == 0
    assert report.query_decision["paired_bootstrap"]["ci95_upper"] < -0.25

    query_path = tmp_path / "query-results.json"
    report_path = tmp_path / "promotion-report.json"
    write_query_results(query_results, query_path)
    write_promotion_report(report, report_path)
    assert load_query_results(query_path) == query_results
    assert load_promotion_report(report_path) == report


def test_promotion_gate_retains_harmful_update_failure() -> None:
    lock = _lock()
    query_results = build_query_results(
        lock,
        rows=_query_rows(lock, harmful_primary=True),
    )
    provider_report = _provider_report(lock)

    report = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256="7" * 64,
    )

    assert report.overall_passed is False
    assert report.query_decision["harmful_accepted_updates_passed"] is False
    assert report.query_decision["query_superiority_passed"] is False
    assert report.query_aggregate["identity"]["harmful_accepted_update_count"] == 3


def test_promotion_gate_retains_exact_fallback_failure() -> None:
    lock = _lock()
    query_results = build_query_results(
        lock,
        rows=_query_rows(lock, break_fallback=True),
    )
    provider_report = _provider_report(lock)

    report = evaluate_heldout_promotion(
        lock,
        query_results,
        provider_report,
        provider_report_sha256="8" * 64,
    )

    assert report.overall_passed is False
    assert report.query_decision["exact_fallback_passed"] is False
    assert report.query_decision["exact_fallback_failure_count"] == 1


def test_provider_report_cannot_change_target_groups() -> None:
    lock = _lock()
    query_results = build_query_results(lock, rows=_query_rows(lock))
    provider_report = _provider_report(lock)
    provider_report["cases"] = provider_report["cases"][:-1]

    with pytest.raises(ValueError, match="incomplete or changed method sets"):
        evaluate_heldout_promotion(
            lock,
            query_results,
            provider_report,
            provider_report_sha256="9" * 64,
        )


def test_provider_report_rejects_duplicate_case_method_records() -> None:
    lock = _lock()
    query_results = build_query_results(lock, rows=_query_rows(lock))
    provider_report = _provider_report(lock)
    cases = provider_report["cases"]
    assert isinstance(cases, list)
    cases.append(dict(cases[0]))

    with pytest.raises(ValueError, match="duplicate case/method"):
        evaluate_heldout_promotion(
            lock,
            query_results,
            provider_report,
            provider_report_sha256="a" * 64,
        )


def test_grouped_freeze_run_verify_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    lock_path = tmp_path / "lock.json"
    provider_path = tmp_path / "provider.json"
    raw_query_path = tmp_path / "query-raw.json"
    output = tmp_path / "output"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")

    assert main(["freeze", str(config_path), "--output", str(lock_path)]) == 0
    lock = load_promotion_lock(lock_path)
    provider_path.write_text(json.dumps(_provider_report(lock)), encoding="utf-8")
    raw_query_path.write_text(
        json.dumps(
            {
                "promotion_lock_id": lock.promotion_lock_id,
                "rows": [row.to_dict() for row in _query_rows(lock)],
                "metadata": {"source": "bayesian-phystwin"},
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run",
                str(lock_path),
                "--provider-report",
                str(provider_path),
                "--query-results",
                str(raw_query_path),
                "--output-dir",
                str(output),
                "--require-pass",
            ]
        )
        == 0
    )
    assert (output / "promotion_report.md").is_file()
    with pytest.raises(FileExistsError, match="output already exists"):
        main(
            [
                "run",
                str(lock_path),
                "--provider-report",
                str(provider_path),
                "--query-results",
                str(raw_query_path),
                "--output-dir",
                str(output),
            ]
        )
    assert (
        main(
            [
                "verify",
                str(lock_path),
                "--provider-report",
                str(provider_path),
                "--query-results",
                str(output / "query_results.sealed.json"),
                "--report",
                str(output / "promotion_report.json"),
            ]
        )
        == 0
    )
