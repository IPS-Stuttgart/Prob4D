from __future__ import annotations

from pathlib import Path

from prob4d.common_mode_stress import (
    CommonModeStressConfig,
    main,
    run_common_mode_stress,
)


def test_default_stress_exposes_failure_and_bias_aware_control() -> None:
    report = run_common_mode_stress(
        CommonModeStressConfig(
            clean_calibration_groups=500,
            bias_calibration_groups=500,
            target_groups=800,
            seed=17,
        )
    )
    assert report.decision == "pass-controlled-common-mode-mechanism"
    assert report.common_mode_audit["low_disagreement_high_error_rate"] > 0.25
    naive = report.methods["naive-independent"]
    aware = report.methods["explicit-shared-bias"]
    assert abs(aware.complete_policy_coverage_90 - 0.9) < abs(
        naive.complete_policy_coverage_90 - 0.9
    )
    assert aware.deployed_rmse_m <= naive.deployed_rmse_m
    assert aware.harmful_accepted_count <= naive.harmful_accepted_count
    assert aware.deployed_rmse_m <= aware.physical_baseline_rmse_m
    assert aware.exact_fallback_reproduced_count == aware.rejected_count
    assert naive.exact_fallback_reproduced_count == naive.rejected_count


def test_stress_is_deterministic() -> None:
    config = CommonModeStressConfig(
        clean_calibration_groups=200,
        bias_calibration_groups=200,
        target_groups=300,
        seed=9,
    )
    assert run_common_mode_stress(config).artifact_id == run_common_mode_stress(
        config
    ).artifact_id


def test_zero_bias_does_not_falsely_pass_failure_exposure() -> None:
    report = run_common_mode_stress(
        CommonModeStressConfig(
            clean_calibration_groups=300,
            bias_calibration_groups=300,
            target_groups=400,
            coherent_bias_std_m=0.0,
            seed=3,
        )
    )
    assert report.gates["coherent_failure_exposed"] is False
    assert report.decision == "controlled-mechanism-gate-failed"


def test_cli_writes_replayable_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    code = main(
        [
            "--output",
            str(output),
            "--seed",
            "5",
            "--clean-calibration-groups",
            "200",
            "--bias-calibration-groups",
            "200",
            "--target-groups",
            "300",
        ]
    )
    assert code == 0
    first = output.read_bytes()
    assert main(
        [
            "--output",
            str(output),
            "--seed",
            "5",
            "--clean-calibration-groups",
            "200",
            "--bias-calibration-groups",
            "200",
            "--target-groups",
            "300",
        ]
    ) == 0
    assert output.read_bytes() == first
