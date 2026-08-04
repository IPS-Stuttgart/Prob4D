from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import prob4d.cycle_guard_conformal_monte_carlo as conformal_study
from prob4d.causal_gauge_graph_monte_carlo import GaugeGraphStudyScenario
from prob4d.cycle_guard_conformal_monte_carlo import (
    CYCLE_GUARD_CONFORMAL_SCHEMA,
    CYCLE_GUARD_CONFORMAL_VERSION,
    run_cycle_guard_conformal_monte_carlo,
)


TEST_SCENARIOS = (
    GaugeGraphStudyScenario("independent_clean", correlation=0.0),
    GaugeGraphStudyScenario("correlated_clean", correlation=0.75),
    GaugeGraphStudyScenario("highly_correlated_clean", correlation=0.95),
    GaugeGraphStudyScenario(
        "correlated_mild_outliers",
        correlation=0.75,
        outlier_probability=1.0,
        outlier_translation=0.10,
    ),
    GaugeGraphStudyScenario(
        "correlated_strong_outliers",
        correlation=0.75,
        outlier_probability=1.0,
        outlier_translation=0.30,
    ),
    GaugeGraphStudyScenario(
        "highly_correlated_strong_outliers",
        correlation=0.95,
        outlier_probability=1.0,
        outlier_translation=0.30,
    ),
)


def test_small_conformal_cycle_guard_study_is_content_addressed(tmp_path: Path) -> None:
    report = run_cycle_guard_conformal_monte_carlo(
        tmp_path,
        scenarios=TEST_SCENARIOS,
        calibration_trials=6,
        target_trials_per_scenario=2,
        calibration_seed=901,
        target_seed=1901,
        empirical_threshold_quantile=0.95,
        conformal_miscoverage=0.25,
        bootstrap_resamples=20,
        source_revision="1" * 40,
    )

    assert report["schema_name"] == CYCLE_GUARD_CONFORMAL_SCHEMA
    assert report["schema_version"] == CYCLE_GUARD_CONFORMAL_VERSION
    assert report["source_revision"] == "1" * 40
    assert len(str(report["report_id"])) == 64
    calibration = report["calibration"]
    conformal = calibration["finite_sample_normalized_cycle_threshold"]
    assert conformal["calibration_count"] == 6
    assert conformal["order_statistic_rank"] == 6
    assert conformal["guaranteed_miscoverage_upper_bound"] <= 0.25
    assert len(report["trials"]) == 6 * 2 * 5
    assert {item["method_id"] for item in report["aggregate"]} == {
        "tree",
        "full_joint_graph",
        "raw_guarded_graph",
        "empirical_normalized_guard",
        "conformal_normalized_guard",
    }
    assert isinstance(report["decision"]["overall_passed"], bool)

    expected_files = {
        "cycle_guard_conformal_monte_carlo.json",
        "cycle_guard_conformal_monte_carlo.csv",
        "cycle_guard_conformal_monte_carlo_trials.csv",
        "cycle_guard_conformal_monte_carlo.md",
    }
    checksum_lines = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert {line.split("  ", 1)[1] for line in checksum_lines} == expected_files
    for line in checksum_lines:
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest() == digest

    loaded = json.loads(
        (tmp_path / "cycle_guard_conformal_monte_carlo.json").read_text(
            encoding="utf-8"
        )
    )
    assert loaded == report


def test_conformal_study_is_deterministic_except_runtime_metadata(tmp_path: Path) -> None:
    first = run_cycle_guard_conformal_monte_carlo(
        tmp_path / "first",
        scenarios=TEST_SCENARIOS,
        calibration_trials=6,
        target_trials_per_scenario=1,
        calibration_seed=111,
        target_seed=222,
        conformal_miscoverage=0.25,
        bootstrap_resamples=10,
        source_revision="2" * 40,
    )
    second = run_cycle_guard_conformal_monte_carlo(
        tmp_path / "second",
        scenarios=TEST_SCENARIOS,
        calibration_trials=6,
        target_trials_per_scenario=1,
        calibration_seed=111,
        target_seed=222,
        conformal_miscoverage=0.25,
        bootstrap_resamples=10,
        source_revision="2" * 40,
    )

    assert first == second


def test_conformal_study_rejects_unresolvable_alpha_before_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_calibration(*args: object, **kwargs: object) -> object:
        raise AssertionError("calibration must not run for an impossible alpha")

    monkeypatch.setattr(
        conformal_study,
        "_calibrate_thresholds",
        unexpected_calibration,
    )

    with pytest.raises(ValueError, match="finite calibration resolution"):
        run_cycle_guard_conformal_monte_carlo(
            tmp_path,
            scenarios=TEST_SCENARIOS,
            calibration_trials=2,
            target_trials_per_scenario=1,
            conformal_miscoverage=0.10,
            bootstrap_resamples=10,
            source_revision="3" * 40,
        )
