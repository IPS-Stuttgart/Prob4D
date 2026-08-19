from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from prob4d.cli import main as grouped_main
from prob4d.command_registry import CommandLifecycle, find_command
from prob4d.study import HeldoutProviderStudy, main, write_study_preflight
from prob4d.study_sensitivity import PairedDifferenceScenarioV1


def _lock() -> SimpleNamespace:
    return SimpleNamespace(
        promotion_lock_id="a" * 64,
        development_group_ids=tuple(f"development-{index:02d}" for index in range(4)),
        calibration_group_ids=tuple(f"calibration-{index:02d}" for index in range(10)),
        target_group_ids=tuple(f"target-{index:02d}" for index in range(12)),
        bootstrap_resamples=5000,
        minimum_mean_accepted_coverage=0.90,
        minimum_target_group_count=12,
        query_superiority_margin_mm=1.0,
        harmful_update_margin_mm=0.25,
        maximum_harmful_accepted_updates=0,
        maximum_worst_group_regression_mm=0.5,
    )


def test_high_level_facade_composes_target_free_reports() -> None:
    study = HeldoutProviderStudy(_lock())
    preflight = study.preflight(
        source_summary_id="b" * 64,
        source_metric="deployed_minus_physical_rmse_mm",
        paired_difference_scenarios=(
            PairedDifferenceScenarioV1("source", 1.0),
        ),
        coverage_levels=(0.90,),
        power_levels=(0.80,),
    )

    assert study.group_counts == {"development": 4, "calibration": 10, "target": 12}
    assert preflight.promotion_lock_id == "a" * 64
    assert preflight.capability.primary_levels_finite is True
    assert preflight.sensitivity.query_margin_detectable is True
    assert preflight.to_dict()["target_outcomes_opened"] is False


def test_preflight_publication_checks_all_destinations_before_writing(tmp_path) -> None:
    preflight = HeldoutProviderStudy(_lock()).preflight(
        source_summary_id="b" * 64,
        source_metric="metric",
        paired_difference_scenarios=(PairedDifferenceScenarioV1("source", 1.0),),
    )
    output_dir = tmp_path / "preflight"
    output_dir.mkdir()
    (output_dir / "study_sensitivity.md").write_text("retained", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_study_preflight(preflight, output_dir)
    assert not (output_dir / "finite_sample_capability.json").exists()


def test_grouped_study_preflight_command_writes_both_reports(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("prob4d.study.load_promotion_lock", lambda _path: _lock())
    output_dir = tmp_path / "preflight"

    assert (
        main(
            [
                str(tmp_path / "lock.json"),
                "--source-summary-id",
                "b" * 64,
                "--source-metric",
                "deployed_minus_physical_rmse_mm",
                "--paired-sd",
                "source=1.0",
                "--coverage",
                "0.90",
                "--power",
                "0.80",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["target_group_count"] == 12
    assert payload["target_outcomes_opened"] is False
    assert (output_dir / "finite_sample_capability.json").is_file()
    assert (output_dir / "finite_sample_capability.md").is_file()
    assert (output_dir / "study_sensitivity.json").is_file()
    assert (output_dir / "study_sensitivity.md").is_file()


def test_command_registry_exposes_non_claim_bearing_study_preflight() -> None:
    command = find_command("prob4d study preflight")
    assert command is not None
    assert command.command_id == "study-preflight"
    assert command.lifecycle is CommandLifecycle.DIAGNOSTIC
    assert command.claim_bearing is False
    assert command.target == "prob4d.study:main"


def test_grouped_cli_routes_study_preflight_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        grouped_main(["study", "preflight", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--source-summary-id" in output
    assert "--paired-sd" in output
    assert "--output-dir" in output
