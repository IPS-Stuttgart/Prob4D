from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.anchor_common_bias_study import (
    StudyConfig,
    finite_sample_upper_threshold,
    main,
    run_study,
    validate_report,
)
from prob4d.cli import main as grouped_main


def _small_config() -> StudyConfig:
    return StudyConfig(
        seed=91,
        calibration_groups=100,
        target_groups=160,
        rows_per_group=64,
        anchor_sigma_ratios=(0.5, 1.0),
        anchor_support_fractions=(0.20, 1.0),
        reference_anchor_sigma_ratio=0.5,
        reference_anchor_support_fraction=0.20,
    )


def test_finite_sample_upper_threshold_uses_registered_order_statistic() -> None:
    scores = np.arange(1.0, 20.0)
    threshold, order, bound = finite_sample_upper_threshold(scores, 0.10)
    assert order == 18
    assert threshold == 18.0
    assert bound == pytest.approx(0.10)


def test_study_is_deterministic_and_content_addressed() -> None:
    first = run_study(_small_config(), source_revision="0" * 40)
    second = run_study(_small_config(), source_revision="0" * 40)
    assert first == second
    validate_report(first)
    assert len(first["report_id"]) == 64


def test_differential_guard_stays_blind_to_shared_bias() -> None:
    report = run_study(_small_config(), source_revision="0" * 40)
    differential = report["differential_provider_guard"]
    assert differential["provider_specific_bias_rejection_rate"] > 0.90
    assert differential["shared_common_bias_rejection_rate"] < 0.15


def test_anchor_power_improves_with_support_at_fixed_precision() -> None:
    report = run_study(_small_config(), source_revision="0" * 40)
    records = [
        item
        for item in report["anchor_common_mode_power_grid"]
        if item["anchor_sigma_ratio"] == 0.5
    ]
    assert records[0]["anchor_support_fraction"] == 0.20
    assert records[1]["anchor_support_fraction"] == 1.0
    assert records[1]["shared_bias_rejection_rate"] >= records[0]["shared_bias_rejection_rate"]


def test_anchor_power_degrades_with_noisier_anchor() -> None:
    report = run_study(_small_config(), source_revision="0" * 40)
    records = {
        (item["anchor_sigma_ratio"], item["anchor_support_fraction"]): item
        for item in report["anchor_common_mode_power_grid"]
    }
    assert (
        records[(0.5, 1.0)]["shared_bias_rejection_rate"]
        >= records[(1.0, 1.0)]["shared_bias_rejection_rate"]
    )


def test_report_tampering_is_rejected() -> None:
    report = run_study(_small_config(), source_revision="0" * 40)
    report["registered_decision"] = "tampered"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_report(report)


def test_cli_writes_replayable_json_and_markdown(tmp_path: Path) -> None:
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"
    exit_code = main(
        [
            "--quick",
            "--seed",
            "20260806",
            "--source-revision",
            "1" * 40,
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )
    assert exit_code in {0, 3}
    report = json.loads(output_json.read_text(encoding="utf-8"))
    validate_report(report)
    markdown = output_markdown.read_text(encoding="utf-8")
    assert report["report_id"] in markdown
    assert "Shared bias rejected with reference anchor" in markdown


def test_config_rejects_non_exchangeable_tiny_calibration() -> None:
    config = StudyConfig(calibration_groups=5)
    with pytest.raises(ValueError, match="at least 20"):
        config.validate()


def test_grouped_cli_routes_anchor_common_bias_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        grouped_main(["diagnostic", "anchor-common-bias", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "independent-anchor detection" in output
    assert "--source-revision" in output


def test_protocol_matches_frozen_default_design() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1] / "protocols" / "anchor-common-bias-study-v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    design = protocol["design"]
    config = StudyConfig()
    assert design["seed"] == config.seed
    assert design["calibration_groups"] == config.calibration_groups
    assert design["target_groups_per_arm"] == config.target_groups
    assert design["rows_per_group"] == config.rows_per_group
    assert design["dimension"] == config.dimension
    assert design["miscoverage"] == config.miscoverage
    assert design["row_quantile"] == config.row_quantile
    assert design["provider_sigma"] == config.provider_sigma
    assert design["provider_cross_correlation"] == config.provider_cross_correlation
    assert design["provider_specific_bias_sigma"] == config.provider_specific_bias_sigma
    assert design["shared_bias_sigma"] == config.shared_bias_sigma
    assert design["shared_bias_row_fraction"] == config.shared_bias_row_fraction
    assert design["anchor_drift_sigma"] == config.anchor_drift_sigma
    assert tuple(design["anchor_sigma_ratios"]) == config.anchor_sigma_ratios
    assert tuple(design["anchor_support_fractions"]) == config.anchor_support_fractions
    assert design["reference_anchor_sigma_ratio"] == config.reference_anchor_sigma_ratio
    assert design["reference_anchor_support_fraction"] == config.reference_anchor_support_fraction
