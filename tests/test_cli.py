from __future__ import annotations

import pytest

from prob4d.cli import main


def test_grouped_cli_lists_provider_and_observation(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "phystwin" in output
    assert "experiment" in output
    assert "identity" in output
    assert "prediction" in output


def test_grouped_cli_lists_explicit_provider_v2_exports(capsys) -> None:
    assert main(["observation"]) == 0
    output = capsys.readouterr().out
    assert "export" in output
    assert "export-calibrated" in output
    assert "export-exploratory" in output
    assert "export-v1" in output
    assert "visual-bias" in output


def test_grouped_cli_requires_explicit_observation_export_mode(capsys) -> None:
    assert main(["observation", "export"]) == 2
    error = capsys.readouterr().err
    assert "intentionally ambiguous" in error
    assert "export-calibrated" in error
    assert "export-exploratory" in error
    assert "export-v1" in error


def test_ambiguous_observation_export_help_is_informational(capsys) -> None:
    assert main(["observation", "export", "--help"]) == 0
    output = capsys.readouterr().out
    assert "Choose one explicit contract" in output
    assert "prob4d-export-observation-belief" in output


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: prob4d" in capsys.readouterr().err


def test_grouped_cli_lists_heldout_provider_experiment(capsys) -> None:
    assert main(["experiment"]) == 0
    output = capsys.readouterr().out
    assert "heldout-provider" in output


def test_grouped_cli_routes_promotion_and_identity_help(capsys) -> None:
    assert main(["experiment", "heldout-provider", "--help"]) == 0
    promotion_help = capsys.readouterr().out
    assert "seal a target-free promotion lock" in promotion_help

    assert main(["identity", "--help"]) == 0
    identity_help = capsys.readouterr().out
    assert "build-mixture" in identity_help
    assert "moment-match" in identity_help


def test_grouped_cli_routes_provider_neutral_prediction_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["prediction", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "import-motioncrafter" in output
    assert "import-vggt" in output
    assert "validate" in output


def test_grouped_cli_routes_vggt_import_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["prediction", "import-vggt", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "integrity-bound VGGT sample" in output
    assert "--sample-id" in output
    assert "--prediction-root" in output


def test_grouped_cli_routes_common_mode_and_visual_bias_help(capsys) -> None:
    with pytest.raises(SystemExit) as stress_exit:
        main(["diagnostic", "common-mode-stress", "--help"])
    assert stress_exit.value.code == 0
    stress_help = capsys.readouterr().out
    assert "coherent visual-bias benchmark" in stress_help

    with pytest.raises(SystemExit) as bias_exit:
        main(["observation", "visual-bias", "--help"])
    assert bias_exit.value.code == 0
    bias_help = capsys.readouterr().out
    assert "explicit low-rank visual-bias nuisance" in bias_help
