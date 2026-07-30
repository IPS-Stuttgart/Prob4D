from __future__ import annotations

from prob4d.cli import main


def test_grouped_cli_lists_provider_and_observation(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "phystwin" in output


def test_grouped_cli_lists_explicit_provider_v2_exports(capsys) -> None:
    assert main(["observation"]) == 0
    output = capsys.readouterr().out
    assert "export" in output
    assert "export-calibrated" in output
    assert "export-exploratory" in output
    assert "export-v1" in output


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
