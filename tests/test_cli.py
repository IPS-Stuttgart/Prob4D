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


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: prob4d" in capsys.readouterr().err
