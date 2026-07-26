from __future__ import annotations

from prob4d.cli import main


def test_grouped_cli_lists_provider_and_observation(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "phystwin" in output


def test_grouped_cli_lists_nested_commands(capsys) -> None:
    assert main(["observation"]) == 0
    assert "export" in capsys.readouterr().out


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: prob4d" in capsys.readouterr().err
