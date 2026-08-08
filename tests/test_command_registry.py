from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

import pytest

from prob4d.cli import _ROUTES, main
from prob4d.command_registry import (
    COMMANDS,
    EXPECTED_LEGACY_ALIASES,
    CommandLifecycle,
    find_command,
    validate_registry,
)


def test_registry_is_the_dispatch_source_of_truth() -> None:
    assert set(_ROUTES) == {command.route for command in COMMANDS}
    aliases = {alias for command in COMMANDS for alias in command.legacy_aliases}
    assert aliases == EXPECTED_LEGACY_ALIASES


def test_registry_matches_installed_legacy_scripts() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = set(configuration["project"]["scripts"])
    assert scripts - {"prob4d"} == EXPECTED_LEGACY_ALIASES


def test_registry_exposes_grouped_replacements_for_previously_legacy_only_tools() -> None:
    expected_routes = {
        ("diagnostic", "finite-sample-preflight"),
        ("diagnostic", "provider-v2-gauge-ablation"),
        ("diagnostic", "visual-bias-calibration"),
        ("observation", "validate"),
        ("provider", "target-admit"),
        ("provider", "target-verify"),
    }
    assert expected_routes <= set(_ROUTES)


def test_commands_list_json_is_machine_readable(capsys) -> None:
    assert main(["commands", "list", "--lifecycle", "stable", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert all(item["lifecycle"] == "stable" for item in payload)
    assert any(item["command_id"] == "observation-export-calibrated" for item in payload)


def test_commands_describe_resolves_id_route_and_alias(capsys) -> None:
    assert main(["commands", "describe", "prob4d-target-admit", "--json"]) == 0
    alias_payload = json.loads(capsys.readouterr().out)
    assert alias_payload["grouped_command"] == "prob4d provider target-admit"
    assert alias_payload["claim_bearing"] is True

    assert main(["commands", "describe", "observation", "validate"]) == 0
    route_output = capsys.readouterr().out
    assert "id: observation-validate" in route_output


def test_commands_migrate_only_accepts_previous_interfaces(capsys) -> None:
    assert main(["commands", "migrate", "prob4d-validate-observation"]) == 0
    assert capsys.readouterr().out.strip() == "prob4d observation validate"

    assert main(["commands", "migrate", "observation-validate"]) == 2
    assert "unknown previous command selector" in capsys.readouterr().err


def test_commands_validate_reports_complete_alias_coverage(capsys) -> None:
    assert main(["commands", "validate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["command_count"] == len(COMMANDS)
    assert payload["legacy_alias_count"] == len(EXPECTED_LEGACY_ALIASES)


def test_registry_rejects_duplicate_routes() -> None:
    duplicate = replace(COMMANDS[1], command_id="duplicate", legacy_aliases=())
    with pytest.raises(ValueError, match="duplicate grouped route"):
        validate_registry((*COMMANDS, duplicate))


def test_find_command_resolves_canonical_route() -> None:
    command = find_command("prob4d observation export-calibrated")
    assert command is not None
    assert command.lifecycle is CommandLifecycle.STABLE


def test_registry_exposes_portable_sparse_gauge_prior() -> None:
    command = find_command("prob4d gauge prior")
    assert command is not None
    assert command.command_id == "gauge-tree-prior-artifact"
    assert command.lifecycle is CommandLifecycle.STABLE
    assert command.claim_bearing is False
