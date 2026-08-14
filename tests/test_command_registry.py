from __future__ import annotations

import importlib
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
    CommandLifecycle,
    find_command,
    validate_registry,
)


def test_registry_is_the_canonical_dispatch_source_of_truth() -> None:
    assert set(_ROUTES) == {command.route for command in COMMANDS}
    for command in COMMANDS:
        descriptor = command.to_dict()
        assert "legacy_aliases" not in descriptor
        assert "previous_routes" not in descriptor
        assert descriptor["grouped_command"].startswith("prob4d ")


def test_package_installs_only_the_grouped_executable() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert configuration["project"]["scripts"] == {"prob4d": "prob4d.cli:main"}


def test_registry_exposes_explicit_grouped_routes() -> None:
    expected_routes = {
        ("diagnostic", "finite-sample-preflight"),
        ("diagnostic", "provider-support-envelope"),
        ("diagnostic", "provider-v2-gauge-ablation"),
        ("diagnostic", "source-covariance-localization"),
        ("diagnostic", "visual-bias-calibration"),
        ("experiment", "fresh-provider-readiness"),
        ("observation", "validate"),
        ("provider", "prefix-admission"),
        ("provider", "target-admit"),
        ("provider", "target-verify"),
    }
    assert expected_routes <= set(_ROUTES)


def test_provider_readiness_routes_preserve_lifecycle_and_claim_boundaries() -> None:
    expected = {
        "provider-support-envelope": (
            CommandLifecycle.DIAGNOSTIC,
            False,
            "prob4d.provider_support_envelope:main",
        ),
        "source-covariance-localization": (
            CommandLifecycle.DIAGNOSTIC,
            False,
            "prob4d.source_covariance_localization:main",
        ),
        "provider-prefix-admission": (
            CommandLifecycle.STABLE,
            True,
            "prob4d.provider_prefix_admission:main",
        ),
        "fresh-provider-readiness": (
            CommandLifecycle.EXPERIMENTAL,
            True,
            "prob4d.fresh_provider_readiness:main",
        ),
    }
    for command_id, (lifecycle, claim_bearing, target) in expected.items():
        command = find_command(command_id)
        assert command is not None
        assert command.lifecycle is lifecycle
        assert command.claim_bearing is claim_bearing
        assert command.requires_gpu is False
        assert command.target == target


def test_provider_readiness_route_targets_are_installed_and_callable() -> None:
    for command_id in (
        "provider-support-envelope",
        "source-covariance-localization",
        "provider-prefix-admission",
        "fresh-provider-readiness",
    ):
        command = find_command(command_id)
        assert command is not None
        module = importlib.import_module(command.module)
        assert callable(getattr(module, command.function))


def test_commands_list_json_is_machine_readable(capsys) -> None:
    assert main(["commands", "list", "--lifecycle", "stable", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert all(item["lifecycle"] == "stable" for item in payload)
    assert any(item["command_id"] == "observation-export-calibrated" for item in payload)
    assert all("legacy_aliases" not in item for item in payload)


def test_commands_describe_resolves_only_id_and_canonical_route(capsys) -> None:
    assert main(["commands", "describe", "target-provider-admit", "--json"]) == 0
    id_payload = json.loads(capsys.readouterr().out)
    assert id_payload["grouped_command"] == "prob4d provider target-admit"
    assert id_payload["claim_bearing"] is True

    assert main(["commands", "describe", "observation", "validate"]) == 0
    route_output = capsys.readouterr().out
    assert "id: observation-validate" in route_output

    assert main(["commands", "describe", "fresh-provider-readiness", "--json"]) == 0
    readiness_payload = json.loads(capsys.readouterr().out)
    assert readiness_payload["grouped_command"] == ("prob4d experiment fresh-provider-readiness")
    assert readiness_payload["claim_bearing"] is True

    assert main(["commands", "describe", "prob4d-target-admit"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_commands_migrate_action_was_removed(capsys) -> None:
    assert main(["commands", "migrate", "prob4d-validate-observation"]) == 2
    error = capsys.readouterr().err
    assert "list|describe|validate" in error


def test_commands_validate_reports_canonical_registry(capsys) -> None:
    assert main(["commands", "validate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["command_count"] == len(COMMANDS)
    assert "legacy_alias_count" not in payload


def test_registry_rejects_duplicate_routes() -> None:
    duplicate = replace(COMMANDS[1], command_id="duplicate")
    with pytest.raises(ValueError, match="duplicate grouped route"):
        validate_registry((*COMMANDS, duplicate))


def test_find_command_resolves_canonical_route() -> None:
    command = find_command("prob4d observation export-calibrated")
    assert command is not None
    assert command.lifecycle is CommandLifecycle.STABLE

    readiness = find_command("prob4d experiment fresh-provider-readiness")
    assert readiness is not None
    assert readiness.lifecycle is CommandLifecycle.EXPERIMENTAL


def test_registry_has_no_provider_v1_or_archived_routes() -> None:
    assert find_command("prob4d observation export-v1") is None
    assert all(command.lifecycle.value != "archived" for command in COMMANDS)
    assert all(command.owner != "provider-v1" for command in COMMANDS)
