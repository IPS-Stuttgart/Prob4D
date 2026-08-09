from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

import pytest

import prob4d.legacy_cli as legacy_cli
from prob4d.command_registry import (
    COMMANDS_BY_LEGACY_ALIAS,
    EXPECTED_LEGACY_ALIASES,
)

WRAPPERS = {
    "prob4d-provider-manifest": "provider_manifest",
    "prob4d-ablate": "ablate",
    "prob4d-ablate-provider-v2-gauge": "ablate_provider_v2_gauge",
    "prob4d-benchmark": "benchmark",
    "prob4d-evaluate-provider": "evaluate_provider",
    "prob4d-export-observation-belief": "export_observation_belief",
    "prob4d-export-calibrated-observation-belief": (
        "export_calibrated_observation_belief"
    ),
    "prob4d-export-exploratory-observation-belief": (
        "export_exploratory_observation_belief"
    ),
    "prob4d-finite-sample-preflight": "finite_sample_preflight",
    "prob4d-motioncrafter": "motioncrafter",
    "prob4d-phystwin": "phystwin",
    "prob4d-phystwin-state": "phystwin_state",
    "prob4d-phystwin-uncertainty": "phystwin_uncertainty",
    "prob4d-sintel-uncertainty": "sintel_uncertainty",
    "prob4d-target-admit": "target_admit",
    "prob4d-target-verify": "target_verify",
    "prob4d-validate-observation": "validate_observation",
    "prob4d-vggt-baseline": "vggt_baseline",
    "prob4d-visual-bias-calibration": "visual_bias_calibration",
}


def test_wrapper_inventory_matches_the_canonical_registry() -> None:
    assert set(WRAPPERS) == EXPECTED_LEGACY_ALIASES
    assert set(legacy_cli.legacy_migration_table()) == EXPECTED_LEGACY_ALIASES
    assert legacy_cli.legacy_migration_table() == {
        alias: COMMANDS_BY_LEGACY_ALIAS[alias].grouped_command
        for alias in sorted(EXPECTED_LEGACY_ALIASES)
    }


def test_pyproject_routes_every_legacy_script_through_one_wrapper_module() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = configuration["project"]["scripts"]

    assert scripts["prob4d"] == "prob4d.cli:main"
    for alias, function_name in WRAPPERS.items():
        assert scripts[alias] == f"prob4d.legacy_cli:{function_name}"


def test_machine_readable_migration_descriptor_is_complete() -> None:
    descriptor = legacy_cli.legacy_migration_descriptor()

    assert descriptor["schema_name"] == "prob4d.legacy-cli-migration"
    assert descriptor["schema_version"] == 1
    assert descriptor["migrations"] == legacy_cli.legacy_migration_table()
    assert "incompatible pre-1.0 release" in descriptor["removal_policy"]


@pytest.mark.parametrize(("alias", "function_name"), sorted(WRAPPERS.items()))
def test_every_wrapper_warns_and_delegates_to_the_registered_target(
    alias: str,
    function_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed = []

    def load_target(command):
        observed.append(command)
        return lambda: 7

    monkeypatch.setattr(legacy_cli, "_load_target", load_target)

    wrapper = getattr(legacy_cli, function_name)
    assert wrapper() == 7

    assert observed == [COMMANDS_BY_LEGACY_ALIAS[alias]]
    message = capsys.readouterr().err.strip()
    assert message == legacy_cli.legacy_deprecation_message(alias)
    assert alias in message
    assert COMMANDS_BY_LEGACY_ALIAS[alias].grouped_command in message


def test_wrapper_normalizes_a_none_status_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(legacy_cli, "_load_target", lambda _command: lambda: None)

    assert legacy_cli.validate_observation() == 0


def test_wrapper_rejects_a_noninteger_target_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(legacy_cli, "_load_target", lambda _command: lambda: "bad")

    with pytest.raises(TypeError, match="non-integer status"):
        legacy_cli.validate_observation()


def test_unknown_alias_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown Prob4D legacy executable"):
        legacy_cli.legacy_deprecation_message("prob4d-not-registered")
    with pytest.raises(RuntimeError, match="unregistered Prob4D legacy executable"):
        legacy_cli._dispatch("prob4d-not-registered")


def test_target_loading_uses_only_the_registry_module_and_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = COMMANDS_BY_LEGACY_ALIAS["prob4d-target-admit"]
    sentinel = lambda: 3
    observed = []

    class Module:
        main_admit = sentinel

    def import_module(name: str):
        observed.append(name)
        return Module()

    monkeypatch.setattr(legacy_cli.importlib, "import_module", import_module)

    assert legacy_cli._load_target(command) is sentinel
    assert observed == [command.module]


def test_missing_registry_target_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = COMMANDS_BY_LEGACY_ALIAS["prob4d-target-admit"]

    monkeypatch.setattr(
        legacy_cli.importlib,
        "import_module",
        lambda _name: object(),
    )

    with pytest.raises(RuntimeError, match="target is not callable"):
        legacy_cli._load_target(command)
