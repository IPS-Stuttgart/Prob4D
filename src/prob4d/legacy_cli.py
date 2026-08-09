"""Registry-driven compatibility wrappers for historical console scripts.

The standalone ``prob4d-*`` executables remain installed for one compatibility
period, but every invocation names the canonical grouped replacement before
lazily dispatching to the unchanged implementation target.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from typing import Any, Final

from .command_registry import (
    COMMANDS_BY_LEGACY_ALIAS,
    EXPECTED_LEGACY_ALIASES,
    CommandSpec,
)

LEGACY_CLI_MIGRATION_SCHEMA: Final = "prob4d.legacy-cli-migration"
LEGACY_CLI_MIGRATION_VERSION: Final = 1
LEGACY_CLI_REMOVAL_POLICY: Final = (
    "Legacy executables are compatibility aliases and may be removed in the next "
    "documented incompatible pre-1.0 release."
)

LegacyTarget = Callable[[], Any]


def legacy_migration_table() -> Mapping[str, str]:
    """Return the canonical legacy-executable to grouped-command mapping."""

    return {
        alias: COMMANDS_BY_LEGACY_ALIAS[alias].grouped_command
        for alias in sorted(EXPECTED_LEGACY_ALIASES)
    }


def legacy_migration_descriptor() -> dict[str, object]:
    """Return deterministic machine-readable migration metadata."""

    return {
        "schema_name": LEGACY_CLI_MIGRATION_SCHEMA,
        "schema_version": LEGACY_CLI_MIGRATION_VERSION,
        "removal_policy": LEGACY_CLI_REMOVAL_POLICY,
        "migrations": dict(legacy_migration_table()),
    }


def legacy_deprecation_message(alias: str) -> str:
    """Render the exact migration guidance for one registered alias."""

    command = COMMANDS_BY_LEGACY_ALIAS.get(alias)
    if command is None:
        raise ValueError(f"unknown Prob4D legacy executable: {alias}")
    return (
        f"DEPRECATION: '{alias}' is a legacy compatibility executable; "
        f"use '{command.grouped_command}'. {LEGACY_CLI_REMOVAL_POLICY}"
    )


def _load_target(command: CommandSpec) -> LegacyTarget:
    module = importlib.import_module(command.module)
    target = getattr(module, command.function, None)
    if not callable(target):
        raise RuntimeError(
            f"legacy command target is not callable: {command.target}"
        )
    return target


def _dispatch(alias: str) -> int:
    command = COMMANDS_BY_LEGACY_ALIAS.get(alias)
    if command is None:
        raise RuntimeError(f"unregistered Prob4D legacy executable: {alias}")
    print(legacy_deprecation_message(alias), file=sys.stderr)
    result = _load_target(command)()
    if result is None:
        return 0
    if type(result) is not int:
        raise TypeError(
            f"legacy command target {command.target} returned a non-integer status"
        )
    return result


def provider_manifest() -> int:
    return _dispatch("prob4d-provider-manifest")


def ablate() -> int:
    return _dispatch("prob4d-ablate")


def ablate_provider_v2_gauge() -> int:
    return _dispatch("prob4d-ablate-provider-v2-gauge")


def benchmark() -> int:
    return _dispatch("prob4d-benchmark")


def evaluate_provider() -> int:
    return _dispatch("prob4d-evaluate-provider")


def export_observation_belief() -> int:
    return _dispatch("prob4d-export-observation-belief")


def export_calibrated_observation_belief() -> int:
    return _dispatch("prob4d-export-calibrated-observation-belief")


def export_exploratory_observation_belief() -> int:
    return _dispatch("prob4d-export-exploratory-observation-belief")


def finite_sample_preflight() -> int:
    return _dispatch("prob4d-finite-sample-preflight")


def motioncrafter() -> int:
    return _dispatch("prob4d-motioncrafter")


def phystwin() -> int:
    return _dispatch("prob4d-phystwin")


def phystwin_state() -> int:
    return _dispatch("prob4d-phystwin-state")


def phystwin_uncertainty() -> int:
    return _dispatch("prob4d-phystwin-uncertainty")


def sintel_uncertainty() -> int:
    return _dispatch("prob4d-sintel-uncertainty")


def target_admit() -> int:
    return _dispatch("prob4d-target-admit")


def target_verify() -> int:
    return _dispatch("prob4d-target-verify")


def validate_observation() -> int:
    return _dispatch("prob4d-validate-observation")


def vggt_baseline() -> int:
    return _dispatch("prob4d-vggt-baseline")


def visual_bias_calibration() -> int:
    return _dispatch("prob4d-visual-bias-calibration")


__all__ = [
    "LEGACY_CLI_MIGRATION_SCHEMA",
    "LEGACY_CLI_MIGRATION_VERSION",
    "LEGACY_CLI_REMOVAL_POLICY",
    "ablate",
    "ablate_provider_v2_gauge",
    "benchmark",
    "evaluate_provider",
    "export_calibrated_observation_belief",
    "export_exploratory_observation_belief",
    "export_observation_belief",
    "finite_sample_preflight",
    "legacy_deprecation_message",
    "legacy_migration_descriptor",
    "legacy_migration_table",
    "motioncrafter",
    "phystwin",
    "phystwin_state",
    "phystwin_uncertainty",
    "provider_manifest",
    "sintel_uncertainty",
    "target_admit",
    "target_verify",
    "validate_observation",
    "vggt_baseline",
    "visual_bias_calibration",
]
