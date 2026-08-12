"""Listing, inspection, and validation for canonical Prob4D commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Final

from .command_registry import (
    COMMANDS,
    CommandLifecycle,
    CommandSpec,
    find_command,
    iter_commands,
    validate_registry,
)

_HELP_FLAGS: Final = frozenset({"-h", "--help"})


def _help() -> str:
    return (
        "usage: prob4d commands <list|describe|validate> [arguments]\n\n"
        "commands:\n"
        "  list                         list entries [--lifecycle VALUE] [--json]\n"
        "  describe SELECTOR            show metadata by id or canonical route\n"
        "  validate                     validate the canonical registry\n\n"
        "lifecycles: stable, experimental, diagnostic\n"
    )


def _parse_selector(arguments: Sequence[str]) -> tuple[str, bool] | None:
    values: list[str] = []
    json_output = False
    for argument in arguments:
        if argument == "--json":
            json_output = True
        elif argument.startswith("-"):
            return None
        else:
            values.append(argument)
    if not values:
        return None
    return " ".join(values), json_output


def _print_list(commands: Sequence[CommandSpec], *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                [command.to_dict() for command in commands],
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not commands:
        print("No commands are registered for this selection.")
        return
    lifecycle_width = max(
        len("LIFECYCLE"),
        *(len(command.lifecycle.value) for command in commands),
    )
    id_width = max(len("ID"), *(len(command.command_id) for command in commands))
    print(f"{'LIFECYCLE':<{lifecycle_width}}  {'ID':<{id_width}}  GPU  CLAIM  COMMAND")
    for command in commands:
        print(
            f"{command.lifecycle.value:<{lifecycle_width}}  "
            f"{command.command_id:<{id_width}}  "
            f"{'yes' if command.requires_gpu else 'no ':<3}  "
            f"{'yes' if command.claim_bearing else 'no ':<5}  "
            f"{command.grouped_command}"
        )


def _print_command(command: CommandSpec, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(command.to_dict(), indent=2, sort_keys=True))
        return
    requirements = ", ".join(command.runtime_requirements) or "none"
    print(f"id: {command.command_id}")
    print(f"lifecycle: {command.lifecycle.value}")
    print(f"command: {command.grouped_command}")
    print(f"owner: {command.owner}")
    print(f"runtime requirements: {requirements}")
    print(f"requires GPU: {str(command.requires_gpu).lower()}")
    print(f"claim-bearing: {str(command.claim_bearing).lower()}")
    print(f"target: {command.target}")
    print(f"description: {command.description}")


def _list(arguments: Sequence[str]) -> int:
    lifecycle: CommandLifecycle | None = None
    json_output = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--json":
            json_output = True
            index += 1
        elif argument == "--lifecycle" and index + 1 < len(arguments):
            try:
                lifecycle = CommandLifecycle(arguments[index + 1])
            except ValueError:
                print("invalid command lifecycle", file=sys.stderr)
                return 2
            index += 2
        else:
            print(_help(), file=sys.stderr, end="")
            return 2
    _print_list(iter_commands(lifecycle=lifecycle), json_output=json_output)
    return 0


def _describe(arguments: Sequence[str]) -> int:
    parsed = _parse_selector(arguments)
    if parsed is None:
        print(_help(), file=sys.stderr, end="")
        return 2
    selector, json_output = parsed
    command = find_command(selector)
    if command is None:
        print(f"unknown command: {selector}", file=sys.stderr)
        return 2
    _print_command(command, json_output=json_output)
    return 0


def _validate(arguments: Sequence[str]) -> int:
    if any(argument not in {"--json"} for argument in arguments):
        print(_help(), file=sys.stderr, end="")
        return 2
    validate_registry()
    summary = {
        "command_count": len(COMMANDS),
        "lifecycles": sorted({command.lifecycle.value for command in COMMANDS}),
        "valid": True,
    }
    if "--json" in arguments:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Command registry is valid: {summary['command_count']} canonical commands.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in _HELP_FLAGS:
        print(_help(), end="")
        return 0
    action = arguments[0]
    if action == "list":
        return _list(arguments[1:])
    if action == "describe":
        return _describe(arguments[1:])
    if action == "validate":
        return _validate(arguments[1:])
    print(_help(), file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
