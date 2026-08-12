"""Grouped, lazily imported command surface for Prob4D."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Final

from .command_registry import COMMANDS, Route

_ROUTES: Final[dict[Route, tuple[str, str, str]]] = {
    command.route: (command.module, command.function, command.description)
    for command in COMMANDS
}


def _ambiguous_observation_export(argv: Sequence[str] | None = None) -> int:
    """Require callers to select an observation-export contract explicitly."""

    arguments = list(() if argv is None else argv)
    help_requested = any(value in {"-h", "--help"} for value in arguments)
    lines = [
        "usage: prob4d observation <export-calibrated|export-exploratory> [arguments]",
        "",
        "'prob4d observation export' is intentionally ambiguous and does not run an "
        "exporter.",
        "",
        "Choose one explicit contract:",
        "  export-calibrated   claim-bearing provider-v2 export",
        "  export-exploratory  labelled provider-v2 control",
    ]
    output = sys.stdout if help_requested else sys.stderr
    print("\n".join(lines), file=output)
    return 0 if help_requested else 2


def _children(prefix: Route) -> list[str]:
    position = len(prefix)
    return sorted(
        {
            route[position]
            for route in _ROUTES
            if route[:position] == prefix and len(route) > position
        }
    )


def _render_help(prefix: Route = ()) -> str:
    command = "prob4d" + (" " + " ".join(prefix) if prefix else "")
    lines = [f"usage: {command} <command> [arguments]", ""]
    if not prefix:
        lines.extend(["Grouped access to Prob4D commands.", ""])
    lines.append("commands:")
    for child in _children(prefix):
        candidate = (*prefix, child)
        route = _ROUTES.get(candidate)
        description = route[2] if route is not None else f"{child} commands"
        lines.append(f"  {child:<28} {description}")
    lines.extend(
        [
            "",
            "Prob4D 0.5 installs only this grouped 'prob4d' executable.",
            "Use 'prob4d commands describe <id-or-route>' for command metadata.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve(arguments: Sequence[str]) -> tuple[Route, list[str]] | None:
    prefix: list[str] = []
    for index, token in enumerate(arguments):
        if token.startswith("-"):
            break
        prefix.append(token)
        candidate = tuple(prefix)
        if candidate in _ROUTES:
            return candidate, list(arguments[index + 1 :])
        if not _children(candidate):
            return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_render_help(), end="")
        return 0

    for length in range(1, len(arguments) + 1):
        prefix = tuple(arguments[:length])
        if arguments[length - 1] in {"-h", "--help"}:
            help_namespace = prefix[:-1]
            if help_namespace == () or _children(help_namespace):
                print(_render_help(help_namespace), end="")
                return 0
            break

    resolved = _resolve(arguments)
    if resolved is None:
        matched_namespace: list[str] = []
        for token in arguments:
            candidate = (*matched_namespace, token)
            if _children(candidate):
                matched_namespace.append(token)
            else:
                break
        if matched_namespace and len(matched_namespace) == len(arguments):
            print(_render_help(tuple(matched_namespace)), end="")
            return 0
        print(_render_help(tuple(matched_namespace)), file=sys.stderr, end="")
        return 2

    route, remaining = resolved
    module_name, function_name, _ = _ROUTES[route]
    function = getattr(importlib.import_module(module_name), function_name)
    return int(function(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
