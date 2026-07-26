"""Grouped, lazily imported command surface for Prob4D."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Final

Route = tuple[str, ...]
_ROUTES: Final[dict[Route, tuple[str, str, str]]] = {
    ("ablate",): (
        "prob4d.experiments",
        "main",
        "run synthetic or real ablations",
    ),
    ("benchmark",): (
        "prob4d.benchmark",
        "main",
        "run the recursive-fusion benchmark",
    ),
    ("motioncrafter",): (
        "prob4d.motioncrafter",
        "main",
        "generate MotionCrafter prediction products",
    ),
    ("observation", "export"): (
        "prob4d.observation_export",
        "main",
        "export a causally sealed observation belief",
    ),
    ("provider", "manifest"): (
        "prob4d.provider_manifest_cli",
        "main",
        "print the observation-provider manifest",
    ),
    ("phystwin", "evaluate"): (
        "prob4d.phystwin_experiment",
        "main",
        "evaluate Prob4D against a PhysTwin trajectory",
    ),
    ("phystwin", "state"): (
        "prob4d.phystwin_state",
        "main",
        "run the causal-prefix PhysTwin state audit",
    ),
    ("phystwin", "uncertainty"): (
        "prob4d.phystwin_uncertainty",
        "main",
        "evaluate empirical visual-flow uncertainty",
    ),
    ("sintel", "uncertainty"): (
        "prob4d.sintel_uncertainty",
        "main",
        "run the held-out Sintel uncertainty analysis",
    ),
    ("vggt", "baseline"): (
        "prob4d.vggt_baseline",
        "main",
        "export the matched VGGT baseline",
    ),
}


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
        lines.extend(
            [
                "Grouped access to stable Prob4D commands.",
                "",
            ]
        )
    lines.append("commands:")
    for child in _children(prefix):
        candidate = (*prefix, child)
        route = _ROUTES.get(candidate)
        description = route[2] if route is not None else f"{child} commands"
        lines.append(f"  {child:<14} {description}")
    lines.extend(
        [
            "",
            "Legacy prob4d-* entry points remain available for compatibility.",
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
