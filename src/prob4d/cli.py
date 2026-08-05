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
        "prob4d.benchmark_safe",
        "main",
        "run the pinned recursive-fusion benchmark",
    ),
    ("diagnostic", "cycle-guard-conformal"): (
        "prob4d.cycle_guard_conformal_monte_carlo",
        "main",
        "evaluate finite-sample calibration of the normalized cycle guard",
    ),
    ("diagnostic", "cycle-guard-monte-carlo"): (
        "prob4d.cycle_guard_monte_carlo",
        "main",
        "compare raw and uncertainty-normalized source cycle guards",
    ),
    ("diagnostic", "gauge-graph"): (
        "prob4d.causal_gauge_graph_ablation",
        "main",
        "compare causal gauge-tree, graph, guarded fallback, and fixed-lag modes",
    ),
    ("diagnostic", "gauge-graph-monte-carlo"): (
        "prob4d.causal_gauge_graph_monte_carlo",
        "main",
        "run calibration-separated correlated gauge-graph Monte Carlo study",
    ),
    ("evaluate", "provider"): (
        "prob4d.provider_evaluation",
        "main",
        "evaluate paired provider artifacts across held-out groups",
    ),
    ("experiment", "heldout-provider"): (
        "prob4d.heldout_promotion",
        "main",
        "freeze, run, and verify the held-out Prob4D-to-BayesianPhysTwin gate",
    ),
    ("identity",): (
        "prob4d.material_identity_cli",
        "main",
        "build, validate, and marginalize material-identity artifacts",
    ),
    ("motioncrafter",): (
        "prob4d.motioncrafter_safe",
        "main",
        "generate crash-safe MotionCrafter prediction products",
    ),
    ("observation", "export"): (
        "prob4d.cli",
        "_ambiguous_observation_export",
        "choose an explicit calibrated, exploratory, or provider-v1 export mode",
    ),
    ("observation", "export-v1"): (
        "prob4d.causal_stream_cli",
        "main",
        "export through the frozen provider-v1 compatibility CLI",
    ),
    ("observation", "export-calibrated"): (
        "prob4d.provider_v2_cli",
        "main_calibrated",
        "export a claim-bearing provider-v2 observation belief",
    ),
    ("observation", "export-exploratory"): (
        "prob4d.provider_v2_cli",
        "main_exploratory",
        "export an explicitly exploratory provider-v2 observation belief",
    ),
    ("project", "identity"): (
        "prob4d.project_identity",
        "main",
        "print the stable project ID and repository aliases",
    ),
    ("provider", "manifest"): (
        "prob4d.provider_manifest_cli",
        "main",
        "print a versioned observation-provider manifest",
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
    ("storage", "benchmark"): (
        "prob4d.prediction_store_benchmark",
        "main",
        "profile eager or memory-mapped prediction loading",
    ),
    ("storage", "materialize"): (
        "prob4d.prediction_store_cli",
        "main_materialize",
        "create a content-addressed memory-mapped prediction store",
    ),
    ("storage", "validate"): (
        "prob4d.prediction_store_cli",
        "main_validate",
        "validate and summarize a memory-mapped prediction store",
    ),
    ("vggt", "baseline"): (
        "prob4d.vggt_baseline",
        "main",
        "export the matched VGGT baseline",
    ),
}


def _ambiguous_observation_export(argv: Sequence[str] | None = None) -> int:
    """Require callers to select an observation-export contract explicitly."""

    arguments = list(() if argv is None else argv)
    help_requested = any(value in {"-h", "--help"} for value in arguments)
    lines = [
        "usage: prob4d observation <export-calibrated|export-exploratory|export-v1> "
        "[arguments]",
        "",
        "'prob4d observation export' is intentionally ambiguous and does not run "
        "an exporter.",
        "",
        "Choose one explicit contract:",
        "  export-calibrated   claim-bearing provider-v2 export",
        "  export-exploratory  labelled provider-v2 control",
        "  export-v1           frozen provider-v1 compatibility export",
        "",
        "The legacy 'prob4d-export-observation-belief' executable remains unchanged.",
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
        lines.append(f"  {child:<20} {description}")
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
