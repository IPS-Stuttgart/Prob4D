"""Canonical metadata for the grouped :mod:`prob4d` command interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final

Route = tuple[str, ...]


class CommandLifecycle(str, Enum):
    """Lifecycle classification for a registered command."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DIAGNOSTIC = "diagnostic"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One lazily imported command and its public metadata."""

    command_id: str
    route: Route
    module: str
    function: str
    description: str
    lifecycle: CommandLifecycle
    owner: str
    legacy_aliases: tuple[str, ...] = ()
    previous_routes: tuple[Route, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    claim_bearing: bool = False
    requires_gpu: bool = False

    @property
    def target(self) -> str:
        return f"{self.module}:{self.function}"

    @property
    def grouped_command(self) -> str:
        return "prob4d " + " ".join(self.route)

    @property
    def previous_grouped_commands(self) -> tuple[str, ...]:
        return tuple("prob4d " + " ".join(route) for route in self.previous_routes)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible registry metadata."""

        payload = asdict(self)
        payload["route"] = list(self.route)
        payload["previous_routes"] = [list(route) for route in self.previous_routes]
        payload["lifecycle"] = self.lifecycle.value
        payload["legacy_aliases"] = list(self.legacy_aliases)
        payload["runtime_requirements"] = list(self.runtime_requirements)
        payload["target"] = self.target
        payload["grouped_command"] = self.grouped_command
        payload["previous_grouped_commands"] = list(self.previous_grouped_commands)
        return payload


def _command(
    command_id: str,
    route: str,
    target: str,
    description: str,
    lifecycle: CommandLifecycle,
    owner: str,
    *,
    aliases: tuple[str, ...] = (),
    requirements: tuple[str, ...] = (),
    claim_bearing: bool = False,
    requires_gpu: bool = False,
) -> CommandSpec:
    module, function = target.split(":", maxsplit=1)
    return CommandSpec(
        command_id=command_id,
        route=tuple(route.split()),
        module=module,
        function=function,
        description=description,
        lifecycle=lifecycle,
        owner=owner,
        legacy_aliases=aliases,
        runtime_requirements=requirements,
        claim_bearing=claim_bearing,
        requires_gpu=requires_gpu,
    )


S = CommandLifecycle.STABLE
E = CommandLifecycle.EXPERIMENTAL
D = CommandLifecycle.DIAGNOSTIC
A = CommandLifecycle.ARCHIVED

COMMANDS: Final[tuple[CommandSpec, ...]] = (
    _command(
        "command-registry",
        "commands",
        "prob4d.command_catalog:main",
        "inspect and validate the canonical command registry",
        S,
        "command-interface",
    ),
    _command(
        "ablation",
        "ablate",
        "prob4d.experiments:main",
        "run synthetic or real ablations",
        E,
        "recursive-fusion-experiments",
        aliases=("prob4d-ablate",),
    ),
    _command(
        "provider-v2-gauge-ablation",
        "diagnostic provider-v2-gauge-ablation",
        "prob4d.provider_v2_gauge_ablation:main",
        "compare provider-v2 gauge backends under the frozen ablation contract",
        D,
        "provider-v2",
        aliases=("prob4d-ablate-provider-v2-gauge",),
    ),
    _command(
        "recursive-fusion-benchmark",
        "benchmark",
        "prob4d.benchmark_safe:main",
        "run the pinned recursive-fusion benchmark",
        S,
        "recursive-fusion",
        aliases=("prob4d-benchmark",),
    ),
    _command(
        "common-mode-stress",
        "diagnostic common-mode-stress",
        "prob4d.common_mode_stress:main",
        "stress coherent visual bias and explicit shared-bias uncertainty",
        D,
        "visual-bias",
    ),
    _command(
        "cross-provider-guard",
        "diagnostic cross-provider-guard",
        "prob4d.cross_provider_guard:main",
        "calibrate provider corroboration without independence assumptions",
        D,
        "provider-calibration",
    ),
    _command(
        "anchor-common-bias",
        "diagnostic anchor-common-bias",
        "prob4d.anchor_common_bias_study:main",
        "quantify independent-anchor power against shared visual bias",
        D,
        "visual-bias",
    ),
    _command(
        "cycle-guard-conformal",
        "diagnostic cycle-guard-conformal",
        "prob4d.cycle_guard_conformal_monte_carlo:main",
        "evaluate finite-sample calibration of the normalized cycle guard",
        D,
        "gauge-graph",
    ),
    _command(
        "cycle-guard-monte-carlo",
        "diagnostic cycle-guard-monte-carlo",
        "prob4d.cycle_guard_monte_carlo:main",
        "compare raw and uncertainty-normalized source cycle guards",
        D,
        "gauge-graph",
    ),
    _command(
        "gauge-graph",
        "diagnostic gauge-graph",
        "prob4d.causal_gauge_graph_ablation:main",
        "compare causal gauge-tree, graph, guarded fallback, and fixed-lag modes",
        D,
        "gauge-graph",
    ),
    _command(
        "gauge-graph-monte-carlo",
        "diagnostic gauge-graph-monte-carlo",
        "prob4d.causal_gauge_graph_monte_carlo:main",
        "run calibration-separated correlated gauge-graph Monte Carlo study",
        D,
        "gauge-graph",
    ),
    _command(
        "gauge-tree-prior-artifact",
        "gauge prior",
        "prob4d.gauge_tree_prior:main",
        "verify or explicitly densify portable sparse gauge-tree priors",
        S,
        "gauge-tree",
    ),
    _command(
        "joint-covariance",
        "diagnostic joint-covariance",
        "prob4d.joint_covariance_metrics:main",
        "evaluate matched residuals under conditional plus low-rank covariance",
        D,
        "observation-covariance",
    ),
    _command(
        "finite-sample-preflight",
        "diagnostic finite-sample-preflight",
        "prob4d.finite_sample_capability:preflight_cli",
        "check whether group counts support the requested finite-sample gate",
        D,
        "finite-sample-calibration",
        aliases=("prob4d-finite-sample-preflight",),
    ),
    _command(
        "visual-bias-calibration",
        "diagnostic visual-bias-calibration",
        "prob4d.visual_bias_calibration:main",
        "fit or validate source-only coherent visual-bias calibration",
        D,
        "visual-bias",
        aliases=("prob4d-visual-bias-calibration",),
    ),
    _command(
        "provider-evaluation",
        "evaluate provider",
        "prob4d.provider_evaluation:main",
        "evaluate paired provider artifacts across held-out groups",
        S,
        "provider-evaluation-v2",
        aliases=("prob4d-evaluate-provider",),
        claim_bearing=True,
    ),
    _command(
        "heldout-provider-promotion",
        "experiment heldout-provider",
        "prob4d.heldout_promotion:main",
        "freeze, run, and verify the held-out Prob4D-to-BayesianPhysTwin gate",
        E,
        "heldout-provider-promotion",
        claim_bearing=True,
    ),
    _command(
        "material-identity",
        "identity",
        "prob4d.material_identity_cli:main",
        "build, validate, and marginalize material-identity artifacts",
        S,
        "material-identity",
    ),
    _command(
        "motioncrafter-producer",
        "motioncrafter",
        "prob4d.motioncrafter_safe:main",
        "generate crash-safe MotionCrafter prediction products",
        S,
        "motioncrafter-producer",
        aliases=("prob4d-motioncrafter",),
        requirements=("MotionCrafter checkout", "model checkpoints", "CUDA"),
        requires_gpu=True,
    ),
    _command(
        "observation-export-guidance",
        "observation export",
        "prob4d.cli:_ambiguous_observation_export",
        "choose an explicit calibrated, exploratory, or provider-v1 export mode",
        S,
        "provider-v2",
    ),
    _command(
        "observation-export-v1",
        "observation export-v1",
        "prob4d.causal_stream_cli:main",
        "export through the frozen provider-v1 compatibility CLI",
        A,
        "provider-v1",
        aliases=("prob4d-export-observation-belief",),
        claim_bearing=True,
    ),
    _command(
        "observation-export-calibrated",
        "observation export-calibrated",
        "prob4d.provider_v2_cli:main_calibrated",
        "export a claim-bearing provider-v2 observation belief",
        S,
        "provider-v2",
        aliases=("prob4d-export-calibrated-observation-belief",),
        claim_bearing=True,
    ),
    _command(
        "observation-export-exploratory",
        "observation export-exploratory",
        "prob4d.provider_v2_cli:main_exploratory",
        "export an explicitly exploratory provider-v2 observation belief",
        D,
        "provider-v2",
        aliases=("prob4d-export-exploratory-observation-belief",),
    ),
    _command(
        "observation-validate",
        "observation validate",
        "prob4d.observation_validation:main",
        "validate and summarize a portable observation belief",
        S,
        "observation-belief-v1",
        aliases=("prob4d-validate-observation",),
    ),
    _command(
        "observation-bias-binding",
        "observation bias-binding",
        "prob4d.observation_bias_binding:main",
        "bind recursive observation factors to their visual-bias stream",
        S,
        "visual-bias",
    ),
    _command(
        "observation-visual-bias",
        "observation visual-bias",
        "prob4d.visual_bias:main",
        "validate explicit shared visual-bias nuisance sidecars",
        S,
        "visual-bias",
    ),
    _command(
        "observation-visual-bias-stream",
        "observation visual-bias-stream",
        "prob4d.visual_bias_stream:main",
        "validate recursive updates that share one visual-bias prior",
        S,
        "visual-bias",
    ),
    _command(
        "prediction-provider",
        "prediction",
        "prob4d.prediction_cli:main",
        "import and validate provider-neutral prediction manifests",
        S,
        "prediction-provider",
    ),
    _command(
        "project-identity",
        "project identity",
        "prob4d.project_identity:main",
        "print the stable project ID and repository aliases",
        S,
        "project-identity",
    ),
    _command(
        "provider-manifest",
        "provider manifest",
        "prob4d.provider_manifest_cli:main",
        "print a versioned observation-provider manifest",
        S,
        "provider-manifest",
        aliases=("prob4d-provider-manifest",),
    ),
    _command(
        "target-provider-admit",
        "provider target-admit",
        "prob4d.target_provider_admission_cli:main_admit",
        "bind target provider manifests before outcome evaluation",
        S,
        "target-provider-admission-v1",
        aliases=("prob4d-target-admit",),
        claim_bearing=True,
    ),
    _command(
        "target-provider-verify",
        "provider target-verify",
        "prob4d.target_provider_admission_cli:main_verify",
        "verify a sealed target provider admission result",
        S,
        "target-provider-admission-v1",
        aliases=("prob4d-target-verify",),
        claim_bearing=True,
    ),
    _command(
        "phystwin-evaluate",
        "phystwin evaluate",
        "prob4d.phystwin_experiment:main",
        "evaluate Prob4D against a PhysTwin trajectory",
        D,
        "phystwin-integration",
        aliases=("prob4d-phystwin",),
        requirements=("PhysTwin dataset",),
    ),
    _command(
        "phystwin-state",
        "phystwin state",
        "prob4d.phystwin_state:main",
        "run the causal-prefix PhysTwin state audit",
        D,
        "phystwin-integration",
        aliases=("prob4d-phystwin-state",),
        requirements=("PhysTwin dataset",),
    ),
    _command(
        "phystwin-uncertainty",
        "phystwin uncertainty",
        "prob4d.phystwin_uncertainty:main",
        "evaluate empirical visual-flow uncertainty",
        D,
        "phystwin-integration",
        aliases=("prob4d-phystwin-uncertainty",),
        requirements=("PhysTwin dataset",),
    ),
    _command(
        "sintel-uncertainty",
        "sintel uncertainty",
        "prob4d.sintel_uncertainty:main",
        "run the held-out Sintel uncertainty analysis",
        D,
        "sintel-integration",
        aliases=("prob4d-sintel-uncertainty",),
        requirements=("Sintel dataset",),
    ),
    _command(
        "storage-benchmark",
        "storage benchmark",
        "prob4d.prediction_store_benchmark:main",
        "profile eager or memory-mapped prediction loading",
        D,
        "prediction-storage",
    ),
    _command(
        "storage-materialize",
        "storage materialize",
        "prob4d.prediction_store_cli:main_materialize",
        "create a content-addressed memory-mapped prediction store",
        S,
        "prediction-storage",
    ),
    _command(
        "storage-validate",
        "storage validate",
        "prob4d.prediction_store_cli:main_validate",
        "validate and summarize a memory-mapped prediction store",
        S,
        "prediction-storage",
    ),
    _command(
        "vggt-baseline",
        "vggt baseline",
        "prob4d.vggt_baseline:main",
        "export the matched VGGT baseline",
        E,
        "vggt-integration",
        aliases=("prob4d-vggt-baseline",),
        requirements=("VGGT checkout", "model checkpoint", "CUDA"),
        requires_gpu=True,
    ),
)

EXPECTED_LEGACY_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "prob4d-provider-manifest",
        "prob4d-ablate",
        "prob4d-ablate-provider-v2-gauge",
        "prob4d-benchmark",
        "prob4d-evaluate-provider",
        "prob4d-export-observation-belief",
        "prob4d-export-calibrated-observation-belief",
        "prob4d-export-exploratory-observation-belief",
        "prob4d-finite-sample-preflight",
        "prob4d-motioncrafter",
        "prob4d-phystwin",
        "prob4d-phystwin-state",
        "prob4d-phystwin-uncertainty",
        "prob4d-sintel-uncertainty",
        "prob4d-target-admit",
        "prob4d-target-verify",
        "prob4d-validate-observation",
        "prob4d-vggt-baseline",
        "prob4d-visual-bias-calibration",
    }
)


def validate_registry(commands: Iterable[CommandSpec] = COMMANDS) -> None:
    """Reject malformed, ambiguous, or incomplete registry entries."""

    command_ids: set[str] = set()
    routes: set[Route] = set()
    previous_routes: set[Route] = set()
    aliases: set[str] = set()
    for command in commands:
        if not command.command_id or command.command_id.startswith("-"):
            raise ValueError(f"invalid command id: {command.command_id!r}")
        if command.command_id in command_ids:
            raise ValueError(f"duplicate command id: {command.command_id}")
        if not command.route or any(not token or token.startswith("-") for token in command.route):
            raise ValueError(f"invalid grouped route: {command.route!r}")
        if command.route in routes:
            raise ValueError("duplicate grouped route: " + " ".join(command.route))
        if not command.module.startswith("prob4d."):
            raise ValueError(f"invalid command module: {command.module}")
        if not command.function or not command.owner or not command.description:
            raise ValueError(f"incomplete command metadata: {command.command_id}")
        for previous_route in command.previous_routes:
            if not previous_route or previous_route in previous_routes:
                raise ValueError(
                    "duplicate or empty previous grouped route: " + " ".join(previous_route)
                )
            previous_routes.add(previous_route)
        for alias in command.legacy_aliases:
            if not alias.startswith("prob4d-") or alias in aliases:
                raise ValueError(f"invalid or duplicate legacy alias: {alias}")
            aliases.add(alias)
        command_ids.add(command.command_id)
        routes.add(command.route)

    collisions = routes & previous_routes
    if collisions:
        rendered = sorted(" ".join(route) for route in collisions)
        raise ValueError("previous grouped route collides with current route: " + str(rendered))
    if aliases != EXPECTED_LEGACY_ALIASES:
        missing = sorted(EXPECTED_LEGACY_ALIASES - aliases)
        extra = sorted(aliases - EXPECTED_LEGACY_ALIASES)
        raise ValueError(f"legacy alias registry drift; missing={missing}, extra={extra}")


validate_registry()

COMMANDS_BY_ID: Final = {command.command_id: command for command in COMMANDS}
COMMANDS_BY_ROUTE: Final = {command.route: command for command in COMMANDS}
COMMANDS_BY_PREVIOUS_ROUTE: Final = {
    route: command for command in COMMANDS for route in command.previous_routes
}
COMMANDS_BY_LEGACY_ALIAS: Final = {
    alias: command for command in COMMANDS for alias in command.legacy_aliases
}


def iter_commands(
    *, lifecycle: CommandLifecycle | None = None
) -> tuple[CommandSpec, ...]:
    """Return registry entries in deterministic command-id order."""

    selected = (
        command
        for command in COMMANDS
        if lifecycle is None or command.lifecycle is lifecycle
    )
    return tuple(sorted(selected, key=lambda command: command.command_id))


def find_command(selector: str) -> CommandSpec | None:
    """Resolve a command ID, current or previous route, or legacy alias."""

    normalized = selector.strip()
    command = COMMANDS_BY_ID.get(normalized)
    if command is None:
        command = COMMANDS_BY_LEGACY_ALIAS.get(normalized)
    route = tuple(normalized.removeprefix("prob4d ").split())
    if command is None:
        command = COMMANDS_BY_ROUTE.get(route)
    if command is None:
        command = COMMANDS_BY_PREVIOUS_ROUTE.get(route)
    return command
