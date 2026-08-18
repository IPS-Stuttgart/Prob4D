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


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One lazily imported canonical command and its public metadata."""

    command_id: str
    route: Route
    module: str
    function: str
    description: str
    lifecycle: CommandLifecycle
    owner: str
    runtime_requirements: tuple[str, ...] = ()
    claim_bearing: bool = False
    requires_gpu: bool = False

    @property
    def target(self) -> str:
        return f"{self.module}:{self.function}"

    @property
    def grouped_command(self) -> str:
        return "prob4d " + " ".join(self.route)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible registry metadata."""

        payload = asdict(self)
        payload["route"] = list(self.route)
        payload["lifecycle"] = self.lifecycle.value
        payload["runtime_requirements"] = list(self.runtime_requirements)
        payload["target"] = self.target
        payload["grouped_command"] = self.grouped_command
        return payload


def _command(
    command_id: str,
    route: str,
    target: str,
    description: str,
    lifecycle: CommandLifecycle,
    owner: str,
    requirements: tuple[str, ...],
    claim_bearing: bool,
    requires_gpu: bool,
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
        runtime_requirements=requirements,
        claim_bearing=claim_bearing,
        requires_gpu=requires_gpu,
    )


S = CommandLifecycle.STABLE
E = CommandLifecycle.EXPERIMENTAL
D = CommandLifecycle.DIAGNOSTIC
CommandRow = tuple[
    str,
    str,
    str,
    str,
    CommandLifecycle,
    str,
    tuple[str, ...],
    bool,
    bool,
]

_COMMAND_ROWS: Final[tuple[CommandRow, ...]] = (
    (
        "command-registry",
        "commands",
        "prob4d.command_catalog:main",
        "inspect and validate the canonical command registry",
        S,
        "command-interface",
        (),
        False,
        False,
    ),
    (
        "ablation",
        "ablate",
        "prob4d.experiments:main",
        "run synthetic or real ablations",
        E,
        "recursive-fusion-experiments",
        (),
        False,
        False,
    ),
    (
        "provider-v2-gauge-ablation",
        "diagnostic provider-v2-gauge-ablation",
        "prob4d.provider_v2_gauge_ablation:main",
        "compare provider-v2 gauge backends under the frozen ablation contract",
        D,
        "provider-v2",
        (),
        False,
        False,
    ),
    (
        "recursive-fusion-benchmark",
        "benchmark",
        "prob4d.benchmark_safe:main",
        "run the pinned recursive-fusion benchmark",
        S,
        "recursive-fusion",
        (),
        False,
        False,
    ),
    (
        "common-mode-stress",
        "diagnostic common-mode-stress",
        "prob4d.common_mode_stress:main",
        "stress coherent visual bias and explicit shared-bias uncertainty",
        D,
        "visual-bias",
        (),
        False,
        False,
    ),
    (
        "cross-provider-guard",
        "diagnostic cross-provider-guard",
        "prob4d.cross_provider_guard:main",
        "calibrate provider corroboration without independence assumptions",
        D,
        "provider-calibration",
        (),
        False,
        False,
    ),
    (
        "anchor-common-bias",
        "diagnostic anchor-common-bias",
        "prob4d.anchor_common_bias_study:main",
        "quantify independent-anchor power against shared visual bias",
        D,
        "visual-bias",
        (),
        False,
        False,
    ),
    (
        "cycle-guard-conformal",
        "diagnostic cycle-guard-conformal",
        "prob4d.cycle_guard_conformal_monte_carlo:main",
        "evaluate finite-sample calibration of the normalized cycle guard",
        D,
        "gauge-graph",
        (),
        False,
        False,
    ),
    (
        "cycle-guard-monte-carlo",
        "diagnostic cycle-guard-monte-carlo",
        "prob4d.cycle_guard_monte_carlo:main",
        "compare raw and uncertainty-normalized source cycle guards",
        D,
        "gauge-graph",
        (),
        False,
        False,
    ),
    (
        "gauge-graph",
        "diagnostic gauge-graph",
        "prob4d.causal_gauge_graph_ablation:main",
        "compare causal gauge-tree, graph, guarded fallback, and fixed-lag modes",
        D,
        "gauge-graph",
        (),
        False,
        False,
    ),
    (
        "gauge-graph-monte-carlo",
        "diagnostic gauge-graph-monte-carlo",
        "prob4d.causal_gauge_graph_monte_carlo:main",
        "run calibration-separated correlated gauge-graph Monte Carlo study",
        D,
        "gauge-graph",
        (),
        False,
        False,
    ),
    (
        "gauge-tree-prior-artifact",
        "gauge prior",
        "prob4d.gauge_tree_prior:main",
        "verify or explicitly densify portable sparse gauge-tree priors",
        S,
        "gauge-tree",
        (),
        False,
        False,
    ),
    (
        "joint-covariance",
        "diagnostic joint-covariance",
        "prob4d.joint_covariance_metrics:main",
        "evaluate matched residuals under conditional plus low-rank covariance",
        D,
        "observation-covariance",
        (),
        False,
        False,
    ),
    (
        "finite-sample-preflight",
        "diagnostic finite-sample-preflight",
        "prob4d.finite_sample_capability:preflight_cli",
        "check whether group counts support the requested finite-sample gate",
        D,
        "finite-sample-calibration",
        (),
        False,
        False,
    ),
    (
        "provider-support-envelope",
        "diagnostic provider-support-envelope",
        "prob4d.provider_support_envelope:main",
        "derive and verify outcome-blind contiguous provider support envelopes",
        D,
        "provider-readiness",
        (),
        False,
        False,
    ),
    (
        "source-covariance-localization",
        "diagnostic source-covariance-localization",
        "prob4d.source_covariance_localization:main",
        "localize source covariance failure before point-model development",
        D,
        "provider-readiness",
        (),
        False,
        False,
    ),
    (
        "target-free-rehearsal",
        "diagnostic target-free-rehearsal",
        "prob4d.target_free_rehearsal:main",
        "rehearse observation admission and adversarial controls without target access",
        D,
        "observation-contract",
        (),
        False,
        False,
    ),
    (
        "visual-bias-calibration",
        "diagnostic visual-bias-calibration",
        "prob4d.visual_bias_calibration:main",
        "fit or validate source-only coherent visual-bias calibration",
        D,
        "visual-bias",
        (),
        False,
        False,
    ),
    (
        "provider-evaluation",
        "evaluate provider",
        "prob4d.provider_evaluation:main",
        "evaluate paired provider artifacts across held-out groups",
        S,
        "provider-evaluation-v2",
        (),
        True,
        False,
    ),
    (
        "heldout-provider-promotion",
        "experiment heldout-provider",
        "prob4d.heldout_promotion:main",
        "freeze, run, and verify the held-out Prob4D-to-BayesianPhysTwin gate",
        E,
        "heldout-provider-promotion",
        (),
        True,
        False,
    ),
    (
        "fresh-provider-readiness",
        "experiment fresh-provider-readiness",
        "prob4d.fresh_provider_readiness:main",
        "compose ordered source gates and authorize one frozen target evaluation",
        E,
        "fresh-provider-readiness",
        (),
        True,
        False,
    ),
    (
        "material-identity",
        "identity",
        "prob4d.material_identity_cli:main",
        "fit, build, validate, and marginalize material-identity artifacts",
        S,
        "material-identity",
        (),
        False,
        False,
    ),
    (
        "motioncrafter-producer",
        "motioncrafter",
        "prob4d.motioncrafter_safe:main",
        "generate crash-safe MotionCrafter prediction products",
        S,
        "motioncrafter-producer",
        ("MotionCrafter checkout", "model checkpoints", "CUDA"),
        False,
        True,
    ),
    (
        "observation-export-guidance",
        "observation export",
        "prob4d.cli:_ambiguous_observation_export",
        "choose a calibrated or exploratory export mode explicitly",
        S,
        "provider-v2",
        (),
        False,
        False,
    ),
    (
        "observation-export-calibrated",
        "observation export-calibrated",
        "prob4d.provider_v2_cli:main_calibrated",
        "export a claim-bearing provider-v2 observation belief",
        S,
        "provider-v2",
        (),
        True,
        False,
    ),
    (
        "observation-export-exploratory",
        "observation export-exploratory",
        "prob4d.provider_v2_cli:main_exploratory",
        "export an explicitly exploratory provider-v2 observation belief",
        D,
        "provider-v2",
        (),
        False,
        False,
    ),
    (
        "observation-validate",
        "observation validate",
        "prob4d.observation_validation:main",
        "validate and summarize a portable observation belief",
        S,
        "observation-belief-v1",
        (),
        False,
        False,
    ),
    (
        "observation-bias-binding",
        "observation bias-binding",
        "prob4d.observation_bias_binding:main",
        "bind recursive observation factors to their visual-bias stream",
        S,
        "visual-bias",
        (),
        False,
        False,
    ),
    (
        "observation-visual-bias",
        "observation visual-bias",
        "prob4d.visual_bias:main",
        "validate explicit shared visual-bias nuisance sidecars",
        S,
        "visual-bias",
        (),
        False,
        False,
    ),
    (
        "observation-visual-bias-stream",
        "observation visual-bias-stream",
        "prob4d.visual_bias_stream:main",
        "validate recursive updates that share one visual-bias prior",
        S,
        "visual-bias",
        (),
        False,
        False,
    ),
    (
        "prediction-provider",
        "prediction",
        "prob4d.prediction_cli:main",
        "import and validate provider-neutral prediction manifests",
        S,
        "prediction-provider",
        (),
        False,
        False,
    ),
    (
        "project-identity",
        "project identity",
        "prob4d.project_identity:main",
        "print the stable project ID and repository aliases",
        S,
        "project-identity",
        (),
        False,
        False,
    ),
    (
        "provider-manifest",
        "provider manifest",
        "prob4d.provider_manifest_cli:main",
        "print a versioned observation-provider manifest",
        S,
        "provider-manifest",
        (),
        False,
        False,
    ),
    (
        "provider-prefix-admission",
        "provider prefix-admission",
        "prob4d.provider_prefix_admission:main",
        "bind support feasibility and calibration transport for one causal prefix",
        S,
        "provider-readiness",
        (),
        True,
        False,
    ),
    (
        "target-provider-admit",
        "provider target-admit",
        "prob4d.target_provider_admission_cli:main_admit",
        "bind target provider manifests before outcome evaluation",
        S,
        "target-provider-admission-v1",
        (),
        True,
        False,
    ),
    (
        "target-provider-verify",
        "provider target-verify",
        "prob4d.target_provider_admission_cli:main_verify",
        "verify a sealed target provider admission result",
        S,
        "target-provider-admission-v1",
        (),
        True,
        False,
    ),
    (
        "phystwin-evaluate",
        "phystwin evaluate",
        "prob4d.phystwin_experiment:main",
        "evaluate Prob4D against a PhysTwin trajectory",
        D,
        "phystwin-integration",
        ("PhysTwin dataset",),
        False,
        False,
    ),
    (
        "phystwin-state",
        "phystwin state",
        "prob4d.phystwin_state:main",
        "run the causal-prefix PhysTwin state audit",
        D,
        "phystwin-integration",
        ("PhysTwin dataset",),
        False,
        False,
    ),
    (
        "phystwin-uncertainty",
        "phystwin uncertainty",
        "prob4d.phystwin_uncertainty:main",
        "evaluate empirical visual-flow uncertainty",
        D,
        "phystwin-integration",
        ("PhysTwin dataset",),
        False,
        False,
    ),
    (
        "sintel-uncertainty",
        "sintel uncertainty",
        "prob4d.sintel_uncertainty:main",
        "run the held-out Sintel uncertainty analysis",
        D,
        "sintel-integration",
        ("Sintel dataset",),
        False,
        False,
    ),
    (
        "storage-benchmark",
        "storage benchmark",
        "prob4d.prediction_store_benchmark:main",
        "profile eager or memory-mapped prediction loading",
        D,
        "prediction-storage",
        (),
        False,
        False,
    ),
    (
        "storage-materialize",
        "storage materialize",
        "prob4d.prediction_store_cli:main_materialize",
        "create a content-addressed memory-mapped prediction store",
        S,
        "prediction-storage",
        (),
        False,
        False,
    ),
    (
        "storage-validate",
        "storage validate",
        "prob4d.prediction_store_cli:main_validate",
        "validate and summarize a memory-mapped prediction store",
        S,
        "prediction-storage",
        (),
        False,
        False,
    ),
    (
        "vggt-baseline",
        "vggt baseline",
        "prob4d.vggt_baseline:main",
        "export the matched VGGT baseline",
        E,
        "vggt-integration",
        ("VGGT checkout", "model checkpoint", "CUDA"),
        False,
        True,
    ),
)

COMMANDS: Final[tuple[CommandSpec, ...]] = tuple(_command(*row) for row in _COMMAND_ROWS)


def validate_registry(commands: Iterable[CommandSpec] = COMMANDS) -> None:
    """Reject malformed, ambiguous, or incomplete canonical entries."""

    command_ids: set[str] = set()
    routes: set[Route] = set()
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
        command_ids.add(command.command_id)
        routes.add(command.route)


validate_registry()

COMMANDS_BY_ID: Final = {command.command_id: command for command in COMMANDS}
COMMANDS_BY_ROUTE: Final = {command.route: command for command in COMMANDS}


def iter_commands(*, lifecycle: CommandLifecycle | None = None) -> tuple[CommandSpec, ...]:
    """Return registry entries in deterministic command-id order."""

    selected = (
        command for command in COMMANDS if lifecycle is None or command.lifecycle is lifecycle
    )
    return tuple(sorted(selected, key=lambda command: command.command_id))


def find_command(selector: str) -> CommandSpec | None:
    """Resolve a canonical command ID or current grouped route."""

    normalized = selector.strip()
    command = COMMANDS_BY_ID.get(normalized)
    if command is not None:
        return command
    route = tuple(normalized.removeprefix("prob4d ").split())
    return COMMANDS_BY_ROUTE.get(route)
