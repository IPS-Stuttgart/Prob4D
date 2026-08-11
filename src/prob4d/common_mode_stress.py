"""Calibration-separated synthetic stress test for coherent visual bias.

The benchmark isolates the failure mode that overlap disagreement cannot reveal:
all prediction members share a coherent error while disagreeing only through
small independent noise.  It compares a visual update that ignores this shared
bias with the same update after marginalizing a separately calibrated bias prior.
The study is controlled mechanism evidence, not real-provider promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from ._atomic_file import atomic_write_text
from ._immutable_json import plain_json
from ._strict_json import load_json_object
from .source_diagnostics import audit_common_mode_failures

COMMON_MODE_STRESS_REPORT_SCHEMA: Final = "prob4d.common-mode-stress-report.v1"
_CHI2_DF3_90: Final = 6.251388631170325
_CHI2_DF3_95: Final = 7.814727903251179


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_real(value: object, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw.item())
    if not np.isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, payload, overwrite=False)


@dataclass(frozen=True)
class CommonModeStressConfig:
    """Predeclared sizes and noise scales for the controlled study."""

    clean_calibration_groups: int = 1200
    bias_calibration_groups: int = 1200
    target_groups: int = 2000
    members_per_group: int = 4
    independent_noise_std_m: float = 0.002
    coherent_bias_std_m: float = 0.012
    physical_prior_std_m: float = 0.008
    state_scale_m: float = 0.05
    seed: int = 20260806
    disagreement_quantile: float = 0.9
    error_quantile: float = 0.9
    minimum_failure_rate: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "clean_calibration_groups",
            "bias_calibration_groups",
            "target_groups",
            "members_per_group",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name=name),
            )
        if self.members_per_group < 2:
            raise ValueError("members_per_group must be at least two")
        for name in (
            "independent_noise_std_m",
            "physical_prior_std_m",
            "state_scale_m",
        ):
            object.__setattr__(
                self,
                name,
                _positive_real(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "coherent_bias_std_m",
            _positive_real(
                self.coherent_bias_std_m,
                name="coherent_bias_std_m",
                allow_zero=True,
            ),
        )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, name="seed"))
        for name in ("disagreement_quantile", "error_quantile"):
            value = _positive_real(getattr(self, name), name=name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
            object.__setattr__(self, name, value)
        failure_rate = _positive_real(
            self.minimum_failure_rate,
            name="minimum_failure_rate",
            allow_zero=True,
        )
        if failure_rate > 1.0:
            raise ValueError("minimum_failure_rate must not exceed one")
        object.__setattr__(self, "minimum_failure_rate", failure_rate)


@dataclass(frozen=True)
class FusionMethodMetrics:
    """Complete guarded-policy metrics for one toy physical update."""

    raw_rmse_m: float
    deployed_rmse_m: float
    physical_baseline_rmse_m: float
    accepted_count: int
    rejected_count: int
    harmful_accepted_count: int
    exact_fallback_reproduced_count: int
    complete_policy_coverage_90: float
    accepted_coverage_90: float | None
    mean_complete_policy_width_m: float
    mean_nll: float
    worst_group_regression_m: float

    def to_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CommonModeStressReport:
    """Deterministic benchmark result and frozen interpretation gates."""

    config: CommonModeStressConfig
    disagreement_threshold_m: float
    error_threshold_m: float
    calibrated_bias_variance_m2: float
    common_mode_audit: Mapping[str, int | float]
    methods: Mapping[str, FusionMethodMetrics]
    gates: Mapping[str, bool]
    decision: str
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if set(self.methods) != {"naive-independent", "explicit-shared-bias"}:
            raise ValueError("stress report method set changed")
        required_gates = {
            "coherent_failure_exposed",
            "bias_aware_coverage_improves",
            "bias_aware_deployed_rmse_not_worse",
            "bias_aware_beats_physical_fallback",
            "bias_aware_harm_not_worse",
            "exact_fallback_reproduced",
        }
        if set(self.gates) != required_gates:
            raise ValueError("stress report gate set changed")
        expected_decision = (
            "pass-controlled-common-mode-mechanism"
            if all(self.gates.values())
            else "controlled-mechanism-gate-failed"
        )
        if self.decision != expected_decision:
            raise ValueError("stress report decision contradicts its gates")
        expected_id = _sha256_json(self.identity_record())
        if self.artifact_id is not None and self.artifact_id != expected_id:
            raise ValueError("common-mode stress report artifact ID mismatch")
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, object]:
        return {
            "schema": COMMON_MODE_STRESS_REPORT_SCHEMA,
            "config": asdict(self.config),
            "disagreement_threshold_m": self.disagreement_threshold_m,
            "error_threshold_m": self.error_threshold_m,
            "calibrated_bias_variance_m2": self.calibrated_bias_variance_m2,
            "common_mode_audit": dict(self.common_mode_audit),
            "methods": {
                name: metrics.to_record() for name, metrics in sorted(self.methods.items())
            },
            "gates": dict(sorted(self.gates.items())),
            "decision": self.decision,
            "claim_boundary": (
                "controlled synthetic mechanism evidence only; no real-provider, "
                "BayesianPhysTwin, Causal4D, calibration-transfer, or safety claim"
            ),
        }

    def to_record(self) -> dict[str, object]:
        return {**self.identity_record(), "artifact_id": self.artifact_id}


def _simulate_panel(
    generator: np.random.Generator,
    *,
    groups: int,
    members: int,
    state_scale: float,
    independent_noise_std: float,
    coherent_bias_std: float,
    physical_prior_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = generator.normal(0.0, state_scale, size=(groups, 3))
    coherent_bias = generator.normal(0.0, coherent_bias_std, size=(groups, 3))
    independent_noise = generator.normal(
        0.0,
        independent_noise_std,
        size=(groups, members, 3),
    )
    predictions = truth[:, None, :] + coherent_bias[:, None, :] + independent_noise
    physical = truth + generator.normal(0.0, physical_prior_std, size=(groups, 3))
    return truth, predictions, physical


def _member_disagreement(predictions: np.ndarray) -> np.ndarray:
    centered = predictions - np.mean(predictions, axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1))


def _coverage_and_nll(
    error: np.ndarray,
    variance: np.ndarray,
) -> tuple[float, float, float]:
    squared_error = np.sum(error * error, axis=1)
    normalized = squared_error / variance
    coverage = float(np.mean(normalized <= _CHI2_DF3_90))
    width = float(np.mean(np.sqrt(_CHI2_DF3_90 * variance)))
    nll = float(
        np.mean(
            0.5
            * (
                3.0 * np.log(2.0 * np.pi * variance)
                + squared_error / variance
            )
        )
    )
    return coverage, width, nll


def _evaluate_method(
    *,
    truth: np.ndarray,
    visual_mean: np.ndarray,
    physical: np.ndarray,
    physical_variance: float,
    visual_variance: float,
) -> FusionMethodMetrics:
    innovation = visual_mean - physical
    innovation_variance = physical_variance + visual_variance
    nis = np.sum(innovation * innovation, axis=1) / innovation_variance
    accepted = nis <= _CHI2_DF3_95

    posterior_variance = 1.0 / (1.0 / physical_variance + 1.0 / visual_variance)
    posterior_mean = posterior_variance * (
        physical / physical_variance + visual_mean / visual_variance
    )
    deployed = np.where(accepted[:, None], posterior_mean, physical)
    deployed_variance = np.where(accepted, posterior_variance, physical_variance)

    raw_error = posterior_mean - truth
    deployed_error = deployed - truth
    baseline_error = physical - truth
    raw_rmse = float(np.sqrt(np.mean(np.sum(raw_error * raw_error, axis=1))))
    deployed_rmse = float(
        np.sqrt(np.mean(np.sum(deployed_error * deployed_error, axis=1)))
    )
    baseline_rmse = float(
        np.sqrt(np.mean(np.sum(baseline_error * baseline_error, axis=1)))
    )
    deployed_loss = np.linalg.norm(deployed_error, axis=1)
    baseline_loss = np.linalg.norm(baseline_error, axis=1)
    harmful = accepted & (deployed_loss > baseline_loss)
    rejected = ~accepted
    exact_fallback = np.all(deployed[rejected] == physical[rejected], axis=1)
    coverage, width, nll = _coverage_and_nll(deployed_error, deployed_variance)
    if np.any(accepted):
        accepted_error = raw_error[accepted]
        accepted_variance = np.full(accepted_error.shape[0], posterior_variance)
        accepted_coverage, _, _ = _coverage_and_nll(
            accepted_error,
            accepted_variance,
        )
    else:
        accepted_coverage = None
    regression = deployed_loss - baseline_loss
    return FusionMethodMetrics(
        raw_rmse_m=raw_rmse,
        deployed_rmse_m=deployed_rmse,
        physical_baseline_rmse_m=baseline_rmse,
        accepted_count=int(np.count_nonzero(accepted)),
        rejected_count=int(accepted.size - np.count_nonzero(accepted)),
        harmful_accepted_count=int(np.count_nonzero(harmful)),
        exact_fallback_reproduced_count=int(np.count_nonzero(exact_fallback)),
        complete_policy_coverage_90=coverage,
        accepted_coverage_90=accepted_coverage,
        mean_complete_policy_width_m=width,
        mean_nll=nll,
        worst_group_regression_m=float(np.max(regression)),
    )


def run_common_mode_stress(
    config: CommonModeStressConfig | None = None,
) -> CommonModeStressReport:
    """Run clean calibration, bias calibration, and a disjoint target panel."""

    actual = CommonModeStressConfig() if config is None else config
    generator = np.random.default_rng(actual.seed)
    clean_truth, clean_predictions, _ = _simulate_panel(
        generator,
        groups=actual.clean_calibration_groups,
        members=actual.members_per_group,
        state_scale=actual.state_scale_m,
        independent_noise_std=actual.independent_noise_std_m,
        coherent_bias_std=0.0,
        physical_prior_std=actual.physical_prior_std_m,
    )
    clean_mean = np.mean(clean_predictions, axis=1)
    clean_disagreement = _member_disagreement(clean_predictions)
    clean_error = np.linalg.norm(clean_mean - clean_truth, axis=1)
    disagreement_threshold = float(
        np.quantile(
            clean_disagreement,
            actual.disagreement_quantile,
            method="higher",
        )
    )
    error_threshold = float(
        np.quantile(clean_error, actual.error_quantile, method="higher")
    )

    bias_truth, bias_predictions, _ = _simulate_panel(
        generator,
        groups=actual.bias_calibration_groups,
        members=actual.members_per_group,
        state_scale=actual.state_scale_m,
        independent_noise_std=actual.independent_noise_std_m,
        coherent_bias_std=actual.coherent_bias_std_m,
        physical_prior_std=actual.physical_prior_std_m,
    )
    bias_mean_error = np.mean(bias_predictions, axis=1) - bias_truth
    empirical_mean_error_variance = float(
        np.mean(np.var(bias_mean_error, axis=0, ddof=1))
    )
    independent_mean_variance = (
        actual.independent_noise_std_m**2 / actual.members_per_group
    )
    calibrated_bias_variance = max(
        empirical_mean_error_variance - independent_mean_variance,
        0.0,
    )

    target_truth, target_predictions, target_physical = _simulate_panel(
        generator,
        groups=actual.target_groups,
        members=actual.members_per_group,
        state_scale=actual.state_scale_m,
        independent_noise_std=actual.independent_noise_std_m,
        coherent_bias_std=actual.coherent_bias_std_m,
        physical_prior_std=actual.physical_prior_std_m,
    )
    target_mean = np.mean(target_predictions, axis=1)
    target_disagreement = _member_disagreement(target_predictions)
    target_error = np.linalg.norm(target_mean - target_truth, axis=1)
    audit = audit_common_mode_failures(
        target_disagreement,
        target_error,
        disagreement_threshold=disagreement_threshold,
        error_threshold=error_threshold,
    )

    physical_variance = actual.physical_prior_std_m**2
    naive_visual_variance = independent_mean_variance
    aware_visual_variance = independent_mean_variance + calibrated_bias_variance
    if aware_visual_variance <= 0.0:
        aware_visual_variance = float(np.finfo(np.float64).tiny)
    methods = {
        "naive-independent": _evaluate_method(
            truth=target_truth,
            visual_mean=target_mean,
            physical=target_physical,
            physical_variance=physical_variance,
            visual_variance=naive_visual_variance,
        ),
        "explicit-shared-bias": _evaluate_method(
            truth=target_truth,
            visual_mean=target_mean,
            physical=target_physical,
            physical_variance=physical_variance,
            visual_variance=aware_visual_variance,
        ),
    }
    naive = methods["naive-independent"]
    aware = methods["explicit-shared-bias"]
    naive_shortfall = abs(naive.complete_policy_coverage_90 - 0.9)
    aware_shortfall = abs(aware.complete_policy_coverage_90 - 0.9)
    gates = {
        "coherent_failure_exposed": (
            audit.low_disagreement_high_error_rate >= actual.minimum_failure_rate
        ),
        "bias_aware_coverage_improves": aware_shortfall < naive_shortfall,
        "bias_aware_deployed_rmse_not_worse": (
            aware.deployed_rmse_m <= naive.deployed_rmse_m + 1e-15
        ),
        "bias_aware_beats_physical_fallback": (
            aware.deployed_rmse_m <= aware.physical_baseline_rmse_m + 1e-15
        ),
        "bias_aware_harm_not_worse": (
            aware.harmful_accepted_count <= naive.harmful_accepted_count
        ),
        "exact_fallback_reproduced": all(
            metrics.exact_fallback_reproduced_count == metrics.rejected_count
            for metrics in methods.values()
        ),
    }
    decision = (
        "pass-controlled-common-mode-mechanism"
        if all(gates.values())
        else "controlled-mechanism-gate-failed"
    )
    return CommonModeStressReport(
        config=actual,
        disagreement_threshold_m=disagreement_threshold,
        error_threshold_m=error_threshold,
        calibrated_bias_variance_m2=calibrated_bias_variance,
        common_mode_audit=audit.to_dict(),
        methods=methods,
        gates=gates,
        decision=decision,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d diagnostic common-mode-stress",
        description="Run the calibration-separated coherent visual-bias benchmark.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=CommonModeStressConfig.seed)
    parser.add_argument(
        "--clean-calibration-groups",
        type=int,
        default=CommonModeStressConfig.clean_calibration_groups,
    )
    parser.add_argument(
        "--bias-calibration-groups",
        type=int,
        default=CommonModeStressConfig.bias_calibration_groups,
    )
    parser.add_argument(
        "--target-groups",
        type=int,
        default=CommonModeStressConfig.target_groups,
    )
    parser.add_argument(
        "--members-per-group",
        type=int,
        default=CommonModeStressConfig.members_per_group,
    )
    parser.add_argument(
        "--coherent-bias-std-m",
        type=float,
        default=CommonModeStressConfig.coherent_bias_std_m,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    config = CommonModeStressConfig(
        clean_calibration_groups=arguments.clean_calibration_groups,
        bias_calibration_groups=arguments.bias_calibration_groups,
        target_groups=arguments.target_groups,
        members_per_group=arguments.members_per_group,
        coherent_bias_std_m=arguments.coherent_bias_std_m,
        seed=arguments.seed,
    )
    report = run_common_mode_stress(config)
    destination = Path(arguments.output)
    record = report.to_record()
    try:
        _atomic_write_json(destination, record)
    except FileExistsError:
        existing = load_json_object(destination, name="common-mode stress report")
        if existing != record:
            raise ValueError(
                "refusing to replace a different common-mode stress report"
            ) from None
    print(json.dumps(report.to_record(), indent=2, sort_keys=True))
    return 0 if report.decision == "pass-controlled-common-mode-mechanism" else 3


__all__ = [
    "COMMON_MODE_STRESS_REPORT_SCHEMA",
    "CommonModeStressConfig",
    "CommonModeStressReport",
    "FusionMethodMetrics",
    "main",
    "run_common_mode_stress",
]


if __name__ == "__main__":
    raise SystemExit(main())
