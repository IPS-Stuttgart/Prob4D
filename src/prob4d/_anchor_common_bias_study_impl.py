"""Calibration-separated study of independent anchors against shared provider bias.

The study deliberately separates two failure modes:

* differential disagreement between two visual providers; and
* a coherent bias shared by both providers, which only an independent anchor can expose.

Complete simulated objects/sessions are the exchangeable units.  Thresholds are
fit on clean calibration groups and applied unchanged to disjoint target groups.
This module is controlled mechanism and acquisition-design evidence, not a real
provider-competence result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

_SCHEMA: Final = "prob4d.anchor-common-bias-study"
_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class StudyConfig:
    """Frozen Monte Carlo design."""

    seed: int = 20_260_806
    calibration_groups: int = 800
    target_groups: int = 1_000
    rows_per_group: int = 256
    dimension: int = 3
    miscoverage: float = 0.05
    row_quantile: float = 0.95
    provider_sigma: float = 1.0
    provider_cross_correlation: float = 0.75
    provider_specific_bias_sigma: float = 1.0
    shared_bias_sigma: float = 1.5
    shared_bias_row_fraction: float = 0.25
    anchor_drift_sigma: float = 1.0
    reference_anchor_sigma_ratio: float = 0.5
    reference_anchor_support_fraction: float = 0.20
    anchor_sigma_ratios: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    anchor_support_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 1.0)

    def validate(self) -> None:
        integer_fields = {
            "seed": self.seed,
            "calibration_groups": self.calibration_groups,
            "target_groups": self.target_groups,
            "rows_per_group": self.rows_per_group,
            "dimension": self.dimension,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.calibration_groups < 20:
            raise ValueError("calibration_groups must be at least 20")
        if self.target_groups < 20:
            raise ValueError("target_groups must be at least 20")
        if self.rows_per_group < 4:
            raise ValueError("rows_per_group must be at least 4")
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        for name, value in {
            "miscoverage": self.miscoverage,
            "row_quantile": self.row_quantile,
            "provider_sigma": self.provider_sigma,
            "provider_cross_correlation": self.provider_cross_correlation,
            "provider_specific_bias_sigma": self.provider_specific_bias_sigma,
            "shared_bias_sigma": self.shared_bias_sigma,
            "shared_bias_row_fraction": self.shared_bias_row_fraction,
            "anchor_drift_sigma": self.anchor_drift_sigma,
            "reference_anchor_sigma_ratio": self.reference_anchor_sigma_ratio,
            "reference_anchor_support_fraction": self.reference_anchor_support_fraction,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.miscoverage < 1.0:
            raise ValueError("miscoverage must lie strictly between zero and one")
        if not 0.0 < self.row_quantile <= 1.0:
            raise ValueError("row_quantile must lie in (0, 1]")
        if self.provider_sigma <= 0.0:
            raise ValueError("provider_sigma must be positive")
        if not 0.0 <= self.provider_cross_correlation < 1.0:
            raise ValueError("provider_cross_correlation must lie in [0, 1)")
        for name in (
            "provider_specific_bias_sigma",
            "shared_bias_sigma",
            "anchor_drift_sigma",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < self.shared_bias_row_fraction <= 1.0:
            raise ValueError("shared_bias_row_fraction must lie in (0, 1]")
        if not 0.0 < self.reference_anchor_support_fraction <= 1.0:
            raise ValueError("reference_anchor_support_fraction must lie in (0, 1]")
        if self.reference_anchor_sigma_ratio <= 0.0:
            raise ValueError("reference_anchor_sigma_ratio must be positive")
        _validate_grid("anchor_sigma_ratios", self.anchor_sigma_ratios)
        _validate_grid("anchor_support_fractions", self.anchor_support_fractions, upper=1.0)
        if self.reference_anchor_sigma_ratio not in self.anchor_sigma_ratios:
            raise ValueError("reference_anchor_sigma_ratio must occur in anchor_sigma_ratios")
        if self.reference_anchor_support_fraction not in self.anchor_support_fractions:
            raise ValueError(
                "reference_anchor_support_fraction must occur in anchor_support_fractions"
            )


def _validate_grid(name: str, values: tuple[float, ...], upper: float | None = None) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    previous = -math.inf
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} values must be real scalars")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} values must be finite and positive")
        if upper is not None and numeric > upper:
            raise ValueError(f"{name} values must not exceed {upper}")
        if numeric <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        previous = numeric


def finite_sample_upper_threshold(
    calibration_scores: np.ndarray, miscoverage: float
) -> tuple[float, int, float]:
    """Return the split-conformal upper order statistic and guaranteed bound."""

    values = np.asarray(calibration_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("calibration_scores must be a nonempty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration_scores must be finite")
    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    order = math.ceil((values.size + 1) * (1.0 - miscoverage))
    order = min(max(order, 1), values.size)
    threshold = float(np.partition(values, order - 1)[order - 1])
    bound = float((values.size + 1 - order) / (values.size + 1))
    return threshold, order, bound


def _direction(dimension: int) -> np.ndarray:
    return np.ones(dimension, dtype=np.float64) / math.sqrt(dimension)


def _correlated_provider_noise(
    rng: np.random.Generator,
    groups: int,
    rows: int,
    dimension: int,
    sigma: float,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    shared = rng.normal(size=(groups, rows, dimension))
    first_private = rng.normal(size=(groups, rows, dimension))
    second_private = rng.normal(size=(groups, rows, dimension))
    shared_scale = math.sqrt(correlation)
    private_scale = math.sqrt(1.0 - correlation)
    first = sigma * (shared_scale * shared + private_scale * first_private)
    second = sigma * (shared_scale * shared + private_scale * second_private)
    return first, second


def _higher_quantile(values: np.ndarray, quantile: float) -> np.ndarray:
    """Deterministic higher empirical quantile along the row axis."""

    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("values must have shape [groups, rows] with rows present")
    index = max(0, math.ceil(quantile * values.shape[1]) - 1)
    return np.partition(values, index, axis=1)[:, index]


def _differential_case_scores(
    rng: np.random.Generator,
    config: StudyConfig,
    groups: int,
    first_bias_sigma: float = 0.0,
    second_bias_sigma: float = 0.0,
    bias_row_fraction: float = 1.0,
) -> np.ndarray:
    first, second = _correlated_provider_noise(
        rng,
        groups,
        config.rows_per_group,
        config.dimension,
        config.provider_sigma,
        config.provider_cross_correlation,
    )
    difference = first - second
    if first_bias_sigma != second_bias_sigma:
        biased = rng.random((groups, config.rows_per_group)) < bias_row_fraction
        amplitude = (
            (first_bias_sigma - second_bias_sigma)
            * config.provider_sigma
            * math.sqrt(config.dimension)
        )
        difference += biased[:, :, None] * (_direction(config.dimension) * amplitude)
    variance = 2.0 * config.provider_sigma**2 * (1.0 - config.provider_cross_correlation)
    row_scores = np.sqrt(np.sum(np.square(difference), axis=2) / (config.dimension * variance))
    return _higher_quantile(row_scores, config.row_quantile)


def _anchor_common_case_scores(
    rng: np.random.Generator,
    config: StudyConfig,
    groups: int,
    anchor_sigma_ratio: float,
    anchor_support_fraction: float,
    shared_bias_sigma: float = 0.0,
    shared_bias_row_fraction: float = 1.0,
    anchor_drift_sigma: float = 0.0,
) -> np.ndarray:
    anchor_rows = max(1, int(round(config.rows_per_group * anchor_support_fraction)))
    first, second = _correlated_provider_noise(
        rng,
        groups,
        anchor_rows,
        config.dimension,
        config.provider_sigma,
        config.provider_cross_correlation,
    )
    anchor_sigma = anchor_sigma_ratio * config.provider_sigma
    anchor = anchor_sigma * rng.normal(size=(groups, anchor_rows, config.dimension))
    common_residual = 0.5 * (first + second) - anchor
    direction = _direction(config.dimension)
    if shared_bias_sigma > 0.0:
        biased = rng.random((groups, anchor_rows)) < shared_bias_row_fraction
        amplitude = shared_bias_sigma * config.provider_sigma * math.sqrt(config.dimension)
        common_residual += biased[:, :, None] * (direction * amplitude)
    if anchor_drift_sigma > 0.0:
        amplitude = anchor_drift_sigma * config.provider_sigma * math.sqrt(config.dimension)
        common_residual -= direction * amplitude
    variance = (
        config.provider_sigma**2 * (1.0 + config.provider_cross_correlation) / 2.0 + anchor_sigma**2
    )
    row_scores = np.sqrt(np.sum(np.square(common_residual), axis=2) / (config.dimension * variance))
    return _higher_quantile(row_scores, config.row_quantile)


def _rate(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(scores) > threshold))


def _seed_sequence(seed: int, *indices: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *indices]))


def run_study(config: StudyConfig, *, source_revision: str) -> dict[str, Any]:
    """Execute the frozen study and return a content-addressed report."""

    config.validate()
    _validate_source_revision(source_revision)

    differential_calibration = _differential_case_scores(
        _seed_sequence(config.seed, 1), config, config.calibration_groups
    )
    differential_threshold, differential_order, differential_bound = finite_sample_upper_threshold(
        differential_calibration, config.miscoverage
    )
    differential_clean = _differential_case_scores(
        _seed_sequence(config.seed, 2), config, config.target_groups
    )
    differential_provider_specific = _differential_case_scores(
        _seed_sequence(config.seed, 3),
        config,
        config.target_groups,
        first_bias_sigma=config.provider_specific_bias_sigma,
        bias_row_fraction=config.shared_bias_row_fraction,
    )
    differential_shared = _differential_case_scores(
        _seed_sequence(config.seed, 4),
        config,
        config.target_groups,
        first_bias_sigma=config.shared_bias_sigma,
        second_bias_sigma=config.shared_bias_sigma,
        bias_row_fraction=config.shared_bias_row_fraction,
    )

    power_grid: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for sigma_index, anchor_sigma_ratio in enumerate(config.anchor_sigma_ratios):
        for support_index, anchor_support_fraction in enumerate(config.anchor_support_fractions):
            calibration = _anchor_common_case_scores(
                _seed_sequence(config.seed, 100, sigma_index, support_index),
                config,
                config.calibration_groups,
                anchor_sigma_ratio,
                anchor_support_fraction,
            )
            threshold, order, bound = finite_sample_upper_threshold(calibration, config.miscoverage)
            clean = _anchor_common_case_scores(
                _seed_sequence(config.seed, 101, sigma_index, support_index),
                config,
                config.target_groups,
                anchor_sigma_ratio,
                anchor_support_fraction,
            )
            shared = _anchor_common_case_scores(
                _seed_sequence(config.seed, 102, sigma_index, support_index),
                config,
                config.target_groups,
                anchor_sigma_ratio,
                anchor_support_fraction,
                shared_bias_sigma=config.shared_bias_sigma,
                shared_bias_row_fraction=config.shared_bias_row_fraction,
            )
            drift = _anchor_common_case_scores(
                _seed_sequence(config.seed, 103, sigma_index, support_index),
                config,
                config.target_groups,
                anchor_sigma_ratio,
                anchor_support_fraction,
                anchor_drift_sigma=config.anchor_drift_sigma,
            )
            record = {
                "anchor_sigma_ratio": float(anchor_sigma_ratio),
                "anchor_support_fraction": float(anchor_support_fraction),
                "anchor_rows": max(1, int(round(config.rows_per_group * anchor_support_fraction))),
                "threshold": threshold,
                "conformal_order": order,
                "guaranteed_miscoverage_upper_bound": bound,
                "clean_false_rejection_rate": _rate(clean, threshold),
                "shared_bias_rejection_rate": _rate(shared, threshold),
                "anchor_drift_rejection_rate": _rate(drift, threshold),
            }
            power_grid.append(record)
            if (
                anchor_sigma_ratio == config.reference_anchor_sigma_ratio
                and anchor_support_fraction == config.reference_anchor_support_fraction
            ):
                reference = record

    if reference is None:  # guarded by validation
        raise RuntimeError("reference anchor design was not evaluated")

    differential = {
        "threshold": differential_threshold,
        "conformal_order": differential_order,
        "guaranteed_miscoverage_upper_bound": differential_bound,
        "clean_false_rejection_rate": _rate(differential_clean, differential_threshold),
        "provider_specific_bias_rejection_rate": _rate(
            differential_provider_specific, differential_threshold
        ),
        "shared_common_bias_rejection_rate": _rate(differential_shared, differential_threshold),
    }
    gates = {
        "differential_clean_false_rejection_le_0_08": (
            differential["clean_false_rejection_rate"] <= 0.08
        ),
        "provider_specific_detection_ge_0_95": (
            differential["provider_specific_bias_rejection_rate"] >= 0.95
        ),
        "differential_shared_bias_rejection_le_0_10": (
            differential["shared_common_bias_rejection_rate"] <= 0.10
        ),
        "reference_anchor_clean_false_rejection_le_0_08": (
            reference["clean_false_rejection_rate"] <= 0.08
        ),
        "reference_anchor_shared_bias_detection_ge_0_90": (
            reference["shared_bias_rejection_rate"] >= 0.90
        ),
        "reference_anchor_drift_rejection_ge_0_95": (
            reference["anchor_drift_rejection_rate"] >= 0.95
        ),
    }
    report: dict[str, Any] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "study_type": "calibration-target-separated-controlled-mechanism",
        "source_revision": source_revision,
        "config": _config_json(config),
        "differential_provider_guard": differential,
        "anchor_common_mode_power_grid": power_grid,
        "reference_anchor_design": reference,
        "registered_gates": gates,
        "registered_decision": (
            "pass-indepent-anchor-mechanism-study"
            if all(gates.values())
            else "completed-negative-independent-anchor-mechanism-study"
        ),
        "interpretation": {
            "differential_guard_role": (
                "detect provider-specific disagreement without claiming absolute correctness"
            ),
            "anchor_guard_role": (
                "reject visual-versus-independent-anchor inconsistency, "
                "including shared visual bias"
            ),
            "fault_attribution": (
                "rejection does not identify whether the visual providers or anchor are wrong"
            ),
        },
        "claim_boundary": [
            "controlled Gaussian mechanism and acquisition-design evidence only",
            "no real provider, physical object, target dataset, or "
            "BayesianPhysTwin outcome was used",
            "independence, covariance, bias occupancy, and exchangeability must be "
            "re-established prospectively",
            "a passing anchor gate does not replace the downstream baseline-relative regret guard",
        ],
    }
    report["report_id"] = _report_id(report)
    return report


def _validate_source_revision(source_revision: str) -> None:
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        raise ValueError("source_revision must be a full 40-character Git SHA")
    if not all(character in "0123456789abcdef" for character in source_revision):
        raise ValueError("source_revision must be lowercase hexadecimal")


def _config_json(config: StudyConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["anchor_sigma_ratios"] = list(config.anchor_sigma_ratios)
    payload["anchor_support_fractions"] = list(config.anchor_support_fractions)
    return payload


def _report_id(report: dict[str, Any]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_report(report: dict[str, Any]) -> None:
    """Recompute the report identity and enforce the closed top-level schema."""

    expected_keys = {
        "schema",
        "schema_version",
        "study_type",
        "source_revision",
        "config",
        "differential_provider_guard",
        "anchor_common_mode_power_grid",
        "reference_anchor_design",
        "registered_gates",
        "registered_decision",
        "interpretation",
        "claim_boundary",
        "report_id",
    }
    if set(report) != expected_keys:
        raise ValueError("anchor-common-bias report has unexpected top-level keys")
    if report["schema"] != _SCHEMA or report["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("anchor-common-bias report schema mismatch")
    report_id = report["report_id"]
    if not isinstance(report_id, str) or len(report_id) != 64:
        raise ValueError("report_id must be a SHA-256 hex string")
    if not all(character in "0123456789abcdef" for character in report_id):
        raise ValueError("report_id must be lowercase hexadecimal")
    if _report_id(report) != report_id:
        raise ValueError("anchor-common-bias report identity mismatch")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(report: dict[str, Any], output_json: Path, output_markdown: Path) -> None:
    validate_report(report)
    _atomic_write(
        output_json,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(output_markdown, render_markdown(report))


def render_markdown(report: dict[str, Any]) -> str:
    validate_report(report)
    differential = report["differential_provider_guard"]
    reference = report["reference_anchor_design"]
    lines = [
        "# Independent-anchor common-bias study",
        "",
        f"Decision: **{report['registered_decision']}**",
        "",
        "## Registered reference result",
        "",
        "| Endpoint | Rate |",
        "| --- | ---: |",
        (
            "| Differential clean false rejection | "
            f"{100.0 * differential['clean_false_rejection_rate']:.2f}% |"
        ),
        (
            "| Provider-specific corruption rejected | "
            f"{100.0 * differential['provider_specific_bias_rejection_rate']:.2f}% |"
        ),
        (
            "| Shared bias rejected by provider disagreement alone | "
            f"{100.0 * differential['shared_common_bias_rejection_rate']:.2f}% |"
        ),
        (
            "| Reference anchor clean false rejection | "
            f"{100.0 * reference['clean_false_rejection_rate']:.2f}% |"
        ),
        (
            "| Shared bias rejected with reference anchor | "
            f"{100.0 * reference['shared_bias_rejection_rate']:.2f}% |"
        ),
        (
            "| Anchor drift inconsistency rejected | "
            f"{100.0 * reference['anchor_drift_rejection_rate']:.2f}% |"
        ),
        "",
        "The reference design uses "
        f"{100.0 * reference['anchor_support_fraction']:.0f}% row support and "
        f"anchor sigma {reference['anchor_sigma_ratio']:.2f} times the provider sigma.",
        "",
        "## Power grid: shared coherent visual bias rejection",
        "",
        "| Anchor sigma / provider sigma | Anchor support | Anchor rows | "
        "Rejection | Clean false rejection |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["anchor_common_mode_power_grid"]:
        lines.append(
            "| "
            f"{record['anchor_sigma_ratio']:.2f} | "
            f"{100.0 * record['anchor_support_fraction']:.0f}% | "
            f"{record['anchor_rows']} | "
            f"{100.0 * record['shared_bias_rejection_rate']:.2f}% | "
            f"{100.0 * record['clean_false_rejection_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            *[f"- {item}" for item in report["claim_boundary"]],
            "",
            f"Report ID: `{report['report_id']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a calibration-separated study of differential provider disagreement "
            "and independent-anchor detection of shared coherent visual bias."
        )
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=StudyConfig.seed)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller deterministic design for smoke tests only.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    config = StudyConfig(seed=arguments.seed)
    if arguments.quick:
        config = StudyConfig(
            seed=arguments.seed,
            calibration_groups=80,
            target_groups=120,
            rows_per_group=64,
            anchor_sigma_ratios=(0.5, 1.0),
            anchor_support_fractions=(0.20, 1.0),
            reference_anchor_sigma_ratio=0.5,
            reference_anchor_support_fraction=0.20,
        )
    report = run_study(config, source_revision=arguments.source_revision)
    write_report(report, arguments.output_json, arguments.output_markdown)
    return 0 if all(report["registered_gates"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
