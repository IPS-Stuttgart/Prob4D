"""Public interface for the independent-anchor common-bias study."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from ._anchor_common_bias_study_impl import (
    StudyConfig,
    _parse_arguments,
    _report_id,
    finite_sample_upper_threshold,
    render_markdown,
    validate_report,
    write_report,
)

_LEGACY_PASS_DECISION: Final = "pass-indepent-anchor-mechanism-study"
_PASS_DECISION: Final = "pass-independent-anchor-mechanism-study"

__all__ = [
    "StudyConfig",
    "finite_sample_upper_threshold",
    "main",
    "render_markdown",
    "run_study",
    "validate_report",
    "write_report",
]


def run_study(config: StudyConfig, *, source_revision: str) -> dict[str, Any]:
    """Run the frozen study with the canonical registered decision identifier."""

    from ._anchor_common_bias_study_impl import run_study as run_implementation

    report = run_implementation(config, source_revision=source_revision)
    if report["registered_decision"] == _LEGACY_PASS_DECISION:
        report["registered_decision"] = _PASS_DECISION
        report["report_id"] = _report_id(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line study through the public interface."""

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
