"""Target-free finite-sample capability audit for held-out promotion protocols.

The audit converts complete calibration and target object/session counts into
explicit split-conformal ranks and small-target diagnostic resolution. It uses
only sealed metadata and never opens provider payloads or target outcomes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ._finite_sample_capability_build import (
    build_finite_sample_capability,
    build_finite_sample_capability_from_cohort_binding,
)
from ._finite_sample_capability_common import (
    DEFAULT_COVERAGE_LEVELS,
    FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY,
    FINITE_SAMPLE_CAPABILITY_SCHEMA,
    FINITE_SAMPLE_CAPABILITY_VERSION,
    coverage,
    split_conformal_level,
)
from ._finite_sample_capability_io import (
    finite_sample_capability_from_dict,
    load_finite_sample_capability,
)
from ._finite_sample_capability_model import FiniteSampleCapabilityV1
from ._finite_sample_capability_output import (
    render_finite_sample_capability_markdown,
    write_finite_sample_capability,
)
from ._finite_sample_capability_records import CalibrationStratumV1
from ._heldout_promotion_lock import load_promotion_lock
from .deform360_cohort_binding import load_deform360_cohort_binding


def _coverage_argument(value: str) -> float:
    try:
        return coverage(float(value), name="coverage")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def preflight_cli(argv: Sequence[str] | None = None) -> int:
    """Run the installed finite-sample preflight command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--cohort-binding", type=Path)
    parser.add_argument(
        "--coverage",
        type=_coverage_argument,
        action="append",
        dest="coverages",
        help="requested nominal coverage; may be repeated (default: 0.80, 0.90, 0.95)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--require-primary-finite", action="store_true")
    parser.add_argument("--require-strata-finite", action="store_true")
    arguments = parser.parse_args(argv)

    lock = load_promotion_lock(arguments.lock)
    requested = DEFAULT_COVERAGE_LEVELS if arguments.coverages is None else arguments.coverages
    if arguments.cohort_binding is None:
        report = build_finite_sample_capability(lock, coverage_levels=requested)
    else:
        report = build_finite_sample_capability_from_cohort_binding(
            lock,
            load_deform360_cohort_binding(arguments.cohort_binding),
            coverage_levels=requested,
        )
    write_finite_sample_capability(
        report,
        arguments.output,
        markdown=arguments.markdown,
    )
    print(
        json.dumps(
            {
                "capability_id": report.capability_id,
                "primary_levels_finite": report.primary_levels_finite,
                "stratum_levels_finite": report.stratum_levels_finite,
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    if arguments.require_primary_finite and not report.primary_levels_finite:
        return 3
    if arguments.require_strata_finite and report.stratum_levels_finite is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(preflight_cli())


__all__ = [
    "DEFAULT_COVERAGE_LEVELS",
    "FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY",
    "FINITE_SAMPLE_CAPABILITY_SCHEMA",
    "FINITE_SAMPLE_CAPABILITY_VERSION",
    "CalibrationStratumV1",
    "FiniteSampleCapabilityV1",
    "build_finite_sample_capability",
    "build_finite_sample_capability_from_cohort_binding",
    "finite_sample_capability_from_dict",
    "load_finite_sample_capability",
    "preflight_cli",
    "render_finite_sample_capability_markdown",
    "split_conformal_level",
    "write_finite_sample_capability",
]
