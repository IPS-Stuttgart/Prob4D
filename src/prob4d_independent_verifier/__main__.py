"""Command-line entry point for independent observation verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .observation_belief import (
    DEFAULT_LIMITS,
    VerificationLimits,
    verify_observation_belief,
    write_verification_report,
)


def _mib(value: str) -> int:
    try:
        amount = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer number of MiB") from error
    if amount <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return amount * 1024**2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prob4d_independent_verifier",
        description=(
            "Independently validate the closed ObservationBeliefV1 NPZ contract "
            "without importing prob4d."
        ),
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--max-archive-mib",
        type=_mib,
        default=DEFAULT_LIMITS.max_archive_bytes,
    )
    parser.add_argument(
        "--max-uncompressed-mib",
        type=_mib,
        default=DEFAULT_LIMITS.max_uncompressed_bytes,
    )
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_LIMITS.max_compression_ratio,
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    limits = VerificationLimits(
        max_members=DEFAULT_LIMITS.max_members,
        max_archive_bytes=arguments.max_archive_mib,
        max_uncompressed_bytes=arguments.max_uncompressed_mib,
        max_compression_ratio=arguments.max_compression_ratio,
    )
    report = verify_observation_belief(arguments.artifact, limits=limits)
    if arguments.report is not None:
        write_verification_report(
            arguments.report,
            report,
            overwrite=arguments.overwrite,
        )
    print(
        json.dumps(
            report.to_dict(),
            sort_keys=True,
            indent=None if arguments.compact else 2,
            separators=(",", ":") if arguments.compact else None,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
