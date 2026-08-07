"""Portable, checksum-bound artifacts for sparse gauge-tree priors.

The numerical prior remains :class:`GaugeTreeSquareRootPriorV1`; this module only
adds a path-independent manifest identity and a checksum-bound ``.npz`` payload.
No dense joint covariance is serialized or materialized during verification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._gauge_tree_prior_artifact_read import load_gauge_tree_prior
from ._gauge_tree_prior_artifact_schema import (
    GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY,
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
    artifact_summary,
    gauge_tree_prior_artifact_id,
    sha256_file,
)
from ._gauge_tree_prior_artifact_write import write_gauge_tree_prior


def _verify_cli(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d gauge prior verify",
        description=(
            "Verify a portable sparse gauge-tree prior without densifying it."
        ),
    )
    parser.add_argument("manifest", type=Path)
    parsed = parser.parse_args(arguments)
    prior = load_gauge_tree_prior(parsed.manifest)
    print(json.dumps(artifact_summary(prior), sort_keys=True, indent=2))
    return 0


def _materialize_cli(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d gauge prior materialize",
        description=(
            "Explicitly materialize a verified sparse prior as a dense .npy file."
        ),
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-gauges", type=int, default=128)
    parsed = parser.parse_args(arguments)
    if parsed.output.exists():
        raise FileExistsError(parsed.output)
    prior = load_gauge_tree_prior(parsed.manifest)
    dense = prior.materialize_dense_covariance(
        maximum_gauges=parsed.maximum_gauges
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("xb") as stream:
        np.save(stream, dense, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                **artifact_summary(prior),
                "dense_output": str(parsed.output),
                "dense_output_sha256": sha256_file(parsed.output),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Verify or explicitly densify a portable gauge-tree prior artifact."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="prob4d gauge prior",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("verify", add_help=False)
    subparsers.add_parser("materialize", add_help=False)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.command == "verify":
        return _verify_cli(remaining)
    if parsed.command == "materialize":
        return _materialize_cli(remaining)
    parser.error("a command is required")
    return 2


__all__ = [
    "GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY",
    "GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA",
    "GAUGE_TREE_PRIOR_ARTIFACT_VERSION",
    "gauge_tree_prior_artifact_id",
    "load_gauge_tree_prior",
    "main",
    "write_gauge_tree_prior",
]
