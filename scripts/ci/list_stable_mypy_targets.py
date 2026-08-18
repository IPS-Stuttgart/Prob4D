#!/usr/bin/env python3
"""Print the authoritative stable-interface MyPy target set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Sequence

ROOT: Final = Path(__file__).resolve().parents[2]
_EXPLICIT_TARGETS: Final[tuple[str, ...]] = (
    "src/prob4d/_provider_evaluation_provider_neutral.py",
    "src/prob4d/_provider_export_core.py",
    "src/prob4d/_version.py",
    "src/prob4d/api/v2.py",
    "src/prob4d/calibration_compatibility.py",
    "src/prob4d/cli.py",
    "src/prob4d/composition_jacobian.py",
    "src/prob4d/covariance_root.py",
    "src/prob4d/observation_factor_stream.py",
    "src/prob4d/prediction_store.py",
    "src/prob4d/prediction_store_benchmark.py",
    "src/prob4d/prediction_store_cli.py",
    "src/prob4d/project_identity.py",
    "src/prob4d/provider_attestation.py",
    "src/prob4d/provider_evaluation_identity.py",
    "src/prob4d/provider_manifest.py",
    "src/prob4d/provider_manifest_cli.py",
    "src/prob4d/provider_v1.py",
    "src/prob4d/provider_v2.py",
    "src/prob4d/provider_v2_cli.py",
    "src/prob4d/provider_v2_factor_bundle.py",
    "src/prob4d/provider_v2_factors.py",
    "src/prob4d/provider_v2_loading.py",
    "src/prob4d/public_api_manifest.py",
    "src/prob4d/runtime_revision.py",
    "src/prob4d/source_diagnostics.py",
    "src/prob4d/sparse_observation_factors.py",
    "src/prob4d/target_free_rehearsal.py",
)
_DYNAMIC_DIRECTORIES: Final[tuple[str, ...]] = (
    "src/prob4d_independent_verifier",
)


def stable_mypy_targets(root: Path = ROOT) -> tuple[str, ...]:
    """Return sorted, duplicate-free paths and fail if any target is missing."""

    targets = set(_EXPLICIT_TARGETS)
    for directory in _DYNAMIC_DIRECTORIES:
        target_root = root / directory
        if not target_root.is_dir():
            raise ValueError(f"stable MyPy target directory is missing: {directory}")
        targets.update(
            path.relative_to(root).as_posix()
            for path in target_root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    ordered = tuple(sorted(targets))
    missing = [path for path in ordered if not (root / path).is_file()]
    if missing:
        raise ValueError(f"stable MyPy targets are missing: {missing}")
    return ordered


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the generated stable-interface MyPy target set."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the target set without printing it",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    targets = stable_mypy_targets()
    if not arguments.check:
        print("\n".join(targets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
