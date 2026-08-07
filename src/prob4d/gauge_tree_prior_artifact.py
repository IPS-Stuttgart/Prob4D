"""Portable content-addressed artifacts for sparse gauge-tree priors.

This additive sidecar lets consumers load the exact causal tree factors without
opening the dense ``7K x 7K`` covariance retained by existing schema-v4 bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._gauge_tree_artifact_common import (
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
    GAUGE_TREE_PRIOR_STORAGE_SEMANTICS,
    GaugeTreePriorArrayMemberV1,
    GaugeTreePriorArtifactV1,
)
from ._gauge_tree_artifact_io import (
    LoadedGaugeTreePriorArtifactV1,
    load_gauge_tree_prior_artifact,
    write_gauge_tree_prior_artifact,
)


def artifact_summary(
    loaded: LoadedGaugeTreePriorArtifactV1,
) -> dict[str, object]:
    """Return a compact JSON-compatible validation summary."""

    prior = loaded.prior
    manifest = loaded.manifest
    payload_bytes = sum(
        member.byte_count
        for member in (
            manifest.parent_indices,
            manifest.transition_matrices,
            manifest.innovation_scale_tril,
        )
    )
    return {
        "valid": True,
        "artifact_id": manifest.artifact_id,
        "prior_id": prior.prior_id,
        "gauge_count": prior.gauge_count,
        "payload_bytes": payload_bytes,
        "factor_storage_nbytes": prior.factor_storage_nbytes,
        "dense_covariance_nbytes": prior.dense_covariance_nbytes,
        "source_joint_covariance_sha256": prior.source_joint_covariance_sha256,
        "storage_semantics": manifest.storage_semantics,
    }


def _print_summary(summary: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(
        "Valid sparse gauge-tree prior artifact: "
        f"{summary['gauge_count']} gauges, "
        f"artifact {summary['artifact_id']}."
    )


def _verify_cli(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d gauge prior verify",
        description="Verify a portable sparse gauge-tree prior without densifying it.",
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parsed = parser.parse_args(list(arguments))
    try:
        summary = artifact_summary(load_gauge_tree_prior_artifact(parsed.manifest))
    except (OSError, ValueError) as error:
        print(f"invalid gauge-tree prior artifact: {error}", file=sys.stderr)
        return 2
    _print_summary(summary, json_output=parsed.json_output)
    return 0


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(f"dense output path crosses symbolic link {candidate}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_dense_no_replace(output: Path, dense: np.ndarray) -> str:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace dense output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, dense, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256_file(temporary)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace dense output {output}"
            ) from error
        published = True
        return digest
    finally:
        temporary.unlink(missing_ok=True)
        if not published and output.is_symlink():
            raise ValueError("dense output destination became a symbolic link")


def _materialize_cli(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="prob4d gauge prior materialize",
        description=(
            "Explicitly materialize a verified sparse prior as a guarded dense NPY file."
        ),
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-gauges", type=int, default=128)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parsed = parser.parse_args(list(arguments))
    try:
        loaded = load_gauge_tree_prior_artifact(parsed.manifest)
        dense = loaded.prior.materialize_dense_covariance(
            maximum_gauges=parsed.maximum_gauges
        )
        digest = _write_dense_no_replace(parsed.output, dense)
    except (OSError, ValueError) as error:
        print(f"unable to materialize gauge-tree prior: {error}", file=sys.stderr)
        return 2
    summary = {
        **artifact_summary(loaded),
        "dense_output": str(parsed.output),
        "dense_output_sha256": digest,
        "maximum_gauges": parsed.maximum_gauges,
    }
    if parsed.json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            f"Materialized {summary['gauge_count']} gauges to {parsed.output} "
            f"with SHA-256 {digest}."
        )
    return 0


def _render_help() -> str:
    return "\n".join(
        [
            "usage: prob4d gauge prior <verify|materialize> [arguments]",
            "",
            "commands:",
            "  verify       verify and summarize a sparse prior artifact",
            "  materialize  explicitly create a guarded dense covariance NPY",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Verify or explicitly densify a portable sparse-prior artifact."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_render_help(), end="")
        return 0
    if arguments[0] == "verify":
        return _verify_cli(arguments[1:])
    if arguments[0] == "materialize":
        return _materialize_cli(arguments[1:])
    # Preserve the original module-level validation form:
    # ``python -m prob4d.gauge_tree_prior_artifact MANIFEST [--json]``.
    return _verify_cli(arguments)


__all__ = [
    "GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA",
    "GAUGE_TREE_PRIOR_ARTIFACT_VERSION",
    "GAUGE_TREE_PRIOR_STORAGE_SEMANTICS",
    "GaugeTreePriorArrayMemberV1",
    "GaugeTreePriorArtifactV1",
    "LoadedGaugeTreePriorArtifactV1",
    "artifact_summary",
    "load_gauge_tree_prior_artifact",
    "main",
    "write_gauge_tree_prior_artifact",
]


if __name__ == "__main__":
    raise SystemExit(main())
