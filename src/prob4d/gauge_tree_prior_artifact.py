"""Portable content-addressed artifacts for sparse gauge-tree priors.

This additive sidecar lets consumers load the exact causal tree factors without
opening the dense ``7K x 7K`` covariance retained by existing schema-v4 bundles.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

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


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and summarize one portable gauge-tree prior artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true", dest="json_output")
    arguments = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        summary = artifact_summary(
            load_gauge_tree_prior_artifact(arguments.manifest)
        )
    except (OSError, ValueError) as error:
        print(f"invalid gauge-tree prior artifact: {error}", file=sys.stderr)
        return 2
    if arguments.json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Valid sparse gauge-tree prior artifact: "
            f"{summary['gauge_count']} gauges, "
            f"artifact {summary['artifact_id']}."
        )
    return 0


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
