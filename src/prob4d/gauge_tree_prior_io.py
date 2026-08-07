"""Public compatibility facade for portable sparse gauge-tree prior artifacts."""

from __future__ import annotations

from pathlib import Path

from ._gauge_tree_artifact_common import (
    GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA,
    GAUGE_TREE_PRIOR_ARTIFACT_VERSION,
)
from ._gauge_tree_artifact_io import LoadedGaugeTreePriorArtifactV1
from .gauge_tree_prior import GaugeTreeSquareRootPriorV1
from .gauge_tree_prior_artifact import (
    gauge_tree_prior_artifact_id,
    load_gauge_tree_prior_artifact,
    write_gauge_tree_prior_artifact,
)

GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY = (
    "portable sparse gauge-tree storage and exact replay identity only; "
    "not provider competence, calibration, physical-query benefit, "
    "Causal4D benefit, deployment safety, or state of the art"
)


def load_gauge_tree_prior(
    manifest_path: str | Path,
) -> GaugeTreeSquareRootPriorV1:
    """Load and verify one portable artifact, returning its immutable prior."""

    return load_gauge_tree_prior_artifact(manifest_path).prior


def write_gauge_tree_prior(
    prior: GaugeTreeSquareRootPriorV1,
    manifest_path: str | Path,
) -> LoadedGaugeTreePriorArtifactV1:
    """Publish and verify one portable three-member sparse-prior artifact."""

    return write_gauge_tree_prior_artifact(prior, manifest_path)


__all__ = [
    "GAUGE_TREE_PRIOR_ARTIFACT_CLAIM_BOUNDARY",
    "GAUGE_TREE_PRIOR_ARTIFACT_SCHEMA",
    "GAUGE_TREE_PRIOR_ARTIFACT_VERSION",
    "gauge_tree_prior_artifact_id",
    "load_gauge_tree_prior",
    "write_gauge_tree_prior",
]
