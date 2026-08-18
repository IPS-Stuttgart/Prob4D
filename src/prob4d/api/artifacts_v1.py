"""Preview version-1 portable-artifact façade backed by :mod:`prob4d.api.v2`."""

from __future__ import annotations

from typing import Final

from .v2 import (
    GaugeTreePriorArtifactV1,
    LoadedGaugeTreePriorArtifactV1,
    LoadedTreeSparseObservationArtifactV1,
    ObservationBeliefExportV1,
    ObservationFactor,
    ObservationFactorBundle,
    ObservationFactorStreamUpdateV1,
    ObservationFactorStreamV1,
    TreeSparseObservationArtifactV1,
    append_observation_factor_bundle,
    gauge_tree_prior_artifact_id,
    load_gauge_tree_prior_artifact,
    load_observation_belief_export,
    load_observation_factor_bundle,
    load_observation_factor_stream,
    load_tree_sparse_observation_artifact,
    save_observation_belief_export,
    write_gauge_tree_prior_artifact,
    write_observation_factor_bundle,
    write_observation_factor_stream,
    write_tree_sparse_observation_artifact,
)

FACADE_VERSION: Final = 1
LIFECYCLE: Final = "preview"

__all__ = [
    "FACADE_VERSION",
    "GaugeTreePriorArtifactV1",
    "LIFECYCLE",
    "LoadedGaugeTreePriorArtifactV1",
    "LoadedTreeSparseObservationArtifactV1",
    "ObservationBeliefExportV1",
    "ObservationFactor",
    "ObservationFactorBundle",
    "ObservationFactorStreamUpdateV1",
    "ObservationFactorStreamV1",
    "TreeSparseObservationArtifactV1",
    "append_observation_factor_bundle",
    "gauge_tree_prior_artifact_id",
    "load_gauge_tree_prior_artifact",
    "load_observation_belief_export",
    "load_observation_factor_bundle",
    "load_observation_factor_stream",
    "load_tree_sparse_observation_artifact",
    "save_observation_belief_export",
    "write_gauge_tree_prior_artifact",
    "write_observation_factor_bundle",
    "write_observation_factor_stream",
    "write_tree_sparse_observation_artifact",
]
