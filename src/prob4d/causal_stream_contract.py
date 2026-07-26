"""Versioned producer contract for strict Prob4D causal observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ._metric_gauge_anchor import (
    METRIC_GAUGE_ANCHOR_SCHEMA,
    METRIC_GAUGE_ANCHOR_VERSION,
    MetricGaugeAnchor,
)
from .observation_contract import ObservationBeliefExportV1

PROB4D_CAUSAL_STREAM_CONTRACT_VERSION = 2
PROB4D_CAUSAL_STREAM_ID = "prob4d:causal-overlap-window-points"
PROB4D_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bind_causal_stream_contract_v2(
    artifact: ObservationBeliefExportV1,
    *,
    metric_anchor: MetricGaugeAnchor,
) -> ObservationBeliefExportV1:
    """Bind the strict joint-gauge stream contract to an exported belief.

    Prob4D 0.2.0 introduced the joint cross-window gauge factor before assigning
    it a provider-specific stream-contract version. This function makes that
    interpretation explicit without changing the neutral ObservationBeliefV1
    schema.
    """

    _require(
        artifact.source_repository == PROB4D_SOURCE_REPOSITORY
        and artifact.stream_id == PROB4D_CAUSAL_STREAM_ID,
        "stream contract v2 can only bind the strict Prob4D causal stream",
    )
    _require(
        artifact.window_names
        and artifact.window_names[0] == metric_anchor.window_id,
        "metric gauge anchor must identify the first exported window",
    )
    metadata = dict(artifact.metadata)
    existing_version = metadata.get(
        "prob4d_causal_stream_contract_version"
    )
    _require(
        existing_version in {None, PROB4D_CAUSAL_STREAM_CONTRACT_VERSION},
        "observation artifact already declares another Prob4D stream contract",
    )
    _require(
        metadata.get("gauge_mode") == "sequential",
        "stream contract v2 requires the causal sequential gauge mode",
    )
    _require(
        metadata.get("joint_cross_window_gauge_covariance_represented") is True,
        "stream contract v2 requires joint cross-window gauge covariance",
    )
    posterior = metadata.get("gauge_posterior")
    _require(
        isinstance(posterior, Mapping),
        "stream contract v2 requires gauge-posterior metadata",
    )
    _require(
        posterior.get("model") == "sequential_joint_spanning_tree_v1",
        "stream contract v2 requires the sequential joint gauge tree",
    )
    _require(
        posterior.get("cross_window_covariance_preserved") is True,
        "stream contract v2 requires preserved cross-window covariance",
    )
    _require(
        posterior.get("fixed_lag_boundary_covariance_is_approximate") is False,
        "stream contract v2 cannot bind approximate fixed-lag covariance",
    )
    existing_anchor = metadata.get("metric_gauge_anchor")
    _require(
        isinstance(existing_anchor, Mapping),
        "exported belief has no metric gauge-anchor metadata",
    )
    _require(
        existing_anchor.get("artifact_id") == metric_anchor.artifact_id,
        "exported metric gauge-anchor identity changed",
    )
    _require(
        existing_anchor.get("source_artifact_sha256")
        == metric_anchor.source_artifact_sha256,
        "exported metric gauge-anchor source digest changed",
    )

    metadata["prob4d_causal_stream_contract_version"] = (
        PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
    )
    metadata["metric_gauge_anchor"] = {
        "schema_name": METRIC_GAUGE_ANCHOR_SCHEMA,
        "schema_version": METRIC_GAUGE_ANCHOR_VERSION,
        "artifact_id": metric_anchor.artifact_id,
        "window_id": metric_anchor.window_id,
        "coordinate_frame": metric_anchor.coordinate_frame,
        "metric_units": "m",
        "source_kind": metric_anchor.source_kind,
        "source_artifact_sha256": metric_anchor.source_artifact_sha256,
        "covariance_treatment": "fixed_external_calibration",
        "metadata": dict(metric_anchor.metadata),
    }
    metadata["prob4d_causal_stream_contract"] = {
        "version": PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
        "gauge_covariance_semantics": (
            "one shared low-rank root of the joint cross-window Sim(3) "
            "covariance induced by the fixed metric anchor and selected "
            "causal gauge tree"
        ),
        "factor_group_semantics": "one shared latent vector across all windows",
        "causal_frame_stop_convention": "exclusive",
    }
    return replace(artifact, metadata=metadata)


__all__ = [
    "PROB4D_CAUSAL_STREAM_CONTRACT_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_SOURCE_REPOSITORY",
    "bind_causal_stream_contract_v2",
]
