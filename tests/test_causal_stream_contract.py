from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d._metric_gauge_anchor import MetricGaugeAnchor
from prob4d.causal_stream_contract import (
    bind_causal_stream_contract_v2,
)
from prob4d.observation_contract import ObservationBeliefExportV1
from prob4d.sim3 import Sim3


def _anchor() -> MetricGaugeAnchor:
    return MetricGaugeAnchor(
        window_id="window-0",
        global_from_local=Sim3.identity(),
        covariance=np.eye(7) * 1e-4,
        coordinate_frame="phystwin-world",
        source_kind="prefix_registration",
        source_artifact_sha256="1" * 64,
        metadata={"calibration_split": "train-prefix"},
    )


def _artifact(anchor: MetricGaugeAnchor) -> ObservationBeliefExportV1:
    return ObservationBeliefExportV1(
        case_id="case",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=2,
        view_names=("camera-0",),
        window_names=("window-0",),
        factor_names=(
            "joint_gauge_latent_0000",
            "joint_gauge_latent_0001",
        ),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="d" * 40,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([1]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]]),
        frame_ids=np.asarray([1]),
        entity_ids=np.asarray([0]),
        view_indices=np.asarray([0]),
        window_indices=np.asarray([0]),
        correlation_group_ids=np.asarray([0]),
        factor_group_ids=np.asarray([0]),
        prior_reliability=np.asarray([0.8]),
        association_probability=np.asarray([1.0]),
        local_covariance_m2=np.eye(3)[None] * 1e-5,
        low_rank_factor_m=np.zeros((1, 3, 2)),
        group_ids=np.asarray([0]),
        group_prior_nominal_probability=np.asarray([1.0]),
        group_composite_weight=np.asarray([1.0]),
        metadata={
            "metric_coordinates": True,
            "metric_units": "m",
            "coordinate_frame": anchor.coordinate_frame,
            "metric_gauge_anchor": {
                "artifact_id": anchor.artifact_id,
                "window_id": anchor.window_id,
                "source_kind": anchor.source_kind,
                "source_artifact_sha256": anchor.source_artifact_sha256,
            },
            "gauge_mode": "sequential",
            "joint_cross_window_gauge_covariance_represented": True,
            "gauge_posterior": {
                "model": "sequential_joint_spanning_tree_v1",
                "window_count": 1,
                "full_dimension": 7,
                "exported_factor_rank": 2,
                "retained_covariance_trace_fraction": 1.0,
                "minimum_retained_gauge_trace": 0.999,
                "cross_window_covariance_preserved": True,
                "fixed_lag_boundary_covariance_is_approximate": False,
                "parent_window_ids": [None],
                "alignments": [],
            },
        },
    )


def test_bind_causal_stream_contract_v2_enriches_anchor_and_version() -> None:
    anchor = _anchor()
    raw = _artifact(anchor)

    bound = bind_causal_stream_contract_v2(raw, metric_anchor=anchor)

    assert bound.metadata["prob4d_causal_stream_contract_version"] == 2
    assert bound.metadata["metric_gauge_anchor"] == {
        "schema_name": "prob4d.metric-gauge-anchor",
        "schema_version": 1,
        "artifact_id": anchor.artifact_id,
        "window_id": "window-0",
        "coordinate_frame": "phystwin-world",
        "metric_units": "m",
        "source_kind": "prefix_registration",
        "source_artifact_sha256": "1" * 64,
        "covariance_treatment": "fixed_external_calibration",
        "metadata": {"calibration_split": "train-prefix"},
    }
    assert bound.artifact_id != raw.artifact_id
    np.testing.assert_array_equal(bound.mean_xyz_m, raw.mean_xyz_m)


def test_bind_causal_stream_contract_rejects_approximate_fixed_lag() -> None:
    anchor = _anchor()
    raw = _artifact(anchor)
    metadata = dict(raw.metadata)
    metadata["gauge_posterior"] = dict(metadata["gauge_posterior"])
    metadata["gauge_posterior"][
        "fixed_lag_boundary_covariance_is_approximate"
    ] = True

    with pytest.raises(ValueError, match="approximate fixed-lag"):
        bind_causal_stream_contract_v2(
            replace(raw, metadata=metadata),
            metric_anchor=anchor,
        )


def test_bind_causal_stream_contract_rejects_anchor_mismatch() -> None:
    anchor = _anchor()
    raw = _artifact(anchor)
    another = replace(anchor, source_artifact_sha256="2" * 64)

    with pytest.raises(ValueError, match="identity changed|source digest changed"):
        bind_causal_stream_contract_v2(raw, metric_anchor=another)
