from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d._metric_gauge_anchor import (
    FIXED_EXTERNAL_CALIBRATION,
    PROPAGATED_JOINT_GAUGE_COVARIANCE,
    MetricGaugeAnchor,
    load_metric_gauge_anchor,
    save_metric_gauge_anchor,
)
from prob4d.observation_contract import ObservationBeliefExportV1
from prob4d.observation_validation import load_observation_belief_export
from prob4d.portable_observation import (
    JOINT_GAUGE_COVARIANCE_LAYOUT,
    JOINT_GAUGE_FACTOR_GROUP_SEMANTICS,
    PROB4D_OBSERVATION_CONTRACT_VERSION,
    enrich_prob4d_observation_belief,
    save_observation_belief_export,
)
from prob4d.sim3 import Sim3


def _anchor(*, uncertain: bool = False, portable: bool = True) -> MetricGaugeAnchor:
    covariance = np.zeros((7, 7))
    if uncertain:
        covariance = np.eye(7) * 1e-6
    return MetricGaugeAnchor(
        window_id="window-0",
        global_from_local=Sim3.identity(),
        covariance=covariance,
        coordinate_frame="phystwin-world",
        source_kind="prefix_registration",
        source_artifact_sha256="1" * 64,
        calibration_artifact_sha256="b" * 64 if portable else None,
        case_id="case",
    )


def _artifact() -> ObservationBeliefExportV1:
    return ObservationBeliefExportV1(
        case_id="case",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=3,
        view_names=("camera-0",),
        window_names=("window-0", "window-1"),
        factor_names=tuple(
            f"joint_gauge_latent_{index:04d}" for index in range(3)
        ),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="d" * 40,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([0, 1]),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        frame_ids=np.asarray([0, 1]),
        entity_ids=np.asarray([0, 1]),
        view_indices=np.zeros(2, dtype=np.int64),
        window_indices=np.asarray([0, 1]),
        correlation_group_ids=np.asarray([0, 1]),
        factor_group_ids=np.zeros(2, dtype=np.int64),
        prior_reliability=np.asarray([0.9, 0.8]),
        association_probability=np.ones(2),
        local_covariance_m2=np.repeat(np.eye(3)[None], 2, axis=0) * 1e-5,
        low_rank_factor_m=np.zeros((2, 3, 3)),
        group_ids=np.asarray([0, 1]),
        group_prior_nominal_probability=np.ones(2),
        group_composite_weight=np.asarray([0.5, 0.5]),
        metadata={
            "metric_coordinates": True,
            "metric_units": "m",
            "coordinate_frame": "phystwin-world",
            "causal_source_lineage": {
                "selected_windows": [
                    {
                        "window_id": "window-0",
                        "payload_sha256": "1" * 64,
                    },
                    {
                        "window_id": "window-1",
                        "payload_sha256": "2" * 64,
                    },
                ]
            },
            "gauge_posterior": {
                "window_count": 2,
                "exported_factor_rank": 3,
                "cross_window_covariance_preserved": True,
            },
        },
    )


def test_contract_v2_embeds_complete_joint_layout_and_anchor() -> None:
    enriched = enrich_prob4d_observation_belief(
        _artifact(),
        metric_anchor=_anchor(),
    )

    assert (
        enriched.metadata["prob4d_observation_contract_version"]
        == PROB4D_OBSERVATION_CONTRACT_VERSION
    )
    assert enriched.metadata["covariance_layout"] == JOINT_GAUGE_COVARIANCE_LAYOUT
    assert (
        enriched.metadata["factor_group_semantics"]
        == JOINT_GAUGE_FACTOR_GROUP_SEMANTICS
    )
    anchor = enriched.metadata["metric_gauge_anchor"]
    assert anchor["world_frame_id"] == "phystwin-world"
    assert anchor["calibration_artifact_sha256"] == "b" * 64
    assert anchor["covariance_treatment"] == FIXED_EXTERNAL_CALIBRATION


def test_uncertain_metric_anchor_is_declared_as_propagated() -> None:
    enriched = enrich_prob4d_observation_belief(
        _artifact(),
        metric_anchor=_anchor(uncertain=True),
    )
    assert (
        enriched.metadata["metric_gauge_anchor"]["covariance_treatment"]
        == PROPAGATED_JOINT_GAUGE_COVARIANCE
    )
    assert enriched.metadata["metric_anchor_covariance_included_in_joint_factor"]


def test_portable_export_rejects_anchor_without_calibration_digest() -> None:
    with pytest.raises(ValueError, match="calibration_artifact_sha256"):
        enrich_prob4d_observation_belief(
            _artifact(),
            metric_anchor=_anchor(portable=False),
        )


def test_atomic_observation_round_trip(tmp_path: Path) -> None:
    artifact = enrich_prob4d_observation_belief(
        _artifact(),
        metric_anchor=_anchor(),
    )
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, artifact)
    restored = load_observation_belief_export(path)

    assert restored.artifact_id == artifact.artifact_id
    assert restored.metadata["covariance_layout"] == JOINT_GAUGE_COVARIANCE_LAYOUT
    assert not list(tmp_path.glob(".observation.npz.*"))


def test_metric_anchor_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "anchor.json"
    save_metric_gauge_anchor(path, _anchor())
    restored = load_metric_gauge_anchor(path)
    assert restored.artifact_id == _anchor().artifact_id

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["calibration_artifact_sha256"] = "e" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id does not match"):
        load_metric_gauge_anchor(path)
