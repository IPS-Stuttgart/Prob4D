from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from prob4d.observation_contract import (
    ObservationBeliefExportV1,
    save_observation_belief_export,
)
from prob4d.provider_v2 import build_provider_attestation, prob4d_provider_manifest
from prob4d.provider_v2_loading import (
    ValidatedClaimBearingObservation,
    load_claim_bearing_observation_belief,
    validate_claim_bearing_observation_belief,
)

REVISION = "a" * 40
GAUGE_CALIBRATION_ID = "1" * 64
POINT_CALIBRATION_ID = "2" * 64


def _attestation() -> dict[str, object]:
    return build_provider_attestation(
        provider_manifest=prob4d_provider_manifest(provider_revision=REVISION),
        provider_revision=REVISION,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": GAUGE_CALIBRATION_ID,
            "point_artifact_id": POINT_CALIBRATION_ID,
        },
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision={
            "expected_revision": REVISION,
            "observed_revision": REVISION,
            "source": "source_checkout",
            "clean_checkout": True,
            "matched": True,
            "independently_verified": True,
        },
    )


def _artifact(
    *,
    caller_metadata: dict[str, object] | None = None,
) -> ObservationBeliefExportV1:
    metadata: dict[str, object] = {
        "coordinate_frame": "phystwin-world",
        "gauge_mode": "sequential",
        "joint_cross_window_gauge_covariance_represented": True,
        "metric_anchor_covariance_in_joint_factor": True,
        "prob4d_causal_stream_contract_version": 2,
        "prob4d_causal_stream_contract": {
            "version": 2,
            "causal_frame_stop_convention": "exclusive",
        },
        "causal_source_lineage": {
            "causal_frame_stop_exclusive": 2,
            "future_prediction_payloads_opened": 0,
            "selected_windows": [
                {
                    "window_id": "window0",
                    "source_frame_max": 0,
                    "payload_sha256": "b" * 64,
                }
            ],
        },
        "gauge_posterior": {
            "model": "sequential_joint_spanning_tree_v1",
            "cross_window_covariance_preserved": True,
            "fixed_lag_boundary_covariance_is_approximate": False,
            "exported_factor_rank": 1,
        },
        "covariance_calibration": {
            "status": "calibrated",
            "gauge_artifact_id": GAUGE_CALIBRATION_ID,
            "point_artifact_id": POINT_CALIBRATION_ID,
            "alignment_count": 1,
            "gauge_calibrated_alignment_count": 1,
            "covariance_fallback_counts": {},
            "uncalibrated_exploratory_covariance_allowed": False,
            "pointwise_covariance_fallback_allowed": False,
        },
        "prob4d_provider_attestation": _attestation(),
        "nested": {"values": [1, {"label": "stable"}]},
    }
    if caller_metadata is not None:
        metadata.update(caller_metadata)
    return ObservationBeliefExportV1(
        case_id="case-a",
        stream_id="prob4d:causal-overlap-window-points",
        causal_frame_stop=2,
        view_names=("camera0",),
        window_names=("window0",),
        factor_names=("joint_gauge_latent_0000",),
        source_repository="FlorianPfaff/Prob4D",
        source_revision=REVISION,
        source_artifact_sha256="c" * 64,
        declared_frame_ids=np.asarray([0], dtype=np.int64),
        mean_xyz_m=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        frame_ids=np.asarray([0], dtype=np.int64),
        entity_ids=np.asarray([0], dtype=np.int64),
        view_indices=np.asarray([0], dtype=np.int64),
        window_indices=np.asarray([0], dtype=np.int64),
        correlation_group_ids=np.asarray([0], dtype=np.int64),
        factor_group_ids=np.asarray([0], dtype=np.int64),
        prior_reliability=np.asarray([1.0], dtype=np.float64),
        association_probability=np.asarray([1.0], dtype=np.float64),
        local_covariance_m2=np.asarray([np.eye(3)], dtype=np.float64),
        low_rank_factor_m=np.asarray([[[0.1], [0.0], [0.0]]], dtype=np.float64),
        group_ids=np.asarray([0], dtype=np.int64),
        group_prior_nominal_probability=np.asarray([1.0], dtype=np.float64),
        group_composite_weight=np.asarray([1.0], dtype=np.float64),
        metadata=metadata,
    )


def test_observation_metadata_is_recursively_immutable_and_hash_stable() -> None:
    caller_metadata = {"caller_owned": {"items": [{"value": 1}]}}
    artifact = _artifact(caller_metadata=caller_metadata)
    artifact_id = artifact.artifact_id

    caller_metadata["caller_owned"]["items"][0]["value"] = 99  # type: ignore[index]
    assert artifact.metadata["caller_owned"]["items"][0]["value"] == 1  # type: ignore[index]
    assert isinstance(artifact.metadata, Mapping)
    assert isinstance(artifact.metadata["nested"]["values"], Sequence)  # type: ignore[index]

    with pytest.raises(TypeError, match="metadata is immutable"):
        artifact.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError, match="metadata is immutable"):
        artifact.metadata["nested"]["values"].append(2)  # type: ignore[index,union-attr]
    with pytest.raises(TypeError, match="metadata is immutable"):
        artifact.metadata["nested"]["values"][1]["label"] = "changed"  # type: ignore[index]

    shallow = copy(artifact.metadata)
    deep = deepcopy(artifact.metadata)
    assert type(shallow) is dict
    assert type(deep) is dict
    assert type(deep["nested"]["values"]) is list
    deep["nested"]["values"].append("mutable")
    assert artifact.artifact_id == artifact_id


def test_strict_claim_bearing_loader_returns_validated_identity(tmp_path: Path) -> None:
    artifact = _artifact()
    path = tmp_path / "observation.npz"
    save_observation_belief_export(path, artifact)

    validated = load_claim_bearing_observation_belief(path)

    assert isinstance(validated, ValidatedClaimBearingObservation)
    assert validated.artifact_id == artifact.artifact_id
    assert validated.provider_manifest_id == _attestation()["provider_manifest_id"]
    assert validated.gauge_calibration_id == GAUGE_CALIBRATION_ID
    assert validated.point_calibration_id == POINT_CALIBRATION_ID
    assert validated.runtime_revision == REVISION


def test_strict_validation_rejects_calibration_identity_drift() -> None:
    artifact = _artifact()
    metadata = deepcopy(artifact.metadata)
    metadata["covariance_calibration"]["gauge_artifact_id"] = "3" * 64
    changed = replace(artifact, metadata=metadata)

    with pytest.raises(ValueError, match="differs from provider attestation"):
        validate_claim_bearing_observation_belief(changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "gauge_calibrated_alignment_count",
            0,
            "uncalibrated gauge alignments",
        ),
        (
            "covariance_fallback_counts",
            {"pointwise": 1},
            "reports covariance fallback use",
        ),
        (
            "uncalibrated_exploratory_covariance_allowed",
            True,
            "cannot allow uncalibrated covariance",
        ),
        (
            "pointwise_covariance_fallback_allowed",
            True,
            "cannot allow pointwise covariance fallback",
        ),
    ],
)
def test_strict_validation_rejects_incomplete_or_fallback_calibration(
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = _artifact()
    metadata = deepcopy(artifact.metadata)
    metadata["covariance_calibration"][field] = value
    changed = replace(artifact, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        validate_claim_bearing_observation_belief(changed)


@pytest.mark.parametrize(
    "field",
    [
        "alignment_count",
        "gauge_calibrated_alignment_count",
        "covariance_fallback_counts",
        "uncalibrated_exploratory_covariance_allowed",
        "pointwise_covariance_fallback_allowed",
    ],
)
def test_strict_validation_requires_complete_calibration_metadata(field: str) -> None:
    artifact = _artifact()
    metadata = deepcopy(artifact.metadata)
    metadata["covariance_calibration"].pop(field)
    changed = replace(artifact, metadata=metadata)

    with pytest.raises(ValueError):
        validate_claim_bearing_observation_belief(changed)


def test_strict_validation_rejects_exploratory_or_noncausal_artifacts() -> None:
    artifact = _artifact()
    metadata = deepcopy(artifact.metadata)
    metadata.pop("prob4d_provider_attestation")
    changed = replace(artifact, metadata=metadata)

    with pytest.raises(ValueError, match="provider attestation"):
        validate_claim_bearing_observation_belief(changed)

    metadata = deepcopy(artifact.metadata)
    metadata["causal_source_lineage"]["selected_windows"][0]["source_frame_max"] = 2
    changed = replace(artifact, metadata=metadata)
    with pytest.raises(ValueError, match="crosses the causal frame boundary"):
        validate_claim_bearing_observation_belief(changed)
