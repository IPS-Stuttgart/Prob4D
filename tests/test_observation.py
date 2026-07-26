import json
from types import SimpleNamespace

import numpy as np
import pytest

from prob4d.observation import ObservationArtifact, SourceWindowProvenance
from prob4d.observation_io import load_observation_artifact, save_observation_artifact


def provenance() -> dict[str, str]:
    return {
        "producer": "Prob4D",
        "producer_revision": "a" * 40,
        "source_model": "MotionCrafter",
        "source_model_revision": "b" * 40,
        "source_manifest_sha256": "c" * 64,
        "method": "prob4d_uniform",
    }


def sequence() -> SimpleNamespace:
    points = np.zeros((2, 1, 2, 3))
    mask = np.ones((2, 1, 2), dtype=bool)
    covariance = np.broadcast_to(np.eye(3), points.shape + (3,)).copy()
    return SimpleNamespace(
        frame_indices=np.array([0, 1]),
        point_map=points,
        valid_mask=mask,
        point_covariance=covariance,
        contributors=np.ones(mask.shape, dtype=np.uint16),
        scene_flow=np.zeros_like(points),
        deform_mask=mask.copy(),
        flow_covariance=covariance.copy(),
    )


def sources() -> tuple[SourceWindowProvenance, ...]:
    return (SourceWindowProvenance("w0", (0, 1), "shared"),)


def test_artifact_round_trip(tmp_path) -> None:
    artifact = ObservationArtifact.from_fused_sequence(
        sequence(),
        sources(),
        coordinate_status="gauge_relative",
        gauge_status="unresolved",
        covariance_units="gauge_unit^2",
        gauge_reference="w0",
        provenance=provenance(),
        causal_max_frame=1,
        global_estimator_source_frame_limit=1,
    )

    manifest = save_observation_artifact(tmp_path / "observation.json", artifact)
    restored = load_observation_artifact(manifest)

    np.testing.assert_array_equal(restored.frame_indices, artifact.frame_indices)
    np.testing.assert_allclose(restored.point_mean, artifact.point_mean)
    np.testing.assert_allclose(restored.point_covariance, artifact.point_covariance)
    assert restored.frame_contributor_window_ids == (("w0",), ("w0",))
    assert restored.summary()["maximum_source_frame_used"] == 1


def test_causal_limit_rejects_future_source_dependency() -> None:
    source = (SourceWindowProvenance("w0", (0, 1, 2), "shared"),)
    with pytest.raises(ValueError, match="after causal_max_frame"):
        ObservationArtifact.from_fused_sequence(
            sequence(),
            source,
            coordinate_status="gauge_relative",
            gauge_status="unresolved",
            covariance_units="gauge_unit^2",
            gauge_reference="w0",
            provenance=provenance(),
            causal_max_frame=1,
            global_estimator_source_frame_limit=2,
        )


def test_metric_coordinates_require_anchored_gauge() -> None:
    with pytest.raises(ValueError, match="anchored gauge"):
        ObservationArtifact.from_fused_sequence(
            sequence(),
            sources(),
            coordinate_status="metric",
            gauge_status="unresolved",
            covariance_units="m^2",
            gauge_reference="world",
            provenance=provenance(),
        )


def test_indefinite_covariance_is_rejected() -> None:
    value = sequence()
    value.point_covariance[0, 0, 0] = np.diag([1.0, 1.0, -0.1])
    with pytest.raises(ValueError, match="positive semidefinite"):
        ObservationArtifact.from_fused_sequence(
            value,
            sources(),
            coordinate_status="gauge_relative",
            gauge_status="unresolved",
            covariance_units="gauge_unit^2",
            gauge_reference="w0",
            provenance=provenance(),
        )


def test_array_hash_detects_mutation(tmp_path) -> None:
    artifact = ObservationArtifact.from_fused_sequence(
        sequence(),
        sources(),
        coordinate_status="gauge_relative",
        gauge_status="unresolved",
        covariance_units="gauge_unit^2",
        gauge_reference="w0",
        provenance=provenance(),
    )
    manifest = save_observation_artifact(tmp_path / "observation.json", artifact)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    array_path = manifest.parent / payload["array_file"]
    with array_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256"):
        load_observation_artifact(manifest)
