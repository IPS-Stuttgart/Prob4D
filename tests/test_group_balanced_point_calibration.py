import numpy as np
import pytest

from prob4d.calibration import (
    PointUncertaintyCalibrationV1,
    fit_group_balanced_point_uncertainty_calibration,
    group_balanced_point_calibration_metadata,
    load_point_uncertainty_calibration,
    save_point_uncertainty_calibration,
)
from prob4d.uncertainty import DepthDisagreementModel, StructuredCovariance


PROVENANCE = {
    "calibration_case_ids": ("large", "small"),
    "source_repository": "FlorianPfaff/Prob4D",
    "source_revision": "a" * 40,
    "motioncrafter_revision": "b" * 40,
    "model_identifier": "motioncrafter@sha256:cafebabe",
    "covariance_method": "depth_disagreement_anisotropic_v1",
    "image_resolution": (320, 640),
    "window_size": 25,
    "window_overlap": 8,
    "covariance_cluster_size": 32,
    "input_artifact_sha256": ("c" * 64,),
}


def calibration_inputs() -> tuple[np.ndarray, StructuredCovariance, np.ndarray]:
    count = 404
    rays = np.zeros((count, 3), dtype=np.float64)
    rays[:, 2] = 1.0
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.ones(count),
        lateral_variance=np.ones(count),
    )
    errors = np.zeros((count, 3), dtype=np.float64)
    errors[:400, 2] = 3.0
    errors[400:, 2] = 1.0
    errors[:400, 0] = np.sqrt(18.0)
    errors[400:, 0] = np.sqrt(2.0)
    groups = np.asarray(["large"] * 400 + ["small"] * 4)
    return errors, covariance, groups


def test_group_balanced_fit_is_bound_into_content_addressed_artifact() -> None:
    errors, covariance, groups = calibration_inputs()

    artifact, report = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="sequence",
        trim_quantile=1.0,
        metadata={"split": "source-family-held-out"},
        **PROVENANCE,
    )

    assert report.group_ids == ("large", "small")
    assert report.group_counts == (400, 4)
    assert report.parallel_scale_update == pytest.approx(5.0)
    assert report.lateral_scale_update == pytest.approx(5.0)
    assert artifact.parallel_scale == pytest.approx(5.0)
    assert artifact.lateral_scale == pytest.approx(5.0)
    record = group_balanced_point_calibration_metadata(artifact)
    assert record is not None
    assert record["group_definition"] == "sequence"
    assert record["report"]["aggregation"] == (
        "equal-group-mean-of-within-group-trimmed-ratios-v1"
    )
    assert artifact.metadata["split"] == "source-family-held-out"
    assert len(artifact.artifact_id) == 64


def test_group_balanced_artifact_is_invariant_to_row_order() -> None:
    errors, covariance, groups = calibration_inputs()
    first, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="sequence",
        trim_quantile=1.0,
        **PROVENANCE,
    )
    generator = np.random.default_rng(9)
    permutation = generator.permutation(len(errors))
    permuted_covariance = StructuredCovariance(
        ray_directions=covariance.ray_directions[permutation],
        parallel_variance=covariance.parallel_variance[permutation],
        lateral_variance=covariance.lateral_variance[permutation],
    )
    second, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors[permutation],
        permuted_covariance,
        groups[permutation],
        group_definition="sequence",
        trim_quantile=1.0,
        **PROVENANCE,
    )

    assert first.artifact_id == second.artifact_id
    assert first.to_dict() == second.to_dict()


def test_group_definition_and_grouping_change_artifact_identity() -> None:
    errors, covariance, groups = calibration_inputs()
    sequence, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="sequence",
        trim_quantile=1.0,
        **PROVENANCE,
    )
    session, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="session",
        trim_quantile=1.0,
        **PROVENANCE,
    )
    regrouped = np.asarray(["combined"] * len(groups))
    combined, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        regrouped,
        group_definition="sequence",
        trim_quantile=1.0,
        **PROVENANCE,
    )

    assert sequence.artifact_id != session.artifact_id
    assert sequence.artifact_id != combined.artifact_id


def test_group_balanced_artifact_round_trips(tmp_path) -> None:
    errors, covariance, groups = calibration_inputs()
    artifact, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="sequence",
        trim_quantile=1.0,
        **PROVENANCE,
    )
    path = tmp_path / "group-balanced-point.json"
    save_point_uncertainty_calibration(artifact, path)

    loaded = load_point_uncertainty_calibration(path)

    assert loaded == artifact
    assert loaded.artifact_id == artifact.artifact_id
    assert group_balanced_point_calibration_metadata(loaded) is not None


def test_reserved_group_metadata_key_is_rejected() -> None:
    errors, covariance, groups = calibration_inputs()

    with pytest.raises(ValueError, match="reserved"):
        fit_group_balanced_point_uncertainty_calibration(
            DepthDisagreementModel(),
            errors,
            covariance,
            groups,
            group_definition="sequence",
            trim_quantile=1.0,
            metadata={"group_balanced_point_uncertainty_calibration": {}},
            **PROVENANCE,
        )


def test_pooled_point_artifact_is_not_mislabelled_group_balanced() -> None:
    errors, covariance, _ = calibration_inputs()
    pooled = PointUncertaintyCalibrationV1.fit(
        DepthDisagreementModel(),
        errors,
        covariance,
        trim_quantile=1.0,
        **PROVENANCE,
    )

    assert group_balanced_point_calibration_metadata(pooled) is None
