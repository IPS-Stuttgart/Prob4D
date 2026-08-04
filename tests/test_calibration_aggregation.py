from dataclasses import replace

import numpy as np
import pytest

from prob4d.calibration import (
    GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
    LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1,
    UPPER_WINSORIZED_MEAN_V1,
    GaugeCovarianceCalibrationV1,
    PointUncertaintyCalibrationV1,
    fit_group_balanced_point_uncertainty_calibration,
    group_balanced_point_calibration_metadata,
    upper_winsorized_mean,
)
from prob4d.gauge import GaugeCovarianceCalibration
from prob4d.uncertainty import DepthDisagreementModel, StructuredCovariance

PROVENANCE = {
    "calibration_case_ids": ("case-a", "case-b"),
    "source_repository": "FlorianPfaff/Prob4D",
    "source_revision": "a" * 40,
    "motioncrafter_revision": "b" * 40,
    "model_identifier": "motioncrafter@sha256:cafebabe",
    "covariance_method": "depth_disagreement_anisotropic_v1",
    "image_resolution": (2, 2),
    "window_size": 3,
    "window_overlap": 1,
    "covariance_cluster_size": 2,
    "input_artifact_sha256": ("c" * 64,),
}


def _group_calibration_inputs() -> tuple[np.ndarray, StructuredCovariance, np.ndarray]:
    rays = np.zeros((6, 3), dtype=np.float64)
    rays[:, 2] = 1.0
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.ones(6),
        lateral_variance=np.ones(6),
    )
    errors = np.zeros((6, 3), dtype=np.float64)
    errors[:, 2] = np.asarray([1.0, 2.0, 10.0, 1.5, 2.5, 20.0])
    groups = np.asarray(["case-a"] * 3 + ["case-b"] * 3)
    return errors, covariance, groups


def test_upper_winsorized_mean_clips_instead_of_dropping_tail_rows() -> None:
    values = np.asarray([1.0, 2.0, 100.0])
    quantile = 2.0 / 3.0
    upper = float(np.quantile(values, quantile))

    observed = upper_winsorized_mean(values, quantile=quantile, minimum=0.0)

    assert observed == pytest.approx(float(np.mean(np.minimum(values, upper))))
    assert observed != pytest.approx(float(np.mean(values[values <= upper])))


def test_upper_winsorized_mean_validates_and_can_canonicalize() -> None:
    values = np.asarray([1e16, 1.0, -1e16, 3.0])
    first = upper_winsorized_mean(
        values,
        quantile=1.0,
        minimum=0.0,
        canonicalize=True,
    )
    second = upper_winsorized_mean(
        values[::-1],
        quantile=1.0,
        minimum=0.0,
        canonicalize=True,
    )

    assert first == second
    with pytest.raises(ValueError, match="nonempty"):
        upper_winsorized_mean(np.asarray([]), quantile=0.99)
    with pytest.raises(ValueError, match="finite"):
        upper_winsorized_mean(np.asarray([1.0, np.inf]), quantile=0.99)
    with pytest.raises(ValueError, match="quantile"):
        upper_winsorized_mean(np.asarray([1.0]), quantile=True)


def test_runtime_calibrators_declare_upper_winsorization() -> None:
    model = DepthDisagreementModel()
    gauge = GaugeCovarianceCalibration(
        scale=1.0,
        rotation=1.0,
        translation=1.0,
        trim_quantile=0.9,
        count=4,
    )

    assert model.calibration_aggregation_semantics == UPPER_WINSORIZED_MEAN_V1
    assert gauge.aggregation_semantics == UPPER_WINSORIZED_MEAN_V1
    assert gauge.winsor_quantile == 0.9


def test_new_group_metadata_uses_explicit_winsorized_semantics() -> None:
    errors, covariance, groups = _group_calibration_inputs()
    artifact, report = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="object",
        trim_quantile=0.8,
        **PROVENANCE,
    )

    record = group_balanced_point_calibration_metadata(artifact)
    assert record is not None
    assert report.aggregation_semantics == GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2
    assert report.winsor_quantile == 0.8
    assert record["report"]["aggregation"] == GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2
    assert record["report"]["winsor_quantile"] == 0.8
    assert record["report"]["trim_quantile"] == 0.8


def test_legacy_misnamed_group_metadata_remains_admissible() -> None:
    errors, covariance, groups = _group_calibration_inputs()
    artifact, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="object",
        trim_quantile=0.8,
        **PROVENANCE,
    )
    metadata = artifact.to_dict()["provenance"]["metadata"]
    report = metadata["group_balanced_point_uncertainty_calibration"]["report"]
    report["aggregation"] = LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1
    report.pop("winsor_quantile")
    legacy = replace(artifact, metadata=metadata)

    record = group_balanced_point_calibration_metadata(legacy)

    assert record is not None
    assert record["report"]["aggregation"] == LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1


def test_new_group_metadata_rejects_disagreeing_quantile_aliases() -> None:
    errors, covariance, groups = _group_calibration_inputs()
    artifact, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="object",
        trim_quantile=0.8,
        **PROVENANCE,
    )
    metadata = artifact.to_dict()["provenance"]["metadata"]
    metadata["group_balanced_point_uncertainty_calibration"]["report"][
        "winsor_quantile"
    ] = 0.7
    malformed = replace(artifact, metadata=metadata)

    with pytest.raises(ValueError, match="quantiles differ"):
        group_balanced_point_calibration_metadata(malformed)



def test_frozen_artifact_descriptors_keep_legacy_field_name_only() -> None:
    gauge = GaugeCovarianceCalibrationV1(
        scale=1.2,
        rotation=1.3,
        translation=1.4,
        count=5,
        trim_quantile=0.9,
        **PROVENANCE,
    )
    point = PointUncertaintyCalibrationV1(
        parallel_floor=1e-4,
        parallel_depth_coefficient=2e-4,
        lateral_floor=3e-4,
        lateral_depth_coefficient=4e-4,
        disagreement_gain=0.5,
        parallel_scale=1.1,
        lateral_scale=1.2,
        count=5,
        trim_quantile=0.9,
        parallel_scale_update=1.1,
        lateral_scale_update=1.2,
        parallel_normalized_mse=1.0,
        lateral_normalized_mse=1.0,
        **PROVENANCE,
    )

    for artifact in (gauge, point):
        calibration = artifact.descriptor()["calibration"]
        assert calibration["trim_quantile"] == 0.9
        assert "winsor_quantile" not in calibration
        assert artifact.winsor_quantile == 0.9


def test_legacy_metadata_rejects_malformed_optional_winsor_alias() -> None:
    errors, covariance, groups = _group_calibration_inputs()
    artifact, _ = fit_group_balanced_point_uncertainty_calibration(
        DepthDisagreementModel(),
        errors,
        covariance,
        groups,
        group_definition="object",
        trim_quantile=0.8,
        **PROVENANCE,
    )
    metadata = artifact.to_dict()["provenance"]["metadata"]
    report = metadata["group_balanced_point_uncertainty_calibration"]["report"]
    report["aggregation"] = LEGACY_GROUP_BALANCED_TRIMMED_RATIOS_V1
    report["winsor_quantile"] = "0.8"
    malformed = replace(artifact, metadata=metadata)

    with pytest.raises(ValueError, match="must be numeric"):
        group_balanced_point_calibration_metadata(malformed)
