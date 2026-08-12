from __future__ import annotations

import numpy as np
import pytest

from prob4d._strict_calibration import PointUncertaintyCalibrationV1
from prob4d.calibration_aggregation import GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2
from prob4d.point_uncertainty_v2 import (
    PointUncertaintyCalibrationPolicyV2,
    PointUncertaintyCalibrationV2,
)
from prob4d.point_uncertainty_v2_promotion import (
    PointUncertaintyPromotionPolicyV1,
    PointUncertaintyPromotionReportV1,
    evaluate_point_uncertainty_v2_promotion,
)

PROVIDER_ID = "a" * 64
COHORT_ID = "b" * 64
LOCALIZATION_ID = "c" * 64
TRAINING_SHA = "d" * 64
VALIDATION_SHA = "e" * 64


def _group_metadata(group_ids: tuple[str, ...]) -> dict[str, object]:
    groups = [
        {
            "group_id": group_id,
            "count": 4,
            "parallel_scale_update": 1.0,
            "lateral_scale_update": 1.0,
            "parallel_normalized_mse": 1.0,
            "lateral_normalized_mse": 1.0,
        }
        for group_id in group_ids
    ]
    return {
        "group_balanced_point_uncertainty_calibration": {
            "group_definition": "object-session",
            "report": {
                "aggregation": GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
                "count": 4 * len(groups),
                "group_count": len(groups),
                "trim_quantile": 0.99,
                "winsor_quantile": 0.99,
                "parallel_scale_update": 1.0,
                "lateral_scale_update": 1.0,
                "parallel_normalized_mse": 1.0,
                "lateral_normalized_mse": 1.0,
                "groups": groups,
            },
        }
    }


def _baseline(
    group_ids: tuple[str, ...] = ("train-a", "train-b"),
) -> PointUncertaintyCalibrationV1:
    return PointUncertaintyCalibrationV1(
        parallel_floor=4.0,
        parallel_depth_coefficient=0.0,
        lateral_floor=4.0,
        lateral_depth_coefficient=0.0,
        disagreement_gain=0.0,
        parallel_scale=1.0,
        lateral_scale=1.0,
        count=8,
        trim_quantile=0.99,
        parallel_scale_update=1.0,
        lateral_scale_update=1.0,
        parallel_normalized_mse=1.0,
        lateral_normalized_mse=1.0,
        calibration_case_ids=group_ids,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="1" * 40,
        motioncrafter_revision="2" * 40,
        model_identifier="test-model",
        covariance_method="group-balanced-test",
        input_artifact_sha256=("3" * 64,),
        metadata=_group_metadata(group_ids),
    )


def _candidate(*, converged: bool = True) -> PointUncertaintyCalibrationV2:
    return PointUncertaintyCalibrationV2(
        provider_manifest_id=PROVIDER_ID,
        cohort_binding_id=COHORT_ID,
        source_covariance_localization_id=LOCALIZATION_ID,
        source_training_sha256=TRAINING_SHA,
        feature_names=("depth",),
        feature_mean=(0.0,),
        feature_scale=(1.0,),
        parallel_coefficients=(0.0, 0.0),
        lateral_reference_coefficients=(0.0, 0.0),
        lateral_orthogonal_coefficients=(0.0, 0.0),
        group_ids=("train-a", "train-b"),
        group_counts=(4, 4),
        policy=PointUncertaintyCalibrationPolicyV2(
            minimum_group_count=2,
            minimum_rows_per_group=1,
        ),
        training_normalized_energy=(1.0, 1.0, 1.0),
        fit_iterations=2,
        fit_converged=converged,
    )


def _policy() -> PointUncertaintyPromotionPolicyV1:
    return PointUncertaintyPromotionPolicyV1(
        minimum_group_count=4,
        minimum_rows_per_group=2,
        minimum_mean_nll_improvement=1.0,
        minimum_group_win_fraction=1.0,
        maximum_coverage_error_increase=0.0,
        maximum_worst_group_coverage_error_increase=0.0,
        maximum_mean_width_ratio=0.75,
        maximum_worst_group_width_ratio=0.75,
        maximum_worst_group_nll_regression=0.0,
    )


def _validation() -> dict[str, object]:
    group_ids = tuple(
        group_id
        for group_id in ("val-a", "val-b", "val-c", "val-d")
        for _ in range(2)
    )
    rows = len(group_ids)
    return {
        "residual_xyz": np.zeros((rows, 3), dtype=np.float64),
        "ray_directions": np.tile(np.asarray([[0.0, 0.0, 1.0]]), (rows, 1)),
        "tangent_reference": np.tile(np.asarray([[1.0, 0.0, 0.0]]), (rows, 1)),
        "features": np.zeros((rows, 1), dtype=np.float64),
        "feature_names": ("depth",),
        "group_ids": group_ids,
        "depth_squared": np.zeros(rows, dtype=np.float64),
        "disagreement_parallel_mean": np.zeros(rows, dtype=np.float64),
        "disagreement_lateral_mean": np.zeros(rows, dtype=np.float64),
    }


def _evaluate(
    *,
    candidate: PointUncertaintyCalibrationV2 | None = None,
    baseline: PointUncertaintyCalibrationV1 | None = None,
    validation: dict[str, object] | None = None,
) -> PointUncertaintyPromotionReportV1:
    data = _validation() if validation is None else validation
    return evaluate_point_uncertainty_v2_promotion(
        _candidate() if candidate is None else candidate,
        baseline_calibration=_baseline() if baseline is None else baseline,
        residual_xyz=data["residual_xyz"],
        ray_directions=data["ray_directions"],
        tangent_reference=data["tangent_reference"],
        features=data["features"],
        feature_names=data["feature_names"],
        group_ids=data["group_ids"],
        depth_squared=data["depth_squared"],
        disagreement_parallel_mean=data["disagreement_parallel_mean"],
        disagreement_lateral_mean=data["disagreement_lateral_mean"],
        provider_manifest_id=PROVIDER_ID,
        cohort_binding_id=COHORT_ID,
        validation_sha256=VALIDATION_SHA,
        policy=_policy(),
    )


def test_disjoint_group_evaluation_promotes_clear_proper_score_gain() -> None:
    report = _evaluate()

    assert report.promote_candidate
    assert report.criteria == {
        "fit_converged": True,
        "minimum_group_count": True,
        "minimum_rows_per_group": True,
        "mean_nll_improvement": True,
        "group_win_fraction": True,
        "coverage_nonworse": True,
        "worst_group_coverage_nonworse": True,
        "width_budget": True,
        "worst_group_width_budget": True,
        "worst_group_nll_regression": True,
    }
    assert report.summary["minimum_group_rows"] == 2
    assert report.summary["worst_group_coverage_error_increase"] == pytest.approx(0.0)
    assert report.summary["worst_group_width_ratio"] == pytest.approx(0.5)
    assert report.summary["group_count"] == 4
    assert report.summary["mean_nll_improvement"] > 2.0
    assert report.summary["group_win_fraction"] == 1.0
    assert report.summary["mean_width_ratio"] == pytest.approx(0.5)
    assert report.validation_group_ids == ("val-a", "val-b", "val-c", "val-d")
    assert len(report.point_uncertainty_promotion_id) == 64


def test_nonconverged_candidate_cannot_be_promoted() -> None:
    report = _evaluate(candidate=_candidate(converged=False))

    assert not report.promote_candidate
    assert not report.criteria["fit_converged"]


def test_validation_groups_must_be_disjoint_from_training() -> None:
    validation = _validation()
    validation["group_ids"] = (
        "train-a",
        "train-a",
        "val-b",
        "val-b",
        "val-c",
        "val-c",
        "val-d",
        "val-d",
    )

    with pytest.raises(ValueError, match="disjoint training and validation"):
        _evaluate(validation=validation)


def test_baseline_and_candidate_must_share_training_roster() -> None:
    with pytest.raises(ValueError, match="same independent training groups"):
        _evaluate(baseline=_baseline(("train-a", "train-c")))


def test_feature_contract_must_match_candidate() -> None:
    validation = _validation()
    validation["feature_names"] = ("different",)

    with pytest.raises(ValueError, match="feature_names"):
        _evaluate(validation=validation)


def test_report_round_trip_replays_derived_decision() -> None:
    report = _evaluate()
    loaded = PointUncertaintyPromotionReportV1.from_dict(report.to_dict())

    assert loaded.to_dict() == report.to_dict()
    tampered = report.to_dict()
    tampered["promote_candidate"] = False
    with pytest.raises(ValueError):
        PointUncertaintyPromotionReportV1.from_dict(tampered)


def test_policy_rejects_negative_harm_budget() -> None:
    with pytest.raises((TypeError, ValueError)):
        PointUncertaintyPromotionPolicyV1(
            minimum_group_count=4,
            minimum_rows_per_group=2,
            minimum_mean_nll_improvement=0.0,
            minimum_group_win_fraction=0.5,
            maximum_coverage_error_increase=0.0,
            maximum_worst_group_coverage_error_increase=0.0,
            maximum_mean_width_ratio=1.0,
            maximum_worst_group_width_ratio=1.0,
            maximum_worst_group_nll_regression=-0.1,
        )
