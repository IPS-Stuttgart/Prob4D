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

P = "a" * 64
C = "b" * 64


def _baseline(groups=("train-a", "train-b")):
    rows = [
        {
            "group_id": group,
            "count": 4,
            "parallel_scale_update": 1.0,
            "lateral_scale_update": 1.0,
            "parallel_normalized_mse": 1.0,
            "lateral_normalized_mse": 1.0,
        }
        for group in groups
    ]
    metadata = {
        "group_balanced_point_uncertainty_calibration": {
            "group_definition": "object-session",
            "report": {
                "aggregation": GROUP_BALANCED_UPPER_WINSORIZED_RATIOS_V2,
                "count": 4 * len(rows),
                "group_count": len(rows),
                "trim_quantile": 0.99,
                "winsor_quantile": 0.99,
                "parallel_scale_update": 1.0,
                "lateral_scale_update": 1.0,
                "parallel_normalized_mse": 1.0,
                "lateral_normalized_mse": 1.0,
                "groups": rows,
            },
        }
    }
    return PointUncertaintyCalibrationV1(
        parallel_floor=4.0,
        parallel_depth_coefficient=0.0,
        lateral_floor=4.0,
        lateral_depth_coefficient=0.0,
        disagreement_gain=0.0,
        parallel_scale=1.0,
        lateral_scale=1.0,
        count=4 * len(groups),
        trim_quantile=0.99,
        parallel_scale_update=1.0,
        lateral_scale_update=1.0,
        parallel_normalized_mse=1.0,
        lateral_normalized_mse=1.0,
        calibration_case_ids=groups,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="1" * 40,
        motioncrafter_revision="2" * 40,
        model_identifier="test-model",
        covariance_method="group-balanced-test",
        input_artifact_sha256=("3" * 64,),
        metadata=metadata,
    )


def _candidate(converged=True):
    return PointUncertaintyCalibrationV2(
        provider_manifest_id=P,
        cohort_binding_id=C,
        source_covariance_localization_id="c" * 64,
        gauge_propagation_readiness_id="f" * 64,
        source_training_sha256="d" * 64,
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


def _policy():
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


def _evaluate(*, candidate=None, baseline=None, group_ids=None, feature_names=("depth",)):
    groups = group_ids or tuple(
        group for group in ("val-a", "val-b", "val-c", "val-d") for _ in range(2)
    )
    n = len(groups)
    return evaluate_point_uncertainty_v2_promotion(
        _candidate() if candidate is None else candidate,
        residual_xyz=np.zeros((n, 3)),
        ray_directions=np.tile([[0.0, 0.0, 1.0]], (n, 1)),
        tangent_reference=np.tile([[1.0, 0.0, 0.0]], (n, 1)),
        features=np.zeros((n, 1)),
        feature_names=feature_names,
        group_ids=groups,
        baseline_calibration=_baseline() if baseline is None else baseline,
        depth_squared=np.zeros(n),
        disagreement_parallel_mean=np.zeros(n),
        disagreement_lateral_mean=np.zeros(n),
        provider_manifest_id=P,
        cohort_binding_id=C,
        validation_sha256="e" * 64,
        policy=_policy(),
    )


def test_disjoint_promotion_passes_all_registered_gates():
    report = _evaluate()
    assert report.promote_candidate
    assert all(report.criteria.values())
    assert report.summary["minimum_group_rows"] == 2
    assert report.summary["mean_nll_improvement"] > 2.0
    assert report.summary["mean_width_ratio"] == pytest.approx(0.5)
    assert report.summary["worst_group_width_ratio"] == pytest.approx(0.5)
    assert len(report.point_uncertainty_promotion_id) == 64


def test_nonconverged_candidate_is_valid_negative():
    report = _evaluate(candidate=_candidate(False))
    assert not report.promote_candidate
    assert not report.criteria["fit_converged"]


def test_train_validation_leakage_and_training_roster_mismatch_fail_closed():
    groups = ("train-a", "train-a", "val-b", "val-b", "val-c", "val-c", "val-d", "val-d")
    with pytest.raises(ValueError, match="disjoint training and validation"):
        _evaluate(group_ids=groups)
    with pytest.raises(ValueError, match="same independent training groups"):
        _evaluate(baseline=_baseline(("train-a", "train-c")))


def test_feature_contract_and_report_tampering_fail_closed():
    with pytest.raises(ValueError, match="feature_names"):
        _evaluate(feature_names=("different",))
    report = _evaluate()
    loaded = PointUncertaintyPromotionReportV1.from_dict(report.to_dict())
    assert loaded.to_dict() == report.to_dict()
    tampered = report.to_dict()
    tampered["promote_candidate"] = False
    with pytest.raises(ValueError):
        PointUncertaintyPromotionReportV1.from_dict(tampered)
