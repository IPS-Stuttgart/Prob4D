from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.evaluation_modes import evaluate_sequence_modes
from prob4d.fusion import FusedSequence
from prob4d.io import save_fused_prediction, save_truth
from prob4d.metrics import TruthSequence, evaluate_sequence, uncertainty_diagnostics
from prob4d.provider_evaluation import run_provider_evaluation


def _prediction(
    truth: TruthSequence,
    error: float,
    *,
    valid_mask: np.ndarray | None = None,
) -> FusedSequence:
    points = truth.point_map.copy()
    points[..., 0] += error
    mask = truth.valid_mask if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    covariance = np.broadcast_to(np.eye(3), points.shape + (3,)).copy()
    return FusedSequence(
        frame_indices=truth.frame_indices,
        point_map=points,
        valid_mask=mask,
        point_covariance=covariance,
        contributors=np.ones(mask.shape, dtype=np.uint16),
    )


def _artifact_metadata() -> dict[str, object]:
    return {
        "prob4d_revision": "a" * 40,
        "motioncrafter_revision": "b" * 40,
        "motioncrafter_seed_policy": "derived-per-call",
        "motioncrafter_model_set_sha256": "c" * 64,
        "prediction_manifest_sha256": "d" * 64,
        "includes_covariance": True,
        "gauge_estimator": "sequential",
        "uncertainty_calibration": "held_out",
    }


def test_truth_sequence_is_canonical_immutable_and_alias_safe() -> None:
    frames = np.array([0, 2], dtype=np.int32)
    points = np.zeros((2, 1, 2, 3), dtype=np.float32)
    points[0, 0, 1] = np.nan
    valid = np.array([[[True, False]], [[True, True]]], dtype=np.uint8)
    flow = np.zeros_like(points)
    flow[0, 0, 1] = np.inf
    deform = np.array([[[True, False]], [[False, True]]], dtype=np.uint8)

    truth = TruthSequence(frames, points, valid, flow, deform)

    assert truth.frame_indices.dtype == np.int64
    assert truth.point_map.dtype == np.float64
    assert truth.valid_mask.dtype == np.bool_
    assert truth.scene_flow is not None
    assert truth.scene_flow.dtype == np.float64
    assert truth.deform_mask is not None
    assert truth.deform_mask.dtype == np.bool_
    assert not np.shares_memory(truth.frame_indices, frames)
    assert not np.shares_memory(truth.point_map, points)
    assert not np.shares_memory(truth.valid_mask, valid)

    frames[0] = 1
    points[1, 0, 0, 0] = 9.0
    valid[1, 0, 0] = 0
    flow[1, 0, 1, 0] = 7.0
    deform[1, 0, 1] = 0
    np.testing.assert_array_equal(truth.frame_indices, [0, 2])
    np.testing.assert_allclose(truth.point_map[1, 0, 0], 0.0)
    assert truth.valid_mask[1, 0, 0]
    np.testing.assert_allclose(truth.scene_flow[1, 0, 1], 0.0)
    assert truth.deform_mask[1, 0, 1]

    for array in (
        truth.frame_indices,
        truth.point_map,
        truth.valid_mask,
        truth.scene_flow,
        truth.deform_mask,
    ):
        assert array is not None
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (np.array([0.0, 1.0]), "contain integers"),
        (np.array([-1, 0]), "non-negative int64"),
        (np.array([0, 0]), "strictly increasing"),
        (np.array([1, 0]), "strictly increasing"),
    ],
)
def test_truth_sequence_rejects_invalid_frame_contract(
    frames: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TruthSequence(
            frame_indices=frames,
            point_map=np.zeros((2, 1, 1, 3)),
            valid_mask=np.ones((2, 1, 1), dtype=bool),
        )


def test_truth_sequence_rejects_nonfinite_active_geometry() -> None:
    points = np.zeros((1, 1, 1, 3))
    points[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="active truth point_map"):
        TruthSequence(
            frame_indices=np.array([0]),
            point_map=points,
            valid_mask=np.ones((1, 1, 1), dtype=bool),
        )

    points.fill(0.0)
    flow = np.zeros_like(points)
    flow[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="active truth scene_flow"):
        TruthSequence(
            frame_indices=np.array([0]),
            point_map=points,
            valid_mask=np.ones((1, 1, 1), dtype=bool),
            scene_flow=flow,
            deform_mask=np.ones((1, 1, 1), dtype=bool),
        )


def test_sequence_metrics_include_frame_balanced_and_uncertainty_scorecard() -> None:
    truth_points = np.zeros((2, 1, 4, 3))
    prediction_points = truth_points.copy()
    prediction_points[1, 0, 0, 0] = 2.0
    valid = np.array(
        [
            [[True, True, True, True]],
            [[True, False, False, False]],
        ]
    )
    covariance = np.broadcast_to(
        np.eye(3) * 0.04,
        prediction_points.shape + (3,),
    ).copy()
    prediction = FusedSequence(
        np.arange(2),
        prediction_points,
        valid,
        covariance,
        np.ones_like(valid, dtype=np.uint16),
    )
    truth = TruthSequence(np.arange(2), truth_points, valid)

    metrics = evaluate_sequence(
        prediction,
        truth,
        align_scale_translation=False,
    )

    np.testing.assert_allclose(metrics.point_rmse, np.sqrt(4.0 / 5.0))
    np.testing.assert_allclose(metrics.frame_balanced_point_rmse, 1.0)
    np.testing.assert_allclose(metrics.mean_covariance_trace, 0.12)
    np.testing.assert_allclose(
        metrics.mean_covariance_log_determinant,
        3.0 * np.log(0.04),
    )
    assert metrics.coverage_50 == pytest.approx(0.8)
    assert metrics.coverage_95 == pytest.approx(0.8)
    assert metrics.coverage_shortfall_95 == pytest.approx(0.15)
    assert metrics.evaluated_frames == 2
    assert metrics.evaluated_flow_points == 0
    assert np.isfinite(metrics.risk_coverage_auc)


def test_selective_risk_is_permutation_invariant_under_uncertainty_ties() -> None:
    errors = np.array(
        [
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ]
    )
    covariances = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    norms = np.ones(4)
    original = uncertainty_diagnostics(errors, covariances, norms)
    permutation = np.array([3, 0, 2, 1])
    permuted = uncertainty_diagnostics(
        errors[permutation],
        covariances[permutation],
        norms[permutation],
    )

    np.testing.assert_allclose(
        original.relative_error_retained_50,
        original.mean_relative_error,
    )
    np.testing.assert_allclose(
        permuted.relative_error_retained_50,
        original.relative_error_retained_50,
    )
    np.testing.assert_allclose(
        permuted.relative_error_retained_80,
        original.relative_error_retained_80,
    )
    np.testing.assert_allclose(
        permuted.risk_coverage_auc,
        original.risk_coverage_auc,
    )


def test_truth_support_mask_controls_fit_and_evaluation_support() -> None:
    truth_points = np.asarray(
        [
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]],
            [[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]],
        ]
    )
    translation = np.asarray([1.0, -0.5, 0.25])
    predicted_points = (truth_points - translation) / 2.0
    prediction = _prediction(
        TruthSequence(np.arange(2), predicted_points, np.ones((2, 1, 3), dtype=bool)),
        0.0,
    )
    truth = TruthSequence(
        np.arange(2),
        truth_points,
        np.ones((2, 1, 3), dtype=bool),
    )
    support = np.zeros_like(truth.valid_mask)
    support[:, :, :2] = True

    result = evaluate_sequence_modes(
        prediction,
        truth,
        prefix_frame_stop_exclusive=1,
        truth_support_mask=support,
    )

    assert result.prefix_aligned is not None
    assert result.prefix_aligned.fit_point_count == 2
    assert result.metric.metrics.evaluated_points == 4
    assert result.metric.metrics.evaluated_frames == 2

    with pytest.raises(ValueError, match="truth_support_mask"):
        evaluate_sequence_modes(
            prediction,
            truth,
            truth_support_mask=np.ones((1, 1, 1), dtype=bool),
        )


def test_provider_evaluation_uses_common_support_and_reports_native_retention(
    tmp_path: Path,
) -> None:
    truth = TruthSequence(
        frame_indices=np.array([0, 1]),
        point_map=np.array(
            [
                [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]],
                [[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
            ]
        ),
        valid_mask=np.ones((2, 1, 2), dtype=bool),
    )
    selective_truth = TruthSequence(
        frame_indices=np.array([1]),
        point_map=truth.point_map[1:],
        valid_mask=np.array([[[True, False]]]),
    )
    truth_path = tmp_path / "truth.npz"
    wide_path = tmp_path / "wide.npz"
    selective_path = tmp_path / "selective.npz"
    save_truth(truth_path, truth)
    save_fused_prediction(
        wide_path,
        _prediction(truth, 0.5),
        method_id="wide",
        fusion_method="uniform",
        metadata=_artifact_metadata(),
    )
    save_fused_prediction(
        selective_path,
        _prediction(selective_truth, 0.25),
        method_id="selective",
        fusion_method="uniform",
        metadata=_artifact_metadata(),
    )
    manifest = {
        "schema_name": "prob4d.provider-evaluation",
        "schema_version": 1,
        "primary_mode": "metric",
        "reference_method": "wide",
        "cases": [
            {
                "case_id": "case",
                "group_id": "group",
                "truth": truth_path.name,
                "predictions": {
                    "wide": wide_path.name,
                    "selective": selective_path.name,
                },
                "boundary_frames": [],
                "prefix_frame_stop_exclusive": None,
            }
        ],
        "metadata": {},
    }
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"

    report = run_provider_evaluation(
        manifest_path,
        output,
        bootstrap_resamples=10,
    )

    records = {record["method_id"]: record for record in report["cases"]}
    wide_record = records["wide"]
    selective_record = records["selective"]
    assert wide_record["evaluation"]["metric"]["metrics"]["evaluated_points"] == 1
    assert (
        selective_record["evaluation"]["metric"]["metrics"]["evaluated_points"]
        == 1
    )
    assert (
        wide_record["native_support_evaluation"]["metric"]["metrics"][
            "evaluated_points"
        ]
        == 4
    )
    assert (
        selective_record["native_support_evaluation"]["metric"]["metrics"][
            "evaluated_points"
        ]
        == 1
    )
    assert wide_record["support"]["common_frame_count"] == 1
    assert wide_record["support"]["common_point_fraction_of_native"] == 0.25
    assert wide_record["support"]["common_frame_fraction_of_native"] == 0.5
    assert selective_record["support"]["common_point_fraction_of_native"] == 1.0
    assert selective_record["support"]["common_frame_fraction_of_native"] == 1.0
    np.testing.assert_allclose(
        report["aggregate"]["wide"]["metrics"][
            "metric.metrics.metric_point_rmse"
        ]["mean"],
        0.5,
    )
    assert report["schema_version"] == 2
    assert report["primary_support"] == "common_across_registered_methods"
    csv_header = output.joinpath("provider_evaluation.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "native_support.metric.metrics.metric_point_rmse" in csv_header
    assert "support.common_point_fraction_of_native" in csv_header
    markdown = output.joinpath("provider_evaluation.md").read_text(encoding="utf-8")
    assert "support shared by truth and every registered method" in markdown


def test_provider_common_flow_requires_every_registered_method(tmp_path: Path) -> None:
    point_map = np.array([[[[0.0, 0.0, 1.0]]]])
    valid = np.ones((1, 1, 1), dtype=bool)
    scene_flow = np.array([[[[1.0, 0.0, 0.0]]]])
    truth = TruthSequence(
        frame_indices=np.array([0]),
        point_map=point_map,
        valid_mask=valid,
        scene_flow=scene_flow,
        deform_mask=valid,
    )
    covariance = np.broadcast_to(np.eye(3), point_map.shape + (3,)).copy()
    with_flow = FusedSequence(
        frame_indices=np.array([0]),
        point_map=point_map,
        valid_mask=valid,
        point_covariance=covariance,
        contributors=np.ones_like(valid, dtype=np.uint16),
        scene_flow=scene_flow,
        deform_mask=valid,
        flow_covariance=covariance,
    )
    without_flow = _prediction(truth, 0.0)
    truth_path = tmp_path / "truth.npz"
    with_flow_path = tmp_path / "with-flow.npz"
    without_flow_path = tmp_path / "without-flow.npz"
    save_truth(truth_path, truth)
    save_fused_prediction(
        with_flow_path,
        with_flow,
        method_id="with_flow",
        fusion_method="uniform",
        metadata=_artifact_metadata(),
    )
    save_fused_prediction(
        without_flow_path,
        without_flow,
        method_id="without_flow",
        fusion_method="uniform",
        metadata=_artifact_metadata(),
    )
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "prob4d.provider-evaluation",
                "schema_version": 1,
                "primary_mode": "metric",
                "reference_method": "without_flow",
                "cases": [
                    {
                        "case_id": "case",
                        "group_id": "group",
                        "truth": truth_path.name,
                        "predictions": {
                            "with_flow": with_flow_path.name,
                            "without_flow": without_flow_path.name,
                        },
                        "boundary_frames": [],
                        "prefix_frame_stop_exclusive": None,
                    }
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    report = run_provider_evaluation(
        manifest_path,
        tmp_path / "output",
        bootstrap_resamples=10,
    )

    records = {record["method_id"]: record for record in report["cases"]}
    with_flow_record = records["with_flow"]
    assert with_flow_record["evaluation"]["metric"]["metrics"]["flow_epe"] is None
    assert (
        with_flow_record["native_support_evaluation"]["metric"]["metrics"][
            "flow_epe"
        ]
        == 0.0
    )
    assert with_flow_record["support"]["all_methods_have_flow"] is False
    assert with_flow_record["support"]["common_valid_flow_points"] == 0
    assert with_flow_record["support"]["common_flow_fraction_of_native"] == 0.0
    assert (
        records["without_flow"]["support"]["common_flow_fraction_of_native"]
        is None
    )


def test_provider_aggregate_records_worst_group_coverage_shortfall(
    tmp_path: Path,
) -> None:
    truth = TruthSequence(
        frame_indices=np.array([0]),
        point_map=np.array([[[[0.0, 0.0, 1.0]]]]),
        valid_mask=np.ones((1, 1, 1), dtype=bool),
    )
    cases = []
    for case_id, group_id, error in (
        ("c1", "g1", 0.0),
        ("c2", "g1", 3.0),
        ("c3", "g2", 5.0),
    ):
        truth_path = tmp_path / f"{case_id}-truth.npz"
        prediction_path = tmp_path / f"{case_id}-prediction.npz"
        save_truth(truth_path, truth)
        save_fused_prediction(
            prediction_path,
            _prediction(truth, error),
            method_id="method",
            fusion_method="uniform",
            metadata=_artifact_metadata(),
        )
        cases.append(
            {
                "case_id": case_id,
                "group_id": group_id,
                "truth": truth_path.name,
                "predictions": {"method": prediction_path.name},
                "boundary_frames": [],
                "prefix_frame_stop_exclusive": None,
            }
        )
    manifest_path = tmp_path / "evaluation.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": "prob4d.provider-evaluation",
                "schema_version": 1,
                "primary_mode": "metric",
                "reference_method": "method",
                "cases": cases,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    report = run_provider_evaluation(
        manifest_path,
        tmp_path / "output",
        bootstrap_resamples=10,
    )

    summary = report["aggregate"]["method"]["metrics"][
        "metric.metrics.coverage_shortfall_95"
    ]
    assert summary["worst_group_id"] == "g2"
    assert summary["worst_group_mean"] == pytest.approx(0.95)
