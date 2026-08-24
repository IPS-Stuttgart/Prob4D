from __future__ import annotations

import numpy as np
import pytest
from test_io import write_problem_bundle

from prob4d.experiments import _gauge_metrics, run_manifest_ablation, run_synthetic_ablation
from prob4d.gauge import GaugeEstimate
from prob4d.sim3 import Sim3
from prob4d.synthetic import make_synthetic_problem


def test_synthetic_ablation_covers_all_requested_variants() -> None:
    rows, calibration = run_synthetic_ablation(
        seed=3,
        num_frames=45,
        height=4,
        width=6,
    )

    assert [row.key for row in rows] == [
        "disjoint",
        "latent_linear",
        "decoded_uniform",
        "precision",
        "ci",
        "ci_smoothed",
        "ci_smoothed_anchored",
    ]
    assert calibration.count > 0
    assert rows[3].sequence_metrics.coverage_95 <= rows[4].sequence_metrics.coverage_95
    assert rows[-1].gauge_metrics is not None


def test_gauge_metrics_do_not_hide_error_in_covariance_nullspace() -> None:
    covariance = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    estimates = {
        "window-a": GaugeEstimate(
            window_id="window-a",
            global_from_local=Sim3(translation=np.array([0.0, 0.0, 0.25])),
            covariance=covariance,
        )
    }
    truth = {"window-a": Sim3.identity()}

    metrics = _gauge_metrics(estimates, truth)

    assert metrics.mean_normalized_squared_error == pytest.approx(0.0)
    assert metrics.minimum_covariance_rank == 6
    assert metrics.support_violation_count == 1
    assert metrics.maximum_nullspace_error_norm == pytest.approx(0.25)
    assert not metrics.all_errors_in_covariance_support


def test_manifest_ablation_uses_held_out_calibration_and_exact_baselines(tmp_path) -> None:
    test_problem = make_synthetic_problem(seed=31, num_frames=45, height=4, width=6, overlap=15)
    calibration_problem = make_synthetic_problem(
        seed=32, num_frames=45, height=4, width=6, overlap=15
    )
    predictions, truth = write_problem_bundle(tmp_path / "test", test_problem)
    calibration_predictions, calibration_truth = write_problem_bundle(
        tmp_path / "calibration", calibration_problem
    )

    rows, calibration, metadata = run_manifest_ablation(
        predictions=predictions,
        truth_path=truth,
        calibration_predictions=calibration_predictions,
        calibration_truth_path=calibration_truth,
    )

    assert len(rows) == 7
    assert calibration.count > 0
    assert rows[0].baseline_source == "upstream_motioncrafter"
    assert rows[1].baseline_source == "upstream_motioncrafter"
    assert metadata["metric_anchor_source"] == "simulated_from_ground_truth"
    assert rows[-1].sequence_metrics.metric_point_rmse < rows[-2].sequence_metrics.metric_point_rmse
