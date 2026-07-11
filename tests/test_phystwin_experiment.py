import json
import pickle
from pathlib import Path

import numpy as np

from prob4d.data import PredictionWindow
from prob4d.phystwin_experiment import (
    ErrorSummary,
    ManualFlowSamples,
    fit_metric_gauge,
    flow_method_metrics,
    load_physics_trajectory,
    load_prediction_product,
)
from prob4d.sim3 import Sim3


def test_fit_metric_gauge_uses_only_preboundary_frames() -> None:
    generator = np.random.default_rng(2)
    local = generator.normal(size=(4, 3, 4, 3))
    transform = Sim3.from_vector(np.array([0.2, 0.1, -0.05, 0.03, 0.4, -0.2, 0.1]))
    truth = transform.transform_points(local)
    truth[2:] += 10.0
    prediction = PredictionWindow(
        "prediction",
        np.array([10, 11, 12, 13]),
        local,
        np.ones(local.shape[:-1], dtype=bool),
    )

    result = fit_metric_gauge(
        prediction,
        truth,
        np.ones(local.shape[:-1], dtype=bool),
        fit_end_frame=12,
        maximum_correspondences=1000,
        seed=0,
    )

    np.testing.assert_allclose(result.transform.as_vector(), transform.as_vector(), atol=1e-10)


def test_training_calibrated_flow_fusion_uses_only_fit_errors() -> None:
    frames = np.array([0, 0, 2, 2])
    truth_current = np.zeros((4, 3))
    truth_next = np.tile([1.0, 0.0, 0.0], (4, 1))
    visual_flow = np.tile([1.2, 0.0, 0.0], (4, 1))
    samples = ManualFlowSamples(
        frame_indices=frames,
        visual_current_world=truth_current,
        visual_flow_world=visual_flow,
        truth_current_world=truth_current,
        truth_next_world=truth_next,
    )
    physics = np.zeros((4, 2, 3))
    physics[:, :, 0] = np.arange(4)[:, None]

    result = flow_method_metrics(samples, physics, None, fit_end_frame=2)

    weight = result["calibration"]["calibrated_visual_physics"]["visual_weight"]
    assert weight < 0.01
    assert (
        result["test"]["calibrated_visual_physics"]["flow_epe"]["mean_m"]
        < result["test"]["visual"]["flow_epe"]["mean_m"]
    )


def test_error_summary_drops_nonfinite_annotation_rows() -> None:
    summary = ErrorSummary.from_vectors(
        np.array([[0.01, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    )

    assert summary.count == 1
    assert summary.mean_m == 0.01


def test_load_physics_trajectory_keeps_surface_contract(tmp_path: Path) -> None:
    trajectory = np.zeros((3, 6, 3), dtype=np.float32)
    final_data = {
        "object_points": np.zeros((3, 2, 3)),
        "surface_points": np.zeros((1, 3)),
    }
    trajectory_path = tmp_path / "trajectory.pkl"
    final_data_path = tmp_path / "final_data.pkl"
    with trajectory_path.open("wb") as handle:
        pickle.dump(trajectory, handle)
    with final_data_path.open("wb") as handle:
        pickle.dump(final_data, handle)

    loaded = load_physics_trajectory(trajectory_path, final_data_path)

    assert loaded.shape == (3, 3, 3)


def test_load_prediction_product_preserves_absolute_frames(tmp_path: Path) -> None:
    points = np.zeros((2, 2, 3, 3), dtype=np.float32)
    prediction = PredictionWindow(
        "baseline",
        np.array([110, 111]),
        points,
        np.ones(points.shape[:-1], dtype=bool),
    )
    prediction.to_npz(tmp_path / "baseline.npz")
    manifest = {
        "format_version": 1,
        "disjoint_baseline": "baseline.npz",
        "latent_linear_baseline": "baseline.npz",
    }
    manifest_path = tmp_path / "predictions.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, _ = load_prediction_product(manifest_path, "disjoint")

    np.testing.assert_array_equal(loaded.frame_indices, [110, 111])
