from prob4d.experiments import run_manifest_ablation, run_synthetic_ablation
from prob4d.synthetic import make_synthetic_problem

from test_io import write_problem_bundle


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


def test_manifest_ablation_uses_held_out_calibration_and_exact_baselines(tmp_path) -> None:
    test_problem = make_synthetic_problem(
        seed=31, num_frames=45, height=4, width=6, overlap=15
    )
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
    assert (
        rows[-1].sequence_metrics.metric_point_rmse
        < rows[-2].sequence_metrics.metric_point_rmse
    )
