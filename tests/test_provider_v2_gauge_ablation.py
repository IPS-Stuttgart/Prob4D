import numpy as np
from test_io import write_problem_bundle

from prob4d.composition_jacobian import composition_jacobian_mode
from prob4d.observation_export import estimate_joint_gauge_tree
from prob4d.provider_v2_gauge_ablation import (
    ALIGNMENT_COVARIANCE_POLICY,
    PROVIDER_V2_COMPOSITION_JACOBIAN_MODE,
    PROVIDER_V2_GAUGE_POSTERIOR_MODE,
    build_provider_v2_alignments,
    estimate_provider_v2_gauge_backend,
    run_provider_v2_gauge_manifest_ablation,
)
from prob4d.sim3 import Sim3
from prob4d.synthetic import make_synthetic_problem


def test_provider_v2_gauge_backend_matches_export_tree() -> None:
    problem = make_synthetic_problem(
        seed=41,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    alignments = build_provider_v2_alignments(problem.overlap_windows)
    initial_covariance = np.diag(np.full(7, 1e-10))

    posterior, marginals = estimate_provider_v2_gauge_backend(
        problem.overlap_windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=initial_covariance,
    )
    with composition_jacobian_mode(PROVIDER_V2_COMPOSITION_JACOBIAN_MODE):
        direct = estimate_joint_gauge_tree(
            problem.overlap_windows,
            alignments,
            initial_transform=Sim3.identity(),
            initial_covariance=initial_covariance,
        )

    assert posterior.mode == PROVIDER_V2_GAUGE_POSTERIOR_MODE
    assert posterior.parent_window_ids == direct.parent_window_ids
    assert posterior.selected_alignment_indices == direct.selected_alignment_indices
    np.testing.assert_allclose(posterior.joint_covariance, direct.joint_covariance)
    for index, window_id in enumerate(posterior.window_ids):
        np.testing.assert_allclose(
            posterior.estimates[window_id].as_vector(),
            direct.estimates[window_id].as_vector(),
        )
        block = slice(7 * index, 7 * (index + 1))
        np.testing.assert_allclose(
            marginals[window_id].covariance,
            direct.joint_covariance[block, block],
        )


def test_provider_v2_gauge_ablation_preserves_seven_row_contract(tmp_path) -> None:
    test_problem = make_synthetic_problem(
        seed=51,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    calibration_problem = make_synthetic_problem(
        seed=52,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    predictions, truth = write_problem_bundle(tmp_path / "test", test_problem)
    calibration_predictions, calibration_truth = write_problem_bundle(
        tmp_path / "calibration",
        calibration_problem,
    )

    rows, calibration, metadata = run_provider_v2_gauge_manifest_ablation(
        predictions=predictions,
        truth_path=truth,
        calibration_predictions=calibration_predictions,
        calibration_truth_path=calibration_truth,
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
    assert rows[0].baseline_source == "upstream_motioncrafter"
    assert rows[1].baseline_source == "upstream_motioncrafter"
    assert all(
        row.baseline_source == "prob4d_provider_v2_gauge" for row in rows[2:]
    )
    assert metadata["benchmark"] == "motioncrafter_manifest_provider_v2_gauge"
    assert metadata["metric_anchor_source"] == "simulated_from_ground_truth"
    assert metadata["gauge_backend"] == {
        "provider_api_version": 2,
        "posterior_mode": PROVIDER_V2_GAUGE_POSTERIOR_MODE,
        "anchored_posterior_mode": PROVIDER_V2_GAUGE_POSTERIOR_MODE,
        "composition_jacobian_mode": PROVIDER_V2_COMPOSITION_JACOBIAN_MODE,
        "joint_cross_window_covariance_available": True,
        "dense_fusion_covariance_adapter": "per_window_marginals",
        "alignment_covariance_policy": ALIGNMENT_COVARIANCE_POLICY,
        "gauge_covariance_calibration_id": None,
        "claim_bearing_provider_export": False,
        "legacy_ablation_runner_unchanged": True,
    }
