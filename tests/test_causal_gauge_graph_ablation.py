from __future__ import annotations

from test_io import write_problem_bundle

from prob4d.causal_gauge_graph import CAUSAL_GAUGE_GRAPH_MODE
from prob4d.causal_gauge_graph_ablation import run_causal_gauge_graph_ablation
from prob4d.synthetic import make_synthetic_problem


def test_causal_gauge_graph_ablation_preserves_paired_four_method_contract(
    tmp_path,
) -> None:
    test_problem = make_synthetic_problem(
        seed=181,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    calibration_problem = make_synthetic_problem(
        seed=182,
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

    rows, calibration, metadata = run_causal_gauge_graph_ablation(
        predictions=predictions,
        truth_path=truth,
        calibration_predictions=calibration_predictions,
        calibration_truth_path=calibration_truth,
    )

    assert [row.key for row in rows] == [
        "tree_ci",
        "sequential_multi_parent_ci",
        "causal_graph_ci",
        "fixed_lag_ci",
    ]
    assert calibration.count > 0
    assert all(
        row.baseline_source == "prob4d_causal_gauge_graph_diagnostic"
        for row in rows
    )
    assert metadata["benchmark"] == "prob4d_causal_gauge_graph_ablation_v1"
    assert metadata["graph"]["mode"] == CAUSAL_GAUGE_GRAPH_MODE
    assert metadata["graph"]["admitted_edge_count"] >= len(
        test_problem.overlap_windows
    ) - 1
    assert metadata["provider_v2_claim_bearing_export"] is False
    assert metadata["target_truth_used_only_for_evaluation"] is True
