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
    assert "cycle_guard_enabled" not in metadata
    assert "guarded_graph" not in metadata
    assert "maximum_cycle_displacement" not in metadata
    assert metadata["graph"]["mode"] == CAUSAL_GAUGE_GRAPH_MODE
    assert metadata["graph"]["admitted_edge_count"] >= len(
        test_problem.overlap_windows
    ) - 1
    assert metadata["provider_v2_claim_bearing_export"] is False
    assert metadata["target_truth_used_only_for_evaluation"] is True


def test_cycle_guard_adds_paired_candidate_without_changing_default_contract(
    tmp_path,
) -> None:
    test_problem = make_synthetic_problem(
        seed=183,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    calibration_problem = make_synthetic_problem(
        seed=184,
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

    rows, _, metadata = run_causal_gauge_graph_ablation(
        predictions=predictions,
        truth_path=truth,
        calibration_predictions=calibration_predictions,
        calibration_truth_path=calibration_truth,
        maximum_cycle_displacement=1.0,
        cycle_representative_radius=1.0,
    )

    assert [row.key for row in rows] == [
        "tree_ci",
        "sequential_multi_parent_ci",
        "causal_graph_ci",
        "guarded_causal_graph_ci",
        "fixed_lag_ci",
    ]
    assert metadata["benchmark"] == "prob4d_causal_gauge_graph_ablation_v2"
    assert metadata["cycle_guard_enabled"] is True
    assert metadata["guarded_graph"]["fallback_applied"] is False
    assert metadata["guarded_graph"]["cycle_audit"]["passed"] is True
    assert metadata["paired_method_order"] == [row.key for row in rows]


def test_failed_cycle_guard_emits_exact_tree_fallback_row(tmp_path) -> None:
    test_problem = make_synthetic_problem(
        seed=185,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )
    calibration_problem = make_synthetic_problem(
        seed=186,
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

    rows, _, metadata = run_causal_gauge_graph_ablation(
        predictions=predictions,
        truth_path=truth,
        calibration_predictions=calibration_predictions,
        calibration_truth_path=calibration_truth,
        maximum_cycle_displacement=1e-12,
        cycle_representative_radius=1.0,
    )

    by_key = {row.key: row for row in rows}
    tree = by_key["tree_ci"].flattened()
    guarded = by_key["guarded_causal_graph_ci"].flattened()
    for field in ("key", "label"):
        tree.pop(field)
        guarded.pop(field)
    assert guarded == tree
    assert metadata["guarded_graph"]["fallback_applied"] is True
    assert metadata["guarded_graph"]["fallback_reason"] == "cycle_inconsistency"
    assert metadata["guarded_graph"]["returned_posterior_mode"] == (
        "sequential_joint_spanning_tree_v1"
    )
