from __future__ import annotations

import numpy as np

from prob4d.causal_gauge_graph import (
    CAUSAL_GAUGE_GRAPH_DEPENDENCE,
    CAUSAL_GAUGE_GRAPH_MODE,
    estimate_causal_multi_edge_gauge_graph,
)
from prob4d.composition_jacobian import composition_jacobian_mode
from prob4d.observation_export import _build_alignments, estimate_joint_gauge_tree
from prob4d.sim3 import Sim3
from prob4d.synthetic import make_synthetic_problem


def _problem():
    return make_synthetic_problem(
        seed=171,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )


def test_single_edge_per_child_matches_analytic_spanning_tree() -> None:
    problem = _problem()
    alignments = _build_alignments(problem.overlap_windows)
    anchor_covariance = np.diag(np.linspace(1e-7, 7e-7, 7))
    with composition_jacobian_mode("analytic"):
        tree = estimate_joint_gauge_tree(
            problem.overlap_windows,
            alignments,
            initial_transform=Sim3.identity(),
            initial_covariance=anchor_covariance,
        )
    selected = [
        alignments[index]
        for index in tree.selected_alignment_indices[1:]
        if index is not None
    ]

    graph, report = estimate_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        selected,
        initial_transform=Sim3.identity(),
        initial_covariance=anchor_covariance,
    )

    np.testing.assert_allclose(
        graph.joint_covariance,
        tree.joint_covariance,
        atol=1e-10,
    )
    for window_id in graph.window_ids:
        np.testing.assert_allclose(
            graph.estimates[window_id].as_vector(),
            tree.estimates[window_id].as_vector(),
            atol=1e-10,
        )
    assert all(
        np.array_equal(step.covariance_intersection_weights, [1.0])
        for step in report.steps
    )


def test_multi_edge_graph_admits_every_prefix_valid_edge_and_is_psd() -> None:
    problem = _problem()
    alignments = _build_alignments(problem.overlap_windows)
    graph, report = estimate_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=np.eye(7) * 1e-6,
    )

    assert graph.mode == CAUSAL_GAUGE_GRAPH_MODE
    assert graph.cross_window_covariance_preserved
    assert report.dependence_semantics == CAUSAL_GAUGE_GRAPH_DEPENDENCE
    assert report.claim_bearing_provider_export is False
    assert report.admitted_edge_count == len(alignments)
    assert len(report.steps[-1].candidate_parent_ids) >= 2
    assert np.min(np.linalg.eigvalsh(graph.joint_covariance)) >= -1e-9
    assert np.any(np.abs(graph.joint_covariance[:7, 7:]) > 0.0)
    for step in report.steps:
        assert np.all(step.covariance_intersection_weights >= 0.0)
        np.testing.assert_allclose(
            np.sum(step.covariance_intersection_weights),
            1.0,
        )


def test_graph_posterior_is_invariant_to_alignment_input_order() -> None:
    problem = _problem()
    alignments = _build_alignments(problem.overlap_windows)
    kwargs = {
        "initial_transform": Sim3.identity(),
        "initial_covariance": np.eye(7) * 1e-6,
    }

    first, first_report = estimate_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        alignments,
        **kwargs,
    )
    second, second_report = estimate_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        list(reversed(alignments)),
        **kwargs,
    )

    np.testing.assert_allclose(
        first.joint_covariance,
        second.joint_covariance,
        atol=1e-10,
    )
    for window_id in first.window_ids:
        np.testing.assert_allclose(
            first.estimates[window_id].as_vector(),
            second.estimates[window_id].as_vector(),
            atol=1e-10,
        )
    assert [step.candidate_parent_ids for step in first_report.steps] == [
        step.candidate_parent_ids for step in second_report.steps
    ]


def test_graph_report_is_json_compatible() -> None:
    problem = _problem()
    alignments = _build_alignments(problem.overlap_windows)
    _, report = estimate_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=np.eye(7) * 1e-6,
    )

    payload = report.to_dict()
    assert payload["mode"] == CAUSAL_GAUGE_GRAPH_MODE
    assert payload["admitted_edge_count"] == len(alignments)
    assert payload["claim_bearing_provider_export"] is False
