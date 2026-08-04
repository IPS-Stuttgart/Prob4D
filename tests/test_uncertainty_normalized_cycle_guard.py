from __future__ import annotations

from dataclasses import replace

import numpy as np

from prob4d.alignment import AlignmentResult, WindowAlignment
from prob4d.composition_jacobian import composition_jacobian_mode
from prob4d.observation_export import _build_alignments, estimate_joint_gauge_tree
from prob4d.sim3 import Sim3
from prob4d.synthetic import make_synthetic_problem
from prob4d.uncertainty_guarded_causal_gauge_graph import (
    UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_MODE,
    estimate_uncertainty_guarded_causal_multi_edge_gauge_graph,
)
from prob4d.uncertainty_normalized_cycles import (
    UNCERTAINTY_NORMALIZED_CYCLE_SEMANTICS,
    audit_uncertainty_normalized_alignment_cycles,
)


def _alignment(
    reference_id: str,
    moving_id: str,
    translation: float,
    *,
    covariance_scale: float,
) -> WindowAlignment:
    return WindowAlignment(
        reference_id=reference_id,
        moving_id=moving_id,
        common_frames=np.array([1, 2], dtype=np.int64),
        result=AlignmentResult(
            transform=Sim3(translation=np.array([translation, 0.0, 0.0])),
            covariance=np.eye(7) * covariance_scale,
            residual_rms=0.01,
            inlier_fraction=1.0,
            num_correspondences=40,
        ),
    )


def _triangle(*, direct_bias: float, covariance_scale: float):
    return (
        _alignment("w0", "w1", 1.0, covariance_scale=covariance_scale),
        _alignment("w1", "w2", 1.0, covariance_scale=covariance_scale),
        _alignment(
            "w0",
            "w2",
            2.0 + direct_bias,
            covariance_scale=covariance_scale,
        ),
    )


def test_normalized_cycle_score_is_zero_for_an_exact_cycle() -> None:
    audit = audit_uncertainty_normalized_alignment_cycles(
        _triangle(direct_bias=0.0, covariance_scale=1e-4),
    )

    assert audit.cycle_count == 1
    assert audit.maximum_observed_normalized_score == 0.0
    assert audit.cycles[0].representative_displacement == 0.0
    assert audit.to_dict()["semantics"] == UNCERTAINTY_NORMALIZED_CYCLE_SEMANTICS


def test_normalized_cycle_score_decreases_with_declared_source_uncertainty() -> None:
    precise = audit_uncertainty_normalized_alignment_cycles(
        _triangle(direct_bias=0.2, covariance_scale=1e-6),
    )
    uncertain = audit_uncertainty_normalized_alignment_cycles(
        _triangle(direct_bias=0.2, covariance_scale=1e-2),
    )

    assert (
        precise.maximum_observed_representative_displacement
        == uncertain.maximum_observed_representative_displacement
    )
    assert (
        precise.maximum_observed_normalized_score
        > 50.0 * uncertain.maximum_observed_normalized_score
    )
    assert precise.cycles[0].minkowski_uncertainty_scale < (
        uncertain.cycles[0].minkowski_uncertainty_scale
    )


def _problem():
    return make_synthetic_problem(
        seed=271,
        num_frames=45,
        height=4,
        width=6,
        overlap=15,
    )


def test_uncertainty_guard_accepts_a_consistent_source_graph() -> None:
    problem = _problem()
    alignments = _build_alignments(problem.overlap_windows)
    audit = audit_uncertainty_normalized_alignment_cycles(alignments)
    threshold = 1.01 * audit.maximum_observed_normalized_score

    posterior, report = estimate_uncertainty_guarded_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=np.eye(7) * 1e-6,
        maximum_normalized_cycle_score=threshold,
    )

    assert posterior.mode == "causal_full_joint_ci_graph_v1"
    assert report.fallback_applied is False
    assert report.cycle_audit.passed is True
    assert report.graph_report is not None
    assert report.to_dict()["mode"] == UNCERTAINTY_GUARDED_CAUSAL_GAUGE_GRAPH_MODE
    assert report.claim_bearing_provider_export is False


def test_uncertainty_guard_returns_the_exact_tree_for_a_precise_biased_edge() -> None:
    problem = _problem()
    alignments = _build_alignments(problem.overlap_windows)
    direct_index = next(
        index
        for index, alignment in enumerate(alignments)
        if alignment.reference_id == problem.overlap_windows[0].window_id
        and alignment.moving_id == problem.overlap_windows[2].window_id
    )
    direct = alignments[direct_index]
    transform = direct.result.transform
    alignments[direct_index] = replace(
        direct,
        result=replace(
            direct.result,
            transform=Sim3(
                scale=transform.scale,
                rotation=transform.rotation,
                translation=transform.translation + np.array([0.5, 0.0, 0.0]),
            ),
        ),
    )
    anchor_covariance = np.eye(7) * 1e-6
    with composition_jacobian_mode("analytic"):
        expected = estimate_joint_gauge_tree(
            problem.overlap_windows,
            alignments,
            initial_transform=Sim3.identity(),
            initial_covariance=anchor_covariance,
        )
    audit = audit_uncertainty_normalized_alignment_cycles(alignments)
    threshold = 0.5 * audit.maximum_observed_normalized_score

    posterior, report = estimate_uncertainty_guarded_causal_multi_edge_gauge_graph(
        problem.overlap_windows,
        alignments,
        initial_transform=Sim3.identity(),
        initial_covariance=anchor_covariance,
        maximum_normalized_cycle_score=threshold,
    )

    assert report.fallback_applied is True
    assert report.fallback_reason == "uncertainty_normalized_cycle_inconsistency"
    assert report.graph_report is None
    np.testing.assert_array_equal(
        posterior.joint_covariance,
        expected.joint_covariance,
    )
    for window_id in posterior.window_ids:
        np.testing.assert_array_equal(
            posterior.estimates[window_id].as_vector(),
            expected.estimates[window_id].as_vector(),
        )
