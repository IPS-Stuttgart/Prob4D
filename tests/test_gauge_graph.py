from dataclasses import dataclass
from itertools import permutations

import numpy as np

from prob4d.gauge_graph import (
    GaugeCandidate,
    GaugeGraphEdge,
    OrderInvariantSequentialGaugeEstimator,
    estimate_joint_gauge_tree,
    fuse_sim3_generalized_ci,
    loop_closure_diagnostics,
    right_invariant_residual,
    select_uncertainty_volume_spanning_tree,
)
from prob4d.sim3 import Sim3


@dataclass(frozen=True)
class Constraint:
    reference_id: str
    moving_id: str
    reference_from_moving: Sim3
    covariance: np.ndarray
    residual_rms: float = 0.0
    num_correspondences: int = 100


def relative(
    reference_id: str,
    moving_id: str,
    gauges: dict[str, Sim3],
    *,
    covariance_scale: float = 1e-4,
    noise: np.ndarray | None = None,
    correspondences: int = 100,
) -> Constraint:
    transform = gauges[reference_id].inverse().compose(gauges[moving_id])
    if noise is not None:
        transform = transform.compose(Sim3.from_vector(noise))
    return Constraint(
        reference_id=reference_id,
        moving_id=moving_id,
        reference_from_moving=transform,
        covariance=np.eye(7) * covariance_scale,
        residual_rms=0.0 if noise is None else float(np.linalg.norm(noise)),
        num_correspondences=correspondences,
    )


def test_constraint_content_id_is_orientation_independent() -> None:
    gauges = {
        "left": Sim3.identity(),
        "right": Sim3.from_vector(
            np.array([0.03, 0.04, -0.02, 0.01, 0.3, 0.1, -0.2])
        ),
    }
    forward = relative(
        "left",
        "right",
        gauges,
        covariance_scale=3e-4,
        correspondences=123,
    )
    edge = GaugeGraphEdge.from_constraint(forward)
    right_from_left, covariance = edge.oriented("right", "left")
    reversed_constraint = Constraint(
        reference_id="right",
        moving_id="left",
        reference_from_moving=right_from_left,
        covariance=covariance,
        residual_rms=forward.residual_rms,
        num_correspondences=forward.num_correspondences,
    )

    assert GaugeGraphEdge.from_constraint(reversed_constraint).edge_id == edge.edge_id


def test_generalized_ci_is_invariant_to_candidate_order() -> None:
    candidates = [
        GaugeCandidate(
            "a",
            Sim3.from_vector(
                np.array([0.01, 0.02, -0.01, 0.0, 0.1, 0.0, 0.0])
            ),
            np.diag([1e-3, 2e-3, 2e-3, 2e-3, 4e-3, 4e-3, 4e-3]),
        ),
        GaugeCandidate(
            "b",
            Sim3.from_vector(
                np.array([0.02, 0.01, -0.02, 0.01, 0.12, -0.01, 0.0])
            ),
            np.diag([2e-3, 1e-3, 1e-3, 1e-3, 3e-3, 3e-3, 3e-3]),
        ),
        GaugeCandidate(
            "c",
            Sim3.from_vector(
                np.array([0.0, 0.03, -0.01, -0.01, 0.09, 0.01, 0.0])
            ),
            np.diag([3e-3, 3e-3, 3e-3, 3e-3, 2e-3, 2e-3, 2e-3]),
        ),
    ]

    baseline = fuse_sim3_generalized_ci(candidates)
    for ordering in permutations(candidates):
        result = fuse_sim3_generalized_ci(ordering)
        assert result.candidate_labels == baseline.candidate_labels
        np.testing.assert_allclose(result.weights, baseline.weights, atol=1e-12)
        np.testing.assert_allclose(
            result.global_from_local.as_vector(),
            baseline.global_from_local.as_vector(),
            atol=1e-11,
        )
        np.testing.assert_allclose(result.covariance, baseline.covariance, atol=1e-11)


def test_sequential_estimator_is_invariant_to_constraint_order() -> None:
    truth = {
        "w0": Sim3.identity(),
        "w1": Sim3.from_vector(
            np.array([0.01, 0.02, 0.0, 0.0, 0.5, 0.0, 0.0])
        ),
        "w2": Sim3.from_vector(
            np.array([0.02, 0.03, 0.01, 0.0, 1.0, 0.1, 0.0])
        ),
    }
    constraints = [
        relative("w0", "w1", truth, covariance_scale=1e-3),
        relative(
            "w0",
            "w2",
            truth,
            covariance_scale=2e-3,
            noise=np.array([0.002, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0]),
        ),
        relative(
            "w1",
            "w2",
            truth,
            covariance_scale=1e-3,
            noise=np.array([-0.001, 0.0, 0.0, 0.0, -0.005, 0.0, 0.0]),
        ),
    ]
    estimator = OrderInvariantSequentialGaugeEstimator()
    baseline = estimator.estimate(["w0", "w1", "w2"], constraints)
    for ordering in permutations(constraints):
        result = estimator.estimate(["w0", "w1", "w2"], ordering)
        for window_id in truth:
            np.testing.assert_allclose(
                result[window_id].global_from_local.as_vector(),
                baseline[window_id].global_from_local.as_vector(),
                atol=1e-11,
            )
            np.testing.assert_allclose(
                result[window_id].covariance,
                baseline[window_id].covariance,
                atol=1e-11,
            )


def test_spanning_tree_is_global_and_order_invariant() -> None:
    truth = {
        "w0": Sim3.identity(),
        "w1": Sim3.from_vector(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ),
        "w2": Sim3.from_vector(
            np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        ),
        "w3": Sim3.from_vector(
            np.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
        ),
    }
    constraints = [
        relative("w0", "w1", truth, covariance_scale=1e-2, correspondences=20),
        relative("w0", "w2", truth, covariance_scale=1e-4, correspondences=200),
        relative("w1", "w2", truth, covariance_scale=1e-4, correspondences=200),
        relative("w1", "w3", truth, covariance_scale=1e-4, correspondences=200),
        relative("w2", "w3", truth, covariance_scale=1e-2, correspondences=20),
    ]
    baseline = select_uncertainty_volume_spanning_tree(tuple(truth), constraints)
    baseline_ids = {edge.edge_id for edge in baseline}
    assert len(baseline_ids) == 3
    for ordering in permutations(constraints):
        result = select_uncertainty_volume_spanning_tree(tuple(truth), ordering)
        assert {edge.edge_id for edge in result} == baseline_ids


def test_joint_tree_preserves_cross_window_covariance_and_detects_bad_loop() -> None:
    truth = {
        "w0": Sim3.identity(),
        "w1": Sim3.from_vector(
            np.array([0.01, 0.01, 0.0, 0.0, 0.4, 0.0, 0.0])
        ),
        "w2": Sim3.from_vector(
            np.array([0.02, 0.02, 0.0, 0.0, 0.8, 0.1, 0.0])
        ),
    }
    good_edges = [
        relative("w0", "w1", truth, covariance_scale=1e-5, correspondences=500),
        relative("w1", "w2", truth, covariance_scale=1e-5, correspondences=500),
    ]
    bad_loop = relative(
        "w0",
        "w2",
        truth,
        covariance_scale=1e-5,
        noise=np.array([0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0]),
        correspondences=10,
    )
    posterior = estimate_joint_gauge_tree(
        tuple(truth),
        [*good_edges, bad_loop],
        root_window_id="w0",
        initial_transform=Sim3.identity(),
        initial_covariance=np.eye(7) * 1e-6,
    )
    cross = posterior.joint_covariance[7:14, 14:21]
    assert np.linalg.norm(cross) > 0.0
    assert np.min(np.linalg.eigvalsh(posterior.joint_covariance)) >= -1e-11
    diagnostics = loop_closure_diagnostics(posterior, [*good_edges, bad_loop])
    assert len(diagnostics) == 1
    assert diagnostics[0].suspicious
    assert diagnostics[0].normalized_innovation_squared > diagnostics[0].threshold


def test_intrinsic_residual_is_finite_across_rotation_branch() -> None:
    first = Sim3.from_vector(
        np.array([0.0, np.pi - 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    second = Sim3.from_vector(
        np.array([0.0, -np.pi + 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    residual = right_invariant_residual(first, second)
    assert np.all(np.isfinite(residual))
    assert np.linalg.norm(residual[1:4]) < 1e-4
