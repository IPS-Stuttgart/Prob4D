from __future__ import annotations

import numpy as np
import pytest

from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)
from prob4d.recursive_task_sufficiency import recursive_linear_task_closure


class DenseReference:
    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.dimension = self.covariance.shape[0]
        self.observation_count = self.dimension // 3

    def solve(self, value: object) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        return np.linalg.solve(
            self.covariance, raw.reshape(self.dimension, -1)
        ).reshape(raw.shape)


def _kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    noise: np.ndarray,
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation_covariance = observation @ covariance @ observation.T + noise
    gain = np.linalg.solve(
        innovation_covariance,
        (covariance @ observation.T).T,
    ).T
    updated_mean = mean + gain @ (value - observation @ mean)
    updated_covariance = covariance - gain @ observation @ covariance
    return updated_mean, 0.5 * (updated_covariance + updated_covariance.T)


def _controlled_recursive_design() -> dict[str, np.ndarray]:
    state_dimension = 20
    closure_dimension = 4
    task_dimension = 3
    observation_dimension = 12

    task_state = np.zeros((closure_dimension, state_dimension))
    task_state[:, :closure_dimension] = np.eye(closure_dimension)
    task = np.eye(closure_dimension)[:task_dimension] @ task_state

    observation_map = np.vstack([np.eye(closure_dimension) for _ in range(3)])
    observation = observation_map @ task_state

    task_transition = np.array(
        [
            [0.92, 0.05, 0.00, 0.28],
            [0.00, 0.90, 0.06, 0.00],
            [0.00, 0.00, 0.88, 0.04],
            [0.00, 0.00, 0.00, 0.80],
        ]
    )
    transition = np.zeros((state_dimension, state_dimension))
    transition[:closure_dimension, :closure_dimension] = task_transition
    transition[closure_dimension:, closure_dimension:] = 0.75 * np.eye(
        state_dimension - closure_dimension
    )
    transition[closure_dimension:, :closure_dimension] = np.linspace(
        -0.03,
        0.03,
        (state_dimension - closure_dimension) * closure_dimension,
    ).reshape(state_dimension - closure_dimension, closure_dimension)

    _, _, right = np.linalg.svd(observation_map.T, full_matrices=True)
    orthogonal_observation_noise = right[closure_dimension:].T[:, :4]
    task_noise = observation_map @ np.diag([0.18, 0.16, 0.14, 0.22])
    irrelevant_noise = orthogonal_observation_noise @ np.diag([0.11, 0.10, 0.09, 0.08])
    shared_factor = np.column_stack((task_noise, irrelevant_noise))

    conditional_noise = 0.04**2 * np.eye(observation_dimension)
    process_noise = np.diag(
        np.concatenate(
            (
                np.full(closure_dimension, 0.01**2),
                np.full(state_dimension - closure_dimension, 0.02**2),
            )
        )
    )
    initial_task_covariance = np.diag([0.15, 0.12, 0.10, 0.20]) ** 2
    initial_covariance = np.zeros((state_dimension, state_dimension))
    initial_covariance[:closure_dimension, :closure_dimension] = initial_task_covariance
    initial_covariance[closure_dimension:, closure_dimension:] = 0.30**2 * np.eye(
        state_dimension - closure_dimension
    )

    return {
        "task_state": task_state,
        "task": task,
        "observation_map": observation_map,
        "observation": observation,
        "task_transition": task_transition,
        "transition": transition,
        "shared_factor": shared_factor,
        "conditional_noise": conditional_noise,
        "process_noise": process_noise,
        "initial_task_covariance": initial_task_covariance,
        "initial_covariance": initial_covariance,
    }


def test_recursive_closure_is_minimal_and_detects_violation() -> None:
    design = _controlled_recursive_design()
    closure = recursive_linear_task_closure(
        design["transition"],
        task_matrix=design["task"],
        observation_matrix=design["observation"],
    )
    assert closure.closure_dimension == 4
    assert closure.state_dimension == 20
    assert closure.task_residual < 1e-12
    assert closure.observation_residual < 1e-12
    assert closure.transition_residual < 1e-12

    violated = design["transition"].copy()
    violated[0, 4] = 0.20
    expanded = recursive_linear_task_closure(
        violated,
        task_matrix=design["task"],
        observation_matrix=design["observation"],
    )
    assert expanded.closure_dimension == 5


def test_closure_aware_compression_is_recursively_exact() -> None:
    design = _controlled_recursive_design()
    task_state = design["task_state"]
    task_decoder = np.eye(4)[:3]
    observation = design["observation"]
    observation_map = design["observation_map"]
    transition = design["transition"]
    task_transition = design["task_transition"]
    shared_factor = design["shared_factor"]
    conditional_noise = design["conditional_noise"]
    full_noise = conditional_noise + shared_factor @ shared_factor.T
    process_noise = design["process_noise"]
    task_process_noise = task_state @ process_noise @ task_state.T

    full_mean = np.zeros(20)
    full_mean[:4] = np.array([0.20, -0.10, 0.05, 0.30])
    full_covariance = design["initial_covariance"].copy()
    closure_mean = full_mean[:4].copy()
    closure_covariance = design["initial_task_covariance"].copy()

    task_only_mean = closure_mean.copy()
    task_only_covariance = closure_covariance.copy()
    rng = np.random.default_rng(20260902)

    closure_mean_errors: list[float] = []
    closure_covariance_errors: list[float] = []
    task_only_errors: list[float] = []
    closure_ranks: list[int] = []
    task_only_ranks: list[int] = []

    for _ in range(8):
        value = rng.normal(scale=0.15, size=observation.shape[0])

        full_mean, full_covariance = _kalman_update(
            full_mean,
            full_covariance,
            observation,
            full_noise,
            value,
        )

        closure_innovation = observation_map @ closure_covariance @ observation_map.T + full_noise
        closure_cross = closure_covariance @ observation_map.T
        closure_compression = compress_shared_factor_for_posterior(
            shared_factor.reshape(-1, 3, shared_factor.shape[1]),
            prior_query_covariance=closure_covariance,
            query_observation_cross_covariance=closure_cross,
            innovation_operator=DenseReference(closure_innovation),
            maximum_rank=4,
        )
        closure_factor = closure_compression.compressed_factor_m.reshape(
            observation.shape[0], -1
        )
        closure_mean, closure_covariance = _kalman_update(
            closure_mean,
            closure_covariance,
            observation_map,
            conditional_noise + closure_factor @ closure_factor.T,
            value,
        )

        task_only_innovation = (
            observation_map @ task_only_covariance @ observation_map.T + full_noise
        )
        task_only_cross = task_decoder @ task_only_covariance @ observation_map.T
        task_only_prior = task_decoder @ task_only_covariance @ task_decoder.T
        task_only_compression = compress_shared_factor_for_posterior(
            shared_factor.reshape(-1, 3, shared_factor.shape[1]),
            prior_query_covariance=task_only_prior,
            query_observation_cross_covariance=task_only_cross,
            innovation_operator=DenseReference(task_only_innovation),
            maximum_rank=3,
        )
        task_only_factor = task_only_compression.compressed_factor_m.reshape(
            observation.shape[0], -1
        )
        task_only_mean, task_only_covariance = _kalman_update(
            task_only_mean,
            task_only_covariance,
            observation_map,
            conditional_noise + task_only_factor @ task_only_factor.T,
            value,
        )

        full_task_mean = task_state @ full_mean
        full_task_covariance = task_state @ full_covariance @ task_state.T
        closure_mean_errors.append(float(np.max(np.abs(closure_mean - full_task_mean))))
        closure_covariance_errors.append(
            float(np.max(np.abs(closure_covariance - full_task_covariance)))
        )
        task_only_errors.append(
            float(np.max(np.abs(task_decoder @ task_only_mean - task_decoder @ full_task_mean)))
        )
        closure_ranks.append(closure_compression.retained_rank)
        task_only_ranks.append(task_only_compression.retained_rank)

        full_mean = transition @ full_mean
        full_covariance = transition @ full_covariance @ transition.T + process_noise
        closure_mean = task_transition @ closure_mean
        closure_covariance = (
            task_transition @ closure_covariance @ task_transition.T + task_process_noise
        )
        task_only_mean = task_transition @ task_only_mean
        task_only_covariance = (
            task_transition @ task_only_covariance @ task_transition.T + task_process_noise
        )

    assert np.linalg.matrix_rank(shared_factor) == 8
    assert closure_ranks == [4] * 8
    assert task_only_ranks == [3] * 8
    assert max(closure_mean_errors) < 1e-12
    assert max(closure_covariance_errors) < 1e-12
    assert task_only_errors[0] < 1e-12
    assert max(task_only_errors[1:]) > 1e-3


def test_recursive_task_closure_validates_inputs_and_is_read_only() -> None:
    closure = recursive_linear_task_closure(
        np.eye(3),
        task_matrix=np.array([[1.0, 0.0, 0.0]]),
        observation_matrix=np.array([[0.0, 1.0, 0.0]]),
    )
    assert closure.closure_dimension == 2
    assert not closure.basis.flags.writeable
    with pytest.raises(ValueError, match="square"):
        recursive_linear_task_closure(
            np.ones((2, 3)),
            task_matrix=np.ones((1, 3)),
            observation_matrix=np.ones((1, 3)),
        )
    with pytest.raises(TypeError, match="real scalar"):
        recursive_linear_task_closure(
            np.eye(3),
            task_matrix=np.ones((1, 3)),
            observation_matrix=np.ones((1, 3)),
            rank_relative_tolerance=True,
        )
