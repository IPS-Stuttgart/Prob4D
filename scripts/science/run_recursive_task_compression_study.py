"""Controlled recursive task-sufficient compression mechanism study.

This script opens no dataset and makes no robotics-performance claim. It tests
whether one-step query-sufficient correlated-noise compression remains valid
under recursion, and whether enlarging the preserved query to the exact linear
task-state closure restores recursive parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

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
            self.covariance,
            raw.reshape(self.dimension, -1),
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


def _design() -> dict[str, np.ndarray]:
    state_dimension = 20
    closure_dimension = 4
    task_dimension = 3

    task_state = np.zeros((closure_dimension, state_dimension))
    task_state[:, :closure_dimension] = np.eye(closure_dimension)
    task_decoder = np.eye(closure_dimension)[:task_dimension]
    task = task_decoder @ task_state

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
    orthogonal_noise = right[closure_dimension:].T[:, :4]
    relevant_noise = observation_map @ np.diag([0.18, 0.16, 0.14, 0.22])
    irrelevant_noise = orthogonal_noise @ np.diag([0.11, 0.10, 0.09, 0.08])
    shared_factor = np.column_stack((relevant_noise, irrelevant_noise))

    conditional_noise = 0.04**2 * np.eye(observation.shape[0])
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
        "task_decoder": task_decoder,
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


def run(protocol: dict[str, object]) -> dict[str, object]:
    design = _design()
    task_state = design["task_state"]
    task_decoder = design["task_decoder"]
    observation = design["observation"]
    observation_map = design["observation_map"]
    transition = design["transition"]
    task_transition = design["task_transition"]
    shared_factor = design["shared_factor"]
    conditional_noise = design["conditional_noise"]
    full_noise = conditional_noise + shared_factor @ shared_factor.T
    process_noise = design["process_noise"]
    task_process_noise = task_state @ process_noise @ task_state.T

    closure = recursive_linear_task_closure(
        transition,
        task_matrix=design["task"],
        observation_matrix=observation,
    )
    violated_transition = transition.copy()
    violated_transition[0, int(protocol["violation_transition_state_index"])] = float(
        protocol["violation_transition_coefficient"]
    )
    violated_closure = recursive_linear_task_closure(
        violated_transition,
        task_matrix=design["task"],
        observation_matrix=observation,
    )

    full_mean = np.zeros(int(protocol["state_dimension"]))
    full_mean[:4] = np.array([0.20, -0.10, 0.05, 0.30])
    full_covariance = design["initial_covariance"].copy()
    closure_mean = full_mean[:4].copy()
    closure_covariance = design["initial_task_covariance"].copy()
    task_only_mean = closure_mean.copy()
    task_only_covariance = closure_covariance.copy()
    rng = np.random.default_rng(int(protocol["seed"]))

    steps: list[dict[str, object]] = []
    for step in range(int(protocol["update_count"])):
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
            maximum_rank=int(protocol["expected_recursive_closure_dimension"]),
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
            maximum_rank=int(protocol["current_task_dimension"]),
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
        steps.append(
            {
                "step": step,
                "closure_retained_rank": closure_compression.retained_rank,
                "current_task_retained_rank": task_only_compression.retained_rank,
                "closure_mean_max_abs_error": float(
                    np.max(np.abs(closure_mean - full_task_mean))
                ),
                "closure_covariance_max_abs_error": float(
                    np.max(np.abs(closure_covariance - full_task_covariance))
                ),
                "current_task_mean_max_abs_error": float(
                    np.max(
                        np.abs(
                            task_decoder @ task_only_mean
                            - task_decoder @ full_task_mean
                        )
                    )
                ),
            }
        )

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

    report = {
        "evidence_class": protocol["evidence_class"],
        "state_dimension": int(protocol["state_dimension"]),
        "current_task_dimension": int(protocol["current_task_dimension"]),
        "recursive_closure": closure.summary(),
        "violated_recursive_closure": violated_closure.summary(),
        "original_shared_factor_rank": int(np.linalg.matrix_rank(shared_factor)),
        "steps": steps,
        "claim_boundary": protocol["claim_boundary"],
    }

    closure_mean_error = max(float(row["closure_mean_max_abs_error"]) for row in steps)
    closure_covariance_error = max(
        float(row["closure_covariance_max_abs_error"]) for row in steps
    )
    current_errors = [float(row["current_task_mean_max_abs_error"]) for row in steps]
    closure_ranks = [int(row["closure_retained_rank"]) for row in steps]
    task_ranks = [int(row["current_task_retained_rank"]) for row in steps]

    if closure.closure_dimension != int(protocol["expected_recursive_closure_dimension"]):
        raise RuntimeError("recursive closure dimension changed")
    if violated_closure.closure_dimension != int(protocol["expected_violated_closure_dimension"]):
        raise RuntimeError("closure violation did not expand the task state")
    if int(np.linalg.matrix_rank(shared_factor)) != int(protocol["original_shared_factor_rank"]):
        raise RuntimeError("shared-factor rank changed")
    if closure_ranks != [int(protocol["expected_closure_factor_rank"])] * len(steps):
        raise RuntimeError("closure-aware retained ranks changed")
    if task_ranks != [int(protocol["expected_current_task_factor_rank"])] * len(steps):
        raise RuntimeError("current-task retained ranks changed")
    if closure_mean_error > float(protocol["closure_mean_parity_absolute_tolerance"]):
        raise RuntimeError("closure-aware recursive mean parity failed")
    if closure_covariance_error > float(
        protocol["closure_covariance_parity_absolute_tolerance"]
    ):
        raise RuntimeError("closure-aware recursive covariance parity failed")
    if current_errors[0] > float(
        protocol["current_task_first_update_parity_absolute_tolerance"]
    ):
        raise RuntimeError("current-task one-step parity failed")
    if max(current_errors[1:]) < float(protocol["minimum_recursive_failure_absolute_error"]):
        raise RuntimeError("current-query-only control did not expose recursive failure")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/recursive-task-compression-controlled-v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["schema"] != "prob4d.recursive-task-compression-controlled.v1":
        raise ValueError("unsupported protocol")
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists; never overwrite a run")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "protocol.json").write_bytes(protocol_bytes)

    report = run(protocol)
    result_bytes = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    (args.output_dir / "result.json").write_bytes(result_bytes)
    manifest = {
        "source_revision": args.source_revision,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "real_provider_evidence": False,
        "real_robot_evidence": False,
        "sealed_data_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
