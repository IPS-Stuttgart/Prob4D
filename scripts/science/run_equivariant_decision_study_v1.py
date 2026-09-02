#!/usr/bin/env python3
"""Deterministic study of action identification under an unresolved SO(2) gauge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.equivariant_decision import (
    certify_gauge_coupled_actions,
    certify_independent_gauge_control,
    so2_covering_radius,
    squared_metric_independent_gauge_losses,
    squared_metric_shared_gauge_losses,
)

SCHEMA = "prob4d.equivariant-decision-study"
SCHEMA_VERSION = 1
SAMPLE_COUNTS = (4, 8, 16, 32, 64)
ACTION_NAMES = ("track-shared-frame", "orthogonal-shared-frame", "zero-fallback")
CLAIM_BOUNDARY = (
    "Controlled SO(2) mechanism evidence. The group action and exact state-action "
    "coupling are supplied. This does not discover a physical symmetry, validate "
    "a learned provider or actuator transform, establish target transport, prove "
    "closed-loop safety, or demonstrate state recovery."
)


def _rotation(angle: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ],
        dtype=np.float64,
    )


def _uniform_shifted_angles(sample_count: int) -> np.ndarray:
    return 2.0 * math.pi * (np.arange(sample_count) + 0.5) / sample_count


def _orbits(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.stack(
        [_rotation(float(angle)) @ np.array([1.0, 0.0]) for angle in angles]
    )
    action = np.stack(
        [
            np.stack(
                (
                    _rotation(float(angle)) @ np.array([1.0, 0.0]),
                    _rotation(float(angle)) @ np.array([0.0, 1.0]),
                    np.zeros(2),
                )
            )
            for angle in angles
        ]
    )
    return state[None, :, :], action[None, :, :, :]


def _dense_world_frame_losses(sample_count: int = 131_072) -> dict[str, float]:
    angles = 2.0 * math.pi * np.arange(sample_count) / sample_count
    state = np.stack(
        (np.cos(angles), np.sin(angles)),
        axis=1,
    )
    representative_action = np.array([1.0, 0.0])
    zero_action = np.zeros(2)
    representative_loss = np.sum(
        np.square(state - representative_action[None, :]),
        axis=1,
    )
    fallback_loss = np.sum(np.square(state - zero_action[None, :]), axis=1)
    return {
        "representative_completion_worst_loss": float(np.max(representative_loss)),
        "representative_completion_mean_loss": float(np.mean(representative_loss)),
        "zero_fallback_worst_loss": float(np.max(fallback_loss)),
        "zero_fallback_mean_loss": float(np.mean(fallback_loss)),
        "representative_completion_harm_fraction_vs_fallback": float(
            np.mean(representative_loss > fallback_loss)
        ),
    }


def _row(sample_count: int) -> dict[str, Any]:
    angles = _uniform_shifted_angles(sample_count)
    state, action = _orbits(angles)
    shared_losses = squared_metric_shared_gauge_losses(state, action)
    independent_losses = squared_metric_independent_gauge_losses(state, action)
    shared = certify_gauge_coupled_actions(
        shared_losses,
        [1.0],
        cover_radius=so2_covering_radius(angles),
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        fallback_action=2,
        regret_tolerance=0.0,
    )
    independent = certify_independent_gauge_control(
        independent_losses,
        [1.0],
        fallback_action=2,
        regret_tolerance=0.0,
    )
    state_coordinate_diameter = float(
        np.max(np.linalg.norm(state[0, :, None, :] - state[0, None, :, :], axis=2))
    )
    state_norm_width = float(np.ptp(np.linalg.norm(state[0], axis=1)))
    return {
        "sample_count": sample_count,
        "cover_radius_rad": so2_covering_radius(angles),
        "state_coordinate_orbit_diameter": state_coordinate_diameter,
        "state_norm_orbit_width": state_norm_width,
        "shared_gauge": {
            "action_names": list(ACTION_NAMES),
            "loss_by_action_first_sample": shared_losses[0, 0].tolist(),
            "maximum_loss_variation_by_action": np.ptp(
                shared_losses[0], axis=0
            ).tolist(),
            "posterior_decision_equivariance_defect_upper": (
                shared.posterior_decision_equivariance_defect_upper.tolist()
            ),
            "posterior_gauge_irrelevant": shared.posterior_gauge_irrelevant,
            "robustly_optimal": shared.robustly_optimal.tolist(),
            "worst_case_regret_upper_bound": (
                shared.worst_case_regret_upper_bound.tolist()
            ),
            "minimax_action": shared.minimax_action,
            "selected_action": shared.selected_action,
            "admitted": shared.admitted,
            "status": shared.status,
        },
        "independent_gauge_control": {
            "worst_case_regret": independent.worst_case_regret.tolist(),
            "minimax_action": independent.minimax_action,
            "selected_action": independent.selected_action,
            "admitted": independent.admitted,
        },
    }


def build_result() -> dict[str, Any]:
    rows = [_row(sample_count) for sample_count in SAMPLE_COUNTS]
    world = _dense_world_frame_losses()
    checks = {
        "state_is_not_identified": all(
            row["state_coordinate_orbit_diameter"] > 1.4 for row in rows
        ),
        "invariant_norm_is_identified": all(
            row["state_norm_orbit_width"] < 1e-12 for row in rows
        ),
        "shared_gauge_action_is_always_exact": all(
            row["shared_gauge"]["status"] == "certified-exactly-optimal"
            and row["shared_gauge"]["selected_action"] == 0
            and row["shared_gauge"]["posterior_gauge_irrelevant"]
            and row["shared_gauge"]["worst_case_regret_upper_bound"][0] < 1e-12
            for row in rows
        ),
        "independent_gauge_control_always_falls_back": all(
            not row["independent_gauge_control"]["admitted"]
            and row["independent_gauge_control"]["selected_action"] == 2
            and row["independent_gauge_control"]["worst_case_regret"][0] > 2.9
            for row in rows
        ),
        "point_completion_is_harmful_on_half_orbit": math.isclose(
            world["representative_completion_harm_fraction_vs_fallback"],
            0.5,
            rel_tol=0.0,
            abs_tol=2.0 / 131_072,
        ),
        "point_completion_worst_loss_exceeds_fallback": (
            world["representative_completion_worst_loss"]
            > world["zero_fallback_worst_loss"] + 2.9
        ),
    }
    decision = (
        "controlled-shared-gauge-passed"
        if all(checks.values())
        else "controlled-shared-gauge-failed"
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
        "group": "SO(2)",
        "state": "x(theta)=R(theta)[1,0]",
        "action_templates": list(ACTION_NAMES),
        "loss": "squared Euclidean state-action distance",
        "sample_counts": list(SAMPLE_COUNTS),
        "rows": rows,
        "world_frame_point_completion": world,
        "checks": checks,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["result_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    result = build_result()
    if result["decision"] != "controlled-shared-gauge-passed":
        raise SystemExit(json.dumps(result["checks"], indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
