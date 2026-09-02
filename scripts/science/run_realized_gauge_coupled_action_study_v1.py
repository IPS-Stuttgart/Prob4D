#!/usr/bin/env python3
"""Controlled study of gauge-coupled actions under realization uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.realized_equivariant_decision import (
    certify_realized_gauge_coupled_actions,
)

SCHEMA = "prob4d.realized-gauge-coupled-action-study"
SCHEMA_VERSION = 1
REALIZATION_RADII = (0.0, 0.2, 0.8, 1.0, 1.2)
CLAIM_BOUNDARY = (
    "Controlled deterministic mechanism evidence. The group action, loss "
    "Lipschitz constants and action-realization radii are supplied assumptions. "
    "This study does not estimate actuator error, validate Causal4D intervention "
    "transport, establish a target-domain probability statement, or certify "
    "deployment safety."
)


def _loss_samples(sample_count: int = 16) -> np.ndarray:
    loss = np.empty((1, sample_count, 3), dtype=np.float64)
    loss[0, :, :] = np.array([0.0, 2.0, 1.0])
    return loss


def _certificate(radius: float, tolerance: float) -> dict[str, Any]:
    result = certify_realized_gauge_coupled_actions(
        _loss_samples(),
        [1.0],
        cover_radius=math.pi / 16.0,
        pairwise_lipschitz=np.zeros((1, 3, 3)),
        realization_radius=[radius, 0.0, 0.0],
        action_loss_lipschitz=[1.0, 1.0, 1.0],
        fallback_action=2,
        regret_tolerance=tolerance,
    )
    return {
        "realization_radius_action_0": radius,
        "regret_tolerance": tolerance,
        "ideal_regret_upper": (
            result.ideal_certificate.worst_case_regret_upper_bound.tolist()
        ),
        "posterior_pairwise_realization_margin": (
            result.posterior_pairwise_realization_margin.tolist()
        ),
        "pairwise_realized_upper": result.pairwise_realized_upper_bound.tolist(),
        "realized_regret_upper": (
            result.worst_case_realized_regret_upper_bound.tolist()
        ),
        "robustly_optimal": result.robustly_optimal_under_realization.tolist(),
        "epsilon_admissible": (
            result.epsilon_admissible_under_realization.tolist()
        ),
        "minimax_action": result.minimax_action,
        "selected_action": result.selected_action,
        "fallback_action": result.fallback_action,
        "admitted": result.admitted,
        "status": result.status,
    }


def build_result() -> dict[str, Any]:
    zero_tolerance = [_certificate(radius, 0.0) for radius in REALIZATION_RADII]
    bounded = _certificate(1.2, 0.25)
    action_zero_regrets = [row["realized_regret_upper"][0] for row in zero_tolerance]
    checks = {
        "ideal_action_has_zero_regret": all(
            row["ideal_regret_upper"][0] < 1e-12 for row in zero_tolerance
        ),
        "realization_regret_is_monotone": all(
            later >= earlier
            for earlier, later in zip(
                action_zero_regrets,
                action_zero_regrets[1:],
                strict=True,
            )
        ),
        "small_radii_preserve_exact_optimality": all(
            row["robustly_optimal"][0]
            and row["selected_action"] == 0
            and row["status"] == "certified-exactly-optimal"
            for row in zero_tolerance[:-1]
        ),
        "large_radius_forces_exact_fallback": bool(
            zero_tolerance[-1]["realized_regret_upper"][0] > 0.19
            and not zero_tolerance[-1]["admitted"]
            and zero_tolerance[-1]["selected_action"] == 2
            and zero_tolerance[-1]["status"] == "fallback-regret-unresolved"
        ),
        "registered_tolerance_recovers_bounded_action": bool(
            bounded["realized_regret_upper"][0] <= 0.25
            and bounded["admitted"]
            and bounded["selected_action"] == 0
            and bounded["status"] == "certified-bounded-regret"
        ),
        "pairwise_margin_matches_K_epsilon_sum": all(
            math.isclose(
                row["posterior_pairwise_realization_margin"][0][2],
                row["realization_radius_action_0"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for row in zero_tolerance
        ),
    }
    decision = (
        "controlled-realized-gauge-action-passed"
        if all(checks.values())
        else "controlled-realized-gauge-action-failed"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
        "realization_radii": list(REALIZATION_RADII),
        "action_loss_lipschitz": [1.0, 1.0, 1.0],
        "zero_tolerance_sweep": zero_tolerance,
        "bounded_regret_case": bounded,
        "checks": checks,
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["result_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    result = build_result()
    if result["decision"] != "controlled-realized-gauge-action-passed":
        raise SystemExit(json.dumps(result["checks"], indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
