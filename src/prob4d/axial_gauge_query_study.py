"""Deterministic development controls for finite axial-gauge query ambiguity.

No provider, dataset, posterior, held-out target, or physical execution is used.
The local reference matches the rank-six analytic factor already used by the
query-observability study; a separate integration test checks that parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .axial_gauge_query import AxialGaugeOrbit, AxialQueryFamily, affine_axial_queries


def run_axial_gauge_query_study() -> dict[str, Any]:
    orbit = AxialGaugeOrbit(pivot=np.zeros(3), axis=np.array([1.0, 0.0, 0.0]))
    support = np.column_stack((np.linspace(-1.0, 1.0, 48), np.zeros((48, 2))))
    point = np.array([[0.0, 0.1, 0.0]])
    radial = affine_axial_queries(
        point,
        np.array([[[0.0, 1.0, 0.0]]]),
        point_group_ids=("shared",),
        orbits={"shared": orbit},
    )
    axial = affine_axial_queries(
        np.array([[0.04, 0.1, 0.0]]),
        np.array([[[1.0, 0.0, 0.0]]]),
        point_group_ids=("shared",),
        orbits={"shared": orbit},
    )
    points = np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]])
    weights = np.array([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]])
    common = affine_axial_queries(
        points,
        weights,
        point_group_ids=("shared", "shared"),
        orbits={"shared": orbit},
    )
    separate = affine_axial_queries(
        points,
        weights,
        point_group_ids=("first", "second"),
        orbits={"first": orbit, "second": orbit},
    )
    losses = affine_axial_queries(
        point,
        np.array([[[0.0, -1.0, 0.0]], [[0.0, 1.0, 0.0]]]),
        offsets=np.array([0.1, 0.1]),
        point_group_ids=("shared",),
        orbits={"shared": orbit},
    )
    common_losses = affine_axial_queries(
        point,
        np.array([[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]]),
        offsets=np.array([0.05, 0.10]),
        point_group_ids=("shared",),
        orbits={"shared": orbit},
    )
    # Independent closed-form local reference: J_y=[.1,0,0,0,0,1,0].
    # Only rotation-x has zero precision. The complete prior is I_7.
    jacobian = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    precision = np.diag([10.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    posterior = np.linalg.solve(np.eye(7) + precision, np.eye(7))
    prior_variance = float((jacobian @ jacobian.T)[0, 0])
    posterior_variance = float((jacobian @ posterior @ jacobian.T)[0, 0])
    variance_reduction = 1.0 - posterior_variance / prior_variance
    worst_ratio = posterior_variance / prior_variance
    local_admitted = variance_reduction >= 0.8 and worst_ratio <= 0.5
    competitor, angles = losses.regret_witness(0)
    at_witness = losses.evaluate(angles)

    def interval(family: AxialQueryFamily) -> list[float]:
        lower, upper = family.bounds()
        return [float(lower[0]), float(upper[0])]

    return {
        "schema_name": "prob4d.axial-gauge-query-development-control",
        "schema_version": 1,
        "classification": "deterministic analytic development evidence",
        "claim_boundary": (
            "Exact only over the declared axial-orbit domain and affine queries; "
            "no provider competence, posterior calibration, physical validation, "
            "BayesianPhysTwin benefit, Causal4D benefit, or deployment safety."
        ),
        "geometry": {
            "axis": [1.0, 0.0, 0.0],
            "pivot": [0.0, 0.0, 0.0],
            "support_points": 48,
            "probe_radius_m": 0.1,
            "maximum_support_motion_m": orbit.maximum_support_motion(support),
        },
        "stationary_derivative_counterexample": {
            "query": "y-coordinate of a 100 mm off-axis point",
            "query_at_reference_m": float(radial.evaluate(np.zeros(1))[0]),
            "first_twist_derivative_m_per_rad": float(radial.sine[0, 0]),
            "second_twist_derivative_m_per_rad2": -float(radial.cosine[0, 0]),
            "finite_orbit_interval_m": interval(radial),
            "local_linear_reference": {
                "factor_rank": 6,
                "direct_observability_fraction": 1.0,
                "metric_variance_reduction_fraction": variance_reduction,
                "worst_supported_variance_ratio": worst_ratio,
                "gate_thresholds": [0.8, 0.8, 0.5],
                "local_gate_admits": local_admitted,
            },
        },
        "positive_control": {
            "query": "axial coordinate of an off-axis point",
            "finite_orbit_interval_m": interval(axial),
        },
        "lineage_control": {
            "query": "difference of equal-radius y-coordinates",
            "shared_angle_interval_m": interval(common),
            "separately_variable_angles_interval_m": interval(separate),
            "boundary": "Group equality is declared, never inferred from coincident axes.",
        },
        "action_control": {
            "losses": ["0.1 - y", "0.1 + y"],
            "nominal_action": 0,
            "nominal_losses_m": losses.evaluate(np.zeros(1)).tolist(),
            "worst_case_regrets_m": losses.worst_case_regrets().tolist(),
            "illustrative_regret_budget_m": 0.05,
            "within_budget": losses.within_regret_budget(0, maximum_regret=0.05),
            "witness_competitor": competitor,
            "witness_angle_rad": angles.tolist(),
            "witness_losses_m": at_witness.tolist(),
        },
        "shared_action_nuisance_control": {
            "losses": ["0.05 + y", "0.10 + y"],
            "marginal_intervals_m": np.column_stack(common_losses.bounds()).tolist(),
            "worst_case_regrets_m": common_losses.worst_case_regrets().tolist(),
            "action_zero_within_zero_budget": common_losses.within_regret_budget(
                0, maximum_regret=0.0
            ),
        },
        "information_boundary": {
            "provider_forward_calls": 0,
            "dataset_records_accessed": 0,
            "protected_targets_opened": 0,
            "physical_executions": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(
        run_axial_gauge_query_study(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite a different retained result.
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"refusing to replace different evidence: {args.output}")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)


if __name__ == "__main__":
    main()
