"""Controlled finite-orbit failure and decision-value study; no provider data.

Run with ``python -m prob4d.axial_query_study --output result.json``.
The study deliberately constructs its model assumptions.  It is not a
calibration experiment, a fresh cohort, or a comparison of real providers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .axial_query_certificate import (
    AxialRotationOrbit,
    HarmonicQuery,
    certify_shared_orbit_advantage,
)

FAMILIES = (
    "stationary-but-ambiguous",
    "positive-whole-orbit-margin",
    "shared-gauge-cancellation",
    "bounded-omitted-discrepancy",
)
ARMS = (
    "local-query-gate-then-plugin",
    "reject-all-deficient",
    "independent-query-intervals",
    "shared-orbit-certificate",
    "shared-orbit-plus-one-bounded-anchor",
)


def _summary() -> dict[str, int | float]:
    return {
        "cases": 0,
        "accepted": 0,
        "sampled_harmful_accepts": 0,
        "admitted_with_possible_harm": 0,
        "exact_fallback_identity_failures": 0,
        "deployed_advantage_sum": 0.0,
        "maximum_possible_harm": 0.0,
    }


def analytic_local_control(orbit: AxialRotationOrbit, radial: np.ndarray) -> dict[str, float]:
    """Reference algebra for the existing scalar local query gate.

    The complete prior is I_7 and visual information is ten times the rank-six
    projector excluding rotation about the line.  The integration test checks
    this control against the repository's actual query_observability API.
    """
    normal = radial / np.linalg.norm(radial)
    jacobian = np.concatenate(([float(normal @ radial)], np.cross(radial, normal), normal))
    nullspace = np.concatenate(([0.0], orbit.axis, np.zeros(3)))
    projector = np.eye(7) - np.outer(nullspace, nullspace)
    posterior = np.linalg.solve(np.eye(7) + 10.0 * projector, np.eye(7))
    energy = float(jacobian @ jacobian)
    direct = float(jacobian @ projector @ jacobian) / energy
    ratio = float(jacobian @ posterior @ jacobian) / energy
    return {
        "direct_observability_fraction": direct,
        "metric_variance_reduction_fraction": 1.0 - ratio,
        "worst_supported_variance_ratio": ratio,
    }


def run_axial_query_study(*, seed: int = 73029, cases_per_family: int = 512) -> dict[str, Any]:
    if isinstance(cases_per_family, bool) or not isinstance(cases_per_family, int):
        raise TypeError("cases_per_family must be an integer")
    if cases_per_family < 1:
        raise ValueError("cases_per_family must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    results = {family: {arm: _summary() for arm in ARMS} for family in FAMILIES}
    max_support_drift = 0.0
    max_local_nullspace_fraction = 0.0
    anchor_truth_exclusion_count = 0
    for family in FAMILIES:
        for case_index in range(cases_per_family):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            origin = rng.normal(scale=0.2, size=3)
            direction = rng.normal(size=3)
            direction -= axis * float(direction @ axis)
            direction /= np.linalg.norm(direction)
            radius = float(rng.uniform(0.01, 0.10))
            radial = radius * direction
            orbit = AxialRotationOrbit(origin, axis, f"{family}/{case_index}")
            support = origin + np.array([-0.4, -0.2, 0.0, 0.2, 0.4])[:, None] * axis
            max_support_drift = max(max_support_drift, orbit.maximum_support_displacement(support))
            q = orbit.affine_query(
                (origin + radial)[None, :],
                direction[None, :],
                offset=-float(direction @ origin),
            )
            base = 4.0 * radius
            error = 0.0
            if family == "shared-gauge-cancellation":
                fallback = HarmonicQuery(
                    base + 2.0 * q.constant, 2.0 * q.cosine, 2.0 * q.sine, orbit.key
                )
                candidate = HarmonicQuery(
                    base - 0.25 * radius + 2.0 * q.constant,
                    2.0 * q.cosine,
                    2.0 * q.sine,
                    orbit.key,
                )
            else:
                gap = (0.25 if family == "stationary-but-ambiguous" else 1.25) * radius
                fallback = HarmonicQuery(base, 0.0, 0.0, orbit.key)
                candidate = HarmonicQuery(base - gap - q.constant, -q.cosine, -q.sine, orbit.key)
                if family == "bounded-omitted-discrepancy":
                    error = 0.5 * radius
            difference = fallback.minus(candidate)
            true_angle = float(rng.uniform(-math.pi, math.pi))
            # Deliberately use the worst signed omitted effect.  This belongs to
            # the stated uniform envelope, not to an estimated noise model.
            true_advantage = difference.evaluate(true_angle) - error
            full_bounds = difference.bounds()
            possible_harm = max(0.0, error - full_bounds.lower)
            local = analytic_local_control(orbit, radial)
            max_local_nullspace_fraction = max(
                max_local_nullspace_fraction, 1.0 - local["direct_observability_fraction"]
            )
            local_admitted = (
                local["direct_observability_fraction"] >= 0.8
                and local["metric_variance_reduction_fraction"] >= 0.8
                and local["worst_supported_variance_ratio"] <= 0.5
                and difference.evaluate(0.0) > 1e-12
            )
            certificate = certify_shared_orbit_advantage(
                fallback_loss=fallback,
                candidate_loss=candidate,
                scope_admitted=True,
                advantage_error_bound=error,
            )
            independent_lower = fallback.bounds().lower - candidate.bounds().upper - error
            # Separate information-budget arm: one synthetic bounded-error
            # metric anchor, with known radius and no fitted calibration.
            anchor_reference = origin + 0.15 * direction
            anchor_observation = orbit.transform(anchor_reference[None, :], true_angle)[0]
            anchor_arc = orbit.bounded_anchor_arc(
                anchor_reference, anchor_observation, error_radius=0.02
            )
            if anchor_arc is None or not anchor_arc.contains(true_angle, atol=1e-12):
                anchor_truth_exclusion_count += 1
            anchored = certify_shared_orbit_advantage(
                fallback_loss=fallback,
                candidate_loss=candidate,
                scope_admitted=True,
                arc=anchor_arc,
                advantage_error_bound=error,
            )
            decisions = {
                ARMS[0]: local_admitted,
                ARMS[1]: False,
                ARMS[2]: independent_lower > 1e-12,
                ARMS[3]: certificate.admitted,
                ARMS[4]: anchored.admitted,
            }
            fallback_object = object()
            candidate_object = object()
            for arm, admitted in decisions.items():
                row = results[family][arm]
                row["cases"] += 1
                row["accepted"] += int(admitted)
                row["sampled_harmful_accepts"] += int(admitted and true_advantage < -1e-12)
                bound_harm = possible_harm
                if arm == ARMS[4]:
                    bound_harm = max(0.0, -(anchored.lower_advantage or 0.0))
                row["admitted_with_possible_harm"] += int(admitted and bound_harm > 1e-12)
                if admitted:
                    row["deployed_advantage_sum"] += true_advantage
                    row["maximum_possible_harm"] = max(row["maximum_possible_harm"], bound_harm)
                selected = candidate_object if admitted else fallback_object
                row["exact_fallback_identity_failures"] += int(
                    not admitted and selected is not fallback_object
                )
    totals = {arm: _summary() for arm in ARMS}
    for family_rows in results.values():
        for arm, row in family_rows.items():
            for key, value in row.items():
                if key == "maximum_possible_harm":
                    totals[arm][key] = max(totals[arm][key], value)
                else:
                    totals[arm][key] += value
    for group in (totals, *results.values()):
        for row in group.values():
            row["acceptance_fraction"] = row["accepted"] / row["cases"]
            row["mean_deployed_advantage"] = row.pop("deployed_advantage_sum") / row["cases"]
    return {
        "schema": "prob4d.finite-orbit-query-mechanism-study",
        "schema_version": 1,
        "seed": seed,
        "cases_per_family": cases_per_family,
        "total_cases": len(FAMILIES) * cases_per_family,
        "evidence_class": "constructed-controlled-mechanism-not-real-provider",
        "uncertainty_scope": "one-shared-axial-orbit-plus-declared-uniform-advantage-error",
        "local_gate_thresholds": {"direct_min": 0.8, "reduction_min": 0.8, "worst_ratio_max": 0.5},
        "additional_anchor_arm": {
            "anchor_radius_m": 0.15,
            "error_bound_m": 0.02,
            "actual_sensor_error_m": 0.0,
            "information_budget": "one-extra-synthetic-metric-anchor",
        },
        "maximum_support_orbit_drift_m": max_support_drift,
        "maximum_local_nullspace_fraction": max_local_nullspace_fraction,
        "anchor_truth_exclusion_count": anchor_truth_exclusion_count,
        "totals": totals,
        "families": results,
        "claim_boundary": [
            "The constructed orbit and error envelope contain the simulated truth by design.",
            (
                "Zero harmful accepts is a conditional algebraic mechanism result, "
                "not measured real-world safety or calibration."
            ),
            (
                "The local arm is an analytic reference control checked against the "
                "existing API by a separate integration test."
            ),
            (
                "The anchor arm has one additional observation "
                "and is not an equal-information comparison."
            ),
            (
                "The object-identity check is a standalone routing control, not execution "
                "of BayesianPhysTwin's complete-belief router."
            ),
            (
                "No real provider, protected source, target, physical simulator, "
                "or Causal4D outcome was accessed."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=73029)
    parser.add_argument("--cases-per-family", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_axial_query_study(seed=args.seed, cases_per_family=args.cases_per_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
