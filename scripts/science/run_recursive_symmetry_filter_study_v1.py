#!/usr/bin/env python3
"""Controlled recursive study of quotient learning without invented gauge information."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prob4d.equivariant_decision import certify_gauge_coupled_actions
from prob4d.recursive_symmetry_filter import (
    predict_cyclic_equivariant,
    uniform_group_belief,
    update_invariant_evidence,
    update_symmetry_breaking_evidence,
)

SCHEMA = "prob4d.recursive-symmetry-filter-study"
SCHEMA_VERSION = 1
GROUP_SIZES = (8, 16, 32, 64)
INVARIANT_UPDATE_COUNT = 6
CLAIM_BOUNDARY = (
    "Deterministic finite-cyclic mechanism evidence. The quotient model, group "
    "action, equivariant transition, invariant likelihood and action loss are "
    "supplied. This does not discover a physical symmetry, validate a learned "
    "provider, establish continuous-group quadrature validity, prove target "
    "transport, or certify deployment safety."
)


def _increment_kernel(class_count: int, group_size: int) -> np.ndarray:
    kernel = np.zeros((class_count, class_count, group_size), dtype=np.float64)
    kernel[:, :, 1 % group_size] = 0.75
    kernel[:, :, -1 % group_size] += 0.25
    return kernel


def _action_loss(group_size: int) -> np.ndarray:
    loss = np.empty((2, group_size, 3), dtype=np.float64)
    loss[0, :, :] = np.array([0.0, 2.0, 1.0])
    loss[1, :, :] = np.array([2.0, 0.0, 1.0])
    return loss


def _run_group_size(group_size: int) -> dict[str, Any]:
    belief = uniform_group_belief([0.5, 0.5], group_size)
    initial_conditional = np.array(
        belief.conditional_group_probability,
        copy=True,
    )
    initial_quotient_entropy = belief.quotient_entropy
    expected_group_entropy = math.log(group_size)
    trace: list[dict[str, Any]] = []
    cumulative_group_information = 0.0
    transition = np.eye(2)
    increment = _increment_kernel(2, group_size)

    for step in range(1, INVARIANT_UPDATE_COUNT + 1):
        belief = predict_cyclic_equivariant(
            belief,
            transition,
            increment,
        )
        prior_conditional = np.array(
            belief.conditional_group_probability,
            copy=True,
        )
        audit = update_invariant_evidence(belief, [0.8, 0.2])
        cumulative_group_information += audit.conditional_group_information
        belief = audit.posterior
        trace.append(
            {
                "step": step,
                "quotient_mass": belief.quotient_mass.tolist(),
                "quotient_entropy": belief.quotient_entropy,
                "expected_conditional_group_entropy": (
                    belief.expected_conditional_group_entropy
                ),
                "joint_information": audit.joint_information,
                "quotient_information": audit.quotient_information,
                "conditional_group_information": (
                    audit.conditional_group_information
                ),
                "chain_rule_error": audit.chain_rule_error,
                "conditional_group_retained_exactly": bool(
                    np.array_equal(
                        prior_conditional,
                        belief.conditional_group_probability,
                    )
                ),
            }
        )

    loss = _action_loss(group_size)
    action_certificate = certify_gauge_coupled_actions(
        loss,
        belief.quotient_mass,
        cover_radius=math.pi / group_size,
        pairwise_lipschitz=np.zeros((2, 3, 3)),
        fallback_action=2,
        regret_tolerance=0.0,
    )

    angles = 2.0 * math.pi * np.arange(group_size) / group_size
    breaking_likelihood = np.exp(4.0 * np.cos(angles))[None, :]
    breaking_likelihood = np.repeat(breaking_likelihood, 2, axis=0)
    breaking_audit = update_symmetry_breaking_evidence(
        belief,
        breaking_likelihood,
    )

    quotient_entropies = [initial_quotient_entropy] + [
        row["quotient_entropy"] for row in trace
    ]
    checks = {
        "every_invariant_update_has_zero_group_information": all(
            row["conditional_group_information"] < 1e-14 for row in trace
        ),
        "every_invariant_update_retains_group_law_exactly": all(
            row["conditional_group_retained_exactly"] for row in trace
        ),
        "quotient_entropy_strictly_decreases": all(
            later < earlier
            for earlier, later in zip(
                quotient_entropies,
                quotient_entropies[1:],
                strict=True,
            )
        ),
        "group_entropy_stays_at_log_cardinality": all(
            abs(
                row["expected_conditional_group_entropy"]
                - expected_group_entropy
            )
            < 1e-12
            for row in trace
        ),
        "complete_initial_group_law_survives": bool(
            np.array_equal(
                belief.conditional_group_probability,
                initial_conditional,
            )
        ),
        "quotient_becomes_decisive": bool(belief.quotient_mass[0] > 0.9997),
        "equivariant_action_becomes_exactly_identified": bool(
            action_certificate.robustly_optimal[0]
            and action_certificate.selected_action == 0
            and action_certificate.posterior_gauge_irrelevant
            and action_certificate.worst_case_regret_upper_bound[0] < 1e-12
        ),
        "symmetry_breaking_evidence_adds_group_information": bool(
            breaking_audit.conditional_group_information > 0.4
            and breaking_audit.posterior.expected_conditional_group_entropy
            < belief.expected_conditional_group_entropy
        ),
        "kl_chain_rule_holds": all(
            row["chain_rule_error"] < 1e-12 for row in trace
        )
        and breaking_audit.chain_rule_error < 1e-12,
    }

    return {
        "group_size": group_size,
        "initial_quotient_mass": [0.5, 0.5],
        "initial_quotient_entropy": initial_quotient_entropy,
        "initial_expected_conditional_group_entropy": expected_group_entropy,
        "invariant_update_count": INVARIANT_UPDATE_COUNT,
        "trace": trace,
        "final_quotient_mass": belief.quotient_mass.tolist(),
        "final_quotient_entropy": belief.quotient_entropy,
        "final_expected_conditional_group_entropy": (
            belief.expected_conditional_group_entropy
        ),
        "cumulative_invariant_group_information": cumulative_group_information,
        "point_completion_specificity_nats": math.log(group_size),
        "action_certificate": {
            "selected_action": action_certificate.selected_action,
            "fallback_action": action_certificate.fallback_action,
            "robustly_optimal": action_certificate.robustly_optimal.tolist(),
            "worst_case_regret_upper_bound": (
                action_certificate.worst_case_regret_upper_bound.tolist()
            ),
            "posterior_gauge_irrelevant": (
                action_certificate.posterior_gauge_irrelevant
            ),
            "status": action_certificate.status,
        },
        "symmetry_breaking_update": {
            "conditional_group_information": (
                breaking_audit.conditional_group_information
            ),
            "group_entropy_before": belief.expected_conditional_group_entropy,
            "group_entropy_after": (
                breaking_audit.posterior.expected_conditional_group_entropy
            ),
            "maximum_conditional_group_change": (
                breaking_audit.maximum_conditional_group_change
            ),
            "chain_rule_error": breaking_audit.chain_rule_error,
        },
        "checks": checks,
    }


def build_result() -> dict[str, Any]:
    rows = [_run_group_size(group_size) for group_size in GROUP_SIZES]
    checks = {
        "all_group_sizes_pass": all(
            all(row["checks"].values()) for row in rows
        ),
        "zero_cumulative_invariant_gauge_information": all(
            row["cumulative_invariant_group_information"] < 1e-13
            for row in rows
        ),
        "point_completion_specificity_grows_with_resolution": all(
            later["point_completion_specificity_nats"]
            > earlier["point_completion_specificity_nats"]
            for earlier, later in zip(rows, rows[1:], strict=True)
        ),
        "same_quotient_posterior_across_group_resolutions": all(
            np.allclose(
                rows[0]["final_quotient_mass"],
                row["final_quotient_mass"],
                rtol=0.0,
                atol=1e-15,
            )
            for row in rows[1:]
        ),
    }
    decision = (
        "controlled-recursive-symmetry-passed"
        if all(checks.values())
        else "controlled-recursive-symmetry-failed"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "claim_boundary": CLAIM_BOUNDARY,
        "group_sizes": list(GROUP_SIZES),
        "invariant_update_count": INVARIANT_UPDATE_COUNT,
        "rows": rows,
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
    if result["decision"] != "controlled-recursive-symmetry-passed":
        raise SystemExit(json.dumps(result["checks"], indent=2, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
