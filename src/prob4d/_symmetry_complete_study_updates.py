"""Bayesian-update panels for the symmetry-complete controlled study."""

from __future__ import annotations

import math

import numpy as np

from ._symmetry_complete_study_common import PROTOCOL, _random_probability
from .symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    SymmetryCompleteBeliefV1,
    audit_point_completion,
    update_symmetry_complete_belief,
)


def _invariant_update_study(
    rng: np.random.Generator,
    *,
    cases: int,
) -> dict[str, float | int]:
    quotient_count = int(PROTOCOL["quotient_count"])
    node_count = int(PROTOCOL["circle_node_count"])
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        node_count,
        group_id="controlled-s1",
    )
    maximum_conditional_change = 0.0
    maximum_gauge_information = 0.0
    maximum_chain_rule_error = 0.0
    minimum_quotient_information = math.inf
    for case_index in range(cases):
        quotient = _random_probability(rng, quotient_count)
        conditionals = np.stack(
            [_random_probability(rng, node_count) for _ in range(quotient_count)]
        )
        prior = SymmetryCompleteBeliefV1(
            quotient,
            conditionals,
            quadrature,
            f"prior-{case_index}",
        )
        quotient_likelihood = np.exp(rng.normal(size=quotient_count))
        likelihood = np.repeat(quotient_likelihood[:, None], node_count, axis=1)
        update = update_symmetry_complete_belief(
            prior,
            likelihood,
            evidence_semantics="orbit-invariant",
            posterior_belief_id=f"posterior-{case_index}",
            whole_group_invariance_certified=True,
        )
        maximum_conditional_change = max(
            maximum_conditional_change,
            update.information.maximum_conditional_l1_change,
        )
        maximum_gauge_information = max(
            maximum_gauge_information,
            update.information.gauge_information_nats,
        )
        maximum_chain_rule_error = max(
            maximum_chain_rule_error,
            abs(
                update.information.total_information_nats
                - update.information.quotient_information_nats
                - update.information.gauge_information_nats
            ),
        )
        minimum_quotient_information = min(
            minimum_quotient_information,
            update.information.quotient_information_nats,
        )
    return {
        "case_count": cases,
        "maximum_conditional_l1_change": maximum_conditional_change,
        "maximum_gauge_information_nats": maximum_gauge_information,
        "maximum_kl_chain_rule_error_nats": maximum_chain_rule_error,
        "minimum_quotient_information_nats": minimum_quotient_information,
    }


def _symmetry_breaking_study(
    rng: np.random.Generator,
    *,
    cases: int,
) -> dict[str, float | int]:
    node_count = int(PROTOCOL["circle_node_count"])
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        node_count,
        group_id="controlled-s1",
    )
    angles = quadrature.nodes[:, 0]
    minimum_gauge_information = math.inf
    minimum_conditional_change = math.inf
    maximum_chain_rule_error = 0.0
    for case_index in range(cases):
        prior = SymmetryCompleteBeliefV1.with_reference_group_law(
            [0.4, 0.6],
            quadrature,
            belief_id=f"breaking-prior-{case_index}",
        )
        phases = rng.uniform(-math.pi, math.pi, size=2)
        strengths = rng.uniform(0.5, 2.0, size=2)
        likelihood = np.stack(
            [
                np.exp(strength * np.cos(angles - phase))
                for strength, phase in zip(strengths, phases, strict=True)
            ]
        )
        update = update_symmetry_complete_belief(
            prior,
            likelihood,
            evidence_semantics="symmetry-breaking",
            posterior_belief_id=f"breaking-posterior-{case_index}",
        )
        minimum_gauge_information = min(
            minimum_gauge_information,
            update.information.gauge_information_nats,
        )
        minimum_conditional_change = min(
            minimum_conditional_change,
            update.information.maximum_conditional_l1_change,
        )
        maximum_chain_rule_error = max(
            maximum_chain_rule_error,
            abs(
                update.information.total_information_nats
                - update.information.quotient_information_nats
                - update.information.gauge_information_nats
            ),
        )
    return {
        "case_count": cases,
        "minimum_gauge_information_nats": minimum_gauge_information,
        "minimum_conditional_l1_change": minimum_conditional_change,
        "maximum_kl_chain_rule_error_nats": maximum_chain_rule_error,
    }


def _point_completion_ladder() -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for node_count in PROTOCOL["completion_node_counts"]:
        quadrature = CompactGroupQuadratureV1.uniform_circle(
            int(node_count),
            group_id="controlled-s1",
        )
        belief = SymmetryCompleteBeliefV1.with_reference_group_law(
            [1.0],
            quadrature,
            belief_id=f"uniform-{node_count}",
        )
        audit = audit_point_completion(belief, [0])
        assert audit.discretized_specificity_nats is not None
        rows.append(
            {
                "node_count": int(node_count),
                "status": audit.status,
                "physical_point_completion_has_finite_kl": (
                    audit.physical_point_completion_has_finite_kl
                ),
                "discretized_specificity_nats": audit.discretized_specificity_nats,
                "expected_log_node_count_nats": math.log(float(node_count)),
                "absolute_error_nats": abs(
                    audit.discretized_specificity_nats - math.log(float(node_count))
                ),
            }
        )
    return rows


__all__ = [
    "_invariant_update_study",
    "_point_completion_ladder",
    "_symmetry_breaking_study",
]
