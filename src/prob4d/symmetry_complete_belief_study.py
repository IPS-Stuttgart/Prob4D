"""Controlled verification study for symmetry-complete beliefs.

This executable validates only algebra and numerical contracts. It does not
open a dataset, infer a physical group, calibrate a cover, verify a real
state--action execution binding, or authorize a paper claim about a learned
provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._symmetry_complete_study_actions import (
    _equivariant_action_coupling_study,
)
from ._symmetry_complete_study_common import PROTOCOL
from ._symmetry_complete_study_decisions import (
    _decision_cover_verification_study,
)
from ._symmetry_complete_study_pairwise import (
    _approximate_pairwise_action_study,
)
from ._symmetry_complete_study_queries import (
    _cover_verification_study,
    _shared_group_dependence_study,
)
from ._symmetry_complete_study_updates import (
    _invariant_update_study,
    _point_completion_ladder,
    _symmetry_breaking_study,
)


def run_study(*, seed: int, cases: int) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(cases, bool) or not isinstance(cases, int):
        raise TypeError("cases must be an integer")
    if cases < 1:
        raise ValueError("cases must be positive")
    rng = np.random.default_rng(seed)
    invariant = _invariant_update_study(rng, cases=cases)
    breaking = _symmetry_breaking_study(rng, cases=cases)
    completion = _point_completion_ladder()
    cover = _cover_verification_study(rng, cases=cases)
    decision_cover = _decision_cover_verification_study(rng, cases=cases)
    shared = _shared_group_dependence_study()
    action_coupling = _equivariant_action_coupling_study()
    approximate_pairwise = _approximate_pairwise_action_study()
    criteria = {
        "invariant_conditionals_preserved_exactly": (
            invariant["maximum_conditional_l1_change"] == 0.0
        ),
        "invariant_evidence_adds_no_gauge_information": (
            invariant["maximum_gauge_information_nats"] <= 1e-14
        ),
        "kl_chain_rule_verified": max(
            float(invariant["maximum_kl_chain_rule_error_nats"]),
            float(breaking["maximum_kl_chain_rule_error_nats"]),
        )
        <= 2e-11,
        "symmetry_breaking_changes_group_law": (
            breaking["minimum_conditional_l1_change"] > 0.0
            and breaking["minimum_gauge_information_nats"] > 0.0
        ),
        "continuous_point_completion_is_singular": all(
            row["status"] == "continuous-singular"
            and not row["physical_point_completion_has_finite_kl"]
            for row in completion
        ),
        "discretized_completion_specificity_matches_log_resolution": max(
            float(row["absolute_error_nats"]) for row in completion
        )
        <= 1e-14,
        "continuous_cover_upper_bounds_exact_harmonic_diameter": (
            cover["minimum_upper_minus_exact_diameter"] >= -2e-12
        ),
        "sample_diameter_is_valid_lower_bound": (
            cover["maximum_sample_minus_exact_diameter"] <= 2e-12
        ),
        "continuous_decision_upper_bounds_exact_regret": (
            decision_cover["minimum_upper_minus_exact_regret_or_gap"] >= -3e-12
        ),
        "continuous_decision_samples_lower_bound_exact_regret": (
            decision_cover["maximum_sample_minus_exact_regret_or_gap"] <= 3e-12
        ),
        "continuous_decision_selected_regret_is_bounded": (
            decision_cover["maximum_selected_true_minus_reported_upper"] <= 3e-12
        ),
        "continuous_decision_has_no_false_admission": (
            decision_cover["false_admission_count"] == 0
        ),
        "continuous_decision_cover_tightens_with_resolution": (
            decision_cover["mean_pairwise_cover_correction_monotone"] is True
        ),
        "continuous_decision_panel_contains_admissible_actions": all(
            row["admitted_case_count"] == cases for row in decision_cover["rows"]
        ),
        "shared_group_draw_preserves_exact_cancellation": (
            shared["maximum_shared_sum_norm"] <= 1e-14
        ),
        "independent_group_draw_destroys_cancellation": (
            shared["independent_mean_squared_sum_norm"] > 1.9
        ),
        "gauge_coupled_action_orbit_is_uniquely_identified": (
            action_coupling["coupled_action_status"] == "certified-admissible"
            and action_coupling["coupled_admissible_action_count"] == 1
            and action_coupling["coupled_selected_action_template"] == 0
        ),
        "decision_equivariance_does_not_require_absolute_loss_invariance": (
            action_coupling["maximum_absolute_loss_range"] > 3.9
            and action_coupling["maximum_pairwise_difference_range"] <= 1e-12
        ),
        "pairwise_equivariance_is_stricter_than_actionwise_lipschitz": (
            action_coupling["actionwise_lipschitz_status"] == "undetermined"
            and action_coupling["pairwise_equivariance_status"] == "certified-admissible"
        ),
        "fixed_frame_and_independent_gauges_do_not_identify_the_action": (
            action_coupling["fixed_frame_status"] == "scope-not-certified"
            and action_coupling["fixed_frame_optimal_action_count"] == 3
            and action_coupling["independent_gauge_optimal_action_count"] == 3
            and action_coupling["shared_gauge_optimal_action_count"] == 1
        ),
        "missing_execution_coupling_fails_closed": (
            action_coupling["missing_coupling_receipt_status"] == "scope-not-certified"
        ),
        "action_certificate_is_group_coordinate_invariant": (
            action_coupling["maximum_regret_change_under_group_coordinate_offset"] <= 2e-12
        ),
        "pairwise_lipschitz_bounds_cover_dense_approximate_regret": (
            approximate_pairwise["minimum_dense_minus_sampled_regret"] >= -3e-6
            and approximate_pairwise["minimum_upper_minus_dense_regret"] >= -3e-6
        ),
        "approximate_pairwise_certificate_identifies_action": (
            approximate_pairwise["pairwise_status"] == "certified-admissible"
            and approximate_pairwise["pairwise_selected_action"] == 0
            and approximate_pairwise["pairwise_upper_regret"][0] == 0.0
        ),
        "approximate_pairwise_regularization_is_strictly_tighter": (
            approximate_pairwise["maximum_sampled_pairwise_difference_range"] > 0.25
            and approximate_pairwise["pairwise_to_actionwise_correction_ratio"] < 0.04
            and approximate_pairwise["actionwise_status"] == "undetermined"
        ),
    }
    return {
        "schema": PROTOCOL["schema"],
        "schema_version": PROTOCOL["schema_version"],
        "seed": seed,
        "cases_per_stochastic_panel": cases,
        "protocol": PROTOCOL,
        "invariant_update": invariant,
        "symmetry_breaking_update": breaking,
        "point_completion_ladder": completion,
        "continuous_cover_verification": cover,
        "continuous_decision_verification": decision_cover,
        "shared_group_dependence": shared,
        "gauge_coupled_action_verification": action_coupling,
        "approximate_pairwise_action_verification": approximate_pairwise,
        "criteria": criteria,
        "decision": (
            "controlled-contract-passed" if all(criteria.values()) else "controlled-contract-failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cases", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(seed=args.seed, cases=args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result["decision"])
    if result["decision"] != "controlled-contract-passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
