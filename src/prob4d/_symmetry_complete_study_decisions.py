"""Continuous-orbit decision panels for the symmetry-complete study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._symmetry_complete_study_common import PROTOCOL, _random_probability
from .symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    SymmetryCompleteBeliefV1,
    certify_compact_group_decision,
)

_ACTION_COUNT = 4


def _exact_harmonic_regret(
    quotient: np.ndarray,
    offset: np.ndarray,
    cosine: np.ndarray,
    sine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_count = offset.shape[1]
    pairwise = np.zeros((action_count, action_count), dtype=np.float64)
    for action in range(action_count):
        for benchmark in range(action_count):
            offset_gap = offset[:, action] - offset[:, benchmark]
            cosine_gap = cosine[:, action] - cosine[:, benchmark]
            sine_gap = sine[:, action] - sine[:, benchmark]
            class_supremum = offset_gap + np.hypot(cosine_gap, sine_gap)
            pairwise[action, benchmark] = float(np.dot(quotient, class_supremum))
    np.fill_diagonal(pairwise, 0.0)
    regret = np.maximum(np.max(pairwise, axis=1), 0.0)
    return pairwise, regret


def _decision_cover_verification_study(
    rng: np.random.Generator,
    *,
    cases: int,
) -> dict[str, Any]:
    quotient_count = int(PROTOCOL["quotient_count"])
    generated: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(cases):
        quotient = _random_probability(rng, quotient_count)
        cosine = rng.normal(scale=0.35, size=(quotient_count, _ACTION_COUNT))
        sine = rng.normal(scale=0.35, size=(quotient_count, _ACTION_COUNT))
        amplitude = np.hypot(cosine, sine)
        offset = (
            1.0
            + amplitude
            + rng.uniform(
                0.0,
                1.0,
                size=(quotient_count, _ACTION_COUNT),
            )
        )
        generated.append((quotient, offset, cosine, sine))

    rows: list[dict[str, float | int]] = []
    global_minimum_upper_margin = math.inf
    global_maximum_lower_overshoot = 0.0
    global_maximum_selected_true_minus_upper = 0.0
    false_admission_count = 0
    mean_pairwise_corrections: list[float] = []
    for node_count in PROTOCOL["cover_node_counts"]:
        quadrature = CompactGroupQuadratureV1.uniform_circle(
            int(node_count),
            group_id="controlled-decision-s1",
        )
        angle = quadrature.nodes[:, 0]
        minimum_upper_margin = math.inf
        maximum_lower_overshoot = 0.0
        maximum_selected_true_minus_upper = 0.0
        admitted_case_count = 0
        row_false_admission_count = 0
        pairwise_correction_sum = 0.0
        for case_index, (quotient, offset, cosine, sine) in enumerate(generated):
            belief = SymmetryCompleteBeliefV1.with_reference_group_law(
                quotient,
                quadrature,
                belief_id=f"decision-{node_count}-{case_index}",
            )
            losses = (
                offset[:, None, :]
                + cosine[:, None, :] * np.cos(angle)[None, :, None]
                + sine[:, None, :] * np.sin(angle)[None, :, None]
            )
            action_lipschitz = np.hypot(cosine, sine)
            exact_pairwise, exact_regret = _exact_harmonic_regret(
                quotient,
                offset,
                cosine,
                sine,
            )
            pairwise_correction = float(
                np.max(
                    np.tensordot(
                        quotient,
                        action_lipschitz[:, :, None] + action_lipschitz[:, None, :],
                        axes=(0, 0),
                    )
                    * quadrature.cover_radius
                )
            )
            tolerance = float(np.min(exact_regret)) + pairwise_correction
            certificate = certify_compact_group_decision(
                belief,
                losses,
                action_loss_lipschitz_by_quotient=action_lipschitz,
                regret_tolerance=tolerance,
                lipschitz_bound_certified=True,
            )
            minimum_upper_margin = min(
                minimum_upper_margin,
                float(np.min(certificate.upper_pairwise_worst_case_loss_gap - exact_pairwise)),
                float(np.min(certificate.upper_worst_case_regret - exact_regret)),
            )
            maximum_lower_overshoot = max(
                maximum_lower_overshoot,
                float(np.max(certificate.sampled_pairwise_worst_case_loss_gap - exact_pairwise)),
                float(np.max(certificate.sampled_worst_case_regret - exact_regret)),
            )
            selected = certificate.minimax_upper_action_index
            maximum_selected_true_minus_upper = max(
                maximum_selected_true_minus_upper,
                float(exact_regret[selected] - certificate.upper_worst_case_regret[selected]),
            )
            false_mask = certificate.tolerance_admissible_action_mask & (
                exact_regret > tolerance + 1e-12
            )
            row_false_admission_count += int(np.count_nonzero(false_mask))
            admitted_case_count += int(certificate.has_tolerance_admissible_action)
            pairwise_correction_sum += float(
                np.max(
                    certificate.upper_pairwise_worst_case_loss_gap
                    - certificate.sampled_pairwise_worst_case_loss_gap
                )
            )
        mean_correction = pairwise_correction_sum / cases
        mean_pairwise_corrections.append(mean_correction)
        global_minimum_upper_margin = min(
            global_minimum_upper_margin,
            minimum_upper_margin,
        )
        global_maximum_lower_overshoot = max(
            global_maximum_lower_overshoot,
            maximum_lower_overshoot,
        )
        global_maximum_selected_true_minus_upper = max(
            global_maximum_selected_true_minus_upper,
            maximum_selected_true_minus_upper,
        )
        false_admission_count += row_false_admission_count
        rows.append(
            {
                "node_count": int(node_count),
                "case_count": cases,
                "admitted_case_count": admitted_case_count,
                "false_admission_count": row_false_admission_count,
                "cover_radius": quadrature.cover_radius,
                "mean_maximum_pairwise_cover_correction": mean_correction,
                "minimum_upper_minus_exact_regret_or_gap": minimum_upper_margin,
                "maximum_sample_minus_exact_regret_or_gap": (maximum_lower_overshoot),
                "maximum_selected_true_minus_reported_upper": (maximum_selected_true_minus_upper),
            }
        )
    correction_monotone = all(
        later <= earlier + 1e-14
        for earlier, later in zip(
            mean_pairwise_corrections,
            mean_pairwise_corrections[1:],
            strict=True,
        )
    )
    return {
        "action_count": _ACTION_COUNT,
        "rows": rows,
        "minimum_upper_minus_exact_regret_or_gap": (global_minimum_upper_margin),
        "maximum_sample_minus_exact_regret_or_gap": (global_maximum_lower_overshoot),
        "maximum_selected_true_minus_reported_upper": (global_maximum_selected_true_minus_upper),
        "false_admission_count": false_admission_count,
        "mean_pairwise_cover_correction_monotone": correction_monotone,
    }


__all__ = ["_decision_cover_verification_study"]
