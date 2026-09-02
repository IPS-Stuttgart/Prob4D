"""Query-orbit panels for the symmetry-complete controlled study."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._symmetry_complete_study_common import PROTOCOL
from .symmetry_complete_belief import (
    CompactGroupQuadratureV1,
    SymmetryCompleteBeliefV1,
    certify_compact_group_query,
    pushforward_shared_group_query,
)


def _cover_verification_study(
    rng: np.random.Generator,
    *,
    cases: int,
) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    global_minimum_upper_margin = math.inf
    global_maximum_lower_overshoot = 0.0
    for node_count in PROTOCOL["cover_node_counts"]:
        quadrature = CompactGroupQuadratureV1.uniform_circle(
            int(node_count),
            group_id="controlled-s1",
        )
        belief = SymmetryCompleteBeliefV1.with_reference_group_law(
            [1.0],
            quadrature,
            belief_id=f"cover-{node_count}",
        )
        angles = quadrature.nodes[:, 0]
        minimum_upper_margin = math.inf
        maximum_lower_overshoot = 0.0
        maximum_interval_width = 0.0
        for case_index in range(cases):
            coefficient = rng.normal(size=(int(PROTOCOL["query_dimension"]), 2))
            atoms = (
                coefficient[:, 0, None] * np.cos(angles)[None, :]
                + coefficient[:, 1, None] * np.sin(angles)[None, :]
            ).T[None, :, :]
            lipschitz = float(np.linalg.svd(coefficient, compute_uv=False)[0])
            exact_diameter = 2.0 * lipschitz
            certificate = certify_compact_group_query(
                belief,
                atoms,
                query_id=f"random-harmonic-{node_count}-{case_index}",
                lipschitz_by_quotient=lipschitz,
                tolerance=0.0,
            )
            lower = certificate.maximum_sample_diameter
            upper = certificate.maximum_upper_diameter
            minimum_upper_margin = min(minimum_upper_margin, upper - exact_diameter)
            maximum_lower_overshoot = max(
                maximum_lower_overshoot,
                lower - exact_diameter,
            )
            maximum_interval_width = max(maximum_interval_width, upper - lower)
        global_minimum_upper_margin = min(
            global_minimum_upper_margin,
            minimum_upper_margin,
        )
        global_maximum_lower_overshoot = max(
            global_maximum_lower_overshoot,
            maximum_lower_overshoot,
        )
        rows.append(
            {
                "node_count": int(node_count),
                "case_count": cases,
                "cover_radius": quadrature.cover_radius,
                "minimum_upper_minus_exact_diameter": minimum_upper_margin,
                "maximum_sample_minus_exact_diameter": maximum_lower_overshoot,
                "maximum_bound_interval_width": maximum_interval_width,
            }
        )
    return {
        "rows": rows,
        "minimum_upper_minus_exact_diameter": global_minimum_upper_margin,
        "maximum_sample_minus_exact_diameter": global_maximum_lower_overshoot,
    }


def _shared_group_dependence_study() -> dict[str, float | int]:
    node_count = 128
    quadrature = CompactGroupQuadratureV1.uniform_circle(
        node_count,
        group_id="controlled-s1",
    )
    belief = SymmetryCompleteBeliefV1.with_reference_group_law(
        [1.0],
        quadrature,
        belief_id="shared-group",
    )
    angles = quadrature.nodes[:, 0]
    first = np.column_stack((np.cos(angles), np.sin(angles)))
    second = -first
    law = pushforward_shared_group_query(
        belief,
        np.concatenate((first, second), axis=1)[None, :, :],
    )
    shared_sum = law.atoms[:, :2] + law.atoms[:, 2:]
    independent_sums = first[:, None, :] + second[None, :, :]
    independent_mean_squared_sum_norm = float(
        np.mean(np.sum(independent_sums * independent_sums, axis=2))
    )
    return {
        "node_count": node_count,
        "maximum_shared_sum_norm": float(np.max(np.linalg.norm(shared_sum, axis=1))),
        "shared_cross_covariance_x": float(law.covariance[0, 2]),
        "shared_cross_covariance_y": float(law.covariance[1, 3]),
        "independent_mean_squared_sum_norm": independent_mean_squared_sum_norm,
    }


__all__ = ["_cover_verification_study", "_shared_group_dependence_study"]
