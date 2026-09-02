"""Shared protocol for the symmetry-complete controlled study."""

from __future__ import annotations

from typing import Any

import numpy as np

PROTOCOL: dict[str, Any] = {
    "schema": "prob4d.symmetry-complete-belief-study",
    "schema_version": 1,
    "study_kind": "controlled-algebraic-verification",
    "quotient_count": 4,
    "circle_node_count": 32,
    "completion_node_counts": [4, 8, 16, 32, 64, 128],
    "cover_node_counts": [8, 16, 32, 64],
    "query_dimension": 3,
    "invariant_likelihood_semantics": "constant-on-prior-supported-orbit-nodes",
    "continuous_group": "S1",
    "group_metric": "wrapped-angle-radians-v1",
    "claim_boundary": (
        "Controlled finite-quadrature and certified-cover verification only. "
        "The study does not infer a symmetry, establish Haar quadrature accuracy, "
        "calibrate a Lipschitz or cover bound, validate a provider, open real data, "
        "prove target transport, authorize deployment, or certify safety."
    ),
}


def _random_probability(rng: np.random.Generator, size: int) -> np.ndarray:
    values = rng.gamma(shape=1.5, scale=1.0, size=size)
    return values / np.sum(values)


__all__ = ["PROTOCOL", "_random_probability"]
