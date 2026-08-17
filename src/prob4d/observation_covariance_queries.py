"""Matrix-free covariance queries for explicit-gauge observation stacks.

The row-level marginal covariance stored by Prob4D contains only one observation
row at a time.  It is therefore insufficient for a query that combines several
rows because shared gauge uncertainty induces cross-row covariance.  This module
projects the exact structured covariance

``blockdiag(R_1, ..., R_M) + J Sigma_g J.T``

without materializing the full ``3M x 3M`` observation covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .observation_factors import StackedObservationFactors
from .sparse_observation_factors import SparseStackedObservationFactors
from .tree_sparse_observation_factors import TreeSparseStackedObservationFactors

FloatArray: TypeAlias = NDArray[np.floating[Any]]
CovarianceComponent: TypeAlias = Literal["conditional", "gauge", "marginal"]
ObservationFactorStack: TypeAlias = (
    StackedObservationFactors
    | SparseStackedObservationFactors
    | TreeSparseStackedObservationFactors
)


def _readonly_symmetric(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (result + result.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(result, symmetric, atol=1e-12 * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    symmetric = symmetric.copy()
    symmetric.setflags(write=False)
    return symmetric


@dataclass(frozen=True, slots=True)
class ProjectedObservationCovariance:
    """Conditional, shared-gauge, and total covariance of one linear query."""

    conditional_covariance: FloatArray
    gauge_covariance: FloatArray
    marginal_covariance: FloatArray

    def __post_init__(self) -> None:
        conditional = _readonly_symmetric(
            self.conditional_covariance,
            name="conditional_covariance",
        )
        gauge = _readonly_symmetric(
            self.gauge_covariance,
            name="gauge_covariance",
        )
        marginal = _readonly_symmetric(
            self.marginal_covariance,
            name="marginal_covariance",
        )
        if conditional.shape != gauge.shape or conditional.shape != marginal.shape:
            raise ValueError("projected covariance components must have equal shape")
        expected = conditional + gauge
        scale = max(float(np.max(np.abs(expected), initial=0.0)), 1.0)
        if not np.allclose(
            marginal,
            expected,
            atol=1e-12 * scale,
            rtol=1e-10,
        ):
            raise ValueError("marginal_covariance must equal conditional plus gauge covariance")
        object.__setattr__(self, "conditional_covariance", conditional)
        object.__setattr__(self, "gauge_covariance", gauge)
        object.__setattr__(self, "marginal_covariance", marginal)

    @property
    def query_dimension(self) -> int:
        """Number of scalar query outputs."""

        return int(self.marginal_covariance.shape[0])

    @property
    def scalar_variance(self) -> float:
        """Return the variance of a scalar query and reject vector-valued queries."""

        if self.query_dimension != 1:
            raise ValueError("scalar_variance requires a one-dimensional query")
        return float(self.marginal_covariance[0, 0])


def _require_stack(value: object) -> ObservationFactorStack:
    if not isinstance(
        value,
        (
            StackedObservationFactors,
            SparseStackedObservationFactors,
            TreeSparseStackedObservationFactors,
        ),
    ):
        raise TypeError(
            "stacked must be a StackedObservationFactors, "
            "SparseStackedObservationFactors, or TreeSparseStackedObservationFactors"
        )
    return value


def _observation_count(stacked: ObservationFactorStack) -> int:
    return int(len(stacked.world_mean_m))


def _validated_component(value: object) -> CovarianceComponent:
    if value not in {"conditional", "gauge", "marginal"}:
        raise ValueError("component must be 'conditional', 'gauge', or 'marginal'")
    return value  # type: ignore[return-value]


def _validated_rhs(
    value: object,
    *,
    observation_count: int,
) -> tuple[np.ndarray, bool]:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == (observation_count, 3):
        result = raw[:, :, None]
        squeeze = True
    elif raw.ndim == 3 and raw.shape[:2] == (observation_count, 3):
        result = raw
        squeeze = False
    else:
        raise ValueError("value must have shape (M, 3) or (M, 3, R)")
    if result.shape[2] < 1 or not np.all(np.isfinite(result)):
        raise ValueError("value must contain finite values and at least one right-hand side")
    return result, squeeze


def _conditional_covariance_action(
    stacked: ObservationFactorStack,
    values: np.ndarray,
) -> np.ndarray:
    local = np.asarray(stacked.conditional_world_covariance_m2, dtype=np.float64)
    count = _observation_count(stacked)
    if local.shape != (count, 3, 3) or not np.all(np.isfinite(local)):
        raise ValueError(
            "conditional_world_covariance_m2 must have finite shape (M, 3, 3)"
        )
    symmetric = 0.5 * (local + local.swapaxes(1, 2))
    scale = np.maximum(np.max(np.abs(symmetric), axis=(1, 2), initial=0.0), 1.0)
    if not np.allclose(
        local,
        symmetric,
        atol=1e-12 * scale[:, None, None],
        rtol=1e-10,
    ):
        raise ValueError("conditional_world_covariance_m2 must be symmetric")
    return np.einsum("mij,mjr->mir", symmetric, values, optimize=True)


def _dense_gauge_covariance_action(
    stacked: StackedObservationFactors,
    values: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(stacked.gauge_jacobian, dtype=np.float64)
    prior = np.asarray(stacked.gauge_prior_covariance, dtype=np.float64)
    count = _observation_count(stacked)
    if jacobian.ndim != 3 or jacobian.shape[:2] != (count, 3):
        raise ValueError("gauge_jacobian must have shape (M, 3, G)")
    gauge_dimension = int(jacobian.shape[2])
    if prior.shape != (gauge_dimension, gauge_dimension):
        raise ValueError("gauge_prior_covariance has changed shape")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(prior)):
        raise ValueError("gauge design and prior covariance must be finite")
    gauge_rhs = np.einsum("mig,mir->gr", jacobian, values, optimize=True)
    gauge_response = prior @ gauge_rhs
    return np.einsum("mig,gr->mir", jacobian, gauge_response, optimize=True)


def _sparse_gauge_covariance_action(
    stacked: SparseStackedObservationFactors,
    values: np.ndarray,
) -> np.ndarray:
    jacobian = np.asarray(stacked.local_gauge_jacobian, dtype=np.float64)
    indices = np.asarray(stacked.gauge_indices, dtype=np.int64)
    prior = np.asarray(stacked.gauge_prior_covariance, dtype=np.float64)
    count = _observation_count(stacked)
    gauge_count = len(stacked.gauge_ids)
    if jacobian.shape != (count, 3, 7):
        raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
    if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= gauge_count):
        raise ValueError("gauge_indices reference an unknown gauge")
    if prior.shape != (7 * gauge_count, 7 * gauge_count):
        raise ValueError("gauge_prior_covariance has changed shape")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(prior)):
        raise ValueError("gauge design and prior covariance must be finite")
    row_contributions = np.einsum("mij,mir->mjr", jacobian, values, optimize=True)
    gauge_rhs = np.zeros((gauge_count, 7, values.shape[2]), dtype=np.float64)
    np.add.at(gauge_rhs, indices, row_contributions)
    gauge_response = (prior @ gauge_rhs.reshape(7 * gauge_count, -1)).reshape(
        gauge_count,
        7,
        values.shape[2],
    )
    return np.einsum(
        "mij,mjr->mir",
        jacobian,
        gauge_response[indices],
        optimize=True,
    )


def _gauge_covariance_action(
    stacked: ObservationFactorStack,
    values: np.ndarray,
) -> np.ndarray:
    if isinstance(stacked, TreeSparseStackedObservationFactors):
        result = np.asarray(stacked.observation_gauge_covariance_action(values))
    elif isinstance(stacked, SparseStackedObservationFactors):
        result = _sparse_gauge_covariance_action(stacked, values)
    else:
        result = _dense_gauge_covariance_action(stacked, values)
    if result.shape != values.shape or not np.all(np.isfinite(result)):
        raise ValueError("gauge covariance action returned malformed values")
    return result


def observation_covariance_action(
    stacked: ObservationFactorStack,
    value: object,
    *,
    component: CovarianceComponent = "marginal",
) -> FloatArray:
    """Apply one structured observation covariance to one or more row vectors.

    ``value`` has shape ``(M, 3)`` or ``(M, 3, R)``.  The returned array has the
    same shape.  The full ``3M x 3M`` covariance is never materialized.
    """

    validated = _require_stack(stacked)
    selected_component = _validated_component(component)
    values, squeeze = _validated_rhs(
        value,
        observation_count=_observation_count(validated),
    )
    result = np.zeros_like(values)
    if selected_component in {"conditional", "marginal"}:
        result += _conditional_covariance_action(validated, values)
    if selected_component in {"gauge", "marginal"}:
        result += _gauge_covariance_action(validated, values)
    return result[:, :, 0] if squeeze else result


def observation_covariance_quadratic(
    stacked: ObservationFactorStack,
    value: object,
    *,
    component: CovarianceComponent = "marginal",
) -> float:
    """Return ``v.T @ Sigma @ v`` without forming ``Sigma``."""

    validated = _require_stack(stacked)
    raw = np.asarray(value, dtype=np.float64)
    count = _observation_count(validated)
    if raw.shape == (3 * count,):
        vector = raw.reshape(count, 3)
    elif raw.shape == (count, 3):
        vector = raw
    else:
        raise ValueError("value must have shape (3M,) or (M, 3)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("value must be finite")
    response = observation_covariance_action(
        validated,
        vector,
        component=component,
    )
    result = float(np.sum(vector * response, dtype=np.float64))
    scale = max(float(np.sum(np.abs(vector * response), dtype=np.float64)), 1.0)
    if result < -1e-10 * scale:
        raise RuntimeError("structured covariance produced a negative quadratic form")
    return max(result, 0.0)


def _validated_query(
    value: object,
    *,
    observation_count: int,
) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape == (observation_count, 3):
        query = raw[None]
    elif raw.ndim == 2 and raw.shape[1:] == (3 * observation_count,):
        query = raw.reshape(raw.shape[0], observation_count, 3)
    elif raw.ndim == 3 and raw.shape[1:] == (observation_count, 3):
        query = raw
    else:
        raise ValueError(
            "query_jacobian must have shape (M, 3), (Q, 3M), or (Q, M, 3)"
        )
    if query.shape[0] < 1 or not np.all(np.isfinite(query)):
        raise ValueError("query_jacobian must contain finite values and at least one query")
    return query


def _project_action(query: np.ndarray, response: np.ndarray) -> np.ndarray:
    projected = np.einsum("qmi,mir->qr", query, response, optimize=True)
    return 0.5 * (projected + projected.T)


def project_observation_covariance(
    stacked: ObservationFactorStack,
    query_jacobian: object,
) -> ProjectedObservationCovariance:
    """Return ``A Sigma A.T`` with conditional and shared-gauge components.

    ``query_jacobian`` may be a scalar row query ``(M, 3)``, a flattened matrix
    ``(Q, 3M)``, or a row-structured matrix ``(Q, M, 3)``.  Only the requested
    ``Q x Q`` covariance is materialized.
    """

    validated = _require_stack(stacked)
    query = _validated_query(
        query_jacobian,
        observation_count=_observation_count(validated),
    )
    right_hand_sides = np.moveaxis(query, 0, -1)
    conditional_response = observation_covariance_action(
        validated,
        right_hand_sides,
        component="conditional",
    )
    gauge_response = observation_covariance_action(
        validated,
        right_hand_sides,
        component="gauge",
    )
    conditional = _project_action(query, conditional_response)
    gauge = _project_action(query, gauge_response)
    return ProjectedObservationCovariance(
        conditional_covariance=conditional,
        gauge_covariance=gauge,
        marginal_covariance=conditional + gauge,
    )


__all__ = [
    "CovarianceComponent",
    "ObservationFactorStack",
    "ProjectedObservationCovariance",
    "observation_covariance_action",
    "observation_covariance_quadratic",
    "project_observation_covariance",
]
