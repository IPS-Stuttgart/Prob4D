"""Experimental shared-gauge curvature moments; not a provider-v2 exporter.

The formulas are the Gaussian moments of a *quadratic Taylor surrogate*.
They are exact for a quadratic map, not an exact nonlinear pushforward, a
calibration certificate, or a generally second-order-accurate covariance.
In particular, linear--cubic terms of the same order are absent in general.
See ``docs/gauge-curvature.md`` for the tangent-null result and scope.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _array(value: Any, *, ndim: int, name: str) -> FloatArray:
    if np.iscomplexobj(np.asarray(value)):
        raise ValueError(f"{name} must be real-valued")
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _finite_result(value: FloatArray, *, name: str) -> FloatArray:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} overflowed or became non-finite")
    return value


@dataclass(frozen=True, slots=True)
class SharedGaugeMoments:
    """Joint quadratic-surrogate moments, stored without a dense covariance.

    Columns of both factors are *shared* across all output coordinates.
    They are not independent per-point noise. Curvature columns are
    orthogonal Gaussian polynomial features, not independent Gaussian
    latent variables and not additional physical degrees of freedom.
    """

    mean: FloatArray
    linear_factor: FloatArray
    curvature_factor: FloatArray
    evaluation_count: int = 0

    def __post_init__(self) -> None:
        mean = _array(self.mean, ndim=1, name="mean")
        linear = _array(self.linear_factor, ndim=2, name="linear_factor")
        curvature = _array(self.curvature_factor, ndim=2, name="curvature_factor")
        if mean.size == 0 or linear.shape[0] != mean.size or curvature.shape[0] != mean.size:
            raise ValueError("mean and factor output dimensions must agree and be positive")
        rank = linear.shape[1]
        if curvature.shape[1] != rank * (rank + 1) // 2:
            raise ValueError("curvature_factor must contain all diagonal and mixed features")
        if (
            isinstance(self.evaluation_count, bool)
            or not isinstance(self.evaluation_count, int)
            or self.evaluation_count < 0
        ):
            raise ValueError("evaluation_count must be a nonnegative integer")
        for name, array in (
            ("mean", mean),
            ("linear_factor", linear),
            ("curvature_factor", curvature),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def input_rank(self) -> int:
        return int(self.linear_factor.shape[1])

    @property
    def covariance_factor(self) -> FloatArray:
        """Return F with Cov[f_quadratic] = F F.T; no compression is performed."""
        return np.concatenate((self.linear_factor, self.curvature_factor), axis=1)

    @property
    def marginal_variance(self) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.sum(self.linear_factor**2, axis=1)
            result += np.sum(self.curvature_factor**2, axis=1)
        return _finite_result(result, name="marginal variance")

    def covariance(self) -> FloatArray:
        """Materialize the joint covariance only when explicitly requested."""
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.linear_factor @ self.linear_factor.T
            result += self.curvature_factor @ self.curvature_factor.T
        return _finite_result(result, name="covariance")

    def covariance_action(self, vector: Any) -> FloatArray:
        """Multiply by the joint covariance, without constructing it."""
        operand = _array(vector, ndim=1, name="vector")
        if operand.shape != self.mean.shape:
            raise ValueError("vector must have the output dimension")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.linear_factor @ (self.linear_factor.T @ operand)
            result += self.curvature_factor @ (self.curvature_factor.T @ operand)
        return _finite_result(result, name="covariance action")

    def project(self, matrix: Any) -> SharedGaugeMoments:
        """Project all outputs jointly through a fixed, outcome-independent map."""
        projection = _array(matrix, ndim=2, name="projection")
        if projection.shape[0] == 0 or projection.shape[1] != self.mean.size:
            raise ValueError(
                "projection must have shape (positive query dimension, output dimension)"
            )
        return SharedGaugeMoments(
            mean=projection @ self.mean,
            linear_factor=projection @ self.linear_factor,
            curvature_factor=projection @ self.curvature_factor,
            evaluation_count=self.evaluation_count,
        )

    def input_cross_covariance(self, covariance_root: Any) -> FloatArray:
        """Return Cov[g, f_quadratic] = L A.T for g = mean + L z.

        The root must use exactly the same whitened coordinates used to
        construct these moments. Shape validation cannot certify that lineage.
        """
        root = _array(covariance_root, ndim=2, name="covariance_root")
        if root.shape[0] == 0 or root.shape[1] != self.input_rank:
            raise ValueError("covariance_root must use the same input rank")
        with np.errstate(over="ignore", invalid="ignore"):
            result = root @ self.linear_factor.T
        return _finite_result(result, name="input cross covariance")


def quadratic_gaussian_moments(
    value_at_mean: Any,
    linear: Any,
    hessian: Any,
) -> SharedGaugeMoments:
    """Compute exact Gaussian moments of a quadratic in whitened coordinates.

    For z ~ N(0, I_r), the output is
    f_i(z) = c_i + A_i z + (1/2) z.T B_i z.
    Arguments have shapes (p,), (p, r), and (p, r, r).
    Each B_i must be symmetric. No covariance-rank truncation is used.
    """
    value = _array(value_at_mean, ndim=1, name="value_at_mean")
    derivative = _array(linear, ndim=2, name="linear")
    second = _array(hessian, ndim=3, name="hessian")
    if value.size == 0 or derivative.shape[0] != value.size:
        raise ValueError("value_at_mean and linear output dimensions must agree")
    rank = derivative.shape[1]
    if second.shape != (value.size, rank, rank):
        raise ValueError("hessian must have shape (output dimension, input rank, input rank)")
    if not np.allclose(second, np.swapaxes(second, 1, 2), atol=1e-12, rtol=1e-10):
        raise ValueError("each output Hessian must be symmetric")
    second = 0.5 * (second + np.swapaxes(second, 1, 2))
    diagonal = np.diagonal(second, axis1=1, axis2=2)
    columns = [diagonal[:, index] / math.sqrt(2.0) for index in range(rank)]
    columns.extend(second[:, i, j] for i in range(rank) for j in range(i + 1, rank))
    curvature = np.column_stack(columns) if columns else np.empty((value.size, 0))
    return SharedGaugeMoments(
        mean=value + 0.5 * np.sum(diagonal, axis=1),
        linear_factor=derivative,
        curvature_factor=curvature,
    )


def finite_difference_gauge_moments(
    function: Callable[[FloatArray], Any],
    mean: Any,
    covariance_root: Any,
    *,
    step: float = 1e-3,
    max_rank: int = 32,
) -> SharedGaugeMoments:
    """Build all local linear, diagonal-curvature, and mixed-curvature factors.

    Differentiates z -> function(mean + covariance_root @ z) near z=0.
    Uses 1 + 2 r**2 function calls; ``step`` is in standardized coordinates.
    This is a derivative stencil, not a Gaussian quadrature rule. The callback
    must be deterministic and may use only the caller-authorized data.

    The caller supplies the *joint* covariance root across all gauges/windows.
    A rank limit raises an error rather than silently discarding uncertainty.
    Check numerical convergence with a second step before scientific use.
    """
    center = _array(mean, ndim=1, name="mean")
    root = _array(covariance_root, ndim=2, name="covariance_root")
    if center.size == 0 or root.shape[0] != center.size:
        raise ValueError("mean and covariance_root input dimensions must agree")
    if isinstance(step, bool) or not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and strictly positive")
    if isinstance(max_rank, bool) or not isinstance(max_rank, int) or max_rank < 0:
        raise ValueError("max_rank must be a nonnegative integer")
    rank = root.shape[1]
    if rank > max_rank:
        raise ValueError("input rank exceeds max_rank; uncertainty was not truncated")
    value = _array(function(center.copy()), ndim=1, name="function output")
    if value.size == 0:
        raise ValueError("function output must be nonempty")

    def evaluate(offset: FloatArray) -> FloatArray:
        argument = _finite_result(center + offset, name="stencil argument")
        output = _array(function(argument), ndim=1, name="function output")
        if output.shape != value.shape:
            raise ValueError("function output shape changed inside the stencil")
        return output

    if rank == 0:
        return SharedGaugeMoments(value, np.empty((value.size, 0)), np.empty((value.size, 0)), 1)
    offsets = float(step) * root
    first_columns: list[FloatArray] = []
    diagonal_columns: list[FloatArray] = []
    for index in range(rank):
        plus = evaluate(offsets[:, index])
        minus = evaluate(-offsets[:, index])
        first_columns.append((plus - minus) / (2.0 * step))
        diagonal_columns.append(((plus - value) + (minus - value)) / step**2)
    curvature_columns = [column / math.sqrt(2.0) for column in diagonal_columns]
    for i in range(rank):
        for j in range(i + 1, rank):
            pp = evaluate(offsets[:, i] + offsets[:, j])
            pm = evaluate(offsets[:, i] - offsets[:, j])
            mp = evaluate(-offsets[:, i] + offsets[:, j])
            mm = evaluate(-offsets[:, i] - offsets[:, j])
            curvature_columns.append(((pp - pm) - (mp - mm)) / (4.0 * step**2))
    return SharedGaugeMoments(
        mean=value + 0.5 * np.sum(np.column_stack(diagonal_columns), axis=1),
        linear_factor=np.column_stack(first_columns),
        curvature_factor=np.column_stack(curvature_columns),
        evaluation_count=1 + 2 * rank**2,
    )


def sim3_chain_gauge_moments(
    transform_vectors: Any,
    joint_covariance_root: Any,
    points_local_m: Any,
    *,
    query_matrix: Any | None = None,
    step: float = 1e-3,
    max_rank: int = 32,
) -> SharedGaugeMoments:
    """Apply local shared curvature to a chain of Prob4D ``Sim3`` transforms.

    Transform vectors use Prob4D's [log_scale, rotvec(3), translation(3)]
    convention. Chain order is T_0.compose(T_1).compose(...).
    The joint root has shape (7*K, r); points have shape (N, 3).
    A fixed query matrix has shape (q, 3*N), with point-major flattening.
    Passing the query evaluates and stores its factors directly.

    This computes smooth forward point coordinates, not a new logarithmic
    gauge state. It does not change branch-cut, support, covariance-calibration,
    export, or downstream admission policies. Conditional point noise is not
    included and must remain a separately calibrated covariance component.
    """
    from .sim3 import Sim3

    vectors = _array(transform_vectors, ndim=2, name="transform_vectors")
    points = _array(points_local_m, ndim=2, name="points_local_m")
    if vectors.shape[0] == 0 or vectors.shape[1] != 7:
        raise ValueError("transform_vectors must have shape (K, 7), K >= 1")
    if points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError("points_local_m must have shape (N, 3), N >= 1")
    projection = None
    if query_matrix is not None:
        projection = _array(query_matrix, ndim=2, name="query_matrix")
        if projection.shape[0] == 0 or projection.shape[1] != points.size:
            raise ValueError("query_matrix must have shape (q, 3*N), q >= 1")

    def transform(flat: FloatArray) -> FloatArray:
        transforms = [Sim3.from_vector(vector) for vector in flat.reshape(vectors.shape)]
        current = transforms[0]
        for following in transforms[1:]:
            current = current.compose(following)
        transformed = current.transform_points(points).reshape(-1)
        return transformed if projection is None else projection @ transformed

    return finite_difference_gauge_moments(
        transform,
        vectors.reshape(-1),
        joint_covariance_root,
        step=step,
        max_rank=max_rank,
    )
