"""Order-invariant generalized covariance intersection on local ``Sim(3)`` tangents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .covariance import (
    covariance_eigendecomposition,
    covariance_statistics,
    regularized_inverse_psd,
)
from .sim3 import Sim3

FloatArray = NDArray[np.floating]


def _readonly_covariance(values: FloatArray, *, name: str) -> FloatArray:
    _, eigenvalues, eigenvectors = covariance_eigendecomposition(
        values,
        name=name,
        eigenvalue_floor=1e-15,
    )
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    result = 0.5 * (result + result.T)
    result.setflags(write=False)
    return result


def _numerical_jacobian(function, vector: FloatArray) -> FloatArray:
    vector = np.asarray(vector, dtype=np.float64)
    baseline = np.asarray(function(vector), dtype=np.float64)
    jacobian = np.empty((baseline.size, vector.size), dtype=np.float64)
    for index in range(vector.size):
        step = 1e-6 * max(1.0, abs(float(vector[index])))
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.asarray(function(plus), dtype=np.float64)
            - np.asarray(function(minus), dtype=np.float64)
        ) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("numerical Sim(3) Jacobian contains non-finite values")
    return jacobian


def _inverse_with_covariance(
    transform: Sim3,
    covariance: FloatArray,
) -> tuple[Sim3, FloatArray]:
    vector = transform.as_vector()
    inverse = transform.inverse()
    jacobian = _numerical_jacobian(
        lambda value: Sim3.from_vector(value).inverse().as_vector(),
        vector,
    )
    propagated = jacobian @ covariance @ jacobian.T
    return inverse, _readonly_covariance(propagated, name="inverted gauge covariance")


def _compose_with_covariance(
    parent: Sim3,
    parent_covariance: FloatArray,
    relative: Sim3,
    relative_covariance: FloatArray,
) -> tuple[Sim3, FloatArray]:
    parent_vector = parent.as_vector()
    relative_vector = relative.as_vector()
    parent_jacobian = _numerical_jacobian(
        lambda value: Sim3.from_vector(value).compose(relative).as_vector(),
        parent_vector,
    )
    relative_jacobian = _numerical_jacobian(
        lambda value: parent.compose(Sim3.from_vector(value)).as_vector(),
        relative_vector,
    )
    covariance = (
        parent_jacobian @ parent_covariance @ parent_jacobian.T
        + relative_jacobian @ relative_covariance @ relative_jacobian.T
    )
    return parent.compose(relative), _readonly_covariance(
        covariance,
        name="composed gauge covariance",
    )


def right_invariant_residual(reference: Sim3, value: Sim3) -> FloatArray:
    """Return local coordinates of ``value`` relative to ``reference``.

    The coordinates are the repository's seven-vector convention applied to the
    right-invariant relative transform.  Unlike subtraction of two global
    rotation vectors, this remains finite when equivalent gauges lie on opposite
    sides of the SO(3) logarithm branch.
    """

    residual = reference.inverse().compose(value).as_vector()
    if not np.all(np.isfinite(residual)):
        raise ValueError("Sim(3) residual contains non-finite values")
    return residual


def constraint_residual(
    measured_reference_from_moving: Sim3,
    predicted_reference_from_moving: Sim3,
) -> FloatArray:
    """Return the intrinsic residual of one relative-gauge measurement."""

    return right_invariant_residual(
        measured_reference_from_moving,
        predicted_reference_from_moving,
    )


@dataclass(frozen=True)
class GaugeCandidate:
    """One uncertain estimate of a window's global gauge."""

    label: str
    global_from_local: Sim3
    covariance: FloatArray

    def __post_init__(self) -> None:
        label = str(self.label)
        if not label:
            raise ValueError("gauge candidate label must be nonempty")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge candidate covariance must have shape (7, 7)")
        object.__setattr__(self, "label", label)
        object.__setattr__(
            self,
            "covariance",
            _readonly_covariance(covariance, name=f"candidate {label!r} covariance"),
        )


@dataclass(frozen=True)
class GeneralizedCIFusionResult:
    """Order-invariant generalized-CI result in a local ``Sim(3)`` tangent."""

    global_from_local: Sim3
    covariance: FloatArray
    candidate_labels: tuple[str, ...]
    weights: FloatArray
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        labels = tuple(map(str, self.candidate_labels))
        if not labels or any(not label for label in labels):
            raise ValueError("CI candidate labels must be nonempty")
        if len(set(labels)) != len(labels):
            raise ValueError("CI candidate labels must be unique")
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        if weights.shape != (len(labels),):
            raise ValueError("CI weights changed shape")
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < -1e-12)
            or not np.isclose(np.sum(weights), 1.0, atol=1e-10)
        ):
            raise ValueError("CI weights must be a finite probability vector")
        weights = np.maximum(weights, 0.0)
        weights /= np.sum(weights)
        weights.setflags(write=False)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("CI covariance must have shape (7, 7)")
        if self.iterations < 1:
            raise ValueError("CI iteration count must be positive")
        object.__setattr__(self, "candidate_labels", labels)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "covariance",
            _readonly_covariance(covariance, name="generalized-CI covariance"),
        )


def _project_probability_simplex(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("simplex projection requires a finite nonempty vector")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    candidates = ordered - cumulative / np.arange(1, len(values) + 1) > 0.0
    if not np.any(candidates):
        return np.full(len(values), 1.0 / len(values))
    rho = int(np.flatnonzero(candidates)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    total = float(np.sum(projected))
    if total <= 0.0:
        return np.full(len(values), 1.0 / len(values))
    return projected / total


def _ci_objective(weights: FloatArray, information: FloatArray) -> float:
    combined = np.einsum("i,ijk->jk", weights, information, optimize=True)
    sign, log_determinant = np.linalg.slogdet(combined)
    if sign <= 0 or not np.isfinite(log_determinant):
        raise ValueError("generalized-CI information matrix is not positive definite")
    return -float(log_determinant)


def generalized_ci_weights(
    covariances: Sequence[FloatArray],
    *,
    maximum_iterations: int = 200,
    tolerance: float = 1e-11,
) -> tuple[FloatArray, FloatArray, int, bool]:
    """Optimize generalized covariance-intersection weights on the simplex.

    The objective is the log determinant of the fused covariance.  Projected
    gradient descent with deterministic backtracking is used to avoid an optional
    SciPy dependency.  Inputs are validated fail-closed before regularization.
    """

    if not covariances:
        raise ValueError("generalized CI requires at least one covariance")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    information = np.stack(
        [
            regularized_inverse_psd(
                covariance,
                name=f"generalized-CI covariance {index}",
            )
            for index, covariance in enumerate(covariances)
        ]
    )
    count = len(information)
    if count == 1:
        return (
            np.ones(1, dtype=np.float64),
            regularized_inverse_psd(information[0], name="generalized-CI information"),
            1,
            True,
        )

    weights = np.full(count, 1.0 / count, dtype=np.float64)
    objective = _ci_objective(weights, information)
    converged = False
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        combined = np.einsum("i,ijk->jk", weights, information, optimize=True)
        covariance = regularized_inverse_psd(
            combined,
            name="generalized-CI information",
        )
        gradient = -np.asarray(
            [np.trace(covariance @ item) for item in information],
            dtype=np.float64,
        )
        centered = gradient - np.mean(gradient)
        scale = max(float(np.linalg.norm(centered)), 1.0)
        step = 1.0 / scale
        accepted = False
        for _ in range(40):
            candidate = _project_probability_simplex(weights - step * gradient)
            difference = candidate - weights
            if np.linalg.norm(difference) <= tolerance:
                weights = candidate
                converged = True
                accepted = True
                break
            candidate_objective = _ci_objective(candidate, information)
            if candidate_objective <= objective + 1e-4 * float(gradient @ difference):
                weights = candidate
                objective = candidate_objective
                accepted = True
                break
            step *= 0.5
        if converged or not accepted:
            break
        if np.linalg.norm(difference) <= tolerance:
            converged = True
            break

    combined = np.einsum("i,ijk->jk", weights, information, optimize=True)
    covariance = regularized_inverse_psd(
        combined,
        name="generalized-CI information",
    )
    return weights, covariance, iteration, converged


def _candidate_in_tangent(
    reference: Sim3,
    candidate: GaugeCandidate,
) -> tuple[FloatArray, FloatArray]:
    candidate_vector = candidate.global_from_local.as_vector()

    def coordinates(value: FloatArray) -> FloatArray:
        return reference.inverse().compose(Sim3.from_vector(value)).as_vector()

    mean = coordinates(candidate_vector)
    jacobian = _numerical_jacobian(coordinates, candidate_vector)
    covariance = jacobian @ candidate.covariance @ jacobian.T
    return mean, _readonly_covariance(
        covariance,
        name=f"candidate {candidate.label!r} tangent covariance",
    )


def _tangent_result(
    reference: Sim3,
    mean: FloatArray,
    covariance: FloatArray,
) -> tuple[Sim3, FloatArray]:
    mean = np.asarray(mean, dtype=np.float64)
    if mean.shape != (7,):
        raise ValueError("fused tangent mean must have shape (7,)")

    def global_coordinates(value: FloatArray) -> FloatArray:
        return reference.compose(Sim3.from_vector(value)).as_vector()

    transform = reference.compose(Sim3.from_vector(mean))
    jacobian = _numerical_jacobian(global_coordinates, mean)
    propagated = jacobian @ covariance @ jacobian.T
    return transform, _readonly_covariance(
        propagated,
        name="fused global gauge covariance",
    )


def fuse_sim3_generalized_ci(
    candidates: Sequence[GaugeCandidate],
    *,
    maximum_manifold_iterations: int = 12,
    maximum_weight_iterations: int = 200,
    tolerance: float = 1e-9,
) -> GeneralizedCIFusionResult:
    """Fuse correlated gauge candidates without depending on input order.

    Candidate labels define the deterministic ordering.  Each iteration maps all
    candidates to the right-invariant tangent at the current estimate, solves one
    generalized-CI problem, and retracts the fused local mean back to ``Sim(3)``.
    """

    if not candidates:
        raise ValueError("at least one gauge candidate is required")
    if maximum_manifold_iterations < 1:
        raise ValueError("maximum_manifold_iterations must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    ordered = tuple(sorted(candidates, key=lambda item: item.label))
    labels = tuple(candidate.label for candidate in ordered)
    if len(set(labels)) != len(labels):
        raise ValueError("gauge candidate labels must be unique")
    if len(ordered) == 1:
        return GeneralizedCIFusionResult(
            global_from_local=ordered[0].global_from_local,
            covariance=ordered[0].covariance,
            candidate_labels=labels,
            weights=np.ones(1),
            iterations=1,
            converged=True,
        )

    ranked = []
    for candidate in ordered:
        _, _, log_determinant = covariance_statistics(
            candidate.covariance,
            name=f"candidate {candidate.label!r} covariance",
        )
        ranked.append((float(log_determinant), candidate.label, candidate))
    current = min(ranked, key=lambda item: (item[0], item[1]))[2].global_from_local

    last_weights = np.full(len(ordered), 1.0 / len(ordered))
    last_covariance = np.eye(7)
    last_transform = current
    converged = False
    iteration = 0
    for iteration in range(1, maximum_manifold_iterations + 1):
        local_means: list[FloatArray] = []
        local_covariances: list[FloatArray] = []
        for candidate in ordered:
            mean, covariance = _candidate_in_tangent(current, candidate)
            local_means.append(mean)
            local_covariances.append(covariance)
        weights, tangent_covariance, _, _ = generalized_ci_weights(
            local_covariances,
            maximum_iterations=maximum_weight_iterations,
        )
        information = np.stack(
            [
                regularized_inverse_psd(
                    covariance,
                    name=f"candidate {label!r} tangent covariance",
                )
                for label, covariance in zip(labels, local_covariances, strict=True)
            ]
        )
        information_vector = np.zeros(7, dtype=np.float64)
        for weight, precision, mean in zip(
            weights,
            information,
            local_means,
            strict=True,
        ):
            information_vector += weight * (precision @ mean)
        tangent_mean = tangent_covariance @ information_vector
        last_transform, last_covariance = _tangent_result(
            current,
            tangent_mean,
            tangent_covariance,
        )
        last_weights = weights
        if np.linalg.norm(tangent_mean) <= tolerance:
            converged = True
            break
        current = last_transform

    return GeneralizedCIFusionResult(
        global_from_local=last_transform,
        covariance=last_covariance,
        candidate_labels=labels,
        weights=last_weights,
        iterations=iteration,
        converged=converged,
    )


__all__ = [
    "GaugeCandidate",
    "GeneralizedCIFusionResult",
    "constraint_residual",
    "fuse_sim3_generalized_ci",
    "generalized_ci_weights",
    "right_invariant_residual",
]
