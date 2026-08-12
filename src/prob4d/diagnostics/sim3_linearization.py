"""Source-only adequacy checks for first-order Gaussian ``Sim(3)`` propagation.

The diagnostic compares Jacobian-propagated moments against streamed nonlinear Monte
Carlo moments under the same frozen Gaussian perturbation. It is deliberately a
certificate, not a covariance repair: an inadequate result is evidence to retain an
explicit gauge latent or an existing exact fallback rather than silently widening a
point covariance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._atomic_file import atomic_write_text
from .._immutable_json import frozen_finite_json_mapping, plain_json
from ..sim3 import Sim3

FloatArray: TypeAlias = NDArray[np.floating]
ExactEvaluator: TypeAlias = Callable[[FloatArray], FloatArray]
PerturbationSide = Literal["left", "right"]
ParameterBlock = Literal["scale", "rotation", "translation"]

SIM3_LINEARIZATION_SCHEMA: Final = "prob4d.gaussian-linearization-adequacy"
SIM3_LINEARIZATION_VERSION: Final = 1
SIM3_LINEARIZATION_CLAIM_BOUNDARY: Final = (
    "This source-only certificate tests local first-order Gaussian moment propagation "
    "against nonlinear Monte Carlo under one frozen perturbation model. It does not "
    "establish provider competence, target transfer, calibrated deployment uncertainty, "
    "BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the art. "
    "An inadequate certificate does not authorize covariance inflation; retain the "
    "explicit gauge latent or an already-declared exact fallback."
)

_CANONICAL_BLOCKS: Final[tuple[ParameterBlock, ...]] = (
    "scale",
    "rotation",
    "translation",
)
_BLOCK_WIDTH: Final[dict[ParameterBlock, int]] = {
    "scale": 1,
    "rotation": 3,
    "translation": 3,
}
_CANONICAL_SLICE: Final[dict[ParameterBlock, slice]] = {
    "scale": slice(0, 1),
    "rotation": slice(1, 4),
    "translation": slice(4, 7),
}


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _exact_positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _exact_nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a genuine integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validated_covariance(value: object, *, dimension: int, name: str) -> FloatArray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (covariance + covariance.T)
    if not np.allclose(covariance, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(symmetric), initial=0.0))
    if minimum_eigenvalue < -(1e-12 + 1e-10 * scale):
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _validated_order(value: Sequence[str]) -> tuple[ParameterBlock, ...]:
    order = tuple(value)
    if order not in {
        ("scale", "rotation", "translation"),
        ("scale", "translation", "rotation"),
        ("rotation", "scale", "translation"),
        ("rotation", "translation", "scale"),
        ("translation", "scale", "rotation"),
        ("translation", "rotation", "scale"),
    }:
        raise ValueError("parameter_order must contain scale, rotation, and translation once")
    return order  # type: ignore[return-value]


def _ordered_to_canonical(
    vector: FloatArray,
    parameter_order: tuple[ParameterBlock, ...],
) -> FloatArray:
    ordered = np.asarray(vector, dtype=np.float64)
    if ordered.shape != (7,) or not np.all(np.isfinite(ordered)):
        raise ValueError("Sim(3) perturbation vector must be finite with shape (7,)")
    canonical = np.empty(7, dtype=np.float64)
    cursor = 0
    for block in parameter_order:
        width = _BLOCK_WIDTH[block]
        canonical[_CANONICAL_SLICE[block]] = ordered[cursor : cursor + width]
        cursor += width
    return canonical


def _normalize_output(
    value: object,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> FloatArray:
    output = np.asarray(value, dtype=np.float64)
    if output.ndim == 1:
        output = output[None, :]
    if output.ndim != 2 or output.shape[0] < 1 or output.shape[1] < 1:
        raise ValueError("exact evaluator output must have shape (items, dimensions)")
    if expected_shape is not None and output.shape != expected_shape:
        raise ValueError("exact evaluator output shape changed across evaluations")
    if not np.all(np.isfinite(output)):
        raise ValueError("exact evaluator output must be finite")
    return output


def _finite_difference_jacobian(
    mean_parameters: FloatArray,
    exact_evaluator: ExactEvaluator,
    *,
    step: float,
    output_shape: tuple[int, int],
) -> FloatArray:
    dimension = mean_parameters.size
    jacobian = np.empty(output_shape + (dimension,), dtype=np.float64)
    for coordinate in range(dimension):
        delta = step * max(1.0, abs(float(mean_parameters[coordinate])))
        plus = mean_parameters.copy()
        minus = mean_parameters.copy()
        plus[coordinate] += delta
        minus[coordinate] -= delta
        plus_output = _normalize_output(exact_evaluator(plus), expected_shape=output_shape)
        minus_output = _normalize_output(exact_evaluator(minus), expected_shape=output_shape)
        jacobian[..., coordinate] = (plus_output - minus_output) / (2.0 * delta)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("finite-difference Jacobian is non-finite")
    return jacobian


def _validated_jacobian(
    value: object,
    *,
    output_shape: tuple[int, int],
    parameter_dimension: int,
) -> FloatArray:
    jacobian = np.asarray(value, dtype=np.float64)
    expected = output_shape + (parameter_dimension,)
    if jacobian.shape == (output_shape[0] * output_shape[1], parameter_dimension):
        jacobian = jacobian.reshape(expected)
    if jacobian.shape != expected:
        raise ValueError(f"jacobian must have shape {expected}")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("jacobian must be finite")
    return jacobian


def _validate_query_projection(value: object, *, output_dimension: int) -> FloatArray:
    projection = np.asarray(value, dtype=np.float64)
    if projection.ndim != 2 or projection.shape[0] < 1:
        raise ValueError("query_projection must be a nonempty matrix")
    if projection.shape[1] != output_dimension:
        raise ValueError("query_projection has an invalid input dimension")
    if not np.all(np.isfinite(projection)):
        raise ValueError("query_projection must be finite")
    return projection


def _sample_gaussian(
    rng: np.random.Generator,
    mean: FloatArray,
    covariance: FloatArray,
    count: int,
) -> FloatArray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-12 + 1e-10 * max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError("covariance became indefinite before sampling")
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))[None, :]
    standard = rng.standard_normal((count, mean.size))
    return mean[None, :] + standard @ factor.T


def _sample_covariance(sum_outer: FloatArray, mean: FloatArray, count: int) -> FloatArray:
    covariance = (sum_outer - count * np.einsum("...i,...j->...ij", mean, mean)) / (count - 1)
    return 0.5 * (covariance + np.swapaxes(covariance, -1, -2))


def _relative_covariance_metrics(
    reference: FloatArray,
    approximate: FloatArray,
) -> tuple[float, float]:
    reference_trace = float(np.trace(reference))
    approximate_trace = float(np.trace(approximate))
    trace_scale = max(abs(reference_trace), np.finfo(np.float64).eps)
    relative_trace_error = abs(approximate_trace - reference_trace) / trace_scale
    reference_frobenius = float(np.linalg.norm(reference, ord="fro"))
    frobenius_scale = max(reference_frobenius, np.finfo(np.float64).eps)
    relative_frobenius_error = (
        float(np.linalg.norm(approximate - reference, ord="fro")) / frobenius_scale
    )
    return relative_trace_error, relative_frobenius_error


def _mean_shift_standard_deviations(
    nonlinear_mean: FloatArray,
    linearized_mean: FloatArray,
    nonlinear_covariance: FloatArray,
) -> float:
    delta = np.asarray(nonlinear_mean - linearized_mean, dtype=np.float64)
    rms_standard_deviation = np.sqrt(
        max(float(np.trace(nonlinear_covariance)) / delta.size, np.finfo(np.float64).eps)
    )
    return float(np.linalg.norm(delta) / rms_standard_deviation)


def _principal_axis_metrics(
    reference: FloatArray,
    approximate: FloatArray,
    *,
    minimum_anisotropy: float,
) -> tuple[float, float | None]:
    if reference.shape[0] < 2:
        return 1.0, None
    reference_values, reference_vectors = np.linalg.eigh(reference)
    approximate_values, approximate_vectors = np.linalg.eigh(approximate)
    order = np.argsort(reference_values)[::-1]
    reference_values = reference_values[order]
    reference_vectors = reference_vectors[:, order]
    approximate_order = np.argsort(approximate_values)[::-1]
    approximate_vectors = approximate_vectors[:, approximate_order]
    second = max(float(reference_values[1]), np.finfo(np.float64).eps)
    anisotropy = float(reference_values[0]) / second
    if anisotropy < minimum_anisotropy:
        return anisotropy, None
    cosine = abs(float(reference_vectors[:, 0] @ approximate_vectors[:, 0]))
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return anisotropy, angle


@dataclass(frozen=True, slots=True)
class LinearizationAdequacyThresholdsV1:
    """Frozen tolerances for a source-only linearization decision."""

    maximum_relative_trace_error: float = 0.10
    maximum_relative_frobenius_error: float = 0.15
    maximum_mean_shift_standard_deviations: float = 0.25
    maximum_principal_axis_angle_degrees: float = 10.0
    minimum_principal_axis_anisotropy: float = 1.25
    maximum_query_relative_trace_error: float = 0.10
    maximum_query_relative_frobenius_error: float = 0.15
    maximum_query_mean_shift_standard_deviations: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "maximum_relative_trace_error",
            "maximum_relative_frobenius_error",
            "maximum_mean_shift_standard_deviations",
            "maximum_principal_axis_angle_degrees",
            "maximum_query_relative_trace_error",
            "maximum_query_relative_frobenius_error",
            "maximum_query_mean_shift_standard_deviations",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name=name))
        anisotropy = _finite_positive(
            self.minimum_principal_axis_anisotropy,
            name="minimum_principal_axis_anisotropy",
        )
        if anisotropy < 1.0:
            raise ValueError("minimum_principal_axis_anisotropy must be at least one")
        object.__setattr__(self, "minimum_principal_axis_anisotropy", anisotropy)

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_relative_trace_error": self.maximum_relative_trace_error,
            "maximum_relative_frobenius_error": self.maximum_relative_frobenius_error,
            "maximum_mean_shift_standard_deviations": self.maximum_mean_shift_standard_deviations,
            "maximum_principal_axis_angle_degrees": self.maximum_principal_axis_angle_degrees,
            "minimum_principal_axis_anisotropy": self.minimum_principal_axis_anisotropy,
            "maximum_query_relative_trace_error": self.maximum_query_relative_trace_error,
            "maximum_query_relative_frobenius_error": self.maximum_query_relative_frobenius_error,
            "maximum_query_mean_shift_standard_deviations": (
                self.maximum_query_mean_shift_standard_deviations
            ),
        }


@dataclass(frozen=True, slots=True)
class GaussianLinearizationAdequacyV1:
    """Replayable source-side decision comparing linear and nonlinear moments."""

    parameterization: str
    parameter_order: tuple[str, ...]
    parameter_dimension: int
    output_shape: tuple[int, int]
    sample_count: int
    batch_size: int
    seed: int
    finite_difference_step: float
    jacobian_validated: bool
    thresholds: LinearizationAdequacyThresholdsV1
    point_diagnostics: tuple[Mapping[str, Any], ...]
    query_diagnostics: Mapping[str, Any] | None
    adequate: bool
    failure_reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    gaussian_linearization_adequacy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parameterization, str) or not self.parameterization:
            raise ValueError("parameterization must be a nonempty string")
        object.__setattr__(self, "parameter_order", tuple(self.parameter_order))
        object.__setattr__(
            self,
            "parameter_dimension",
            _exact_positive_integer(self.parameter_dimension, name="parameter_dimension"),
        )
        if (
            type(self.output_shape) is not tuple
            or len(self.output_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
                for value in self.output_shape
            )
        ):
            raise ValueError("output_shape must contain two positive integers")
        object.__setattr__(self, "output_shape", tuple(int(value) for value in self.output_shape))
        object.__setattr__(
            self,
            "sample_count",
            _exact_positive_integer(self.sample_count, name="sample_count", minimum=2),
        )
        object.__setattr__(
            self,
            "batch_size",
            _exact_positive_integer(self.batch_size, name="batch_size"),
        )
        object.__setattr__(self, "seed", _exact_nonnegative_integer(self.seed, name="seed"))
        object.__setattr__(
            self,
            "finite_difference_step",
            _finite_positive(self.finite_difference_step, name="finite_difference_step"),
        )
        if type(self.jacobian_validated) is not bool:
            raise ValueError("jacobian_validated must be a Boolean")
        if type(self.adequate) is not bool:
            raise ValueError("adequate must be a Boolean")
        object.__setattr__(
            self,
            "point_diagnostics",
            tuple(
                frozen_finite_json_mapping(record, name="point diagnostic")
                for record in self.point_diagnostics
            ),
        )
        query = (
            None
            if self.query_diagnostics is None
            else frozen_finite_json_mapping(self.query_diagnostics, name="query diagnostics")
        )
        object.__setattr__(self, "query_diagnostics", query)
        reasons = tuple(self.failure_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("failure_reasons must contain nonempty strings")
        if len(reasons) != len(set(reasons)):
            raise ValueError("failure_reasons must be unique")
        if self.adequate != (len(reasons) == 0):
            raise ValueError("adequate must agree with failure_reasons")
        object.__setattr__(self, "failure_reasons", reasons)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="linearization metadata"),
        )
        object.__setattr__(
            self,
            "gaussian_linearization_adequacy_id",
            _sha256_json(self._content_dict()),
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": SIM3_LINEARIZATION_SCHEMA,
            "schema_version": SIM3_LINEARIZATION_VERSION,
            "parameterization": self.parameterization,
            "parameter_order": list(self.parameter_order),
            "parameter_dimension": self.parameter_dimension,
            "output_shape": list(self.output_shape),
            "sample_count": self.sample_count,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "finite_difference_step": self.finite_difference_step,
            "jacobian_validated": self.jacobian_validated,
            "thresholds": self.thresholds.to_dict(),
            "point_diagnostics": [plain_json(record) for record in self.point_diagnostics],
            "query_diagnostics": (
                None if self.query_diagnostics is None else plain_json(self.query_diagnostics)
            ),
            "adequate": self.adequate,
            "failure_reasons": list(self.failure_reasons),
            "metadata": plain_json(self.metadata),
            "claim_boundary": SIM3_LINEARIZATION_CLAIM_BOUNDARY,
        }

    def to_dict(self) -> dict[str, object]:
        result = self._content_dict()
        result["gaussian_linearization_adequacy_id"] = self.gaussian_linearization_adequacy_id
        return result


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assess_gaussian_linearization(
    mean_parameters: object,
    covariance: object,
    exact_evaluator: ExactEvaluator,
    *,
    jacobian: object | None = None,
    query_projection: object | None = None,
    thresholds: LinearizationAdequacyThresholdsV1 | None = None,
    sample_count: int = 4096,
    batch_size: int = 256,
    seed: int = 0,
    finite_difference_step: float = 1e-6,
    jacobian_atol: float = 1e-6,
    jacobian_rtol: float = 1e-4,
    parameterization: str = "generic-gaussian",
    parameter_order: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> GaussianLinearizationAdequacyV1:
    """Compare nonlinear Monte Carlo moments with a first-order Gaussian approximation.

    ``exact_evaluator`` maps one parameter vector to a finite ``(items, dimensions)``
    array. Monte Carlo outputs are processed in bounded batches; only first and second
    moments are retained. If ``jacobian`` is supplied it is checked against an
    independent central-difference Jacobian before it is used.
    """

    mean = np.asarray(mean_parameters, dtype=np.float64)
    if mean.ndim != 1 or mean.size < 1 or not np.all(np.isfinite(mean)):
        raise ValueError("mean_parameters must be a finite one-dimensional vector")
    covariance_matrix = _validated_covariance(
        covariance,
        dimension=mean.size,
        name="parameter covariance",
    )
    count = _exact_positive_integer(sample_count, name="sample_count", minimum=2)
    chunk = _exact_positive_integer(batch_size, name="batch_size")
    random_seed = _exact_nonnegative_integer(seed, name="seed")
    step = _finite_positive(finite_difference_step, name="finite_difference_step")
    absolute_tolerance = _finite_nonnegative(jacobian_atol, name="jacobian_atol")
    relative_tolerance = _finite_nonnegative(jacobian_rtol, name="jacobian_rtol")
    threshold_set = thresholds or LinearizationAdequacyThresholdsV1()

    linearized_mean = _normalize_output(exact_evaluator(mean))
    output_shape = linearized_mean.shape
    finite_difference = _finite_difference_jacobian(
        mean,
        exact_evaluator,
        step=step,
        output_shape=output_shape,
    )
    if jacobian is None:
        linearization_jacobian = finite_difference
        jacobian_validated = False
    else:
        linearization_jacobian = _validated_jacobian(
            jacobian,
            output_shape=output_shape,
            parameter_dimension=mean.size,
        )
        if not np.allclose(
            linearization_jacobian,
            finite_difference,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
            maximum_absolute_error = float(
                np.max(np.abs(linearization_jacobian - finite_difference), initial=0.0)
            )
            raise ValueError(
                "supplied Jacobian does not match central differences; "
                f"maximum absolute error={maximum_absolute_error:.6g}"
            )
        jacobian_validated = True

    # Keep only per-item covariance blocks; no dense cross-item covariance is formed.
    linearized_covariance = np.einsum(
        "mip,pq,mjq->mij",
        linearization_jacobian,
        covariance_matrix,
        linearization_jacobian,
    )

    projection = None
    linearized_query_mean = None
    linearized_query_covariance = None
    if query_projection is not None:
        projection = _validate_query_projection(
            query_projection,
            output_dimension=output_shape[0] * output_shape[1],
        )
        flat_jacobian = linearization_jacobian.reshape(-1, mean.size)
        query_jacobian = projection @ flat_jacobian
        linearized_query_mean = projection @ linearized_mean.reshape(-1)
        linearized_query_covariance = query_jacobian @ covariance_matrix @ query_jacobian.T

    sum_output = np.zeros(output_shape, dtype=np.float64)
    sum_output_outer = np.zeros(
        (output_shape[0], output_shape[1], output_shape[1]),
        dtype=np.float64,
    )
    sum_query = None
    sum_query_outer = None
    if projection is not None:
        sum_query = np.zeros(projection.shape[0], dtype=np.float64)
        sum_query_outer = np.zeros((projection.shape[0], projection.shape[0]), dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    processed = 0
    while processed < count:
        current = min(chunk, count - processed)
        parameters = _sample_gaussian(rng, mean, covariance_matrix, current)
        outputs = np.empty((current,) + output_shape, dtype=np.float64)
        for local_index, parameter_vector in enumerate(parameters):
            outputs[local_index] = _normalize_output(
                exact_evaluator(parameter_vector),
                expected_shape=output_shape,
            )
        sum_output += np.sum(outputs, axis=0)
        sum_output_outer += np.einsum("bmi,bmj->mij", outputs, outputs)
        if projection is not None and sum_query is not None and sum_query_outer is not None:
            projected = outputs.reshape(current, -1) @ projection.T
            sum_query += np.sum(projected, axis=0)
            sum_query_outer += projected.T @ projected
        processed += current

    nonlinear_mean = sum_output / count
    nonlinear_covariance = _sample_covariance(sum_output_outer, nonlinear_mean, count)

    point_diagnostics: list[dict[str, object]] = []
    maximum_trace_error = 0.0
    maximum_frobenius_error = 0.0
    maximum_mean_shift = 0.0
    maximum_axis_angle = 0.0
    anisotropic_point_count = 0
    for index in range(output_shape[0]):
        trace_error, frobenius_error = _relative_covariance_metrics(
            nonlinear_covariance[index],
            linearized_covariance[index],
        )
        mean_shift = _mean_shift_standard_deviations(
            nonlinear_mean[index],
            linearized_mean[index],
            nonlinear_covariance[index],
        )
        anisotropy, axis_angle = _principal_axis_metrics(
            nonlinear_covariance[index],
            linearized_covariance[index],
            minimum_anisotropy=threshold_set.minimum_principal_axis_anisotropy,
        )
        if axis_angle is not None:
            anisotropic_point_count += 1
            maximum_axis_angle = max(maximum_axis_angle, axis_angle)
        maximum_trace_error = max(maximum_trace_error, trace_error)
        maximum_frobenius_error = max(maximum_frobenius_error, frobenius_error)
        maximum_mean_shift = max(maximum_mean_shift, mean_shift)
        point_diagnostics.append(
            {
                "item_index": index,
                "relative_trace_error": trace_error,
                "relative_frobenius_error": frobenius_error,
                "mean_shift_standard_deviations": mean_shift,
                "nonlinear_trace": float(np.trace(nonlinear_covariance[index])),
                "linearized_trace": float(np.trace(linearized_covariance[index])),
                "principal_axis_anisotropy": anisotropy,
                "principal_axis_angle_degrees": axis_angle,
            }
        )

    query_diagnostics = None
    query_trace_error = 0.0
    query_frobenius_error = 0.0
    query_mean_shift = 0.0
    if (
        projection is not None
        and sum_query is not None
        and sum_query_outer is not None
        and linearized_query_mean is not None
        and linearized_query_covariance is not None
    ):
        nonlinear_query_mean = sum_query / count
        nonlinear_query_covariance = _sample_covariance(
            sum_query_outer,
            nonlinear_query_mean,
            count,
        )
        query_trace_error, query_frobenius_error = _relative_covariance_metrics(
            nonlinear_query_covariance,
            linearized_query_covariance,
        )
        query_mean_shift = _mean_shift_standard_deviations(
            nonlinear_query_mean,
            linearized_query_mean,
            nonlinear_query_covariance,
        )
        query_diagnostics = {
            "query_dimension": projection.shape[0],
            "relative_trace_error": query_trace_error,
            "relative_frobenius_error": query_frobenius_error,
            "mean_shift_standard_deviations": query_mean_shift,
            "nonlinear_trace": float(np.trace(nonlinear_query_covariance)),
            "linearized_trace": float(np.trace(linearized_query_covariance)),
        }

    failure_reasons: list[str] = []
    if maximum_trace_error > threshold_set.maximum_relative_trace_error:
        failure_reasons.append("point-trace-distortion")
    if maximum_frobenius_error > threshold_set.maximum_relative_frobenius_error:
        failure_reasons.append("point-frobenius-distortion")
    if maximum_mean_shift > threshold_set.maximum_mean_shift_standard_deviations:
        failure_reasons.append("point-mean-shift")
    if (
        anisotropic_point_count > 0
        and maximum_axis_angle > threshold_set.maximum_principal_axis_angle_degrees
    ):
        failure_reasons.append("point-principal-axis-rotation")
    if query_diagnostics is not None:
        if query_trace_error > threshold_set.maximum_query_relative_trace_error:
            failure_reasons.append("query-trace-distortion")
        if query_frobenius_error > threshold_set.maximum_query_relative_frobenius_error:
            failure_reasons.append("query-frobenius-distortion")
        if query_mean_shift > threshold_set.maximum_query_mean_shift_standard_deviations:
            failure_reasons.append("query-mean-shift")

    diagnostic_metadata = dict(metadata or {})
    diagnostic_metadata.update(
        {
            "maximum_point_relative_trace_error": maximum_trace_error,
            "maximum_point_relative_frobenius_error": maximum_frobenius_error,
            "maximum_point_mean_shift_standard_deviations": maximum_mean_shift,
            "maximum_principal_axis_angle_degrees": maximum_axis_angle,
            "anisotropic_point_count": anisotropic_point_count,
        }
    )
    return GaussianLinearizationAdequacyV1(
        parameterization=parameterization,
        parameter_order=tuple(parameter_order),
        parameter_dimension=mean.size,
        output_shape=output_shape,
        sample_count=count,
        batch_size=chunk,
        seed=random_seed,
        finite_difference_step=step,
        jacobian_validated=jacobian_validated,
        thresholds=threshold_set,
        point_diagnostics=tuple(point_diagnostics),
        query_diagnostics=query_diagnostics,
        adequate=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        metadata=diagnostic_metadata,
    )


def assess_sim3_linearization(
    mean_transform: Sim3,
    covariance: object,
    points: object,
    *,
    perturbation_side: PerturbationSide = "left",
    parameter_order: Sequence[str] = _CANONICAL_BLOCKS,
    jacobian: object | None = None,
    query_projection: object | None = None,
    thresholds: LinearizationAdequacyThresholdsV1 | None = None,
    sample_count: int = 4096,
    batch_size: int = 256,
    seed: int = 0,
    finite_difference_step: float = 1e-6,
    jacobian_atol: float = 1e-6,
    jacobian_rtol: float = 1e-4,
    metadata: Mapping[str, Any] | None = None,
) -> GaussianLinearizationAdequacyV1:
    """Certify a local Gaussian perturbation around one ``Sim3`` transform.

    The covariance is interpreted in the declared block order. For ``left``
    perturbations the sampled transform is ``delta.compose(mean_transform)``; for
    ``right`` perturbations it is ``mean_transform.compose(delta)``. The perturbation
    mean is the identity transform, so the exact nonlinear comparison uses the same
    local coordinates as first-order gauge propagation.
    """

    if perturbation_side not in {"left", "right"}:
        raise ValueError("perturbation_side must be 'left' or 'right'")
    order = _validated_order(parameter_order)
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3 or point_array.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must be finite")

    def evaluator(ordered_delta: FloatArray) -> FloatArray:
        canonical_delta = _ordered_to_canonical(ordered_delta, order)
        delta = Sim3.from_vector(canonical_delta)
        transform = (
            delta.compose(mean_transform)
            if perturbation_side == "left"
            else mean_transform.compose(delta)
        )
        return transform.transform_points(point_array)

    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            "mean_transform_vector": mean_transform.as_vector().tolist(),
            "point_count": int(point_array.shape[0]),
            "perturbation_side": perturbation_side,
        }
    )
    return assess_gaussian_linearization(
        np.zeros(7, dtype=np.float64),
        covariance,
        evaluator,
        jacobian=jacobian,
        query_projection=query_projection,
        thresholds=thresholds,
        sample_count=sample_count,
        batch_size=batch_size,
        seed=seed,
        finite_difference_step=finite_difference_step,
        jacobian_atol=jacobian_atol,
        jacobian_rtol=jacobian_rtol,
        parameterization=f"sim3-{perturbation_side}-perturbation",
        parameter_order=order,
        metadata=merged_metadata,
    )


def write_linearization_certificate(
    path: str | Path,
    certificate: GaussianLinearizationAdequacyV1,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one deterministic certificate using the shared race-safe publisher."""

    destination = Path(path)
    content = (
        json.dumps(
            certificate.to_dict(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    atomic_write_text(destination, content, overwrite=overwrite)
    return destination


def _load_npz(path: Path) -> dict[str, FloatArray]:
    allowed = {"mean_transform", "covariance", "points", "jacobian", "query_projection"}
    with np.load(path, allow_pickle=False) as archive:
        unknown = set(archive.files) - allowed
        if unknown:
            raise ValueError(f"unexpected NPZ members: {sorted(unknown)}")
        required = {"mean_transform", "covariance", "points"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"missing NPZ members: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _parse_parameter_order(value: str) -> tuple[ParameterBlock, ...]:
    return _validated_order(tuple(part.strip() for part in value.split(",")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ with mean_transform, covariance, and points")
    parser.add_argument("--output", type=Path, required=True, help="JSON certificate path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-inadequate", action="store_true")
    parser.add_argument("--perturbation-side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--parameter-order",
        default="scale,rotation,translation",
        help="comma-separated permutation of scale, rotation, translation",
    )
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--max-relative-trace-error", type=float, default=0.10)
    parser.add_argument("--max-relative-frobenius-error", type=float, default=0.15)
    parser.add_argument("--max-mean-shift-std", type=float, default=0.25)
    parser.add_argument("--max-principal-axis-angle-deg", type=float, default=10.0)
    parser.add_argument("--min-principal-axis-anisotropy", type=float, default=1.25)
    parser.add_argument("--max-query-relative-trace-error", type=float, default=0.10)
    parser.add_argument("--max-query-relative-frobenius-error", type=float, default=0.15)
    parser.add_argument("--max-query-mean-shift-std", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    arrays = _load_npz(arguments.input)
    mean_vector = np.asarray(arrays["mean_transform"], dtype=np.float64)
    if mean_vector.shape != (7,) or not np.all(np.isfinite(mean_vector)):
        raise ValueError("mean_transform must be a finite canonical Sim(3) vector with shape (7,)")
    thresholds = LinearizationAdequacyThresholdsV1(
        maximum_relative_trace_error=arguments.max_relative_trace_error,
        maximum_relative_frobenius_error=arguments.max_relative_frobenius_error,
        maximum_mean_shift_standard_deviations=arguments.max_mean_shift_std,
        maximum_principal_axis_angle_degrees=arguments.max_principal_axis_angle_deg,
        minimum_principal_axis_anisotropy=arguments.min_principal_axis_anisotropy,
        maximum_query_relative_trace_error=arguments.max_query_relative_trace_error,
        maximum_query_relative_frobenius_error=arguments.max_query_relative_frobenius_error,
        maximum_query_mean_shift_standard_deviations=arguments.max_query_mean_shift_std,
    )
    certificate = assess_sim3_linearization(
        Sim3.from_vector(mean_vector),
        arrays["covariance"],
        arrays["points"],
        perturbation_side=arguments.perturbation_side,
        parameter_order=_parse_parameter_order(arguments.parameter_order),
        jacobian=arrays.get("jacobian"),
        query_projection=arrays.get("query_projection"),
        thresholds=thresholds,
        sample_count=arguments.samples,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        finite_difference_step=arguments.finite_difference_step,
        metadata={"input_npz_sha256": hashlib.sha256(arguments.input.read_bytes()).hexdigest()},
    )
    write_linearization_certificate(
        arguments.output,
        certificate,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(certificate.to_dict(), sort_keys=True, indent=2, allow_nan=False))
    return 2 if arguments.fail_on_inadequate and not certificate.adequate else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GaussianLinearizationAdequacyV1",
    "LinearizationAdequacyThresholdsV1",
    "assess_gaussian_linearization",
    "assess_sim3_linearization",
    "main",
    "write_linearization_certificate",
]
