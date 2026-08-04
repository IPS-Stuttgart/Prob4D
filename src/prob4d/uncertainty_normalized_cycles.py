"""Source-only uncertainty-normalized cycle diagnostics for causal gauge graphs.

The existing raw cycle audit intentionally avoids a chi-square interpretation
because overlapping alignment edges have unknown cross-correlation.  This module
adds a scalar normalization that remains meaningful under unknown correlation:
for each representative point, the root expected squared displacement contributed
by each edge covariance is computed separately and the component root scales are
summed.  Minkowski's inequality makes that sum an upper bound on the root expected
squared difference for arbitrary cross-correlation between the three edge errors.

The resulting dimensionless score is still calibrated empirically on source-only
clean groups.  It is not treated as a chi-square statistic and cannot consume
truth, BayesianPhysTwin innovations, or downstream physical residuals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .alignment import WindowAlignment
from .alignment_cycles import alignment_edge_id
from .composition_jacobian import analytic_sim3_compose_jacobians
from .sim3 import Sim3, skew, so3_log, so3_right_jacobian

FloatArray = NDArray[np.floating]
UNCERTAINTY_NORMALIZED_CYCLE_SEMANTICS: Final = (
    "representative-point-minkowski-normalized-sim3-cycle-v1"
)


@dataclass(frozen=True)
class UncertaintyNormalizedCycleResidual:
    """One direct-versus-two-edge cycle with a conservative source scale."""

    reference_id: str
    middle_id: str
    moving_id: str
    direct_edge_id: str
    first_path_edge_id: str
    second_path_edge_id: str
    representative_displacement: float
    uncertainty_normalized_score: float
    minkowski_uncertainty_scale: float
    direct_uncertainty_scale: float
    first_path_uncertainty_scale: float
    second_path_uncertainty_scale: float
    direct_residual_rms: float
    maximum_path_residual_rms: float
    minimum_correspondences: int
    passed: bool | None = None

    def __post_init__(self) -> None:
        identifiers = (
            self.reference_id,
            self.middle_id,
            self.moving_id,
            self.direct_edge_id,
            self.first_path_edge_id,
            self.second_path_edge_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("normalized cycle identifiers must be nonempty text")
        if len({self.reference_id, self.middle_id, self.moving_id}) != 3:
            raise ValueError("a normalized gauge cycle requires three windows")
        diagnostics = np.asarray(
            [
                self.representative_displacement,
                self.uncertainty_normalized_score,
                self.minkowski_uncertainty_scale,
                self.direct_uncertainty_scale,
                self.first_path_uncertainty_scale,
                self.second_path_uncertainty_scale,
                self.direct_residual_rms,
                self.maximum_path_residual_rms,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("normalized cycle diagnostics must be finite and nonnegative")
        minimum_correspondences = int(self.minimum_correspondences)
        if (
            minimum_correspondences != self.minimum_correspondences
            or minimum_correspondences < 0
        ):
            raise ValueError("minimum_correspondences must be a nonnegative integer")
        if self.passed is not None and not isinstance(self.passed, (bool, np.bool_)):
            raise ValueError("passed must be boolean or None")
        object.__setattr__(self, "minimum_correspondences", minimum_correspondences)
        if self.passed is not None:
            object.__setattr__(self, "passed", bool(self.passed))

    @property
    def cycle_id(self) -> str:
        return f"{self.reference_id}<-{self.middle_id}<-{self.moving_id}"

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["cycle_id"] = self.cycle_id
        return result


@dataclass(frozen=True)
class UncertaintyNormalizedCycleAudit:
    """Aggregate source-only decision over all available directed triangles."""

    alignment_count: int
    window_count: int
    cycle_count: int
    representative_radius: float
    minimum_uncertainty_scale: float
    maximum_normalized_score: float | None
    failed_cycle_count: int
    mean_normalized_score: float
    median_normalized_score: float
    maximum_observed_normalized_score: float
    maximum_observed_representative_displacement: float
    cycles: tuple[UncertaintyNormalizedCycleResidual, ...]

    def __post_init__(self) -> None:
        for name in (
            "alignment_count",
            "window_count",
            "cycle_count",
            "failed_cycle_count",
        ):
            original = getattr(self, name)
            value = int(original)
            if value != original or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, value)
        cycles = tuple(self.cycles)
        if len(cycles) != self.cycle_count:
            raise ValueError("cycle_count differs from normalized cycle records")
        if self.failed_cycle_count > self.cycle_count:
            raise ValueError("failed_cycle_count cannot exceed cycle_count")
        radius = float(self.representative_radius)
        floor = float(self.minimum_uncertainty_scale)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("representative_radius must be finite and positive")
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("minimum_uncertainty_scale must be finite and positive")
        threshold = self.maximum_normalized_score
        if threshold is not None and (not np.isfinite(threshold) or threshold <= 0.0):
            raise ValueError("maximum_normalized_score must be positive when supplied")
        diagnostics = np.asarray(
            [
                self.mean_normalized_score,
                self.median_normalized_score,
                self.maximum_observed_normalized_score,
                self.maximum_observed_representative_displacement,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("aggregate normalized diagnostics must be finite and nonnegative")
        expected_failed = sum(cycle.passed is False for cycle in cycles)
        if expected_failed != self.failed_cycle_count:
            raise ValueError("failed_cycle_count differs from normalized pass flags")
        if threshold is None and any(cycle.passed is not None for cycle in cycles):
            raise ValueError("cycles cannot carry pass flags without a threshold")
        if threshold is not None and any(cycle.passed is None for cycle in cycles):
            raise ValueError("every cycle must carry a pass flag with a threshold")
        ordered = tuple(sorted(cycles, key=lambda item: item.cycle_id))
        if cycles != ordered:
            raise ValueError("normalized cycles must use canonical cycle ordering")
        object.__setattr__(self, "representative_radius", radius)
        object.__setattr__(self, "minimum_uncertainty_scale", floor)
        object.__setattr__(self, "cycles", cycles)

    @property
    def passed(self) -> bool | None:
        if self.maximum_normalized_score is None:
            return None
        return self.failed_cycle_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": UNCERTAINTY_NORMALIZED_CYCLE_SEMANTICS,
            "statistical_interpretation": (
                "empirically calibrated source-only score; the per-point denominator "
                "is a Minkowski upper bound for arbitrary edge-error correlation, not "
                "a chi-square covariance"
            ),
            "alignment_count": self.alignment_count,
            "window_count": self.window_count,
            "cycle_count": self.cycle_count,
            "representative_radius": self.representative_radius,
            "minimum_uncertainty_scale": self.minimum_uncertainty_scale,
            "maximum_normalized_score": self.maximum_normalized_score,
            "failed_cycle_count": self.failed_cycle_count,
            "passed": self.passed,
            "mean_normalized_score": self.mean_normalized_score,
            "median_normalized_score": self.median_normalized_score,
            "maximum_observed_normalized_score": (
                self.maximum_observed_normalized_score
            ),
            "maximum_observed_representative_displacement": (
                self.maximum_observed_representative_displacement
            ),
            "cycles": [cycle.to_dict() for cycle in self.cycles],
        }


def _representative_points(radius: float) -> FloatArray:
    axes = radius * np.eye(3, dtype=np.float64)
    return np.concatenate((np.zeros((1, 3)), axes, -axes), axis=0)


def _validated_covariance(value: FloatArray, *, name: str) -> FloatArray:
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 7 x 7 covariance")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    tolerance = 1e-10 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if float(np.min(eigenvalues)) < 0.0:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors * np.maximum(eigenvalues, 0.0)
        ) @ eigenvectors.T
    return covariance


def _point_jacobian(transform: Sim3, point: FloatArray) -> FloatArray:
    point = np.asarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("representative point must contain three finite values")
    rotation_vector = so3_log(transform.rotation)
    if np.pi - float(np.linalg.norm(rotation_vector)) <= 1e-7:
        raise ValueError("point Jacobian is undefined at the SO(3) log branch cut")
    rotated_scaled = transform.scale * transform.rotation @ point
    jacobian = np.zeros((3, 7), dtype=np.float64)
    jacobian[:, 0] = rotated_scaled
    jacobian[:, 1:4] = (
        -transform.scale
        * transform.rotation
        @ skew(point)
        @ so3_right_jacobian(rotation_vector)
    )
    jacobian[:, 4:7] = np.eye(3)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("representative-point Jacobian is nonfinite")
    return jacobian


def _root_expected_squared_norm(
    jacobian: FloatArray,
    covariance: FloatArray,
) -> float:
    propagated = jacobian @ covariance @ jacobian.T
    variance = float(np.trace(0.5 * (propagated + propagated.T)))
    tolerance = 1e-12 * max(float(np.linalg.norm(propagated, ord="fro")), 1.0)
    if variance < -tolerance:
        raise ValueError("propagated representative-point variance is negative")
    return math_sqrt_nonnegative(variance)


def math_sqrt_nonnegative(value: float) -> float:
    """Return a finite square root after clipping roundoff-scale negativity."""

    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("uncertainty scale must be finite")
    return float(np.sqrt(max(numeric, 0.0)))


def _cycle_residual(
    direct: WindowAlignment,
    first: WindowAlignment,
    second: WindowAlignment,
    *,
    representative_radius: float,
    minimum_uncertainty_scale: float,
    threshold: float | None,
) -> UncertaintyNormalizedCycleResidual:
    direct_transform = direct.result.transform
    first_transform = first.result.transform
    second_transform = second.result.transform
    composed = first_transform.compose(second_transform)
    compose_first, compose_second = analytic_sim3_compose_jacobians(
        first_transform,
        second_transform,
    )
    direct_covariance = _validated_covariance(
        direct.result.covariance,
        name=f"{alignment_edge_id(direct)} covariance",
    )
    first_covariance = _validated_covariance(
        first.result.covariance,
        name=f"{alignment_edge_id(first)} covariance",
    )
    second_covariance = _validated_covariance(
        second.result.covariance,
        name=f"{alignment_edge_id(second)} covariance",
    )

    squared_displacements: list[float] = []
    squared_normalized: list[float] = []
    direct_scales: list[float] = []
    first_scales: list[float] = []
    second_scales: list[float] = []
    total_scales: list[float] = []
    for point in _representative_points(representative_radius):
        direct_point = direct_transform.transform_points(point[None, :])[0]
        composed_point = composed.transform_points(point[None, :])[0]
        displacement = float(np.linalg.norm(direct_point - composed_point))
        direct_jacobian = _point_jacobian(direct_transform, point)
        composed_point_jacobian = _point_jacobian(composed, point)
        first_jacobian = composed_point_jacobian @ compose_first
        second_jacobian = composed_point_jacobian @ compose_second
        direct_scale = _root_expected_squared_norm(
            direct_jacobian,
            direct_covariance,
        )
        first_scale = _root_expected_squared_norm(
            first_jacobian,
            first_covariance,
        )
        second_scale = _root_expected_squared_norm(
            second_jacobian,
            second_covariance,
        )
        total_scale = max(
            direct_scale + first_scale + second_scale,
            minimum_uncertainty_scale,
        )
        squared_displacements.append(displacement**2)
        squared_normalized.append((displacement / total_scale) ** 2)
        direct_scales.append(direct_scale)
        first_scales.append(first_scale)
        second_scales.append(second_scale)
        total_scales.append(total_scale)

    representative_displacement = math_sqrt_nonnegative(
        float(np.mean(squared_displacements))
    )
    normalized_score = math_sqrt_nonnegative(float(np.mean(squared_normalized)))
    passed = None if threshold is None else normalized_score <= threshold
    return UncertaintyNormalizedCycleResidual(
        reference_id=direct.reference_id,
        middle_id=first.moving_id,
        moving_id=direct.moving_id,
        direct_edge_id=alignment_edge_id(direct),
        first_path_edge_id=alignment_edge_id(first),
        second_path_edge_id=alignment_edge_id(second),
        representative_displacement=representative_displacement,
        uncertainty_normalized_score=normalized_score,
        minkowski_uncertainty_scale=math_sqrt_nonnegative(
            float(np.mean(np.square(total_scales)))
        ),
        direct_uncertainty_scale=math_sqrt_nonnegative(
            float(np.mean(np.square(direct_scales)))
        ),
        first_path_uncertainty_scale=math_sqrt_nonnegative(
            float(np.mean(np.square(first_scales)))
        ),
        second_path_uncertainty_scale=math_sqrt_nonnegative(
            float(np.mean(np.square(second_scales)))
        ),
        direct_residual_rms=float(direct.result.residual_rms),
        maximum_path_residual_rms=max(
            float(first.result.residual_rms),
            float(second.result.residual_rms),
        ),
        minimum_correspondences=min(
            direct.result.num_correspondences,
            first.result.num_correspondences,
            second.result.num_correspondences,
        ),
        passed=passed,
    )


def audit_uncertainty_normalized_alignment_cycles(
    alignments: list[WindowAlignment] | tuple[WindowAlignment, ...],
    *,
    representative_radius: float = 1.0,
    minimum_uncertainty_scale: float = 1e-12,
    maximum_normalized_score: float | None = None,
) -> UncertaintyNormalizedCycleAudit:
    """Audit direct edges against two-edge paths using a source uncertainty scale."""

    radius = float(representative_radius)
    floor = float(minimum_uncertainty_scale)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("representative_radius must be finite and positive")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("minimum_uncertainty_scale must be finite and positive")
    threshold = maximum_normalized_score
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("maximum_normalized_score must be positive when supplied")

    edges: dict[tuple[str, str], WindowAlignment] = {}
    windows: set[str] = set()
    for alignment in alignments:
        key = (alignment.reference_id, alignment.moving_id)
        if key in edges:
            raise ValueError(
                f"duplicate directed alignment edge {alignment_edge_id(alignment)!r}"
            )
        edges[key] = alignment
        windows.update(key)

    cycles: list[UncertaintyNormalizedCycleResidual] = []
    for reference_id, moving_id in sorted(edges):
        direct = edges[(reference_id, moving_id)]
        for middle_id in sorted(windows - {reference_id, moving_id}):
            first = edges.get((reference_id, middle_id))
            second = edges.get((middle_id, moving_id))
            if first is None or second is None:
                continue
            cycles.append(
                _cycle_residual(
                    direct,
                    first,
                    second,
                    representative_radius=radius,
                    minimum_uncertainty_scale=floor,
                    threshold=threshold,
                )
            )
    cycles.sort(key=lambda item: item.cycle_id)
    scores = np.asarray(
        [cycle.uncertainty_normalized_score for cycle in cycles],
        dtype=np.float64,
    )
    displacements = np.asarray(
        [cycle.representative_displacement for cycle in cycles],
        dtype=np.float64,
    )

    def mean_or_zero(values: FloatArray) -> float:
        return 0.0 if values.size == 0 else float(np.mean(values))

    def median_or_zero(values: FloatArray) -> float:
        return 0.0 if values.size == 0 else float(np.median(values))

    def max_or_zero(values: FloatArray) -> float:
        return 0.0 if values.size == 0 else float(np.max(values))

    return UncertaintyNormalizedCycleAudit(
        alignment_count=len(edges),
        window_count=len(windows),
        cycle_count=len(cycles),
        representative_radius=radius,
        minimum_uncertainty_scale=floor,
        maximum_normalized_score=threshold,
        failed_cycle_count=sum(cycle.passed is False for cycle in cycles),
        mean_normalized_score=mean_or_zero(scores),
        median_normalized_score=median_or_zero(scores),
        maximum_observed_normalized_score=max_or_zero(scores),
        maximum_observed_representative_displacement=max_or_zero(displacements),
        cycles=tuple(cycles),
    )


__all__ = [
    "UNCERTAINTY_NORMALIZED_CYCLE_SEMANTICS",
    "UncertaintyNormalizedCycleAudit",
    "UncertaintyNormalizedCycleResidual",
    "audit_uncertainty_normalized_alignment_cycles",
]
