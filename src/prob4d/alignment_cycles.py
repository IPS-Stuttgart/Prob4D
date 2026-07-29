"""Deterministic cycle-consistency audits for overlapping-window gauges.

Prob4D's strict observation provider uses a causal spanning tree so correlated
alignment edges are not multiplied as if independent. Non-tree edges still
contain useful diagnostic information: a direct alignment can be compared with
the transform obtained through one intermediate window. This module reports
that disagreement without assigning a chi-square interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .alignment import WindowAlignment


@dataclass(frozen=True)
class AlignmentCycleResidual:
    """One directed triangle ``reference <- middle <- moving``."""

    reference_id: str
    middle_id: str
    moving_id: str
    direct_edge_id: str
    first_path_edge_id: str
    second_path_edge_id: str
    log_scale_error: float
    rotation_error_rad: float
    translation_error: float
    representative_displacement: float
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
        if any(not str(value) for value in identifiers):
            raise ValueError("cycle identifiers must not be empty")
        if len({self.reference_id, self.middle_id, self.moving_id}) != 3:
            raise ValueError("a gauge cycle requires three distinct windows")
        diagnostics = np.asarray(
            [
                self.log_scale_error,
                self.rotation_error_rad,
                self.translation_error,
                self.representative_displacement,
                self.direct_residual_rms,
                self.maximum_path_residual_rms,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("cycle diagnostics must be finite and non-negative")
        minimum_correspondences = int(self.minimum_correspondences)
        if (
            minimum_correspondences != self.minimum_correspondences
            or minimum_correspondences < 0
        ):
            raise ValueError("minimum_correspondences must be a non-negative integer")
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
class AlignmentCycleAudit:
    """Aggregate audit over every available directed three-window cycle."""

    alignment_count: int
    window_count: int
    cycle_count: int
    representative_radius: float
    maximum_representative_displacement: float | None
    failed_cycle_count: int
    mean_representative_displacement: float
    median_representative_displacement: float
    maximum_observed_representative_displacement: float
    maximum_rotation_error_rad: float
    maximum_translation_error: float
    maximum_log_scale_error: float
    cycles: tuple[AlignmentCycleResidual, ...]

    def __post_init__(self) -> None:
        integer_fields = (
            "alignment_count",
            "window_count",
            "cycle_count",
            "failed_cycle_count",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value != getattr(self, name) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, value)
        cycles = tuple(self.cycles)
        if len(cycles) != self.cycle_count:
            raise ValueError("cycle_count differs from the cycle records")
        if self.failed_cycle_count > self.cycle_count:
            raise ValueError("failed_cycle_count cannot exceed cycle_count")
        if not np.isfinite(self.representative_radius) or self.representative_radius <= 0.0:
            raise ValueError("representative_radius must be finite and positive")
        threshold = self.maximum_representative_displacement
        if threshold is not None and (not np.isfinite(threshold) or threshold <= 0.0):
            raise ValueError(
                "maximum_representative_displacement must be positive when supplied"
            )
        diagnostics = np.asarray(
            [
                self.mean_representative_displacement,
                self.median_representative_displacement,
                self.maximum_observed_representative_displacement,
                self.maximum_rotation_error_rad,
                self.maximum_translation_error,
                self.maximum_log_scale_error,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(diagnostics)) or np.any(diagnostics < 0.0):
            raise ValueError("aggregate cycle diagnostics must be finite and non-negative")
        expected_failed = sum(cycle.passed is False for cycle in cycles)
        if expected_failed != self.failed_cycle_count:
            raise ValueError("failed_cycle_count differs from cycle pass flags")
        if threshold is None and any(cycle.passed is not None for cycle in cycles):
            raise ValueError("cycles cannot carry pass flags without a threshold")
        if threshold is not None and any(cycle.passed is None for cycle in cycles):
            raise ValueError("every cycle must carry a pass flag when a threshold is used")
        ordered = tuple(sorted(cycles, key=lambda item: item.cycle_id))
        if cycles != ordered:
            raise ValueError("cycles must use canonical cycle-ID ordering")
        object.__setattr__(self, "cycles", cycles)

    @property
    def passed(self) -> bool | None:
        if self.maximum_representative_displacement is None:
            return None
        return self.failed_cycle_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "semantics": "direct-versus-two-edge-sim3-cycle-displacement-v1",
            "statistical_interpretation": "unnormalized diagnostic; edge dependence unknown",
            "alignment_count": self.alignment_count,
            "window_count": self.window_count,
            "cycle_count": self.cycle_count,
            "representative_radius": self.representative_radius,
            "maximum_representative_displacement": (
                self.maximum_representative_displacement
            ),
            "failed_cycle_count": self.failed_cycle_count,
            "passed": self.passed,
            "mean_representative_displacement": self.mean_representative_displacement,
            "median_representative_displacement": self.median_representative_displacement,
            "maximum_observed_representative_displacement": (
                self.maximum_observed_representative_displacement
            ),
            "maximum_rotation_error_rad": self.maximum_rotation_error_rad,
            "maximum_translation_error": self.maximum_translation_error,
            "maximum_log_scale_error": self.maximum_log_scale_error,
            "cycles": [cycle.to_dict() for cycle in self.cycles],
        }


def alignment_edge_id(alignment: WindowAlignment) -> str:
    """Return the canonical directed edge identity ``reference<-moving``."""

    return f"{alignment.reference_id}<-{alignment.moving_id}"


def _representative_points(radius: float) -> np.ndarray:
    axes = radius * np.eye(3, dtype=np.float64)
    return np.concatenate((np.zeros((1, 3)), axes, -axes), axis=0)


def _cycle_residual(
    direct: WindowAlignment,
    first: WindowAlignment,
    second: WindowAlignment,
    *,
    representative_radius: float,
    threshold: float | None,
) -> AlignmentCycleResidual:
    direct_transform = direct.result.transform
    composed = first.result.transform.compose(second.result.transform)
    residual = direct_transform.inverse().compose(composed).as_vector()
    points = _representative_points(representative_radius)
    direct_points = direct_transform.transform_points(points)
    composed_points = composed.transform_points(points)
    displacement = float(
        np.sqrt(np.mean(np.sum((direct_points - composed_points) ** 2, axis=1)))
    )
    passed = None if threshold is None else displacement <= threshold
    return AlignmentCycleResidual(
        reference_id=direct.reference_id,
        middle_id=first.moving_id,
        moving_id=direct.moving_id,
        direct_edge_id=alignment_edge_id(direct),
        first_path_edge_id=alignment_edge_id(first),
        second_path_edge_id=alignment_edge_id(second),
        log_scale_error=abs(float(residual[0])),
        rotation_error_rad=float(np.linalg.norm(residual[1:4])),
        translation_error=float(np.linalg.norm(residual[4:7])),
        representative_displacement=displacement,
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


def audit_alignment_cycles(
    alignments: list[WindowAlignment] | tuple[WindowAlignment, ...],
    *,
    representative_radius: float = 1.0,
    maximum_representative_displacement: float | None = None,
) -> AlignmentCycleAudit:
    """Compare direct edges with every available directed two-edge path.

    The audit intentionally reports raw transform disagreement. It does not
    whiten the residual with edge covariances because overlapping-window edges
    generally share frames, pixels, and a model backbone, and their unavailable
    cross-covariance prevents a valid chi-square interpretation.
    """

    radius = float(representative_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("representative_radius must be finite and positive")
    threshold = maximum_representative_displacement
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "maximum_representative_displacement must be positive when supplied"
            )

    edges: dict[tuple[str, str], WindowAlignment] = {}
    windows: set[str] = set()
    for alignment in alignments:
        key = (alignment.reference_id, alignment.moving_id)
        if key in edges:
            raise ValueError(f"duplicate directed alignment edge {alignment_edge_id(alignment)!r}")
        edges[key] = alignment
        windows.update(key)

    cycles: list[AlignmentCycleResidual] = []
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
                    threshold=threshold,
                )
            )
    cycles.sort(key=lambda item: item.cycle_id)
    displacements = np.asarray(
        [cycle.representative_displacement for cycle in cycles],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [cycle.rotation_error_rad for cycle in cycles],
        dtype=np.float64,
    )
    translations = np.asarray(
        [cycle.translation_error for cycle in cycles],
        dtype=np.float64,
    )
    scales = np.asarray(
        [cycle.log_scale_error for cycle in cycles],
        dtype=np.float64,
    )

    def mean_or_zero(values: np.ndarray) -> float:
        return 0.0 if not len(values) else float(np.mean(values))

    def median_or_zero(values: np.ndarray) -> float:
        return 0.0 if not len(values) else float(np.median(values))

    def max_or_zero(values: np.ndarray) -> float:
        return 0.0 if not len(values) else float(np.max(values))

    return AlignmentCycleAudit(
        alignment_count=len(edges),
        window_count=len(windows),
        cycle_count=len(cycles),
        representative_radius=radius,
        maximum_representative_displacement=threshold,
        failed_cycle_count=sum(cycle.passed is False for cycle in cycles),
        mean_representative_displacement=mean_or_zero(displacements),
        median_representative_displacement=median_or_zero(displacements),
        maximum_observed_representative_displacement=max_or_zero(displacements),
        maximum_rotation_error_rad=max_or_zero(rotations),
        maximum_translation_error=max_or_zero(translations),
        maximum_log_scale_error=max_or_zero(scales),
        cycles=tuple(cycles),
    )


__all__ = [
    "AlignmentCycleAudit",
    "AlignmentCycleResidual",
    "alignment_edge_id",
    "audit_alignment_cycles",
]
