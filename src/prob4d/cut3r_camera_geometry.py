"""Translation-invariant camera geometry for CUT3R prediction windows.

CUT3R stores points in one sequence-local common frame.  A viewing ray therefore
must originate at the frame's camera centre, not at the arbitrary common-frame
origin.  This module recovers each camera centre from the common-frame points and
unit rays and uses the resulting camera-relative ranges for depth-aware
uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ._scientific_scalars import require_finite_real, require_genuine_integer
from .data import PredictionWindow
from .uncertainty import (
    DepthDisagreementModel,
    DisagreementEvidence,
    StructuredCovariance,
)

CUT3R_CAMERA_GEOMETRY_SEMANTICS: Final = (
    "common-frame-points-plus-camera-origin-unit-rays-v1"
)
CUT3R_CAMERA_GEOMETRY_CLAIM_BOUNDARY: Final = (
    "Camera-centre recovery and translation-invariant range propagation establish "
    "geometric semantics only. They do not establish CUT3R provider competence, "
    "uncertainty calibration, BayesianPhysTwin benefit, Causal4D intervention "
    "benefit, deployment safety, or state of the art."
)


def _immutable_float64(value: object) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    payload = array.tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class CameraRelativeGeometryV1:
    """Recovered camera centres and ranges for one prediction window."""

    camera_centers_local: np.ndarray
    range_local: np.ndarray
    frame_condition_numbers: tuple[float, ...]
    frame_max_perpendicular_residual_local: tuple[float, ...]

    def __post_init__(self) -> None:
        centers = np.asarray(self.camera_centers_local, dtype=np.float64)
        ranges = np.asarray(self.range_local, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
            raise ValueError("camera_centers_local must have nonempty shape (T, 3)")
        if ranges.ndim != 3 or ranges.shape[0] != centers.shape[0]:
            raise ValueError("range_local must have shape (T, H, W)")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(ranges)):
            raise ValueError("camera-relative geometry must be finite")
        if np.any(ranges < 0.0):
            raise ValueError("camera-relative ranges must be non-negative")
        frame_count = centers.shape[0]
        if len(self.frame_condition_numbers) != frame_count or len(
            self.frame_max_perpendicular_residual_local
        ) != frame_count:
            raise ValueError("camera-relative frame diagnostics changed length")
        conditions = tuple(
            require_finite_real(
                value,
                name=f"frame_condition_numbers[{index}]",
                minimum=1.0,
            )
            for index, value in enumerate(self.frame_condition_numbers)
        )
        residuals = tuple(
            require_finite_real(
                value,
                name=f"frame_max_perpendicular_residual_local[{index}]",
                minimum=0.0,
            )
            for index, value in enumerate(
                self.frame_max_perpendicular_residual_local
            )
        )
        object.__setattr__(self, "camera_centers_local", _immutable_float64(centers))
        object.__setattr__(self, "range_local", _immutable_float64(ranges))
        object.__setattr__(self, "frame_condition_numbers", conditions)
        object.__setattr__(
            self,
            "frame_max_perpendicular_residual_local",
            residuals,
        )

    @property
    def frame_count(self) -> int:
        return int(self.camera_centers_local.shape[0])

    @property
    def maximum_condition_number(self) -> float:
        return max(self.frame_condition_numbers)

    @property
    def maximum_perpendicular_residual_local(self) -> float:
        return max(self.frame_max_perpendicular_residual_local)

    def summary(self) -> dict[str, object]:
        return {
            "semantics": CUT3R_CAMERA_GEOMETRY_SEMANTICS,
            "frame_count": self.frame_count,
            "maximum_condition_number": self.maximum_condition_number,
            "maximum_perpendicular_residual_local": (
                self.maximum_perpendicular_residual_local
            ),
            "claim_boundary": CUT3R_CAMERA_GEOMETRY_CLAIM_BOUNDARY,
        }


def _recover_frame_camera_geometry(
    points: np.ndarray,
    rays: np.ndarray,
    *,
    absolute_tolerance_local: float,
    relative_tolerance: float,
    maximum_condition_number: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if points.ndim != 2 or points.shape[1] != 3 or rays.shape != points.shape:
        raise ValueError("camera geometry rows must have matching shape (N, 3)")
    if points.shape[0] < 3:
        raise ValueError("camera-centre recovery requires at least three valid rays")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(rays)):
        raise ValueError("camera geometry rows must be finite")

    ray_norms = np.linalg.norm(rays, axis=1)
    if not np.allclose(ray_norms, 1.0, atol=1e-7, rtol=1e-6):
        raise ValueError("camera geometry requires normalized viewing rays")
    unit_rays = rays / ray_norms[:, None]

    row_count = float(points.shape[0])
    normal_matrix = row_count * np.eye(3) - unit_rays.T @ unit_rays
    normal_matrix = 0.5 * (normal_matrix + normal_matrix.T)
    eigenvalues = np.linalg.eigvalsh(normal_matrix)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if largest <= 0.0 or smallest <= 0.0:
        raise ValueError("camera rays do not identify a finite camera centre")
    condition_number = largest / smallest
    if condition_number > maximum_condition_number:
        raise ValueError(
            "camera-centre recovery is ill-conditioned: "
            f"condition_number={condition_number:.6g}, "
            f"maximum={maximum_condition_number:.6g}"
        )

    # Solve in coordinates centred on the observed points.  The exact least-squares
    # problem is translation equivariant; centring also avoids subtracting large
    # common-frame offsets when the sequence origin is far from the camera.
    point_reference = np.mean(points, axis=0)
    centered_points = points - point_reference
    ray_point_projection = np.einsum(
        "ni,ni->n",
        unit_rays,
        centered_points,
        optimize=True,
    )
    right_hand_side = np.sum(
        centered_points - unit_rays * ray_point_projection[:, None],
        axis=0,
    )
    try:
        camera_offset = np.linalg.solve(normal_matrix, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera centre could not be recovered from points and rays") from error
    camera_center = point_reference + camera_offset

    displacement = points - camera_center
    ranges = np.einsum(
        "ni,ni->n",
        displacement,
        unit_rays,
        optimize=True,
    )
    range_scale = max(float(np.max(np.abs(ranges))), np.finfo(np.float64).tiny)
    positive_tolerance = absolute_tolerance_local + relative_tolerance * range_scale
    if np.any(ranges <= -positive_tolerance):
        raise ValueError("camera rays point away from one or more valid CUT3R points")
    ranges = np.maximum(ranges, 0.0)

    perpendicular = displacement - ranges[:, None] * unit_rays
    maximum_residual = float(np.max(np.linalg.norm(perpendicular, axis=1)))
    allowed_residual = absolute_tolerance_local + relative_tolerance * range_scale
    if maximum_residual > allowed_residual:
        raise ValueError(
            "common-frame points and camera rays are geometrically inconsistent: "
            f"maximum_perpendicular_residual={maximum_residual:.6g}, "
            f"allowed={allowed_residual:.6g}"
        )
    return camera_center, ranges, condition_number, maximum_residual


def recover_camera_relative_geometry(
    window: PredictionWindow,
    *,
    absolute_tolerance_local: float = 1e-5,
    relative_tolerance: float = 1e-5,
    maximum_condition_number: float = 1e12,
    minimum_valid_rays_per_frame: int = 3,
) -> CameraRelativeGeometryV1:
    """Recover camera centres and ranges without using the common-frame origin."""

    absolute_tolerance = require_finite_real(
        absolute_tolerance_local,
        name="absolute_tolerance_local",
        minimum=0.0,
    )
    relative = require_finite_real(
        relative_tolerance,
        name="relative_tolerance",
        minimum=0.0,
    )
    maximum_condition = require_finite_real(
        maximum_condition_number,
        name="maximum_condition_number",
        minimum=1.0,
    )
    minimum_rays = require_genuine_integer(
        minimum_valid_rays_per_frame,
        name="minimum_valid_rays_per_frame",
        minimum=3,
    )
    if window.ray_directions is None:
        raise ValueError(
            "camera-relative depth requires explicit camera-origin ray_directions"
        )

    points = np.asarray(window.point_map, dtype=np.float64)
    valid = np.asarray(window.valid_mask, dtype=bool)
    rays = np.asarray(window.rays(dtype=np.float64), dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("prediction window points must have shape (T, H, W, 3)")
    if valid.shape != points.shape[:-1] or rays.shape != points.shape:
        raise ValueError("prediction window geometry arrays changed shape")

    centers = np.empty((points.shape[0], 3), dtype=np.float64)
    ranges = np.zeros(valid.shape, dtype=np.float64)
    condition_numbers: list[float] = []
    maximum_residuals: list[float] = []
    for frame_index in range(points.shape[0]):
        frame_valid = valid[frame_index]
        valid_count = int(np.count_nonzero(frame_valid))
        if valid_count < minimum_rays:
            raise ValueError(
                "camera-centre recovery has insufficient valid rays in frame "
                f"{frame_index}: count={valid_count}, minimum={minimum_rays}"
            )
        (
            center,
            frame_ranges,
            condition_number,
            maximum_residual,
        ) = _recover_frame_camera_geometry(
            points[frame_index][frame_valid],
            rays[frame_index][frame_valid],
            absolute_tolerance_local=absolute_tolerance,
            relative_tolerance=relative,
            maximum_condition_number=maximum_condition,
        )
        centers[frame_index] = center
        ranges[frame_index][frame_valid] = frame_ranges
        condition_numbers.append(condition_number)
        maximum_residuals.append(maximum_residual)

    return CameraRelativeGeometryV1(
        camera_centers_local=centers,
        range_local=ranges,
        frame_condition_numbers=tuple(condition_numbers),
        frame_max_perpendicular_residual_local=tuple(maximum_residuals),
    )


@dataclass(frozen=True)
class CameraRelativeDepthDisagreementModel(DepthDisagreementModel):
    """Depth-disagreement model using recovered camera-relative range."""

    geometry_absolute_tolerance_local: float = 1e-5
    geometry_relative_tolerance: float = 1e-5
    geometry_maximum_condition_number: float = 1e12
    geometry_minimum_valid_rays_per_frame: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "geometry_absolute_tolerance_local",
            require_finite_real(
                self.geometry_absolute_tolerance_local,
                name="geometry_absolute_tolerance_local",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "geometry_relative_tolerance",
            require_finite_real(
                self.geometry_relative_tolerance,
                name="geometry_relative_tolerance",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "geometry_maximum_condition_number",
            require_finite_real(
                self.geometry_maximum_condition_number,
                name="geometry_maximum_condition_number",
                minimum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "geometry_minimum_valid_rays_per_frame",
            require_genuine_integer(
                self.geometry_minimum_valid_rays_per_frame,
                name="geometry_minimum_valid_rays_per_frame",
                minimum=3,
            ),
        )

    def recover_geometry(
        self,
        window: PredictionWindow,
    ) -> CameraRelativeGeometryV1:
        return recover_camera_relative_geometry(
            window,
            absolute_tolerance_local=self.geometry_absolute_tolerance_local,
            relative_tolerance=self.geometry_relative_tolerance,
            maximum_condition_number=self.geometry_maximum_condition_number,
            minimum_valid_rays_per_frame=(
                self.geometry_minimum_valid_rays_per_frame
            ),
        )

    def predict(
        self,
        window: PredictionWindow,
        evidence: DisagreementEvidence | None = None,
    ) -> StructuredCovariance:
        geometry = self.recover_geometry(window)
        depth_squared = np.square(geometry.range_local)
        parallel = self.parallel_floor + self.parallel_depth_coefficient * depth_squared
        lateral = self.lateral_floor + self.lateral_depth_coefficient * depth_squared
        if evidence is not None:
            if evidence.count.shape != window.shape:
                raise ValueError("disagreement evidence shape does not match window")
            parallel += 0.5 * self.disagreement_gain * evidence.parallel_mean
            lateral += 0.5 * self.disagreement_gain * evidence.lateral_mean
        parallel *= self.parallel_scale
        lateral *= self.lateral_scale
        return StructuredCovariance(
            ray_directions=window.rays(),
            parallel_variance=np.maximum(parallel, np.finfo(np.float64).eps),
            lateral_variance=np.maximum(lateral, np.finfo(np.float64).eps),
        )


__all__ = [
    "CUT3R_CAMERA_GEOMETRY_CLAIM_BOUNDARY",
    "CUT3R_CAMERA_GEOMETRY_SEMANTICS",
    "CameraRelativeDepthDisagreementModel",
    "CameraRelativeGeometryV1",
    "recover_camera_relative_geometry",
]
