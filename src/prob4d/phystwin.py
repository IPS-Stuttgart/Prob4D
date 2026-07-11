"""Metric RGB-D and correspondence helpers for PhysTwin experiment zero."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .metrics import TruthSequence

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CoverResizeCrop:
    """MotionCrafter's cover-resize and center-crop image geometry."""

    source_height: int
    source_width: int
    target_height: int
    target_width: int
    resized_height: int
    resized_width: int
    crop_row: int
    crop_column: int

    @classmethod
    def from_shapes(
        cls,
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int,
    ) -> CoverResizeCrop:
        values = (source_height, source_width, target_height, target_width)
        if any(value < 1 for value in values):
            raise ValueError("source and target image dimensions must be positive")
        scale = max(target_height / source_height, target_width / source_width)
        resized_height = int(round(source_height * scale))
        resized_width = int(round(source_width * scale))
        return cls(
            source_height=source_height,
            source_width=source_width,
            target_height=target_height,
            target_width=target_width,
            resized_height=resized_height,
            resized_width=resized_width,
            crop_row=(resized_height - target_height) // 2,
            crop_column=(resized_width - target_width) // 2,
        )

    def target_to_source(self, target_xy: FloatArray) -> FloatArray:
        """Map target pixel centers to continuous source pixel coordinates."""

        coordinates = np.asarray(target_xy, dtype=np.float64)
        if coordinates.shape[-1] != 2:
            raise ValueError("pixel coordinates must end in (x, y)")
        result = np.empty_like(coordinates)
        result[..., 0] = (
            (coordinates[..., 0] + self.crop_column + 0.5)
            * self.source_width
            / self.resized_width
            - 0.5
        )
        result[..., 1] = (
            (coordinates[..., 1] + self.crop_row + 0.5)
            * self.source_height
            / self.resized_height
            - 0.5
        )
        return result

    def source_to_target(self, source_xy: FloatArray) -> FloatArray:
        """Map source pixel centers into MotionCrafter's cropped image."""

        coordinates = np.asarray(source_xy, dtype=np.float64)
        if coordinates.shape[-1] != 2:
            raise ValueError("pixel coordinates must end in (x, y)")
        result = np.empty_like(coordinates)
        result[..., 0] = (
            (coordinates[..., 0] + 0.5) * self.resized_width / self.source_width
            - 0.5
            - self.crop_column
        )
        result[..., 1] = (
            (coordinates[..., 1] + 0.5) * self.resized_height / self.source_height
            - 0.5
            - self.crop_row
        )
        return result

    def target_source_grid(self) -> FloatArray:
        target_y, target_x = np.meshgrid(
            np.arange(self.target_height, dtype=np.float64),
            np.arange(self.target_width, dtype=np.float64),
            indexing="ij",
        )
        return self.target_to_source(np.stack((target_x, target_y), axis=-1))

    def sample_nearest(self, source: np.ndarray) -> np.ndarray:
        """Sample a source image on the target grid with nearest neighbors."""

        if source.shape[:2] != (self.source_height, self.source_width):
            raise ValueError("source image shape does not match crop geometry")
        coordinates = self.target_source_grid()
        columns = np.clip(
            np.rint(coordinates[..., 0]).astype(np.int64), 0, self.source_width - 1
        )
        rows = np.clip(
            np.rint(coordinates[..., 1]).astype(np.int64), 0, self.source_height - 1
        )
        return source[rows, columns]


@dataclass(frozen=True)
class PhysTwinCase:
    """A calibrated official PhysTwin RGB-D interaction."""

    root: Path
    intrinsics: FloatArray
    camera_to_world: FloatArray
    frame_count: int
    fps: float
    source_width: int
    source_height: int

    @classmethod
    def from_directory(cls, root: str | Path) -> PhysTwinCase:
        path = Path(root).resolve()
        metadata_path = path / "metadata.json"
        calibration_path = path / "calibrate.pkl"
        if not metadata_path.is_file() or not calibration_path.is_file():
            raise ValueError(f"{path} is not a calibrated PhysTwin case")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with calibration_path.open("rb") as handle:
            camera_to_world = np.asarray(pickle.load(handle), dtype=np.float64)
        intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
        width, height = (int(value) for value in metadata["WH"])
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError("PhysTwin intrinsics must have shape (C, 3, 3)")
        if camera_to_world.shape != (intrinsics.shape[0], 4, 4):
            raise ValueError("PhysTwin camera calibration count is inconsistent")
        return cls(
            root=path,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            frame_count=int(metadata["frame_num"]),
            fps=float(metadata["fps"]),
            source_width=width,
            source_height=height,
        )

    @property
    def camera_count(self) -> int:
        return int(self.intrinsics.shape[0])

    def _validate_camera(self, camera: int) -> None:
        if not 0 <= camera < self.camera_count:
            raise ValueError(f"camera must be between 0 and {self.camera_count - 1}")

    def _validate_frame(self, frame: int) -> None:
        if not 0 <= frame < self.frame_count:
            raise ValueError(f"frame must be between 0 and {self.frame_count - 1}")

    def color_video(self, camera: int) -> Path:
        self._validate_camera(camera)
        path = self.root / "color" / f"{camera}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def load_depth_m(self, frame: int, camera: int) -> FloatArray:
        self._validate_frame(frame)
        self._validate_camera(camera)
        path = self.root / "depth" / str(camera) / f"{frame}.npy"
        depth = np.asarray(np.load(path), dtype=np.float64) / 1000.0
        if depth.shape != (self.source_height, self.source_width):
            raise ValueError(f"unexpected depth shape in {path}")
        return depth

    def load_processed_masks(self) -> dict:
        path = self.root / "mask" / "processed_masks.pkl"
        with path.open("rb") as handle:
            masks = pickle.load(handle)
        if len(masks) != self.frame_count:
            raise ValueError("processed mask frame count is inconsistent")
        return masks

    def object_mask(self, masks: dict, frame: int, camera: int) -> BoolArray:
        self._validate_frame(frame)
        self._validate_camera(camera)
        mask = np.asarray(masks[frame][camera]["object"], dtype=bool)
        if mask.shape != (self.source_height, self.source_width):
            raise ValueError("processed object mask shape is inconsistent")
        return mask

    def metric_point_map(
        self,
        frame: int,
        camera: int,
        crop: CoverResizeCrop,
        *,
        masks: dict | None = None,
        object_only: bool = True,
        minimum_depth_m: float = 0.2,
        maximum_depth_m: float = 1.5,
    ) -> tuple[FloatArray, BoolArray]:
        """Unproject one camera onto MotionCrafter's model grid in world meters."""

        if crop.source_height != self.source_height or crop.source_width != self.source_width:
            raise ValueError("crop source geometry does not match PhysTwin metadata")
        depth = crop.sample_nearest(self.load_depth_m(frame, camera))
        valid = np.isfinite(depth) & (depth > minimum_depth_m) & (depth < maximum_depth_m)
        if object_only:
            actual_masks = self.load_processed_masks() if masks is None else masks
            valid &= crop.sample_nearest(self.object_mask(actual_masks, frame, camera))

        source_coordinates = crop.target_source_grid()
        intrinsic = self.intrinsics[camera]
        camera_points = np.stack(
            (
                (source_coordinates[..., 0] - intrinsic[0, 2]) / intrinsic[0, 0] * depth,
                (source_coordinates[..., 1] - intrinsic[1, 2]) / intrinsic[1, 1] * depth,
                depth,
            ),
            axis=-1,
        )
        world = np.einsum(
            "ij,...j->...i",
            self.camera_to_world[camera, :3, :3],
            camera_points,
        ) + self.camera_to_world[camera, :3, 3]
        world[~valid] = 0.0
        return world.astype(np.float32), valid

    def metric_truth(
        self,
        frame_indices: NDArray[np.integer],
        camera: int,
        crop: CoverResizeCrop,
        *,
        object_only: bool = True,
    ) -> TruthSequence:
        frames = np.asarray(frame_indices, dtype=np.int64)
        if frames.ndim != 1 or frames.size == 0:
            raise ValueError("frame_indices must be a non-empty vector")
        masks = self.load_processed_masks() if object_only else None
        point_maps = []
        valid_masks = []
        for frame in frames:
            points, valid = self.metric_point_map(
                int(frame),
                camera,
                crop,
                masks=masks,
                object_only=object_only,
            )
            point_maps.append(points)
            valid_masks.append(valid)
        return TruthSequence(
            frame_indices=frames,
            point_map=np.stack(point_maps),
            valid_mask=np.stack(valid_masks),
        )

    def project_world(self, points: FloatArray, camera: int) -> tuple[FloatArray, FloatArray]:
        """Project world points to source pixels and return positive camera depth."""

        self._validate_camera(camera)
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("world points must end in three coordinates")
        world_to_camera = np.linalg.inv(self.camera_to_world[camera])
        camera_points = np.einsum(
            "ij,...j->...i", world_to_camera[:3, :3], points
        ) + world_to_camera[:3, 3]
        depth = camera_points[..., 2]
        intrinsic = self.intrinsics[camera]
        safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.0)
        pixels = np.stack(
            (
                intrinsic[0, 0] * camera_points[..., 0] / safe_depth + intrinsic[0, 2],
                intrinsic[1, 1] * camera_points[..., 1] / safe_depth + intrinsic[1, 2],
            ),
            axis=-1,
        )
        return pixels, depth


def sample_vector_field_nearest(
    field: FloatArray,
    coordinates_xy: FloatArray,
    *,
    valid_mask: BoolArray | None = None,
) -> tuple[FloatArray, BoolArray]:
    """Sample an image vector field at continuous pixel coordinates."""

    values = np.asarray(field)
    coordinates = np.asarray(coordinates_xy, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("field must have shape (H, W, 3)")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    rows = np.rint(coordinates[:, 1]).astype(np.int64)
    columns = np.rint(coordinates[:, 0]).astype(np.int64)
    active = (
        (rows >= 0)
        & (rows < values.shape[0])
        & (columns >= 0)
        & (columns < values.shape[1])
    )
    clipped_rows = np.clip(rows, 0, values.shape[0] - 1)
    clipped_columns = np.clip(columns, 0, values.shape[1] - 1)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape[:2]:
            raise ValueError("valid_mask shape does not match field")
        active &= mask[clipped_rows, clipped_columns]
    sampled = values[clipped_rows, clipped_columns]
    return sampled, active


def deterministic_subsample(
    points: FloatArray,
    maximum_points: int,
    *,
    seed: int,
) -> FloatArray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point sets must have shape (N, 3)")
    if maximum_points < 1:
        raise ValueError("maximum_points must be positive")
    if points.shape[0] <= maximum_points:
        return points
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(points.shape[0], maximum_points, replace=False))
    return points[indices]


def directed_nearest_distances(
    source: FloatArray,
    target: FloatArray,
    *,
    chunk_size: int = 512,
) -> FloatArray:
    _, distances = nearest_neighbor_indices(source, target, chunk_size=chunk_size)
    return distances


def nearest_neighbor_indices(
    source: FloatArray,
    target: FloatArray,
    *,
    chunk_size: int = 512,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Return each source point's nearest target index and metric distance."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1:] != (3,)
        or target.shape[1:] != (3,)
    ):
        raise ValueError("source and target must have shape (N, 3)")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("point sets must not be empty")
    indices = np.empty(source.shape[0], dtype=np.int64)
    distances = np.empty(source.shape[0], dtype=np.float64)
    for start in range(0, source.shape[0], chunk_size):
        stop = min(start + chunk_size, source.shape[0])
        squared = np.sum((source[start:stop, None, :] - target[None, :, :]) ** 2, axis=-1)
        local_indices = np.argmin(squared, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local_indices])
    return indices, distances


@dataclass(frozen=True)
class PointSetMetrics:
    source_to_target_mean_m: float
    target_to_source_mean_m: float
    symmetric_mean_m: float
    target_to_source_p95_m: float
    target_coverage_10mm: float
    target_coverage_20mm: float
    source_count: int
    target_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def point_set_metrics(source: FloatArray, target: FloatArray) -> PointSetMetrics:
    source_distances = directed_nearest_distances(source, target)
    target_distances = directed_nearest_distances(target, source)
    return PointSetMetrics(
        source_to_target_mean_m=float(np.mean(source_distances)),
        target_to_source_mean_m=float(np.mean(target_distances)),
        symmetric_mean_m=float(0.5 * (np.mean(source_distances) + np.mean(target_distances))),
        target_to_source_p95_m=float(np.quantile(target_distances, 0.95)),
        target_coverage_10mm=float(np.mean(target_distances <= 0.01)),
        target_coverage_20mm=float(np.mean(target_distances <= 0.02)),
        source_count=int(source_distances.size),
        target_count=int(target_distances.size),
    )
