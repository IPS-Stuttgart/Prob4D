"""Metric RGB-D and correspondence helpers for PhysTwin experiment zero."""

from __future__ import annotations

import importlib
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .metrics import TruthSequence

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


def _integer_at_least(value: int, *, name: str, minimum: int) -> int:
    """Return a non-coercive integer that satisfies a lower bound."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        comparator = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    return result


def _positive_real(value: float, *, name: str) -> float:
    """Return a finite positive real scalar without accepting Boolean aliases."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _strict_json_object(path: Path) -> dict[str, Any]:
    """Load one finite JSON object while rejecting duplicate keys and symlinks."""

    if path.is_symlink():
        raise ValueError(f"refusing symbolic-link metadata path: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return loaded


def _legacy_numpy_pickle_globals() -> dict[tuple[str, str], Any]:
    """Return the minimal NumPy globals required by official legacy artifacts."""

    allowed: dict[tuple[str, str], Any] = {
        ("numpy", "dtype"): np.dtype,
        ("numpy", "ndarray"): np.ndarray,
    }
    module_attributes = {
        "numpy.core.multiarray": ("_reconstruct", "scalar"),
        "numpy._core.multiarray": ("_reconstruct", "scalar"),
        "numpy.core.numeric": ("_frombuffer",),
        "numpy._core.numeric": ("_frombuffer",),
    }
    for module_name, attributes in module_attributes.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                module = importlib.import_module(module_name)
                values = tuple(getattr(module, attribute, None) for attribute in attributes)
        except ImportError:
            continue
        for attribute, value in zip(attributes, values, strict=True):
            if value is not None:
                allowed[(module_name, attribute)] = value
    return allowed


_LEGACY_NUMPY_PICKLE_GLOBALS = _legacy_numpy_pickle_globals()


class _RestrictedLegacyUnpickler(pickle.Unpickler):
    """Unpickle only primitive containers and the NumPy array reconstruction API."""

    def find_class(self, module: str, name: str) -> Any:
        allowed = _LEGACY_NUMPY_PICKLE_GLOBALS.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(
                f"legacy PhysTwin pickle requests forbidden global {module}.{name}"
            )
        return allowed


def _load_trusted_legacy_pickle(path: Path, *, description: str) -> Any:
    """Load a legacy official dataset pickle through a restricted unpickler.

    The old PhysTwin dataset stores calibration and mask arrays as pickles. This
    adapter permits only NumPy's array reconstruction primitives; arbitrary
    globals and symbolic-link substitution fail closed. Portable Prob4D artifacts
    never use pickle.
    """

    if path.is_symlink():
        raise ValueError(f"refusing symbolic-link {description} path: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with path.open("rb") as handle:
            return _RestrictedLegacyUnpickler(handle).load()
    except (pickle.UnpicklingError, AttributeError, EOFError, TypeError, ValueError) as error:
        raise ValueError(f"invalid or unsafe {description} pickle: {path}") from error


def _readonly_float64(value: Any) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


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
        source_height = _integer_at_least(
            source_height,
            name="source_height",
            minimum=1,
        )
        source_width = _integer_at_least(
            source_width,
            name="source_width",
            minimum=1,
        )
        target_height = _integer_at_least(
            target_height,
            name="target_height",
            minimum=1,
        )
        target_width = _integer_at_least(
            target_width,
            name="target_width",
            minimum=1,
        )
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
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
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
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
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
    """A validated calibrated official PhysTwin RGB-D interaction."""

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
        metadata = _strict_json_object(metadata_path)
        required = {"intrinsics", "WH", "frame_num", "fps"}
        missing = sorted(required - metadata.keys())
        if missing:
            raise ValueError(f"PhysTwin metadata is missing required fields: {missing}")

        wh = metadata["WH"]
        if not isinstance(wh, list) or len(wh) != 2:
            raise ValueError("PhysTwin WH must be a two-element JSON array")
        width = _integer_at_least(wh[0], name="PhysTwin width", minimum=1)
        height = _integer_at_least(wh[1], name="PhysTwin height", minimum=1)
        frame_count = _integer_at_least(
            metadata["frame_num"],
            name="PhysTwin frame_num",
            minimum=1,
        )
        fps = _positive_real(metadata["fps"], name="PhysTwin fps")

        camera_to_world = np.asarray(
            _load_trusted_legacy_pickle(
                calibration_path,
                description="PhysTwin calibration",
            ),
            dtype=np.float64,
        )
        intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
        if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
            raise ValueError("PhysTwin intrinsics must have shape (C, 3, 3)")
        if intrinsics.shape[0] == 0:
            raise ValueError("PhysTwin must contain at least one camera")
        if camera_to_world.shape != (intrinsics.shape[0], 4, 4):
            raise ValueError("PhysTwin camera calibration count is inconsistent")
        if not np.all(np.isfinite(intrinsics)):
            raise ValueError("PhysTwin intrinsics must be finite")
        if not np.all(np.isfinite(camera_to_world)):
            raise ValueError("PhysTwin camera transforms must be finite")
        if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(intrinsics[:, 1, 1] <= 0.0):
            raise ValueError("PhysTwin focal lengths must be positive")
        expected_intrinsic_row = np.array([0.0, 0.0, 1.0])
        if not np.allclose(
            intrinsics[:, 2, :],
            expected_intrinsic_row,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("PhysTwin intrinsics must use a homogeneous final row")
        expected_transform_row = np.array([0.0, 0.0, 0.0, 1.0])
        if not np.allclose(
            camera_to_world[:, 3, :],
            expected_transform_row,
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError("PhysTwin camera transforms must be homogeneous")
        rotations = camera_to_world[:, :3, :3]
        gram = np.einsum("...ji,...jk->...ik", rotations, rotations)
        if not np.allclose(gram, np.eye(3), atol=1e-5, rtol=1e-5):
            raise ValueError("PhysTwin camera rotations must be orthonormal")
        determinants = np.linalg.det(rotations)
        if not np.allclose(determinants, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("PhysTwin camera rotations must be proper")

        return cls(
            root=path,
            intrinsics=_readonly_float64(intrinsics),
            camera_to_world=_readonly_float64(camera_to_world),
            frame_count=frame_count,
            fps=fps,
            source_width=width,
            source_height=height,
        )

    @property
    def camera_count(self) -> int:
        return int(self.intrinsics.shape[0])

    def _validate_camera(self, camera: int) -> None:
        camera = _integer_at_least(camera, name="camera", minimum=0)
        if camera >= self.camera_count:
            raise ValueError(f"camera must be between 0 and {self.camera_count - 1}")

    def _validate_frame(self, frame: int) -> None:
        frame = _integer_at_least(frame, name="frame", minimum=0)
        if frame >= self.frame_count:
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
        depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64) / 1000.0
        if depth.shape != (self.source_height, self.source_width):
            raise ValueError(f"unexpected depth shape in {path}")
        return depth

    def load_processed_masks(self) -> dict | list | tuple:
        path = self.root / "mask" / "processed_masks.pkl"
        masks = _load_trusted_legacy_pickle(
            path,
            description="PhysTwin processed-mask",
        )
        if not isinstance(masks, (dict, list, tuple)):
            raise ValueError("processed masks must be an indexed mapping or sequence")
        if len(masks) != self.frame_count:
            raise ValueError("processed mask frame count is inconsistent")
        return masks

    def object_mask(
        self,
        masks: dict | list | tuple,
        frame: int,
        camera: int,
    ) -> BoolArray:
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
        masks: dict | list | tuple | None = None,
        object_only: bool = True,
        minimum_depth_m: float = 0.2,
        maximum_depth_m: float = 1.5,
    ) -> tuple[FloatArray, BoolArray]:
        """Unproject one camera onto MotionCrafter's model grid in world meters."""

        if crop.source_height != self.source_height or crop.source_width != self.source_width:
            raise ValueError("crop source geometry does not match PhysTwin metadata")
        minimum_depth_m = _positive_real(minimum_depth_m, name="minimum_depth_m")
        maximum_depth_m = _positive_real(maximum_depth_m, name="maximum_depth_m")
        if maximum_depth_m <= minimum_depth_m:
            raise ValueError("maximum_depth_m must exceed minimum_depth_m")
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
        raw_frames = np.asarray(frame_indices)
        if not np.issubdtype(raw_frames.dtype, np.integer):
            raise ValueError("frame_indices must contain integers")
        frames = np.asarray(raw_frames, dtype=np.int64)
        if frames.ndim != 1 or frames.size == 0:
            raise ValueError("frame_indices must be a non-empty vector")
        if np.any(frames < 0) or np.any(frames >= self.frame_count):
            raise ValueError("frame_indices lie outside the PhysTwin case")
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
        """Project finite world points to source pixels and positive camera depth."""

        self._validate_camera(camera)
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError("world points must end in three coordinates")
        if not np.all(np.isfinite(points)):
            raise ValueError("world points must be finite")
        rotation = self.camera_to_world[camera, :3, :3]
        translation = self.camera_to_world[camera, :3, 3]
        camera_points = np.einsum(
            "ji,...j->...i",
            rotation,
            points - translation,
        )
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
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("field spatial dimensions must be nonempty")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    finite = np.all(np.isfinite(coordinates), axis=1)
    safe_coordinates = np.where(np.isfinite(coordinates), coordinates, 0.0)
    rows = np.rint(safe_coordinates[:, 1]).astype(np.int64)
    columns = np.rint(safe_coordinates[:, 0]).astype(np.int64)
    active = (
        finite
        & (rows >= 0)
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
    if not np.all(np.isfinite(points)):
        raise ValueError("point sets must be finite")
    maximum_points = _integer_at_least(
        maximum_points,
        name="maximum_points",
        minimum=1,
    )
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
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("source and target point sets must be finite")
    chunk_size = _integer_at_least(chunk_size, name="chunk_size", minimum=1)
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
