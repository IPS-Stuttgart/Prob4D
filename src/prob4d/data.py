"""Validated data contracts for decoded MotionCrafter predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]


def _readonly(value: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    """Return a defensive, read-only NumPy copy."""

    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PredictionWindow:
    """Decoded predictions for one local MotionCrafter temporal window.

    Point maps and scene flow use the window's local world gauge. Absolute
    ``frame_indices`` identify duplicate frames across overlapping windows.
    Every NumPy field is defensively copied and made read-only so a validated
    window cannot be mutated after it has entered a content-addressed artifact.
    """

    window_id: str
    frame_indices: IntArray
    point_map: FloatArray
    valid_mask: BoolArray
    scene_flow: FloatArray | None = None
    deform_mask: BoolArray | None = None
    ray_directions: FloatArray | None = None

    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64).copy()
        point_map = np.asarray(self.point_map, dtype=np.float64).copy()
        valid_mask = np.asarray(self.valid_mask, dtype=bool).copy()

        if not window_id:
            raise ValueError("window_id must not be empty")
        if frame_indices.ndim != 1 or frame_indices.size == 0:
            raise ValueError("frame_indices must be a non-empty one-dimensional array")
        if np.any(frame_indices < 0):
            raise ValueError("frame_indices must be non-negative")
        if np.any(np.diff(frame_indices) <= 0):
            raise ValueError("frame_indices must be strictly increasing")
        if point_map.ndim != 4 or point_map.shape[-1] != 3:
            raise ValueError("point_map must have shape (T, H, W, 3)")
        if valid_mask.shape != point_map.shape[:-1]:
            raise ValueError("valid_mask must have shape (T, H, W)")
        if point_map.shape[0] != frame_indices.size:
            raise ValueError("frame_indices length must match point_map time dimension")
        if not np.all(np.isfinite(point_map[valid_mask])):
            raise ValueError("valid point_map entries must be finite")

        scene_flow = self._optional_vector_field("scene_flow", self.scene_flow, point_map)
        deform_mask = self._optional_mask("deform_mask", self.deform_mask, valid_mask)
        rays = self._optional_vector_field("ray_directions", self.ray_directions, point_map)

        if (scene_flow is None) != (deform_mask is None):
            raise ValueError("scene_flow and deform_mask must either both be present or absent")
        if scene_flow is not None and not np.all(np.isfinite(scene_flow[deform_mask])):
            raise ValueError("active scene_flow entries must be finite")
        if rays is not None:
            if not np.all(np.isfinite(rays[valid_mask])):
                raise ValueError("valid ray directions must be finite")
            ray_norm = np.linalg.norm(rays, axis=-1)
            if np.any(valid_mask & (ray_norm <= np.finfo(np.float64).eps)):
                raise ValueError("valid ray directions must be nonzero")
            normalize = ray_norm > np.finfo(np.float64).eps
            rays[normalize] /= ray_norm[normalize, None]

        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "frame_indices", _readonly(frame_indices))
        object.__setattr__(self, "point_map", _readonly(point_map))
        object.__setattr__(self, "valid_mask", _readonly(valid_mask))
        object.__setattr__(
            self,
            "scene_flow",
            None if scene_flow is None else _readonly(scene_flow),
        )
        object.__setattr__(
            self,
            "deform_mask",
            None if deform_mask is None else _readonly(deform_mask),
        )
        object.__setattr__(self, "ray_directions", None if rays is None else _readonly(rays))

    @staticmethod
    def _optional_vector_field(
        name: str, value: FloatArray | None, reference: FloatArray
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).copy()
        if array.shape != reference.shape:
            raise ValueError(f"{name} must have shape {reference.shape}")
        return array

    @staticmethod
    def _optional_mask(
        name: str, value: BoolArray | None, reference: BoolArray
    ) -> np.ndarray | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=bool).copy()
        if array.shape != reference.shape:
            raise ValueError(f"{name} must have shape {reference.shape}")
        return array

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the ``(T, H, W)`` sample grid shape."""

        return self.point_map.shape[:-1]

    @property
    def start_frame(self) -> int:
        return int(self.frame_indices[0])

    @property
    def stop_frame(self) -> int:
        """Return the exclusive final frame index for unit-stride windows."""

        return int(self.frame_indices[-1]) + 1

    def local_index(self, frame_index: int) -> int:
        position = int(np.searchsorted(self.frame_indices, frame_index))
        if position == self.frame_indices.size or self.frame_indices[position] != frame_index:
            raise KeyError(f"frame {frame_index} is not in window {self.window_id!r}")
        return position

    def common_frames(self, other: PredictionWindow) -> IntArray:
        return np.intersect1d(self.frame_indices, other.frame_indices, assume_unique=True)

    def rays(self) -> FloatArray:
        """Return normalized rays, falling back to directions from the local origin."""

        if self.ray_directions is not None:
            return self.ray_directions.copy()
        norms = np.linalg.norm(self.point_map, axis=-1, keepdims=True)
        return np.divide(
            self.point_map,
            norms,
            out=np.zeros_like(self.point_map),
            where=norms > np.finfo(np.float64).eps,
        )

    def to_npz(self, path: str | Path) -> None:
        """Write a portable, self-describing prediction window."""

        payload: dict[str, np.ndarray] = {
            "window_id": np.asarray(self.window_id),
            "frame_indices": self.frame_indices,
            "point_map": self.point_map.astype(np.float32),
            "valid_mask": self.valid_mask,
        }
        if self.scene_flow is not None:
            payload["scene_flow"] = self.scene_flow.astype(np.float32)
            payload["deform_mask"] = self.deform_mask
        if self.ray_directions is not None:
            payload["ray_directions"] = self.ray_directions.astype(np.float32)
        np.savez_compressed(Path(path), **payload)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        start_frame: int | None = None,
        window_id: str | None = None,
    ) -> PredictionWindow:
        """Read a window, requiring explicit frame metadata when absent upstream."""

        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "point_map" not in data or "valid_mask" not in data:
                raise ValueError(f"{path} does not contain point_map and valid_mask")
            time_steps = int(data["point_map"].shape[0])
            if "frame_indices" in data:
                frame_indices = data["frame_indices"]
            elif start_frame is not None:
                frame_indices = np.arange(start_frame, start_frame + time_steps)
            else:
                raise ValueError(
                    "MotionCrafter files without frame_indices require start_frame explicitly"
                )

            stored_id = str(data["window_id"].item()) if "window_id" in data else path.stem
            return cls(
                window_id=window_id or stored_id,
                frame_indices=frame_indices,
                point_map=data["point_map"],
                valid_mask=data["valid_mask"],
                scene_flow=data["scene_flow"] if "scene_flow" in data else None,
                deform_mask=data["deform_mask"] if "deform_mask" in data else None,
                ray_directions=data["ray_directions"] if "ray_directions" in data else None,
            )
