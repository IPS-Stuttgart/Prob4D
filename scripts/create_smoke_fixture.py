#!/usr/bin/env python3
"""Create a tiny rendered video and metric truth for integration smoke tests.

The fixture is intentionally not a benchmark: its simple image rendering is not
photometrically coupled to the complete 3D point map. It exists only to verify
the MotionCrafter-to-Prob4D artifact and evaluation path end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from prob4d.io import save_truth
from prob4d.metrics import TruthSequence


def create_fixture(
    output_directory: Path,
    *,
    num_frames: int,
    height: int,
    width: int,
    direction: int,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rows, columns = np.mgrid[:height, :width]
    normalized_row = (rows - 0.5 * (height - 1)) / height
    normalized_column = (columns - 0.5 * (width - 1)) / width
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = np.clip(127 + 100 * normalized_column, 0, 255)
    base[..., 1] = np.clip(127 + 100 * normalized_row, 0, 255)
    base[..., 2] = 80

    point_map = np.empty((num_frames, height, width, 3), dtype=np.float32)
    scene_flow = np.zeros_like(point_map)
    frames: list[np.ndarray] = []
    velocity = direction * 0.02
    for frame_index in range(num_frames):
        center_column = (width // 4 + direction * 7 * frame_index) % width
        object_mask = (columns - center_column) ** 2 + (rows - height // 2) ** 2 < (
            height // 7
        ) ** 2
        image = base.copy()
        image[object_mask] = np.array([255, 230, 40], dtype=np.uint8)
        frames.append(image)

        depth = 5.0 + 0.25 * np.sin(8.0 * normalized_column) + 0.1 * normalized_row
        point_map[frame_index, ..., 0] = normalized_column * depth
        point_map[frame_index, ..., 1] = normalized_row * depth
        point_map[frame_index, ..., 2] = depth
        point_map[frame_index, object_mask, 0] += velocity * frame_index
        scene_flow[frame_index, object_mask, 0] = velocity

    video_path = output_directory / "video.mp4"
    imageio.mimwrite(video_path, frames, fps=10, codec="libx264", quality=8)
    deform_mask = np.ones((num_frames, height, width), dtype=bool)
    deform_mask[-1] = False
    truth_path = output_directory / "truth.npz"
    save_truth(
        truth_path,
        TruthSequence(
            frame_indices=np.arange(num_frames),
            point_map=point_map,
            valid_mask=np.ones((num_frames, height, width), dtype=bool),
            scene_flow=scene_flow,
            deform_mask=deform_mask,
        ),
    )
    return video_path, truth_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--num-frames", type=int, default=26)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--direction", type=int, choices=[-1, 1], default=1)
    arguments = parser.parse_args()
    video, truth = create_fixture(
        arguments.output_directory,
        num_frames=arguments.num_frames,
        height=arguments.height,
        width=arguments.width,
        direction=arguments.direction,
    )
    print(video)
    print(truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
