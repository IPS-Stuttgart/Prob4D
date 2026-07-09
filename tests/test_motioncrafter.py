from pathlib import Path

import pytest

from prob4d.motioncrafter import MotionCrafterRunConfig


def test_motioncrafter_config_validates_window_geometry() -> None:
    with pytest.raises(ValueError, match="overlap"):
        MotionCrafterRunConfig(
            upstream_root=Path("upstream"),
            video_path=Path("video.mp4"),
            output_directory=Path("output"),
            window_size=25,
            overlap=25,
        )

    with pytest.raises(ValueError, match="divisible"):
        MotionCrafterRunConfig(
            upstream_root=Path("upstream"),
            video_path=Path("video.mp4"),
            output_directory=Path("output"),
            height=321,
        )
