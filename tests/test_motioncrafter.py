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


def test_motioncrafter_config_validates_source_frame_selection() -> None:
    base = {
        "upstream_root": Path("upstream"),
        "video_path": Path("video.mp4"),
        "output_directory": Path("output"),
    }
    with pytest.raises(ValueError, match="frame_start"):
        MotionCrafterRunConfig(**base, frame_start=-1)
    with pytest.raises(ValueError, match="frame_stop"):
        MotionCrafterRunConfig(**base, frame_start=10, frame_stop=10)
    with pytest.raises(ValueError, match="frame_stride"):
        MotionCrafterRunConfig(**base, frame_stride=0)
