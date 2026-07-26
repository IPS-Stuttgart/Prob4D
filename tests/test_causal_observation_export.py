from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from prob4d.io import PredictionBundle
from prob4d.observation_export import admit_source_causal_windows


@dataclass
class FakeWindow:
    window_id: str
    frame_indices: np.ndarray


def bundle(*windows: FakeWindow) -> PredictionBundle:
    records = [
        {
            "window_id": window.window_id,
            "path": f"windows/{window.window_id}.npz",
            "start_frame": int(window.frame_indices[0]),
            "stop_frame": int(window.frame_indices[-1]) + 1,
        }
        for window in windows
    ]
    return PredictionBundle(
        manifest_path=Path("predictions.json"),
        overlap_windows=list(windows),
        disjoint_baseline=None,
        latent_linear_baseline=None,
        metadata={"overlap_windows": records},
    )


def test_crossing_source_window_is_rejected() -> None:
    first = FakeWindow("window_0000", np.arange(0, 10))
    crossing = FakeWindow("window_0001", np.arange(8, 18))
    selected, admitted, rejected = admit_source_causal_windows(
        bundle(first, crossing), causal_frame_stop=12
    )
    assert [window.window_id for window in selected.overlap_windows] == [
        "window_0000"
    ]
    assert [record["window_id"] for record in admitted] == ["window_0000"]
    assert [record["window_id"] for record in rejected] == ["window_0001"]


def test_no_complete_window_fails_closed() -> None:
    crossing = FakeWindow("window_0000", np.arange(8, 18))
    with pytest.raises(ValueError, match="no complete MotionCrafter"):
        admit_source_causal_windows(
            bundle(crossing), causal_frame_stop=12
        )
