from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from prob4d.phystwin_experiment import load_physics_trajectory


def mark_pickle_execution(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class UnsafePicklePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return mark_pickle_execution, (str(self.marker),)


def _dump(path: Path, value: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _valid_final_data() -> dict[str, np.ndarray]:
    return {
        "object_points": np.zeros((1, 2, 3), dtype=np.float64),
        "surface_points": np.zeros((1, 3), dtype=np.float64),
    }


def test_legacy_physics_trajectory_loads_numpy_only_payloads(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.pkl"
    final_data_path = tmp_path / "final-data.pkl"
    trajectory = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    _dump(trajectory_path, trajectory)
    _dump(final_data_path, _valid_final_data())

    loaded = load_physics_trajectory(trajectory_path, final_data_path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded, trajectory[:, :3])


def test_legacy_physics_trajectory_rejects_executable_pickle(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.pkl"
    final_data_path = tmp_path / "final-data.pkl"
    marker = tmp_path / "trajectory-pickle-executed"
    _dump(trajectory_path, UnsafePicklePayload(marker))
    _dump(final_data_path, _valid_final_data())

    with pytest.raises(ValueError, match="invalid or unsafe physics trajectory pickle"):
        load_physics_trajectory(trajectory_path, final_data_path)
    assert not marker.exists()


def test_legacy_final_data_rejects_executable_pickle(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.pkl"
    final_data_path = tmp_path / "final-data.pkl"
    marker = tmp_path / "final-data-pickle-executed"
    _dump(trajectory_path, np.zeros((2, 4, 3), dtype=np.float64))
    _dump(final_data_path, UnsafePicklePayload(marker))

    with pytest.raises(ValueError, match="invalid or unsafe PhysTwin final data pickle"):
        load_physics_trajectory(trajectory_path, final_data_path)
    assert not marker.exists()


def test_legacy_final_data_rejects_non_dictionary_payload(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.pkl"
    final_data_path = tmp_path / "final-data.pkl"
    _dump(trajectory_path, np.zeros((2, 4, 3), dtype=np.float64))
    _dump(final_data_path, [np.zeros((1, 2, 3)), np.zeros((1, 3))])

    with pytest.raises(ValueError, match="PhysTwin final data must be a dictionary"):
        load_physics_trajectory(trajectory_path, final_data_path)
