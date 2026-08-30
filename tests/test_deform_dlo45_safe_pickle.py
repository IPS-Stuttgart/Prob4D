from __future__ import annotations

import importlib.util
import io
import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/audit_deform_dlo45_observability_v1.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "audit_deform_dlo45_observability_v1",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _primitive_trajectory() -> list[list[list[float]]]:
    return [
        [[float(frame), float(vertex), 0.0] for vertex in range(5)]
        for frame in range(2)
    ]


def test_primitive_only_unpickler_accepts_nested_numeric_containers() -> None:
    module = _load_script()
    value = _primitive_trajectory()

    loaded = module.load_primitive_pickle(io.BytesIO(pickle.dumps(value, protocol=4)))

    assert loaded == value


def test_primitive_only_unpickler_rejects_executable_globals() -> None:
    module = _load_script()

    with pytest.raises(pickle.UnpicklingError, match="global object builtins.len"):
        module.load_primitive_pickle(io.BytesIO(pickle.dumps(len, protocol=4)))


def test_load_trajectory_accepts_the_pinned_primitive_layout(tmp_path: Path) -> None:
    module = _load_script()
    value = _primitive_trajectory()
    path = tmp_path / "trajectory.pkl"
    path.write_bytes(pickle.dumps(value, protocol=4))

    trajectory = module.load_trajectory(path)

    assert trajectory.shape == (2, 5, 3)
    assert trajectory.dtype == np.float64
    assert np.allclose(trajectory, np.asarray(value, dtype=np.float64))
