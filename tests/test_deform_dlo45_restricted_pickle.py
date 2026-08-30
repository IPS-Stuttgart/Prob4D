from __future__ import annotations

import importlib.util
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_deform_dlo45_observability_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dlo45_observability_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restricted_loader_accepts_numeric_numpy_trajectory(tmp_path: Path) -> None:
    module = _load_module()
    expected = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)
    path = tmp_path / "trajectory.pkl"
    path.write_bytes(pickle.dumps(expected, protocol=4))

    actual = module.load_trajectory(path)

    np.testing.assert_array_equal(actual, expected)


class _CommandPayload:
    def __reduce__(self):
        return os.system, ("echo unsafe > should-not-exist",)


def test_restricted_loader_rejects_arbitrary_globals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    path = tmp_path / "malicious.pkl"
    path.write_bytes(pickle.dumps(_CommandPayload(), protocol=4))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(pickle.UnpicklingError, match="forbidden pickle global"):
        module.load_trajectory(path)

    assert not (tmp_path / "should-not-exist").exists()
