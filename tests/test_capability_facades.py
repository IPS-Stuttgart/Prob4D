from __future__ import annotations

import subprocess
import sys
from types import ModuleType

from prob4d.api import artifacts_v1, calibration_v1, covariance_v1, geometry_v1, provider_v1
from prob4d.api import v2 as api_v2

_FACADES: tuple[ModuleType, ...] = (
    geometry_v1,
    artifacts_v1,
    covariance_v1,
    calibration_v1,
    provider_v1,
)


def test_capability_facades_are_exact_additive_api_v2_aliases() -> None:
    for facade in _FACADES:
        assert facade.FACADE_VERSION == 1
        assert facade.LIFECYCLE == "preview"
        assert facade.__all__ == sorted(facade.__all__)
        assert len(facade.__all__) == len(set(facade.__all__))
        for name in facade.__all__:
            assert not name.startswith("_")
            if name in {"FACADE_VERSION", "LIFECYCLE"}:
                continue
            assert name in api_v2.__all__
            assert getattr(facade, name) is getattr(api_v2, name)


def test_capability_facades_do_not_load_optional_gpu_stacks() -> None:
    script = """
import sys

import prob4d.api.artifacts_v1
import prob4d.api.calibration_v1
import prob4d.api.covariance_v1
import prob4d.api.geometry_v1
import prob4d.api.provider_v1

forbidden = {"torch", "diffusers", "decord"}
loaded = sorted(forbidden & set(sys.modules))
if loaded:
    raise SystemExit(f"optional GPU dependencies loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
