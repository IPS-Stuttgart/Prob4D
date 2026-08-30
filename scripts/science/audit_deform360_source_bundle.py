#!/usr/bin/env python3
"""Bind the Deform360 metadata audit to the reviewed processed repository."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPLEMENTATION_PATH = Path(__file__).with_name("audit_deform360_source_bundle_impl_v1.py")
_SPEC = importlib.util.spec_from_file_location(
    "prob4d_deform360_source_bundle_audit_impl_v1",
    _IMPLEMENTATION_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Deform360 audit implementation: {_IMPLEMENTATION_PATH}")
_IMPLEMENTATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPLEMENTATION)
_IMPLEMENTATION.EXPECTED_SOURCE_ROOT = Path(
    "/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository"
)

for _name in dir(_IMPLEMENTATION):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPLEMENTATION, _name)

if __name__ == "__main__":
    raise SystemExit(_IMPLEMENTATION.main())
