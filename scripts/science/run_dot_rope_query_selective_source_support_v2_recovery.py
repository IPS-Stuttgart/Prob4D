#!/usr/bin/env python3
"""Run frozen DOT source-support v2 with one dataset-free smoke repair.

The scientific implementation remains byte-pinned. This wrapper changes only
its CUT3R smoke fixture: synthetic frames are written into a fresh child
folder rather than into ``TemporaryDirectory``'s already-created root.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

_V2_PATH = Path(__file__).with_name("run_dot_rope_query_selective_source_support_v2.py")
_V2_GIT_BLOB_SHA1 = "7d63f9d3b718f53b15036e55d538add2560e855a"


def _git_blob_sha1(payload: bytes) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _load_v2() -> ModuleType:
    source = _V2_PATH.read_bytes()
    measured = _git_blob_sha1(source)
    if measured != _V2_GIT_BLOB_SHA1:
        raise RuntimeError("frozen source-support v2 implementation changed")
    spec = importlib.util.spec_from_file_location(
        "dot_rope_query_selective_source_support_v2_frozen",
        _V2_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen source-support v2 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_runtime_smoke_compatibility(v2: ModuleType) -> None:
    original_run_base_provider = v2._run_base_provider

    def run_base_provider(args: Any, command: str) -> int:
        if command != "runtime-smoke":
            return int(original_run_base_provider(args, command))

        protocol = v2._load_protocol(args.protocol)
        v2._require_execution_identity(args.request_id, args.prob4d_revision)
        base = v2._load_script(
            "run_dot_rope_cut3r_native_provider.py",
            "dot_r11_r20_source_provider_recovery",
            v2.BASE_PROVIDER_BLOB,
        )
        adapted = v2._adapted_provider_protocol(protocol)
        base._load_protocol = lambda _path: adapted
        original_make_frames = base._make_synthetic_frames

        def make_frames(destination: Path, count: int) -> list[Path]:
            return list(original_make_frames(destination / "frames", count))

        base._make_synthetic_frames = make_frames
        return int(base.runtime_smoke(args))

    v2._run_base_provider = run_base_provider


def main() -> int:
    v2 = _load_v2()
    _install_runtime_smoke_compatibility(v2)
    return int(v2.main())


if __name__ == "__main__":
    raise SystemExit(main())
