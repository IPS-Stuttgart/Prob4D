#!/usr/bin/env python3
"""Run one routed-camera component through the frozen DOT source-support v2 code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SOURCE = Path(__file__).with_name("run_dot_rope_query_selective_source_support_v2.py")
CAMERAS = {"cam001", "cam002", "cam003", "cam004", "cam005"}
ALL_SOURCE = {f"R{index:02d}" for index in range(11, 21)}
SHARED_3D_CAMERA = "cam001"


def _load():
    spec = importlib.util.spec_from_file_location("dot_source_support_component_delegate", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-support v2 delegate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--source-sequences-json", required=True)
    known, remainder = parser.parse_known_args()
    if known.camera not in CAMERAS:
        raise ValueError("camera is outside the frozen DOT roster")
    source_sequences = json.loads(known.source_sequences_json)
    if not isinstance(source_sequences, list) or not source_sequences:
        raise ValueError("source component requires a nonempty sequence list")
    if len(source_sequences) != len(set(source_sequences)):
        raise ValueError("source component sequence list contains duplicates")
    if any(sequence not in ALL_SOURCE for sequence in source_sequences):
        raise ValueError("source component escaped R11-R20")

    module = _load()
    module.CAMERA = known.camera
    module.SOURCE_SEQUENCES = list(source_sequences)

    original_load_script = module._load_script

    def load_script(filename: str, name: str, expected_blob: str):
        loaded = original_load_script(filename, name, expected_blob)
        if filename == "audit_dot_rope_marker_support.py":
            def coordinate_member(sequence: str, dimension: int, frame: int) -> str:
                camera = known.camera if dimension == 2 else SHARED_3D_CAMERA
                return (
                    f"{sequence}/coordinates/{dimension}d/"
                    f"frame{frame:06d}_{camera}.txt"
                )

            loaded._coordinate_member = coordinate_member
        return loaded

    module._load_script = load_script
    sys.argv = [SOURCE.name, *remainder]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
