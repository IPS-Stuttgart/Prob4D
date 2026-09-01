#!/usr/bin/env python3
"""Invoke the camera-routing confirmation materializer with the final frozen calibration binding."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

DELEGATE = Path(__file__).with_name(
    "materialize_dot_rope_camera_routing_confirmation_request_v1.py"
)
PROTOCOL = Path("protocols/dot-rope-query-selective-camera-routing-confirmation-v1.json")
PROTOCOL_ID = "b9267992484516a88d73637c813ae463f27e4a7b2586f805d394bea45e5f537c"
SOURCE_CALIBRATION_ID = "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
DEPENDENCE_ALPHA = 0.85


def main() -> int:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("final confirmation protocol identity changed")
    prerequisite = value.get("prerequisite") or {}
    if prerequisite.get("source_calibration_id") != SOURCE_CALIBRATION_ID:
        raise ValueError("source calibration identity changed")
    if float(prerequisite.get("selected_dependence_alpha")) != DEPENDENCE_ALPHA:
        raise ValueError("source-selected dependence alpha changed")
    if float((value.get("factor") or {}).get("selected_dependence_alpha")) != DEPENDENCE_ALPHA:
        raise ValueError("factor dependence alpha changed")

    spec = importlib.util.spec_from_file_location("camera_routing_request_v1", DELEGATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load confirmation request materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL_ID = PROTOCOL_ID
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
