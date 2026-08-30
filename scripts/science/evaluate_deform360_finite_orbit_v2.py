#!/usr/bin/env python3
"""V2 wrapper for the reviewed finite-orbit evaluator.

V2 reuses the v1 estimator and metrics while increasing the object-disjoint
calibration cohort from at most two objects to approximately one third of the
available non-target objects.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

PROTOCOL_ID = "deform360-finite-orbit-real-data-v2"


def load_v1():
    path = Path(__file__).with_name("evaluate_deform360_finite_orbit_v1.py")
    spec = importlib.util.spec_from_file_location(
        "prob4d_deform360_finite_orbit_v1", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_v1()
    module.PROTOCOL_ID = PROTOCOL_ID

    def split_fold(object_ids, held_out):
        others = [value for value in object_ids if value != held_out]
        ordered = sorted(
            others,
            key=lambda value: module.sha256_bytes(
                f"{PROTOCOL_ID}:{held_out}:{value}".encode()
            ),
        )
        if len(ordered) < 5:
            raise ValueError(
                "at least six complete objects are required for v2 folds"
            )
        calibration_count = max(3, int(math.floor(len(ordered) / 3)))
        calibration_count = min(calibration_count, len(ordered) - 2)
        return ordered[calibration_count:], ordered[:calibration_count]

    original_report = module.report

    def report(summary):
        return original_report(summary).replace(
            "Deform360 finite-orbit real-data pilot v1",
            "Deform360 finite-orbit real-data pilot v2",
        )

    module.split_fold = split_fold
    module.report = report
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
