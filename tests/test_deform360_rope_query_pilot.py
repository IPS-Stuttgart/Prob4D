from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "science" / "run_deform360_rope_query_pilot.py"
PROTOCOL = Path(__file__).parents[1] / "protocols" / "deform360-rope-query-pilot-v1.json"
SPEC = importlib.util.spec_from_file_location("deform360_rope_query_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_is_frozen_and_disjoint() -> None:
    protocol = MODULE.read_protocol(PROTOCOL)
    split = protocol["episode_split"]
    assert split == {
        "source": [0, 1, 2, 3, 4],
        "calibration": [5, 6],
        "target": [7, 8, 9],
    }
    assert protocol["dataset"]["runner_root"] == (
        "/mnt/seagate10tb/florianpfaff/datasets/deform360"
    )
    assert protocol["information_boundary"]["target_used_for_fitting"] is False
    assert protocol["information_boundary"]["target_used_for_calibration"] is False


def test_mechanism_self_test() -> None:
    MODULE.self_test()


def test_shared_paired_bound_can_preserve_cancellation() -> None:
    values = [
        {"candidate_error": 0.19, "fallback_error": 0.20, "advantage": 0.01},
        {"candidate_error": 0.79, "fallback_error": 0.80, "advantage": 0.01},
    ]
    shared = MODULE.conservative_quantile(
        (value["advantage"] for value in values), 0.10
    )
    independent = MODULE.conservative_quantile(
        (value["fallback_error"] for value in values), 0.10
    ) - MODULE.conservative_quantile(
        (value["candidate_error"] for value in values), 0.90
    )
    assert shared > 0.0
    assert independent < 0.0


def test_query_orbit_invariance_and_sensitivity() -> None:
    current = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    future = np.asarray([[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]])
    axis = np.asarray([1.0, 0.0, 0.0])
    span = [
        MODULE.query_value(current, future, axis, "span_change", gauge)
        for gauge in (0, 1)
    ]
    named = [
        MODULE.query_value(current, future, axis, "named_endpoint_progress", gauge)
        for gauge in (0, 1)
    ]
    assert span[0] == span[1]
    assert named[0] != named[1]
