from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_tracking_cloth_finite_orbit_real_v1.py"


def _load_module() -> ModuleType:
    name = "tracking_cloth_finite_orbit"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_coordinate_groups_accept_common_marker_header_conventions() -> None:
    module = _load_module()
    headers = [
        "time",
        "marker_01_x [m]",
        "marker_01_y [m]",
        "marker_01_z [m]",
        "X-marker-02",
        "Y-marker-02",
        "Z-marker-02",
        "3x",
        "3y",
        "3z",
        "force_x",
        "force_y",
        "force_z",
    ]
    groups = module._coordinate_groups(headers)
    assert sorted(groups) == ["3", "marker_01", "marker_02"]


def test_line_orbit_has_zero_local_radial_derivative_but_finite_width() -> None:
    module = _load_module()
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 100.0])
    p = np.array([30.0, 0.0, 40.0])
    axial, radius = module._line_geometry(a, b, p)
    assert axial == 40.0
    assert radius == 30.0
    local_derivative = -radius * math.sin(0.0)
    assert local_derivative == 0.0
    assert 2.0 * radius == 60.0


def test_hidden_angle_sequence_is_deterministic_and_exposes_local_harm() -> None:
    module = _load_module()
    radius = 25.0
    harmful = []
    for index in range(10000):
        theta = 2.0 * math.pi * module._uniform_from_key(20260831, "g", index) - math.pi
        truth = radius * math.cos(theta)
        local_error = (truth - radius) ** 2
        fallback_error = truth**2
        harmful.append(local_error > fallback_error)
    fraction = float(np.mean(harmful))
    assert 0.64 < fraction < 0.69
    assert module._uniform_from_key(1, "a") == module._uniform_from_key(1, "a")


def test_marker_reader_handles_semicolon_decimal_comma_and_meter_units(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "sample.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(
            [
                "time",
                "m1_x [m]",
                "m1_y [m]",
                "m1_z [m]",
                "m2_x [m]",
                "m2_y [m]",
                "m2_z [m]",
                "m3_x [m]",
                "m3_y [m]",
                "m3_z [m]",
            ]
        )
        writer.writerow(["0", "0,0", "0,0", "0,0", "0,0", "0,0", "0,1", "0,03", "0,0", "0,04"])
    values, scale, details = module._read_markers(path, ["m1", "m2", "m3"])
    assert scale == 1000.0
    assert details["rows"] == 1
    np.testing.assert_allclose(values[0, 1], [0.0, 0.0, 100.0])
    np.testing.assert_allclose(values[0, 2], [30.0, 0.0, 40.0])


def test_source_triplet_selection_finds_nondegenerate_geometry() -> None:
    module = _load_module()
    samples = []
    for shift in np.linspace(0.0, 10.0, 20):
        samples.append(
            [
                [shift, 0.0, 0.0],
                [shift, 0.0, 100.0],
                [shift + 30.0, 0.0, 40.0],
                [shift + 2.0, 0.0, 50.0],
            ]
        )
    triplet, details = module._select_marker_triplet(
        np.asarray(samples),
        ["a", "b", "probe", "near"],
        minimum_anchor_distance=20.0,
        minimum_probe_radius=5.0,
    )
    assert len(set(triplet)) == 3
    assert details["anchor_distance_q10_mm"] >= 20.0
    assert details["probe_radius_q10_mm"] >= 5.0


def test_gaussian_metrics_are_finite() -> None:
    module = _load_module()
    metrics = module._gaussian_metrics(1.0, 0.5, 4.0, 1.6448536269514722)
    assert metrics["squared_error"] == 0.25
    assert math.isfinite(metrics["nll"])
    assert metrics["covered"] == 1.0
