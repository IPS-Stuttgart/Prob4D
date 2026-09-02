from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "run_tracking_cloth_rank_distortion.py"
)
SPEC = importlib.util.spec_from_file_location("run_tracking_cloth_rank_distortion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _protocol(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": MODULE.SCHEMA,
        "fold_count": 3,
        "lag_frames": 3,
        "horizon_frames": 6,
        "stride_frames": 6,
        "maximum_windows_per_recording": 16,
        "minimum_recordings_per_size": 3,
        "joint_covariance_shrinkage": 0.15,
        "joint_covariance_ridge_fraction": 1e-5,
        "maximum_conditional_block_fraction": 0.20,
        "factor_eigenvalue_relative_tolerance": 1e-12,
        "rank_relative_tolerance": 1e-12,
        "retained_ranks": [0, 1, 2, 3],
        "optimality_relative_tolerance": 1e-8,
        "strict_improvement_relative_tolerance": 1e-8,
        "required_maximum_relative_parity_error": 1e-7,
    }
    value.update(updates)
    return value


def _grid(size: str) -> np.ndarray:
    if size == "A2":
        rows, columns = 4, 5
        width, height = 0.42, 0.594
    else:
        rows, columns = 3, 4
        width, height = 0.297, 0.420
    return np.asarray(
        [
            (x, y, 0.0)
            for y in np.linspace(-height / 2.0, height / 2.0, rows)
            for x in np.linspace(-width / 2.0, width / 2.0, columns)
        ],
        dtype=np.float64,
    )


def _write_recording(
    path: Path,
    *,
    size: str,
    seed: int,
    frame_count: int = 120,
    marker_count_override: int | None = None,
    millimetres: bool = False,
) -> None:
    rng = np.random.default_rng(seed)
    base = _grid(size)
    if marker_count_override is not None:
        if marker_count_override > len(base):
            extra = np.repeat(base[-1:], marker_count_override - len(base), axis=0)
            base = np.concatenate((base, extra), axis=0)
        else:
            base = base[:marker_count_override]
    marker_count = len(base)
    time = np.arange(frame_count, dtype=np.float64) / 120.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Format Version", "1.23"])
        writer.writerow(["Capture Frame Rate", "120"])
        writer.writerow(
            ["", ""]
            + [field for marker in range(marker_count) for field in (str(marker + 1), "", "")]
        )
        writer.writerow(
            ["", ""] + [field for _ in range(marker_count) for field in ("Position", "", "")]
        )
        writer.writerow(
            ["Frame", "Time (Seconds)"]
            + [field for _ in range(marker_count) for field in ("X", "Y", "Z")]
        )
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for frame, timestamp in enumerate(time):
            translation = np.asarray(
                [
                    0.05 * np.sin(2.0 * np.pi * 0.7 * timestamp + phase),
                    0.02 * np.cos(2.0 * np.pi * 0.4 * timestamp),
                    0.03 * np.sin(2.0 * np.pi * 0.9 * timestamp),
                ]
            )
            deformation = np.column_stack(
                (
                    0.010 * np.sin(2.0 * np.pi * timestamp + 3.0 * base[:, 1]),
                    0.005 * np.sin(3.0 * np.pi * timestamp + 4.0 * base[:, 0]),
                    0.020 * np.sin(4.0 * np.pi * timestamp + 2.0 * base[:, 0] + 3.0 * base[:, 1]),
                )
            )
            positions = base + translation + deformation + 0.0002 * rng.normal(size=base.shape)
            if millimetres:
                positions = 1000.0 * positions
            writer.writerow([frame, timestamp, *positions.reshape(-1).tolist()])


def test_parse_publisher_style_recording_and_detect_units(tmp_path: Path) -> None:
    path = tmp_path / "Free-hanging" / "cotton_A3_shake_fast_hands.csv"
    _write_recording(path, size="A3", seed=1, millimetres=True)
    recording = MODULE.parse_recording(path, tmp_path)
    assert recording.size == "A3"
    assert recording.marker_count == 12
    assert recording.positions_m.shape == (120, 12, 3)
    assert recording.original_coordinate_scale_to_m == pytest.approx(0.001)
    samples = MODULE.make_samples(recording, _protocol())
    assert samples.observations_m.shape[1:] == (12, 3)
    assert samples.queries_m.shape[1:] == (3,)
    assert 0 < samples.retained_window_count <= 16


def test_rod_or_stick_marker_recording_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "Hitting" / "cotton_A2_hitting.csv"
    _write_recording(path, size="A2", seed=2, marker_count_override=22)
    with pytest.raises(ValueError, match="rod/stick"):
        MODULE.parse_recording(path, tmp_path)


def test_recording_disjoint_real_trajectory_study_builds_rank_frontier(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    for size, offset in (("A2", 0), ("A3", 100)):
        for index in range(6):
            _write_recording(
                dataset / "Free-hanging" / f"material_{size}_shake_fast_hands_{index}.csv",
                size=size,
                seed=offset + index,
            )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(_protocol(), sort_keys=True),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = MODULE.run(dataset, protocol_path, output, "a" * 40)
    assert result["status"] == "evaluated-real-rank-distortion"
    aggregate = result["aggregate"]
    assert aggregate["fold_count"] == 6
    assert aggregate["retained_ranks"] == [0, 1, 2, 3]
    assert aggregate["numerical_exact_ranks"] == [3]
    assert aggregate["original_shared_rank_min"] >= 36
    assert aggregate["maximum_optimality_violation"] < 1e-8

    for rank_record in aggregate["ranks"]:
        optimum = rank_record["methods"]["optimal_generalized_eigen"]
        response = rank_record["methods"]["response_svd"]
        covariance = rank_record["methods"]["covariance_pca"]
        assert (
            optimum["normalized_covariance_trace_loss_mean"]
            <= response["normalized_covariance_trace_loss_mean"] + 1e-8
        )
        assert (
            optimum["normalized_covariance_trace_loss_mean"]
            <= covariance["normalized_covariance_trace_loss_mean"] + 1e-8
        )

    exact = next(record for record in aggregate["ranks"] if record["retained_rank"] == 3)
    parity = aggregate["exact_rank_full_parity"]
    assert parity["maximum_relative_gain_error"] < 1e-8
    assert parity["maximum_relative_posterior_covariance_error"] < 1e-8
    assert parity["maximum_realized_mean_difference_m"] < 1e-10
    np.testing.assert_allclose(
        exact["methods"]["optimal_generalized_eigen"]["query_rmse_m"],
        aggregate["full"]["query_rmse_m"],
        atol=1e-12,
        rtol=1e-10,
    )
    assert aggregate["exact_rank_shared_factor_payload_reduction_ratio"] > 10.0
    assert not list(output.rglob("*.csv"))
    assert (output / "result.json").is_file()
    assert (output / "inventory.json").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "summary.md").is_file()


def test_balanced_hash_assignment_populates_all_folds() -> None:
    records = [
        MODULE.RecordingSamples(
            relative_path=f"recording-{index}.csv",
            size="A2",
            observations_m=np.zeros((1, 20, 3)),
            queries_m=np.zeros((1, 3)),
            horizon_seconds=np.ones(1),
            candidate_window_count=1,
            retained_window_count=1,
        )
        for index in range(11)
    ]
    assignments = MODULE._fold_assignments(records, 5)
    counts = [sum(value == fold for value in assignments.values()) for fold in range(5)]
    assert max(counts) - min(counts) <= 1
    assert set(assignments) == {record.relative_path for record in records}
