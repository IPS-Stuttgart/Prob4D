from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "information_contract_v1"
    / "adapters"
    / "deform_dlo45_retrospective_v1.py"
)
SPEC = importlib.util.spec_from_file_location("deform_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def _trajectory(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(adapter.FRAME_COUNT, dtype=np.float64)
    result = np.zeros((adapter.FRAME_COUNT, adapter.NODE_COUNT, 3), dtype=np.float64)
    left = np.column_stack(
        (
            0.0003 * time,
            0.005 * np.sin(time / 41.0),
            0.12 + np.zeros_like(time),
        )
    )
    right = np.column_stack(
        (
            0.22 + 0.0002 * time,
            0.012 * np.sin(time / 53.0),
            0.12 + np.zeros_like(time),
        )
    )
    for node, weight in enumerate(np.linspace(0.0, 1.0, adapter.NODE_COUNT)):
        result[:, node] = (1.0 - weight) * left + weight * right
        result[:, node, 2] += 0.015 * np.sin(time / 19.0 + weight)
        result[:, node] += rng.normal(scale=1e-5, size=(adapter.FRAME_COUNT, 3))
    return result


def test_prediction_never_reads_held_internal_future() -> None:
    spec = adapter.WindowSpec(5, 25, 25, 0.85)
    first = _trajectory()
    second = first.copy()
    current = 29
    second[current + 1 : current + 26, adapter.INTERNAL] += 100.0

    prediction_a, truth_a = adapter._prediction_and_truth(first, current, spec)
    prediction_b, truth_b = adapter._prediction_and_truth(second, current, spec)

    np.testing.assert_array_equal(prediction_a, prediction_b)
    assert not np.array_equal(truth_a, truth_b)


def test_registered_future_action_nodes_are_causal_inputs() -> None:
    spec = adapter.WindowSpec(5, 25, 25, 0.85)
    first = _trajectory()
    second = first.copy()
    current = 29
    second[current + 1 : current + 26, adapter.ACTION_NODES, 0] += 0.02

    prediction_a, _ = adapter._prediction_and_truth(first, current, spec)
    prediction_b, _ = adapter._prediction_and_truth(second, current, spec)

    assert not np.array_equal(prediction_a, prediction_b)


def test_source_covariance_is_positive_definite() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(200, 3))
    covariance = adapter._regularized_covariance(values, 1e-10)
    np.linalg.cholesky(covariance)
    np.testing.assert_allclose(covariance, covariance.T)


def test_same_mean_controls_preserve_prediction() -> None:
    prediction = np.arange(60, dtype=np.float64).reshape(20, 3) / 1000.0
    covariance = adapter.SourceCovariance(
        local_m2=np.diag([1e-4, 2e-4, 3e-4]),
        shared_m2=np.diag([4e-4, 5e-4, 6e-4]),
        source_window_count=10,
        source_row_count=200,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        means = []
        for mode in (
            "full-source-fitted-low-rank",
            "same-mean-marginal-matched-diagonal",
            "same-mean-overconfident-scale-0.01",
        ):
            path = Path(directory) / f"{mode}.npz"
            adapter._write_submission_payload(
                path,
                prediction,
                covariance,
                mode=mode,
            )
            with np.load(path, allow_pickle=False) as archive:
                means.append(np.array(archive["prediction_mean_xyz_m"], copy=True))
                local = np.asarray(archive["conditional_covariance_m2"])
                np.linalg.cholesky(local)
        for mean in means:
            np.testing.assert_array_equal(mean, prediction)


def test_template_payload_key_detection_is_prefix_aware() -> None:
    case = {
        "case_id": "template",
        "submission_payload": "cases/template.npz",
        "submission_payload_sha256": "a" * 64,
        "challenge_payload_sha256": "b" * 64,
    }
    assert adapter._payload_keys(case) == (
        "submission_payload",
        "submission_payload_sha256",
    )
