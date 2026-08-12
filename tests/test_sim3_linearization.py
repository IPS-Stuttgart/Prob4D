from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.diagnostics.sim3_linearization import (
    LinearizationAdequacyThresholdsV1,
    assess_gaussian_linearization,
    assess_sim3_linearization,
    main,
    write_linearization_certificate,
)
from prob4d.sim3 import Sim3


def _small_covariance() -> np.ndarray:
    return np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5])


def _loose_thresholds() -> LinearizationAdequacyThresholdsV1:
    return LinearizationAdequacyThresholdsV1(
        maximum_relative_trace_error=0.20,
        maximum_relative_frobenius_error=0.20,
        maximum_mean_shift_standard_deviations=0.20,
        maximum_principal_axis_angle_degrees=20.0,
        maximum_query_relative_trace_error=0.20,
        maximum_query_relative_frobenius_error=0.20,
        maximum_query_mean_shift_standard_deviations=0.20,
    )


def test_small_sim3_perturbation_is_adequate_and_deterministic() -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    first = assess_sim3_linearization(
        Sim3.identity(),
        _small_covariance(),
        points,
        thresholds=_loose_thresholds(),
        sample_count=4096,
        batch_size=257,
        seed=4,
    )
    second = assess_sim3_linearization(
        Sim3.identity(),
        _small_covariance(),
        points,
        thresholds=_loose_thresholds(),
        sample_count=4096,
        batch_size=257,
        seed=4,
    )

    assert first.adequate
    assert first.failure_reasons == ()
    assert first.to_dict() == second.to_dict()
    assert len(first.point_diagnostics) == 2


def test_zero_tolerances_fail_closed() -> None:
    certificate = assess_sim3_linearization(
        Sim3.identity(),
        _small_covariance(),
        np.array([[1.0, 2.0, 3.0]]),
        thresholds=LinearizationAdequacyThresholdsV1(
            maximum_relative_trace_error=0.0,
            maximum_relative_frobenius_error=0.0,
            maximum_mean_shift_standard_deviations=0.0,
            maximum_principal_axis_angle_degrees=0.0,
            maximum_query_relative_trace_error=0.0,
            maximum_query_relative_frobenius_error=0.0,
            maximum_query_mean_shift_standard_deviations=0.0,
        ),
        sample_count=512,
        seed=1,
    )

    assert not certificate.adequate
    assert certificate.failure_reasons


def test_supplied_jacobian_is_independently_checked() -> None:
    points = np.array([[1.0, 0.0, 0.0]])
    bad_jacobian = np.zeros((1, 3, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="supplied Jacobian does not match"):
        assess_sim3_linearization(
            Sim3.identity(),
            _small_covariance(),
            points,
            jacobian=bad_jacobian,
            sample_count=32,
        )


def test_generic_linear_evaluator_supports_query_projection() -> None:
    matrix = np.array([[2.0, -1.0], [0.5, 3.0]])

    def evaluator(parameters: np.ndarray) -> np.ndarray:
        return (matrix @ parameters)[None, :]

    certificate = assess_gaussian_linearization(
        np.zeros(2),
        np.diag([0.2, 0.1]),
        evaluator,
        query_projection=np.array([[1.0, -0.25]]),
        thresholds=LinearizationAdequacyThresholdsV1(
            maximum_relative_trace_error=0.20,
            maximum_relative_frobenius_error=0.20,
            maximum_mean_shift_standard_deviations=0.20,
            maximum_principal_axis_angle_degrees=20.0,
            maximum_query_relative_trace_error=0.20,
            maximum_query_relative_frobenius_error=0.20,
            maximum_query_mean_shift_standard_deviations=0.20,
        ),
        sample_count=4096,
        seed=5,
    )

    assert certificate.adequate
    assert certificate.query_diagnostics is not None
    assert certificate.query_diagnostics["query_dimension"] == 1


def test_certificate_writer_is_no_clobber(tmp_path: Path) -> None:
    certificate = assess_sim3_linearization(
        Sim3.identity(),
        _small_covariance(),
        np.array([[1.0, 0.0, 0.0]]),
        thresholds=_loose_thresholds(),
        sample_count=512,
        seed=3,
    )
    destination = tmp_path / "certificate.json"
    write_linearization_certificate(destination, certificate)
    with pytest.raises(FileExistsError):
        write_linearization_certificate(destination, certificate)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["gaussian_linearization_adequacy_id"] == (
        certificate.gaussian_linearization_adequacy_id
    )


def test_cli_writes_certificate_and_can_fail_on_inadequate(tmp_path: Path) -> None:
    input_path = tmp_path / "case.npz"
    output_path = tmp_path / "certificate.json"
    np.savez(
        input_path,
        mean_transform=np.zeros(7),
        covariance=_small_covariance(),
        points=np.array([[1.0, 0.0, 0.0]]),
    )

    status = main(
        [
            str(input_path),
            "--output",
            str(output_path),
            "--samples",
            "512",
            "--max-relative-trace-error",
            "0",
            "--max-relative-frobenius-error",
            "0",
            "--max-mean-shift-std",
            "0",
            "--max-principal-axis-angle-deg",
            "0",
            "--fail-on-inadequate",
        ]
    )

    assert status == 2
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["adequate"] is False
