from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d.diagnostics.sim3_linearization import (
    GaussianLinearizationAdequacyV1,
    LinearizationAdequacyThresholdsV1,
    write_linearization_certificate,
)
from prob4d.diagnostics.sim3_linearization_certificate import (
    gaussian_linearization_adequacy_from_dict,
    load_gaussian_linearization_adequacy,
    main,
)


def _certificate() -> GaussianLinearizationAdequacyV1:
    return GaussianLinearizationAdequacyV1(
        parameterization="sim3-left-perturbation",
        parameter_order=("scale", "rotation", "translation"),
        parameter_dimension=7,
        output_shape=(1, 3),
        sample_count=4096,
        batch_size=256,
        seed=17,
        finite_difference_step=1.0e-6,
        jacobian_validated=True,
        thresholds=LinearizationAdequacyThresholdsV1(),
        point_diagnostics=(
            {
                "item_index": 0,
                "relative_trace_error": 0.01,
                "relative_frobenius_error": 0.02,
                "mean_shift_standard_deviations": 0.03,
                "nonlinear_trace": 1.0,
                "linearized_trace": 1.01,
                "principal_axis_anisotropy": 1.1,
                "principal_axis_angle_degrees": None,
            },
        ),
        query_diagnostics={
            "query_dimension": 1,
            "relative_trace_error": 0.01,
            "relative_frobenius_error": 0.02,
            "mean_shift_standard_deviations": 0.03,
            "nonlinear_trace": 1.0,
            "linearized_trace": 1.01,
        },
        adequate=True,
        failure_reasons=(),
        metadata={
            "mean_transform_vector": [0.0] * 7,
            "point_count": 1,
            "perturbation_side": "left",
        },
    )


def test_strict_certificate_round_trip(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "certificate.json"
    write_linearization_certificate(path, certificate)

    loaded = load_gaussian_linearization_adequacy(path)

    assert loaded.gaussian_linearization_adequacy_id == (
        certificate.gaussian_linearization_adequacy_id
    )
    assert loaded.to_dict() == certificate.to_dict()
    assert main([str(path)]) == 0


def test_strict_certificate_rejects_identity_tampering() -> None:
    payload = _certificate().to_dict()
    payload["gaussian_linearization_adequacy_id"] = "0" * 64

    with pytest.raises(ValueError, match="identity mismatch"):
        gaussian_linearization_adequacy_from_dict(payload)


def test_strict_certificate_rejects_point_roster_tampering() -> None:
    payload = _certificate().to_dict()
    payload["point_diagnostics"] = []

    with pytest.raises(ValueError, match="count must match"):
        gaussian_linearization_adequacy_from_dict(payload)


def test_strict_certificate_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    payload = json.dumps(_certificate().to_dict(), sort_keys=True)
    path.write_text(payload[:-1] + ', "adequate": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_gaussian_linearization_adequacy(path)


def test_strict_certificate_replays_decision_from_metrics() -> None:
    certificate = _certificate()
    payload = certificate.to_dict()
    point = dict(payload["point_diagnostics"][0])
    point["relative_trace_error"] = 0.5
    payload["point_diagnostics"] = [point]
    forged = GaussianLinearizationAdequacyV1(
        parameterization=payload["parameterization"],
        parameter_order=tuple(payload["parameter_order"]),
        parameter_dimension=payload["parameter_dimension"],
        output_shape=tuple(payload["output_shape"]),
        sample_count=payload["sample_count"],
        batch_size=payload["batch_size"],
        seed=payload["seed"],
        finite_difference_step=payload["finite_difference_step"],
        jacobian_validated=payload["jacobian_validated"],
        thresholds=certificate.thresholds,
        point_diagnostics=(point,),
        query_diagnostics=payload["query_diagnostics"],
        adequate=True,
        failure_reasons=(),
        metadata=payload["metadata"],
    )

    with pytest.raises(ValueError, match="decision does not match diagnostics"):
        gaussian_linearization_adequacy_from_dict(forged.to_dict())


def test_strict_certificate_rejects_hidden_principal_axis_angle() -> None:
    certificate = _certificate()
    payload = certificate.to_dict()
    point = dict(payload["point_diagnostics"][0])
    point["principal_axis_anisotropy"] = 2.0
    point["principal_axis_angle_degrees"] = None
    forged = GaussianLinearizationAdequacyV1(
        parameterization=payload["parameterization"],
        parameter_order=tuple(payload["parameter_order"]),
        parameter_dimension=payload["parameter_dimension"],
        output_shape=tuple(payload["output_shape"]),
        sample_count=payload["sample_count"],
        batch_size=payload["batch_size"],
        seed=payload["seed"],
        finite_difference_step=payload["finite_difference_step"],
        jacobian_validated=payload["jacobian_validated"],
        thresholds=certificate.thresholds,
        point_diagnostics=(point,),
        query_diagnostics=payload["query_diagnostics"],
        adequate=True,
        failure_reasons=(),
        metadata=payload["metadata"],
    )

    with pytest.raises(ValueError, match="angle presence changed"):
        gaussian_linearization_adequacy_from_dict(forged.to_dict())
