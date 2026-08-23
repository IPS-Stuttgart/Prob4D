from __future__ import annotations

import json
from pathlib import Path

import pytest

from prob4d._cut3r_recovery_v2_exact import (
    _exact_group_bootstrap,
    _leave_one_group_out,
    _triplet_v2,
)
from prob4d._cut3r_recovery_v2_report import (
    _build_v2_from_validated_v1_report,
)
from prob4d._cut3r_recovery_v2_spec import (
    build_cut3r_recurrent_state_recovery_v2_specification,
    load_cut3r_recurrent_state_recovery_v2_specification,
)
from prob4d.prediction_cli import main as prediction_main

_SPECIFICATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "cut3r_recurrent_state_recovery_v2.json"
)
_METRICS = (
    "point_rmse_m",
    "endpoint_rmse_m",
    "proper_score",
    "seam_rmse_m",
    "absolute_drift_slope_m_per_frame",
)


def _metric_values(native: float, restarted: float, fused: float) -> dict[str, object]:
    return {
        "native_continuous": native,
        "restarted_newest": restarted,
        "restarted_prob4d_fused": fused,
        "prob4d_gain": restarted - fused,
        "recurrence_gap": restarted - native,
        "recovery_fraction": (restarted - fused) / (restarted - native),
        "status": "defined",
    }


def _validated_v1_report() -> dict[str, object]:
    groups = [
        {
            "group_id": group_id,
            "technical_failure_code": None,
            "metrics": {
                metric: _metric_values(0.6, 1.0, 0.8) for metric in _METRICS
            },
        }
        for group_id in ("source-a", "source-b")
    ]
    return {
        "comparison_lock_id": "a" * 64,
        "comparison_protocol_name": "v2-unit-test",
        "group_unit": "complete-object-or-acquisition-session",
        "source_evaluation_group_ids": ["source-a", "source-b"],
        "group_count": 2,
        "evaluable_group_count": 2,
        "technical_failure_count": 0,
        "evidence": {
            "fusion_source_competence_report_v2_id": "b" * 64,
            "recurrence_source_competence_report_v2_id": "c" * 64,
            "byte_identical_restarted_newest_rows": True,
        },
        "recovery_definition": (
            "(restarted-newest - restarted-prob4d-fused) / "
            "(restarted-newest - native-continuous)"
        ),
        "groups": groups,
        "recurrent_state_recovery_report_id": "d" * 64,
    }


def test_checked_in_v2_specification_is_content_addressed_and_positive() -> None:
    specification = load_cut3r_recurrent_state_recovery_v2_specification(
        _SPECIFICATION_PATH
    )

    assert len(specification["analysis_specification_id"]) == 64
    assert specification["maximum_exact_group_count"] == 10
    assert specification["leave_one_group_out"] is True
    assert specification["source_outcomes_opened_before_specification"] is False
    assert specification["target_access"] == "forbidden"
    assert all(
        value > 0.0
        for value in specification["minimum_recurrence_gap_by_metric"].values()
    )


def test_v2_specification_rejects_zero_recurrence_gap_floor() -> None:
    raw = json.loads(_SPECIFICATION_PATH.read_text(encoding="utf-8"))
    raw.pop("analysis_specification_id")
    raw["minimum_recurrence_gap_by_metric"]["endpoint_rmse_m"] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        build_cut3r_recurrent_state_recovery_v2_specification(raw)


def test_exact_group_bootstrap_enumerates_all_ordered_resamples() -> None:
    result = _exact_group_bootstrap(
        [(0.6, 1.0, 0.8), (0.6, 1.0, 0.8)],
        denominator_threshold=1e-4,
        confidence_level=0.95,
        minimum_valid_denominator_probability=0.8,
        maximum_exact_group_count=10,
        point_status="defined",
    )

    assert result["count_vector_count"] == 3
    assert result["ordered_resample_count"] == 4
    assert result["valid_denominator_probability"] == 1.0
    assert result["prob4d_gain_interval"]["lower"] == pytest.approx(0.2)
    assert result["prob4d_gain_interval"]["upper"] == pytest.approx(0.2)
    assert result["recovery_fraction_interval"]["lower"] == pytest.approx(0.5)
    assert result["recovery_fraction_interval"]["upper"] == pytest.approx(0.5)


def test_exact_group_bootstrap_covers_the_frozen_ten_group_limit() -> None:
    result = _exact_group_bootstrap(
        [(0.6, 1.0, 0.8)] * 10,
        denominator_threshold=1e-4,
        confidence_level=0.95,
        minimum_valid_denominator_probability=0.8,
        maximum_exact_group_count=10,
        point_status="defined",
    )

    assert result["count_vector_count"] == 92378
    assert result["ordered_resample_count"] == 10**10
    assert result["valid_denominator_probability"] == 1.0


def test_ratio_interval_is_withheld_when_denominator_support_is_unstable() -> None:
    result = _exact_group_bootstrap(
        [(0.5, 1.0, 0.8), (1.2, 1.0, 0.8)],
        denominator_threshold=1e-4,
        confidence_level=0.95,
        minimum_valid_denominator_probability=0.8,
        maximum_exact_group_count=10,
        point_status="defined",
    )

    assert result["valid_denominator_probability"] == pytest.approx(0.75)
    assert (
        result["recovery_fraction_interval"]["interval_status"]
        == "insufficient-valid-denominator-probability"
    )
    assert result["recovery_fraction_interval"]["lower"] is None
    assert result["prob4d_gain_interval"]["lower"] is not None
    assert result["recurrence_gap_interval"]["lower"] is not None


def test_recovery_is_undefined_below_practical_separation_floor() -> None:
    result = _triplet_v2(
        0.99995,
        1.0,
        0.8,
        denominator_threshold=1e-4,
    )

    assert result["status"] == (
        "undefined-recurrence-gap-not-practically-separated"
    )
    assert result["recovery_fraction"] is None
    assert result["prob4d_gain"] == pytest.approx(0.2)


def test_nonfinite_derived_gain_fails_closed() -> None:
    with pytest.raises(ValueError, match="Prob4D gain must remain finite"):
        _triplet_v2(
            0.0,
            1e308,
            -1e308,
            denominator_threshold=1.0,
        )


def test_leave_one_group_out_reports_gain_sign_reversal() -> None:
    point = _triplet_v2(0.6, 1.0, 0.8, denominator_threshold=1e-4)
    result = _leave_one_group_out(
        ["source-a", "source-b"],
        [(0.6, 1.0, 0.5), (0.6, 1.0, 1.1)],
        denominator_threshold=1e-4,
        point=point,
    )

    assert result["summary"]["prob4d_gain_sign_reversal"] is True
    assert result["summary"]["minimum_prob4d_gain"] == pytest.approx(-0.1)
    assert result["summary"]["maximum_prob4d_gain"] == pytest.approx(0.5)


def test_v2_payload_foregrounds_gain_and_is_deterministic() -> None:
    specification = load_cut3r_recurrent_state_recovery_v2_specification(
        _SPECIFICATION_PATH
    )
    validated_v1 = _validated_v1_report()

    first = _build_v2_from_validated_v1_report(validated_v1, specification)
    second = _build_v2_from_validated_v1_report(validated_v1, specification)

    assert first == second
    assert first["primary_endpoint"] == "prob4d_gain"
    assert first["exact_small_sample_inference"] is True
    assert len(first["recurrent_state_recovery_v2_report_id"]) == 64
    endpoint = first["aggregate"]["endpoint_rmse_m"]
    assert endpoint["prob4d_gain"] == pytest.approx(0.2)
    assert endpoint["recovery_fraction"] == pytest.approx(0.5)
    assert endpoint["exact_group_bootstrap"]["ordered_resample_count"] == 4


def test_prediction_cli_dispatches_cut3r_recovery_v2_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        prediction_main(["cut3r-recovery-v2", "--help"])

    assert caught.value.code == 0
    assert "denominator-safe" in capsys.readouterr().out.lower()
