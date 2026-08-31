from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence/deform-dlo45-query-observability-heldout-v1"
SUMMARY_PATH = EVIDENCE_DIR / "summary.json"
VALIDATION_PATH = EVIDENCE_DIR / "validation-manifest.json"
EVALUATION_REQUEST_PATH = (
    ROOT
    / "protocols/execution_requests/"
    "deform_dlo45_query_observability_eval_v1.json"
)
SOURCE_GATE_REQUEST_PATH = (
    ROOT
    / "protocols/execution_requests/"
    "deform_dlo45_query_gate_source_v1.json"
)
RESULT_ID = "1ac8cd083b39877888ea0eb2f4b9400ca89eda09436f25f5f0a6f43b154b1007"
SOURCE_GEOMETRY_RESULT_ID = (
    "04f0df72492e97de2b16b7db57da707c97a37c1fcc545f6e6853f60498fe58a9"
)
SOURCE_GATE_RESULT_ID = (
    "a3b48a522e509e53935cc42c9f1cd293cd5f7753057979c99c743a10d74c14e2"
)


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    assert isinstance(value, dict)
    return value


def _mean(summary: dict[str, Any], query: str, method: str, metric: str) -> float:
    return float(summary["equal_file_results"][query][method][metric]["mean"])


def test_evidence_identity_and_information_order_are_frozen() -> None:
    summary = _load(SUMMARY_PATH)
    validation = _load(VALIDATION_PATH)
    evaluation_request = _load(EVALUATION_REQUEST_PATH)
    source_gate_request = _load(SOURCE_GATE_REQUEST_PATH)

    assert summary["heldout_result_id"] == RESULT_ID
    assert validation["frozen_heldout_result_id"] == RESULT_ID
    assert summary["decision"] == validation["expected_decision"] == "pass"
    assert summary["accounting"]["independent_groups"] == 28
    assert validation["expected_independent_groups"] == 28
    assert all(summary["criteria"].values())

    source_freeze = summary["source_freeze"]
    assert source_freeze["source_geometry_result_id"] == SOURCE_GEOMETRY_RESULT_ID
    assert source_freeze["source_gate_result_id"] == SOURCE_GATE_RESULT_ID
    assert evaluation_request["source_geometry_result_id"] == SOURCE_GEOMETRY_RESULT_ID
    assert evaluation_request["source_gate_result_id"] == SOURCE_GATE_RESULT_ID
    assert validation["expected_source_geometry_result_id"] == SOURCE_GEOMETRY_RESULT_ID
    assert validation["expected_source_gate_result_id"] == SOURCE_GATE_RESULT_ID
    assert source_freeze["rank_threshold"] == evaluation_request["rank_threshold"]
    assert source_freeze["query_gate"] == evaluation_request["query_gate"]
    assert source_gate_request["rank_threshold"] == source_freeze["rank_threshold"]

    boundary = summary["information_boundary"]
    assert boundary["source_gate_frozen_before_opening"] is True
    assert boundary["evaluation_outcomes_opened"] is True
    assert boundary["post_open_retuning_permitted"] is False
    assert validation["post_open_retuning_permitted"] is False
    assert boundary == evaluation_request["information_boundary"]


def test_query_aware_result_has_the_registered_selective_behavior() -> None:
    summary = _load(SUMMARY_PATH)

    centroid_query_aware_rmse = _mean(
        summary,
        "segment_centroid",
        "query_aware",
        "rmse_mm",
    )
    centroid_fallback_rmse = _mean(
        summary,
        "segment_centroid",
        "physical_fallback",
        "rmse_mm",
    )
    assert centroid_query_aware_rmse < centroid_fallback_rmse
    assert _mean(
        summary,
        "segment_centroid",
        "query_aware",
        "accepted_fraction",
    ) == 1.0
    assert _mean(
        summary,
        "segment_centroid",
        "query_aware",
        "harmful_fraction_vs_fallback",
    ) == 0.0

    assert _mean(
        summary,
        "off_axis_probe",
        "query_aware",
        "accepted_fraction",
    ) == 0.0
    assert _mean(
        summary,
        "off_axis_probe",
        "query_aware",
        "exact_fallback_fraction",
    ) == 1.0
    assert _mean(
        summary,
        "off_axis_probe",
        "query_aware",
        "harmful_fraction_vs_fallback",
    ) == 0.0
    assert _mean(
        summary,
        "off_axis_probe",
        "observable_subspace_unconditional",
        "harmful_fraction_vs_fallback",
    ) > 0.0
    assert _mean(
        summary,
        "off_axis_probe",
        "invalid_full_rank_completion",
        "mean_gaussian_nll",
    ) > _mean(
        summary,
        "off_axis_probe",
        "physical_fallback",
        "mean_gaussian_nll",
    )


def test_calibration_limitation_remains_explicit_and_machine_readable() -> None:
    summary = _load(SUMMARY_PATH)
    limitation = summary["limitations"]["accepted_centroid_covariance_underdispersed"]

    assert limitation["nominal_90pct_coverage"]["mean"] < 0.9
    assert limitation["normalized_nees"]["mean"] > 1.0
    assert summary["limitations"]["post_open_retuning_permitted"] is False
    assert summary["limitations"]["provider_competence_tested"] is False
    assert "not learned-provider competence" in _load(VALIDATION_PATH)[
        "claim_boundary"
    ]
