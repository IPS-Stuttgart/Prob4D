from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/query-message-overlap-controlled-v1"
PROTOCOL = ROOT / "protocols/query-message-overlap-study-v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_id(value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    identifier = unsigned.pop(field)
    assert isinstance(identifier, str)
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == identifier
    return identifier


def test_retained_query_message_overlap_evidence_is_byte_bound() -> None:
    assert {path.name for path in EVIDENCE.iterdir()} == {
        "README.md",
        "manifest.json",
        "summary.json",
    }
    manifest = _load(EVIDENCE / "manifest.json")
    summary = _load(EVIDENCE / "summary.json")
    protocol = _load(PROTOCOL)

    assert manifest["schema"] == "prob4d.query-message-overlap-controlled-evidence.v1"
    assert manifest["schema_version"] == 1
    assert manifest["decision"] == "controlled-overlap-passed"
    assert manifest["protocol_id"] == _content_id(protocol, "protocol_id")
    assert manifest["result_id"] == summary["result_id"]
    assert summary["decision"] == manifest["decision"]
    assert summary["protocol_id"] == manifest["protocol_id"]

    committed = manifest["committed_files_sha256"]
    assert committed == {
        "protocols/query-message-overlap-study-v1.json": _sha256(PROTOCOL),
        "summary.json": _sha256(EVIDENCE / "summary.json"),
    }
    payloads = manifest["artifact_payloads_sha256"]
    assert payloads == {
        "protocols/query-message-overlap-study-v1.json": (
            "231f22630ffdcf05d81ee445f6fcf185b34e761303307a9fb9128f8bc63748fb"
        ),
        "result.json": (
            "5a75ae8e32a23ec022e5509b365e17044b86924b29af4fad582ad26747fcc978"
        ),
        "summary.json": (
            "5c1370b5029219098356f4f446d7620d1ca1d02d185368bd46b7cb69ea482787"
        ),
    }
    assert committed["summary.json"] == payloads["summary.json"]
    assert committed["protocols/query-message-overlap-study-v1.json"] == (
        payloads["protocols/query-message-overlap-study-v1.json"]
    )

    source = manifest["source"]
    assert source == {
        "repository": "IPS-Stuttgart/Prob4D",
        "evaluated_head_sha": "8b44e274f4d4dfddec643a18cd5d0f21551b87eb",
        "workflow_run_id": 33565333587,
        "workflow_id": 347918200,
        "artifact_id": 9822877862,
        "artifact_name": (
            "query-message-overlap-study-v1-"
            "08440588fd9b4078fbaca698c6c769964d8f321c"
        ),
        "artifact_sha256": (
            "c5a1a911c87701b57ef0eeb7002a3b69b474867c4a6842b05e80f8f560659b34"
        ),
    }

    assert manifest["accounting"] == {
        "correlation_count": 6,
        "sample_count_per_correlation": 65536,
        "complete_case_count": 393216,
        "independent_unit": "simulated_complete_query_case",
    }


def test_retained_query_message_overlap_summary_preserves_claim_boundary() -> None:
    manifest = _load(EVIDENCE / "manifest.json")
    summary = _load(EVIDENCE / "summary.json")["summary"]

    assert summary["naive_high_correlation_minimum_normalized_nees"] > 1.59
    assert summary["naive_high_correlation_maximum_coverage"] < 0.73
    assert summary["ci_maximum_normalized_nees"] < 1.0
    assert summary["ci_minimum_coverage"] > 0.9
    assert summary["ci_improves_over_either_single_window"] is True
    assert summary["maximum_duplicate_mean_error"] == 0.0
    assert summary["maximum_duplicate_covariance_error"] == 0.0
    assert summary["maximum_api_covariance_error"] == 0.0
    assert summary["maximum_api_mean_error"] < 1e-15

    assert manifest["information_boundary"] == {
        "controlled_linear_gaussian_only": True,
        "dataset_payload_opened": False,
        "dot_source_executed": False,
        "dot_confirmation_access_authorized": False,
        "learned_provider_calibration_established": False,
        "real_data_utility_established": False,
        "robot_safety_established": False,
    }
