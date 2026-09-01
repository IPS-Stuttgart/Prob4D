from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_query_selective_source_support_v2.py"
PROTOCOL = ROOT / "protocols/dot-rope-query-selective-source-support-v2.json"

SPEC = importlib.util.spec_from_file_location("dot_source_support_v2_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_protocol_identity_and_clean_split() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(protocol)
    protocol_id = unsigned.pop("protocol_id")
    assert content_id(unsigned) == protocol_id
    assert protocol["source_sequences"] == [f"R{index:02d}" for index in range(11, 21)]
    assert protocol["confirmation_sequences"] == [
        f"R{index:02d}" for index in range(21, 31)
    ]
    assert protocol["reserved_sequences"] == "R31-R70"
    assert protocol["r04_r10_diagnostic"]["decision"] == "heldout-support-negative"
    assert protocol["information_boundary"]["r04_r10_reused_for_tuning"] is False
    assert protocol["information_boundary"]["r21_r30_payloads_opened"] is False
    assert protocol["support_selection"]["outcome_metrics_used_for_selection"] is False


def test_request_authorizes_source_only(tmp_path: Path) -> None:
    protocol_blob = "1" * 40
    request = {
        "schema": MODULE.REQUEST_SCHEMA,
        "schema_version": MODULE.SCHEMA_VERSION,
        "protocol_path": PROTOCOL.as_posix(),
        "protocol_git_blob_sha": protocol_blob,
        "r04_r10_diagnostic": {
            "decision": "heldout-support-negative",
            "result_id": MODULE.R04_RESULT_ID,
            "marker_support_id": MODULE.R04_SUPPORT_ID,
            "provider_bundle_id": MODULE.R04_PROVIDER_ID,
        },
        "source_sequences": MODULE.SOURCE_SEQUENCES,
        "confirmation_sequences": MODULE.CONFIRMATION_SEQUENCES,
        "reserved_sequences": MODULE.RESERVED_SEQUENCES,
        "normal_view_source_prediction_authorized": True,
        "source_marker_support_qualification_authorized": True,
        "confirmation_prediction_authorized": False,
        "confirmation_marker_access_authorized": False,
        "post_source_tuning_authorized": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "claim_boundary": "source-only test",
    }
    request["request_id"] = content_id(request)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    validated = MODULE.validate_request(path, PROTOCOL, protocol_blob)
    assert validated["request_id"] == request["request_id"]

    request["confirmation_prediction_authorized"] = True
    unsigned = {key: value for key, value in request.items() if key != "request_id"}
    request["request_id"] = content_id(unsigned)
    path.write_text(json.dumps(request), encoding="utf-8")
    try:
        MODULE.validate_request(path, PROTOCOL, protocol_blob)
    except ValueError as error:
        assert "exceeds the source-only boundary" in str(error)
    else:
        raise AssertionError("confirmation access must remain forbidden")


def test_source_selector_is_deterministic_and_support_first() -> None:
    summaries = [
        {
            "candidate_id": "more-conditioned",
            "supported_sequences": 8,
            "rank_six_sequences": 10,
            "worst_normalized_support_margin": 3.0,
            "worst_observable_condition_ratio": 0.5,
            "selected_frame_count": 6,
        },
        {
            "candidate_id": "more-supported",
            "supported_sequences": 9,
            "rank_six_sequences": 9,
            "worst_normalized_support_margin": 1.0,
            "worst_observable_condition_ratio": 0.01,
            "selected_frame_count": 8,
        },
    ]
    selected = MODULE._select_candidate_summaries(summaries)
    assert selected["candidate_id"] == "more-supported"


def test_workflow_never_names_confirmation_archive() -> None:
    workflow = (
        ROOT / ".github/workflows/dot-rope-query-selective-source-support-v2.yml"
    ).read_text(encoding="utf-8")
    assert "R11-20.zip" in workflow
    assert "R21-30.zip" not in workflow
    assert "gpuserver4090" in workflow
    assert "workstation1" in workflow
