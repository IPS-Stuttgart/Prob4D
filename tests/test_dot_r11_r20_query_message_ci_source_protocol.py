from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/dot-r11-r20-query-message-ci-source-v1.json"


def _canonical_id(value: dict[str, object]) -> str:
    payload = dict(value)
    protocol_id = payload.pop("protocol_id")
    assert isinstance(protocol_id, str)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_dot_query_message_source_protocol_is_rank_agnostic_and_target_closed() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert value["schema"] == "prob4d.dot-r11-r20-query-message-ci-source-protocol"
    assert value["schema_version"] == 1
    assert value["stage"] == "development-only-source"
    assert value["protocol_id"] == _canonical_id(value)

    dataset = value["dataset"]
    assert dataset["source_sequences"] == [f"R{index:02d}" for index in range(11, 21)]
    assert dataset["confirmation_sequences"] == [
        f"R{index:02d}" for index in range(21, 31)
    ]
    assert dataset["reserved_sequences"] == "R31-R70"
    assert dataset["source_archive"] == {
        "name": "R11-20.zip",
        "md5": "23ce3e7067465d3edabe20b4c7cfa388",
    }

    provider = value["sealed_provider"]
    assert provider["workflow_run_id"] == 33552798863
    assert provider["execution_revision"] == "c64765ea766e667a566e1b565e8ed01ffd734e53"
    assert provider["artifact_id"] == 9818146750
    assert (
        provider["artifact_digest"]
        == "sha256:70c2cec1cf33b65ae6653a3839fb4f74d57023d1d49cced6c61e6948f4d7b8a6"
    )
    assert (
        provider["provider_seal_id"]
        == "38ea78e8bf44cbeaedeeadaee862af3cc6369d35d7e3b5a2b5fac0f020c7145b"
    )
    assert provider["source_marker_payloads_opened_when_predictions_were_sealed"] is False
    assert provider["confirmation_payloads_opened"] is False
    routed = {
        sequence: component["camera"]
        for component in provider["components"]
        for sequence in component["sequences"]
    }
    assert routed == {
        "R11": "cam005",
        "R12": "cam005",
        "R13": "cam001",
        "R14": "cam001",
        "R15": "cam001",
        "R16": "cam001",
        "R17": "cam002",
        "R18": "cam001",
        "R19": "cam001",
        "R20": "cam005",
    }

    diagnostic = value["terminal_rank_diagnostic"]
    assert diagnostic["decision"] == "camera-routing-provider-rank-negative"
    assert diagnostic["factor_rank_counts"] == {"6": 1, "7": 9}
    assert diagnostic["supported_sequences"] == ["R18"]
    assert value["promotion"]["rank_six_required"] is False
    assert value["promotion"]["automatic_confirmation_authorization"] is False

    contract = value["message_contract"]
    assert contract["implementation"] == "prob4d.query_message"
    assert contract["same_query_required"] is True
    assert contract["byte_identical_anchor_required"] is True
    assert contract["negative_information_rejected"] is True
    assert contract["observation_likelihood_preserved"] is False
    assert contract["posterior_parity_relative_tolerance"] <= 1e-9
    assert contract["duplicate_message_mean_relative_tolerance"] <= 1e-12
    assert contract["duplicate_message_covariance_relative_tolerance"] <= 1e-12

    primary = value["source_methods"]["primary"]
    assert primary["id"] == "two_window_query_ci_equal"
    assert primary["messages"] == ["window_a", "window_b"]
    assert primary["weights"] == [0.5, 0.5]
    assert sum(primary["weights"]) + primary["prior_weight"] == 1.0
    assert primary["weight_selection_uses_source_outcomes"] is False
    assert "naive_independent_message_sum" in value["source_methods"]["comparators"]
    assert "continuous_provider_query" in value["source_methods"]["comparators"]

    boundary = value["information_boundary"]
    assert boundary == {
        "source_sequences_previously_opened": True,
        "source_provider_may_be_reused_without_rerun": True,
        "confirmation_access_authorized": False,
        "r21_r30_payloads_opened": False,
        "r31_r70_payloads_opened": False,
        "execution_request_present": False,
        "target_side_retuning_allowed": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
    }
