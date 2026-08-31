from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/science/run_dot_rope_query_selective_heldout.py"
PROTOCOL = ROOT / "protocols/dot-rope-query-selective-heldout-v1.json"
MODULE_NAME = "dot_rope_query_selective_heldout_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


def test_protocol_is_content_addressed_and_target_closed() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id")
    assert content_id(unsigned) == protocol_id
    assert value["target_sequences"] == [f"R{index:02d}" for index in range(11, 31)]
    assert value["reserved_sequences"] == "R31-R70"
    assert value["archives"] == [
        {
            "name": "R11-20.zip",
            "md5": "23ce3e7067465d3edabe20b4c7cfa388",
            "sequences": [f"R{index:02d}" for index in range(11, 21)],
        },
        {
            "name": "R21-30.zip",
            "md5": "8aee77f79d1aff6e1f3fd21886b251a0",
            "sequences": [f"R{index:02d}" for index in range(21, 31)],
        },
    ]
    assert value["prerequisite"]["required_decision"] == "heldout-strong-positive"
    assert value["information_boundary"]["r04_r10_confirmation_reused_for_tuning"] is False
    assert value["evaluation"]["means_and_admission_sealed_before_3d_marker_access"] is True


def _request(protocol_blob: str) -> dict:
    value = {
        "schema": MODULE.REQUEST_SCHEMA,
        "schema_version": 1,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_git_blob_sha": protocol_blob,
        "target_sequences": list(MODULE.TARGET_SEQUENCES),
        "reserved_sequences": MODULE.RESERVED_SEQUENCES,
        "prerequisite": {
            "protocol_id": MODULE.PREREQUISITE_PROTOCOL_ID,
            "source_calibration_id": MODULE.SOURCE_CALIBRATION_ID,
            "run_id": 33363832286,
            "artifact_id": 123456789,
            "artifact_name": "dot-rope-heldout-confirmation-result",
            "artifact_digest": "sha256:" + "a" * 64,
            "evaluation_id": "b" * 64,
            "marker_support_id": "c" * 64,
            "decision": MODULE.PREREQUISITE_DECISION,
        },
        "normal_view_prediction_authorized": True,
        "marker_2d_factor_seal_authorized": True,
        "marker_3d_scoring_authorized": True,
        "post_open_tuning_authorized": False,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "claim_boundary": "frozen test request",
    }
    value["request_id"] = content_id(value)
    return value


def test_request_binds_strong_prerequisite_and_recomputes_identity(tmp_path: Path) -> None:
    protocol_blob = "1" * 40
    value = _request(protocol_blob)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    validated = MODULE.validate_request(
        path,
        PROTOCOL.relative_to(ROOT),
        protocol_blob,
    )
    assert validated["request_id"] == value["request_id"]
    assert validated["prerequisite"]["decision"] == "heldout-strong-positive"

    value["prerequisite"]["decision"] = "heldout-directional-positive"
    value["request_id"] = content_id(
        {key: item for key, item in value.items() if key != "request_id"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="strong-positive"):
        MODULE.validate_request(path, PROTOCOL.relative_to(ROOT), protocol_blob)


def _synthetic_rows(sequence_count: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sequence_records = []
    for index in range(sequence_count):
        sequence = f"R{11 + index:02d}"
        sequence_records.append(
            {
                "sequence": sequence,
                "support": "supported",
                "queries": {
                    "centerline_centroid": {
                        "admitted": True,
                        "predictions": {
                            "query_aware": {"exact_fallback": False}
                        },
                    },
                    "off_axis_probe": {
                        "admitted": False,
                        "predictions": {
                            "query_aware": {"exact_fallback": True}
                        },
                    },
                },
            }
        )
        for query in MODULE.QUERIES:
            for method in MODULE.METHODS:
                if query == "centerline_centroid":
                    rmse = 0.20 if method == "physical_fallback" else 0.10
                    nll = 2.0 if method == "physical_fallback" else 1.0
                    exact = method == "physical_fallback"
                    accepted = method not in {"physical_fallback"}
                else:
                    rmse = 0.20
                    nll = 2.0
                    exact = method in {
                        "physical_fallback",
                        "full_rank_only",
                        "query_aware",
                    }
                    accepted = not exact
                    if method == "observable_subspace_unconditional":
                        rmse = 0.24
                        nll = 3.0
                rows.append(
                    {
                        "sequence": sequence,
                        "query": query,
                        "method": method,
                        "rmse_fraction_of_provider_span": rmse,
                        "normalized_nll_per_dimension": nll,
                        "normalized_nees": 1.0,
                        "covered_90": True,
                        "covered_95": True,
                        "harmful_vs_fallback": rmse > 0.20,
                        "accepted": accepted,
                        "exact_fallback": exact,
                    }
                )
    return rows, {"sequence_records": sequence_records}


def test_registered_classification_is_complete_sequence_based() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows, seal = _synthetic_rows(18)
    aggregate, comparisons = MODULE._aggregate_rows(rows, protocol)
    decision, checks = MODULE._classification(
        protocol,
        seal,
        rows,
        aggregate,
        comparisons,
    )
    assert decision == "query-selective-strong-positive"
    assert all(checks.values())
    centroid = comparisons["centerline_centroid"]["query_aware"]
    assert centroid["fallback_minus_method_rmse"]["lower_95"] > 0.0
    assert centroid["fallback_minus_method_nll"]["lower_95"] > 0.0


def test_deterministic_normal_is_orthogonal_and_repeatable() -> None:
    points = np.array([[float(index), 0.0, 0.0] for index in range(12)])
    first = MODULE._deterministic_normal(points)
    second = MODULE._deterministic_normal(points)
    np.testing.assert_allclose(first, second)
    assert abs(float(first @ (points[-1] - points[0]))) < 1e-12
