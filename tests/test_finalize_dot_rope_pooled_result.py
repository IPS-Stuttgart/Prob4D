from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "finalize_dot_rope_pooled_result.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dot_result_finalizer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_bundle(module, root: Path) -> tuple[dict, dict]:
    result = {
        "schema": module.SOURCE_RESULT_SCHEMA,
        "schema_version": 1,
        "decision": module.SOURCE_DECISION,
        "request_id": "1" * 64,
        "provider_bundle_id": "2" * 64,
        "provider_prob4d_revision": "3" * 40,
        "evaluator_prob4d_revision": "4" * 40,
        "predecessor_evaluation_id": "5" * 64,
        "runtime_artifact_id": "6" * 64,
        "protocol_id": "7" * 64,
        "marker_support_id": "8" * 64,
        "information_boundary": {
            "opened_sequences": ["R01", "R02", "R03"],
            "reserved_sequences": "R04-R70",
            "target_payloads_opened": False,
        },
        "marker_sampling": {"selected_coordinate_candidate": "columns-0-1:pixel-zero-based"},
        "marker_support_audit": {"audit_id": "9" * 64, "run_id": 10},
        "aggregate_methods": [
            {
                "method": "pointwise_quadratic",
                "mean_normalized_nll_per_dimension": 0.75,
                "covered_95_count": 3,
                "sequence_count": 3,
                "mean_predictive_sd_fraction_of_span": 1.0,
            }
        ],
        "method_rows": [{"method": "pointwise_quadratic", "mahalanobis": 2.0}],
        "sequences": [
            {
                "sequence": "R01",
                "point_metrics": {
                    "continuous_rmse_fraction_of_span": 0.1,
                    "identity_stitch_rmse_fraction_of_span": 0.2,
                    "estimated_stitch_rmse_fraction_of_span": 0.15,
                    "oracle_window_rmse_fraction_of_span": 0.05,
                },
            }
        ],
        "opened_marker_members": [{"member": "R01/coordinates/2d/frame000001_cam001.txt"}],
        "claim_boundary": "source-only",
    }
    result["evaluation_id"] = content_id(result)
    support = {
        "schema": "prob4d.dot-rope-cut3r-pooled-marker-support",
        "schema_version": 1,
        "request_id": "a" * 64,
        "repository_revision": "4" * 40,
        "marker_support_audit": {"audit_id": "9" * 64, "run_id": 10},
        "information_boundary": {"target_payloads_opened": False},
    }
    support["support_id"] = content_id(support)
    result_dir = root / "result"
    _write_json(result_dir / "result.json", result)
    _write_json(result_dir / "marker-support.json", support)
    (result_dir / "method-summary.csv").write_text(
        "method,sequence_count\npointwise_quadratic,3\n",
        encoding="utf-8",
    )
    (result_dir / "sequence-methods.csv").write_text(
        "sequence,method\nR01,pointwise_quadratic\n",
        encoding="utf-8",
    )
    (result_dir / "summary.md").write_text(
        "# Source\n\nEvaluation ID: `stale`\n",
        encoding="utf-8",
    )
    _write_json(root / "official-archive-metadata.json", {"filename": "R01-10.zip"})
    return result, support


def _request(module, source_result: dict) -> dict:
    request = {
        "claim_boundary": "metadata-only finalization",
        "execution_nonce": "test-1",
        "marker_support_audit_id": "9" * 64,
        "no_dataset_access": True,
        "no_scores_recomputed": True,
        "pooled_request_id": "a" * 64,
        "provider_artifact_name": "sealed-provider",
        "provider_bundle_id": "2" * 64,
        "provider_request_id": "1" * 64,
        "provider_run_id": 11,
        "schema": module.REQUEST_SCHEMA,
        "schema_version": module.REQUEST_SCHEMA_VERSION,
        "scientific_payload_id": content_id(module.scientific_payload(source_result)),
        "source_artifact_digest": "sha256:" + "b" * 64,
        "source_artifact_id": 12,
        "source_artifact_name": "source-evidence",
        "source_head_sha": "4" * 40,
        "source_run_id": 13,
        "source_unfinalized_evaluation_id": source_result["evaluation_id"],
    }
    request["request_id"] = content_id(request)
    return request


def test_finalization_repairs_request_binding_without_changing_scores(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "source"
    source_result, _ = _source_bundle(module, source_root)
    request = _request(module, source_result)
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)
    validated = module.validate_request(request_path)

    receipt = module.finalize(
        validated,
        source_root,
        tmp_path / "output",
        "c" * 40,
    )
    finalized = json.loads((tmp_path / "output" / "result.json").read_text(encoding="utf-8"))

    assert finalized["request_id"] == request["pooled_request_id"]
    assert finalized["provider_request_id"] == request["provider_request_id"]
    assert finalized["unfinalized_evaluation_id"] == source_result["evaluation_id"]
    assert finalized["aggregate_methods"] == source_result["aggregate_methods"]
    assert finalized["method_rows"] == source_result["method_rows"]
    assert finalized["sequences"] == source_result["sequences"]
    assert finalized["scientific_payload_id"] == request["scientific_payload_id"]
    assert finalized["evaluation_id"] == content_id(
        {key: value for key, value in finalized.items() if key != "evaluation_id"}
    )
    assert receipt["scientific_payload_unchanged"] is True
    summary = (tmp_path / "output" / "summary.md").read_text(encoding="utf-8")
    assert finalized["evaluation_id"] in summary
    assert source_result["evaluation_id"] in summary


def test_finalization_rejects_scientific_payload_drift(tmp_path: Path) -> None:
    module = _load_module()
    source_root = tmp_path / "source"
    source_result, _ = _source_bundle(module, source_root)
    request = _request(module, source_result)
    request["scientific_payload_id"] = "d" * 64
    request["request_id"] = content_id(
        {key: value for key, value in request.items() if key != "request_id"}
    )

    with pytest.raises(ValueError, match="scientific payload identity changed"):
        module.finalize(request, source_root, tmp_path / "output", "c" * 40)
