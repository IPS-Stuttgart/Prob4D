from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_dot_rope_cut3r_heldout_confirmation.py"
PROTOCOL = ROOT / "protocols" / "dot-rope-cut3r-heldout-confirmation-v1.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_cut3r_heldout_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_content_addressed_and_confirmation_is_frozen() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id")
    assert content_id(unsigned) == protocol_id
    assert value["confirmation_sequences"] == [
        "R04",
        "R05",
        "R06",
        "R07",
        "R08",
        "R09",
        "R10",
    ]
    assert value["reserved_sequences"] == "R11-R70"
    assert value["source_calibration"]["selected_alpha"] == 0.85
    assert value["source_calibration"]["calibration_id"] == (
        "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
    )
    assert value["information_boundary"]["source_calibration_frozen"] is True
    assert value["information_boundary"]["confirmation_tuning_authorized"] is False


def test_request_validator_recomputes_identity(tmp_path: Path) -> None:
    module = _load_script()
    protocol_blob = "1" * 40
    request = {
        "schema": "prob4d.dot-rope-cut3r-heldout-confirmation-request",
        "schema_version": 1,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_git_blob_sha": protocol_blob,
        "confirmation_sequences": [
            "R04",
            "R05",
            "R06",
            "R07",
            "R08",
            "R09",
            "R10",
        ],
        "reserved_sequences": "R11-R70",
        "source_calibration_id": (
            "943339ac864fda04cc59081bc81a605576b3c90bf0aa996aea00b00335cfc0c7"
        ),
        "selected_alpha": 0.85,
        "normal_view_prediction_authorized": True,
        "marker_evaluation_authorized": True,
        "bayesian_phystwin_executed": False,
        "causal4d_executed": False,
        "claim_boundary": "held out only",
    }
    request["request_id"] = content_id(request)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    result = module.validate_request(path, PROTOCOL.relative_to(ROOT), protocol_blob)
    assert result["request_id"] == request["request_id"]
    assert result["reserved_sequences"] == "R11-R70"


def test_paired_sequence_bootstrap_and_classification_are_deterministic() -> None:
    module = _load_script()
    rows = []
    for index, sequence in enumerate(module.CONFIRMATION_SEQUENCES):
        selected = 1.0 + 0.01 * index
        for method, offset in (
            (module.SELECTED_METHOD, 0.0),
            ("pointwise_quadratic", 0.3),
            ("shared_quadratic_curvature", 0.8),
            ("local_first_order", 1.2),
        ):
            rows.append(
                {
                    "sequence": sequence,
                    "method": method,
                    "normalized_nll_per_dimension": selected + offset,
                }
            )
    comparisons = {
        comparator: module._paired_difference(
            rows,
            module.SELECTED_METHOD,
            comparator,
            replicates=2000,
            seed=17 + index,
        )
        for index, comparator in enumerate(
            ["pointwise_quadratic", "shared_quadratic_curvature", "local_first_order"]
        )
    }
    assert module._classification(comparisons) == "heldout-strong-positive"
    assert comparisons["pointwise_quadratic"]["sequence_wins"] == 7
    assert comparisons["pointwise_quadratic"]["upper_95"] < 0.0


def test_base_protocol_adaptation_does_not_mutate_frozen_protocol() -> None:
    module = _load_script()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    adapted = module._base_protocol(protocol)
    assert "source_sequences" not in protocol
    assert adapted["source_sequences"] == module.CONFIRMATION_SEQUENCES
    assert adapted["reserved_sequences"] == "R11-R70"
