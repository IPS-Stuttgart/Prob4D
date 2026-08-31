from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.dependence_tempering import temper_shared_dependence
from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "calibrate_dot_rope_dependence_tempering.py"
PROTOCOL = ROOT / "protocols" / "dot-rope-dependence-tempering-source-v1.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("dot_dependence_tempering", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tempering_preserves_diagonal_and_endpoints() -> None:
    marginal = np.diag([2.0, 3.0, 5.0])
    shared = np.asarray(
        [
            [2.0, 0.6, -0.2],
            [0.6, 3.0, 0.7],
            [-0.2, 0.7, 5.0],
        ]
    )
    np.testing.assert_allclose(temper_shared_dependence(marginal, shared, 0.0), marginal)
    np.testing.assert_allclose(temper_shared_dependence(marginal, shared, 1.0), shared)
    middle = temper_shared_dependence(marginal, shared, 0.25)
    np.testing.assert_array_equal(np.diag(middle), np.diag(marginal))
    np.testing.assert_allclose(middle - np.diag(np.diag(middle)), 0.25 * (shared - marginal))
    assert np.min(np.linalg.eigvalsh(middle)) >= -1.0e-12


def test_tempering_rejects_mismatched_marginals_and_invalid_strength() -> None:
    marginal = np.eye(2)
    shared = np.asarray([[2.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="matching marginal variances"):
        temper_shared_dependence(marginal, shared, 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        temper_shared_dependence(marginal, marginal, -0.1)


def test_protocol_has_canonical_identity_and_keeps_confirmation_closed() -> None:
    value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    unsigned = dict(value)
    protocol_id = unsigned.pop("protocol_id")
    assert content_id(unsigned) == protocol_id
    assert value["source_sequences"] == ["R01", "R02", "R03"]
    assert value["reserved_sequences"] == "R04-R70"
    assert value["confirmation_candidate"] == "R04-R10"
    assert value["confirmation_reserved_after_candidate"] == "R11-R70"
    assert value["confirmation_payloads_opened"] is False
    assert value["means_held_fixed"] is True
    assert value["alpha_grid"][0] == 0.0
    assert value["alpha_grid"][-1] == 1.0


def test_source_selection_is_worst_sequence_first() -> None:
    module = _load_script()
    strengths = [0.0, 0.5, 1.0]
    rows = []
    values = {
        0.0: {"R01": 1.0, "R02": 1.0, "R03": 5.0},
        0.5: {"R01": 2.0, "R02": 2.0, "R03": 3.0},
        1.0: {"R01": 0.0, "R02": 0.0, "R03": 4.0},
    }
    for alpha, by_sequence in values.items():
        for sequence, nll in by_sequence.items():
            rows.append(
                {
                    "method": module.alpha_method_name(alpha),
                    "sequence": sequence,
                    "normalized_nll_per_dimension": nll,
                }
            )
    selected, table = module.select_strength(rows, strengths)
    assert selected == 0.5
    assert [row["alpha"] for row in table] == strengths


def test_request_validator_binds_protocol_blob(tmp_path: Path) -> None:
    module = _load_script()
    protocol_blob = "1" * 40
    request = {
        "schema": "prob4d.dot-rope-dependence-tempering-source-request",
        "schema_version": 1,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "protocol_git_blob_sha": protocol_blob,
        "provider_run_id": 33329701704,
        "provider_artifact_name": "dot-rope-cut3r-sealed-provider-33329701704-1",
        "source_sequences": ["R01", "R02", "R03"],
        "reserved_sequences": "R04-R70",
        "source_calibration_authorized": True,
        "confirmation_payloads_opened": False,
        "claim_boundary": "source only",
    }
    request["request_id"] = content_id(request)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    result = module.validate_request(
        path,
        PROTOCOL.relative_to(ROOT),
        protocol_blob,
    )
    assert result["request_id"] == request["request_id"]
    assert result["provider_run_id"] == 33329701704
