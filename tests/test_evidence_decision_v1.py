from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from prob4d.evidence_decision_v1 import (
    EVIDENCE_DECISION_JSON_SCHEMA_SHA256,
    EVIDENCE_DECISION_SCHEMA,
    EVIDENCE_DECISION_SCHEMA_VERSION,
    evidence_decision_contract_identity,
    load_evidence_decision_v1,
    main,
    require_authorized_evidence_decision_v1,
    require_prob4d_evidence_binding_v1,
    require_repository_binding_v1,
    validate_evidence_decision_v1,
)

BPT_REVISION = "4ee702f5130cfedbea7bce6be5e72483c92f63da"
PROB4D_REVISION = "9212f3e039b9e3662099d05c62b1b4c72a7530c4"
CAUSAL4D_REVISION = "50e3682a5dbf976b20cc9115b6e7a975d0144ea5"


def _reseal(payload: dict[str, object]) -> dict[str, object]:
    descriptor = copy.deepcopy(payload)
    descriptor.pop("decision_id", None)
    repositories = descriptor.get("repositories")
    if isinstance(repositories, list):
        primary = [item for item in repositories if item.get("role") == "primary"]
        related = [item for item in repositories if item.get("role") != "primary"]
        descriptor["repositories"] = primary + sorted(
            related,
            key=lambda item: (item.get("role"), item.get("repository")),
        )
    payload["decision_id"] = hashlib.sha256(
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _authorized() -> dict[str, object]:
    return _reseal(
        {
            "claim_authorized": True,
            "claim_id": "contract.conformance.authorized",
            "created_utc": "2026-08-10T12:00:00+00:00",
            "evidence_fingerprint": "2" * 64,
            "evidence_level": 3,
            "evidence_summary_sha256": "3" * 64,
            "limitations": [
                "synthetic conformance vector; not scientific evidence"
            ],
            "metadata": {
                "source_contract_revision": BPT_REVISION,
                "synthetic": True,
            },
            "metric": {
                "comparison": "reference_contract",
                "name": "synthetic_contract_metric",
                "observed_value": 1.0,
                "rule": "observed_value_ge_threshold",
                "threshold_value": 1.0,
                "unit": "dimensionless",
            },
            "protocol_id": "evidence-decision-conformance-v1",
            "repositories": [
                {
                    "dirty": False,
                    "repository": "IPS-Stuttgart/BayesianPhysTwin",
                    "revision": BPT_REVISION,
                    "role": "primary",
                },
                {
                    "dirty": False,
                    "repository": "IPS-Stuttgart/Causal4D",
                    "revision": CAUSAL4D_REVISION,
                    "role": "downstream",
                },
                {
                    "dirty": False,
                    "repository": "IPS-Stuttgart/Prob4D",
                    "revision": PROB4D_REVISION,
                    "role": "observation",
                },
            ],
            "run_classification": "confirmatory",
            "run_manifest_id": "4" * 64,
            "schema_name": EVIDENCE_DECISION_SCHEMA,
            "schema_version": EVIDENCE_DECISION_SCHEMA_VERSION,
            "status": "pass",
        }
    )


def _degraded() -> dict[str, object]:
    payload = _authorized()
    payload.update(
        claim_authorized=False,
        claim_id="contract.conformance.degraded",
        created_utc="2026-08-10T12:05:00Z",
        evidence_level=1,
        limitations=["synthetic degraded vector", "dirty observation repository"],
        run_classification="diagnostic",
        status="degraded",
    )
    payload["metric"]["observed_value"] = 0.5
    payload["metric"]["rule"] = "diagnostic_only"
    payload["metric"]["threshold_value"] = None
    payload["repositories"][2]["dirty"] = True
    return _reseal(payload)


def test_contract_identity_is_content_locked() -> None:
    identity = evidence_decision_contract_identity()

    assert EVIDENCE_DECISION_JSON_SCHEMA_SHA256 == (
        "d5615258c6cf666d0ed9684a87930989adf91817fe99b0387e83a31479dcd465"
    )
    assert identity == {
        "schema_name": EVIDENCE_DECISION_SCHEMA,
        "schema_version": EVIDENCE_DECISION_SCHEMA_VERSION,
        "json_schema_sha256": EVIDENCE_DECISION_JSON_SCHEMA_SHA256,
        "source_repository": "IPS-Stuttgart/BayesianPhysTwin",
        "source_revision": BPT_REVISION,
    }


def test_authorized_decision_and_prob4d_binding_round_trip(tmp_path: Path) -> None:
    payload = _authorized()
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    decision = load_evidence_decision_v1(path)

    assert decision.as_dict() == payload
    assert decision.computed_decision_id == payload["decision_id"]
    assert [repository.role for repository in decision.repositories] == [
        "primary",
        "downstream",
        "observation",
    ]
    require_authorized_evidence_decision_v1(
        decision,
        claim_id="contract.conformance.authorized",
        protocol_id="evidence-decision-conformance-v1",
        minimum_evidence_level=3,
    )
    binding = require_prob4d_evidence_binding_v1(
        decision,
        expected_revision=PROB4D_REVISION,
    )
    assert binding.role == "observation"
    with pytest.raises(TypeError):
        decision.metadata["synthetic"] = False


def test_degraded_decision_is_valid_but_not_authorized() -> None:
    decision = validate_evidence_decision_v1(_degraded())

    assert decision.status == "degraded"
    binding = require_prob4d_evidence_binding_v1(decision, require_clean=False)
    assert binding.dirty
    with pytest.raises(ValueError, match="does not authorize"):
        require_authorized_evidence_decision_v1(decision)
    with pytest.raises(ValueError, match="dirty"):
        require_prob4d_evidence_binding_v1(decision, require_clean=True)


def test_loader_rejects_invalid_json_duplicate_keys_and_constants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decision.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_evidence_decision_v1(path)

    path.write_text('{"decision_id":"a","decision_id":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_evidence_decision_v1(path)

    path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_evidence_decision_v1(path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_evidence_decision_v1(path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update(unexpected=True), "unknown"),
        (lambda value: value.pop("claim_id"), "missing"),
        (lambda value: value.update(schema_name="other"), "unsupported.*schema"),
        (lambda value: value.update(schema_version=2), "unsupported.*version"),
        (lambda value: value.update(status="unknown"), "unsupported.*status"),
        (
            lambda value: value.update(run_classification="unknown"),
            "unsupported run classification",
        ),
        (lambda value: value.update(claim_authorized=1), "must be boolean"),
        (lambda value: value.update(evidence_level=4), "evidence_level"),
        (
            lambda value: value.update(created_utc="2026-08-10T12:00:00+0000"),
            "UTC suffix",
        ),
        (lambda value: value.update(claim_id=" padded "), "canonical nonempty"),
        (
            lambda value: value["metric"].update(observed_value="one"),
            "finite number",
        ),
        (lambda value: value["metric"].update(extra=True), "unknown"),
        (
            lambda value: value["repositories"][0].update(revision="A" * 40),
            "lowercase Git revision",
        ),
        (
            lambda value: value["repositories"][0].update(
                repository="bad/name/extra"
            ),
            "owner/name",
        ),
        (
            lambda value: value["repositories"][0].update(role="unknown"),
            "unsupported.*role",
        ),
    ],
)
def test_schema_and_scalar_drift_fail_closed(mutator, message: str) -> None:
    payload = _authorized()
    mutator(payload)
    try:
        _reseal(payload)
    except (TypeError, ValueError):
        pass
    with pytest.raises(ValueError, match=message):
        validate_evidence_decision_v1(payload)


def test_authorization_and_repository_invariants_fail_closed() -> None:
    payload = _authorized()
    payload["status"] = "fail"
    _reseal(payload)
    with pytest.raises(ValueError, match="passing decision"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["run_classification"] = "exploratory"
    _reseal(payload)
    with pytest.raises(ValueError, match="confirmatory"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["repositories"][1]["dirty"] = True
    _reseal(payload)
    with pytest.raises(ValueError, match="dirty repository"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["repositories"] = payload["repositories"][1:]
    _reseal(payload)
    with pytest.raises(ValueError, match="exactly one primary"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["repositories"][1]["role"] = "primary"
    _reseal(payload)
    with pytest.raises(ValueError, match="exactly one primary"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["repositories"][1]["repository"] = payload["repositories"][0][
        "repository"
    ]
    _reseal(payload)
    with pytest.raises(ValueError, match="names must be unique"):
        validate_evidence_decision_v1(payload)


def test_limitations_metadata_and_digest_fail_closed() -> None:
    payload = _degraded()
    payload["limitations"] = []
    _reseal(payload)
    with pytest.raises(ValueError, match="at least one limitation"):
        validate_evidence_decision_v1(payload)

    payload = _degraded()
    payload["limitations"] = ["same", "same"]
    _reseal(payload)
    with pytest.raises(ValueError, match="limitations must be unique"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["metadata"] = {"value": float("inf")}
    payload["decision_id"] = "0" * 64
    with pytest.raises(ValueError, match="non-finite"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["metadata"] = {1: "value"}
    payload["decision_id"] = "0" * 64
    with pytest.raises(ValueError, match="string keys"):
        validate_evidence_decision_v1(payload)

    circular: dict[str, object] = {}
    circular["value"] = circular
    payload = _authorized()
    payload["metadata"] = circular
    payload["decision_id"] = "0" * 64
    with pytest.raises(ValueError, match="circular mapping"):
        validate_evidence_decision_v1(payload)

    payload = _authorized()
    payload["claim_id"] = "changed"
    with pytest.raises(ValueError, match="digest does not match"):
        validate_evidence_decision_v1(payload)


def test_authorized_semantic_binding_errors() -> None:
    decision = validate_evidence_decision_v1(_authorized())

    with pytest.raises(ValueError, match="claim_id"):
        require_authorized_evidence_decision_v1(decision, claim_id="other")
    with pytest.raises(ValueError, match="protocol_id"):
        require_authorized_evidence_decision_v1(decision, protocol_id="other")
    with pytest.raises(ValueError, match="minimum_evidence_level"):
        require_authorized_evidence_decision_v1(
            decision,
            minimum_evidence_level=4,
        )

    lower = _authorized()
    lower["evidence_level"] = 1
    _reseal(lower)
    with pytest.raises(ValueError, match="required evidence level"):
        require_authorized_evidence_decision_v1(
            lower,
            minimum_evidence_level=2,
        )


def test_repository_binding_errors() -> None:
    decision = validate_evidence_decision_v1(_authorized())

    with pytest.raises(ValueError, match="revision does not match"):
        require_prob4d_evidence_binding_v1(
            decision,
            expected_revision="0" * 40,
        )
    with pytest.raises(ValueError, match="role is not allowed"):
        require_prob4d_evidence_binding_v1(
            decision,
            allowed_roles=("dependency",),
        )
    with pytest.raises(ValueError, match="unsupported repository role"):
        require_prob4d_evidence_binding_v1(
            decision,
            allowed_roles=("bad",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="nonempty and unique"):
        require_repository_binding_v1(decision, repository_names=())
    with pytest.raises(ValueError, match="nonempty and unique"):
        require_repository_binding_v1(
            decision,
            repository_names=("IPS-Stuttgart/Prob4D",) * 2,
        )
    with pytest.raises(ValueError, match="exactly one matching"):
        require_repository_binding_v1(
            decision,
            repository_names=("IPS-Stuttgart/Unknown",),
        )

    payload = _authorized()
    payload["repositories"].append(
        {
            "dirty": False,
            "repository": "FlorianPfaff/Prob4D",
            "revision": "0" * 40,
            "role": "dependency",
        }
    )
    _reseal(payload)
    with pytest.raises(ValueError, match="exactly one Prob4D"):
        require_prob4d_evidence_binding_v1(payload)


def test_cli_validates_and_prints_decision(tmp_path: Path, capsys) -> None:
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(_authorized()), encoding="utf-8")

    assert (
        main(
            [
                str(path),
                "--require-authorized",
                "--expected-prob4d-revision",
                PROB4D_REVISION,
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["decision_id"] == _authorized()["decision_id"]
