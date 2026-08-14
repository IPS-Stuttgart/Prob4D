from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prob4d.fresh_provider_readiness import (
    FreshProviderCohortLockV1,
    FreshProviderReadinessRequestV1,
    ReadinessGateV1,
    evaluate_fresh_provider_readiness,
    write_fresh_provider_readiness_decision,
)
from prob4d.provider_readiness_matrix import (
    PROVIDER_READINESS_MATRIX_DECISION_SPEC_SCHEMA,
    PROVIDER_READINESS_MATRIX_LOCK_SPEC_SCHEMA,
    PROVIDER_READINESS_MATRIX_SELECTION_RULE,
    PROVIDER_READINESS_MATRIX_VERSION,
    ProviderReadinessMatrixEntryV1,
    ProviderReadinessMatrixLockV1,
    ProviderReadinessMatrixProviderV1,
    ProviderReadinessMatrixRequestV1,
    authorize_provider_readiness_matrix_target,
    build_provider_readiness_matrix_lock,
    build_provider_readiness_matrix_request,
    evaluate_provider_readiness_matrix,
    load_provider_readiness_matrix_decision,
    load_provider_readiness_matrix_lock,
    readiness_matrix_provider_metadata,
    write_provider_readiness_matrix_decision,
    write_provider_readiness_matrix_lock,
)

_GATE_ORDER = (
    "support-feasibility",
    "source-mean",
    "identity-reliability",
    "gauge-dependence",
    "point-covariance",
    "query-relevance",
    "exact-fallback",
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provider(provider_id: str, priority: int) -> ProviderReadinessMatrixProviderV1:
    return ProviderReadinessMatrixProviderV1(
        provider_id=provider_id,
        priority=priority,
        provider_repository=f"example/{provider_id}",
        provider_revision=_digest(provider_id)[:40],
        model_set_id=_digest(provider_id + ":model"),
        loader_id=_digest(provider_id + ":loader"),
        promotion_lock_id=_digest(provider_id + ":promotion"),
        adapter_identity_id=_digest(provider_id + ":adapter"),
        adapter_conformance_id=_digest(provider_id + ":conformance"),
    )


def _lock(*providers: ProviderReadinessMatrixProviderV1) -> ProviderReadinessMatrixLockV1:
    return ProviderReadinessMatrixLockV1(
        matrix_id="fresh-provider-matrix-v1",
        source_spec_sha256="e" * 64,
        source_repository="IPS-Stuttgart/Prob4D",
        source_revision="a" * 40,
        cohort_binding_id="b" * 64,
        query_definition_id="c" * 64,
        fallback_identity_id="d" * 64,
        development_group_ids=("development-0", "development-1"),
        calibration_group_ids=("calibration-0", "calibration-1"),
        target_group_ids=("target-0", "target-1"),
        confirmation_group_ids=("confirmation-0",),
        comparison_policy={
            "group_definition": "complete-object-or-session-v1",
            "gate_policy_id": "f" * 64,
        },
        providers=providers,
    )


def _decision(
    lock: ProviderReadinessMatrixLockV1,
    provider_id: str,
    *,
    terminal_gate: str | None,
    metadata_override: dict[str, object] | None = None,
):
    provider = lock.provider(provider_id)
    cohort = FreshProviderCohortLockV1(
        protocol_id=f"protocol-{provider_id}",
        source_repository=lock.source_repository,
        source_revision=lock.source_revision,
        provider_repository=provider.provider_repository,
        provider_revision=provider.provider_revision,
        model_set_id=provider.model_set_id,
        loader_id=provider.loader_id,
        cohort_binding_id=lock.cohort_binding_id,
        promotion_lock_id=provider.promotion_lock_id,
        query_definition_id=lock.query_definition_id,
        fallback_identity_id=lock.fallback_identity_id,
        development_group_ids=lock.development_group_ids,
        calibration_group_ids=lock.calibration_group_ids,
        target_group_ids=lock.target_group_ids,
        confirmation_group_ids=lock.confirmation_group_ids,
    )
    gates: list[ReadinessGateV1] = []
    terminal_seen = False
    for name in _GATE_ORDER:
        if terminal_seen:
            gates.append(ReadinessGateV1(name, "not-evaluated", None))
        elif name == terminal_gate:
            gates.append(
                ReadinessGateV1(
                    name,
                    "fail",
                    _digest(provider_id + ":" + name),
                    reason_codes=("registered-negative",),
                )
            )
            terminal_seen = True
        else:
            gates.append(
                ReadinessGateV1(
                    name,
                    "pass",
                    _digest(provider_id + ":" + name),
                )
            )
    metadata: dict[str, object] = readiness_matrix_provider_metadata(lock, provider_id)
    if metadata_override is not None:
        metadata.update(metadata_override)
    request = FreshProviderReadinessRequestV1(
        cohort_lock=cohort,
        gates=tuple(gates),
        metadata=metadata,
    )
    return evaluate_fresh_provider_readiness(request)


def _entry(
    lock: ProviderReadinessMatrixLockV1,
    provider_id: str,
    terminal_gate: str | None,
) -> ProviderReadinessMatrixEntryV1:
    provider = lock.provider(provider_id)
    return ProviderReadinessMatrixEntryV1(
        provider_id=provider_id,
        priority=provider.priority,
        decision_file_sha256=_digest(provider_id + ":file"),
        decision=_decision(lock, provider_id, terminal_gate=terminal_gate),
    )


def _request(
    lock: ProviderReadinessMatrixLockV1,
    *entries: ProviderReadinessMatrixEntryV1,
) -> ProviderReadinessMatrixRequestV1:
    return ProviderReadinessMatrixRequestV1(
        matrix_lock_file_sha256="1" * 64,
        matrix_lock=lock,
        source_spec_sha256="2" * 64,
        entries=entries,
    )


def test_matrix_selects_only_first_ready_provider_by_frozen_priority() -> None:
    lock = _lock(
        _provider("provider-a", 10),
        _provider("provider-b", 20),
        _provider("provider-c", 30),
    )
    request = _request(
        lock,
        _entry(lock, "provider-b", None),
        _entry(lock, "provider-a", None),
        _entry(lock, "provider-c", "source-mean"),
    )
    decision = evaluate_provider_readiness_matrix(request)

    assert decision.matrix_status == "provider-selected"
    assert decision.ready_provider_ids == ("provider-a", "provider-b")
    assert decision.selected_provider_id == "provider-a"
    assert decision.unselected_ready_provider_ids == ("provider-b",)
    assert decision.target_evaluation_budget == 1
    selected = [
        item for item in decision.provider_results if item["selected_for_target_evaluation"]
    ]
    assert len(selected) == 1
    assert selected[0]["provider_id"] == "provider-a"

    authorization = authorize_provider_readiness_matrix_target(decision)
    assert authorization.selected_provider_id == "provider-a"
    assert authorization.target_evaluation_budget == 1


def test_matrix_retains_no_ready_result_without_authorization() -> None:
    lock = _lock(_provider("provider-a", 10), _provider("provider-b", 20))
    decision = evaluate_provider_readiness_matrix(
        _request(
            lock,
            _entry(lock, "provider-a", "support-feasibility"),
            _entry(lock, "provider-b", "point-covariance"),
        )
    )
    assert decision.matrix_status == "no-provider-ready"
    assert decision.selected_provider_id is None
    assert decision.point_uncertainty_provider_ids == ("provider-b",)
    assert decision.target_evaluation_budget == 0
    with pytest.raises(ValueError, match="selected no target provider"):
        authorize_provider_readiness_matrix_target(decision)


def test_matrix_rejects_decision_without_exact_lock_and_policy_binding() -> None:
    lock = _lock(_provider("provider-a", 10), _provider("provider-b", 20))
    changed = ProviderReadinessMatrixEntryV1(
        provider_id="provider-b",
        priority=20,
        decision_file_sha256=_digest("provider-b:file"),
        decision=_decision(
            lock,
            "provider-b",
            terminal_gate=None,
            metadata_override={"provider_readiness_matrix_policy_id": "0" * 64},
        ),
    )
    with pytest.raises(ValueError, match="differs from the matrix lock"):
        _request(lock, _entry(lock, "provider-a", None), changed)


def _lock_spec() -> dict[str, object]:
    return {
        "schema": PROVIDER_READINESS_MATRIX_LOCK_SPEC_SCHEMA,
        "schema_version": PROVIDER_READINESS_MATRIX_VERSION,
        "matrix_id": "fresh-provider-matrix-v1",
        "selection_rule": PROVIDER_READINESS_MATRIX_SELECTION_RULE,
        "maximum_target_evaluations": 1,
        "source_repository": "IPS-Stuttgart/Prob4D",
        "source_revision": "a" * 40,
        "cohort_binding_id": "b" * 64,
        "query_definition_id": "c" * 64,
        "fallback_identity_id": "d" * 64,
        "development_group_ids": ["development-0", "development-1"],
        "calibration_group_ids": ["calibration-0", "calibration-1"],
        "target_group_ids": ["target-0", "target-1"],
        "confirmation_group_ids": ["confirmation-0"],
        "comparison_policy": {
            "group_definition": "complete-object-or-session-v1",
            "gate_policy_id": "f" * 64,
        },
        "providers": [
            {
                **{
                    key: value
                    for key, value in _provider("provider-b", 20).to_dict().items()
                    if key != "provider_identity_id"
                }
            },
            {
                **{
                    key: value
                    for key, value in _provider("provider-a", 10).to_dict().items()
                    if key != "provider_identity_id"
                }
            },
        ],
        "source_payloads_opened": False,
        "source_outcomes_opened": False,
        "target_payloads_opened": False,
        "target_outcomes_opened": False,
        "confirmation_payloads_opened": False,
        "metadata": {"stage": "before-source-execution"},
    }


def test_lock_freeze_and_matrix_decision_build_replay(tmp_path: Path) -> None:
    lock_spec_path = tmp_path / "matrix-lock-spec.json"
    lock_spec_path.write_text(
        json.dumps(_lock_spec(), indent=2) + "\n",
        encoding="utf-8",
    )
    lock = build_provider_readiness_matrix_lock(lock_spec_path)
    lock_path = tmp_path / "matrix-lock.json"
    write_provider_readiness_matrix_lock(lock_path, lock)
    assert load_provider_readiness_matrix_lock(lock_path).provider_readiness_matrix_lock_id == (
        lock.provider_readiness_matrix_lock_id
    )

    decisions = tmp_path / "decisions"
    decisions.mkdir()
    first_path = decisions / "provider-a.json"
    second_path = decisions / "provider-b.json"
    write_fresh_provider_readiness_decision(
        first_path,
        _decision(lock, "provider-a", terminal_gate=None),
    )
    write_fresh_provider_readiness_decision(
        second_path,
        _decision(lock, "provider-b", terminal_gate="identity-reliability"),
    )
    decision_spec = {
        "schema": PROVIDER_READINESS_MATRIX_DECISION_SPEC_SCHEMA,
        "schema_version": PROVIDER_READINESS_MATRIX_VERSION,
        "matrix_lock_path": "matrix-lock.json",
        "entries": [
            {
                "provider_id": "provider-b",
                "decision_path": "decisions/provider-b.json",
                "metadata": {},
            },
            {
                "provider_id": "provider-a",
                "decision_path": "decisions/provider-a.json",
                "metadata": {},
            },
        ],
        "metadata": {"stage": "source-only"},
    }
    decision_spec_path = tmp_path / "matrix-decisions.json"
    decision_spec_path.write_text(
        json.dumps(decision_spec, indent=2) + "\n",
        encoding="utf-8",
    )

    request = build_provider_readiness_matrix_request(decision_spec_path)
    assert [entry.provider_id for entry in request.entries] == [
        "provider-a",
        "provider-b",
    ]
    decision = evaluate_provider_readiness_matrix(request)
    output = tmp_path / "matrix-decision.json"
    write_provider_readiness_matrix_decision(output, decision)
    loaded = load_provider_readiness_matrix_decision(output)
    assert loaded.provider_readiness_matrix_decision_id == (
        decision.provider_readiness_matrix_decision_id
    )


def test_matrix_lock_must_precede_source_access() -> None:
    record = _lock_spec()
    record["source_outcomes_opened"] = True
    with pytest.raises(ValueError, match="must precede source"):
        ProviderReadinessMatrixLockV1(
            matrix_id=record["matrix_id"],
            source_spec_sha256="e" * 64,
            source_repository=record["source_repository"],
            source_revision=record["source_revision"],
            cohort_binding_id=record["cohort_binding_id"],
            query_definition_id=record["query_definition_id"],
            fallback_identity_id=record["fallback_identity_id"],
            development_group_ids=tuple(record["development_group_ids"]),
            calibration_group_ids=tuple(record["calibration_group_ids"]),
            target_group_ids=tuple(record["target_group_ids"]),
            confirmation_group_ids=tuple(record["confirmation_group_ids"]),
            comparison_policy=record["comparison_policy"],
            providers=(_provider("provider-a", 10), _provider("provider-b", 20)),
            source_outcomes_opened=True,
        )


def test_matrix_decision_tampering_fails_exact_replay(tmp_path: Path) -> None:
    lock = _lock(_provider("provider-a", 10), _provider("provider-b", 20))
    decision = evaluate_provider_readiness_matrix(
        _request(
            lock,
            _entry(lock, "provider-a", None),
            _entry(lock, "provider-b", "source-mean"),
        )
    )
    path = tmp_path / "matrix.json"
    write_provider_readiness_matrix_decision(path, decision)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["selected_provider_id"] = "provider-b"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="replay changed"):
        load_provider_readiness_matrix_decision(path)
