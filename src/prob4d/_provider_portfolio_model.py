"""Strict ordered-gate model for the provider portfolio artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal, cast

from ._immutable_json import plain_json
from ._strict_json import (
    require_exact_fields,
    require_exact_string,
    require_finite_json_mapping,
    require_mapping,
    require_sha256,
)

PROVIDER_PORTFOLIO_SCHEMA: Final = "prob4d.provider-portfolio"
PROVIDER_PORTFOLIO_VERSION: Final = 1
PROVIDER_PORTFOLIO_CLAIM_BOUNDARY: Final = (
    "This artifact constrains provider-development concurrency and records ordered "
    "gate evidence. It does not establish provider accuracy, uncertainty calibration, "
    "physical-query improvement, BayesianPhysTwin admission, Causal4D intervention "
    "benefit, protected-target authorization, deployment safety, or state of the art."
)

ProviderRole = Literal["primary", "alternative", "parked"]
ProviderStatus = Literal["active", "parked", "promoted", "rejected", "archived"]
GateDecision = Literal["not-started", "in-progress", "passed", "failed"]

PROVIDER_STAGES: Final[tuple[str, ...]] = (
    "support",
    "means",
    "identity",
    "gauge-dependence",
    "conditional-covariance",
    "query-value",
)
ACTIVE_PROVIDER_ROLES: Final[tuple[ProviderRole, ...]] = ("primary", "alternative")
MAX_ACTIVE_PRIMARY: Final = 1
MAX_ACTIVE_ALTERNATIVE: Final = 1

ENTRY_FIELDS: Final = frozenset(
    {
        "provider_id",
        "provider_family",
        "role",
        "status",
        "gates",
        "point_covariance_development_authorized",
        "metadata",
    }
)
GATE_FIELDS: Final = frozenset({"decision", "evidence_id"})
POLICY_FIELDS: Final = frozenset(
    {
        "ordered_stages",
        "max_active_primary",
        "max_active_alternative",
        "active_roles",
    }
)


def canonical_policy() -> dict[str, object]:
    return {
        "ordered_stages": list(PROVIDER_STAGES),
        "max_active_primary": MAX_ACTIVE_PRIMARY,
        "max_active_alternative": MAX_ACTIVE_ALTERNATIVE,
        "active_roles": list(ACTIVE_PROVIDER_ROLES),
    }


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Boolean")
    return cast(bool, value)


def _exact_list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return cast(list[Any], value)


def _role(value: object, *, name: str) -> ProviderRole:
    role = require_exact_string(value, name=name)
    if role not in {"primary", "alternative", "parked"}:
        raise ValueError(f"{name} is not a supported provider role")
    return cast(ProviderRole, role)


def _status(value: object, *, name: str) -> ProviderStatus:
    status = require_exact_string(value, name=name)
    if status not in {"active", "parked", "promoted", "rejected", "archived"}:
        raise ValueError(f"{name} is not a supported provider status")
    return cast(ProviderStatus, status)


def _decision(value: object, *, name: str) -> GateDecision:
    decision = require_exact_string(value, name=name)
    if decision not in {"not-started", "in-progress", "passed", "failed"}:
        raise ValueError(f"{name} is not a supported gate decision")
    return cast(GateDecision, decision)


def _evidence_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return require_sha256(value, name=name)


def _normalize_gate(
    value: object,
    *,
    provider_id: str,
    stage: str,
) -> dict[str, object]:
    name = f"entry {provider_id!r} gate {stage!r}"
    gate = require_mapping(value, name=name)
    require_exact_fields(gate, GATE_FIELDS, name=name)
    decision = _decision(gate["decision"], name=f"{name}.decision")
    evidence_id = _evidence_digest(gate["evidence_id"], name=f"{name}.evidence_id")
    if decision in {"passed", "failed"} and evidence_id is None:
        raise ValueError(f"{name} requires an evidence digest")
    if decision in {"not-started", "in-progress"} and evidence_id is not None:
        raise ValueError(f"{name} cannot bind evidence before a decision")
    return {"decision": decision, "evidence_id": evidence_id}


def _decisions(gates: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        cast(Mapping[str, object], gates[stage])["decision"]
        for stage in PROVIDER_STAGES
    )


def _validate_gate_order(gates: Mapping[str, object], *, provider_id: str) -> None:
    decisions = _decisions(gates)
    active = tuple(index for index, value in enumerate(decisions) if value == "in-progress")
    failed = tuple(index for index, value in enumerate(decisions) if value == "failed")
    if len(active) > 1:
        raise ValueError(f"entry {provider_id!r} has more than one in-progress gate")
    if len(failed) > 1:
        raise ValueError(f"entry {provider_id!r} has more than one failed gate")

    terminal = active + failed
    if terminal:
        stop = terminal[0]
        if decisions[:stop] != ("passed",) * stop:
            raise ValueError(
                f"entry {provider_id!r} must pass every earlier gate before advancing"
            )
        if any(value != "not-started" for value in decisions[stop + 1 :]):
            raise ValueError(
                f"entry {provider_id!r} has decisions after its active or failed gate"
            )
        return

    passed = 0
    for value in decisions:
        if value != "passed":
            break
        passed += 1
    if any(value != "not-started" for value in decisions[passed:]):
        raise ValueError(
            f"entry {provider_id!r} gate decisions must form one passed prefix"
        )


def _normalize_gates(value: object, *, provider_id: str) -> dict[str, object]:
    gates = require_mapping(value, name=f"entry {provider_id!r}.gates")
    require_exact_fields(
        gates,
        frozenset(PROVIDER_STAGES),
        name=f"entry {provider_id!r}.gates",
    )
    normalized = {
        stage: _normalize_gate(gates[stage], provider_id=provider_id, stage=stage)
        for stage in PROVIDER_STAGES
    }
    _validate_gate_order(normalized, provider_id=provider_id)
    return normalized


def _validate_status(
    *,
    provider_id: str,
    role: ProviderRole,
    status: ProviderStatus,
    gates: Mapping[str, object],
    point_authorized: bool,
) -> None:
    decisions = _decisions(gates)
    in_progress = tuple(value for value in decisions if value == "in-progress")
    failed = tuple(value for value in decisions if value == "failed")
    all_passed = all(value == "passed" for value in decisions)

    if status == "active":
        if role not in ACTIVE_PROVIDER_ROLES:
            raise ValueError(f"active entry {provider_id!r} must have an active role")
        if len(in_progress) != 1 or failed or all_passed:
            raise ValueError(
                f"active entry {provider_id!r} requires exactly one in-progress gate"
            )
    elif status == "promoted":
        if role != "primary" or not all_passed:
            raise ValueError(
                f"promoted entry {provider_id!r} must be primary with every gate passed"
            )
    elif status == "rejected":
        if role not in ACTIVE_PROVIDER_ROLES:
            raise ValueError(
                f"rejected entry {provider_id!r} must retain its primary or alternative role"
            )
        if len(failed) != 1 or in_progress:
            raise ValueError(
                f"rejected entry {provider_id!r} requires exactly one failed gate"
            )
    elif status in {"parked", "archived"}:
        if in_progress or failed:
            raise ValueError(
                f"{status} entry {provider_id!r} cannot have an active or failed gate"
            )
        if status == "parked" and role != "parked":
            raise ValueError(f"parked entry {provider_id!r} must use the parked role")

    if point_authorized and decisions[:4] != ("passed",) * 4:
        raise ValueError(
            f"entry {provider_id!r} cannot authorize conditional covariance before "
            "support, means, identity, and gauge-dependence pass"
        )
    conditional = decisions[4]
    if conditional == "in-progress" and not point_authorized:
        raise ValueError(
            f"entry {provider_id!r} cannot develop conditional covariance without "
            "source-localization authorization"
        )


def _normalize_entry(value: object, *, index: int) -> dict[str, object]:
    entry = require_mapping(value, name=f"entries[{index}]")
    require_exact_fields(entry, ENTRY_FIELDS, name=f"entries[{index}]")
    provider_id = require_exact_string(
        entry["provider_id"],
        name=f"entries[{index}].provider_id",
    )
    provider_family = require_exact_string(
        entry["provider_family"],
        name=f"entry {provider_id!r}.provider_family",
    )
    role = _role(entry["role"], name=f"entry {provider_id!r}.role")
    status = _status(entry["status"], name=f"entry {provider_id!r}.status")
    gates = _normalize_gates(entry["gates"], provider_id=provider_id)
    point_authorized = _exact_bool(
        entry["point_covariance_development_authorized"],
        name=f"entry {provider_id!r}.point_covariance_development_authorized",
    )
    metadata = require_finite_json_mapping(
        entry["metadata"],
        name=f"entry {provider_id!r}.metadata",
    )
    _validate_status(
        provider_id=provider_id,
        role=role,
        status=status,
        gates=gates,
        point_authorized=point_authorized,
    )
    return {
        "provider_id": provider_id,
        "provider_family": provider_family,
        "role": role,
        "status": status,
        "gates": gates,
        "point_covariance_development_authorized": point_authorized,
        "metadata": plain_json(metadata),
    }


def normalize_entries(value: object) -> list[dict[str, object]]:
    entries = [
        _normalize_entry(item, index=index)
        for index, item in enumerate(_exact_list(value, name="entries"))
    ]
    if not entries:
        raise ValueError("entries must contain at least one provider")
    identifiers = [cast(str, entry["provider_id"]) for entry in entries]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("provider_id values must be unique")
    entries.sort(key=lambda entry: cast(str, entry["provider_id"]))
    _validate_budget(entries)
    return entries


def _validate_budget(entries: Sequence[Mapping[str, object]]) -> None:
    primary = sum(
        entry["status"] == "active" and entry["role"] == "primary"
        for entry in entries
    )
    alternative = sum(
        entry["status"] == "active" and entry["role"] == "alternative"
        for entry in entries
    )
    if primary > MAX_ACTIVE_PRIMARY:
        raise ValueError("provider portfolio exceeds the active-primary budget")
    if alternative > MAX_ACTIVE_ALTERNATIVE:
        raise ValueError("provider portfolio exceeds the active-alternative budget")
    if alternative and not primary:
        raise ValueError("an active alternative requires one active primary provider")
