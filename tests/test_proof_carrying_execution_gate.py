from __future__ import annotations

import copy
import json
from pathlib import Path

from prob4d_independent_verifier.execution_gate import (
    PROOF_CARRYING_FACTOR_KIND,
    PROOF_CARRYING_ORBIT_KIND,
    build_execution_context,
    compute_execution_context_id,
    main as execution_gate_main,
    render_execution_context,
    verify_for_execution,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "proof4d"


def _load(name: str) -> dict[str, object]:
    value = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _local_bindings(certificate: dict[str, object]) -> dict[str, object]:
    query = _object(certificate["query"])
    decision = _object(certificate["decision"])
    provenance = _object(certificate["provenance"])
    return {
        "certificate_id": certificate["certificate_id"],
        "input_digest": provenance["input_digest"],
        "source_factor_digest": provenance["source_factor_digest"],
        "query_id": query["query_id"],
        "query_program_digest": query["query_program_digest"],
        "fallback_id": decision["fallback_id"],
    }


def _orbit_bindings(certificate: dict[str, object]) -> dict[str, object]:
    orbit = _object(certificate["orbit"])
    actions = _object(certificate["actions"])
    candidate = _object(actions["candidate"])
    fallback = _object(actions["fallback"])
    decision = _object(certificate["decision"])
    provenance = _object(certificate["provenance"])
    return {
        "certificate_id": certificate["certificate_id"],
        "input_digest": provenance["input_digest"],
        "shared_gauge_id": orbit["shared_gauge_id"],
        "support_receipt_digest": orbit["support_receipt_digest"],
        "candidate_action_id": candidate["action_id"],
        "fallback_action_id": fallback["action_id"],
        "candidate_loss_program_digest": candidate["loss_program_digest"],
        "fallback_loss_program_digest": fallback["loss_program_digest"],
        "fallback_id": decision["fallback_id"],
        "fallback_digest": decision["fallback_digest"],
        "admission_policy_digest": provenance["admission_policy_digest"],
    }


def _context(certificate: dict[str, object]) -> dict[str, object]:
    kind = certificate["certificate_kind"]
    assert isinstance(kind, str)
    bindings = (
        _local_bindings(certificate)
        if kind == PROOF_CARRYING_FACTOR_KIND
        else _orbit_bindings(certificate)
    )
    return build_execution_context(certificate_kind=kind, bindings=bindings)


def test_matching_local_context_authorizes_admitted_certificate(tmp_path: Path) -> None:
    certificate = _load("linear-query-supported.json")
    context = _context(certificate)
    context_path = tmp_path / "context.json"
    context_path.write_text(render_execution_context(context), encoding="utf-8")

    report = verify_for_execution(
        EXAMPLES / "linear-query-supported.json",
        context_path,
    )

    assert report.valid
    assert report.admitted
    assert report.decision == "verified-admit"
    assert report.certificate_decision == "verified-admit"
    assert execution_gate_main(
        [
            str(EXAMPLES / "linear-query-supported.json"),
            str(context_path),
            "--compact",
        ]
    ) == 0


def test_matching_orbit_context_authorizes_global_action(tmp_path: Path) -> None:
    certificate = _load("axial-action-supported.json")
    context = _context(certificate)
    context_path = tmp_path / "context.json"
    context_path.write_text(render_execution_context(context), encoding="utf-8")

    report = verify_for_execution(
        EXAMPLES / "axial-action-supported.json",
        context_path,
    )

    assert report.valid
    assert report.admitted
    assert report.decision == "verified-admit"
    assert execution_gate_main(
        [
            str(EXAMPLES / "axial-action-supported.json"),
            str(context_path),
            "--compact",
        ]
    ) == 0


def test_valid_rejected_certificate_remains_rejected_in_matching_context() -> None:
    certificate = _load("axial-action-rejected.json")
    report = verify_for_execution(certificate, _context(certificate))

    assert report.valid
    assert not report.admitted
    assert report.decision == "verified-reject"
    assert report.certificate_decision == "verified-reject"
    assert report.reason_codes == ("nonpositive-robust-advantage",)


def test_valid_local_certificate_cannot_be_replayed_for_new_input() -> None:
    certificate = _load("linear-query-supported.json")
    bindings = _local_bindings(certificate)
    bindings["input_digest"] = "sha256:" + "1" * 64
    context = build_execution_context(
        certificate_kind=PROOF_CARRYING_FACTOR_KIND,
        bindings=bindings,
    )

    report = verify_for_execution(certificate, context)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("execution-context-mismatch",)
    assert "input_digest" in report.detail


def test_valid_orbit_certificate_cannot_be_replayed_for_new_policy() -> None:
    certificate = _load("axial-action-supported.json")
    bindings = _orbit_bindings(certificate)
    bindings["admission_policy_digest"] = "sha256:" + "2" * 64
    context = build_execution_context(
        certificate_kind=PROOF_CARRYING_ORBIT_KIND,
        bindings=bindings,
    )

    report = verify_for_execution(certificate, context)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("execution-context-mismatch",)
    assert "admission_policy_digest" in report.detail


def test_valid_orbit_certificate_cannot_be_replayed_for_new_fallback() -> None:
    certificate = _load("axial-action-supported.json")
    bindings = _orbit_bindings(certificate)
    bindings["fallback_digest"] = "sha256:" + "3" * 64
    context = build_execution_context(
        certificate_kind=PROOF_CARRYING_ORBIT_KIND,
        bindings=bindings,
    )

    report = verify_for_execution(certificate, context)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("execution-context-mismatch",)
    assert "fallback_digest" in report.detail


def test_context_kind_mismatch_fails_closed() -> None:
    local_certificate = _load("linear-query-supported.json")
    orbit_certificate = _load("axial-action-supported.json")

    report = verify_for_execution(local_certificate, _context(orbit_certificate))

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("execution-context-kind-mismatch",)


def test_context_content_id_detects_tampering() -> None:
    certificate = _load("linear-query-supported.json")
    context = _context(certificate)
    bindings = _object(context["bindings"])
    bindings["query_id"] = "different-query"

    report = verify_for_execution(certificate, context)

    assert not report.valid
    assert report.reason_codes == ("context-id-mismatch",)


def test_resealed_context_still_cannot_misrepresent_certificate() -> None:
    certificate = _load("linear-query-supported.json")
    context = copy.deepcopy(_context(certificate))
    bindings = _object(context["bindings"])
    bindings["query_id"] = "different-query"
    context["context_id"] = compute_execution_context_id(context)

    report = verify_for_execution(certificate, context)

    assert not report.valid
    assert report.reason_codes == ("execution-context-mismatch",)
    assert "query_id" in report.detail


def test_duplicate_context_keys_fail_closed(tmp_path: Path) -> None:
    certificate = _load("linear-query-supported.json")
    rendered = render_execution_context(_context(certificate))
    duplicate = rendered.replace(
        "{\n",
        '{\n  "schema": "duplicate",\n',
        1,
    )
    context_path = tmp_path / "duplicate-context.json"
    context_path.write_text(duplicate, encoding="utf-8")

    report = verify_for_execution(
        EXAMPLES / "linear-query-supported.json",
        context_path,
    )

    assert not report.valid
    assert report.reason_codes == ("duplicate-json-key",)
