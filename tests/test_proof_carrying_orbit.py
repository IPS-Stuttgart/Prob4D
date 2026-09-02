from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from prob4d.axial_query_certificate import AngleArc, HarmonicQuery
from prob4d.proof_carrying_orbit import (
    DEFAULT_ORBIT_ASSUMPTION_IDS,
    build_axial_orbit_action_certificate,
    compute_proof_carrying_orbit_id,
    render_proof_carrying_orbit,
    write_proof_carrying_orbit,
)
from prob4d_independent_verifier.orbit_advantage import (
    main as orbit_verify_main,
)
from prob4d_independent_verifier.orbit_advantage import (
    verify_proof_carrying_orbit_action,
)

_FULL_CIRCLE = AngleArc()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _query(constant: float, cosine: float, sine: float) -> HarmonicQuery:
    return HarmonicQuery(
        constant=constant,
        cosine=cosine,
        sine=sine,
        orbit_key=("test:shared-twist-v1", (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
    )


def _certificate(
    *,
    fallback: HarmonicQuery | None = None,
    candidate: HarmonicQuery | None = None,
    scope_admitted: bool = True,
    arc: AngleArc | None = _FULL_CIRCLE,
) -> dict[str, object]:
    return build_axial_orbit_action_certificate(
        fallback_loss=_query(4.0, 0.0, 0.0) if fallback is None else fallback,
        candidate_loss=_query(1.0, 0.5, 0.0) if candidate is None else candidate,
        scope_admitted=scope_admitted,
        support_receipt_digest=_digest(b"registered orbit support") if scope_admitted else None,
        candidate_action_id="action:candidate-v1",
        fallback_action_id="action:fallback-v1",
        candidate_loss_program_digest=_digest(b"candidate loss program"),
        fallback_loss_program_digest=_digest(b"fallback loss program"),
        fallback_id="belief:caller-owned-physical-fallback-v1",
        fallback_digest=_digest(b"complete fallback belief"),
        input_digest=_digest(b"sealed physical input"),
        admission_policy_digest=_digest(b"registered robust-action policy"),
        arc=arc,
        advantage_error_bound=0.1,
        required_margin=0.5,
        assumption_ids=(
            *DEFAULT_ORBIT_ASSUMPTION_IDS,
            "controlled-orbit-certificate-test-v1",
        ),
    )


def _reseal(certificate: dict[str, object]) -> None:
    certificate["certificate_id"] = compute_proof_carrying_orbit_id(certificate)


def test_full_orbit_robust_advantage_verifies_and_admits(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "supported.json"
    write_proof_carrying_orbit(path, certificate)

    report = verify_proof_carrying_orbit_action(path)

    assert report.valid
    assert report.admitted
    assert report.decision == "verified-admit"
    assert report.reason_codes == ()
    assert report.metrics["nominal_lower_advantage"] == pytest.approx(2.5)
    assert report.metrics["robust_lower_advantage"] == pytest.approx(2.4)
    assert report.metrics["robust_admission_margin"] == pytest.approx(1.9)
    assert orbit_verify_main([str(path), "--compact"]) == 0


def test_sign_changing_advantage_is_valid_but_rejected(tmp_path: Path) -> None:
    certificate = _certificate(
        fallback=_query(1.0, 0.0, 0.0),
        candidate=_query(1.0, 1.0, 0.0),
    )
    path = tmp_path / "rejected.json"
    write_proof_carrying_orbit(path, certificate)

    report = verify_proof_carrying_orbit_action(path)

    assert report.valid
    assert not report.admitted
    assert report.decision == "verified-reject"
    assert report.reason_codes == ("nonpositive-robust-advantage",)
    assert orbit_verify_main([str(path), "--compact"]) == 2


def test_partial_arc_checks_interior_stationary_points() -> None:
    certificate = _certificate(
        fallback=_query(3.0, 1.0, 0.0),
        candidate=_query(1.0, 0.0, 0.0),
        arc=AngleArc(center=0.0, half_width=0.5),
    )

    report = verify_proof_carrying_orbit_action(certificate)

    assert report.admitted
    assert report.metrics["nominal_upper_advantage"] == pytest.approx(3.0)
    assert report.metrics["nominal_lower_advantage"] == pytest.approx(2.0 + math.cos(0.5))


def test_infeasible_arc_is_valid_and_fails_closed() -> None:
    certificate = _certificate(arc=None)

    report = verify_proof_carrying_orbit_action(certificate)

    assert report.valid
    assert not report.admitted
    assert report.reason_codes == ("infeasible-anchor-support",)
    assert report.metrics["advantage_bounds_feasible"] == 0.0


def test_unadmitted_scope_is_valid_and_fails_closed_without_receipt() -> None:
    certificate = _certificate(scope_admitted=False)

    report = verify_proof_carrying_orbit_action(certificate)

    assert report.valid
    assert not report.admitted
    assert report.reason_codes == ("orbit-model-scope-not-admitted",)


def test_resealed_false_admission_is_invalid() -> None:
    certificate = _certificate(
        fallback=_query(1.0, 0.0, 0.0),
        candidate=_query(1.0, 1.0, 0.0),
    )
    decision = certificate["decision"]
    assert isinstance(decision, dict)
    decision["admitted"] = True
    decision["reason_codes"] = []
    _reseal(certificate)

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("producer-decision-mismatch",)


def test_resealed_advantage_witness_tampering_is_invalid() -> None:
    certificate = _certificate()
    advantage = certificate["advantage"]
    assert isinstance(advantage, dict)
    difference = advantage["difference_harmonic"]
    assert isinstance(difference, dict)
    difference["constant"] = float(difference["constant"]) + 1.0
    _reseal(certificate)

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert report.reason_codes == ("advantage-witness-mismatch",)


def test_resealed_reported_bound_tampering_is_invalid() -> None:
    certificate = _certificate()
    advantage = certificate["advantage"]
    assert isinstance(advantage, dict)
    advantage["reported_robust_lower"] = 100.0
    _reseal(certificate)

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert report.reason_codes == ("reported-robust-lower-mismatch",)


def test_resealed_nonunit_axis_is_invalid() -> None:
    certificate = _certificate()
    orbit = certificate["orbit"]
    assert isinstance(orbit, dict)
    orbit["axis"] = [0.0, 0.0, 2.0]
    _reseal(certificate)

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert report.reason_codes == ("orbit-axis-not-unit",)


def test_resealed_missing_support_receipt_is_invalid() -> None:
    certificate = _certificate()
    orbit = certificate["orbit"]
    assert isinstance(orbit, dict)
    orbit["support_receipt_digest"] = None
    _reseal(certificate)

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert report.reason_codes == ("missing-support-receipt",)


def test_widened_claim_scope_is_invalid_even_when_resealed() -> None:
    certificate = _certificate()
    certificate["claim_scope"] = "unconditional-deployment-safety"
    _reseal(certificate)

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert report.reason_codes == ("invalid-claim-scope",)


def test_content_id_detects_unsealed_tampering() -> None:
    certificate = _certificate()
    provenance = certificate["provenance"]
    assert isinstance(provenance, dict)
    provenance["producer"] = "tampered"

    report = verify_proof_carrying_orbit_action(certificate)

    assert not report.valid
    assert report.reason_codes == ("certificate-id-mismatch",)


def test_writer_is_immutable_and_mapping_verification_is_nonmutating(tmp_path: Path) -> None:
    certificate = _certificate()
    before = copy.deepcopy(certificate)
    path = tmp_path / "certificate.json"
    write_proof_carrying_orbit(path, certificate)

    with pytest.raises(FileExistsError):
        write_proof_carrying_orbit(path, certificate)

    assert verify_proof_carrying_orbit_action(certificate).admitted
    assert certificate == before
    assert json.loads(path.read_text(encoding="utf-8")) == certificate
    assert render_proof_carrying_orbit(certificate).endswith("\n")


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    certificate = _certificate()
    rendered = render_proof_carrying_orbit(certificate)
    duplicate = rendered.replace("{\n", '{\n  "schema": "duplicate",\n', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")

    report = verify_proof_carrying_orbit_action(path)

    assert not report.valid
    assert report.reason_codes == ("duplicate-json-key",)


def test_producer_rejects_scope_admission_without_support_receipt() -> None:
    with pytest.raises(ValueError, match="support_receipt_digest"):
        build_axial_orbit_action_certificate(
            fallback_loss=_query(2.0, 0.0, 0.0),
            candidate_loss=_query(1.0, 0.0, 0.0),
            scope_admitted=True,
            support_receipt_digest=None,
            candidate_action_id="candidate",
            fallback_action_id="fallback",
            candidate_loss_program_digest=_digest(b"candidate"),
            fallback_loss_program_digest=_digest(b"fallback"),
            fallback_id="fallback-belief",
            fallback_digest=_digest(b"fallback belief"),
            input_digest=_digest(b"input"),
            admission_policy_digest=_digest(b"policy"),
        )
