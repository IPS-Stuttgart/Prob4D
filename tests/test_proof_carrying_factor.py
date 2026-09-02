from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.observable_gauge import (
    IID_OBSERVABLE_INFORMATION,
    CentroidGaugeChart,
    ObservableGaugeFactor,
)
from prob4d.proof_carrying_factor import (
    DEFAULT_ASSUMPTION_IDS,
    build_observable_gauge_query_certificate,
    compute_proof_carrying_factor_id,
    render_proof_carrying_factor,
    write_proof_carrying_factor,
)
from prob4d.sim3 import Sim3
from prob4d_independent_verifier.proof_carrying import (
    main as proof4d_verify_main,
)
from prob4d_independent_verifier.proof_carrying import (
    verify_proof_carrying_factor,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _twist_ambiguous_factor() -> ObservableGaugeFactor:
    identity = np.eye(7)
    observable_indices = [0, 2, 3, 4, 5, 6]
    return ObservableGaugeFactor(
        chart=CentroidGaugeChart(
            linearization=Sim3.identity(),
            source_centroid=np.zeros(3),
            cloud_scale=1.0,
        ),
        observable_basis=identity[:, observable_indices],
        nullspace_basis=identity[:, 1:2],
        observable_information=10.0 * np.eye(6),
        normalized_geometry_spectrum=np.array(
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
        ),
        rank_threshold=0.1,
        residual_rms=0.01,
        residual_variance=1e-4,
        inlier_fraction=1.0,
        num_correspondences=8,
        covariance_method=IID_OBSERVABLE_INFORMATION,
    )


def _certificate(query_jacobian: np.ndarray) -> dict[str, object]:
    return build_observable_gauge_query_certificate(
        _twist_ambiguous_factor(),
        query_jacobian_local=query_jacobian,
        query_id="example:registered-linear-query-v1",
        query_program_digest=_digest(b"registered linear query program"),
        fallback_id="example:caller-owned-physical-fallback-v1",
        input_digest=_digest(b"sealed provider input"),
        assumption_ids=(
            *DEFAULT_ASSUMPTION_IDS,
            "controlled-test-factor-v1",
        ),
    )


def _reseal(certificate: dict[str, object]) -> None:
    certificate["certificate_id"] = compute_proof_carrying_factor_id(certificate)


def test_supported_certificate_verifies_and_admits(tmp_path: Path) -> None:
    certificate = _certificate(np.eye(7)[[0, 4]])
    path = tmp_path / "supported.json"

    write_proof_carrying_factor(path, certificate)
    report = verify_proof_carrying_factor(path)

    assert report.valid
    assert report.admitted
    assert report.decision == "verified-admit"
    assert report.reason_codes == ()
    assert report.certificate_id == certificate["certificate_id"]
    assert report.metrics["relative_query_nullspace_sensitivity"] == pytest.approx(0.0)
    assert proof4d_verify_main([str(path), "--compact"]) == 0


def test_unsupported_certificate_is_valid_but_rejected(tmp_path: Path) -> None:
    certificate = _certificate(np.eye(7)[[1]])
    path = tmp_path / "unsupported.json"
    write_proof_carrying_factor(path, certificate)

    report = verify_proof_carrying_factor(path)

    assert report.valid
    assert not report.admitted
    assert report.decision == "verified-reject"
    assert report.reason_codes == (
        "query-nullspace-sensitivity-exceeds-threshold",
    )
    assert report.metrics["relative_query_nullspace_sensitivity"] == pytest.approx(1.0)
    assert proof4d_verify_main([str(path), "--compact"]) == 2


def test_content_id_detects_unsealed_tampering() -> None:
    certificate = _certificate(np.eye(7)[[0]])
    query = certificate["query"]
    assert isinstance(query, dict)
    query["query_id"] = "tampered-query"

    report = verify_proof_carrying_factor(certificate)

    assert not report.valid
    assert not report.admitted
    assert report.decision == "invalid-fail-closed"
    assert report.reason_codes == ("certificate-id-mismatch",)


def test_resealed_false_admission_fails_closed() -> None:
    certificate = _certificate(np.eye(7)[[1]])
    decision = certificate["decision"]
    assert isinstance(decision, dict)
    decision["admitted"] = True
    _reseal(certificate)

    report = verify_proof_carrying_factor(certificate)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("producer-decision-mismatch",)


def test_resealed_query_witness_tampering_fails_closed() -> None:
    certificate = _certificate(np.eye(7)[[0]])
    query = certificate["query"]
    assert isinstance(query, dict)
    coordinates = query["observable_coordinates"]
    assert isinstance(coordinates, list)
    assert isinstance(coordinates[0], list)
    coordinates[0][0] = float(coordinates[0][0]) + 0.25
    _reseal(certificate)

    report = verify_proof_carrying_factor(certificate)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("query-observable-witness-mismatch",)


def test_resealed_subspace_tampering_fails_closed() -> None:
    certificate = _certificate(np.eye(7)[[0]])
    factor = certificate["factor"]
    assert isinstance(factor, dict)
    nullspace = factor["nullspace_basis"]
    assert isinstance(nullspace, list)
    nullspace[0][0] = 1.0
    nullspace[1][0] = 0.0
    _reseal(certificate)

    report = verify_proof_carrying_factor(certificate)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("subspace-basis-not-orthonormal",)


def test_widened_claim_scope_fails_closed_even_when_resealed() -> None:
    certificate = _certificate(np.eye(7)[[0]])
    certificate["claim_scope"] = "global-nonlinear-deployment-safety"
    _reseal(certificate)

    report = verify_proof_carrying_factor(certificate)

    assert not report.valid
    assert not report.admitted
    assert report.reason_codes == ("invalid-claim-scope",)


def test_writer_is_immutable_by_default(tmp_path: Path) -> None:
    certificate = _certificate(np.eye(7)[[0]])
    path = tmp_path / "certificate.json"
    write_proof_carrying_factor(path, certificate)

    with pytest.raises(FileExistsError):
        write_proof_carrying_factor(path, certificate)

    assert json.loads(path.read_text(encoding="utf-8")) == certificate
    assert render_proof_carrying_factor(certificate).endswith("\n")


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    certificate = _certificate(np.eye(7)[[0]])
    rendered = render_proof_carrying_factor(certificate)
    duplicate = rendered.replace(
        "{\n",
        '{\n  "schema": "duplicate",\n',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")

    report = verify_proof_carrying_factor(path)

    assert not report.valid
    assert report.reason_codes == ("duplicate-json-key",)


def test_mapping_verification_does_not_mutate_the_certificate() -> None:
    certificate = _certificate(np.eye(7)[[0]])
    before = copy.deepcopy(certificate)

    report = verify_proof_carrying_factor(certificate)

    assert report.admitted
    assert certificate == before


def test_checked_examples_match_the_independent_verifier() -> None:
    root = Path(__file__).parents[1] / "examples" / "proof4d"
    supported = verify_proof_carrying_factor(root / "linear-query-supported.json")
    rejected = verify_proof_carrying_factor(root / "linear-query-rejected.json")

    assert supported.decision == "verified-admit"
    assert supported.admitted
    assert rejected.decision == "verified-reject"
    assert rejected.valid
    assert not rejected.admitted
