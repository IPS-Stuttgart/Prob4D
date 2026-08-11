from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from prob4d.query_covariance_preservation import (
    QueryCovarianceCandidateV1,
    QueryCovariancePreservationCertificateV1,
    QueryCovariancePreservationPolicyV1,
    load_query_covariance_preservation,
    write_query_covariance_preservation,
)


def _policy(**overrides: float) -> QueryCovariancePreservationPolicyV1:
    values = {
        "relative_rank_tolerance": 1e-10,
        "maximum_relative_trace_distortion": 0.1,
        "maximum_relative_frobenius_distortion": 0.1,
        "minimum_directional_variance_ratio": 0.9,
        "maximum_directional_variance_ratio": 1.1,
        "maximum_unsupported_trace_fraction": 0.0,
    }
    values.update(overrides)
    return QueryCovariancePreservationPolicyV1(**values)


def _candidate(
    candidate_id: str,
    covariance: np.ndarray,
) -> QueryCovarianceCandidateV1:
    return QueryCovarianceCandidateV1(
        candidate_id=candidate_id,
        representation=candidate_id,
        covariance=covariance,
        estimated_memory_bytes=128,
        estimated_runtime_seconds=0.01,
    )


def _certificate(
    reference: np.ndarray,
    candidates: tuple[QueryCovarianceCandidateV1, ...],
    *,
    policy: QueryCovariancePreservationPolicyV1 | None = None,
) -> QueryCovariancePreservationCertificateV1:
    return QueryCovariancePreservationCertificateV1(
        query_definition_id="1" * 64,
        observation_artifact_id="2" * 64,
        reference_representation="full-joint",
        reference_covariance=reference,
        policy=_policy() if policy is None else policy,
        candidates=candidates,
    )


def test_exact_and_small_perturbation_preserve_query_covariance() -> None:
    reference = np.diag([2.0, 1.0])
    certificate = _certificate(
        reference,
        (
            _candidate("exact", reference),
            _candidate("near", np.diag([1.9, 1.0])),
        ),
    )

    assert certificate.all_preserved
    assert certificate.preserved_candidate_ids == ("exact", "near")
    exact = certificate.results[0]
    assert exact.minimum_directional_variance_ratio == pytest.approx(1.0)
    assert exact.maximum_directional_variance_ratio_error == pytest.approx(0.0)


def test_directional_understatement_fails_even_when_average_trace_is_close() -> None:
    reference = np.eye(2)
    candidate = np.diag([0.5, 1.5])
    certificate = _certificate(reference, (_candidate("anisotropic", candidate),))

    result = certificate.results[0]
    assert not result.preserved
    assert "directional-understatement-exceeded" in result.failure_reasons
    assert "directional-overstatement-exceeded" in result.failure_reasons
    assert result.relative_trace_distortion == pytest.approx(0.0)


def test_nullspace_leakage_is_reported_separately() -> None:
    reference = np.diag([2.0, 0.0])
    candidate = np.diag([2.0, 1.0])
    certificate = _certificate(reference, (_candidate("leaky", candidate),))

    result = certificate.results[0]
    assert result.minimum_directional_variance_ratio == pytest.approx(1.0)
    assert result.unsupported_trace_fraction == pytest.approx(1.0 / 3.0)
    assert "unsupported-query-trace-exceeded" in result.failure_reasons


def test_from_projections_uses_total_covariance_only() -> None:
    reference = SimpleNamespace(total_covariance=np.diag([2.0, 1.0]))
    candidate = SimpleNamespace(total_covariance=np.diag([2.0, 1.0]))

    certificate = QueryCovariancePreservationCertificateV1.from_projections(
        query_definition_id="1" * 64,
        observation_artifact_id="2" * 64,
        reference_representation="full-joint",
        reference_projection=reference,
        candidate_projections={"tree-sparse": candidate},
        policy=_policy(),
    )

    assert certificate.preserved_candidate_ids == ("tree-sparse",)


def test_invalid_covariance_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        _candidate("bad", np.diag([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive trace"):
        _certificate(np.zeros((2, 2)), (_candidate("zero", np.zeros((2, 2))),))


def test_round_trip_recomputes_every_result(tmp_path) -> None:
    reference = np.diag([2.0, 1.0])
    certificate = _certificate(reference, (_candidate("exact", reference),))
    path = tmp_path / "certificate.json"
    write_query_covariance_preservation(path, certificate)

    loaded = load_query_covariance_preservation(path)
    assert loaded.to_dict() == certificate.to_dict()
    with pytest.raises(FileExistsError):
        write_query_covariance_preservation(path, certificate)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["relative_trace_distortion"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="derived fields changed"):
        load_query_covariance_preservation(path)
