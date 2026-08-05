import copy
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
    load_material_identity_mixture,
    marginalize_identity_log_likelihoods,
    moment_match_gaussian_identity_hypotheses,
    write_material_identity_mixture,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
REVISION_C = "c" * 40
REVISION_D = "d" * 40
RESULT_E = "e" * 64
RESULT_F = "f" * 64


def make_mixture(
    *,
    reverse: bool = False,
    null_log_weight: float = 0.0,
) -> MaterialIdentityMixtureV1:
    candidates = [
        MaterialIdentityCandidateV1(
            source_endpoint=None,
            association_result_id=None,
            source_score=None,
            calibrated_log_weight=null_log_weight,
            metadata={"role": "exact-fallback"},
        ),
        MaterialIdentityCandidateV1(
            source_endpoint=LocalTrackEndpoint("window-0", 3),
            association_result_id=RESULT_E,
            source_score=0.9,
            calibrated_log_weight=np.log(3.0),
            metadata={"support": 4},
        ),
        MaterialIdentityCandidateV1(
            source_endpoint=LocalTrackEndpoint("window-1", 8),
            association_result_id=RESULT_F,
            source_score=0.7,
            calibrated_log_weight=np.log(2.0),
        ),
    ]
    if reverse:
        candidates.reverse()
    return MaterialIdentityMixtureV1(
        target_endpoint=LocalTrackEndpoint("window-2", 5),
        window_order=("window-0", "window-1", "window-2"),
        causal_frame_stop=42,
        association_rule_id=SHA_A,
        calibration_id=SHA_B,
        tracklet_producer_revision=REVISION_C,
        association_revision=REVISION_D,
        candidates=tuple(candidates),
        metadata={"experiment": "source-only"},
    )


def test_candidate_order_is_canonical_and_content_addressed() -> None:
    first = make_mixture()
    second = make_mixture(reverse=True)

    assert first.mixture_id == second.mixture_id
    assert first.candidate_ids == second.candidate_ids
    assert first.candidates[0].source_endpoint is None
    np.testing.assert_allclose(first.probabilities, [1.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0])
    assert first.null_probability == pytest.approx(1.0 / 6.0)
    assert first.effective_hypothesis_count > 2.0


def test_null_only_mixture_reproduces_exact_likelihood_and_moments() -> None:
    mixture = MaterialIdentityMixtureV1(
        target_endpoint=LocalTrackEndpoint("newest", 0),
        window_order=("newest",),
        causal_frame_stop=10,
        association_rule_id=SHA_A,
        calibration_id=SHA_B,
        tracklet_producer_revision=REVISION_C,
        association_revision=REVISION_D,
        candidates=(
            MaterialIdentityCandidateV1(
                source_endpoint=None,
                association_result_id=None,
                source_score=None,
                calibrated_log_weight=12.0,
            ),
        ),
    )
    candidate_ids = mixture.candidate_ids
    likelihood = marginalize_identity_log_likelihoods(
        mixture,
        candidate_ids,
        np.array([-4.25]),
    )
    assert likelihood.log_marginal_likelihood == pytest.approx(-4.25)
    np.testing.assert_array_equal(likelihood.posterior_probabilities, [1.0])

    mean = np.array([[1.0, -2.0]])
    covariance = np.array([[[0.4, 0.1], [0.1, 0.8]]])
    result = moment_match_gaussian_identity_hypotheses(
        mixture,
        candidate_ids,
        mean,
        covariance,
    )
    np.testing.assert_array_equal(result.mean, mean[0])
    np.testing.assert_array_equal(result.covariance, covariance[0])
    np.testing.assert_array_equal(result.between_hypothesis_covariance, np.zeros((2, 2)))


def test_likelihood_marginalization_uses_stable_logsumexp() -> None:
    mixture = make_mixture()
    log_likelihoods = np.array([-1000.0, -999.0, -1004.0])
    result = marginalize_identity_log_likelihoods(
        mixture,
        mixture.candidate_ids,
        log_likelihoods,
    )

    prior = mixture.probabilities
    expected_terms = prior * np.exp(log_likelihoods - np.max(log_likelihoods))
    expected_probabilities = expected_terms / np.sum(expected_terms)
    expected_log_marginal = np.max(log_likelihoods) + np.log(np.sum(expected_terms))
    np.testing.assert_allclose(result.posterior_probabilities, expected_probabilities)
    assert result.log_marginal_likelihood == pytest.approx(expected_log_marginal)
    assert result.posterior_probabilities[1] > 0.85


def test_zero_likelihood_power_returns_source_prior_without_nan() -> None:
    mixture = make_mixture()
    result = marginalize_identity_log_likelihoods(
        mixture,
        mixture.candidate_ids,
        np.array([-np.inf, -5.0, -np.inf]),
        likelihood_power=0.0,
    )

    assert result.log_marginal_likelihood == pytest.approx(0.0)
    np.testing.assert_allclose(result.posterior_probabilities, mixture.probabilities)


def test_moment_matching_retains_between_hypothesis_uncertainty() -> None:
    mixture = make_mixture(null_log_weight=np.log(2.0))
    probabilities = np.array([0.25, 0.5, 0.25])
    means = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    covariances = np.repeat(np.eye(2)[None, :, :] * 0.1, 3, axis=0)

    result = moment_match_gaussian_identity_hypotheses(
        mixture,
        mixture.candidate_ids,
        means,
        covariances,
        probabilities=probabilities,
    )

    np.testing.assert_allclose(result.mean, [1.0, 0.5])
    np.testing.assert_allclose(result.within_hypothesis_covariance, np.eye(2) * 0.1)
    np.testing.assert_allclose(
        result.between_hypothesis_covariance,
        [[1.0, -0.5], [-0.5, 0.75]],
    )
    np.testing.assert_allclose(
        result.covariance,
        result.within_hypothesis_covariance + result.between_hypothesis_covariance,
    )


def test_candidate_alignment_and_invalid_covariance_fail_closed() -> None:
    mixture = make_mixture()
    reversed_ids = tuple(reversed(mixture.candidate_ids))
    with pytest.raises(ValueError, match="exactly match"):
        marginalize_identity_log_likelihoods(
            mixture,
            reversed_ids,
            np.zeros(3),
        )
    with pytest.raises(ValueError, match="impossible"):
        marginalize_identity_log_likelihoods(
            mixture,
            mixture.candidate_ids,
            np.full(3, -np.inf),
        )
    bad_covariance = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    bad_covariance[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        moment_match_gaussian_identity_hypotheses(
            mixture,
            mixture.candidate_ids,
            np.zeros((3, 2)),
            bad_covariance,
        )


def test_duplicate_source_endpoint_and_missing_null_are_rejected() -> None:
    linked = MaterialIdentityCandidateV1(
        source_endpoint=LocalTrackEndpoint("source", 1),
        association_result_id=RESULT_E,
        source_score=0.5,
        calibrated_log_weight=0.0,
    )
    settings = {
        "target_endpoint": LocalTrackEndpoint("target", 2),
        "window_order": ("source", "target"),
        "causal_frame_stop": 5,
        "association_rule_id": SHA_A,
        "calibration_id": SHA_B,
        "tracklet_producer_revision": REVISION_C,
        "association_revision": REVISION_D,
    }
    with pytest.raises(ValueError, match="exactly one null"):
        MaterialIdentityMixtureV1(candidates=(linked,), **settings)
    null = MaterialIdentityCandidateV1(
        source_endpoint=None,
        association_result_id=None,
        source_score=None,
        calibrated_log_weight=0.0,
    )
    with pytest.raises(ValueError, match="unique"):
        MaterialIdentityMixtureV1(candidates=(null, linked, linked), **settings)
    same_window = MaterialIdentityCandidateV1(
        source_endpoint=LocalTrackEndpoint("target", 7),
        association_result_id=RESULT_F,
        source_score=0.4,
        calibrated_log_weight=0.0,
    )
    with pytest.raises(ValueError, match="precede the target"):
        MaterialIdentityMixtureV1(candidates=(null, same_window), **settings)
    future = MaterialIdentityCandidateV1(
        source_endpoint=LocalTrackEndpoint("future", 9),
        association_result_id=RESULT_F,
        source_score=0.4,
        calibrated_log_weight=0.0,
    )
    with pytest.raises(ValueError, match="precede the target"):
        MaterialIdentityMixtureV1(candidates=(null, future), **settings)
    with pytest.raises(ValueError, match="last in window_order"):
        MaterialIdentityMixtureV1(
            candidates=(null, linked),
            **{**settings, "window_order": ("target", "source")},
        )


def test_metadata_is_recursively_immutable() -> None:
    mixture = make_mixture()
    with pytest.raises(TypeError, match="immutable"):
        mixture.metadata["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError, match="immutable"):
        mixture.candidates[0].metadata["new"] = 1  # type: ignore[index]
    copied = copy.deepcopy(mixture.metadata)
    copied["new"] = 1


def test_atomic_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    mixture = make_mixture()
    path = tmp_path / "mixture.json"
    write_material_identity_mixture(path, mixture)
    loaded = load_material_identity_mixture(path)
    assert loaded == mixture
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_material_identity_mixture(path, mixture)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mixture_id"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mixture ID mismatch"):
        load_material_identity_mixture(path)


def test_loader_rejects_duplicate_keys_and_candidate_id_tampering(tmp_path: Path) -> None:
    mixture = make_mixture()
    path = tmp_path / "mixture.json"
    write_material_identity_mixture(path, mixture)

    duplicate = path.read_text(encoding="utf-8").replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_material_identity_mixture(path)

    write_material_identity_mixture(path, mixture, overwrite=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidates"][0]["candidate_id"] = "1" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate ID mismatch"):
        load_material_identity_mixture(path)


def test_noncanonical_scalar_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        LocalTrackEndpoint("window", True)
    with pytest.raises(ValueError, match="real number"):
        MaterialIdentityCandidateV1(
            source_endpoint=None,
            association_result_id=None,
            source_score=None,
            calibrated_log_weight="0",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at most 1.0"):
        MaterialIdentityCandidateV1(
            source_endpoint=LocalTrackEndpoint("window", 1),
            association_result_id=RESULT_E,
            source_score=1.1,
            calibrated_log_weight=0.0,
        )
