from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from prob4d.correlation_group_robust_likelihood import (
    CORRELATION_GROUP_ROBUST_CLAIM_BOUNDARY,
    CORRELATION_GROUP_ROBUST_SCHEMA,
    CORRELATION_GROUP_ROBUST_VERSION,
    GAUSSIAN_GROUP_LIKELIHOOD_V1,
    CorrelationGroupContaminationSpecV1,
    CorrelationGroupResidualV1,
    SourceCorrelationGroupMixtureSelectionV1,
    evaluate_correlation_group_mixture,
    select_source_correlation_group_mixture,
)


def _group(
    group_id: str,
    residual_x: float,
    *,
    sample_count: int = 1,
    rank: int = 0,
) -> CorrelationGroupResidualV1:
    residual = np.zeros((sample_count, 3), dtype=np.float64)
    residual[:, 0] = residual_x
    local = np.repeat(np.eye(3, dtype=np.float64)[None, ...], sample_count, axis=0)
    factor = np.zeros((sample_count, 3, rank), dtype=np.float64)
    if rank:
        factor[:, 0, 0] = 0.5
    return CorrelationGroupResidualV1(group_id, residual, local, factor)


def _robust_spec() -> CorrelationGroupContaminationSpecV1:
    return CorrelationGroupContaminationSpecV1(0.2, 25.0)


def _mixed_source_groups() -> tuple[CorrelationGroupResidualV1, ...]:
    values = (0.0, 0.2, -0.2, 0.1, 0.3, -0.1, 8.0, 9.0)
    return tuple(_group(f"group-{index:02d}", value) for index, value in enumerate(values))


def test_spec_identity_and_gaussian_fallback_are_deterministic() -> None:
    first = CorrelationGroupContaminationSpecV1(0.2, 25.0)
    second = CorrelationGroupContaminationSpecV1(np.float64(0.2), np.float64(25.0))

    assert first == second
    assert first.spec_id == second.spec_id
    assert not first.is_gaussian_fallback
    assert GAUSSIAN_GROUP_LIKELIHOOD_V1.is_gaussian_fallback
    assert first.summary()["contamination_probability"] == 0.2


@pytest.mark.parametrize(
    ("probability", "inflation", "message"),
    [
        (True, 1.0, "genuine real"),
        (-0.1, 2.0, "must lie"),
        (1.0, 2.0, "must lie"),
        (0.0, 2.0, "Gaussian fallback"),
        (0.2, 1.0, "inflation_factor > 1"),
        (0.2, np.nan, "finite"),
    ],
)
def test_invalid_specs_fail_closed(
    probability: object,
    inflation: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CorrelationGroupContaminationSpecV1(probability, inflation)  # type: ignore[arg-type]


def test_group_owns_normalized_immutable_arrays_and_content_identity() -> None:
    residual = np.array([[1.0, 0.0, 0.0]])
    local = np.eye(3)[None, ...]
    local[0, 0, 1] = 1e-14
    factor = np.ones((1, 3, 1), dtype=np.float64)

    group = CorrelationGroupResidualV1("group-a", residual, local, factor)
    residual[...] = 9.0
    local[...] = 9.0
    factor[...] = 9.0

    np.testing.assert_allclose(group.residual_xyz_m, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(group.local_covariance_m2, np.eye(3)[None, ...], atol=1e-13)
    np.testing.assert_allclose(group.low_rank_factor_m, np.ones((1, 3, 1)))
    for value in (
        group.residual_xyz_m,
        group.local_covariance_m2,
        group.low_rank_factor_m,
    ):
        assert not value.flags.writeable
    assert len(group.source_id) == 64
    assert group.sample_count == 1
    assert group.dimension == 3

    changed = _group("group-a", 1.1, rank=1)
    assert changed.source_id != group.source_id


def test_group_rejects_invalid_geometry_and_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="group_id"):
        _group(1, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        _group("", 0.0)
    with pytest.raises(ValueError, match="residual_xyz_m"):
        CorrelationGroupResidualV1(
            "a",
            np.ones((1, 2)),
            np.eye(3)[None, ...],
            np.empty((1, 3, 0)),
        )
    indefinite = np.diag([1.0, 1.0, -1.0])[None, ...]
    with pytest.raises(ValueError, match="positive definite"):
        CorrelationGroupResidualV1(
            "a",
            np.zeros((1, 3)),
            indefinite,
            np.empty((1, 3, 0)),
        )
    asymmetric = np.eye(3)[None, ...]
    asymmetric[0, 0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetric"):
        CorrelationGroupResidualV1(
            "a",
            np.zeros((1, 3)),
            asymmetric,
            np.empty((1, 3, 0)),
        )
    residual = np.zeros((1, 3))
    residual[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        CorrelationGroupResidualV1(
            "a",
            residual,
            np.eye(3)[None, ...],
            np.empty((1, 3, 0)),
        )


def test_mixture_matches_dense_scaled_gaussian_formula() -> None:
    group = _group("group-a", 4.0, sample_count=2, rank=1)
    spec = _robust_spec()

    evaluation = evaluate_correlation_group_mixture(group, spec)

    local = np.zeros((6, 6), dtype=np.float64)
    local[:3, :3] = np.eye(3)
    local[3:, 3:] = np.eye(3)
    factor = group.low_rank_factor_m.reshape(6, 1)
    covariance = local + factor @ factor.T
    residual = group.residual_xyz_m.reshape(6)
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign == 1.0
    mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    gaussian_nll = 0.5 * (
        6 * np.log(2.0 * np.pi) + log_determinant + mahalanobis
    )
    inflated_nll = 0.5 * (
        6 * np.log(2.0 * np.pi)
        + log_determinant
        + 6 * np.log(spec.inflation_factor)
        + mahalanobis / spec.inflation_factor
    )
    expected_log_likelihood = np.logaddexp(
        np.log1p(-spec.contamination_probability) - gaussian_nll,
        np.log(spec.contamination_probability) - inflated_nll,
    )

    assert evaluation.mahalanobis_squared == pytest.approx(mahalanobis)
    assert evaluation.joint_log_determinant == pytest.approx(log_determinant)
    assert evaluation.gaussian_nll == pytest.approx(gaussian_nll)
    assert evaluation.inflated_nll == pytest.approx(inflated_nll)
    assert evaluation.mixture_nll == pytest.approx(-expected_log_likelihood)


def test_posterior_responsibility_is_group_shared_and_bounded() -> None:
    spec = _robust_spec()
    nominal = evaluate_correlation_group_mixture(_group("nominal", 0.0), spec)
    contaminated = evaluate_correlation_group_mixture(_group("outlier", 8.0), spec)

    assert nominal.posterior_contamination_probability < 0.01
    assert contaminated.posterior_contamination_probability > 0.999
    assert nominal.posterior_expected_precision_multiplier > 0.99
    assert contaminated.posterior_expected_precision_multiplier == pytest.approx(
        1.0 / spec.inflation_factor,
        rel=1e-3,
    )
    for result in (nominal, contaminated):
        assert 0.0 <= result.posterior_contamination_probability <= 1.0
        assert 1.0 / spec.inflation_factor <= (
            result.posterior_expected_precision_multiplier
        ) <= 1.0


def test_gaussian_fallback_is_exact() -> None:
    evaluation = evaluate_correlation_group_mixture(
        _group("group-a", 3.0, rank=1),
        GAUSSIAN_GROUP_LIKELIHOOD_V1,
    )

    assert evaluation.mixture_nll == evaluation.gaussian_nll
    assert evaluation.inflated_nll == evaluation.gaussian_nll
    assert evaluation.posterior_contamination_probability == 0.0
    assert evaluation.posterior_expected_precision_multiplier == 1.0
    assert evaluation.nll_advantage_over_gaussian_per_dimension == 0.0


def test_source_selection_promotes_robust_candidate_for_repeated_outlier_groups() -> None:
    robust = _robust_spec()

    selection = select_source_correlation_group_mixture(
        _mixed_source_groups(),
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, robust),
    )

    assert selection.robust_supported
    assert selection.selected_spec_id == robust.spec_id
    assert selection.unconstrained_spec_id == robust.spec_id
    assert selection.decision_reasons == ()
    assert selection.mean_heldout_advantage_per_dimension > 2.0
    assert selection.harmful_group_fraction == 0.0
    assert selection.final_candidate_fold_fraction == 1.0
    assert all(fold.selected_is_robust for fold in selection.folds)


def test_clean_source_prefers_gaussian_by_deterministic_complexity_tie_break() -> None:
    groups = tuple(_group(f"group-{index:02d}", 0.0) for index in range(8))

    selection = select_source_correlation_group_mixture(
        groups,
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, _robust_spec()),
    )

    assert not selection.robust_supported
    assert selection.selected_spec.is_gaussian_fallback
    assert selection.unconstrained_spec.is_gaussian_fallback
    assert selection.decision_reasons == ("full-source-selection-is-gaussian",)
    assert all(not fold.selected_is_robust for fold in selection.folds)


def test_small_nominal_harm_margin_is_explicit_and_can_force_fallback() -> None:
    robust = _robust_spec()

    selection = select_source_correlation_group_mixture(
        _mixed_source_groups(),
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, robust),
        maximum_heldout_nll_harm_per_dimension=0.01,
        maximum_harmful_group_fraction=0.0,
    )

    assert not selection.robust_supported
    assert selection.unconstrained_spec_id == robust.spec_id
    assert selection.selected_spec.is_gaussian_fallback
    assert selection.harmful_group_fraction == pytest.approx(0.75)
    assert selection.decision_reasons == (
        "harmful-heldout-group-fraction-exceeds-maximum",
    )


def test_nested_heldout_score_can_reject_one_outlier_despite_full_source_fit() -> None:
    values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0)
    groups = tuple(_group(f"group-{index:02d}", value) for index, value in enumerate(values))
    robust = CorrelationGroupContaminationSpecV1(0.1, 25.0)

    selection = select_source_correlation_group_mixture(
        groups,
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, robust),
        maximum_harmful_group_fraction=1.0,
    )

    assert selection.unconstrained_spec_id == robust.spec_id
    assert not selection.robust_supported
    assert selection.selected_spec.is_gaussian_fallback
    assert selection.mean_heldout_advantage_per_dimension < 0.0
    assert "heldout-mean-nll-advantage-below-minimum" in selection.decision_reasons


def test_insufficient_group_count_is_valid_negative_evidence() -> None:
    robust = _robust_spec()
    groups = (_group("a", 8.0), _group("b", 9.0))

    selection = select_source_correlation_group_mixture(
        groups,
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, robust),
        minimum_group_count=4,
        maximum_harmful_group_fraction=1.0,
        minimum_final_candidate_fold_fraction=0.0,
    )

    assert not selection.robust_supported
    assert selection.selected_spec.is_gaussian_fallback
    assert "insufficient-independent-source-groups" in selection.decision_reasons


def test_group_and_candidate_input_order_do_not_change_selection_identity() -> None:
    groups = _mixed_source_groups()
    robust = _robust_spec()

    forward = select_source_correlation_group_mixture(
        groups,
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, robust),
    )
    reversed_input = select_source_correlation_group_mixture(
        tuple(reversed(groups)),
        (robust, GAUSSIAN_GROUP_LIKELIHOOD_V1),
    )

    assert forward.selection_id == reversed_input.selection_id
    assert forward.summary() == reversed_input.summary()
    np.testing.assert_array_equal(
        forward.nll_per_dimension,
        reversed_input.nll_per_dimension,
    )


def test_direct_selection_construction_reorders_candidate_rows_and_replays() -> None:
    original = select_source_correlation_group_mixture(
        _mixed_source_groups(),
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, _robust_spec()),
    )
    reversed_candidates = tuple(reversed(original.candidates))
    reversed_scores = original.nll_per_dimension[::-1]

    reconstructed = SourceCorrelationGroupMixtureSelectionV1(
        group_ids=original.group_ids,
        group_source_ids=original.group_source_ids,
        candidates=reversed_candidates,
        nll_per_dimension=reversed_scores,
        minimum_group_count=original.minimum_group_count,
        minimum_mean_heldout_advantage_per_dimension=(
            original.minimum_mean_heldout_advantage_per_dimension
        ),
        maximum_heldout_nll_harm_per_dimension=(
            original.maximum_heldout_nll_harm_per_dimension
        ),
        maximum_harmful_group_fraction=original.maximum_harmful_group_fraction,
        minimum_final_candidate_fold_fraction=(
            original.minimum_final_candidate_fold_fraction
        ),
        tie_tolerance=original.tie_tolerance,
        relative_rank_tolerance=original.relative_rank_tolerance,
    )

    assert reconstructed.selection_id == original.selection_id
    assert reconstructed.summary() == original.summary()
    assert not reconstructed.nll_per_dimension.flags.writeable


def test_duplicate_groups_or_candidate_grid_without_one_fallback_fail_closed() -> None:
    duplicate_groups = (_group("a", 0.0), _group("a", 8.0))
    robust = _robust_spec()
    with pytest.raises(ValueError, match="group IDs must be unique"):
        select_source_correlation_group_mixture(
            duplicate_groups,
            (GAUSSIAN_GROUP_LIKELIHOOD_V1, robust),
        )
    groups = (_group("a", 0.0), _group("b", 8.0))
    with pytest.raises(ValueError, match="one Gaussian fallback"):
        select_source_correlation_group_mixture(groups, (robust,))
    with pytest.raises(ValueError, match="unique"):
        select_source_correlation_group_mixture(
            groups,
            (GAUSSIAN_GROUP_LIKELIHOOD_V1, GAUSSIAN_GROUP_LIKELIHOOD_V1),
        )


def test_selection_thresholds_reject_coercive_or_out_of_range_values() -> None:
    selection = select_source_correlation_group_mixture(
        _mixed_source_groups(),
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, _robust_spec()),
    )

    with pytest.raises(TypeError, match="minimum_group_count"):
        replace(selection, minimum_group_count=True)
    with pytest.raises(ValueError, match="must be nonnegative"):
        replace(selection, maximum_heldout_nll_harm_per_dimension=-0.1)
    with pytest.raises(ValueError, match="must lie"):
        replace(selection, maximum_harmful_group_fraction=1.1)
    with pytest.raises(ValueError, match="must be finite"):
        replace(selection, tie_tolerance=np.nan)
    with pytest.raises(ValueError, match="must lie"):
        replace(selection, relative_rank_tolerance=1.0)


def test_summary_keeps_responsibility_separate_and_states_claim_boundary() -> None:
    evaluation = evaluate_correlation_group_mixture(_group("a", 8.0), _robust_spec())
    selection = select_source_correlation_group_mixture(
        _mixed_source_groups(),
        (GAUSSIAN_GROUP_LIKELIHOOD_V1, _robust_spec()),
    )

    evaluation_summary = evaluation.summary()
    selection_summary = selection.summary()
    assert "posterior_contamination_probability" in evaluation_summary
    assert "association_probability" not in evaluation_summary
    assert "prior_reliability" not in evaluation_summary
    assert selection_summary["schema"] == CORRELATION_GROUP_ROBUST_SCHEMA
    assert selection_summary["version"] == CORRELATION_GROUP_ROBUST_VERSION
    assert selection_summary["claim_boundary"] == CORRELATION_GROUP_ROBUST_CLAIM_BOUNDARY
