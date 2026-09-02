"""Apply the reviewed repeated-eigenvalue semantics to PR #478.

This branch-local helper is intentionally temporary.  It performs exact,
fail-closed text replacements so the GitHub-hosted runner can validate and
commit the resulting source, tests, and theorem note.
"""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement target, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    source = "src/prob4d/posterior_rank_distortion.py"
    tests = "tests/test_posterior_rank_distortion.py"
    docs = "docs/posterior-rank-distortion-theory.md"

    if "boundary_generalized_eigengap" in Path(source).read_text(encoding="utf-8"):
        print("Repeated-eigenvalue semantics are already present.")
        return

    replace_once(
        source,
        '''class PosteriorRankDistortionPoint:
    """One globally optimal retained-rank point for the registered objective."""

    retained_rank: int
    discarded_dimension: int
    latent_projection: Array
    compressed_factor_m: Array
    optimal_normalized_covariance_trace_loss: float
    audited_normalized_covariance_trace_loss: float
    maximum_normalized_covariance_contraction: float
    mean_shift_risk: float
    mean_shift_risk_upper_bound: float
    exact_posterior: bool
''',
        '''class PosteriorRankDistortionPoint:
    """One globally optimal point, including uniqueness of its rank cut.

    ``optimal_subspace_unique`` is true only when the generalized-eigenvalue
    boundary is strict (or the retained/discarded subspace is trivial). When
    false, the optimum value is unique but multiple optimal factor covariances
    exist.
    """

    retained_rank: int
    discarded_dimension: int
    latent_projection: Array
    compressed_factor_m: Array
    optimal_normalized_covariance_trace_loss: float
    audited_normalized_covariance_trace_loss: float
    maximum_normalized_covariance_contraction: float
    mean_shift_risk: float
    mean_shift_risk_upper_bound: float
    boundary_generalized_eigengap: float | None
    optimal_subspace_unique: bool
    exact_posterior: bool
''',
    )
    replace_once(
        source,
        '''            "mean_shift_risk": self.mean_shift_risk,
            "mean_shift_risk_upper_bound": self.mean_shift_risk_upper_bound,
            "exact_posterior": self.exact_posterior,
''',
        '''            "mean_shift_risk": self.mean_shift_risk,
            "mean_shift_risk_upper_bound": self.mean_shift_risk_upper_bound,
            "boundary_generalized_eigengap": self.boundary_generalized_eigengap,
            "optimal_subspace_unique": self.optimal_subspace_unique,
            "exact_posterior": self.exact_posterior,
''',
    )
    replace_once(
        source,
        '''            mean_shift_risk=0.0,
            mean_shift_risk_upper_bound=0.0,
            exact_posterior=True,
''',
        '''            mean_shift_risk=0.0,
            mean_shift_risk_upper_bound=0.0,
            boundary_generalized_eigengap=None,
            optimal_subspace_unique=True,
            exact_posterior=True,
''',
    )
    replace_once(
        source,
        '''    for retained_rank in range(rank + 1):
        discarded_dimension = rank - retained_rank
        if discarded_dimension:
''',
        '''    for retained_rank in range(rank + 1):
        discarded_dimension = rank - retained_rank
        if 0 < discarded_dimension < rank:
            boundary_gap = max(
                float(
                    eigenvalues[discarded_dimension]
                    - eigenvalues[discarded_dimension - 1]
                ),
                0.0,
            )
            optimal_subspace_unique = boundary_gap > tolerance * scale
        else:
            boundary_gap = None
            optimal_subspace_unique = True

        if discarded_dimension:
''',
    )
    replace_once(
        source,
        '''                mean_shift_risk=mean_shift_risk,
                mean_shift_risk_upper_bound=mean_bound,
                exact_posterior=audited_trace <= 1e-10 * max(1.0, optimal_trace),
''',
        '''                mean_shift_risk=mean_shift_risk,
                mean_shift_risk_upper_bound=mean_bound,
                boundary_generalized_eigengap=boundary_gap,
                optimal_subspace_unique=optimal_subspace_unique,
                exact_posterior=audited_trace <= 1e-10 * max(1.0, optimal_trace),
''',
    )
    replace_once(
        source,
        '''The objective is posterior covariance contraction, not observation likelihood,
full posterior KL, or end-to-end task loss.  Every returned point independently
audits the exact contraction and expected posterior-normalized mean-shift risk.
''',
        '''The objective is posterior covariance contraction, not observation likelihood,
full posterior KL, or end-to-end task loss.  A rank cut inside a repeated
generalized-eigenvalue block has a unique optimum value but a non-unique factor
subspace; every point reports that distinction and independently audits the exact
contraction and expected posterior-normalized mean-shift risk.
''',
    )

    replace_once(
        tests,
        '''    for rank in range(8):
        first = original.point(rank).compressed_factor_m.reshape(36, rank)
        second = latent_changed.point(rank).compressed_factor_m.reshape(36, rank)
        np.testing.assert_allclose(first @ first.T, second @ second.T, atol=2e-10)
''',
        '''    nonunique_ranks: list[int] = []
    for rank in range(8):
        first_point = original.point(rank)
        second_point = latent_changed.point(rank)
        np.testing.assert_allclose(
            first_point.audited_normalized_covariance_trace_loss,
            second_point.audited_normalized_covariance_trace_loss,
            atol=2e-11,
            rtol=2e-10,
        )
        assert first_point.optimal_subspace_unique == second_point.optimal_subspace_unique
        if not first_point.optimal_subspace_unique:
            nonunique_ranks.append(rank)
            continue
        first = first_point.compressed_factor_m.reshape(36, rank)
        second = second_point.compressed_factor_m.reshape(36, rank)
        np.testing.assert_allclose(first @ first.T, second @ second.T, atol=2e-10)
    assert nonunique_ranks == [4, 5, 6]
''',
    )
    replace_once(
        tests,
        '''    np.testing.assert_allclose(
        original.generalized_eigenvalues,
        query_changed.generalized_eigenvalues,
        atol=2e-11,
        rtol=2e-10,
    )


def test_budget_selection_is_fail_closed_to_a_valid_rank() -> None:
''',
        '''    np.testing.assert_allclose(
        original.generalized_eigenvalues,
        query_changed.generalized_eigenvalues,
        atol=2e-11,
        rtol=2e-10,
    )
    for rank in range(8):
        original_point = original.point(rank)
        changed_point = query_changed.point(rank)
        np.testing.assert_allclose(
            original_point.audited_normalized_covariance_trace_loss,
            changed_point.audited_normalized_covariance_trace_loss,
            atol=2e-11,
            rtol=2e-10,
        )
        assert original_point.optimal_subspace_unique == changed_point.optimal_subspace_unique


def test_budget_selection_is_fail_closed_to_a_valid_rank() -> None:
''',
    )
    replace_once(
        tests,
        '''    assert result.original_rank == 0
    assert result.point(0).exact_posterior
''',
        '''    assert result.original_rank == 0
    assert result.point(0).exact_posterior
    assert result.point(0).optimal_subspace_unique
    assert result.point(0).boundary_generalized_eigengap is None
''',
    )

    replace_once(
        docs,
        '''An optimal discarded subspace is the span of the `d` generalized eigenvectors
with smallest eigenvalues.  Its Euclidean orthogonal complement gives the
rank-`k=r-d` retained projector `V`.  The subspaces are nested, so all retained
ranks form one deterministic frontier.
''',
        '''An optimal discarded subspace is the span of the `d` generalized eigenvectors
with smallest eigenvalues.  Its Euclidean orthogonal complement gives the
rank-`k=r-d` retained projector `V`.  Choosing nested eigenspaces gives a nested
frontier.  If the rank boundary cuts through a repeated generalized-eigenvalue
block, the optimum value remains unique but the optimizing subspace is not.  The
implementation reports the boundary eigengap and an `optimal_subspace_unique`
flag; factor-covariance representatives should only be compared across latent
reparameterizations at strict boundaries.
''',
    )

    print("Applied repeated-eigenvalue semantics.")


if __name__ == "__main__":
    main()
