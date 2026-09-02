"""One-shot reviewed transformation for the spectral validity certificate."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

SOURCE = Path("src/prob4d/posterior_rank_distortion.py")
TESTS = Path("tests/test_posterior_rank_distortion.py")
DOCS = Path("docs/posterior-rank-distortion-theory.md")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one replacement target, found {count}; "
            f"target={old[:200]!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


source_text = SOURCE.read_text(encoding="utf-8")
if "minimum_possible_maximum_contraction" in source_text:
    print("spectral validity certificate already applied")
    raise SystemExit(0)

replace_once(
    SOURCE,
    dedent(
        '''\
        The objective is posterior covariance contraction, not observation likelihood,
        full posterior KL, or end-to-end task loss. A rank cut inside a repeated
        generalized-eigenvalue block has a unique optimum value but a non-unique factor
        subspace; every point reports that distinction and independently audits the exact
        contraction and expected posterior-normalized mean-shift risk.
        '''
    ),
    dedent(
        '''\
        After whitening by the innovation-remainder and full query-posterior metrics,
        every contraction is ``H.T @ Pi @ H`` for an orthogonal discarded-subspace
        projector ``Pi``. Singular-value interlacing therefore makes the same nested
        frontier componentwise spectral-minimal, not merely trace-optimal. In particular,
        it gives the globally minimum possible worst-direction contraction and an exact
        feasibility certificate for positive-definite reduced query posteriors.

        The objective family is posterior covariance contraction, not observation
        likelihood, full posterior KL, or end-to-end task loss. A rank cut inside a
        repeated generalized-eigenvalue block has a unique optimum value but a non-unique
        factor subspace; every point reports that distinction and independently audits the
        exact contraction and expected posterior-normalized mean-shift risk.
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
        class PosteriorRankDistortionPoint:
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
        '''
    ),
    dedent(
        '''\
        class PosteriorRankDistortionPoint:
            """One globally optimal point, including rank validity and uniqueness.

            ``optimal_subspace_unique`` is true only when the generalized-eigenvalue
            boundary is strict (or the retained/discarded subspace is trivial). When
            false, the optimum value is unique but multiple optimal factor covariances
            exist.

            ``minimum_possible_maximum_contraction`` is the global minimum, over all
            equal-rank factor projections, of the largest posterior-normalized covariance
            contraction eigenvalue. Therefore ``valid_projection_exists_at_rank`` is an
            existence certificate, while ``posterior_valid`` audits the returned frontier
            representative itself.
            """

            retained_rank: int
            discarded_dimension: int
            latent_projection: Array
            compressed_factor_m: Array
            optimal_normalized_covariance_trace_loss: float
            audited_normalized_covariance_trace_loss: float
            maximum_normalized_covariance_contraction: float
            minimum_possible_maximum_contraction: float
            mean_shift_risk: float
            mean_shift_risk_upper_bound: float
            boundary_generalized_eigengap: float | None
            optimal_subspace_unique: bool
            valid_projection_exists_at_rank: bool
            posterior_valid: bool
            exact_posterior: bool
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                    "maximum_normalized_covariance_contraction": (
                        self.maximum_normalized_covariance_contraction
                    ),
                    "mean_shift_risk": self.mean_shift_risk,
        '''
    ),
    dedent(
        '''\
                    "maximum_normalized_covariance_contraction": (
                        self.maximum_normalized_covariance_contraction
                    ),
                    "minimum_possible_maximum_contraction": (
                        self.minimum_possible_maximum_contraction
                    ),
                    "mean_shift_risk": self.mean_shift_risk,
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                    "boundary_generalized_eigengap": self.boundary_generalized_eigengap,
                    "optimal_subspace_unique": self.optimal_subspace_unique,
                    "exact_posterior": self.exact_posterior,
        '''
    ),
    dedent(
        '''\
                    "boundary_generalized_eigengap": self.boundary_generalized_eigengap,
                    "optimal_subspace_unique": self.optimal_subspace_unique,
                    "valid_projection_exists_at_rank": (
                        self.valid_projection_exists_at_rank
                    ),
                    "posterior_valid": self.posterior_valid,
                    "exact_posterior": self.exact_posterior,
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
            def point(self, retained_rank: int) -> PosteriorRankDistortionPoint:
                if not 0 <= retained_rank <= self.original_rank:
                    raise ValueError("retained_rank lies outside the frontier")
                return self.points[retained_rank]

            def minimum_rank_for_trace_budget(self, budget: float) -> PosteriorRankDistortionPoint:
        '''
    ),
    dedent(
        '''\
            def point(self, retained_rank: int) -> PosteriorRankDistortionPoint:
                if not 0 <= retained_rank <= self.original_rank:
                    raise ValueError("retained_rank lies outside the frontier")
                return self.points[retained_rank]

            @property
            def minimum_valid_retained_rank(self) -> int:
                """Smallest rank for which any projected factor can remain Bayesian-valid."""
                return self.minimum_rank_for_spectral_budget(1.0).retained_rank

            def minimum_rank_for_spectral_budget(
                self,
                budget: float,
            ) -> PosteriorRankDistortionPoint:
                """Return the least rank whose globally minimal worst contraction is below budget."""
                if isinstance(budget, (bool, np.bool_)) or not isinstance(
                    budget, (int, float, np.integer, np.floating)
                ):
                    raise TypeError("budget must be a positive real scalar")
                threshold = float(budget)
                if not np.isfinite(threshold) or threshold <= 0.0:
                    raise ValueError("budget must be finite and positive")
                for point in self.points:
                    if point.minimum_possible_maximum_contraction < threshold:
                        return point
                return self.points[-1]

            def minimum_rank_for_trace_budget(self, budget: float) -> PosteriorRankDistortionPoint:
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                    "shared_precision_max_eigenvalue": self.shared_precision_max_eigenvalue,
                    "generalized_eigenvalues": self.generalized_eigenvalues.tolist(),
        '''
    ),
    dedent(
        '''\
                    "shared_precision_max_eigenvalue": self.shared_precision_max_eigenvalue,
                    "minimum_valid_retained_rank": self.minimum_valid_retained_rank,
                    "generalized_eigenvalues": self.generalized_eigenvalues.tolist(),
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                    "claim_boundary": (
                        "globally optimal normalized posterior-covariance trace contraction "
                        "within one frozen U->UV factor-projection family; not observation evidence"
                    ),
        '''
    ),
    dedent(
        '''\
                    "claim_boundary": (
                        "spectral-majorization and trace-optimal posterior-covariance "
                        "contraction within one frozen U->UV factor-projection family; "
                        "not observation evidence"
                    ),
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                    maximum_normalized_covariance_contraction=0.0,
                    mean_shift_risk=0.0,
        '''
    ),
    dedent(
        '''\
                    maximum_normalized_covariance_contraction=0.0,
                    minimum_possible_maximum_contraction=0.0,
                    mean_shift_risk=0.0,
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                    boundary_generalized_eigengap=None,
                    optimal_subspace_unique=True,
                    exact_posterior=True,
        '''
    ),
    dedent(
        '''\
                    boundary_generalized_eigengap=None,
                    optimal_subspace_unique=True,
                    valid_projection_exists_at_rank=True,
                    posterior_valid=True,
                    exact_posterior=True,
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                optimal_trace = max(float(cumulative[discarded_dimension]), 0.0)
                audit_scale = max(optimal_trace, audited_trace, 1.0)
        '''
    ),
    dedent(
        '''\
                minimum_maximum_contraction = (
                    max(float(eigenvalues[discarded_dimension - 1]), 0.0)
                    if discarded_dimension
                    else 0.0
                )
                spectral_scale = max(
                    minimum_maximum_contraction,
                    maximum_contraction,
                    1.0,
                )
                if (
                    abs(maximum_contraction - minimum_maximum_contraction)
                    > 1e-9 * spectral_scale
                ):
                    raise ValueError(
                        "generalized-eigenvalue spectral-minimax identity failed its audit"
                    )
                valid_projection_exists = minimum_maximum_contraction < 1.0
                posterior_valid = maximum_contraction < 1.0

                optimal_trace = max(float(cumulative[discarded_dimension]), 0.0)
                audit_scale = max(optimal_trace, audited_trace, 1.0)
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                        maximum_normalized_covariance_contraction=maximum_contraction,
                        mean_shift_risk=mean_shift_risk,
        '''
    ),
    dedent(
        '''\
                        maximum_normalized_covariance_contraction=maximum_contraction,
                        minimum_possible_maximum_contraction=(
                            minimum_maximum_contraction
                        ),
                        mean_shift_risk=mean_shift_risk,
        '''
    ),
)
replace_once(
    SOURCE,
    dedent(
        '''\
                        boundary_generalized_eigengap=boundary_gap,
                        optimal_subspace_unique=optimal_subspace_unique,
                        exact_posterior=audited_trace <= 1e-10 * max(1.0, optimal_trace),
        '''
    ),
    dedent(
        '''\
                        boundary_generalized_eigengap=boundary_gap,
                        optimal_subspace_unique=optimal_subspace_unique,
                        valid_projection_exists_at_rank=valid_projection_exists,
                        posterior_valid=posterior_valid,
                        exact_posterior=audited_trace <= 1e-10 * max(1.0, optimal_trace),
        '''
    ),
)

replace_once(
    TESTS,
    dedent(
        '''\
        def direct_trace_distortion(
            full_posterior: np.ndarray,
            reduced_posterior: np.ndarray,
        ) -> float:
            loss = full_posterior - reduced_posterior
            return float(np.trace(np.linalg.solve(full_posterior, loss)))
        '''
    ),
    dedent(
        '''\
        def normalized_contraction_spectrum(
            full_posterior: np.ndarray,
            reduced_posterior: np.ndarray,
        ) -> np.ndarray:
            loss = full_posterior - reduced_posterior
            root = np.linalg.cholesky(full_posterior)
            left = np.linalg.solve(root, loss)
            normalized = np.linalg.solve(root, left.T).T
            normalized = 0.5 * (normalized + normalized.T)
            return np.linalg.eigvalsh(normalized)


        def direct_trace_distortion(
            full_posterior: np.ndarray,
            reduced_posterior: np.ndarray,
        ) -> float:
            return float(
                np.sum(
                    normalized_contraction_spectrum(
                        full_posterior,
                        reduced_posterior,
                    )
                )
            )
        '''
    ),
)
replace_once(
    TESTS,
    dedent(
        '''\
                dense_distortion = direct_trace_distortion(full_posterior, reduced_posterior)
                np.testing.assert_allclose(
                    point.audited_normalized_covariance_trace_loss,
        '''
    ),
    dedent(
        '''\
                spectrum = normalized_contraction_spectrum(
                    full_posterior,
                    reduced_posterior,
                )
                dense_distortion = float(np.sum(spectrum))
                np.testing.assert_allclose(
                    point.maximum_normalized_covariance_contraction,
                    max(float(spectrum[-1]), 0.0),
                    atol=2e-11,
                    rtol=2e-10,
                )
                np.testing.assert_allclose(
                    point.minimum_possible_maximum_contraction,
                    point.maximum_normalized_covariance_contraction,
                    atol=2e-11,
                    rtol=2e-10,
                )
                assert point.valid_projection_exists_at_rank == (
                    point.minimum_possible_maximum_contraction < 1.0
                )
                assert point.posterior_valid == (
                    float(np.linalg.eigvalsh(reduced_posterior)[0]) > 0.0
                )
                np.testing.assert_allclose(
                    point.audited_normalized_covariance_trace_loss,
        '''
    ),
)
start = TESTS.read_text(encoding="utf-8").index(
    "def test_generalized_eigen_frontier_beats_sampled_same_rank_subspaces()"
)
end = TESTS.read_text(encoding="utf-8").index(
    "def test_existing_euclidean_svd_order_is_not_globally_trace_optimal()",
    start,
)
text = TESTS.read_text(encoding="utf-8")
replacement = dedent(
    '''\
    def test_generalized_eigen_frontier_beats_sampled_same_rank_subspaces() -> None:
        factor, prior, cross, remainder, innovation = model(seed=9, rank=4, qdim=2)
        result = frontier(factor, prior, cross, innovation)
        _, full_posterior = posterior(prior, cross, innovation)
        rng = np.random.default_rng(20260902)

        for retained_rank in (1, 2, 3):
            point = result.point(retained_rank)
            optimum_trace = point.audited_normalized_covariance_trace_loss
            optimum_maximum = point.minimum_possible_maximum_contraction
            for _ in range(300):
                projection = _random_orthonormal_columns(rng, 4, retained_rank)
                reduced = factor @ projection
                _, candidate_posterior = posterior(
                    prior,
                    cross,
                    remainder + reduced @ reduced.T,
                )
                spectrum = normalized_contraction_spectrum(
                    full_posterior,
                    candidate_posterior,
                )
                candidate_trace = float(np.sum(spectrum))
                candidate_maximum = max(float(spectrum[-1]), 0.0)
                assert candidate_trace >= optimum_trace - 2e-10
                assert candidate_maximum >= optimum_maximum - 2e-10


    '''
)
TESTS.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
replace_once(
    TESTS,
    dedent(
        '''\
                assert first_point.optimal_subspace_unique == second_point.optimal_subspace_unique
                if not first_point.optimal_subspace_unique:
        '''
    ),
    dedent(
        '''\
                assert first_point.optimal_subspace_unique == second_point.optimal_subspace_unique
                np.testing.assert_allclose(
                    first_point.minimum_possible_maximum_contraction,
                    second_point.minimum_possible_maximum_contraction,
                    atol=2e-11,
                    rtol=2e-10,
                )
                assert (
                    first_point.valid_projection_exists_at_rank
                    == second_point.valid_projection_exists_at_rank
                )
                if not first_point.optimal_subspace_unique:
        '''
    ),
)
replace_once(
    TESTS,
    dedent(
        '''\
            with pytest.raises(ValueError, match="budget"):
                result.minimum_rank_for_trace_budget(-0.1)


        def test_zero_shared_rank_and_invalid_inputs() -> None:
        '''
    ),
    dedent(
        '''\
            with pytest.raises(ValueError, match="budget"):
                result.minimum_rank_for_trace_budget(-0.1)

            assert result.minimum_rank_for_spectral_budget(1.0).retained_rank == 0
            with pytest.raises(TypeError, match="budget"):
                result.minimum_rank_for_spectral_budget(True)
            with pytest.raises(ValueError, match="budget"):
                result.minimum_rank_for_spectral_budget(0.0)


        def test_spectral_certificate_proves_lower_rank_posterior_infeasible() -> None:
            factor = np.array([[np.sqrt(0.99)], [0.0], [0.0]])
            prior = np.ones((1, 1))
            cross = np.array([[np.sqrt(0.99), 0.0, 0.0]])
            remainder = np.diag([0.01, 1.0, 1.0])
            innovation = remainder + factor @ factor.T
            result = frontier(factor, prior, cross, innovation)

            invalid = result.point(0)
            np.testing.assert_allclose(
                invalid.minimum_possible_maximum_contraction,
                9801.0,
                atol=1e-8,
                rtol=1e-12,
            )
            assert not invalid.valid_projection_exists_at_rank
            assert not invalid.posterior_valid
            assert result.minimum_valid_retained_rank == 1
            assert result.minimum_rank_for_spectral_budget(1.0).retained_rank == 1

            valid = result.point(1)
            assert valid.minimum_possible_maximum_contraction == 0.0
            assert valid.valid_projection_exists_at_rank
            assert valid.posterior_valid
            assert valid.exact_posterior


        def test_zero_shared_rank_and_invalid_inputs() -> None:
        '''
    ),
)
replace_once(
    TESTS,
    dedent(
        '''\
            assert result.point(0).optimal_subspace_unique
            assert result.point(0).boundary_generalized_eigengap is None
        '''
    ),
    dedent(
        '''\
            assert result.point(0).optimal_subspace_unique
            assert result.point(0).boundary_generalized_eigengap is None
            assert result.point(0).minimum_possible_maximum_contraction == 0.0
            assert result.point(0).valid_projection_exists_at_rank
            assert result.point(0).posterior_valid
            assert result.minimum_valid_retained_rank == 0
        '''
    ),
)

replace_once(
    DOCS,
    dedent(
        '''\
        This is a **global optimum only for this registered distortion and this
        `U -> U V` factor family**.  It is not a claim of globally optimal Bayesian model
        reduction.

        ### Exact theorem as the zero-distortion endpoint
        '''
    ),
    dedent(
        r'''\
        ### Stronger theorem: spectral majorization and validity feasibility

        Define the whitened latent response

        \[
        H=M^{-1/2}RP^{-1/2}.
        \]

        For any discarded span `N`, let

        \[
        X=M^{1/2}N(N^\top MN)^{-1/2},\qquad X^\top X=I.
        \]

        The normalized posterior contraction is then

        \[
        W=P^{-1/2}L_NP^{-1/2}=H^\top XX^\top H.
        \]

        Let `h=rank(H)=rank(R)` and write the positive generalized eigenvalues in
        ascending order as

        \[
        0<\mu_1\le\cdots\le\mu_h.
        \]

        Retaining rank `k<h` forces at least `s=h-k` query-relevant directions into
        the discarded subspace.  Singular-value interlacing gives, for every equal-rank
        projection and contraction eigenvalues `omega_1 >= ... >= omega_s > 0`,

        \[
        \omega_i\ge\mu_{s-i+1},\qquad i=1,\ldots,s.
        \]

        The generalized-eigen frontier attains equality componentwise.  Consequently it
        minimizes every monotone unitarily invariant norm of `W`, including both

        \[
        \min\operatorname{tr}(W)=\sum_{i=1}^{h-k}\mu_i
        \]

        and

        \[
        \boxed{\min\lambda_{\max}(W)=\mu_{h-k}.}
        \]

        This yields an exact rank-feasibility certificate.  Because a reduced query
        posterior is positive definite iff `lambda_max(W)<1`, an admissible rank-`k`
        factor projection exists iff

        \[
        \boxed{\mu_{h-k}<1}
        \]

        for `k<h`; every `k>=h` has the zero-distortion exact solution.  Equivalently,
        the minimum valid retained rank is the number of positive generalized
        eigenvalues greater than or equal to one.

        Therefore an invalid generalized-eigen frontier point is not merely a poor
        optimizer: no equal-rank orthogonal projection in the registered factor family
        can produce a positive-definite query posterior.  The implementation reports
        this distinction as `valid_projection_exists_at_rank` and audits the returned
        representative separately as `posterior_valid`.

        These are **global optima only for posterior covariance contraction within the
        registered `U -> U V` factor family**.  They are not claims of globally optimal
        Bayesian model reduction or observation-likelihood preservation.

        ### Exact theorem as the zero-distortion endpoint
        '''
    ),
)

print("spectral validity certificate transformation applied")
