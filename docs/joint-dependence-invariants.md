# Joint-dependence invariants

Prob4D's central statistical object is not a collection of independent point
covariances. For admitted observation rows, the linearized model is

\[
y = \mu + J_g\,\delta g + \varepsilon,
\qquad
\delta g \sim \mathcal N(0,\Sigma_g),
\qquad
\varepsilon \sim \mathcal N(0,R),
\]

where `R` is block diagonal conditional point uncertainty and the gauge latent
`delta g` is shared by every row that depends on the same window-gauge tree.
The resulting joint covariance is

\[
\Sigma_y = R + J_g\Sigma_gJ_g^\top.
\]

The tests in `tests/test_joint_dependence_invariants.py` make four consequences
of this model executable.

## 1. Explicit and collapsed Gaussian representations agree

An affine transformation of jointly Gaussian variables is Gaussian. Therefore,
marginalizing the explicit gauge latent gives exactly

\[
y \sim \mathcal N(\mu,\Sigma_y).
\]

A downstream estimator may consequently keep gauge errors as explicit nuisance
variables or consume the correctly collapsed joint covariance. For the same
mean, residual, conditional covariance, Jacobian, and gauge prior, both routes
must produce the same inverse action, determinant, and Gaussian negative log
likelihood.

This is a representation equivalence, not permission to discard cross-row
covariance. A collapsed artifact is equivalent only when it preserves the full
joint covariance semantics required by the downstream query.

## 2. Rowwise marginal proper scores are not joint proper scores

Consider two scalar rows with independent conditional noise variance
\(\sigma^2\) and one shared additive gauge error with variance \(\tau^2\):

\[
\Sigma =
\begin{bmatrix}
\sigma^2+\tau^2 & \tau^2\\
\tau^2 & \sigma^2+\tau^2
\end{bmatrix}.
\]

The common residual \([a,a]^\top\) and contrast residual \([a,-a]^\top\) have
the same rowwise values and therefore receive identical sums of marginal
Gaussian scores. Jointly, however, they lie in different eigendirections:

- the common mode has variance \(\sigma^2+2\tau^2\);
- the contrast mode has variance \(\sigma^2\).

A joint score correctly treats the contrast as much less plausible when the
conditional noise is small. Summing rowwise marginal scores erases precisely the
dependence information that Prob4D is intended to preserve.

## 3. Query covariance can cancel or amplify shared uncertainty

For the same two-row model, the average query

\[
q_{+}=\tfrac12(y_1+y_2)
\]

retains the complete shared gauge variance, whereas the difference query

\[
q_{-}=y_1-y_2
\]

cancels it exactly. Their variances are

\[
\operatorname{Var}(q_+)=\tfrac12\sigma^2+\tau^2,
\qquad
\operatorname{Var}(q_-)=2\sigma^2.
\]

Consequently, a collection of per-row marginal covariance blocks is
insufficient for physical queries spanning multiple rows. The query Jacobian
must act on the complete conditional-plus-shared covariance model.

## 4. Structured factor storage avoids dense quadratic growth

A dense covariance for `M` three-dimensional rows requires storage proportional
to \((3M)^2\). The covariance-root Woodbury backend retains one conditional
`3 x 3` factor per row plus gauge-root and latent factors. With fixed shared
latent rank, its cached storage grows linearly in `M`; the input factor stack is
already owned separately. The causal tree backend similarly retains fixed-size
row and seven-dimensional gauge blocks rather than a dense observation
covariance.

The executable storage invariant compares otherwise identical two-row and
128-row stacks. It requires the structured-to-dense storage ratio to decrease
with observation count and to remain below one percent for the registered
128-row shared-rank example. This is a deterministic storage check, not a timing
benchmark or a universal hardware-performance claim.

## Ownership and claim boundary

Prob4D owns the observation-space conditional covariance, shared gauge
structure, exact joint Gaussian operations, and query projection. BayesianPhysTwin
owns the physical residual or query Jacobian, update guard, identifiability test,
and exact physical fallback. Causal4D consumes only an accepted BayesianPhysTwin
belief and owns intervention inference.

These invariants establish algebraic consistency and the necessity of retaining
shared dependence. They do not establish real-provider competence, calibrated
physical uncertainty, downstream physical benefit, Causal4D intervention
benefit, deployment safety, or state of the art.
