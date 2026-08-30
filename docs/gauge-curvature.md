# Shared-gauge curvature for physical 4D queries

**Status:** experimental, additive numerical method; not a provider-v2 exporter,
calibration certificate, or change to any existing admission rule.

## Motivation and the exact boundary of the finding

Prob4D's source-only closure diagnostic compares analytic first-order propagation
with a spherical-radial rule evaluating `+/- sqrt(r) L[:, j]`. The relevant
upstream implementation is `_sigma_points` in
`src/prob4d/_gauge_linearization_numerics.py`, inspected at commit
`224d13dc9a93731ac5297b479eb1e121b3dbe659` (blob
`9dd800dbdd9b2f262eec0a50ddc3aaa2d871873c`). See the existing
[gauge linearization closure guide](gauge-linearization-closure.md).

Those two approximations can agree on zero query uncertainty even when the
nonlinear query has positive variance. This is a query-level blind spot, **not**
a demonstrated false pass of the entire existing closure/admission workflow.
Other point-level, mean-shift, support, or physical guards may still reject.
The original diagnostic and its frozen artifacts are intentionally unchanged.

Take a point `(0, 0, ell)` and two independent zero-mean Gaussian angles with
variance `sigma^2`. Compose `Rz(alpha) Ry(beta)` and query the y coordinate:

\[
q = \ell\sin\alpha\sin\beta,\qquad
\mathbb E q=0,\qquad
\operatorname{Var}(q)=\ell^2
\left(\frac{1-e^{-2\sigma^2}}{2}\right)^2.
\]

The first derivative at zero vanishes. Every axis sigma point also makes one
angle zero. Both approximations therefore report zero query variance. With
`ell = 0.5 m` and `sigma = 0.1 rad`, the exact standard deviation is
`4.950331673 mm`. A multiplicative covariance inflation cannot recover a
missing direction whose original covariance is exactly zero.

## Classical quadratic moments, represented jointly

For a *complete joint* root `L`, write `g = mu + L z`, `z ~ N(0, I_r)`. Let a
quadratic surrogate for each output coordinate be

\[
\widehat f_i(z)=c_i+a_i^\top z+\tfrac12 z^\top B_i z.
\]

Its exact Gaussian moments are

\[
m_i=c_i+\tfrac12\operatorname{tr}(B_i),\qquad
\Sigma_{ij}=a_i^\top a_j+\tfrac12\operatorname{tr}(B_i B_j).
\]

These are established Gaussian moment identities, not a new filtering rule.
They follow by expanding the polynomial and using zero odd Gaussian moments
and the Gaussian fourth-moment identity.

Store the joint covariance as `F F.T` with columns

\[
F=\left[A,\;\{B_{:,kk}/\sqrt2\}_k,\;\{B_{:,kl}\}_{k<l}\right].
\]

The associated centered polynomial features are `z_k`,
`(z_k^2 - 1)/sqrt(2)`, and `z_k z_l`. They have identity second-moment matrix
but are **not independent Gaussian variables**. This representation is a
joint, positive-semidefinite second-moment representation, not an exact
Gaussian pushforward and not a collection of new physical gauge freedoms.

A fixed linear query `C` uses `C m` and `C F`, preserving correlations and
cancellations across points. Adding a separate diagonal variance to every point
is not equivalent. Exact quadratic moments are invariant under orthogonal
changes of whitening basis; axis cubature need not be.

The input/output cross covariance of the surrogate is `L A.T`. Curvature
features contribute no input cross covariance under the stated Gaussian input,
because the relevant odd moments vanish. Calling this API with a different
root of the same shape does not establish correct source lineage.

## A useful query-specific local result

Let `q(epsilon) = C f(mu + epsilon L z)` with fixed `C`, and suppose the
first-order query sensitivity `C J L` is zero. Assume a third-order Taylor
expansion with an `L2` remainder of order `epsilon^4`, and finite moments of
the displayed derivatives. If `B_i` is the Hessian of query coordinate i in
these whitened coordinates, then

\[
\mathbb E q_i = q_i(0)+\frac{\epsilon^2}{2}\operatorname{tr}(B_i)
+O(\epsilon^4),
\]
\[
\operatorname{Cov}(q_i,q_j)
=\frac{\epsilon^4}{2}\operatorname{tr}(B_i B_j)+O(\epsilon^6).
\]

**Proof.** The linear term is absent. After centering, the leading random term
is the centered quadratic polynomial of order `epsilon^2`. Its covariance is
the Gaussian fourth-moment expression above. The covariance between the
quadratic and cubic terms vanishes by Gaussian sign symmetry. Covariance with
the fourth-order remainder and the cubic variance are of order `epsilon^6`
under the assumed `L2` control. The mean of the cubic term is zero.

Thus quadratic propagation recovers the leading nonzero uncertainty for this
first-order-null query class. This does not imply a globally conservative
bound. For a general query with nonzero linear sensitivity, omitted
linear--cubic cross terms also occur at order `epsilon^4`; the quadratic
surrogate is **not** a complete covariance expansion through that order.

## Implementation

`src/prob4d/gauge_curvature.py` adds an experimental module only. The stable
`prob4d.api.v2` facade, CLI registry, artifact schemas, and existing closure
rule are not modified.

- `quadratic_gaussian_moments` consumes exact local quadratic coefficients.
- `finite_difference_gauge_moments` computes linear, diagonal, and *all mixed*
  curvature coefficients using a centered derivative stencil.
- `sim3_chain_gauge_moments` evaluates actual Prob4D `Sim3` composition and
  transforms the local points, optionally projecting directly to a fixed query.
- `SharedGaugeMoments` supports joint covariance factors, covariance actions,
  marginal variances, fixed-query projection, and input cross covariance.

The derivative stencil uses `1 + 2 r^2` function calls. It is not an integration
rule. Output factors require `p * [r + r(r+1)/2]` entries; direct projection
replaces p by the query dimension. The default rank cap is 32 and raises an
error rather than truncating uncertainty. No horizon-independent cost,
automatic sparse reduction, or runtime speedup has been demonstrated.

```python
import numpy as np
from prob4d.gauge_curvature import sim3_chain_gauge_moments

# Prob4D vector convention: [log_scale, rotvec(3), translation(3)].
# T0 = Rz(alpha), T1 = Ry(beta), independent sigma = 0.1 rad.
root = np.zeros((14, 2))
root[3, 0] = 0.1
root[9, 1] = 0.1
moments = sim3_chain_gauge_moments(
    transform_vectors=np.zeros((2, 7)),
    joint_covariance_root=root,
    points_local_m=np.array([[0.0, 0.0, 0.5]]),
    query_matrix=np.array([[0.0, 1.0, 0.0]]),
    step=1e-3,
)
print(1000 * np.sqrt(moments.marginal_variance))  # approximately [5.0] mm
print(moments.evaluation_count)  # 9
```

## Controlled study and required baselines

After applying the patch to the pinned Prob4D checkout and installing it:

```bash
python -m pytest tests/test_gauge_curvature.py tests/test_gauge_curvature_study.py
python examples/gauge_curvature_study.py --output-dir outputs/gauge-curvature-v1
```

The script refuses to overwrite a nonempty output directory. It verifies that
both imported modules come from this checkout and that `sim3.py` has the
pinned upstream Git blob identity. Its manifest records Python/NumPy versions,
source hashes and output hashes. Candidate commit is null because the supplied
patch was not pushed to GitHub.

The study includes analytic truth, third-degree spherical-radial cubature,
classical fifth-degree cubature, three-node-per-dimension Gauss--Hermite, and
15-node-per-dimension Gauss--Hermite as an independent numerical reference.
At rank two, three-node tensor Gauss--Hermite uses the same nine calls and is
more accurate on the sine-product example. Classical fifth-degree cubature
also has quadratic node count; no new asymptotic quadrature complexity is
claimed. Its signed weights can yield a non-PSD covariance for some nonlinear
maps, but that fact is not evidence of a performance advantage on real data.

A separate scalar state-update analogue has observations
`y_j = x + g + e_j`, with common sine-product gauge error g and independent
point errors e_j. Hence the averaged measurement variance is
`Var(g) + Var(e_j)/N`, not `(Var(g) + Var(e_j))/N`. The study compares
uncorrected, pointwise-curvature, joint-curvature, quadrature, oracle-second-
moment, and exact physical-prior fallback arms. Risks are analytic; coverage
is measured separately using independent synthetic episodes.

The full result bundle belongs to the paper repository at
`evidence/prob4d_curvature_v1/`. It is a local exploratory mechanism study,
not a preregistered real-data or cross-repository result. An oracle using the
correct second moments is an optimal linear estimator, not the exact Bayesian
posterior for non-Gaussian sine-product noise.

## Integration and evidence boundaries

Do not feed polynomial features into existing gauge-state fields or silently
relax a claim-bearing loader. A future provider integration must specify the
new shared-feature semantics, cross-window/root lineage, conditional-noise
model, calibration partition, and downstream complete-belief routing in a
separately versioned experiment. Conditional point noise is not included in
this prototype. It may require further nonlinear marginalization if its
covariance depends on gauge.

Only the forward point map is differentiated here. The implementation does
not certify a logarithmic gauge state, remove existing branch-cut checks,
choose provider support, authorize target access, or select physical updates.
Neither BayesianPhysTwin nor Causal4D was executed in this study. Existing
fallback behavior might already reject the harmful unguarded comparator.

Near-nominal NEES does not imply distributional calibration. Gaussian intervals
are measured only for the stated controlled case. Larger gauge variances can
make the Taylor model inaccurate: at sigma = 0.35 rad the example variance is
overestimated by 27.12%. More points do not supply more independent gauge
realizations or independent physical objects.

## References and inspected source

1. Shishan Yang and Marcus Baum. *Second-Order Extended Kalman Filter for
   Extended Object and Group Tracking*. 2016. arXiv:1604.00219.
2. Maria V. Kulikova and Gennady Yu. Kulikov. *MATLAB-based general approach
   for square-root extended-unscented and fifth-degree cubature Kalman
   filtering methods*. 2023. arXiv:2312.02846.
3. Prob4D, commit `224d13dc9a93731ac5297b479eb1e121b3dbe659`,
   `_gauge_linearization_numerics.py` and `docs/gauge-linearization-closure.md`.
4. Prob4D `sim3.py`, Git blob
   `dc84a889c1aecb6b7d7c16ad83604861744b995d`, verified byte-for-byte in local
   validation. Its existing license and attribution are unchanged.
