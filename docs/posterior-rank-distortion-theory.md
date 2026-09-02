# Posterior rank--distortion frontier for a supplied correlated-noise factor

This note records the candidate theorem behind the research branch.  It extends
Prob4D's exact fixed-query shared-factor compression to nonzero distortion, but
only inside the same constrained factor-projection family.  It does **not** claim
that generalized-eigenvalue Bayesian reduction or goal-oriented posterior
approximation is new.

## Registered Gaussian interface

Let the registered query `q` and innovation `y` be jointly Gaussian with

\[
\operatorname{Cov}\begin{bmatrix}q\\y\end{bmatrix}
=
\begin{bmatrix}Q&C\\C^\top&S\end{bmatrix},
\qquad S=A+UU^\top,
\]

where `A` contains every covariance term that must remain unchanged and `U` is
the supplied shared-noise factor.  The full query posterior covariance is

\[
P=Q-CS^{-1}C^\top\succ0.
\]

For an orthogonal latent split `[V,N]`, retain `UV` and discard `UN`.  Define

\[
G=U^\top S^{-1}U,\qquad M=I-G\succ0,
\qquad R=U^\top S^{-1}C^\top.
\]

The exact covariance downdate identity gives

\[
P_V=P-L_N,
\qquad
L_N=R^\top N(N^\top M N)^{-1}N^\top R\succeq0.
\]

Thus every inexact factor reduction makes the registered query posterior more
confident than the full model.  This contraction must be audited; it is not a
free low-rank approximation.

## Theorem: globally optimal normalized trace contraction

Use the query-coordinate-invariant distortion

\[
D(N)=\operatorname{tr}(P^{-1}L_N).
\]

Set

\[
B=R P^{-1}R^\top\succeq0.
\]

Then

\[
D(N)=
\operatorname{tr}\!\left[
(N^\top M N)^{-1}(N^\top B N)
\right].
\]

For discarded dimension `d`, the generalized Ky Fan/Rayleigh--Ritz principle
therefore gives

\[
\boxed{
D_d^*=\sum_{i=1}^{d}\lambda_i
}
\]

where

\[
B x_i=\lambda_i Mx_i,
\qquad
0\le\lambda_1\le\cdots\le\lambda_r.
\]

An optimal discarded subspace is the span of the `d` generalized eigenvectors
with smallest eigenvalues.  Its Euclidean orthogonal complement gives the
rank-`k=r-d` retained projector `V`.  Choosing nested eigenspaces gives a nested
frontier.  If the rank boundary cuts through a repeated generalized-eigenvalue
block, the optimum value remains unique but the optimizing subspace is not.  The
implementation reports the boundary eigengap and an `optimal_subspace_unique`
flag; factor-covariance representatives should only be compared across latent
reparameterizations at strict boundaries.

This is a **global optimum only for this registered distortion and this
`U -> U V` factor family**.  It is not a claim of globally optimal Bayesian model
reduction.

### Exact theorem as the zero-distortion endpoint

Because

\[
B=(P^{-1/2}R^\top)^\top(P^{-1/2}R^\top),
\]

`rank(B)=rank(R)`.  Hence exactly `r-rank(R)` generalized eigenvalues are zero
in exact arithmetic.  Zero distortion is possible iff the discarded subspace
lies in `null(R^T)`, so the minimum retained zero-distortion rank is

\[
\operatorname{rank}(R)
=
\operatorname{rank}(U^\top S^{-1}C^\top),
\]

recovering the existing exact theorem.

### Why the old SVD ordering is not generally optimal

The current exact compressor orders directions by the Euclidean SVD of the
posterior-whitened response.  That is sufficient to expose its null space and
therefore gives the correct zero-distortion rank.  For nonzero distortion it
ignores the remainder metric `M=I-G`.  The generalized-eigen frontier uses that
metric and can strictly improve the same-rank posterior contraction.  A frozen
controlled counterexample is required before this distinction is promoted.

## Bayesian validity boundary

Define the normalized contraction

\[
W=P^{-1/2}L_NP^{-1/2}\succeq0.
\]

The reduced query posterior is positive definite iff

\[
\lambda_{\max}(W)<1.
\]

Since `lambda_max(W) <= trace(W)=D(N)`, the simple registered condition

\[
D(N)<1
\]

is sufficient for a valid reduced Gaussian query posterior.  Implementations
must also audit the sharper maximum eigenvalue directly.  Invalid rank points
may be retained as negative diagnostics but must not be deployed as Bayesian
posteriors.  The complete factor is always the terminal valid zero-distortion
fallback.

## Decision-regret corollary

The distortion can be converted into a task-level certificate rather than being
reported only as a covariance number.

Let every registered action loss be `L`-Lipschitz in the full-posterior
Mahalanobis query metric:

\[
|\ell(q,a)-\ell(q',a)|
\le
L\,\|P^{-1/2}(q-q')\|_2.
\]

Let `p_y` be the full query posterior and `p^V_y` the valid reduced posterior at
innovation `y`.  For any action, Kantorovich--Rubinstein plus
`W_1 <= W_2` gives

\[
|\mathbb E_{p_y}\ell-\mathbb E_{p^V_y}\ell|
\le L W_{2,P}(p_y,p^V_y),
\]

where the Wasserstein metric is evaluated after whitening by `P^{-1/2}`.
Therefore, if the reduced posterior chooses an action by Bayes risk, its regret
under the full posterior is bounded by

\[
\operatorname{Regret}(y)
\le 2L W_{2,P}(p_y,p^V_y).
\]

### Expected predictive regret

Under the full predictive innovation distribution, the normalized expected
posterior-mean shift satisfies

\[
\mathbb E_y\|P^{-1/2}(m^V_y-m_y)\|_2^2
\le
\frac{\gamma}{1-\gamma}D(N),
\qquad
\gamma=\lambda_{\max}(G)<1.
\]

For the covariance part,

\[
\sum_i(1-\sqrt{1-\mu_i})^2
\le\sum_i\mu_i=D(N),
\]

where `mu_i` are the eigenvalues of `W`.  Consequently

\[
\mathbb E_y W_{2,P}^2(p_y,p^V_y)
\le
\frac{D(N)}{1-\gamma}
\]

and

\[
\boxed{
\mathbb E_y[\operatorname{Regret}]
\le
2L\sqrt{\frac{D(N)}{1-\gamma}}.
}
\]

Because `gamma` is fixed by the full supplied factor, the generalized-eigen
subspace that globally minimizes `D(N)` at a given rank also globally minimizes
this registered regret upper bound at that rank.

### Uniform certificate inside a registered NIS gate

For innovations satisfying

\[
y^\top S^{-1}y\le\rho^2,
\]

the same operator bound yields

\[
W_{2,P}^2(p_y,p^V_y)
\le
D(N)\left(1+\rho^2\frac{\gamma}{1-\gamma}\right),
\]

hence

\[
\boxed{
\operatorname{Regret}(y)
\le
2L\sqrt{
D(N)\left(1+\rho^2\frac{\gamma}{1-\gamma}\right)
}.
}
\]

This gives a fail-closed design rule: for a preregistered action-loss Lipschitz
constant, NIS radius, and regret tolerance, choose the smallest retained rank
whose certified bound passes; otherwise use the full factor.

The corollary is a **certificate**, not a claim that the bound equals realized
regret or that the rank-`k` factor is globally optimal for arbitrary downstream
losses.

## Prior-art boundary

The broad mathematical ingredients are established:

- generalized Rayleigh/Ky Fan variational principles;
- goal-oriented Bayesian low-rank approximation and matrix-pencil constructions
  (including Spantini et al.);
- Wasserstein and Lipschitz risk perturbation bounds; and
- lossless sensor transformations, including correlated measurement-noise
  settings.

The candidate contribution is the constrained composition around a physical
estimation interface: retain every measurement row and every non-shared
covariance term, modify only a supplied correlated-noise factor, recover the
exact theorem at zero distortion, expose the complete rank frontier, audit
Bayesian validity, attach decision-regret certificates, and fail closed to the
original factor.  A specialist prior-art review is still required before any
first/unique wording.
