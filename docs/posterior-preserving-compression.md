# Posterior-preserving shared-noise compression

Status: **experimental local-Gaussian kernel**. This is not a provider/export
change, a replacement calibration, or authorization to open any data. It answers
a conditioning failure exposed by a controlled counterexample, not a retained
real-provider failure. Existing terminal protocols and provider gates are unchanged.

## The scientific distinction

The existing [query-preserving compressor](query-preserving-compression.md)
controls observation covariance projected through supplied Jacobians. It does
not promise to preserve a Bayesian posterior after that observation is combined
with a physical prior. The existing [query conditioner](query-posterior-conditioning.md)
computes this posterior, but does not construct a reduced shared-noise factor.

Here the compressed quantity is specifically the **latent shared observation-noise
factor**, while the complete observation vector is retained. The chosen subspace
depends on the complete physical prior and observation model. A low-energy mode
can matter critically because conditioning weights directions by precision.

This does not contradict the existing marginal compressor's documented guarantee.
In particular, using the actual full-model Bayesian gain as its projection and
preserving that projected covariance exactly already gives the same subspace
condition below. The new kernel constructs this prior-aware subspace directly,
establishes its minimal rank, and audits conditional rather than marginal errors.

## Fixed-model theorem

Let the centered joint Gaussian query and innovation have covariance blocks

$$
\operatorname{Cov}(q,y)=C,\qquad \operatorname{Cov}(q)=Q,\qquad
S=\operatorname{Cov}(y)=A+UU^\top,\quad A\succ0.
$$

For example, $A=D+HPH^\top$, $C=LPH^\top$ and $Q=LPL^\top$ for
$q=Lx$, $x\sim\mathcal N(m,P)$, and observation noise $D+UU^\top$.
All blocks, means, row order and linearizations are held fixed.

Choose an orthonormal latent projection $V$ and replace $U$ by $UV$:

$$
S_V=A+UVV^\top U^\top.
$$

**Proposition.** The query posterior mean for every innovation and the query
posterior covariance are unchanged if and only if

$$
\operatorname{range}(U^\top S^{-1}C^\top)\subseteq\operatorname{range}(V).
$$

Within this family of orthogonal latent projections, the minimum retained rank is

$$
k_* = \operatorname{rank}(U^\top S^{-1}C^\top)\leq\dim(q).
$$

This is a minimum within the stated factor-projection family, not among all
possible representations of a posterior. The count does not depend directly on
the number of points, windows, or gauge variables. Stacking several queries into
one joint query preserves their joint covariance, not just each marginal.

### Proof and error identity

Let $N$ span the orthogonal complement of $V$, $W=UN$, and
$T=I-W^\top S^{-1}W$. Since $A\succ0$, $S_V\succ0$ and $T\succ0$.
The inverse downdate identity gives

$$
S_V^{-1}=S^{-1}+S^{-1}WT^{-1}W^\top S^{-1}.
$$

Set $E=CS^{-1}W$, $K=CS^{-1}$ and $K_V=CS_V^{-1}$. Then

$$
K_V-K=ET^{-1}W^\top S^{-1},
$$

$$
P_{q\mid y}-P_{q\mid y,V}=ET^{-1}E^\top\succeq0.
$$

The covariance difference is zero exactly when $E=0$, which is equivalent to
the range condition and also makes the gain difference zero. Any containing
subspace must have at least $k_*$ columns, and a basis of the displayed range
attains this bound. This proves necessity, sufficiency and minimality.

The same identity quantifies numerical truncation. For innovations drawn from
the full model, the mean-error covariance is

$$
\mathbb E[(K_V-K)y y^\top(K_V-K)^\top]
=ET^{-1}(W^\top S^{-1}W)T^{-1}E^\top.
$$

Thus a small discarded singular value is not enough: the downdate core may
amplify it. The implementation checks gain error, posterior-relative covariance
loss and expected posterior-normalized mean error before admitting a reduction.

## Counterexample to marginal/trace-only reasoning

Use a scalar $x\sim\mathcal N(0,1)$, $H=(1,1,0)^\top$,
$D=\operatorname{diag}(1,10^{-4},1)$ and

$$
U=\begin{bmatrix}1000&0\\0&1\\0&0\end{bmatrix}.
$$

Keeping only the first column retains $10^6/(10^6+1)$ of the shared covariance
trace and exactly preserves the marginal projection $J=(1,0,0)$. Nevertheless,
the scalar posterior variance falls from approximately $0.500025$ to
$0.00009999$: about 5,001-fold overconfidence relative to the full model.
The posterior-aware one-column projection preserves the full posterior instead.
The selected $J$ is not the full Bayesian gain $K$; conflating them is the failure.

## Experimental API and numerical behavior

```python
from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)

result = compress_shared_factor_for_posterior(
    shared_factor_m,                    # (N, 3, R)
    prior_query_covariance=query_prior, # (Q, Q)
    query_observation_cross_covariance=query_cross, # (Q, N, 3) or (Q, 3N)
    innovation_operator=full_innovation_operator,
    maximum_rank=query_dimension,
)
```

The operator must solve the **full** innovation covariance, not conditional noise
alone. Prob4D's existing cached observation and low-rank innovation operators
satisfy the solve protocol. A batched solve obtains responses to $U$ and $C^\top$.
The SVD is query-posterior-whitened, avoiding dependence of the exact selected
subspace on nonsingular query units or basis changes. Only numerical tolerances
can affect near-rank-deficient decisions. Singular values at a truncation boundary
are followed by the actual conditional-error audit, not accepted blindly.

The kernel requires strictly positive-definite query prior and posterior
covariance for relative auditing; dependent or deterministic query coordinates
must first be reduced by the caller. It validates finite real inputs, covariance
symmetry, posterior consistency, and the positive-definite innovation remainder.
The caller must supply a valid SPD solver. It does not verify an arbitrary
solver's complete unseen covariance or provenance.

An insufficient rank cap returns an independent read-only copy of the **exact
original factor** and an identity latent projection. It never substitutes an
underqualified truncation or adds a ridge. Invalid inputs raise an exception.
No fallback result authorizes a physical update: BayesianPhysTwin still owns
complete-belief selection and its exact physical fallback.

## Cost and the necessary simple baseline

Construction needs the original factorization, responses to $R+Q$ right-hand
sides, and an $R$-dimensional Gram/SVD audit. This implementation is **not** a
streaming, subquadratic-rank construction and does not lower the initial peak
memory. A reduced factor can lower a subsequently stored shared-factor payload
from $3NR$ to $3Nk$ scalars. This is not a measured total-system speedup.

For an immutable query and model, caching $(K,P_{q\mid y})$ is simpler and may
be smaller/faster still. The controlled study therefore includes a
`cached-full-query-message` reference. A paper must demonstrate why a consumer
needs a factor-level representation, or report the direct-message baseline as
preferable. Do not compare reduced factors against unnecessarily repeated full
inference while omitting this cache baseline.

## What is NOT preserved

Observation log determinants, full joint observation likelihoods, other queries,
and later recursive updates generally change. Do not use the compressed factor
for provider competence scores, model comparison, mixture weights or robust
reweighting. Use the full model for these quantities. Reconstruct the projection
when the query, physical prior, cross covariance, linearization, observation row
set, or robust weights change. A mixture of locally exact Gaussian queries need
not have preserved mixture probabilities or a preserved global posterior.

An existing `ObservationBeliefV1` or claim-bearing factor artifact is never
modified or relabeled by this module. It is outside `prob4d.api.v2` and returns
only an experimental query-bound result. Causal4D may consume only an admitted
BayesianPhysTwin belief, not this reduced observation as a new general provider.

## Reproduction and paper boundary

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_posterior_preserving_compression.py \
  tests/test_posterior_compression_operator_integration.py

OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python \
  scripts/science/run_posterior_compression_study.py \
  --protocol protocols/posterior-compression-controlled-v1.json \
  --source-revision "$(git rev-parse HEAD)" \
  --output-dir outputs/posterior-compression-controlled-v1
```

The study has twelve fixed geometry/window designs and 4,096 complete independent
Gaussian draws per design. Shared errors follow a correlated seven-coordinate
linearized Sim(3) chain. It compares full covariance, posterior-preserving rank
three, equal-rank covariance PCA, conditional-only noise, and a cached full query
message. All methods use paired draws. No real model, dataset, calibration cohort
or sealed target is accessed. Generated paper evidence belongs in the paper repo.

The mathematical proposition and controlled study support a narrower
conditioning-exact representation claim. They do not establish useful provider
means, real calibration, physical benefit, Causal4D benefit, safety or state of the
art. They cannot rescue the terminal MotionCrafter/Deform360 experiments.

Goal-oriented Bayesian reduction is established prior work: Spantini, Cui,
Willcox, Tenorio and Marzouk, *Goal-Oriented Optimal Approximations of Bayesian
Linear Inverse Problems*, SIAM J. Scientific Computing 39(5), S167--S196 (2017),
DOI: `10.1137/16M1082123`, arXiv: `1607.01881`. That work studies optimal low-rank
query-posterior covariance updates and posterior-mean maps. The present candidate
contribution is the explicit shared-noise-factor projection, exact range/rank
condition, error identity and Prob4D integration. No claim of first-ever
goal-oriented inference or exhaustive novelty clearance is made.
