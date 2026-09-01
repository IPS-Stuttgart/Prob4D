# Factorized marginal-preserving shared dependence

Status: **experimental covariance kernel and designed algebra control**. This
module does not fit a dependence strength and does not change any frozen DOT
confirmation protocol.

## Motivation

A learned 4-D provider may exhibit both local uncertainty and a shared restart,
gauge, or representation effect. Treating all outputs as independent discards
that dependence. Treating all uncertainty as one perfectly shared latent can be
far too restrictive. The source-only DOT study therefore considered the family

\[
\Sigma_\alpha=(1-\alpha)\Sigma_{\mathrm{local}}+
\alpha\Sigma_{\mathrm{shared}},\qquad 0\leq\alpha\leq 1,
\]

while holding the predictive mean and every marginal variance fixed. The
source-frozen DOT candidate is \(\alpha=0.85\); its independent confirmation is
separate from this implementation.

`BlockSharedGaussianCovariance` gives this family a direct latent
interpretation and avoids materializing a dense joint covariance.

## Model

Let output group \(i\) have dimension \(D\), local marginal covariance
\(M_i\), and rows \(F_i\in\mathbb R^{D\times R}\) of a shared factor. The
constructor requires

\[
F_iF_i^\top=M_i
\]

for every group. Define

\[
\varepsilon_i\sim\mathcal N(0,M_i),\qquad
z\sim\mathcal N(0,I_R),
\]

with all \(\varepsilon_i\) independent of each other and of \(z\). Then

\[
x_i=\sqrt{1-\alpha}\,\varepsilon_i+
\sqrt{\alpha}\,F_i z
\]

has joint covariance

\[
\operatorname{Cov}(x)=
(1-\alpha)\operatorname{blockdiag}(M_1,\ldots,M_N)
+\alpha FF^\top.
\]

For every \(\alpha\),

\[
\operatorname{Cov}(x_i)=M_i.
\]

Only cross-group covariance changes:

\[
\operatorname{Cov}(x_i,x_j)=\alpha F_iF_j^\top,
\qquad i\ne j.
\]

This is stronger than preserving scalar diagonal entries: complete vector-valued
marginal covariance blocks are preserved exactly.

## Stable solve and log determinant

For \(\alpha<1\), let

\[
B=(1-\alpha)\operatorname{blockdiag}(M_i),\qquad
U=\sqrt{\alpha}F.
\]

The implementation evaluates

\[
(B+UU^\top)^{-1}
=B^{-1}-B^{-1}U(I+U^\top B^{-1}U)^{-1}U^\top B^{-1}
\]

and

\[
\log\det(B+UU^\top)
=\log\det B+\log\det(I+U^\top B^{-1}U)
\]

using independent \(D\times D\) block solves and one \(R\times R\) core.
This supports exact Gaussian quadratic forms and normalized joint NLL without a
large dense matrix.

The fully shared endpoint \(\alpha=1\) remains valid for covariance moments and
sampling, but may be singular whenever \(R<ND\). Precision operations reject
that case rather than silently adding jitter or changing the registered model.

## Complexity and storage

The retained arrays require

\[
O\!\left(ND(D+R)\right)
\]

storage. One solve requires block-local work plus a rank-sized core, rather than
a dense \(O((ND)^3)\) factorization.

For the paper-scale example \(N=2048\), \(D=3\), and \(R=7\):

| Representation | Float64 storage |
|---|---:|
| Dense \(6144\times6144\) covariance | 301,989,888 bytes |
| Marginal blocks plus shared factor | 491,520 bytes |
| **Reduction** | **614.4×** |

This is an algebraic storage comparison. It is not an end-to-end runtime claim;
provider execution and query construction may dominate total cost.

## Reproduction

```bash
python -m pytest -q tests/test_factorized_dependence.py
python scripts/science/run_factorized_shared_dependence_control.py \
  --protocol protocols/factorized-shared-dependence-control-v1.json \
  --output outputs/factorized-shared-dependence-control-v1/result.json
```

The control compares factorized solves, log determinants, quadratic forms, and
Gaussian NLL with an explicitly materialized dense covariance. It also checks
Monte Carlo moments and reports the paper-scale storage calculation. The output
is created exclusively and includes the protocol identity, source hash, runtime,
and a content identity.

## Integration boundary

The caller must supply a scientifically justified shared factor and dependence
strength. The factor may come from a finite gauge orbit, a shared latent model,
or another covariance construction, but those semantics are not inferred here.
A source-fitted \(\alpha\) must be frozen before held-out evaluation.

Do not use this class to:

- change a predictive mean;
- retrofit dependence using held-out outcomes;
- claim that provider restarts are independent observations;
- replace a singular likelihood with fabricated information;
- infer a safety decision from NLL alone.

The current DOT evidence specifically rejected adding a predictable restart
residual as a deterministic mean correction. That negative control motivates a
covariance interpretation, but the R04-R10 confirmation is still required for a
held-out empirical claim.
