# First-order Sim(3) linearization adequacy

`prob4d.diagnostics.sim3_linearization` is a source-only diagnostic for deciding
whether a frozen local Gaussian `Sim(3)` approximation is sufficiently accurate
for downstream uncertainty propagation.

It compares the first-order Jacobian approximation with nonlinear Monte Carlo under
the same local perturbation model. The Monte Carlo path is streamed in bounded
batches and retains only first and second moments.

## What is checked

For every supplied point, the certificate reports:

- relative covariance-trace error;
- relative covariance Frobenius error;
- nonlinear mean shift in empirical RMS standard deviations; and
- principal-axis rotation when the nonlinear covariance is sufficiently anisotropic.

An optional query projection can be supplied to evaluate the same trace,
Frobenius, and mean-shift diagnostics in a frozen downstream physical-query space.

A caller-supplied analytic Jacobian is not trusted blindly. The module validates it
against an independent central-difference Jacobian before using it.

## Perturbation convention

The convenience `assess_sim3_linearization` API supports both local conventions:

- `left`: `delta.compose(mean_transform)`;
- `right`: `mean_transform.compose(delta)`.

The seven perturbation coordinates are the existing Prob4D `Sim3` coordinates:
log-scale, three rotation-vector coordinates, and translation. Their block order is
explicit and may be any permutation of `scale`, `rotation`, and `translation`.

## CLI

Create an NPZ containing:

- `mean_transform`: canonical Prob4D `Sim3.as_vector()` array with shape `(7,)`;
- `covariance`: local perturbation covariance with shape `(7, 7)`;
- `points`: finite point array with shape `(N, 3)`;
- optionally `jacobian`: array with shape `(N, 3, 7)`; and
- optionally `query_projection`: matrix with shape `(Q, 3N)`.

Then run:

```bash
python -m prob4d.diagnostics.sim3_linearization source_case.npz \
  --output certificate.json \
  --fail-on-inadequate
```

`--parameter-order` accepts a comma-separated permutation such as
`rotation,translation,scale`. The covariance and optional supplied Jacobian are
interpreted in that declared order.

The command returns status `2` for a valid but inadequate certificate when
`--fail-on-inadequate` is enabled. Malformed inputs fail closed.

## Scientific boundary

The certificate is diagnostic evidence, not a covariance repair. A failure does not
authorize target-side retuning or silent covariance inflation. The intended response
is to retain the explicit shared gauge latent, use an already-declared exact fallback,
or revise the source-side approximation under a new frozen protocol.

Passing this diagnostic does not establish real-provider competence, target transfer,
BayesianPhysTwin benefit, Causal4D intervention benefit, deployment safety, or state
of the art.
