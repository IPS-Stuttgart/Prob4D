# Certified query identifiability under approximate finite orbits

Status: **experimental fail-closed certificate and designed robustness control**.

The recording-disjoint Tracking Cloth result shows that a local query gate can
accept a query that changes over an unresolved finite orbit. That primary result
uses a controlled exact SO(2) ambiguity. This module addresses the principal
robustness objection: in practice the orbit coefficients, or a numerical orbit
sample, may only be approximately known.

## Exact axial linear-query diameter

For a vector query whose axial orbit is

\[
q(\theta)=c+B
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix},
\qquad B\in\mathbb R^{d\times 2},
\]

the complete Euclidean orbit diameter is

\[
d(B)=\max_{\theta,\phi}\|q(\theta)-q(\phi)\|_2
    =2\sigma_{\max}(B).
\]

The equality follows because differences between antipodal circle points span
all vectors of norm two. `batch_axial_orbit_diameters` evaluates this expression
from the two-by-two Gram matrix, vectorized over any number of queries.

A local derivative at one representative is

\[
q'(\theta_0)=B[-\sin\theta_0,\cos\theta_0]^\top.
\]

It can vanish even when \(d(B)>0\). For example, the scalar query
\(q(\theta)=\cos\theta\) has \(q'(0)=0\) and diameter two.

## Certified coefficient-error interval

Let \(\widehat B\) be the estimated coefficient matrix and suppose a separately
justified bound satisfies

\[
\|B-\widehat B\|_2\leq\eta.
\]

Singular-value perturbation gives

\[
|\sigma_{\max}(B)-\sigma_{\max}(\widehat B)|\leq\eta,
\]

and therefore

\[
d(B)\in
\left[
\max(0,d(\widehat B)-2\eta),
 d(\widehat B)+2\eta
\right].
\]

For an invariance tolerance \(\tau\), the gate has three outcomes:

- **certified invariant:** the upper diameter bound is at most \(\tau\);
- **certified variant:** the lower diameter bound exceeds \(\tau\);
- **undetermined:** the interval crosses \(\tau\).

Only the first outcome admits an update. The undetermined case uses the caller's
registered fallback. Consequently a valid coefficient-error bound prevents a
truly variant query from being admitted, while larger estimation error manifests
as more fallback rather than hidden harmful acceptance.

The module does not infer \(\eta\). It may come from a geometric perturbation
bound, a source-calibrated bootstrap, a provider ensemble, or another method
whose validity is separately established. Supplying an optimistic bound voids
the certificate.

## Certified sampled nonlinear orbits

For a general periodic query \(q(\theta)\), suppose

\[
\|q(\theta)-q(\phi)\|_2\leq L|\theta-\phi|
\]

under circular angular distance. A uniform grid of \(K\) samples has angular
cover radius \(\pi/K\). If its finite-sample diameter is \(d_K\), then

\[
d_K\leq d(q)\leq d_K+2L\pi/K.
\]

`certify_uniformly_sampled_periodic_query` applies the same three-way decision
to this interval. This turns orbit discretization into an explicit conservatism
budget. Uncertified finite sampling can miss near-boundary variant queries;
the Lipschitz margin converts that error into fallback.

## Registered robustness study

The protocol in `protocols/robust-orbit-query-stress-v1.json` freezes:

- 20,000 balanced invariant/variant query orbits;
- three-dimensional vector queries;
- 50% stationary-representative cases;
- coefficient-error bounds from zero to 0.8 times the invariance tolerance;
- odd orbit grids of 7, 15, 31, and 63 samples;
- 10,000 near-boundary variant queries for the sampling test;
- a 4,096-query computational control.

It compares:

1. the certified bounded-error gate;
2. a naive gate using only the estimated diameter;
3. a nominal local-derivative gate;
4. Lipschitz-certified uniform orbit sampling;
5. naive uniform orbit sampling.

The registered scientific checks require zero harmful acceptance from both
certified gates, a nonzero failure rate for the naive approximations, and dense
SVD parity for the exact two-column diameter. Useful-query acceptance and
fallback are reported across the full error sweep; no claim is based on hiding
that conservatism.

Run:

```bash
python scripts/science/run_robust_orbit_query_stress.py \
  --protocol protocols/robust-orbit-query-stress-v1.json \
  --output outputs/robust-orbit-query-stress-v1/result.json
```

The output path is exclusive and includes the complete protocol, all rows,
runtime information, registered checks, and a canonical artifact identity.

## Relationship to real-data evidence

The public Tracking Cloth experiment remains the claim-bearing real-trajectory
mechanism result:

- 24 source recordings;
- 15 outcome-blind, header-supported held-out recordings;
- 1,803 controlled real-trajectory cases;
- approximately 65.1% harmful radial updates from the local gate;
- zero harmful accepted radial updates from the finite-orbit gate;
- invariant-query acceptance of one.

This robustness study does not create another independent real-data cohort. It
shows that the method has a mathematically controlled non-ideal regime: bounded
orbit-estimation and discretization errors enlarge fallback rather than silently
admitting an uncertified query.

## Claim boundary

This module certifies only the supplied mathematical error contract. It does not
establish that a learned provider's error bound is valid, infer the physical
symmetry from images, recover cloth state, or prove downstream BayesianPhysTwin
benefit. The CUT3R/DOT learned-provider confirmation remains separately reported
as support-negative. A future end-to-end experiment must estimate and validate
the orbit uncertainty on source groups before using this certificate on held-out
provider outcomes.
