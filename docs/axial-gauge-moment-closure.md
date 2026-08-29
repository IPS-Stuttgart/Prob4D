# Finite-angle closure for an unobserved axial gauge

Status: **experimental conditional kernel and designed analytic control**.
No real provider, protected target, BayesianPhysTwin execution, or Causal4D
execution is used. The stable exporter, local query gate, and exact-fallback
semantics are unchanged.

## Scientific question

The observable-subspace factor preserves a missing local gauge direction.
The query-conditioned gate then projects covariance through a local Jacobian.
Neither operation alone establishes that a query is constant over the complete
finite set of transformations that the measurement cannot distinguish.

For points on a known line, rotation about that line leaves their positions
unchanged. An off-axis probe has radial query

\[
q(\theta)=r\cos\theta,\qquad q'(0)=0.
\]

The derivative is zero although the query varies over the unobserved rotation.
A local scalar-query certificate can therefore miss this uncertainty. The
existing full-position gate does see its first-order tangential component;
this is not a claim that every query or every sigma-point scheme fails.

`prob4d.axial_gauge_moments` retains the finite axial orbit and propagates its
first two circular moments into **joint** point and linear-query moments.
`tests/test_axial_gauge_observability_integration.py` connects the counterexample
to the existing local gate without changing its documented local semantics.

## Exact conditional propagation

Let the unit axis be \(u\), the pivot be \(c\), and the reference point be \(p_i\).
Define

\[
z_i=c+uu^\top(p_i-c),\quad
 a_i=(I-uu^\top)(p_i-c),\quad b_i=u\times a_i.
\]

Rodrigues' formula gives the complete orbit

\[
p_i(\theta)=z_i+a_i\cos\theta+b_i\sin\theta.
\]

For \(h(\theta)=[\cos\theta,\sin\theta]^\top\), retain
\(m=E[h]\) and \(C=\operatorname{Cov}(h)\). These are equivalent to the first
two complex trigonometric moments, not a complete angular density. Stack
\(A_i=[a_i,b_i]\) into \(A\). Then

\[
E[p]=z+Am,\qquad \operatorname{Cov}(p)=ACA^\top=LL^\top,
\quad L=AC^{1/2}.
\]

The **shared axial contribution has rank at most two**, regardless of the
number of points. The implementation stores an `N x 3 x 2` factor rather than
a dense `3N x 3N` covariance. It never interprets the rows as independent noise.
For any declared linear query \(q=Wp\), its moments are exactly

\[
E[q]=W(z+Am),\qquad \operatorname{Cov}(q)=(WL)(WL)^\top.
\]

The componentwise full-orbit bounds are

\[
(Wz)_j\ \pm\ \sqrt{(Wa)_j^2+(Wb)_j^2}.
\]

These are bounds over the entire circle, not credible intervals, a joint box
whose corners are simultaneously attainable, or necessarily the support of the
supplied angular law. A component is constant over the circle precisely when
both its cosine and sine coefficients vanish. Checking only its derivative at
one angle is insufficient.

### Angular laws

`CircularMoments2` accepts a validated mean/covariance of `(cos, sin)`, provides
uniform and wrapped-normal constructors, and computes exact moments of a finite
weighted angular law. Finite laws may be multimodal.

For a centered wrapped normal with underlying variance \(v\),

\[
E[r\cos\theta]=r e^{-v/2},\qquad
\operatorname{Var}(r\cos\theta)=\tfrac12 r^2(1-e^{-v})^2.
\]

The implementation uses `expm1` to preserve the small radial term when \(v\)
is tiny. It does not replace the user's angular law with a fitted Gaussian or
infer an angular prior from evaluation outcomes.

## Limits that matter for integration

The axis, pivot, scale, and reference geometry are fixed or conditioned on.
This is **not** exact propagation of a general seven-dimensional Sim(3)
Gaussian. A small local information eigenvalue is not proof of an exact finite
symmetry: near-collinear geometry can contain weak but genuine twist evidence.
The caller must justify the orbit from its measurement model and retain that
weak evidence when it exists.

If other gauge variables \(\eta\) are uncertain or correlated with the angle,
use conditional moments and the law of total covariance:

\[
E[p]=E_\eta[E[p\mid\eta]],\quad
\operatorname{Cov}(p)=E_\eta[\operatorname{Cov}(p\mid\eta)]
 +\operatorname{Cov}_\eta(E[p\mid\eta]).
\]

A full implementation of that integration is future work; it is not provided
by this conditional kernel. Do not add this term to an existing gauge
covariance that already contains the same twist. Independent readout noise may
be added only under an explicit independence assumption. Nonlinear physical
queries require additional integration, not the linear `project` shortcut.

Moment exactness does not establish calibrated probabilities, Gaussian tails,
physical correctness, or safe admission. BayesianPhysTwin still owns complete
candidate construction and exact caller-owned fallback; Causal4D consumes only
a separately admitted physical belief.

## Reproduction

```bash
python -m pytest -q \
  tests/test_axial_gauge_moments.py \
  tests/test_axial_gauge_observability_integration.py
python -m prob4d.axial_gauge_moment_study \
  --protocol protocols/axial-gauge-moment-closure-v1.json \
  --output outputs/axial-gauge-moment-closure-v1/result.json
```

The output is created exclusively: an existing result is never overwritten.
It records the complete protocol, source-file hashes, runtime, all method rows,
and a canonical content identity. Final paper-facing result bytes belong in
`FlorianPfaff/BayesianPhysTwin-Paper`, not this code repository.

The designed control uses a 100 mm off-axis radius, a wrapped-normal angular
standard deviation sweep of 0.1, 0.5, 1, 2, and 3 radians, and independent 5 mm
readout noise. It retains first order, second order, two-point spherical-radial
cubature, Gauss-Hermite with 5 and 32 nodes, exact circular moments, and a
separate 128-node numerical reference. A 20 mm standard-deviation screen is
illustrative only and is not a production guard or a safety threshold.

At one radian, the expected radial mean and standard deviation are 60.653 mm
and 44.976 mm. First order reports 100 mm and 5 mm. Two symmetric cubature
points also report zero axial radial variance because their cosines coincide.
**32-node Gauss-Hermite agrees with the exact result** in this primary case.
It must remain in any paper comparison. The Gaussian expected score evaluates
moment approximations under the designed law; it is not empirical Gaussian
calibration. The shared-copy control is an algebraic dependence check, not 64
independent observations or physical trials.

## Contribution and prior art

Neither unobservable registration directions nor circular moments are new.
Localizability-aware registration is established, for example by X-ICP [1].
Directional estimation already provides analytic wrapped-normal trigonometric
moments, circular mixtures, and moment-preserving deterministic sampling [2].
Classical Rodrigues propagation and the law of total covariance are used here.

The candidate Prob4D contribution is narrower: preserve an unresolved finite
rotation in a learned-4D gauge, distinguish local derivative support from
finite-orbit query invariance, and retain its nonlinear **shared** uncertainty
through physical-query projection. Exact rank-two propagation gives a compact
implementation and an analytic check on numerical integration. It is not a
new general theorem about observability or a claim to outperform sufficiently
accurate quadrature.

General query-quotient lifting and physical-identifiability theory already
belong to the ecosystem's theory companion. This module does not reclaim those
theorems or require another standalone manuscript.

## What would make the paper materially stronger

The largest missing evidence remains one fresh real-provider result. Complete
the existing source-only PointWorld/Flat'n'Fold qualification in issue #333,
then separately freeze a held-out provider study. Do not reopen the terminal
MotionCrafter, official-Hub Deform360, v6.1, or CUT3R cohorts.

Only use this axial path where source geometry and the measurement model
justify it. Register queries and source/calibration thresholds before opening
held-out outcomes. Include on-axis and off-axis queries, shared and incorrectly
independent covariance ablations, local Gaussian and strong nonlinear
quadrature baselines, unchanged physical fallback, and the strongest simple
deterministic comparator. Score provider quality separately from downstream
physical-query value; use garment/object-session inference units, proper
scores, interval width, admitted-update harm, fallback frequency, and paired
query error. Report negatives and unsupported geometries without replacement.

PointWorld's action-conditioned point-flow model [3] is a candidate provider,
not evidence that it works on Flat'n'Fold or benefits BayesianPhysTwin.
Causal4D is optional downstream evidence and cannot rescue an upstream failure.
No current paper requires new hardware acquisition.

## References

1. Tuna et al., *X-ICP: Localizability-Aware LiDAR Registration for Robust
   Localization in Extreme Environments*, arXiv:2211.16335.
   https://arxiv.org/abs/2211.16335
2. Kurz et al., *Directional Statistics and Filtering Using libDirectional*,
   Journal of Statistical Software 89(4), 2019; arXiv:1712.09718.
   https://www.jstatsoft.org/article/view/v089i04
3. Huang et al., *PointWorld: Scaling 3D World Models for In-The-Wild Robotic
   Manipulation*, arXiv:2601.03782, 2026.
   https://arxiv.org/abs/2601.03782
