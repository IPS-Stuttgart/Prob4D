# Continuous axial-symmetry certificates with calibrated support

Status: **experimental mechanism and real-trajectory evaluation**.

This extension separates two questions that the finite-orbit work deliberately
left open:

1. Can the registered axial ambiguity be optimized over the complete continuous
   group rather than a finite angle roster?
2. Can the size of the admitted ambiguity support be calibrated from real
   source trajectories rather than supplied as a controlled hidden angle?

The answer is conditional. The geometry below is exact for a declared
`SO(2)` axial orbit. The statistical statement is split-conformal marginal
coverage for an exchangeable recording-level score. It is not conditional
coverage, selective risk control, provider competence, or deployment safety.

## Exact continuous vector-query diameter

For a vector first-harmonic query

\[
q(\theta)=c+a\cos\theta+b\sin\theta ,
\]

let \(M=[a\ b]\). For any linear weighting matrix \(W\),

\[
\operatorname{diam}_{W}\bigl(q(SO(2))\bigr)
=
\sup_{\theta,\phi}
\|W(q(\theta)-q(\phi))\|_2
=
2\sigma_{\max}(WM).
\]

The upper bound follows because the difference between two unit-circle points
has norm at most two. Equality is attained by antipodal points in the right
singular direction of \(WM\). For a rotating 3-D point this gives twice the
largest weighted orbit radius without discretizing the angle.

An additive query-space ball of radius \(r_q\) enlarges the diameter by
\(2\|W\|_2r_q\). The implementation rejects when the declared model scope is
not admitted or when the exact diameter exceeds the registered tolerance.

## Group-conformal axial tube

A causal predictor supplies a representative point \(\widehat p\) on an axial
orbit. For real point \(p\), let \(\widehat\theta\) be its closest continuous
orbit angle and let \(d\) be the remaining Euclidean distance to that orbit.
With radial scale \(r>0\) and a predeclared angular scale \(\theta_0>0\), use

\[
s(p)=\max\left\{
\frac{|\widehat\theta|}{\theta_0},
\frac{d}{r}
\right\}.
\]

One complete recording contributes one score, here the conservative empirical
90th percentile of its case scores. Given \(n\) exchangeable calibration-group
scores, the finite threshold is their

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil
\]

order statistic. When \(k>n\), the implementation returns no finite bound and
the caller must reject. Otherwise, the next exchangeable recording score is
below the threshold with marginal probability at least \(1-\alpha\).

For calibrated threshold \(\widehat q\), a case receives the continuous support

\[
|\theta|\leq \min(\pi,\widehat q\theta_0),
\qquad
\|e\|_2\leq \widehat q r .
\]

The guarantee concerns the registered recording score. It does not imply
conditional coverage for a selected subgroup, selective coverage after
admission, or future support completeness under distribution shift.

## Exact query and action bounds

A scalar affine query remains a first harmonic over the angle arc. Its exact
continuous extrema are computed analytically and enlarged by \(L\widehat q r\)
for Euclidean Lipschitz constant \(L\).

For an orbiting point \(x(\theta)\) and fixed target \(y\),
\(\|x(\theta)-y\|_2^2\) is also a first harmonic. Comparing candidate point
\(c\) and fallback point \(f\) under the same latent angle gives

\[
D(x)=\|x-f\|_2^2-\|x-c\|_2^2 .
\]

If \(x=x(\theta)+e\), the orbit-model error in this advantage is exactly linear
in \(e\) and satisfies

\[
|D(x)-D(x(\theta))|
\leq 2\|c-f\|_2\|e\|_2.
\]

This supplies a deterministic error envelope for the calibrated tube. The
candidate is admitted only when the minimum continuous-arc advantage minus this
envelope is strictly positive. Rejection returns the exact axis-center fallback.

## Tracking Cloth study

The workflow uses the complete public Tracking Cloth Deformation collection on
the `gpuserver4090` runner. It admits the 80 cloth-only recordings identified by
their explicit Motive marker layouts:

- A2: 20 markers;
- A3: 12 markers.

Rod/stick-augmented layouts remain visible as support exclusions. Every accepted
recording selects two axis markers and one off-axis probe from an initial causal
prefix. At a later case the current axis is observed and the probe is hidden. A
transported constant-angular-velocity construction based on two earlier
horizon-spaced frames supplies the representative.

Five recording-disjoint folds are evaluated separately by cloth size and at
horizons 3, 6, 12, and 24 frames. A held-fold trajectory never contributes to
its calibration threshold. Complete recordings are the empirical units; frames,
coordinates, and cases remain nested.

The registered comparisons are:

- local representative plug-in;
- continuous source-calibrated `SO(2)` tube certificate;
- complete full-circle continuous certificate;
- finite 8/16/32-angle query-interval approximations;
- exact axis-center fallback.

The experiment reports support and query-interval coverage, accepted-update
harm, acceptance, fallback identity, RMSE, interval width, and the gap between
finite angle grids and analytic continuous extrema. A positive horizon requires
nontrivial acceptance, lower harmful acceptance than the local rule, exact
fallback, continuous-support coverage, and full-circle rejection.

## Claim boundary

The trajectories and prediction residuals are real. The missing-probe condition
is simulated, and the representative is a fixed kinematic predictor rather than
a learned visual provider. The result cannot establish calibrated conditional
uncertainty, selective risk control, unique physical-state recovery,
BayesianPhysTwin or Causal4D benefit, deployment safety, or state of the art.
