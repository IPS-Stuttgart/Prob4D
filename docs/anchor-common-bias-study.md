# Independent-anchor common-bias study

The cross-provider corroboration guard deliberately does not claim that agreement
between two visual providers establishes absolute geometric correctness. Two
providers that consume the same video can share camera, training-set, scale,
occlusion, and reconstruction biases. Their difference cancels a coherent error
that moves both predictions in the same direction.

This study quantifies the complementary role of an **independent metric anchor**.
It is a calibration/target-separated controlled mechanism experiment and an
acquisition-design calculation. It is not evidence that any real provider or
anchor is competent.

## Statistical model

For one matched 3-D row, the two provider errors are

\[
e_1, e_2 \sim \mathcal N(0, \Sigma_p),
\qquad
\operatorname{Cov}(e_1,e_2)=\rho\Sigma_p,
\]

and the independent anchor error is

\[
e_a \sim \mathcal N(0, \Sigma_a).
\]

The differential provider residual is

\[
d=e_1-e_2.
\]

It detects provider-specific corruption, but a shared bias cancels exactly. The
anchor common-mode residual is

\[
c=\frac{(e_1+b_1)+(e_2+b_2)}{2}-e_a.
\]

For isotropic registered covariances, its variance is

\[
\operatorname{Var}(c)
=\frac{1+\rho}{2}\sigma_p^2+\sigma_a^2.
\]

The study normalizes both residuals by their complete covariance, computes one
row score from normalized Mahalanobis energy, and reduces each complete simulated
object/session to the frozen higher row quantile. It then fits a split-conformal
upper threshold across clean calibration groups. Frames and rows are never used
as exchangeable calibration units.

## Frozen design

The version-1 protocol uses:

- 800 clean calibration groups;
- 1,000 disjoint target groups per arm;
- 256 candidate rows per group;
- three coordinates per row;
- provider cross-correlation 0.75;
- nominal miscoverage 5%;
- the 95th row-score quantile;
- provider-specific corruption of 1.0 provider sigma;
- shared coherent bias of 1.5 provider sigma on 25% of rows; and
- anchor precision/support grids declared in
  `protocols/anchor-common-bias-study-v1.json`.

The reference acquisition design uses an anchor standard deviation of 0.5 times
the provider standard deviation and random anchor support on 20% of rows. This is
a controlled design point, not a recommendation for a real sensor until its
covariance, independence, registration, and support process have been calibrated
on complete source objects or sessions.

## Registered gates

The controlled study passes only when all of these predeclared checks hold:

1. differential clean false rejection is at most 8%;
2. provider-specific corruption detection is at least 95%;
3. differential rejection of shared common bias is at most 10%, preserving the
   required limitation control;
4. reference-anchor clean false rejection is at most 8%;
5. reference-anchor rejection of shared visual bias is at least 90%; and
6. reference-anchor rejection of an anchor-drift inconsistency is at least 95%.

The last endpoint is an inconsistency test, not fault attribution. A rejection
cannot determine whether the providers, the anchor, their registration, or their
covariance model is wrong.

## Run

```bash
prob4d diagnostic anchor-common-bias \
  --source-revision "$(git rev-parse HEAD)" \
  --output-json outputs/anchor-common-bias/report.json \
  --output-markdown outputs/anchor-common-bias/report.md
```

The report is content-addressed and binds the exact 40-character source revision.
The workflow runs the full frozen design, verifies the report identity, checks all
registered gates, and uploads both JSON and Markdown evidence.

## Interpretation

A positive result supports the following narrow design conclusion: under the
registered Gaussian dependence model, a sparse independent anchor can recover
power against a coherent visual bias that provider disagreement alone must miss.
The power grid exposes how that result changes with anchor precision and support.

It does **not** establish:

- real MotionCrafter, VGGT, CUT3R, V-DPM, or other provider competence;
- independence or calibration of a real RGB-D, tactile, LiDAR, kinematic, or
  manually registered anchor;
- correct material identity or registration;
- target-object coverage;
- BayesianPhysTwin physical-query improvement;
- Causal4D intervention benefit; or
- deployment safety.

A real promotion gate must freeze provider and anchor revisions, calibrate their
joint covariance and support on independent source groups, use fresh physical
objects or sessions, retain exact fallback, and separately pass the downstream
BayesianPhysTwin regret guard.
