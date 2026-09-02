# Continuous axial decision-risk calibration

This stacked experiment follows the retained negative result from the support-only
continuous-\(SO(2)\) study. That study calibrated a valid compatibility tube but
used an axis-center fallback that was much weaker than its kinematic candidate.
The result established that support validity alone does not make abstention
useful.

The v2 study changes the scientific question before opening its registered
distribution-shift target:

> Can an exact continuous-orbit advantage calculation be conformalized directly
> at the recording level so that accepted candidate updates are non-harmful
> relative to a competent transported zero-angular-velocity fallback?

## Exact base advantage

The candidate and fallback are points on the same current axial orbit:

- candidate: transported constant angular velocity;
- fallback: transported zero angular velocity.

For the source-calibrated continuous angle arc and Euclidean remainder tube, the
implementation computes the exact harmonic lower bound

\[
b_i =
\min_{\theta\in\mathcal A}
\left[
\|q(\theta)-f_i\|^2-\|q(\theta)-c_i\|^2
\right]
-
2\|c_i-f_i\|r_i .
\]

The final term is the exact Lipschitz envelope of the loss difference for an
additive Euclidean perturbation of radius \(r_i\).

## Signed recording-group calibration

Let the realized fallback-minus-candidate advantage be

\[
a_i=\|x_i-f_i\|^2-\|x_i-c_i\|^2 .
\]

For every risk-calibration recording \(g\), define the signed maximum deficit

\[
S_g=\max_{i\in g}(b_i-a_i).
\]

The score is deliberately signed. A negative conformal threshold may remove
systematic conservatism from the geometric base bound without weakening the
finite-sample statement. Given the one-sided split-conformal upper threshold
\(\tau\), the next exchangeable recording satisfies, with marginal probability
at least \(1-\alpha\),

\[
a_i\ge b_i-\tau
\quad\text{simultaneously for every case }i
\]

whenever its recording score is covered. The candidate is admitted only if

\[
b_i-\tau > 0.
\]

Therefore every accepted case in a covered recording is non-harmful relative to
the registered fallback.

This is a group-level simultaneous-case statement, not conditional or selective
coverage. It relies on exchangeability of complete recording scores; a
distribution-shift target can only test transfer empirically.

## Information order

The 32 source A2 shaking/twisting recordings are split before trajectory
evaluation, stratified by material:

- 8 recordings select one global anchor/probe triplet from causal prefixes;
- 12 disjoint recordings calibrate the continuous support;
- 12 further disjoint recordings calibrate signed decision-risk deficit.

Only marker headers from the collision cohort may be used before the calibration
seal, solely to identify labels common to all layouts. Collision trajectory
values are opened only after `calibration.json` has been written and hashed.

The 56 collision-family recordings were already opened by the earlier
finite-orbit mechanism experiment. Consequently, v2 is a related-access
distribution-shift diagnostic, not an independent prospective confirmation.
No target-side tuning or provider promotion is authorized.

## Required comparisons

The experiment reports:

1. transported zero-angular-velocity fallback;
2. transported constant-angular-velocity candidate;
3. support-only continuous certificate;
4. signed risk-calibrated continuous certificate;
5. complete-circle continuous certificate as a conservative control.

Primary outputs are recording risk-bound coverage, acceptance, harmful accepted
updates, exact fallback, and squared-error RMSE. Results are also stratified by
table collision, stick hitting, and self-collision.

A positive result requires at least two horizons to meet all frozen criteria,
including at least 90% recording-level risk-bound coverage, nonzero acceptance,
at most 1% harmful updates among accepted cases, strict harm reduction relative
to unconditional candidate use, RMSE no worse than the zero-velocity fallback
and support-only policy, exact fallback, and complete-circle rejection.

No result from this capsule establishes learned visual-provider competence,
unique physical-state recovery, deployment safety, or state of the art.
