# Query identifiability with an estimated equivalence orbit

The exact finite-orbit gate assumes that the unresolved orbit is known. A real
system generally has only an estimated orbit. This note gives a conditional
robustness certificate that separates the query geometry from the quality of
the orbit estimator.

Let `O` be the true equivalence orbit, `O_hat` an estimated orbit, and `q` an
`L`-Lipschitz physical query. If

\[
d_H(O,\widehat O)\leq \delta,
\]

then

\[
\left|\operatorname{diam}q(O)-
       \operatorname{diam}q(\widehat O)\right|
\leq 2L\delta.
\]

For arbitrary `x,y` in `O`, choose `x_hat,y_hat` in `O_hat` within `delta`.
The triangle inequality gives

\[
\|q(x)-q(y)\|
\leq \|q(\widehat x)-q(\widehat y)\|+2L\delta.
\]

Taking the supremum yields the upper bound. Reversing the roles of the two sets
gives the absolute-difference statement. The factor two is tight, for example
when an estimated singleton query set lies midway between the two endpoints of
the true query set.

A sufficient robust admission rule is therefore

\[
\operatorname{diam}q(\widehat O)+2L\delta\leq \varepsilon_q,
\]

where `epsilon_q` is the maximum allowed unresolved variation of the registered
query. The rule is agnostic to how the orbit is represented. It applies to
sampled multimodal sets, finite group orbits, and continuous Lie-group orbits.

`prob4d.orbit_identifiability.QueryOrbitCertificate` implements this arithmetic
and fails closed on invalid radii, Lipschitz constants, or tolerances.
`paired_orbit_error_bound` provides a convenient upper bound when true and
estimated orbits share a parameterization. It is deliberately named a bound,
not an exact Hausdorff calculation.

## What the certificate does not supply

The guarantee is conditional on a valid `delta`. A source-calibrated quantile,
bootstrap radius, or learned confidence estimate is not automatically a
deterministic Hausdorff bound. Its transfer must be evaluated separately, and a
probabilistic bound must be reported with its coverage level. Likewise, a local
Jacobian norm is not a global Lipschitz constant unless the relevant domain and
regularity assumptions justify it.

The Tracking Cloth robustness protocol perturbs the two axis-defining anchors,
selects one query-range threshold on source recordings, and evaluates the
resulting acceptance-versus-harm curve on the already opened v3 targets. That
experiment tests graceful degradation of the plug-in orbit estimate; it does
not turn the conditional theorem into an empirical coverage guarantee and is
not a second held-out confirmation.

## Paper-facing role

The exact-axis theorem and recording-disjoint v3 result establish the failure of
local query observability under a controlled hidden orbit. The estimated-axis
stress test addresses sensitivity to imperfect geometry. This certificate
states the principled extension needed when an orbit-set error bound is
available. Together they support a focused claim about query-level
identifiability and guarded fallback, not a claim of learned-provider competence
or complete physical-state recovery.
