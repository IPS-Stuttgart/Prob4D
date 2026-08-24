# Prob4D scientific kernel

This page gives the shortest technical description of the method and its
cross-repository role. Detailed contracts, numerical safeguards, and experiment
protocols remain in the linked documents.

## Scope

Prob4D is an uncertain 4-D observation feeder. It fuses overlapping causal
prediction windows, preserves their shared geometric uncertainty, and exports a
portable observation for a downstream Bayesian physical estimator. Prob4D does
not decide whether a physical-state update is accepted.

The one-way ownership boundary is

```text
4-D provider
    -> Prob4D observation and covariance
    -> BayesianPhysTwin guarded belief or exact physical fallback
    -> Causal4D abduction, intervention, and prediction
```

## Window gauges

For window \(k\), let \(p_{ki}\in\mathbb{R}^3\) be a local point and let

\[
g_k =
\begin{bmatrix}
\rho_k & \phi_k^\top & t_k^\top
\end{bmatrix}^\top
\in\mathbb{R}^7
\]

denote its `Sim(3)` gauge: log scale \(\rho_k\), rotation vector \(\phi_k\), and
translation \(t_k\). Its world-frame point is

\[
y_{ki} = \exp(\rho_k) R(\phi_k) p_{ki} + t_k.
\]

The causal gauge tree starts from an uncertain metric anchor and uncertain
relative gauges. Linearizing each composition gives

\[
\delta g_k \approx A_k\,\delta g_{\operatorname{parent}(k)}
                  + B_k\,\delta r_k.
\]

Stacking the anchor and relative-gauge errors into \(\eta\) yields

\[
\delta G = L\eta,\qquad
\Sigma_G = L\Sigma_\eta L^\top.
\]

The off-diagonal blocks of \(\Sigma_G\) are part of the model: windows share the
metric anchor and earlier relative-gauge errors. Provider-v2 uses analytic
composition Jacobians and fails closed near the nondifferentiable
\(\operatorname{SO}(3)\) logarithm branch at angle \(\pi\).

## Observation covariance

After stacking all admitted rows, Prob4D uses the linearized observation model

\[
y = \mu + J_G\delta G + \varepsilon,\qquad
\delta G\sim\mathcal N(0,\Sigma_G),\quad
\varepsilon\sim\mathcal N(0,R),
\]

with

\[
\Sigma_y = R + J_G\Sigma_GJ_G^\top.
\]

Here \(R\) is block-diagonal conditional point uncertainty. It must exclude the
gauge contribution. Adding both a gauge-marginal point covariance and an
explicit gauge prior would double count uncertainty.

`ObservationFactorBundle` retains \(R\), \(J_G\), and the complete joint
\(\Sigma_G\). This is the preferred representation when BayesianPhysTwin keeps
gauge errors as explicit nuisance variables. A collapsed observation belief is
appropriate only when the declared covariance semantics preserve the required
cross-row dependence.

## Proper scores and physical queries

For a residual \(r\in\mathbb{R}^{3M}\), the joint Gaussian negative log
likelihood is

\[
\frac{1}{2}\left(
r^\top\Sigma_y^{-1}r +
\log\det\Sigma_y +
3M\log(2\pi)
\right).
\]

Rowwise marginal scores are generally wrong because they omit cross-row
covariance. Prob4D therefore supplies structured inverse actions, log
determinants, and proper scores without materializing the full
\(3M\times3M\) covariance.

For a downstream linear query with Jacobian \(A\), the propagated covariance is

\[
\Sigma_q = A\Sigma_yA^\top
         = ARA^\top + (AJ_G)\Sigma_G(AJ_G)^\top.
\]

The decomposition separates conditional point uncertainty from shared gauge
uncertainty while holding the observation mean and physical query fixed.

## Reliability remains separate

Prob4D does not collapse distinct evidence quantities into one confidence
number:

- association probability supports the named material point or entity;
- prior reliability is source-side evidence that a row is nominal;
- nominal-component probability belongs to a correlation group; and
- composite weight limits dependent groups from contributing repeated
  information.

BayesianPhysTwin owns the robust likelihood, physical identifiability test,
prospective update guard, and exact complete-belief fallback.

## Minimal factor-side use

```python
from prob4d.api.v2 import (
    build_observation_gaussian_operator,
    load_claim_bearing_observation_factor_bundle,
    project_observation_covariance,
    stack_sparse_observation_factors,
)

validated = load_claim_bearing_observation_factor_bundle(
    "outputs/case-a/factors.claim.json"
)
stacked = stack_sparse_observation_factors(validated.bundle)

operator = build_observation_gaussian_operator(stacked)
joint_nll = operator.gaussian_nll(residual_xyz_m)

query_covariance = project_observation_covariance(
    stacked,
    physical_query_jacobian,
).marginal_covariance
```

The residual and physical-query Jacobian belong to the downstream estimator.
Loading and algebraic consistency are not provider-competence evidence.

## Scientific decision order

Independent units are complete physical objects or acquisition sessions, never
frames, points, tracks, views, or pixels. A real provider advances only through

```text
support
  -> source means
  -> identity and reliability
  -> gauge and dependence
  -> linearization closure
  -> conditional point covariance
  -> physical-query relevance
  -> one frozen target evaluation
```

A failure stops or redirects that provider version. Downstream performance
cannot rescue an upstream negative. Richer point covariance is authorized only
when useful means and identities have passed and the remaining source failure is
explicitly localized to conditional point covariance.

The current highest-priority execution is the
[CUT3R source qualification](cut3r-qualification-runbook.md), coordinated with
[issue #49](https://github.com/IPS-Stuttgart/Prob4D/issues/49).

## Detailed references

- [Architecture and repository boundary](architecture.md)
- [Provider API v2](provider-v2.md)
- [Unfused observation-factor bundle](observation-factor-bundle.md)
- [Structured observation covariance queries](observation-covariance-queries.md)
- [Executable joint-dependence invariants](joint-dependence-invariants.md)
- [Analytic gauge propagation](analytic-gauge-propagation.md)
- [Provider readiness localization](provider-readiness-localization.md)
