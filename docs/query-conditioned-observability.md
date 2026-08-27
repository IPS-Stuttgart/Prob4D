# Query-conditioned observability for partial 4D gauge factors

Status: **experimental scientific kernel and deterministic mechanism evidence**.
This path is not yet admitted to the claim-bearing provider-v2 export.

## Why this is needed

A rank-deficient overlap can be informative without identifying every component
of its relative Sim(3) gauge. The existing observable-subspace factor retains
that valid information and leaves the missing directions to a complete prior.
The remaining deployment question is query-specific:

> Does the retained information actually constrain the physical quantity that a
> downstream Bayesian twin is about to use?

A centerline overlap, for example, does not observe twist about the line. It can
still locate points on that line accurately. The same factor may be unsafe for
an off-axis contact point, gripper pose, or counterfactual probe whose position
depends strongly on the missing twist. Pooled reconstruction error does not
express this distinction.

`prob4d.query_observability` projects one
`ObservableGaugeFactor` through a declared downstream query. It reports direct
geometric support separately from covariance reduction mediated by the prior,
then exposes a source-frozen gate. The caller retains ownership of exact physical
fallback.

## Formulation

Let the local gauge coordinate in the factor's centroid-normalized chart be

\[
\boldsymbol\zeta =
[\delta\ell,\delta\boldsymbol\phi^\top,
 \delta\boldsymbol\tau^\top]^\top \in \mathbb R^7 .
\]

An observable-subspace factor has information

\[
\boldsymbol\Lambda_{\mathrm{obs}}
=
\boldsymbol U_r \boldsymbol\Lambda_r \boldsymbol U_r^\top ,
\]

where \(\boldsymbol U_r\) spans the retained observable directions and
\(\boldsymbol N\) spans its orthogonal nullspace. For a downstream query
\(\boldsymbol q(\boldsymbol\zeta)\), freeze its local Jacobian

\[
\boldsymbol J_q =
\left.\frac{\partial\boldsymbol q}{\partial\boldsymbol\zeta}\right|_{\boldsymbol 0}
\]

and a positive-definite output metric \(\boldsymbol M_q\). The metric is
important for mixed-unit or differently weighted query outputs; omitting it
selects the identity metric explicitly.

### Direct query observability

The direct fraction is

\[
s_{\mathrm{dir}} =
\frac{\lVert\boldsymbol M_q^{1/2}\boldsymbol J_q\boldsymbol U_r\rVert_F^2}
{\lVert\boldsymbol M_q^{1/2}\boldsymbol J_q\boldsymbol U_r\rVert_F^2+
 \lVert\boldsymbol M_q^{1/2}\boldsymbol J_q\boldsymbol N\rVert_F^2}.
\]

This is a geometric statement. It does not increase merely because the prior
correlates an unobserved direction with an observed one. The complementary
nullspace-sensitivity fraction is \(1-s_{\mathrm{dir}}\).

### Complete-prior covariance reduction

For a full-rank local prior
\(\boldsymbol P^-\), the posterior covariance is

\[
\boldsymbol P^+ =
[(\boldsymbol P^-)^{-1}+\boldsymbol\Lambda_{\mathrm{obs}}]^{-1}.
\]

The corresponding query covariances are

\[
\boldsymbol\Sigma_q^- =
\boldsymbol J_q\boldsymbol P^-\boldsymbol J_q^\top,\qquad
\boldsymbol\Sigma_q^+ =
\boldsymbol J_q\boldsymbol P^+\boldsymbol J_q^\top.
\]

The metric-weighted variance reduction is

\[
s_{\mathrm{red}} =
1-
\frac{\operatorname{tr}(\boldsymbol M_q\boldsymbol\Sigma_q^+)}
     {\operatorname{tr}(\boldsymbol M_q\boldsymbol\Sigma_q^-)}.
\]

This quantity may be positive even when \(s_{\mathrm{dir}}=0\), because a
correlated complete prior can transmit information between gauge directions.
The implementation deliberately reports both quantities instead of calling
prior-mediated reduction direct visual observability.

The worst supported variance ratio is the largest eigenvalue of

\[
(\boldsymbol\Sigma_q^-)^{-1/2}
\boldsymbol\Sigma_q^+
(\boldsymbol\Sigma_q^-)^{-1/2}
\]

on the positive-variance support of \(\boldsymbol\Sigma_q^-\). It catches a
single unresolved query direction that can be hidden by a favorable average
trace reduction.

## Query gate

`QueryObservabilityGate` applies three independently frozen thresholds:

1. minimum direct observability fraction;
2. minimum metric-weighted variance reduction; and
3. maximum worst-direction variance ratio.

A rejection returns stable reason codes. It does **not** mutate the candidate,
replace covariance, invent a ridge, or implement fallback itself. BayesianPhysTwin
must either select a complete admitted candidate or return its exact caller-owned
physical belief.

Thresholds, the query Jacobian, and the query metric must be selected on
development/source or calibration groups before any protected target outcome is
opened. A target-side threshold adjustment would be a new experiment.

## Point-position helper

For a source point \(\boldsymbol x\), the transformed-point query has Jacobian

\[
\boldsymbol J_x =
\begin{bmatrix}
\boldsymbol q & -[\boldsymbol q]_\times & \rho\boldsymbol I_3
\end{bmatrix},
\]

where \(\boldsymbol q\) is the fitted point relative to the transformed centroid
and \(\rho\) is the chart cloud scale. Use:

```python
from prob4d.query_observability import (
    QueryObservabilityGate,
    evaluate_query_observability,
    point_position_query_jacobian,
)

jacobian = point_position_query_jacobian(factor, source_point)
report = evaluate_query_observability(
    factor,
    prior_covariance_local=complete_prior_covariance,
    query_jacobian_local=jacobian,
    query_metric=point_metric,
)
decision = frozen_gate.evaluate(report)
```

Arbitrary nonlinear physical queries remain supported by supplying their own
local Jacobian.

## Deterministic control

The checked-in control isolates one missing rotation-about-line direction. It
uses the same complete identity prior and one frozen gate for three cases.

| Case | Rank | Direct fraction | Variance reduction | Worst ratio | Decision |
|---|---:|---:|---:|---:|---|
| Point on observed line | 6 | 1.000 | 0.909 | 0.091 | admit |
| Distant off-axis probe | 6 | 0.679 | 0.618 | 0.965 | reject |
| Invalid full-rank completion, same probe | 7 | 1.000 | 0.909 | 0.091 | admit |

The invalid completion is intentionally included as a failure control. It
demonstrates that query conditioning cannot repair a factor that already
fabricated information in the geometric nullspace. Preserving rank deficiency
is therefore a prerequisite for a meaningful downstream gate.

Reproduce the result with:

```bash
PYTHONPATH=src python -m prob4d.query_observability_study \
  --output evidence/query-observability-control-v1/result.json
```

## Paper contribution enabled by this kernel

The defensible Prob4D companion-paper claim is not that geometric degeneracy was
newly discovered. Localizability-aware registration already studies weak pose
directions. The contribution is the complete chain:

1. retain a rank-deficient **Sim(3) likelihood** from overlapping learned 4D
   predictions rather than reject it or numerically complete it;
2. fuse it with a complete correlated prior without pretending that the visual
   provider measured its nullspace;
3. project the resulting belief into a declared action-conditioned physical
   query;
4. distinguish direct geometric support from prior-mediated uncertainty
   reduction;
5. admit or reject the complete candidate prospectively; and
6. expose exact fallback, harmful accepted updates, proper scores, and
   worst-group regret as primary endpoints.

This moves the scientific object from “better aligned 4D reconstruction” to
“which physical decisions are identified by an uncertain 4D observation?”

## Required real-provider promotion

The highest-value promotion path remains PointWorld on a fresh,
garment-disjoint Flat'n'Fold robot cohort after issue #333 passes every
source-support gate.

Before target access, freeze:

- exact PointWorld checkpoint, runtime, normalization statistics, and input
  lineage;
- persistent sparse point identities and cross-window association;
- camera/action/metric-frame support;
- source-only covariance and reliability calibration;
- the physical query Jacobian and output metric;
- query-observability thresholds;
- the BayesianPhysTwin regret guard and exact fallback; and
- garment/session statistical units and one-shot target scoring.

The held-out comparison should include:

1. unchanged physical fallback;
2. direct PointWorld or strongest simple deterministic comparator;
3. full-rank/ridge alignment failure control;
4. observable-subspace Prob4D candidate without query admission;
5. query-conditioned Prob4D candidate; and
6. guarded BayesianPhysTwin deployment with exact fallback.

Provider competence and downstream physical-query value must be reported
separately. Primary endpoints are object/session-clustered proper score, physical
query error, accepted-update harm, worst-group regret, calibration and width,
admission/fallback rate, and off-support query behavior. Causal4D may consume
only the admitted belief and remains an optional downstream experiment.

## Claim boundary

The current implementation and deterministic control establish mechanism and
software evidence only. They do not establish real-provider competence,
held-out calibration, BayesianPhysTwin benefit, Causal4D benefit, deployment
safety, or state of the art. A material paper claim requires the prospective
fresh-provider experiment above.
