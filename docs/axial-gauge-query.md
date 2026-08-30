# Nonlinear axial-gauge query laws

**Experimental, conditional method and controlled study. Not a production provider update.**

## Scientific purpose

The observable-subspace factor avoids inventing information in a deficient
Sim(3) gauge. The query-observability diagnostic then distinguishes direct
geometric information from prior-mediated variance reduction. Both are local
Gaussian constructions. A further issue remains: an unresolved rotation acts
nonlinearly on an off-axis physical query. Retaining its tangent variance does
not preserve the resulting distribution, and even exact Euclidean mean and
covariance do not determine event probabilities.

This module adds the following narrowly bounded method step:

> Preserve the conditional angular law along an axial gauge orbit, update it
> only through a declared likelihood, and propagate the same angular draw into
> every coordinate of a joint physical query.

It provides exact finite-quadrature posterior updates and pushforwards, a
first/two-harmonic moment calculation, joint query covariance, mixture density,
and halfspace probability. It changes neither stable provider-v2 export nor
BayesianPhysTwin's complete-belief routing or exact fallback.

## Conditional model and its limits

Let eta denote the observable geometry/gauge and let theta denote rotation
about its reference line. If the **entire positional likelihood**, including
its covariance model, is invariant along that orbit, then

    p(theta | eta, D) = p(theta | eta).

This is a conditional statement. It does not imply that eta and theta are
independent, that the angular prior is uniform, or that the marginal angular
posterior is unchanged when the posterior over eta changes. For uncertain eta,
a full belief must mix the eta-specific query laws with the posterior eta
weights and preserve each prior conditional. This module operates at one fixed
eta; it is not that complete-belief constructor.

`AxialGaugeOrbit.from_line` checks only exact **mean geometry**, to a declared
relative numerical tolerance. A local rank deficiency alone does not prove a
global symmetry. Normals, appearance, anisotropic transformed source covariance,
or other likelihood terms may carry angular information even for a collinear
mean. Do not use the geometry check to erase that information.

For a declared axis and fixed observable geometry, the optional
`condition_on_correspondences` operation instead evaluates the actual nonlinear
positional likelihood on the angular nodes. Weakly curved or off-axis geometry
then updates the angle. The complete residual covariance may include
cross-point dependence; it must be positive definite and fixed with respect to
the angle. Unknown source coordinates, angle-dependent covariance, uncertain
associations, and joint scale/rotation/translation inference are outside this
likelihood's scope. Only observations allowed by the caller's causal/source
protocol may be supplied.

## Exact moment calculation

For axis a, line point c, and nominal query point q, write

    b = c + a a^T (q - c)
    u = (I - a a^T) (q - c)
    v = a cross u
    q(theta) = b + u cos(theta) + v sin(theta).

Let m1 = E[exp(i theta)] and m2 = E[exp(2 i theta)]. Then

    E[q] = b + u Re(m1) + v Im(m1)

and the covariance is `[u v] C [u v]^T`, where

    C11 = (1 + Re(m2))/2 - Re(m1)^2
    C22 = (1 - Re(m2))/2 - Im(m1)^2
    C12 = Im(m2)/2 - Re(m1) Im(m1).

Stacking the u and v vectors for several points gives their **joint** covariance.
The same angular atom must be used for every point. For two antipodal probes,
independent marginal draws would invent variance in their fixed midpoint.

These are standard trigonometric-moment identities, not new circular-statistics
theorems. `moments` is exact for the supplied finite angular law. A periodic
quadrature approximates a continuous law; grid refinement is a numerical check,
not a universal error bound.

## Why covariance correction alone is insufficient

A uniform angular law and equal masses at angles 0, 2pi/3, and -2pi/3 have the
same first and second trigonometric moments: both vanish. For an off-axis probe
at radius r, both therefore give the same Cartesian mean and covariance.
Nevertheless, the probability that its radial reference coordinate is positive
is 1/2 under the uniform law and 1/3 under the threefold law.

Thus a Gaussian retaining even the exact mean and covariance cannot distinguish
these two decisions. The threefold law's third harmonic matters. Under the
threefold law, predicting 1/2 instead of 1/3 adds 1/36 to expected Brier loss.
This counterexample is deterministic; the smooth controlled study adds small
angular and readout noise and tests density and decision value separately.

## Use with a verified rank-six factor

```python
import numpy as np
from prob4d.axial_gauge import AxialGaugeOrbit, CircularQuadrature

# Existing factor.chart.linearization maps the source cloud into reference units.
reference_overlap = factor.chart.linearization.transform_points(source_overlap)
orbit = AxialGaugeOrbit.from_line(reference_overlap)
# Check that the full declared likelihood, not just the mean cloud, admits this
# axial model. The caller supplies the conditional angle law in this axis convention.
angular = CircularQuadrature(angles=registered_angles, weights=conditional_angle_mass)
reference_queries = factor.chart.linearization.transform_points(source_query_points)
query = orbit.pushforward(reference_queries, angular, noise_covariance=query_readout_covariance)

# query.atoms contains complete point-major flattened query vectors.
mean = query.mean
joint_covariance = query.covariance
probability = query.halfspace_probability(registered_normal, registered_threshold)
```

`from_line` fixes the largest-magnitude axis component positive. Angular priors
must use that convention; changing the axis sign requires changing angle signs.
The readout noise covariance is independent of the angular atom. The noiseless
pushforward remains a discrete measure: `logpdf` refuses to label it a continuous
density. A positive-definite readout covariance makes the Gaussian-mixture
Lebesgue density well-defined.

## Reproduce the controlled study

```bash
python -m pytest -q tests/test_axial_gauge.py
python -m prob4d.axial_gauge_study \
  --source-revision "$(git rev-parse HEAD)" \
  --output outputs/axial-gauge-query-control-v1/result.json
```

The source-frozen protocol is embedded in the study module and copied with its
SHA-256 into the result. It contains two independent seeds, five angular regimes,
16,384 independent gauge/readout draws per seed and regime, a 50 mm off-axis
probe, and 3 mm isotropic readout noise. Each draw is one statistical unit;
coordinates and quadrature atoms are not independent units. Truth angles are
sampled continuously by a separate simulator, never from the predictive grid.

The three arms use exactly the same conditional geometry, prior information,
and known readout noise:

1. tangent Gaussian: nominal-position linearization with unwrapped angular variance;
2. exact-moment Gaussian: the nonlinear mean and complete covariance, but Gaussian shape;
3. axial-orbit mixture: the full shared-angle law with the same additive noise.

The second and third arms have matched mean and covariance by construction. Their
NLL and Brier differences test distribution shape, not an improved mean or a
larger covariance. The primary comparison is their paired NLL, with a fixed
halfspace Brier score as a separate decision diagnostic. Per-seed results,
Monte Carlo intervals, numerical moment mismatch, and 512-to-1024-node density
refinement are retained. No threshold is tuned to the simulated outcomes.

Finalized numerical evidence and interpretation belong in
`FlorianPfaff/BayesianPhysTwin-Paper`, not in a public provider-promotion claim.
The local tests also cover likelihood-invariant prior preservation, genuine
angular updating, cross-point noise, rigid-frame equivariance, geometry
rejection, continuous wrapped-normal moment identities, and singular-density
failure handling.

## Relation to existing work and paper positioning

Non-Gaussian rotational uncertainty, symmetry-aware pose inference, and circular
moments are established topics. Relevant primary sources include:

- Murphy et al., *Implicit-PDF*, ICML 2021:
  https://proceedings.mlr.press/v139/murphy21a.html
- Maken, Ramos, and Ott, *Estimating Motion Uncertainty with Bayesian ICP*, 2020:
  https://arxiv.org/abs/2004.07973
- Kurz et al., *Directional Statistics and Filtering Using libDirectional*:
  https://arxiv.org/abs/1712.09718
- Tuna et al., *X-ICP: Localizability-Aware LiDAR Registration*:
  https://arxiv.org/abs/2211.16335

The candidate Prob4D contribution is the narrower connection from deficient
learned-window gauges to joint physical-query laws and measurable decision loss
that survives exact mean/covariance matching. It is not the invention of a
circular distribution, rotational symmetry, quadrature, or Bayesian conditioning.
It specializes, rather than duplicates, the paper repository's general
query-quotient lifting arguments.

A stronger empirical paper still needs a separately frozen real-provider study,
including a complete-belief integration and query-level proper-score/value
comparison. The current PointWorld source-qualification route remains separate;
this experiment opens no PointWorld, Flat'n'Fold, MotionCrafter, Deform360,
BayesianPhysTwin, or Causal4D outcome and does not reopen any terminal cohort.
