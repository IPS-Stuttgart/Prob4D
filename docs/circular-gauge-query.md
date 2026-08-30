# Nonlinear physical queries under an unresolved circular gauge

**Status:** experimental, locally implemented and tested on 2026-08-30. Additive
patch reviewed against Prob4D `224d13dc9a93731ac5297b479eb1e121b3dbe659`.
No stable API, existing estimator, admission decision, or historical evidence is
changed. This is not a real-provider or deployment result.

## Why this is a scientific extension rather than another adapter

The existing `observable_gauge` path retains a rank-deficient Sim(3) information
factor. `query_observability` projects that factor with a local query Jacobian.
The separate gauge-linearization-closure diagnostic compares first-order
propagation with a spherical-radial reference using `2r` symmetric points for a
rank-`r` Gaussian coordinate covariance. These are useful, explicitly local
instruments, but neither a local derivative nor that specific cubature rule
establishes correctness for a nonlinear physical-query distribution.

A circular ambiguity gives a rigorous example. Rotation about a line can be
unobserved by collinear geometric correspondences. An off-axis point then lies
on a circle, not on a Gaussian tangent line. In particular, a radial coordinate
can have zero local derivative and strictly positive variance. The symmetric
rank-one two-point cubature rule also misses that variance exactly.

The extension propagates an **explicit conditional circular prior** through
such queries, preserving a single shared phase across points and time. It
provides exact conditional moments in a two-column covariance factor and
joint linear-constraint event probabilities through unions of circular arcs.

## Premises: local nullspace is not a global symmetry certificate

Let `psi` denote the observable quotient and `phi` a residual rotation. The
required likelihood property is the GLOBAL conditional invariance

    L(y | psi, phi) = L(y | psi) for every supported phi.

Under that premise,

    p(phi | psi, y) = p(phi | psi).

The conditional prior must be retained, not replaced by a uniform density, a
zero twist, an arbitrary Gaussian, or an unconditional phase marginal. Prior
correlation between `psi` and `phi` matters.

A rank-six local Jacobian alone does not prove this premise. Nearly collinear
geometry, oriented anisotropic likelihoods, physical contact, and additional
sensors can make the rotation weakly informative rather than exactly
unobserved. `validate_declared_line_support` only checks the supplied geometry
against a declared line within numerical tolerance. It does not prove
likelihood invariance. The caller owns that additional premise.

The current implementation conditions on `psi`. It does not perform the
outer integral over a complete physical belief. For a complete belief, use

    E[q | y] = E_psi[ E[q | psi, y] ],
    Cov(q | y) = E_psi[ Cov(q | psi, y) ]
                + Cov_psi( E[q | psi, y] ).

For a physical-particle mixture the corresponding sums must use the existing
particle weights and each particle's conditional phase prior. That integration
is a future adapter, not something claimed to have been executed here.

## Exact conditional moment proposition

For a fixed observable quotient, stack any number of points, frames, or linear
query projections into

    q(phi) = c + a cos(phi) + b sin(phi) = c + A t(phi),
    A = [a b],  t(phi) = [cos(phi), sin(phi)]^T.

Let `z_k = E[exp(i k phi)]`. Then

    m = E[t] = [Re(z_1), Im(z_1)]^T,

    E[t t^T] = 1/2 [[1 + Re(z_2), Im(z_2)],
                       [Im(z_2), 1 - Re(z_2)]],

    C = E[t t^T] - m m^T,
    E[q] = c + A m,
    Cov(q) = A C A^T.

**Proof.** Expand the affine expression for `q`. Use
`cos^2(phi)=(1+cos(2phi))/2`,
`sin^2(phi)=(1-cos(2phi))/2`, and
`sin(phi)cos(phi)=sin(2phi)/2`. No small-angle approximation is used.

The complete stacked covariance has rank at most two for a fixed quotient,
although the latent phase has only one dimension. A factor `A C^(1/2)` stores
all cross-point and cross-time covariance in linear memory. Its two columns
are harmonic covariance modes, not two independent phases.

For a mixture of wrapped normals and a uniform component,

    z_k = sum_j w_j exp(i k mu_j - k^2 sigma_j^2/2),  k != 0,
    z_0 = 1.

The uniform contribution vanishes for nonzero harmonics. The implementation
uses component-centered covariance formulas and total covariance to avoid
catastrophic cancellation for a narrow wrapped normal.

## Analytic counterexample to the specified local and cubature rules

Take a 100-mm off-axis point rotating about the z-axis:

    q(phi) = r [cos(phi), sin(phi)],
    phi ~ wrapped Normal(0, sigma^2).

The radial coordinate has

    E[q_x] = r exp(-sigma^2/2),
    Var(q_x) = r^2/2 (1 - exp(-sigma^2))^2,
    Var(q_y) = r^2/2 (1 - exp(-2 sigma^2)).

For every `sigma > 0`, the radial variance is strictly positive. However,
`d q_x/d phi` at zero is zero, so first-order propagation assigns zero radial
variance. The rank-one spherical-radial rule evaluates `phi=+sigma` and
`phi=-sigma`. Both have the same cosine, so that rule also assigns exactly
zero radial variance. Its nonlinear mean correction does not repair the
missing second moment.

At `r=0.1 m`, `sigma=1 rad`, the exact radial mean is `60.653066 mm` and the
standard deviation is `44.697673 mm`. For the illustrative event `q_x<50 mm`,
the exact probability is `0.2950083103791666`. Both specified approximations
assign probability zero because their deterministic radial prediction lies
above the threshold. A model-conditional 10% gate would incorrectly admit
those approximations.

This does NOT show that all sigma-point methods fail. Higher-order
Gauss-Hermite rules recover the smooth moments, and moment-matched circular
sampling is established prior work. The bundled study retains all declared
Gauss-Hermite orders rather than selecting the weakest one.

## Why moment matching is still insufficient for event probabilities

An exactly moment-matched Gaussian assigns probability `0.4058102397` to the
same event, rather than `0.2950083104`. Likewise, 17-node Gauss-Hermite recovers
the mean and variance to essentially machine precision but its direct
indicator quadrature gives `0.2470489567`. Finite-node integration of a
discontinuous indicator need not converge monotonically.

The claim is not that quadrature cannot solve this problem. It is that accurate
mean and covariance, or agreement of a limited cubature reference, is not an
event-probability certificate. Here the circular structure admits a direct
calculation without indicator quadrature.

## Shared-phase joint-event proposition

For one shared phase and registered affine constraints, write failure as

    exists j: c_j + a_j cos(phi) + b_j sin(phi) > 0.

Each nonconstant inequality is an arc with center `atan2(b_j,a_j)` and
half-width `acos(-c_j / hypot(a_j,b_j))`, with constant, full-circle, and empty
cases handled separately. Splitting at the period boundary and merging the
arcs gives disjoint intervals `[l_i,u_i]` on `[0,2 pi]`.

For a uniform prior, the probability is total arc length divided by `2 pi`.
For a wrapped-normal component,

    P = sum_i sum_{k in Z}
        [Phi((u_i + 2 pi k - mu)/sigma)
         - Phi((l_i + 2 pi k - mu)/sigma)].

Mixture probabilities are the corresponding weighted sums. Finite summation
over complete periods supplies a lower value and an upper value obtained by
adding the omitted unwrapped-normal tails. This bounds truncation only; it is
not formal floating-point interval arithmetic and does not bound model error.

**Proof.** Rewrite each inequality as a shifted cosine, solve its superlevel
set, and take the union. Integrate a wrapped density over that union by summing
its lifted real-line normal components. Disjointness prevents double counting.
The omitted union mass cannot exceed the omitted mass of the complete periods.

A repeated constraint changes neither the arc union nor its probability.
Treating frames as independent does not have this invariance. Five identical
10%-risk constraints have joint risk 10%, not `1-0.9^5=40.951%`. Five disjoint
phase-violation arcs of the same size have joint risk 50%, also not 40.951%.
Thus an independence assumption can be either conservative or anticonservative.

## Use

```python
import numpy as np
from prob4d.circular_gauge_query import (
    AffineCircularQuery,
    CircularPrior,
    bounded_risk_admissible,
    path_violation_probability,
    point_rotation_orbit,
)

# This must be an explicitly justified CONDITIONAL prior, not a convenience default.
prior = CircularPrior.wrapped_normal(0.0, 1.0, prior_id="source-frozen-conditional-prior")
orbit = point_rotation_orbit(
    np.array([[0.1, 0.0, 0.0]]),
    axis_origin=np.zeros(3),
    axis_direction=np.array([0.0, 0.0, 1.0]),
)
factored = orbit.low_rank_moments(prior)  # mean plus (3 x 2) covariance factor

# Positive violation means x < 0.05 m. There is no additional state/noise uncertainty here.
violation = orbit.project(np.array([[-1.0, 0.0, 0.0]]), offset=np.array([0.05]))
bounds = path_violation_probability(violation, prior)
admitted = bounded_risk_admissible(bounds, maximum_risk=0.10)
assert not admitted
```

The consumer must retain the complete caller-owned fallback on rejection.
Nothing in this experimental module replaces a BayesianPhysTwin belief or
issues a Causal4D action. Its risk calculation is conditional on the declared
model, not a robot safety certificate.

## Reproduction and evidence location

```bash
PYTHONPATH=src python -m pytest -q tests/test_circular_gauge_query.py
PYTHONPATH=src python scripts/science/circular_gauge_query_study.py \
  --output-dir /tmp/circular-gauge-control-new --plots
python scripts/science/verify_circular_gauge_query_evidence.py \
  --evidence-dir /tmp/circular-gauge-control-new --source-root .
```

The runtime module needs NumPy only. Tests use pytest; one numerical integration
test additionally uses SciPy and is skipped if it is unavailable. Optional
figures require Matplotlib. The evaluated environment ran all 36 tests without
skips. Repository-wide CI, formatting, lint, installed-wheel integration, and
provider execution were not run in this read-only session.

Paper-facing JSON, CSV, figures, validation records, and interpretation belong
in `FlorianPfaff/BayesianPhysTwin-Paper`, not in the public runtime code tree.

## Relation to existing work and claim limits

Circular moments, wrapped-normal filtering, deterministic circular sampling,
Rao-Blackwellization, and degeneracy-aware registration are not new inventions.
Relevant primary sources include:

- Kurz, Gilitschenski, and Hanebeck, *Recursive Bayesian Filtering in Circular
  State Spaces*, IEEE AES Magazine 31(3), 2016, DOI 10.1109/MAES.2016.150083;
  author preprint: https://arxiv.org/abs/1501.05151.
- Kurz, Gilitschenski, Siegwart, and Hanebeck, *Methods for Deterministic
  Approximation of Circular Densities*, JAIF 11(2), 2016, pp. 138–156;
  https://isif.org/media/methods-deterministic-approximation-circular-densities.
- Kurz et al., including Florian Pfaff, *Directional Statistics and Filtering
  Using libDirectional*, JSS 89(4), 2019, DOI 10.18637/jss.v089.i04.
- Tuna et al., *X-ICP: Localizability-Aware LiDAR Registration for Robust
  Localization in Extreme Environments*, IEEE T-RO 40, 2024, pp. 452–471;
  https://arxiv.org/abs/2211.16335.
- Huang, Mourikis, and Roumeliotis, *Observability-based Rules for Designing
  Consistent EKF SLAM Estimators*, IJRR 29(5), 2010,
  DOI 10.1177/0278364909353640.

The candidate contribution is the narrower connection between retained
unobservable 4-D gauge structure, nonlinear physical-query validity, shared
trajectory uncertainty, and conditional exact fallback. The analytic
counterexample and implementation establish a mechanism, not a field-wide
novelty or empirical superiority claim.

A material paper improvement still needs source evidence that this ambiguity
occurs in the selected provider and a frozen downstream query evaluation. If
that source check finds no material circular ambiguity, retain this as a
bounded theory/diagnostic extension rather than force it into the main claim.
