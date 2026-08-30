# Finite-orbit query certificates

## Scientific purpose and scope

The local diagnostic in `query_observability.py` reports the sensitivity of a
query Jacobian to observable and null directions. It does not assert a global
invariance theorem. This extension asks a different question: can an action
comparison remain valid for **every finite rotation left unresolved by an exact
line-shaped overlap**?

`prob4d.axial_query_certificate` preserves that ambiguity rather than completing
a deficient information matrix or treating a zero derivative as a finite-angle
certificate. This is an experimental, conditional-model kernel. It is not a
replacement for source support, identity, covariance, or provider-value gates.

## 1. Exact correspondence geometry

Assume labeled point correspondences, at least two distinct source points on one
line, exact agreement, positive Sim(3) scale, and proper rotations. Let `g0` be
one exact fit. Any other fit composed with `inverse(g0)` fixes every target
correspondence. The distance between two distinct fixed points forces relative
scale one; their difference forces the relative rotation to fix the line's
unit direction. The remaining transformations are exactly rotations about the
target line. Conversely, every such rotation fixes the target correspondences.

This identifies an SO(2) stabilizer. It is **not** implied by a generic rank-six
matrix. With noisy or near-collinear correspondences, the exact-fit argument no
longer applies. `maximum_support_displacement` reports twice the maximum radial
distance to the declared line; it does not silently replace a nearly straight
cloud by an exact symmetry using a numerical rank threshold.

## 2. A local false assurance

Take the overlap on the z-axis and a probe at `(1, 0, 0)`. Its x-coordinate is

```text
q(theta) = cos(theta).
```

At the identity representative its derivative in the unresolved rotation is
zero, but its full-orbit range is `[-1, 1]`. Inflating variance in that direction
cannot change the first-order query covariance when the corresponding Jacobian
column is zero.

Under the analytic rank-six factor with information `10 P_observed` and complete
prior `I_7`, the existing scalar query diagnostic returns direct support `1`,
variance reduction `10/11`, and worst variance ratio `1/11`. It passes thresholds
`(0.8, 0.8, 0.5)`. The new integration test calls the existing API and exhibits an
affine fallback-minus-candidate loss `0.25 + cos(theta)`, whose lower bound is
`-0.75`. Local support therefore cannot substitute for a finite-orbit action
certificate. This is a limitation of the local approximation, not evidence that
the existing implementation calculates its declared local quantities wrongly.

## 3. Exact finite-angle bounds

For line origin `o`, unit direction `u`, and representative point `x_j`, write

```text
p_j = u u^T (x_j - o)
r_j = (I - u u^T) (x_j - o)
x_j(theta) = o + p_j + cos(theta) r_j + sin(theta) (u cross r_j).
```

Every scalar affine query `q = b + sum_j w_j^T x_j(theta)` has the form

```text
q(theta) = C + A cos(theta) + B sin(theta),
C = b + sum_j w_j^T (o + p_j),
A = sum_j w_j^T r_j,
B = sum_j w_j^T (u cross r_j).
```

Its full-circle extrema are exactly `C +/- hypot(A, B)`. On a closed circular
arc, it suffices to evaluate the two endpoints and the maximum/minimum stationary
angles that belong to the arc. The implementation handles wraparound and
singleton arcs. A zero derivative at zero establishes `B = 0`, not `A = 0`.

This also covers isotropic squared-distance objectives to fixed target points:
`||x_j(theta) - y_j||^2` is a first harmonic because the rotating radial component
has constant norm. Its coefficients are

```text
C_j = ||o + p_j - y_j||^2 + ||r_j||^2,
A_j = 2 (o + p_j - y_j)^T r_j,
B_j = 2 (o + p_j - y_j)^T (u cross r_j).
```

Nonnegative weighted sums retain this form. General simulator rollouts,
anisotropic quadratic objectives, contact changes, and arbitrary nonlinear
queries do not automatically have this representation.

## 4. Compare actions under the SAME unresolved gauge

Let `L_f` and `L_c` denote fallback and candidate losses with a shared latent
angle and the same representative convention. Form `D = L_f - L_c` **before**
optimizing over the angle. Bounding the two losses independently loses their
shared dependence. For example,

```text
L_f(theta) = 4 + 2 cos(theta) + 3 sin(theta),
L_c(theta) = 3.75 + 2 cos(theta) + 3 sin(theta)
```

has advantage exactly `0.25`, although independent intervals overlap widely.

An explicit `shared_gauge_id` is mandatory. Matching origins and axes does not
prove shared uncertainty: two independent gauge variables with identical
geometry must not cancel. The key also binds the geometric convention. A
source/protocol owner must justify the identity and representative; the kernel
does not infer them from coincident numbers.

**Conditional dominance proposition.** Suppose the true shared angle belongs to
the supplied nonempty arc, and the true advantage differs from the represented
`D(theta)` by at most `epsilon`, uniformly on that support. If

```text
min_arc D(theta) - epsilon > required_margin,
```

then the candidate improves the fallback loss by more than `required_margin`
for every admitted state. Proof: subtract the uniform error bound pointwise and
then use the minimum. The real-arithmetic full-circle criterion is
`C_D - hypot(A_D, B_D) - epsilon > required_margin`.

The implementation also requires a caller-owned `scope_admitted` flag and a
strict numerical slack. A false scope or empty support always rejects. The flag
is not evidence on its own. `epsilon` must cover all omitted effects on the
advantage: observed gauge coordinates, model discrepancy, unmodeled nonlinear
responses, and relevant action uncertainty. It is not interchangeable with a
standard deviation. A Gaussian prior with unbounded support does not imply a
finite deterministic envelope. A statistically constructed support would need
its own coverage argument, including selection and dependence.

Floating-point formulas are not outward-rounded interval arithmetic. The
explicit numerical slack is an engineering tolerance in loss units, not a
proved rounding-error enclosure or a deployment-safety theorem.

## 5. One bounded-error metric anchor

A separately admitted point anchor restricts angles by
`||x_a(theta) - y_a|| <= radius`. Decomposing the points into axial and radial
components gives

```text
squared_residual(theta) = K - 2 (a cos(theta) + b sin(theta)).
```

Thus its feasible support is a circular arc obtained from
`cos(theta - atan2(b,a)) >= (K - radius^2)/(2 hypot(a,b))`. The implementation
returns a full circle for an uninformative feasible anchor and `None` for
inconsistent support; it never accepts vacuously from an empty set. An on-axis
anchor cannot resolve rotation. The radius is a supplied bound, not a learned
confidence interval. Reusing a visual point as an independent new sensor is not
justified.

## Usage and downstream ownership

```python
import numpy as np
from prob4d.axial_query_certificate import (
    AxialRotationOrbit,
    certify_shared_orbit_advantage,
)

orbit = AxialRotationOrbit(
    origin=np.zeros(3),
    axis=np.array([0.0, 0.0, 1.0]),
    shared_gauge_id="source-bound-factor-and-representative-id",
)
fallback = orbit.affine_query([[1, 0, 0]], [[0, 0, 0]], offset=4.0)
candidate = orbit.affine_query([[1, 0, 0]], [[-1, 0, 0]], offset=3.75)
certificate = certify_shared_orbit_advantage(
    fallback_loss=fallback,
    candidate_loss=candidate,
    scope_admitted=True,  # justified analytically for this exact constructed example
)
assert not certificate.admitted
assert certificate.lower_advantage == -0.75
```

In a real integration, Prob4D supplies an admitted geometric factor and its
lineage; the application supplies query coefficients and a justified envelope.
The certificate is only an additional guard input. BayesianPhysTwin's
`inference.v2` policy must still construct its own `CompleteBeliefGuardDecisionV1`
and route a rejection to the exact original complete baseline belief. This
module does not replace that router or authorize a latent-state correction.
Causal4D can consume a selected BayesianPhysTwin belief, but this extension does
not establish counterfactual or intervention validity.

## Reproduction and evidence boundary

```bash
python -m pytest -q tests/test_axial_query_certificate.py \
  tests/test_axial_query_certificate_integration.py
python -m prob4d.axial_query_study --seed 73029 --cases-per-family 512 \
  --output outputs/finite-orbit-query-v1/result.json
```

The four constructed families deliberately distinguish a stationary but
ambiguous query, a positive whole-orbit margin, shared-gauge cancellation, and
an omitted-effect envelope. They are failure controls, not a sampled deployment
population. The separate anchor arm has one additional observation and is not
an equal-information comparator. The study's standalone object-identity check
is not execution of BayesianPhysTwin's full-belief router. Paper-facing JSON,
run identities, and tables belong in `FlorianPfaff/BayesianPhysTwin-Paper`.

The next empirical step is an additional source-developed, frozen method arm
for the PointWorld/Flat'n'Fold path tracked in #333, followed by a fresh
object/garment-disjoint evaluation only when its existing source gates authorize
it. Freeze one query, correct shared identities, an admitted support/envelope,
a coverage/width trade-off, fallback, and all decision thresholds before target
access. Report accepted-update harm, acceptance, proper scores where a genuine
predictive distribution is available, and clustered decision value. Preserve
all unsupported cases and negatives. Do not reopen any terminal MotionCrafter,
Deform360, or CUT3R cohort or call this mechanism result provider competence.

## Related work and defensible positioning

Symmetry-aware pose sets are established prior work: Bregier et al., *Defining
the Pose of any 3D Rigid Object and an Associated Distance*, arXiv:1612.04631.
Distributional pose estimates for uncertainty-aware manipulation are also
studied, including Jin et al., *SE(3)-PoseFlow*, arXiv:2511.01501. The proposed
contribution is not the invention of pose ambiguity, SO(2) geometry, harmonic
optimization, or robust dominance. It is a finite-orbit, shared-dependence query
certificate and explicit local-to-finite failure control for this probabilistic
4D-to-physical-query pipeline. Establishing broad novelty requires a fuller
related-work comparison, and establishing practical value requires fresh
provider evidence.
