# Finite axial-gauge ambiguity for physical queries

## Purpose and status

This experimental, NumPy-only analysis extends the **local**
[query-observability diagnostic](query-conditioned-observability.md) with exact
finite-orbit bounds for a restricted but useful geometry: one free rotation
about a registered line per shared gauge group, with all other transformation
parameters fixed. It also supplies sharp worst-case regret for actual affine
action losses that share those same gauges.

The existing local diagnostic is not mathematically incorrect. The new control
shows why its direct-observability fraction must not be interpreted as a global
invariance or decision certificate. A small first derivative does not establish
that a query is constant on the finite observation-equivalence class.

This is analytic development evidence, not real-provider competence,
calibration, BayesianPhysTwin benefit, Causal4D benefit, or deployment safety.
No provider, dataset, protected target, or physical execution is used. No
existing result, production exporter, stable API, or admission rule is changed.

## A falsifying control with a positive counterpart

An exactly collinear overlap lies on the x-axis. Rotation about that axis leaves
all its point coordinates unchanged. A probe initially at `(0, 0.1, 0)` has

$$q(\phi)=0.1\cos\phi,\qquad q'(0)=0,\qquad q''(0)=-0.1.$$

For the existing analytic rank-six factor, the scalar y-position Jacobian has
zero projection onto the missing rotation-x direction. With its complete
identity prior and observable precision 10, the local report gives direct
support 1, variance reduction 10/11, and worst variance ratio 1/11. The existing
illustrative `(0.8, 0.8, 0.5)` gate admits this scalar query. Nevertheless, the
finite y-query range over a full rotation is `[-100, 100] mm`.

These comparisons concern **different guarantees**. A full-circle ambiguity
bound is deliberately stronger than a prior-weighted local uncertainty
calculation. The control does not establish that the local posterior's actual
credible interval undercovers or that its Bayesian decision is wrong.

There are also useful positive controls. The axial coordinate of an off-axis
probe remains fixed. Two points with the same transverse displacement have an
exactly invariant y-difference when they share one angle. Treating their angles
as separately variable destroys that cancellation; falsely merging unrelated
angles invents it.

## Domain assumptions

The caller registers metric-space points, their persistent identities, a pivot
and unit axis for each group, and which points share exactly the same angle.
Each group angle ranges over a full circle; the joint domain is the Cartesian
product of those circles. This is a set-valued domain, **not** an assertion of
statistical independence. A smaller or coupled domain is conservatively
contained by this product, but the bounds need not be sharp on that smaller
domain.

A rank-six information matrix alone does not prove the presence of an exact
axial symmetry. The caller must establish that the finite transformations are
admissible for the relevant observation model. An exactly collinear support
has zero geometric motion under its axial orbit. A near-collinear support does
not: `maximum_support_motion` returns twice its maximum distance to the axis.
That is a geometric motion bound, not a bound on likelihood ratio, calibration,
or posterior probability. Additional observations, metric anchors, covariance
orientation, priors, or physical constraints can restrict or break the proposed
symmetry and must not be ignored.

Points are already expressed in the aligned metric frame. Uncertainty in scale,
axis location, other gauge directions, identity, deformable dynamics, and
nonlinear physical queries is outside this exact special case. The helper does
not automatically turn a provider dependence-group label into a shared angle.

## Exact query bounds

Let `a_g` be a unit axis through `c_g`, and decompose each point in group `g` as

$$d_i=p_i-c_g,\quad d_i^\parallel=a_ga_g^\top d_i,
\quad d_i^\perp=d_i-d_i^\parallel.$$

Rodrigues' formula gives

$$p_i(\phi_g)=c_g+d_i^\parallel
 +d_i^\perp\cos\phi_g+(a_g\times d_i^\perp)\sin\phi_g.$$

For affine queries `q_k = b_k + sum_i w_ki^T p_i`, collect contributions
**inside each shared group before taking an amplitude**:

$$q_k(\phi)=C_k+\sum_g[A_{kg}\cos\phi_g+B_{kg}\sin\phi_g].$$

Consequently the sharp componentwise interval is

$$q_k\in\left[C_k-\sum_g\sqrt{A_{kg}^2+B_{kg}^2},
                 C_k+\sum_g\sqrt{A_{kg}^2+B_{kg}^2}\right].$$

Proof: each harmonic has extrema plus or minus its amplitude, attained at
`atan2(B_kg, A_kg)` and that angle plus pi. Product-domain freedom lets the
extrema be attained simultaneously for one query. Different query endpoints
need not be jointly attainable: the interval collection is not an exact
rectangular joint feasible set.

Global invariance holds exactly when every `A_kg` and `B_kg` is zero. Stationarity
at the reference only requires every `B_kg` to be zero. This isolates the
first-order blind spot without a Monte Carlo tolerance or an arbitrary grid.

## Sharp action regret preserves common uncertainty

When the query family contains actual lower-is-better affine action losses
`L_k`, define regret for a caller-selected action `k` by

$$R_k=\sup_\phi\left[L_k(\phi)-\min_j L_j(\phi)\right].$$

For finitely many actions this has the closed form

$$R_k=\max_j\left\{C_k-C_j+
 \sum_g\sqrt{(A_{kg}-A_{jg})^2+(B_{kg}-B_{jg})^2}\right\}.$$

Proof: interchange the supremum and the finite maximum over `j`, and apply the
harmonic bound to each loss difference. Including `j=k` ensures nonnegative
regret. `regret_witness(k)` returns a competing action and an angle vector
attaining this bound. Query contrasts share the same angles, so amplitudes are
computed **after subtraction**, not from separate marginal interval endpoints.

For `L_0=0.1-y` and `L_1=0.1+y`, the reference selects action 0 with zero loss.
At half a turn, action 0 has loss 200 mm while action 1 has zero loss. Both
worst-case regrets are 200 mm, so an illustrative 50 mm regret budget rejects
that reference-selected action. This is a constructed loss, not robot-control
or physical-state estimation performance.

Conversely, `L_0=0.05+y` and `L_1=0.10+y` have overlapping marginal loss
intervals but action 0 is uniformly preferable: its exact regret is zero.
The common uncertain term cancels. The method therefore does not simply
reject everything with a broad marginal interval.

## Reproduce and use

From an installed source checkout:

```bash
python -m pytest tests/test_axial_gauge_query.py
python -m prob4d.axial_gauge_query_study \
  --output outputs/axial-gauge-query-development-v1.json
```

The output is labelled deterministic development evidence. Repeating the command
at the same path is allowed only for identical bytes; different existing output
is never overwritten. The tests include an independent quaternion-rotation
reference, attaining extrema, common-nuisance cancellation, frame and axis-sign
invariance, malformed inputs, and parity with the actual existing local gate.
The integration test must run in a complete checkout; it does not silently skip
when the existing modules cannot be imported. Random algebraic test cases are
development controls, not independent empirical trials.

```python
import numpy as np
from prob4d.axial_gauge_query import AxialGaugeOrbit, affine_axial_queries

orbit = AxialGaugeOrbit(np.zeros(3), np.array([1.0, 0.0, 0.0]))
losses = affine_axial_queries(
    np.array([[0.0, 0.1, 0.0]]),
    np.array([[[0.0, -1.0, 0.0]], [[0.0, 1.0, 0.0]]]),
    offsets=np.array([0.1, 0.1]),
    point_group_ids=("window-gauge-0",),
    orbits={"window-gauge-0": orbit},
)
regret = losses.worst_case_regrets()  # [0.2, 0.2], in these loss units
within_budget = losses.within_regret_budget(0, maximum_regret=0.05)  # False
```

`within_regret_budget` executes no action and changes no belief. A downstream caller
must retain every existing provider, calibration, identifiability, and support
gate, then route rejection through its existing complete-belief fallback.
Passing this additional check cannot rescue an upstream failure. Applying a
position bound to a nonlinear simulator loss without a valid reduction does not
produce an action-regret guarantee.

The formulas are analytic, but the implementation evaluates them in float64,
not certified interval arithmetic. A caller-frozen nonnegative numerical margin
can make a threshold decision more conservative; it is not a proof of a
floating-point enclosure. The implementation rejects nonfinite values rather
than manufacturing a bounded result.

## Relation to existing work and paper ownership

Weak-direction-aware registration is established prior art: see Tuna et al.,
[X-ICP](https://arxiv.org/abs/2211.16335). Group-invariant observable/unobservable
decompositions are also established; see Shen and Leok,
[Geometric Symmetry Reduction of the Unobservable Subspace for Kalman
Filtering](https://arxiv.org/abs/1901.03474). Rodrigues' formula, harmonic support
functions, and worst-case regret are not claimed as new generic mathematics.
This note does not establish an exhaustive novelty claim.

The bounded contribution candidate is the finite-gauge, shared-lineage query
and decision audit for partially observable learned 4D windows, together with
its stationary-derivative counterexample and cancellation-preserving positive
controls. The existing [Gaussian linearization-closure
diagnostic](gauge-linearization-closure.md) instead checks numerical moment
closure under a declared joint Gaussian. Neither result subsumes the other's
assumptions or gives empirical calibration.

This belongs to the Prob4D geometric observation boundary. It does not duplicate
the general query-quotient, Jeffrey-lift, or cross-intervention theorems owned by
the existing theory companion in the paper repository. No fourth manuscript or
new physical acquisition is proposed. Paper-facing development results and
interpretation belong in `FlorianPfaff/BayesianPhysTwin-Paper`; this public
repository owns the method, tests, and reproducer.

Real-provider value remains a separate empirical question. The existing
PointWorld/Flat'n'Fold source-qualification route in issue #333 remains governed
by its own unresolved source requirements and target-access rules. This method
neither authorizes its execution nor makes it a dependency for the already
bounded BayesianPhysTwin and Causal4D manuscripts.
