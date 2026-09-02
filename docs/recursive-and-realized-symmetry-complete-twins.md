# Recursive and realized symmetry-complete physical twins

This note records the two extensions needed to turn one-step compact-orbit
query gating into a credible physical-twin architecture:

1. **recursive gauge preservation**—invariant evidence may accumulate quotient
   information over time but must not accumulate unsupported group information;
2. **realized-intervention robustness**—an equivariant action certificate must
   include the discrepancy between the intended group-transported action and the
   action physically realized by the robot.

Together with `equivariant-decision-identifiability.md`, the operational claim is
now:

> A physical twin may repeatedly learn everything needed for a downstream
> relative action while never selecting the unresolved physical gauge. It may
> execute that action only when the complete-orbit regret remains admissible
> after adding a declared actuator-realization margin; otherwise it returns the
> exact caller-owned fallback.

## 1. Recursive gauge preservation

Let `c_t` be a quotient class and `g_t` a finite cyclic approximation of a
compact group. The recursive belief is disintegrated as

\[
p_t(c,g)=p_t(c)\,p_t(g\mid c).
\]

An equivariant transition is specified by a quotient transition and a group
increment kernel,

\[
p(c_{t+1},g_{t+1}\mid c_t,g_t)
=
T(c_{t+1}\mid c_t)
K(g_{t+1}g_t^{-1}\mid c_t,c_{t+1}).
\]

The increment depends on the relative group motion, not the absolute gauge.
Prediction is therefore a quotient transition followed by circular convolution
of the conditional group law.

For invariant evidence with likelihood `L(y | c)`, the exact update is

\[
p_{t+1}(c) \propto L(y\mid c)p_t(c),
\qquad
p_{t+1}(g\mid c)=p_t(g\mid c).
\]

The implementation retains the conditional probability array exactly, including
for a class whose posterior quotient mass becomes zero. A zero-mass joint row
cannot determine its conditional group law; constructing such a belief without
an explicit continuation conditional therefore fails closed.

### Information audit

Every update reports

\[
D_{\mathrm{KL}}(p_{t+1}(c,g)\|p_t(c,g))
=
D_{\mathrm{KL}}(p_{t+1}(c)\|p_t(c))
+
\mathbb E_{p_{t+1}(c)}
D_{\mathrm{KL}}(p_{t+1}(g\mid c)\|p_t(g\mid c)).
\]

The last term is the **gauge-information expenditure**. It is required to be
zero, with byte-identical conditional arrays, for invariant evidence. A separate
symmetry-breaking update accepts a gauge-sensitive likelihood and reports the
positive expenditure rather than disguising it as quotient information.

This creates an auditable recursive invariant:

\[
\sum_{t\in\mathcal I}
\mathbb E_{p_{t+1}(c)}
D_{\mathrm{KL}}(p_{t+1}(g\mid c)\|p_t(g\mid c))=0
\]

for the registered set of invariant updates `I`.

## 2. Decision identification after repeated invariant updates

Repeated invariant evidence can concentrate the quotient posterior even while
`p(g | c)` remains maximally diffuse. A gauge-coupled action template can then
be uniquely or approximately identified from the quotient masses.

The controlled recursive study starts with two equally likely quotient classes
and a uniform cyclic group conditional. Six identical invariant observations
with likelihood ratio `0.8 : 0.2` yield

\[
p(c=0\mid y_{1:6})
=
\frac{0.8^6}{0.8^6+0.2^6}
=
\frac{4096}{4097}
\approx 0.999756.
\]

For every tested group cardinality from 8 to 64:

- quotient entropy decreases strictly at every update;
- conditional group entropy remains exactly `log K`;
- cumulative group-information expenditure is zero;
- the same quotient-level action becomes exact zero-regret;
- a subsequent explicitly symmetry-breaking likelihood adds positive group
  information and lowers group entropy.

This demonstrates recursive **task identification without gauge
identification**, not merely a one-step posterior construction.

## 3. Realized-intervention robustness

The ideal equivariant loss is

\[
L_c(g,a)=\ell(g\cdot x_c, g\cdot a).
\]

Let the robot execute `u_a(g)` instead. Assume a registered action-space metric
and deterministic realization radius

\[
d_A(u_a(g),g\cdot a)\leq\varepsilon_{ca}.
\]

If the loss for template `a` is `K_ca`-Lipschitz in the action argument, then

\[
\ell(g\cdot x_c,u_a(g))
\leq
\ell(g\cdot x_c,g\cdot a)+K_{ca}\varepsilon_{ca}.
\]

For an action pair `(a,b)`, the realized pairwise gap is bounded by

\[
D^{ab}_{c,\mathrm{real}}(g)
\leq
D^{ab}_{c,\mathrm{ideal}}(g)
+
K_{ca}\varepsilon_{ca}
+
K_{cb}\varepsilon_{cb}.
\]

Combining this with the compact-orbit cover gives

\[
\overline\Delta_{\mathrm{real}}(a,b)
\leq
\sum_c\lambda_c
\left[
\max_s D^{ab}_{c,\mathrm{ideal}}(g_{cs})
+L_{cab}\rho_c
+K_{ca}\varepsilon_{ca}
+K_{cb}\varepsilon_{cb}
\right].
\]

The realized-action regret certificate is

\[
\overline R_{\mathrm{real}}(a)
=
\max_b\overline\Delta_{\mathrm{real}}(a,b).
\]

This cleanly separates three contributions:

\[
\boxed{
\text{realized regret bound}
=
\text{sampled structural gap}
+
\text{continuous-orbit margin}
+
\text{intervention-realization margin}
}
\]

The implementation reports all terms independently. A large realization radius
cannot be hidden inside a generic covariance; it either converts exact
optimality into an explicit bounded-regret claim or forces exact fallback.

## 4. Controlled realization sweep

The registered ideal losses are `(0, 2, 1)` for the useful action, an
orthogonal action, and fallback. Only the useful action receives realization
uncertainty, with unit action-Lipschitz loss.

At zero regret tolerance:

- radii `0.0`, `0.2`, `0.8`, and `1.0` preserve exact optimality;
- radius `1.2` yields a realized regret upper bound of `0.2` and forces fallback.

With a separately registered regret tolerance of `0.25`, the same radius `1.2`
case is admitted as explicitly bounded regret. This is not a safety claim; it is
a deterministic consequence of the supplied radius and Lipschitz assumptions.

## 5. Required Causal4D interface

For a claim-bearing physical experiment, Causal4D should provide an immutable
intervention receipt containing at least:

- group and metric identity;
- state-orbit and action-transport program digests;
- a common transform-instance or frame identifier binding state and action;
- intended action-template identity;
- commanded and realized intervention provenance;
- the deterministic or calibrated realization radius;
- the domain and confidence statement supporting that radius;
- the loss program and action-Lipschitz certificate;
- caller-owned fallback identity.

The receipt must distinguish:

1. **algebraic coupling:** state and action use the same registered transform;
2. **physical realization:** the actuator stays inside the declared radius;
3. **statistical transport:** the radius remains valid on the target population.

Prob4D can verify the first two inputs algebraically and consume their margin.
BayesianPhysTwin should own the final act-or-fallback decision and any outer
target-valid calibration. None of the three layers should silently infer the
others.

## 6. Strong paper formulation

The strongest defensible central statement is no longer only

> invariant queries can be answered without selecting a gauge.

It is

> **Quotient information can accumulate recursively without gauge hallucination;
> symmetry-sensitive actions can nevertheless be identified through shared
> state-action equivariance; and bounded actuator mismatch enters as an explicit
> additive regret term.**

This yields a strict hierarchy:

\[
\text{state identification}
\Longrightarrow
\text{fixed-coordinate query identification}
\Longrightarrow
\text{fixed-action identification},
\]

but additionally

\[
\text{equivariant action identification}
\not\Longrightarrow
\text{state or fixed-coordinate query identification}.
\]

That last separation is the category-changing result. It explains how a robot
can act correctly in a physically shared frame while remaining uncertain about
an arbitrary global gauge.

## 7. Remaining empirical promotion

The current controlled studies verify the algebra, while the merged Tracking
Cloth result supplies real-trajectory evidence for finite-orbit rejection and
exact fallback under a controlled hidden `SO(2)` gauge. A decisive new public
study still needs to validate the shared action transform and realization radius
on an object- or recording-disjoint cohort.

A suitable protocol would compare:

- gauge-coupled equivariant action;
- MAP/world-frame completion;
- independent state/action gauges;
- invariant-only rejection;
- covariance inflation;
- exact fallback.

The independent unit must be the complete object, specimen, or recording. A
finite target-valid realization radius must be calibrated on separate units
before the held outcomes are opened. Without that step, the result remains a
structural certificate rather than a deployment or safety guarantee.
