# Equivariant decision identifiability under a shared gauge

## Motivation

A symmetry-complete belief should not automatically reject every
symmetry-sensitive quantity.  Some downstream actions transform with the same
unresolved physical gauge as the state.  In that case the absolute state and
absolute action are not identified, while their physical relation and the
resulting loss can be identified exactly.

For example, let an unresolved axial angle rotate a deformable object's local
transverse direction.  A world-frame point completion chooses one unsupported
angle.  An independent-gauge approximation is also wrong: it lets the object and
the action rotate separately, destroying a real shared latent dependence.  A
frame-coupled actuator instead applies the same transform to the action template
that acts on the state.

This module extends symmetry-complete inference from

> invariant query or exact fallback

to

> invariant query, gauge-coupled equivariant action, or exact fallback.

## Decision-equivariant loss

Let a compact group `G` act on physical states and action templates.  For
quotient class `c`, choose one state representative `x_c`.  The loss after a
shared group transform is

\[
L_c(g,a)=\ell(g\cdot x_c,\,g\cdot a).
\]

The strongest condition is joint loss invariance,

\[
L_c(g,a)=L_c(e,a).
\]

For decisions, however, an action-independent offset is irrelevant.  It is
enough that every pairwise difference is invariant:

\[
L_c(g,a)-L_c(g,b)
=
L_c(e,a)-L_c(e,b).
\]

This is **decision equivariance**.  It says that all members of one physical
orbit have the same action-difference signature.  Therefore they belong to one
decision-sufficient class even if the state, query, or raw loss is not invariant.

### Proposition: gauge cancellation

Let `lambda_c` be posterior quotient masses and let the conditional gauge law
inside every class be arbitrary.  If the pairwise loss differences above are
constant on every orbit, then

\[
\mathbb E[L(G,a)-L(G,b)]
=
\sum_c \lambda_c
  \bigl(L_c(e,a)-L_c(e,b)\bigr),
\]

independently of every unresolved conditional distribution `p(g | c)`.
Consequently, the Bayes-action set and regret vector depend only on the quotient
posterior.  The group state may remain completely unidentified.

This is stronger than an invariant-query gate.  It can authorize a
symmetry-sensitive physical action without constructing a point state.

## Complete-orbit certificate from a finite cover

Suppose a finite set `{g_cs}` is a certified `rho_c`-net of each compact orbit.
For action pair `(a,b)`, assume a valid Lipschitz constant `K_cab` for

\[
D_c^{ab}(g)=L_c(g,a)-L_c(g,b).
\]

Then

\[
\max_s D_c^{ab}(g_{cs})
\leq
\sup_{g\in G}D_c^{ab}(g)
\leq
\max_s D_c^{ab}(g_{cs})+K_{cab}\rho_c.
\]

The posterior pairwise upper bound is

\[
\overline\Delta(a,b)=
\sum_c\lambda_c
\left[
\max_s D_c^{ab}(g_{cs})+K_{cab}\rho_c
\right],
\]

and the action-wise regret bound is

\[
\overline R(a)=\max_b\overline\Delta(a,b).
\]

An action is certified optimal when every pairwise upper bound in its row is
nonpositive.  It is uniformly `epsilon`-admissible when
`overline R(a) <= epsilon`.  If no minimax action meets the registered tolerance,
the implementation returns the caller-owned fallback.

When every `rho_c` is zero, the supplied finite group is exhaustive and the
certificate is exact.  For a uniform `S`-point `SO(2)` grid,

\[
\rho=\pi/S,
\]

so `S >= ceil(pi K / eta)` is sufficient to limit one pairwise cover margin to
`eta`.

## Why independent gauges are not conservative physics

A common approximation propagates each state point, query, or action under an
independent group draw.  That may preserve one-dimensional marginals while
removing exact cross-variable cancellation.  Its loss model is

\[
\widetilde L_c(g_x,g_a,a)
=
\ell(g_x\cdot x_c,\,g_a\cdot a),
\]

with `g_x` and `g_a` varied independently.  This defines a larger, physically
different ambiguity set.  It can make a perfectly determined relative action
appear unresolved.

The module includes this construction only as a negative control.  It is not a
valid substitute for a shared-gauge posterior unless the application really has
independent transforms.

## Operational contract

A gauge-coupled action certificate is meaningful only when the following items
are registered and independently checked:

1. **Group action on state.** The orbit represents a genuine unresolved physical
   equivalence, not arbitrary augmentation.
2. **Group action on action.** The actuator or command adapter exposes the
   corresponding transformation of an action template.
3. **Shared transform.** State and action use the same group element.  A content
   or provenance identifier should bind both paths.
4. **Loss semantics.** The registered physical loss is evaluated after the
   shared transformation.
5. **Continuous scope.** A finite sample is a complete group only when the group
   is finite.  A continuous claim needs a valid cover and pairwise Lipschitz
   bound, or an exact analytic extremum.
6. **Fallback ownership.** The caller supplies the complete fallback action;
   Prob4D does not synthesize one.

Malformed arrays, negative masses or radii, asymmetric pairwise Lipschitz
bounds, invalid metrics, and unsupported action indices fail closed.

## Controlled adversarial study

`scripts/science/run_gauge_coupled_action_study_v1.py` uses

\[
x(\theta)=R(\theta)[1,0]^\top
\]

and three action templates:

- the same shared-frame direction;
- the shared-frame orthogonal direction;
- the zero-vector fallback.

The state-coordinate orbit has diameter two, so the state is not identified.
The shared-gauge losses are nevertheless constant:

\[
L_{\rm track}=0,\qquad
L_{\rm orthogonal}=2,\qquad
L_{\rm fallback}=1.
\]

The first action has exact zero regret on every sampled and unsampled group
element.  In contrast:

- a world-frame point completion is worse than fallback on two thirds of the
  continuous orbit and has worst loss four rather than one;
- the independent-gauge control destroys the cancellation, cannot certify the
  useful action at zero tolerance, and returns fallback.

This is the intended strictness result:

\[
\text{state unidentified}
\quad\not\Rightarrow\quad
\text{action unidentified}.
\]

## Integration across the physical-twin stack

- **Prob4D** carries the quotient belief, conditional group law, shared-gauge
  query pushforward, and this action-difference certificate.
- **Causal4D** should register how commanded interventions transform under the
  group and bind the realized intervention to the same group provenance.
- **BayesianPhysTwin** should combine the group-coupled structural regret with
  finite-sample transport uncertainty and execute the exact caller-owned
  fallback when either the symmetry or regret contract fails.

The important interface object is not a point pose.  It is a tuple containing
quotient evidence, unresolved group support, an action transport, a physical
loss, and a verifiable shared-gauge receipt.

## Scientific boundary

This result does not discover the symmetry group, prove that a learned provider
is invariant, validate an actuator transform, estimate a Lipschitz constant,
turn a numerical quadrature into Haar integration, establish unseen-object
transport, or certify deployment safety.  Those require separate evidence.

The next claim-bearing experiment should use an object- or recording-disjoint
provider that emits a source-qualified group action and shared state/action
provenance.  It should compare the shared-gauge policy against MAP completion,
independent-gauge uncertainty, covariance inflation, invariant-only rejection,
and exact fallback on held physical outcomes.
