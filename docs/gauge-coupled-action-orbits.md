# Gauge-coupled action orbits under unresolved symmetry

## Purpose

A symmetry-complete belief can leave a group coordinate unresolved without
forcing every downstream robot decision to abstain. The important distinction is
between:

- a **fixed-frame action**, whose physical meaning changes across compatible
  group representatives; and
- a **gauge-coupled action template**, whose command is transformed by the same
  group element as the state representation.

This module certifies the second object. It does not estimate the group element
and it does not select one state representative.

## Decision-equivariance theorem

Let `c` index a finite quotient class, let `g` be an unresolved compact-group
coordinate, and let `alpha_a` denote action template `a`. The caller supplies a
coupled state/action construction

\[
  x_c(g), \qquad u_a(g),
\]

where the same `g` is used in both branches. Define the pairwise action-loss
difference

\[
  d_{c,a,b}(g)
  = \ell\!\left(x_c(g),u_a(g)\right)
    -\ell\!\left(x_c(g),u_b(g)\right).
\]

Assume that every pairwise difference is invariant over the complete group:

\[
  d_{c,a,b}(g)=d_{c,a,b}
  \quad \text{for all }g.
\]

For any conditional group laws `rho_c` and fixed quotient masses `lambda_c`,

\[
\begin{aligned}
  &\mathbb E_{c\sim\lambda,\,g\sim\rho_c}
  \left[
    \ell\!\left(x_c(g),u_a(g)\right)
    -\ell\!\left(x_c(g),u_b(g)\right)
  \right]\\
  &\qquad = \sum_c \lambda_c d_{c,a,b}.
\end{aligned}
\]

Thus every Bayes action comparison and every action regret depends only on the
quotient masses. The complete conditional group posterior can remain unresolved.

Joint invariance of the absolute loss,

\[
  \ell(gx,gu)=\ell(x,u),
\]

is sufficient but not necessary. The theorem still applies when

\[
  \ell\!\left(x_c(g),u_a(g)\right)
  = r_{c,a}+\kappa_c(g),
\]

because the action-independent gauge term `kappa_c(g)` cancels from every
pairwise comparison. This is why the pairwise certificate can be strictly tighter
than applying separate Lipschitz bounds to every absolute action loss.

## Exact finite and continuous scope

`certify_gauge_coupled_action_orbit` receives sampled coupled losses with shape

```text
(quotient_count, group_node_count, action_count)
```

and verifies that sampled pairwise differences are invariant up to a registered
numerical tolerance.

For a finite group, the supplied nodes must be a certified zero-radius exhaustive
quadrature. For a continuous group, equality at finitely many nodes is not enough:
the caller must separately certify pairwise-difference invariance on the complete
group. Missing scope fails closed.

The returned lower and upper pairwise gaps retain the sampled numerical range
rather than silently replacing it by one point value. The action is admissible
only when its upper regret is within the registered tolerance.

## Shared-gauge execution receipt

The theorem is operational only when state and action use one shared group
realization. `GaugeCouplingReceiptV1` therefore binds:

- the group identifier;
- a state-orbit identifier;
- an action-orbit identifier;
- a coupling identifier;
- certification that state and action use the same group element; and
- certification that the deployed command generator preserves that coupling.

The receipt is caller-owned provenance. Prob4D validates its internal consistency
but does not infer or prove that a robot controller actually satisfies it.

This distinction matters. Replacing one shared latent group element by independent
state and action draws can preserve both marginals while changing the task loss
and the optimal decision.

## Strict controlled example

The controlled study uses two quotient states on the unit circle with masses
`0.75` and `0.25`, and three action templates. State and action are co-rotated by
one unresolved `SO(2)` element. The absolute losses also contain a common term

\[
  \kappa(\theta)=2+2\cos(3\theta),
\]

whose range is four. Absolute losses are therefore strongly gauge variant, while
all pairwise action-loss differences are invariant.

The coupled expected losses are

```text
[3, 5, 4]
```

and uniquely identify action template zero with regret vector

```text
[0, 2, 1].
```

Two matched controls destroy that result:

- fixed world-frame actions have expected losses `[4, 4, 4]`; and
- independently drawn state/action gauges also have expected losses `[4, 4, 4]`.

The controls have the same state and action marginals as the shared-gauge
construction. Only their dependence differs.

On an eight-node circle cover, the generic action-wise Lipschitz certificate adds
`(L_a+L_b) rho` to every comparison and remains `undetermined`, because it charges
the common gauge term twice. The pairwise decision-equivariance certificate
cancels that nuisance term and certifies the unique action orbit with zero regret.

## Robotics interpretation

A valid coupling can arise when the command is defined in an object-attached,
sensor-attached, or otherwise shared equivariant frame. Examples include:

- moving along a direction defined by two observed object anchors;
- selecting a grasp or pull template expressed in a shared local frame;
- executing an action field that co-transforms with a reconstructed object orbit;
- changing coordinates in both the world model and controller without changing
the physical command.

A latent physical ambiguity does not automatically provide such an execution
binding. If the actuator requires an unknown fixed world direction, the receipt
must remain uncertified and the policy falls back.

## Relationship to equivariant policy learning

Equivariant policies, value functions, and diffusion policies are established
methods. This contribution does not claim that co-transforming actions is new.
The narrower contribution is an auditable Bayesian decision interface that:

1. preserves the unresolved group posterior rather than canonicalizing it;
2. distinguishes fixed-frame actions from shared-gauge action orbits;
3. exploits invariance of action-loss differences rather than requiring
   invariance of every absolute loss;
4. proves that the selected action template is independent of every compatible
   conditional group law; and
5. fails closed unless a state--action execution-coupling receipt and complete-
   group invariance receipt are present.

## Empirical promotion path

The current controlled panel executes no robot command. A claim-bearing public-
data study should construct object-attached action templates from independently
registered anchors and compare:

- the shared-gauge action orbit;
- a MAP/canonical representative action;
- fixed world-frame actions;
- independent matched-marginal gauges;
- the generic action-wise Lipschitz certificate; and
- exact fallback.

Tracking Cloth can provide the first controlled real-geometry demonstration.
Deform360 or PokeFlex can provide stronger object-disjoint action evidence when a
causal prefix or matched-reset protocol makes the candidate action outcomes
observable. Complete objects or reset episodes—not frames—must remain the
statistical units.

## Claim boundary

The certificate proves a decision statement only under the supplied group,
quotient, coupled loss, complete-group pairwise-difference invariance, and
execution-coupling receipt. It does not discover a symmetry, validate the physical
loss, infer the execution binding, recover the latent group coordinate, establish
counterfactual outcomes for unexecuted commands, authorize deployment, or certify
safety.
