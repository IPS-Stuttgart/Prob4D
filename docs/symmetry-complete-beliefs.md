# Symmetry-complete Bayesian beliefs

## Scientific purpose

A physical observation can identify a quotient state while leaving a compact
symmetry group unresolved. Choosing one group representative in that situation
adds information that the observation did not provide. This module represents
the missing information explicitly instead of hiding it in a point estimate or
an inflated Gaussian covariance.

The operational object is a factorized belief

\[
  p(z,g)=\lambda_z\,\rho_z(g),
\]

where `z` is a finite quotient class and `g` is a declared compact-group
coordinate. `lambda` is the quotient belief and `rho_z` is the conditional law
along the unresolved group orbit. A finite quadrature carries the numerical law;
continuous-group query claims additionally require a certified metric cover and
Lipschitz bound.

This layer generalizes the existing axial `SO(2)` kernels in
`axial_gauge.py` and `axial_query_certificate.py`. Those modules retain useful
closed forms for rotations about one line. The new layer supplies the generic
Bayesian semantics needed to preserve an unresolved finite or continuous compact
group across quotient updates and vector-valued physical queries.

## 1. Symmetry-preserving Bayesian update

Let a compact group `G` act within every quotient class. Assume the prior has a
regular conditional factorization

\[
  p_0(z,g)=\lambda_0(z)\rho_0(g\mid z).
\]

If the observation likelihood is invariant on the complete group orbit,

\[
  L(y\mid z,g)=L(y\mid z),
\]

then Bayes' rule gives

\[
\begin{aligned}
  p(z,g\mid y)
  &\propto L(y\mid z)\lambda_0(z)\rho_0(g\mid z)\\
  &=\lambda_y(z)\rho_0(g\mid z).
\end{aligned}
\]

Therefore

\[
  \rho_y(g\mid z)=\rho_0(g\mid z).
\]

Orbit-invariant evidence may update quotient masses but must not alter the
conditional group law. `update_symmetry_complete_belief` enforces this
separation:

- finite groups are checked exhaustively over every supported node;
- continuous groups require both nodewise equality and a caller-supplied
  complete-group invariance certificate;
- non-invariant likelihoods are rejected in invariant mode;
- explicitly symmetry-breaking observations use a separate mode and may update
  the group conditional.

Equality at finitely many continuous-group quadrature nodes is not a proof of
whole-group invariance. The external certificate is a source/protocol obligation
and is not inferred from the flag itself.

## 2. Information decomposition

For any posterior that remains absolutely continuous with the prior,

\[
  D_{\mathrm{KL}}(p_1\Vert p_0)
  =D_{\mathrm{KL}}(\lambda_1\Vert\lambda_0)
   +\sum_z \lambda_1(z)
    D_{\mathrm{KL}}\!\left(
      \rho_1(\cdot\mid z)\Vert\rho_0(\cdot\mid z)
    \right).
\]

The implementation reports the first term as quotient information and the
second as gauge information. An admitted orbit-invariant update must have
exactly zero conditional-array change and zero gauge information, up to the
fail-closed numerical contract.

This makes a previously implicit distinction auditable:

- **supported information** changes the quotient belief;
- **symmetry-breaking information** changes a conditional group law under an
  explicit likelihood;
- **unsupported specificity** chooses a group representative without such
  evidence.

## 3. Why point completion is not benign

For a finite conditional group law, completing the belief to one supported node
adds

\[
  -\log \rho_0(g^\star\mid z)
\]

nats in class `z`. The total diagnostic is the quotient-weighted sum. Under a
uniform `K`-element group this is `log(K)`.

For a continuous conditional density, a Dirac point completion is singular with
respect to the original law and has infinite KL divergence. A finite quadrature
reports `log(K)` only as a resolution-dependent numerical diagnostic; it is not
a finite physical information claim. The audit therefore labels a continuous
point completion `continuous-singular` even when its selected quadrature node
has positive numerical mass.

A selected node outside positive conditional quadrature support is reported as
`outside-support`; no fictitious finite penalty is returned.

## 4. Shared-group query pushforward

A physical query often contains many points or coordinates controlled by one
shared latent gauge. For query atoms

\[
  q_{z,k}=q(z,g_k),
\]

`pushforward_shared_group_query` constructs the complete mixture with weights

\[
  w_{z,k}=\lambda_z\rho_z(g_k\mid z).
\]

Every coordinate in `q_{z,k}` uses the same group node. The implementation never
creates independent per-point gauge draws. That distinction is structural: two
points that are exact negatives under one shared rotation have perfect negative
cross-covariance and cancel exactly, whereas independent gauge draws destroy the
cancellation while preserving both marginal laws.

The returned `GaussianQueryMixture` retains the full discrete pushforward and an
optional shared additive readout covariance. Its moment methods are summaries of
the mixture, not a claim that the group-pushforward distribution is Gaussian.

## 5. Continuous compact-group query certificate

Let `S={g_k}` be a finite `rho`-net of a compact group under a declared metric.
For a vector query that is `L`-Lipschitz in that same metric, define the sampled
diameter

\[
  D_S=\max_{s,t\in S}\lVert q(s)-q(t)\rVert_2.
\]

For arbitrary `g,h`, choose nearest nodes `s,t`. The triangle inequality gives

\[
\begin{aligned}
  \lVert q(g)-q(h)\rVert_2
  &\leq \lVert q(g)-q(s)\rVert_2
       +\lVert q(s)-q(t)\rVert_2
       +\lVert q(t)-q(h)\rVert_2\\
  &\leq D_S+2L\rho.
\end{aligned}
\]

Consequently,

\[
  D_S\leq \operatorname{diam}q(G)\leq D_S+2L\rho.
\]

The certificate has three scientific outcomes plus one scope failure:

- `certified-invariant`: the upper bound is at most the registered tolerance;
- `certified-variant`: the sampled lower bound exceeds the tolerance;
- `undetermined`: the certified interval crosses the tolerance;
- `scope-not-certified`: the cover itself is not certified.

Only `certified-invariant` is admitted. `undetermined` and uncertified scope fail
closed. Exact finite groups are the special case `rho=0`.

For the circle group with uniformly spaced angles and wrapped angular distance,
the exact cover radius is `pi/K`. The controlled study verifies the certificate
against the analytic diameter `2 sigma_max(B)` for random vector first-harmonic
queries

\[
  q(\theta)=B[\cos\theta,\sin\theta]^T.
\]

## 6. Relationship to existing real-data evidence

The merged Tracking Cloth experiment already provides a real-trajectory failure
control for one continuous hidden `SO(2)` gauge:

- 15 support-qualified held recordings and 1,803 controlled geometry cases;
- the local gate admitted every radial query;
- approximately 65.1% of those local admissions were harmful;
- the complete-orbit gate rejected every radial query, admitted every invariant
  query, and reproduced the exact fallback on rejection.

That result validates the need for complete-orbit reasoning under a controlled
hidden gauge. It does not validate this generic compact-group posterior with a
learned visual provider. The new module is a method-level generalization and
must not relabel the existing cohort as fresh evidence.

The current uncertain-axis and approximate-orbit studies remain complementary:
they bound errors in a particular `SO(2)` orbit representation. This module
instead defines how a valid unresolved group law propagates through Bayesian
updates, information accounting, and joint query distributions.

## 7. Controlled verification

```bash
python -m pytest -q \
  tests/test_symmetry_complete_belief.py \
  tests/test_symmetry_complete_belief_study.py \
  tests/test_axial_gauge.py \
  tests/test_axial_query_certificate.py

python -m prob4d.symmetry_complete_belief_study \
  --seed 20260902 \
  --cases 512 \
  --output outputs/symmetry-complete-belief-v1/result.json
```

The study verifies:

- exact preservation of arbitrary conditional group laws under invariant
  evidence;
- zero invented gauge information and the KL chain rule;
- explicit group-law change under symmetry-breaking evidence;
- singular continuous point completion and its `log(K)` discretization ladder;
- the continuous metric-cover bound over random vector harmonic queries;
- exact shared-gauge cancellation versus a matched independent-gauge control.

It is algebraic evidence only. No dataset is opened.

## 8. Integration and claim boundary

Prob4D owns the quotient/group belief and query pushforward. BayesianPhysTwin may
consume the resulting complete mixture or a decision certificate, but must
retain its own exact complete-belief fallback. Causal4D may register which
interventions preserve or break the group symmetry; this module does not infer
that causal status.

The representation is exact only for the supplied finite quotient, numerical
group law, prior support, likelihood values, and declared symmetry semantics. A
continuous query certificate additionally requires a valid metric cover radius
and Lipschitz bound. The module does not:

- discover or validate a physical group action;
- establish that a likelihood is invariant merely because sampled values match;
- turn a quadrature into exact Haar integration for arbitrary functions;
- calibrate geometry, cover, or Lipschitz errors;
- validate a learned provider or source-to-target transport;
- identify the complete physical state;
- authorize deployment or certify safety.

The decisive next empirical step is an object- or recording-disjoint provider
that outputs a source-qualified group action and whole-group invariance receipt.
Its target evaluation should compare symmetry-complete propagation against MAP
representatives, independent per-point gauges, covariance inflation, and exact
fallback while keeping complete physical objects or recordings as statistical
units.
