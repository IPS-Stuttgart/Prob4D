# Finite-action certificates over unresolved compact groups

## Purpose

A symmetry-complete physical belief can remain deliberately noncommittal about a
compact-group coordinate while a downstream finite action is already uniformly
good. This module connects the compact-group posterior to the decision-
identifiability interface without selecting a gauge representative.

The caller supplies:

- posterior masses `lambda_c` for a finite quotient;
- a declared compact-group domain within every quotient class;
- a finite metric cover `S_c` with radius `rho_c`;
- registered losses `ell(c, g_k, a)` for every sampled group node and action;
- certified Lipschitz bounds `L_c,a` for each action loss in the same metric;
- a regret tolerance `epsilon`.

The result bounds regret over **every conditional group completion** compatible
with the declared quotient masses, not only over the numerical posterior weights
stored for moment propagation.

## Exact product-simplex adversary

Let `q_c` be an arbitrary probability law over the group coordinate in quotient
class `c`. The complete-belief ambiguity set is

\[
  \mathcal Q(\lambda)
  =\left\{q(c,g)=\lambda_c q_c(g):q_c\in\Delta(G_c)\right\}.
\]

For actions `a` and `b`, define

\[
  d_{c,a,b}(g)=\ell(c,g,a)-\ell(c,g,b).
\]

Because every class conditional can concentrate independently on its own
maximizer,

\[
\begin{aligned}
  \sup_{q\in\mathcal Q(\lambda)}
  \mathbb E_q[d_{C,a,b}(G)]
  &=\sum_c\lambda_c\sup_{g\in G_c}d_{c,a,b}(g).
\end{aligned}
\]

The robust regret of action `a` is therefore

\[
  R(a)=\max_b\sum_c\lambda_c\sup_{g\in G_c}d_{c,a,b}(g).
\]

This is the compact-group analogue of the finite query-quotient certificate in
BayesianPhysTwin. It does not require the conditional group probabilities and it
does not infer a point representative.

## Certified finite-cover bounds

Suppose `S_c` is a certified `rho_c`-net under the declared group metric. If
`ell(c, ·, a)` is `L_c,a`-Lipschitz, then

\[
  d_{c,a,b}
  \text{ is }(L_{c,a}+L_{c,b})\text{-Lipschitz}.
\]

Let

\[
  m_{c,a,b}
  =\max_{s\in S_c}d_{c,a,b}(s).
\]

For any group element, its nearest cover node is at most `rho_c` away, hence

\[
  m_{c,a,b}
  \leq \sup_{g\in G_c}d_{c,a,b}(g)
  \leq m_{c,a,b}+(L_{c,a}+L_{c,b})\rho_c.
\]

After quotient weighting,

\[
\begin{aligned}
  \underline\Delta(a,b)
  &=\sum_c\lambda_c m_{c,a,b},\\
  \overline\Delta(a,b)
  &=\underline\Delta(a,b)
    +\sum_c\lambda_c(L_{c,a}+L_{c,b})\rho_c.
\end{aligned}
\]

Thus

\[
  \underline R(a)=\max_b\underline\Delta(a,b),
  \qquad
  \overline R(a)=\max_b\overline\Delta(a,b)
\]

satisfy

\[
  \underline R(a)\leq R(a)\leq\overline R(a).
\]

Consequences:

- if `upper_regret[a] <= epsilon`, action `a` is uniformly
  `epsilon`-optimal for every declared group completion;
- if `lower_regret[a] > epsilon` for every action, the certificate proves that
  no registered action meets the tolerance;
- otherwise the finite cover is inconclusive and the caller must fall back;
- for an exact finite group, `rho_c=0` and the lower and upper certificates are
  identical.

## Fail-closed scope

`certify_compact_group_decision` reports one of four outcomes:

- `certified-admissible`;
- `certified-no-admissible-action`;
- `undetermined`;
- `scope-not-certified`.

Only `certified-admissible` can expose an admissible action. All other outcomes
require the caller-owned complete fallback.

The scope checks are intentionally strict:

- a custom cover radius is uncertified unless explicitly accompanied by a cover
  certificate;
- a positive Lipschitz correction is uncertified unless its Lipschitz bound is
  explicitly certified;
- sampled equality is not a continuous-group certificate;
- malformed losses, dimensions, radii, tolerances, or booleans fail closed.

The reported arrays are immutable and retain sampled lower bounds separately
from certified upper bounds.

## Controlled analytic verification

The deterministic study draws vector-valued first-harmonic action losses

\[
  \ell_{c,a}(\theta)
  =\beta_{c,a}+u_{c,a}\cos\theta+v_{c,a}\sin\theta.
\]

Their exact pairwise orbit supremum is available analytically:

\[
  \sup_\theta [\ell_{c,a}(\theta)-\ell_{c,b}(\theta)]
  =\beta_{c,a}-\beta_{c,b}
   +\sqrt{(u_{c,a}-u_{c,b})^2+(v_{c,a}-v_{c,b})^2}.
\]

The study checks across a circle-resolution ladder that:

- every sampled pairwise gap and regret is a valid lower bound;
- every certified upper gap and regret covers the analytic value;
- the selected action's analytic regret is below its reported upper bound;
- no action is falsely admitted;
- the cover correction decreases as the certified circle cover is refined;
- two executions with the same seed produce byte-identical evidence.

This is an algebraic verification of the certificate, not empirical validation
of a physical symmetry or learned provider.

## Non-free group actions and stabilizers

The v1 object is a probability law over a **declared numerical group
coordinate**. A physical group action need not be free: a stabilizer

\[
  H_x=\{g:g\cdot x=x\}
\]

can make several group coordinates represent the same physical state. This does
not invalidate query pushforwards or robust decision bounds—duplicate physical
states simply yield duplicate query or loss atoms. It does affect the
interpretation of point-completion information:

- without further structure, the reported finite-group point specificity is
  specificity in the supplied group coordinate;
- interpreting it as physical-state specificity requires an injective/free
  orbit chart or an explicit homogeneous-space reduction `G/H_x`;
- varying stabilizers, orbit singularities, and quotient topology changes are
  outside the v1 contract and must not be hidden by relabeling group nodes.

## Integration boundary

Prob4D owns the unresolved group law, query pushforward, and these structural
regret bounds. BayesianPhysTwin may combine the resulting certificate with its
registered action semantics and exact fallback. Causal4D may register which
observations or interventions preserve or break the symmetry.

The method does not establish that the group action, losses, cover, Lipschitz
constants, quotient masses, or transport assumptions are physically correct. It
does not calibrate target-domain regret, validate a learned provider, identify a
unique state, authorize deployment, or certify safety.
