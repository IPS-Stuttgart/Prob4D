# Recursive task-sufficient correlated-uncertainty compression

This note records an experimental extension of Prob4D's fixed-query posterior-preserving
shared-noise compression. It is a research direction, not a release claim.

## Motivation

The existing compressor preserves one registered Gaussian query posterior exactly, but it
explicitly does not preserve an arbitrary subsequent update. A current task can therefore be
posterior-exact while an omitted state function changes a later task through the dynamics.

The recursive extension separates two questions:

1. **Which state functions must survive so that the registered task can be filtered exactly over
   time?**
2. **Given that recursively sufficient task state, which directions of a supplied correlated
   measurement-noise factor are required for its posterior update?**

The first question is classical functional-observer / invariant-subspace territory. The second
uses Prob4D's existing posterior-preserving shared-factor theorem. The candidate contribution is
their explicit composition into an exact, fail-closed correlated-uncertainty interface for a
registered recursive task.

## Strict LTI task-state closure

Consider

\[
 x_{t+1}=F x_t+w_t, \qquad y_t=H x_t+v_t, \qquad q_t=Lx_t,
\]

with exogenous Gaussian process and measurement noises. Define the row space

\[
 \mathcal T = \operatorname{span}\{\operatorname{row}(L),
                                   \operatorname{row}(H),
                                   \mathcal T F\}.
\]

`recursive_linear_task_closure` computes the smallest fixed row space satisfying this closure by
starting from `row([L; H])` and repeatedly adding its pullback through `F` until the rank stops
growing. For an orthonormal row basis `T` of this space,

\[
 z_t=T x_t,
\]

there exist exact reduced matrices

\[
 L=B T, \qquad H=D T, \qquad T F=A T.
\]

Hence

\[
 z_{t+1}=A z_t+T w_t, \qquad y_t=D z_t+v_t, \qquad q_t=B z_t,
\]

is an exact linear-Gaussian task-state model. If the closure grows to the complete state, the
algorithm reports the complete state; it does not manufacture a lower-dimensional claim.

The minimum-space statement here is only within the stated fixed-row-space invariance
requirements. Functional observers, functional observability, reduced-order observers, and state
aggregation are established prior work and must be credited explicitly in any manuscript.

## Composition with correlated-noise compression

Suppose the innovation covariance at one update contains a supplied shared factor

\[
 S_t=A_t^{(y)}+U_tU_t^\top.
\]

For the recursively sufficient state `z_t`, the existing Prob4D result retains

\[
 \operatorname{range}
 \left(U_t^\top S_t^{-1}\operatorname{Cov}(z_t,y_t)^\top\right).
\]

The retained factor rank is therefore at most `dim(z_t)`. The factor replacement preserves the
posterior mean of `z_t` for every innovation and its posterior covariance. Because `z_t` itself is
closed under the registered dynamics and observation model, equal posteriors at step `t` imply
equal priors at step `t+1`; induction then yields equal registered task posteriors for the whole
filtering sequence.

This composition does **not** preserve the observation likelihood. It also does not establish
recursive exactness when the dynamics, observation map, robust weights, query family, or
linearization leave the registered closure. Such changes require rebuilding or enlarging the
closure and recomputing the factor projection.

## Decisive controlled mechanism

`tests/test_recursive_task_sufficiency.py` contains a deliberately adversarial 20-dimensional
linear-Gaussian example:

- the current physical task is three-dimensional;
- the exact recursive task state is four-dimensional because a fourth component drives a future
  task coordinate;
- the supplied correlated measurement-noise factor has rank eight;
- current-task compression retains rank three and is exact at the first update, but it alters the
  fourth state component and subsequently changes the task posterior;
- closure-aware compression retains rank four and matches the full recursive task-state filter to
  numerical precision for every tested update; and
- introducing one nuisance-to-task transition term expands the audited closure from four to five
  dimensions instead of silently accepting the old rank.

The important comparison is therefore not `full state` versus `small query` in one update. It is

\[
 \text{current query rank} < \text{recursive task-closure rank} \ll
 \text{state / measurement-history dimension},
\]

with exact fallback when the middle quantity expands.

## Evidence still needed for a large robotics claim

The controlled result establishes only the algebraic mechanism. A paper-level robotics claim
would additionally need a preregistered real or closed-loop study in which:

1. correlated pseudo-measurement rank grows with views, windows, or history;
2. the registered task closure remains substantially smaller;
3. the closure is chosen without target outcomes;
4. full and recursively compressed estimators produce the same task posterior / controller action
   within numerical tolerance; and
5. generic equal-rank truncation or current-query-only compression fails on the same protocol.

A changing query or action family is especially valuable because it removes the strongest current
objection: for one immutable query, directly caching its posterior message is simpler and faster.
