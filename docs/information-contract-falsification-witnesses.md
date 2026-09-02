# Source-selected falsification witnesses for 4-D information contracts

## Purpose

The information-contract benchmark reports accuracy, calibration, dependence,
query admissibility, decision regret, fallback, and communication separately.
A fixed scorecard can still miss a provider's weakest physical direction if the
registered queries happen not to align with it.

This module adds a source-only auditor. Inside a caller-registered linear query
span it selects the query with the largest ratio between empirical squared error
and reported uncertainty, freezes that query as a content-addressed witness, and
evaluates it on held independent groups without target-side reselection.

The intended scientific question is:

> Can a source-selected, physically registered query expose a held failure or a
> provider ranking reversal that average point error conceals?

The current implementation is a deterministic development mechanism. It does
not yet constitute a prospective public-provider result.

## Registered source problem

Let `e_i` be the residual vector for source case `i`, let `P_i` be the covariance
reported by the audited submission, and let `g(i)` identify an independent
physical object, session, or complete trajectory. The benchmark first constructs
equal-group moments

\[
S=\frac{1}{G}\sum_{g=1}^G
  \frac{1}{n_g}\sum_{i:g(i)=g}e_i e_i^\top,
\qquad
P=\frac{1}{G}\sum_{g=1}^G
  \frac{1}{n_g}\sum_{i:g(i)=g}P_i.
\]

Nested points, frames, and windows therefore cannot dominate the witness merely
because one trajectory contains more cases.

The auditor does not search every vector in state space. The challenge owner
registers a basis `B` before source outcomes are opened. Every admissible query
has the form

\[
q=Bz.
\]

The selected witness maximizes

\[
\kappa^\star
 =\max_{z\ne 0}
 \frac{z^\top B^\top S Bz}
      {z^\top B^\top P Bz}.
\]

This is the largest generalized eigenvalue of

\[
B^\top S Bz=\kappa B^\top P Bz.
\]

The implementation solves the symmetric whitened problem, checks the generalized
eigen residual, normalizes the state-space query to unit Euclidean norm, and
fixes its sign deterministically. It records the complete source payload identity,
query-family identity, coefficients, vector, source objective, per-group ratios,
and information-order declaration.

## Held evaluation

A held manifest binds two or more provider submissions to the exact witness ID.
Every submission must use the same case and group roster. The evaluator computes,
without changing the query:

- equal-group coordinate RMSE;
- equal-group selected-query RMSE;
- selected-query normalized squared error;
- absolute log calibration error;
- Gaussian NLL;
- nominal marginal coverage; and
- per-group query diagnostics.

It reports point-accuracy, query-calibration, and query-NLL orderings separately.
A ranking reversal is present when the point-accuracy winner differs from the
provider closest to unit normalized query error.

A retrospective target remains labelled `retrospective-diagnostic`. Only a
separately sealed target whose witness was frozen before target opening can be
labelled `prospective-held-confirmation`. The JSON declaration is necessary but
not sufficient for a claim: a claim-bearing release must also retain workflow
chronology and target-opening receipts.

## Controlled anti-gaming result

Run:

```bash
python -m prob4d.information_contract_witness smoke /tmp/witness-smoke
```

The deterministic fixture registers three orthogonal regional-displacement
coordinates. On four source groups, Provider A has the lower point error but
underreports variance by a factor of ten along the first coordinate. The source
auditor selects that coordinate exactly. On six held groups:

- Provider A remains the point-accuracy winner;
- Provider A has selected-query normalized error 10;
- Provider B has larger point RMSE but selected-query normalized error 1; and
- the point-accuracy and query-calibration rankings reverse.

This is a conformance control, not empirical provider evidence. Its purpose is to
ensure that the benchmark can express the paper-defining failure before public
targets are opened.

## Query-family governance

An unconstrained adversarial query is easy to make uninterpretable. A
claim-bearing query family should therefore freeze:

1. physical units and coordinate frame;
2. locality or support constraints;
3. invariance requirements, such as zero-sum translation-invariant weights;
4. coefficient or operator normalization;
5. source and target independent units;
6. provider and covariance identities;
7. the selection objective and deterministic tie rule; and
8. the complete target roster.

Examples include endpoint-relative displacement, regional centroid, span,
bending, clearance, and contact-local deformation. The present module accepts a
general linear basis but does not certify that the basis is physically meaningful.

## Claim boundary

The generalized-eigenvector solution is exact for the registered linear span and
the equal-source-group second-moment ratio. It is not an exhaustive adversary
over nonlinear tasks. Source maximization can overfit, so a held result requires
independent target objects or sessions. A ranking reversal does not establish
robot-action safety, causal validity, covariance calibration outside the suite,
or state of the art.

## Next empirical stage

The first bounded public-data use is a retrospective DEFORM DLO4/DLO5 adapter:

- build a source-only prediction model from frozen training roles;
- fit covariance on a disjoint calibration role;
- select a query from a disjoint source-test role;
- freeze the witness;
- evaluate it on complete held evaluation trajectories;
- compare a structured covariance with its same-marginal diagonal control; and
- retain the result as retrospective because those targets were opened in earlier
  work.

The claim-bearing promotion remains a prospective two-provider by two-dataset
release with source-selected witnesses frozen before held targets are opened.
