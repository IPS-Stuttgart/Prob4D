# Status: information-contract benchmark v1

## Completed in the benchmark line

- Provider-neutral JSON evaluator and deterministic command-line interface.
- Versioned protocol and formal suite JSON Schema.
- Seven-system controlled conformance suite.
- Separate accuracy, joint-probabilistic, query/decision, fallback,
  communication, and cross-provider axes.
- Exact finite-support pairwise gaps and worst-case regret recomputation.
- Equal-independent-unit aggregation plus descriptive pooled RMSE.
- Missing-covariance `not_evaluated` semantics.
- Shared-common-bias, dependence-destruction, unsupported-specificity, and
  malformed-contract controls.
- Focused Ruff/pytest/deterministic-generation workflows.
- Retained controlled report, validation receipt, and hashes.

## Strengthening in the sealed and public-data branches

- Challenge-owned truth is physically separated from provider-owned output.
- The case roster, both payloads, and dataset manifest are independently
  SHA-256 bound.
- Submissions cannot redefine target truth, query, nullspace, quotient, loss,
  realized outcome, or physical fallback.
- Retrospective and prospective information orders are distinct result classes.
- A finite-query task complements the local tangent-nullspace check.
- Communication accounting charges provider-owned bytes rather than joined
  truth-plus-submission bytes.
- The public DEFORM DLO4/DLO5 adapter reconstructs 532 truth-separated cases
  nested in 28 complete held trajectories and reproduces the retained result.

## Source-selected falsification witnesses

The stacked witness branch searches a registered linear physical-query span on
source groups only. It solves the generalized eigenproblem for the largest
equal-group empirical-error to reported-variance ratio, freezes the selected
query as a content-addressed witness, and evaluates that exact query on held
submissions without target-side reselection.

The deterministic control establishes the benchmark-defining possibility:

- Provider A has lower point RMSE;
- the source-only auditor selects the exact underreported direction;
- on held groups Provider A has normalized query error 10;
- Provider B has higher point RMSE but normalized query error 1; and
- point-accuracy and physical-query calibration rankings reverse.

## Retrospective public DEFORM results

Two source-selected diagnostics have completed on the official DLO4/DLO5
release. In each, 80 training trajectories fit source quantities, 32 disjoint
training trajectories select the query, and 28 complete evaluation trajectories
are scored without target query reselection.

### Spatial overconfidence witness

A frozen metric 3-D direction remains strongly overconfident on the held
trajectories: normalized query error is `18.2447` and nominal-90% coverage is
`55.47%`, versus `0.9122` and `91.26%` for the corresponding full covariance.
The registered method pair did not exhibit a point/query ranking reversal; that
negative fact is retained.

### Dependence-sensitive trajectory witness

A 12-D query combines terminal and horizon-average centroid, terminal half-span,
and temporal centroid change. Full and diagonal submissions have identical
means, residuals, coordinate RMSE, and coordinate marginal variances; only
off-diagonal dependence differs.

On the frozen held query:

- full dependence: normalized query error `1.0193`, 90% coverage `89.85%`,
  query NLL `-1.6180`;
- marginal-matched diagonal: normalized query error `3.3831`, coverage `67.48%`,
  query NLL `-1.0318`;
- full-dependence NLL gain: `0.5862` nats per query case.

A secondary complete-trajectory analysis gives 21/28 NLL wins, a paired mean
NLL gain of `0.5862` with bootstrap interval `[0.3298, 0.8709]`, and an exact
paired sign-test p-value of `0.01254`. This analysis was added after inspection
of the retrospective aggregate and remains explicitly post-hoc.

Compact evidence and immutable artifact provenance are retained under:

```text
evidence/information-contract-deform-falsification-witness-v1/
evidence/information-contract-deform-dependence-witness-v1/
```

## Controlled findings

These are deterministic conformance checks, not empirical provider performance:

- the lowest-RMSE system has zero registered coverage and normalized NEES 450;
- full and diagonal covariance systems can share means and coordinate marginals
  but differ in joint likelihood;
- a locally admitted query can fail finite quotient-class invariance;
- an artificially refined quotient can admit a harmful nonfallback action;
- query-sufficient communication can reduce 17,000 bytes to 1,000 bytes with
  posterior error `1e-12`; and
- corroborating providers can both be wrong under shared common bias.

## Not completed

- No prospective public target has been opened for witness confirmation.
- No learned-provider ranking is claimed.
- The query basis is registered by the auditor; physical semantics are not
  inferred automatically.
- No target-domain safety or state-of-the-art claim exists.
- No scalar overall score exists.

## Next bounded milestone

1. Generate the corresponding Tracking Cloth dependence/compression witness
   from original recording-level artifacts.
2. Freeze a prospective two-provider by two-dataset protocol using separated
   challenge/submission payloads and source-only witness selection.
3. Require a non-artificial held ranking reversal or a statistically supported
   contract distinction under tied point accuracy.
4. Publish every positive, negative, support-negative, and technical-failure
   outcome without replacement.

A standalone benchmark manuscript is not promoted until a prospectively frozen
witness produces a statistically supported held result and the required
dependence, ambiguity, common-bias, and fallback controls all remain visible.
