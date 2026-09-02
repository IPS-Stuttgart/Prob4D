# Status: information-contract benchmark v1

## Completed in the benchmark branch

- Provider-neutral JSON evaluator and deterministic command-line interface.
- Versioned protocol and formal self-contained suite JSON Schema.
- Seven-system controlled conformance suite.
- Separate accuracy, joint-probabilistic, query/decision, fallback,
  communication, and cross-provider axes.
- Exact finite-support pairwise gaps and worst-case regret recomputation.
- Equal-independent-unit aggregation plus descriptive pooled RMSE.
- Missing-covariance `not_evaluated` semantics.
- Shared-common-bias, dependence-destruction, unsupported-specificity, and
  malformed-contract controls.
- Focused Ruff/pytest/deterministic-generation workflows.
- Retained `controlled_report.json`, `validation_receipt.json`, and hashes.
- Staged public-data promotion plan and paper novelty-ownership issue.

## Further hardening in the sealed-submission branch

The benchmark now has a truth-separated public-evaluation interface:

- challenge-owned truth, queries, ambiguity sets, losses, and fallback;
- submission-owned means, covariance, admissions, certificates, and actions;
- exact case-roster equality and SHA-256 binding on both sides;
- rejection of target truth or realized losses smuggled into a submission;
- explicit retrospective versus prospective information-order classes;
- producer, model, calibration, frame, causal-cutoff, and dependence identities;
- provider-only communication accounting rather than charging truth bytes; and
- an exact finite-query axis over quotient classes in addition to the local
  linear nullspace check.

The deterministic control includes a query that the local differential test
admits but the finite quotient rejects. Both decisions are scored separately;
the benchmark therefore does not allow local stationarity to masquerade as
global query identification.

See
[`docs/information-contract-sealed-submissions.md`](../../docs/information-contract-sealed-submissions.md).

## Controlled findings

These findings are deterministic conformance checks, not empirical provider
performance:

- the lowest-RMSE system has zero registered coverage and normalized NEES 450;
- full and diagonal covariance systems have identical means and coordinate
  marginals, but different joint likelihoods;
- the ambiguity-aware contract identifies the invariant query, rejects the
  sensitive query, and returns exact fallback;
- a local nullspace check can admit a query that varies inside a finite
  observation-equivalence class;
- an artificially refined quotient admits a harmful nonfallback action;
- the communication control reduces 17,000 bytes to 1,000 bytes with posterior
  error `1e-12` under tolerance `1e-9`; and
- two corroborating providers are both inaccurate and share one input
  dependence group.

## Not completed

- No public dataset has been opened by the benchmark implementation branches.
- No learned provider has been run for a new benchmark result.
- No public provider ranking exists.
- No uncertainty calibration, query validity, or decision utility is claimed on
  a new empirical cohort.
- No scalar overall score exists.
- The sealed evaluator validates declarations and hashes; an external workflow
  receipt is still required to prove prospective information order.

## Next bounded milestone

1. Implement a byte-parity retrospective DEFORM DLO4/DLO5 adapter from the
   original public trajectories and frozen BayesianPhysTwin experiment code.
   Challenge truth/losses and provider decisions must be stored separately.
2. Implement the Tracking Cloth dependence/compression adapter from the original
   per-window artifacts, grouped by recording or specimen.
3. Compare at least two provider contracts on at least two independently
   collected public datasets.
4. Freeze a prospective challenge before target opening and retain every
   positive, negative, support-negative, and technical-failure outcome.

A standalone benchmark manuscript is not promoted until the prospective stage
contains at least one statistically supported ranking reversal and the required
negative controls without target-side policy selection.
