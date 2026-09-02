# Status: information-contract benchmark v1

## Completed in this branch

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
- Retained `controlled_report.json`, `validation_receipt.json`, and hashes.
- Staged public-data promotion plan and paper novelty-ownership issue.

## Controlled findings

These findings are deterministic conformance checks, not empirical provider
performance:

- the lowest-RMSE system has zero registered coverage and normalized NEES 450;
- full and diagonal covariance systems have identical means and coordinate
  marginals, but different joint likelihoods;
- the ambiguity-aware contract identifies the invariant query, rejects the
  sensitive query, and returns exact fallback;
- an artificially refined quotient admits a harmful nonfallback action;
- the communication control reduces 17,000 bytes to 1,000 bytes with posterior
  error `1e-12` under tolerance `1e-9`; and
- two corroborating providers are both inaccurate and share one input dependence
  group.

## Not completed

- No public dataset has been opened for this benchmark branch.
- No learned provider has been run.
- No public provider ranking exists.
- No uncertainty calibration, query validity, or decision utility is claimed on
  a new empirical cohort.
- No scalar overall score exists.

## Next bounded milestone

1. Implement byte-parity retrospective adapters for the already retained
   Tracking Cloth communication/dependence result and DEFORM decision/fallback
   result. This remains adapter validation and does not create new novelty.
2. Freeze a prospective two-provider by two-dataset protocol.
3. Seal provider manifests and predictions before held-out scoring.
4. Publish the first multi-axis public scorecard, including every positive,
   negative, support-negative, and technical-failure outcome.

A standalone benchmark manuscript is not promoted until the prospective stage
contains at least one statistically supported ranking reversal and the required
negative controls without target-side policy selection.
