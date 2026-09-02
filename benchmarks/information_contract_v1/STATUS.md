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

## Strengthening in the sealed-submission branch

- Challenge-owned truth is physically separated from provider-owned output.
- The case roster, both payloads, and the dataset manifest are independently
  SHA-256 bound.
- Submissions cannot redefine target truth, query, nullspace, quotient, loss,
  realized outcome, or physical fallback.
- Challenge payloads cannot carry provider means, uncertainty, admissions,
  certificates, or selected actions.
- Retrospective and prospective information orders are distinct result classes;
  retrospective replay cannot claim prospective held-out status.
- A new `finite_query` task tests query constancy over each finite supported
  quotient class, complementing rather than replacing the local tangent-
  nullspace check.
- The scorecard explicitly counts local admissions that fail the finite test.
- Communication file size charges only provider-owned submission bytes, while
  retaining challenge and temporary joined sizes for audit.
- New adversarial tests cover target smuggling, provider smuggling, false finite
  admission, incomplete case rosters, false prospective relabelling, and byte-
  identical replay.

## Controlled findings

These are deterministic conformance checks, not empirical provider performance:

- the lowest-RMSE system has zero registered coverage and normalized NEES 450;
- full and diagonal covariance systems have identical means and coordinate
  marginals, but different joint likelihoods;
- the ambiguity-aware contract identifies the invariant query, rejects the
  sensitive query, and returns exact fallback;
- the sealed control contains a locally admitted query that is rejected by the
  finite quotient-class range test;
- an artificially refined quotient admits a harmful nonfallback action;
- the communication control reduces 17,000 bytes to 1,000 bytes with posterior
  error `1e-12` under tolerance `1e-9`; and
- two corroborating providers are both inaccurate and share one input dependence
  group.

## Not completed

- No new public dataset has been opened for this benchmark branch.
- No learned provider has been run.
- No public provider ranking exists.
- No uncertainty calibration, query validity, or decision utility is claimed on
  a new empirical cohort.
- No scalar overall score exists.

## Next bounded milestone

1. Generate a retrospective DEFORM DLO4/DLO5 adapter directly from the pinned
   public trajectories and the original per-decision evaluator. It must reproduce
   the retained result before it is used as an adapter reference.
2. Generate the corresponding Tracking Cloth dependence/compression adapter from
   original recording-level artifacts.
3. Freeze a prospective two-provider by two-dataset protocol using the separated
   challenge/submission format.
4. Seal provider manifests and predictions before held-out scoring.
5. Publish the first multi-axis public scorecard, including every positive,
   negative, support-negative, and technical-failure outcome.

A standalone benchmark manuscript is not promoted until the prospective stage
contains at least one statistically supported ranking reversal and the required
negative controls without target-side policy selection.
