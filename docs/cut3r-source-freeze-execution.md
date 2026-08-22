# Retained CUT3R source-freeze execution

This workflow provides the first write-free, protected execution step for the
registered Deform360 CUT3R comparison.

## Trigger and trust boundary

`.github/workflows/cut3r-source-freeze-execution.yml` has two paths:

- pull requests run hosted contract checks only;
- creation or modification of
  `protocols/execution_requests/cut3r_deform360_source_freeze_v1.json` on merged
  `main` authorizes one protected self-hosted source-freeze run.

The self-hosted job receives no write-capable repository token. It checks out
the exact merged `main` revision, verifies the content-addressed request and
source-protocol Git blob, builds one wheel from that revision, and uses only the
protected source-side paths already required by the reviewed manual workflow.

A separate GitHub-hosted reporting job may write one status comment to issue
#49. No write token is exposed to the self-hosted runner.

## Retained outputs

A support-positive execution uploads:

- `cut3r-deform360-source-freeze.json`;
- `cut3r-comparison-spec.json`;
- `cut3r-comparison-lock.json`;
- `cut3r-comparison-summary.json`;
- a sanitized log;
- exact environment and host summaries;
- `execution-summary.json`; and
- `SHA256SUMS`.

A support-negative execution retains the source-freeze decision and does not
create comparison outputs. Exit status 3 is treated as a valid scientific
negative rather than a retryable infrastructure failure.

## Information boundary

This workflow hashes source videos, sidecars, calibration, the CUT3R checkout and
checkpoint, and the reviewed Prob4D wheel. It does not decode source RGB frames,
run CUT3R, open provider predictions, inspect source residuals or truth, access
future geometry, open any confirmation or target payload, or inspect a
BayesianPhysTwin physical innovation.

After a support-positive artifact is independently inspected, its exact
source-freeze and comparison-lock JSON files must be committed and reviewed
before the three causal CUT3R arms are executed.
