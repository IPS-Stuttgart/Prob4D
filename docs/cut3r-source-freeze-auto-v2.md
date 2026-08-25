# Automatic retained CUT3R source freeze v2

This execution route runs the already registered Deform360 source-input freeze
from reviewed `main`. The scientific inputs are unchanged. It supports both the
original content-addressed push trigger and an exact manual retry of a cancelled
historical merged-main execution.

## Authorization

The retained request is

```text
protocols/execution_requests/cut3r_deform360_source_freeze_v2.json
```

### Original push route

An ordinary, non-forced push to `refs/heads/main` is accepted only when that push
changes the request file. The workflow verifies its content identity, replays the
exact source-protocol Git blob, and binds the resulting merged revision by full
SHA.

### Exact historical retry

A maintainer may dispatch the workflow from `main` with:

- the complete 40-character SHA of the historical merged-main execution; and
- the complete 64-character retained request ID.

The hosted authorization job proves that the historical revision is an ancestor
of current `main`, that its request blob is byte-identical to the current retained
request, that the request remains content-addressed and target-closed, that the
historical source-protocol blob still matches the request, and that the reviewed
execution driver exists at that revision. The self-hosted job keeps the current
merged-main checkout as a read-only control plane and materializes the authorized
historical revision in an isolated detached worktree. It builds the scientific
wheel from that historical worktree and validates the historical request and
protocol there. Only the current control-plane driver performs custody and
publication orchestration; it receives no permission to change scientific input
bytes.

This separation permits a reviewed post-execution custody repair without
silently advancing the scientific implementation. The retry does not create a
new request or change the cohort, provider, checkpoint, prefix, support rule,
package wheel, or information order. Both the control-plane and historical
execution SHAs are recorded in retained evidence.

## Hosted configuration preflight

Before a self-hosted job is queued, a hosted job checks that these repository
variable names are configured:

```text
BPT_CHECKOUT
CUT3R_CHECKOUT
CUT3R_CHECKPOINT
DEFORM360_PROCESSED_ROOT
```

Only set/unset state is evaluated. Variable values and retained filesystem paths
are never written to the report, logs, job summary, or issue comment. An
incomplete configuration fails closed before the self-hosted job is scheduled.
Actual path type, repository, checkpoint, and retained-input verification remains
inside the read-only self-hosted job.

## Bounded runner queue

A hosted watchdog monitors the workflow's self-hosted execution job through the
GitHub Actions jobs API. If no runner accepts the job within 20 minutes, the
watchdog first publishes a target-closed timeout receipt to issue 49 and then
cancels the workflow run. This prevents an unavailable runner from leaving the
scientific request silently queued for many hours. It does not inspect runner
host details, retained paths, provider inputs, or outcomes.

The exact retry may be dispatched again after a matching
`[self-hosted, Linux, X64, nvidia-smi]` runner is online. The same request ID and
historical execution SHA must be supplied.

## Execution

The self-hosted job has `contents: read` only. It receives no pull-request fields,
no writable repository token, and no secret-valued command input. The workflow:

1. materializes the exact authorized revision in an isolated detached worktree;
2. builds one wheel from that historical scientific revision;
3. installs that wheel in an isolated environment;
4. verifies the retained CUT3R checkout, checkpoint, Deform360 processed source
   root, BayesianPhysTwin selection lock, protocol, wheel, and input sidecars;
5. executes `build_cut3r_deform360_source_freeze.py`;
6. retains either `source-support-freeze-ready` or the registered
   `insufficient-common-camera-support` negative;
7. creates and verifies the immutable CUT3R comparison lock only after a support
   pass;
8. sanitizes protected filesystem paths before publication;
9. uploads checksummed evidence; and
10. posts authorization, blocker, timeout, and terminal receipts to issue 49 from
   hosted jobs.

The v2 request supersedes the earlier request only because that run produced no
observable terminal receipt. It does not change the ten source groups, twelve
forbidden confirmation groups, CUT3R revision, checkpoint rule, camera-panel
policy, prefix, evaluation interval, or comparison arms.

## Boundary

This stage does not execute CUT3R, decode RGB frames, open source residuals or
truth, fit uncertainty, open confirmation or target payloads, run
BayesianPhysTwin, or run Causal4D. A support-positive receipt authorizes only
publication of the exact source-freeze and comparison-lock bytes. A
support-negative receipt is the terminal result for this frozen source design.
Configuration and runner-timeout receipts are operational negatives with zero
scientific evidence.
