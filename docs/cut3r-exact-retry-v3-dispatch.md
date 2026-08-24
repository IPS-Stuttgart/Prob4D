# Exact retained CUT3R retry v3

The original retained CUT3R source-freeze run reached the trusted GPU runner but
failed before provider execution. Its registered terminal decision was
`workflow-failed-before-decision`: no source-freeze result, CUT3R prediction,
source residual, truth, confirmation payload, target payload, BayesianPhysTwin
result, or Causal4D result was produced.

The run nevertheless retained one small Actions artifact because the protected
workflow deliberately uploads diagnostics after a failed self-hosted job. The
artifact contains only environment, host, and request-verification records. A
blanket requirement that the old run have zero Actions artifacts therefore
rejected a scientifically empty failure for the wrong reason.

## Exact admitted diagnostic

The v3 dispatcher admits only this immutable artifact:

```text
artifact ID: 9532584642
name: cut3r-source-freeze-v2-failed-32621813949-2
size: 3106 bytes
SHA-256: a7805e079ccb367d56634c62bb91a79fdac71babaa69c233e485135b9243a0a0
run: 32621813949
historical revision: 8b923e8cd67ca65f09312cffe305e36852f36fbb
```

It also binds the failed retained-data job `97376973894` and the exact outcomes
of the workspace, checkout, wheel-build, source-freeze, upload, and cleanup
steps. Any additional artifact, result-like artifact, changed digest, changed
size, changed workflow/run/job identity, successful source-freeze job, or step
outcome drift fails closed.

This is narrower than allowing arbitrary failure artifacts. It is an exact
custody exception for one already inspected diagnostic bundle.

## Exact zero-execution queue-timeout replacement

The first v3 target retry, workflow run `32764290533`, passed its hosted contract,
exact historical authorization, repository-variable preflight, and queued-run
receipt, but no retained runner accepted its execution job before the bounded
20-minute watchdog fired. The workflow ended `cancelled` without a retained
execution step and produced zero Actions artifacts.

After the reviewed runner-routing repair, the dispatcher may replace this one
specific retry only after independently re-reading GitHub and verifying:

- run ID `32764290533`, the exact target workflow, `main`, `workflow_dispatch`,
  registered head revision, terminal `cancelled` state, and no other run identity;
- the exact seven-job roster and job IDs from the timeout execution;
- success of authorization, hosted contract, variable preflight, queued receipt,
  and terminal receipt;
- cancelled retained-execution job `97550358844` with zero executed steps;
- watchdog job `97550358906` whose bounded cancellation step completed
  successfully; and
- zero Actions artifacts.

Any drift fails closed. After a replacement is created, every newer target retry
is duplicate-protected regardless of whether it is queued, running, or complete.
This exception therefore cannot be used to repeat a run that reached retained
execution or produced scientific evidence.

## Exact command

After the workflow is merged to `main`, only this exact comment by
`FlorianPfaff` on issue 49 is admitted:

```text
/prob4d-dispatch-cut3r-source-freeze-v3 8b923e8cd67ca65f09312cffe305e36852f36fbb 8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e
```

The GitHub-hosted dispatcher checks out the exact default-branch event revision,
replays the existing historical retry authorization, verifies the exact
`host-workstation2` target binding together with the fail-closed
`workstation2`/Linux/X64/`nvidia-smi` runtime checks, and resolves any
already-created target retry before dispatching. Repeated commands cannot create
duplicate target executions.

## Next operational gate

The current target workflow performs a hosted set/unset preflight for:

```text
BPT_CHECKOUT
CUT3R_CHECKOUT
CUT3R_CHECKPOINT
DEFORM360_PROCESSED_ROOT
```

Only their names and readiness state may be published. If any variable remains
unset, the target workflow stops before queueing the self-hosted job and opens no
retained input. If all are configured, the exact historical request is queued on
the exact retained runner and remains bounded by its runner-acceptance watchdog.

## Scientific boundary

This correction changes no provider, checkpoint, source cohort, camera panel,
prefix, support rule, comparison arm, covariance rule, target roster, or
analysis. It authorizes no source or target outcome. Scientific evidence begins
only when the separately protected target workflow emits its registered terminal
source-freeze artifact and decision.
