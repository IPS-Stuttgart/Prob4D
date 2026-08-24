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

## Exact command

After the workflow is merged to `main`, only this exact comment by
`FlorianPfaff` on issue 49 is admitted:

```text
/prob4d-dispatch-cut3r-source-freeze-v3 8b923e8cd67ca65f09312cffe305e36852f36fbb 8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e
```

The GitHub-hosted dispatcher checks out the exact default-branch event revision,
replays the existing historical retry authorization, verifies the routed
retained-data/CUT3R runner labels, and resolves any already-created target retry
before dispatching.

The first v3 target execution, run `32764290533` at control-plane revision
`78a209c2b217c264ab8b7bebfcc42fe7cd7d2ebf`, was cancelled by its bounded
runner-acceptance watchdog before the retained job began. Retained job
`97550358844` has no steps and the run has zero artifacts. The dispatcher may
treat only this exact immutable state as a non-execution and issue one fresh
current-`main` dispatch of the same historical request. It re-reads and binds
the exact run, all seven job identities and outcomes, the empty retained-job
step list, and the zero-artifact inventory before doing so.

Any drift fails closed. Any other existing target run, or any second run created
after this exception is consumed, remains deduplicated. No new scientific
request, provider choice, data selection, or outcome access is authorized.

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
the routed runner and remains bounded by its runner-acceptance watchdog.

## Scientific boundary

This correction changes no provider, checkpoint, source cohort, camera panel,
prefix, support rule, comparison arm, covariance rule, target roster, or
analysis. It authorizes no source or target outcome. Scientific evidence begins
only when the separately protected target workflow emits its registered terminal
source-freeze artifact and decision.
