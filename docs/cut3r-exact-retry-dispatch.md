# Exact retained CUT3R source-freeze retry dispatch

This one-shot hosted control-plane workflow replaces a stale queued execution with
the already registered exact historical retry. It exists to make the retained
source-freeze request executable through the routed runner labels introduced on
`main` without changing any scientific input.

The merge-triggered request is:

```text
protocols/dispatch_requests/cut3r_deform360_source_freeze_exact_retry_v1.json
```

It binds:

- historical execution revision
  `8b923e8cd67ca65f09312cffe305e36852f36fbb`;
- retained request ID
  `8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e`;
- stale workflow run `32621813949`;
- the exact target workflow and `main` ref; and
- the routed runner labels `data-prob4d-deform360-source-v1` and
  `prob4d-cut3r` in addition to the standard self-hosted GPU labels.

## Execution order

On pull requests, the workflow validates only the request, workflow policy,
action pins, and exact historical retry authorization. It performs no Actions
mutation.

On the ordinary merged-main push that first adds the request, the hosted dispatch
job:

1. proves that the request file changed on a non-forced push to `main`;
2. replays `cut3r_execution_preflight.py authorize-retry`, including ancestry,
   byte-identical request, and source-protocol checks;
3. verifies that the target workflow contains the routed retained-data and CUT3R
   capability labels;
4. verifies that stale run `32621813949` is the expected historical push run;
5. refuses to continue if its source-freeze job succeeded or if the run retained
   any Actions artifact;
6. cancels the stale active queue and waits for a terminal zero-evidence state;
7. dispatches `cut3r-source-freeze-auto-v2.yml` from current `main` with the exact
   historical revision and retained request ID; and
8. posts a target-closed operational receipt to issue 49.

The dispatched workflow independently repeats the historical authorization,
checks repository-variable presence without publishing values, uses a read-only
token on the self-hosted job, and cancels itself if no matching runner accepts it
within the bounded queue interval.

## Scientific boundary

This dispatch changes no provider, checkpoint, source cohort, camera panel,
prefix, support criterion, comparison arm, target roster, calibration rule, or
analysis. The helper has no retained filesystem path, provider input, model
checkpoint, source residual, truth, confirmation payload, or target outcome.

A successful dispatch is operational evidence only. Scientific evidence begins
only if the separately authorized source-freeze workflow emits its registered
terminal artifact. The ordered support, means, identity, gauge/dependence,
linearization, covariance, and physical-query gates remain unchanged.
