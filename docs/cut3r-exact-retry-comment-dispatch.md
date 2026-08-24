# Exact CUT3R retry from issue 49

This hosted control-plane workflow provides an independently observable trigger
for the already registered retained CUT3R source-freeze retry. It is intentionally
separate from retained-data execution and has no access to provider checkouts,
checkpoints, videos, predictions, residuals, truth, confirmation objects, target
objects, BayesianPhysTwin artifacts, or Causal4D artifacts.

## Exact command

After the workflow is present on `main`, only this exact comment on issue 49 is
admitted:

```text
/prob4d-dispatch-cut3r-source-freeze-v2 8b923e8cd67ca65f09312cffe305e36852f36fbb 8f3c9fba12f8a16895edce89d7a92e4806a43cb2f34b5a05faff71945809b63e
```

The comment author and workflow actor must both be `FlorianPfaff`. Near matches,
comments on other issues, edits, pull-request input, and comments from other
accounts produce no privileged job.

## Ordered checks

The GitHub-hosted job:

1. publishes an accepted-command receipt containing its own workflow run;
2. checks out the exact default-branch revision attached to the comment event;
3. proves full historical ancestry and byte-identical retained request/protocol
   content using `cut3r_execution_preflight.py authorize-retry`;
4. verifies the routed retained-data/CUT3R labels on the target workflow;
5. lists target-workflow `workflow_dispatch` runs created after the original
   dispatch-control merge;
6. resolves and publishes any existing retry rather than dispatching a duplicate;
7. otherwise verifies that historical run `32621813949` is terminal and
   unsuccessful at the retained-data job, then admits only the reviewed
   failure-evidence artifact `9532584642` with its exact name, byte size,
   historical run, branch, and revision binding;
8. dispatches the exact historical revision and request ID once; and
9. polls the Actions API until it can publish the concrete target run ID.

The concurrency group serializes exact command events. A repeated command
therefore resolves the already-created retry rather than creating another
scientific execution.

The admitted artifact contains only request verification, execution-environment,
and host records from the configuration failure on `workstation1`. Any additional,
renamed, resized, expired, or differently bound artifact keeps the retry closed.

## Scientific boundary

This workflow changes no provider, checkpoint, source group, camera panel,
prefix, support rule, comparison arm, uncertainty rule, target roster, or
analysis. A dispatch or resolver receipt is operational evidence only. The target
workflow still owns repository-variable checks, read-only self-hosted execution,
bounded runner acceptance, input custody, sanitization, immutable evidence, and
the terminal support decision.
