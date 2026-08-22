# Automatic retained CUT3R source freeze v2

This execution route runs the already registered Deform360 source-input freeze from
one reviewed push to `main`. It exists because the linked GitHub capability cannot
dispatch or approve the older protected-environment workflow. The scientific
inputs are unchanged.

## Authorization

The trigger is the content-addressed request

```text
protocols/execution_requests/cut3r_deform360_source_freeze_v2.json
```

The workflow accepts only an ordinary, non-forced push to `refs/heads/main`. It
checks that the request file changed in that push, verifies its content identity,
replays the exact source-protocol Git blob, and checks out the resulting merged
revision by full SHA.

The self-hosted job has `contents: read` only. It receives no pull-request fields,
no writable repository token, and no secret-valued command input. The merge into
`main` is the authorization event; this v2 route deliberately does not use a
GitHub Environment approval.

## Execution

The workflow:

1. builds one wheel from the exact merged revision;
2. installs that wheel in an isolated environment;
3. verifies the retained CUT3R checkout, checkpoint, Deform360 processed source
   root, BayesianPhysTwin selection lock, protocol, wheel, and input sidecars;
4. executes `build_cut3r_deform360_source_freeze.py`;
5. retains either `source-support-freeze-ready` or the registered
   `insufficient-common-camera-support` negative;
6. creates and verifies the immutable CUT3R comparison lock only after a support
   pass;
7. sanitizes protected filesystem paths before publication;
8. uploads checksummed evidence; and
9. posts queued and terminal workflow pointers to issue 49 from hosted jobs.

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
