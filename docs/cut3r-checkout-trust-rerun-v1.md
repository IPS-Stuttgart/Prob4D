# Exact retained CUT3R isolated-HOME repair v2

The exact retained CUT3R source-freeze run `32771242880`, attempt 2, reached the
sole retained Deform360 runner and completed checkout, Python provisioning, and
installation of the reviewed Prob4D wheel. It stopped before CUT3R inference
because the source-freeze builder could not resolve `HEAD` in the retained CUT3R
checkout.

The bounded source-freeze failure capsule is:

```text
run: 32771242880
attempt: 2
execute job: 97702294765
artifact: 9551181122
name: cut3r-source-freeze-v2-failed-32771242880-2
size: 4384 bytes
SHA-256: d1eff3af637eb297e72334693b1c51723f4eb9487a6cf6a8957d130bc34b9721
```

The first repair helper, run `32817085071`, established two additional facts
without opening any source outcome:

1. command-scoped `safe.directory` recovered the exact frozen CUT3R revision
   `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` and a clean worktree;
2. the runner does not permit a passwordless system Git-config write.

That helper therefore stopped before requesting a rerun. No CUT3R prediction,
source residual, source truth, confirmation payload, target payload,
BayesianPhysTwin result, or Causal4D result was produced by either failure.

## Exact v2 command

After this workflow is merged to `main`, only this exact issue-49 comment from
`FlorianPfaff` is accepted:

```text
/prob4d-repair-cut3r-isolated-home-and-rerun-v2 32771242880 32817085071 9551181122 d1eff3af637eb297e72334693b1c51723f4eb9487a6cf6a8957d130bc34b9721
```

The hosted authorization binds the exact source-freeze run, attempt, seven-job
roster, failed execution step, single diagnostic artifact, prior helper run,
prior helper job roster, and zero helper artifacts. It also requires the source
workflow to remain at attempt 2, so a previously executed attempt cannot be
silently repeated.

## Attempt-specific isolated HOME

The historical source-freeze workflow deliberately creates a fresh HOME at:

```text
${RUNNER_TEMP}/prob4d-cut3r-source-freeze-v2-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}/home
```

For the authorized rerun, a short-lived watcher is armed on `workstation2`
before GitHub requests attempt 3. The watcher:

- first verifies the retained checkout under command-scoped trust;
- requires the exact frozen CUT3R revision and a completely clean worktree;
- refuses any pre-existing attempt-3 workspace or `.gitconfig`;
- waits only for the exact run-32771242880, attempt-3 isolated HOME;
- atomically writes one mode-0600 `.gitconfig` containing the exact
  `safe.directory` entry;
- writes a bounded marker containing no retained path; and
- exits.

The process is detached only so that the single self-hosted runner can accept the
original job. It has a fixed timeout, no repository token with write permission,
and no ability to dispatch Actions or write issue comments. A GitHub-hosted job
alone requests rerun of failed job `97702294765`.

The historical source-freeze job then runs unchanged. Its existing cleanup
removes the complete attempt-specific workspace, including the temporary Git
configuration. A workflow-run cleanup removes only the watcher request, script,
PID, log, marker, and any exact leftover attempt-3 workspace.

No system or user-global Git configuration is changed.

## Scientific boundary

This is an execution-environment repair for a failure before provider execution.
It changes no provider, checkpoint, source object/session roster, camera panel,
causal prefix, metric anchor, support rule, seed, covariance method, target
roster, or analysis. Scientific evidence begins only if the original frozen
source-freeze workflow publishes its registered terminal source-support decision.
