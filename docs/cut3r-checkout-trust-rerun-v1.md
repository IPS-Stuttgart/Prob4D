# Exact retained CUT3R checkout-trust repair v1

The exact retained CUT3R source-freeze run `32771242880`, attempt 2, reached the
sole retained Deform360 runner and completed checkout, Python provisioning, and
installation of the reviewed Prob4D wheel. It stopped before CUT3R inference
because the source-freeze builder could not resolve `HEAD` in the retained CUT3R
checkout.

The bounded failure artifact is fixed as:

```text
run: 32771242880
attempt: 2
execute job: 97702294765
artifact: 9551181122
name: cut3r-source-freeze-v2-failed-32771242880-2
size: 4384 bytes
SHA-256: d1eff3af637eb297e72334693b1c51723f4eb9487a6cf6a8957d130bc34b9721
```

Its diagnostic log contains no CUT3R prediction, source residual, source truth,
confirmation payload, target payload, BayesianPhysTwin result, or Causal4D
result. The first failing operation is the read-only Git revision lookup.

## Exact repair command

After this workflow is merged to `main`, only this exact issue-49 comment from
`FlorianPfaff` is accepted:

```text
/prob4d-repair-cut3r-checkout-trust-and-rerun-v1 32771242880 9551181122 d1eff3af637eb297e72334693b1c51723f4eb9487a6cf6a8957d130bc34b9721
```

The hosted authorization job binds the exact failed workflow, attempt, seven-job
roster, failed execution step, successful setup and evidence-retention steps,
and single diagnostic artifact. It authorizes no changed provider, checkpoint,
source group, camera, prefix, seed, covariance method, target group, or analysis.

## Bounded workstation repair

On `workstation2`, the workflow first tries the unchanged Git revision lookup. If
that works and the checkout has the frozen CUT3R revision
`8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`, no trust entry is added. If the
ordinary lookup fails, one command-scoped `safe.directory` probe must recover the
same exact revision and a completely clean worktree. Only then may the workflow
add the exact retained checkout path temporarily to Git's system protected
configuration.

The path value is never printed or posted. The self-hosted job has read-only
repository permissions and cannot dispatch Actions or write issue comments.
A separate GitHub-hosted job rechecks that the failed workflow is still attempt 2
and reruns only job `97702294765`. GitHub reruns that job and its dependent
terminal receipt under the original run identity.

When run `32771242880` finishes attempt 3 or later, a separate self-hosted cleanup
job removes only the exact temporary `safe.directory` entry if this workflow
added it. A GitHub-hosted receipt reports cleanup without publishing the retained
path.

## Scientific boundary

This is an execution-environment repair for a failure before provider execution.
It does not reinterpret the failed run and cannot turn it into scientific
evidence. Scientific evidence begins only if the original frozen source-freeze
job publishes its registered source-support decision. The existing ordered stop
sequence and target-closed boundary remain authoritative.
