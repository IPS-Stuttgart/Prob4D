# Trusted self-hosted exact-head validation

Prob4D uses GitHub-hosted runners for ordinary pull-request checks. Pull-request
source must not select, trigger, or redirect a job onto a persistent self-hosted
runner merely through a branch name, changed workflow, or dispatch input.

The only repository workflow allowed to execute reviewed pull-request source on
`workstation2` is `.github/workflows/trusted-self-hosted-validation.yml`. It is a
manual, exact-revision validation path for work that genuinely needs local GPU,
large-memory, or approved data access.

The workflow and this repository document the intended controls, but repository
files cannot prove current external settings or host state. Record those through
the separately validated, machine-readable
[trusted validation boundary attestation](trusted-validation-boundary-attestation.md).
An unverified template or repository-local statement never counts as operational
proof.

## Required repository environment

Create an environment named exactly:

```text
trusted-self-hosted-validation
```

Protect it before the workflow is used:

- require at least one independent reviewer;
- do not attach environment secrets;
- do not attach repository, deployment, cloud, or package-write credentials;
- restrict deployment branches so the workflow definition must come from
  `main`;
- do not permit the pull-request author to approve their own execution where the
  repository settings support that restriction; and
- require the approver to compare the displayed pull-request number and full
  40-character head SHA with the reviewed source.

The workflow itself also verifies, on a GitHub-hosted runner and before any
self-hosted checkout, that:

- it was dispatched from `refs/heads/main`;
- the pull request is open and based on `main`;
- the pull request belongs to this repository rather than a fork; and
- the requested SHA is exactly the current pull-request head.

A stale SHA, changed head, fork, closed pull request, non-main base, or dispatch
from another ref fails before self-hosted execution.

## Operation

From the default branch, open **Actions → Trusted exact-head validation → Run
workflow** and provide:

1. the open pull-request number;
2. the exact reviewed 40-character head SHA; and
3. the fixed reviewed profile: `full-validation` or `production-memory`.

The self-hosted job pauses at the protected environment. After approval, it
checks out only the authorized SHA with persisted Git credentials disabled,
verifies a clean exact checkout, creates isolated home/cache/temp/virtualenv
directories below `RUNNER_TEMP`, and runs policy checks before general project
code. It records the source, base, pull request, runner, Python version,
distribution digests, selected validation lanes, and final job status in a
compact evidence artifact.

The ordinary pull-request test, quality, contract, provider-neutral, visual-bias,
identity, memory, and controlled-stress workflows remain GitHub-hosted. A
self-hosted validation request is therefore explicit and exceptional rather than
a fallback selected by pull-request-controlled source.

## Runner hardening outside GitHub Actions

Environment approval is an authorization barrier, not a sandbox. Once approved,
the reviewed revision can execute arbitrary code as the runner operating-system
account. The host must therefore be configured defensively:

- use a dedicated non-administrator runner account;
- remove cloud, SSH, package-registry, and personal credentials from that account;
- mount only explicitly approved datasets, preferably read-only;
- keep unopened target cohorts outside the runner's ordinary namespace until a
  separately authorized protocol requires them;
- restrict access to unrelated repositories, user homes, caches, services, and
  network destinations;
- use disposable workspaces and inspect cleanup failures; and
- rotate or rebuild the runner after a suspected boundary violation.

The workflow redirects common Python, XDG, Hugging Face, pip, and temporary paths
to a run-specific directory and removes that directory after artifact upload.
This reduces accidental state reuse but does not prevent deliberately malicious
code from reading any resource available to the runner account.

## Evidence and claim boundary

A successful run proves that one exact source revision passed the selected
implementation checks in the recorded environment. A ready boundary attestation
additionally records independent evidence for the GitHub environment, runner
host, dataset namespace, and registered positive and negative controls at one
snapshot. Neither establishes:

- observation accuracy or uncertainty calibration;
- BayesianPhysTwin or Causal4D benefit;
- permission to open a held-out or confirmatory cohort;
- deployment or safety readiness; or
- reproducibility on a different host or dependency resolution.

Claim-bearing experiments must continue to use their separately frozen protocol,
information-order, dataset, calibration, artifact, and evidence-publication
contracts.
