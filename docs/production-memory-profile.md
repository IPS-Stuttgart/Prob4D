# Production memory profiling

Prob4D provides fresh-process memory measurements at the production
`25 x 320 x 640` window shape. This is an engineering evidence surface, not a
reconstruction-accuracy or downstream-physics experiment.

Issue #50 is complete. The repository retains the bounded fusion, evaluation,
loading, and artifact-parity implementations plus their recorded evidence. New
profiling work should be opened only when a concrete regression or bottleneck is
identified.

## Covered phases

The standard workflow runs each measured phase in a separate Python process:

1. dense covariance-intersection fusion with configurable contributor and tile
   counts;
2. bounded provider evaluation with optional dense scene flow and configurable
   evaluation chunks; and
3. optional eager-NPZ and read-only mmap-NPY loading when a prediction bundle is
   explicitly staged in the ephemeral job.

Separate processes prevent an earlier phase's high-water RSS mark from being
attributed to a later phase. Every JSON report records the exact repository
revision, Python and NumPy versions, host platform, configuration, timing, peak
process RSS, retained-array accounting, and a deterministic output digest.

## Runner and invocation

`.github/workflows/production-memory-profile.yml` always uses a GitHub-hosted
runner. Pull-request source and workflow inputs cannot redirect it to a persistent
self-hosted machine.

A pull request that changes the workflow or one of its benchmark surfaces runs
the synthetic profile automatically. For a manual hosted run, open **Actions →
Production memory profile → Run workflow**. The defaults are:

```text
frames: 25
height: 320
width: 640
contributors: 3
fusion tile size: 16384
evaluation chunk size: 65536
include flow: true
```

The optional `prediction_manifest` and `prediction_store` values refer only to
paths already staged inside that ephemeral hosted job. They do not provide access
to files on `workstation2` or another persistent host.

Hardware-specific validation on `workstation2` uses the separate
`Trusted exact-head validation` workflow. That workflow must be dispatched from
`main`, verifies an open same-repository pull request and its exact current
40-character head SHA on a hosted runner, and enters the protected
`trusted-self-hosted-validation` environment before checking out reviewed source.
See [trusted self-hosted exact-head validation](trusted-self-hosted-validation.md).

## Evidence artifact

Each standard run uploads one `production-memory-profile-<run-id>` artifact
containing:

- `host.txt`, including CPU, RAM, Python, NumPy, and any available driver
  identity;
- one JSON and captured stdout file per executed phase;
- `summary.json`, binding the phase reports to the workflow revision and runner;
- `SHA256SUMS` for every evidence file.

The workflow fails closed when a required phase does not report positive peak
RSS or when a report revision differs from `GITHUB_SHA`.

The protected exact-head workflow emits a separate compact validation artifact
that binds the approved pull request, exact head and base revisions, runner,
distribution hashes, selected validation lanes, and final job status.

## Claim boundary

Synthetic full-resolution runs establish only resource behavior for the selected
host and configuration. Actual-bundle loading adds storage evidence but does not
establish visual accuracy, uncertainty calibration, BayesianPhysTwin acceptance,
Causal4D benefit, deployment safety, or permission to open target data.

A successful protected self-hosted run is privileged implementation validation
for one exact revision. It does not replace a frozen scientific protocol or its
information-order and evidence-publication requirements.
