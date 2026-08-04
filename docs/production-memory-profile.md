# Production memory profiling

Prob4D provides a workflow for reproducible, fresh-process memory measurements
at the production `25 x 320 x 640` window shape. It is an engineering evidence
surface for issue #50, not a reconstruction or downstream physics experiment.

## Covered phases

The workflow runs each measured phase in a separate Python process:

1. dense covariance-intersection fusion with configurable contributor and tile
   counts;
2. metric, prefix-aligned, and oracle-aligned provider evaluation with optional
   dense scene flow and configurable evaluation chunks;
3. optional eager-NPZ and read-only mmap-NPY loading for an actual prediction
   bundle already present on the runner.

Separate processes prevent an earlier phase's high-water RSS mark from being
attributed to a later phase. Every JSON report records the exact repository
revision, Python and NumPy versions, host platform, configuration, timing, peak
process RSS, retained-array accounting, and a deterministic output digest.

## Runner and invocation

The checked-in workflow uses `ubuntu-latest` by default. Manual dispatches can
select `self-hosted` to target the IPS runner labels:

```text
self-hosted, Linux, X64, nvidia-smi
```

A pull request that changes the workflow or one of its benchmark surfaces runs
the full synthetic profile automatically. For a manual run, open **Actions →
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

The optional `prediction_manifest` and `prediction_store` inputs are absolute
paths already present on the selected runner. They are primarily useful with
the self-hosted option because GitHub-hosted runners are ephemeral. When
supplied, the workflow additionally runs the verified eager and mmap loading
benchmarks in fresh processes. These paths are never uploaded; only their
content-addressed identities and measurements are included in the JSON evidence.

## Evidence artifact

Each run uploads one `production-memory-profile-<run-id>` artifact containing:

- `host.txt`, including CPU, RAM, Python, NumPy, and any available GPU/driver
  identity;
- one JSON and captured stdout file per executed phase;
- `summary.json`, binding the phase reports to the workflow revision and runner;
- `SHA256SUMS` for every evidence file.

The workflow fails closed when a required phase does not report positive peak
RSS or when a report revision differs from `GITHUB_SHA`.

## Claim boundary and remaining work

Synthetic full-resolution runs establish only resource behavior for the selected
host and configuration. Optional actual-bundle loading adds storage evidence but
does not establish visual accuracy, calibration, Bayesian-PhysTwin acceptance,
or Causal4D benefit.

Issue #50 remains open until one frozen real multi-overlap bundle is profiled
phase by phase through loading, disagreement, gauge estimation, fusion,
provider export, and evaluation. The workflow is the repeatable hardware and
artifact substrate for that final evidence gate.
