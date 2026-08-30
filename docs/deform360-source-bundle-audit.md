# Deform360 processed source-bundle metadata audit

This audit is the first source-only step toward a fresh real-provider study for
Prob4D issue #49. It inventories the canonical processed Deform360 repository on
the self-hosted runner labelled `gpuserver4090`:

```text
/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository
```

The audit establishes whether the reviewed path is present, traversable, and
bounded enough for later source-side adapter and query work. It does not test a
4-D provider, fit covariance, inspect prediction residuals, select a gate, or
open a held-out target.

## Information boundary

The runner may read directory entries and `lstat` metadata only. Dataset files
are never opened. The implementation:

- rejects a symlink as the source root;
- never follows symlinks below the root;
- skips a path component before `stat` or descent when its normalized name
  contains `confirmation`, `fresh-validation`, `held-v8`, `shadow`, or `target`;
- records only relative path, entry type, POSIX mode, and size in the metadata
  manifest;
- excludes modification times and file contents from the manifest;
- caps traversal at 1,000,000 admitted entries and depth 64; and
- writes evidence only to an isolated directory under `RUNNER_TEMP`.

The checked protocol is
[`protocols/deform360-source-bundle-audit-v1.json`](../protocols/deform360-source-bundle-audit-v1.json).
Its content address binds the fixed root, runner label, forbidden tokens, limits,
and all negative authorization flags.

## Trigger and authorization

The protected workflow is
[`.github/workflows/deform360-source-bundle-audit.yml`](../.github/workflows/deform360-source-bundle-audit.yml).
A run is admitted only when a non-forced push to `main` changes exactly:

```text
protocols/execution_requests/deform360_source_bundle_audit_v1.json
```

The request must bind the exact Git blob of the merged protocol. The hosted
authorization job validates that identity before the self-hosted job is eligible.
The self-hosted job has read-only repository permission and receives no
repository write token. Issue reporting is isolated in a later hosted job.

## Evidence

A successful run uploads `result.json` and `summary.md`. The result contains:

- a deterministic metadata-manifest SHA-256;
- regular-file, directory, symlink, and other-entry counts;
- total regular-file bytes;
- depth and extension summaries;
- bounded top-level, sample-path, and largest-file summaries;
- counts of forbidden path components skipped;
- bounded metadata-error accounting; and
- explicit flags stating that no file content, prediction, residual, target, or
  dataset mutation was authorized.

The independent statistical units and any final Deform360 source/calibration/
target split are not defined by this inventory. Historical object exclusions
must be resolved and a new protocol must be frozen before a fresh target is
opened.

## Claim boundary

A positive decision means only that the fixed processed source bundle is
available and completely inventoried under the declared limits. It does not
establish provider competence, covariance calibration, query benefit,
BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the
art.
