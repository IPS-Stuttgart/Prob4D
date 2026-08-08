# Portable sparse gauge-tree prior artifacts

`GaugeTreeSquareRootPriorV1` represents Prob4D's production causal gauge tree
with one parent, one `7 x 7` transition, and one independent innovation
Cholesky factor per window. The in-memory backend removes the dense `7K x 7K`
prior after construction, but schema-v4 observation-factor bundles still carry
that dense matrix.

`prob4d.gauge_tree_prior_artifact` adds a separate portable sidecar for consumers
that need the exact tree prior without opening the dense covariance first. It is
additive: provider-v1, provider-v2, schema-v4 bundles, and all frozen artifact
identities remain unchanged.

## Artifact layout

A manifest binds three ordinary, non-pickled NPY files:

```text
prior.json
gauge-tree-prior-parent-indices-<sha256>.npy
gauge-tree-prior-transition-matrices-<sha256>.npy
gauge-tree-prior-innovation-scale-tril-<sha256>.npy
```

The arrays have the fixed portable dtypes and shapes:

```text
parent_indices:         <i8, shape K
transition_matrices:    <f8, shape K x 7 x 7
innovation_scale_tril:  <f8, shape K x 7 x 7
```

The manifest records, for every member:

- the content-addressed filename;
- exact file SHA-256 and byte count;
- dtype and shape; and
- a canonical array identity over dtype, shape, and values.

It also binds the complete gauge order, sparse-prior identity, representation
semantics, and optional source dense-covariance identity. The manifest's own
artifact ID is independent of its directory and filename.

## Writing and loading

```python
from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.gauge_tree_prior_artifact import (
    load_gauge_tree_prior_artifact,
    write_gauge_tree_prior_artifact,
)

prior = GaugeTreeSquareRootPriorV1.from_transition_covariances(
    gauge_ids=gauge_ids,
    parent_indices=parent_indices,
    transition_matrices=transition_matrices,
    innovation_covariances=innovation_covariances,
)

published = write_gauge_tree_prior_artifact(
    prior,
    "outputs/case-a/gauge-tree-prior.json",
)
loaded = load_gauge_tree_prior_artifact(
    "outputs/case-a/gauge-tree-prior.json"
)
assert loaded.prior.prior_id == published.prior.prior_id
```

Direct construction from transition and innovation factors gives the intended
low-memory path. Conversion from an existing dense schema-v4 bundle is also
possible through `GaugeTreeSquareRootPriorV1.from_dense_covariance`, but that
transitional route necessarily opens the dense matrix once before publishing the
sparse sidecar.

Validate an artifact from an installed package with:

```bash
python -m prob4d.gauge_tree_prior_artifact \
  outputs/case-a/gauge-tree-prior.json --json
```

## Fail-closed publication and loading

Publication is create-if-absent and idempotent for identical bytes. It refuses to
replace a different manifest or payload. Payload names are derived from their
exact byte digests, and the manifest is published only after all members exist.
A failed final publication can therefore leave only harmless content-addressed
orphan members, never a valid manifest pointing to incomplete data.

Before NumPy allocates an array, the loader independently reads the NPY magic,
version, little-endian header length, shape, dtype, memory order, and exact
header-plus-data size from the already bounded file snapshot. This prevents a
small, correctly rehashed payload from advertising an unbounded allocation
shape or a framing layout that differs from the manifest.

Loading rejects:

- duplicate or unknown JSON fields and non-finite JSON constants;
- absolute, nested, traversal, or backslash-separated member paths;
- symbolic-link manifest, directory, or payload components;
- source mutation while a file is read;
- a manifest larger than 1 MiB or an NPY member whose declared/actual size
  exceeds the exact array bytes plus a 64 KiB header allowance;
- unsupported or oversized NPY headers, Fortran-order arrays, and any mismatch
  between header shape/dtype/framing and the manifest;
- file hash, byte-count, dtype, shape, or canonical-array mismatch;
- object arrays, pickled data, malformed NPY data, and trailing bytes;
- changed gauge order, tree semantics, sparse-prior identity, or source binding;
  and
- any factor set rejected by `GaugeTreeSquareRootPriorV1`.

There is no jitter, covariance repair, pseudoinverse, or silent fallback during
loading.

## Storage and compatibility boundary

The payload grows as `O(K)` rather than `O(K^2)`. For 64 gauges, the numerical
factor arrays require about 50 KiB, compared with about 1.53 MiB for the dense
float64 covariance. NPY framing adds only a small fixed overhead per member.

This sidecar does not remove the dense covariance from existing schema-v4 files.
A producer or orchestrator must explicitly publish and select the sparse
artifact, and downstream code must bind its `artifact_id` or `prior_id`. A later
provider or observation-factor schema may adopt these semantics natively only
through a separately versioned contract and compatibility gate.

## Claim boundary

Portable storage, exact identity checks, and lower loading complexity are
engineering properties. They do not establish provider accuracy, uncertainty
calibration, physical-query identifiability, BayesianPhysTwin acceptance,
Causal4D intervention benefit, deployment safety, or state of the art. The
independent-object provider and guarded-query promotion gate remains unchanged.
