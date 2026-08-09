# Portable tree-sparse observation artifacts

`TreeSparseStackedObservationFactors` can now be persisted without serializing a
dense `7K x 7K` joint gauge covariance. The artifact is additive and does not
change schema-v4 `ObservationFactorBundle` bytes or identities.

## Stored representation

The observation manifest binds one portable gauge-tree prior artifact and these
content-addressed non-pickled NPY members:

```text
world_mean_m:                      M x 3 float64
conditional_world_covariance_m2:  M x 3 x 3 float64
local_gauge_jacobian:              M x 3 x 7 float64
gauge_indices:                     M int64
association_probability:           M float64
prior_reliability:                 M float64
prior_nominal_probability:         M float64
composite_weight:                  M float64
point_ids:                         M int64
frame_indices:                     M int64
view_indices:                      M int64
factor_indices:                    M int64
correlation_group_indices:         M int64
```

View, factor, and correlation-group identities use canonical sorted string tables
plus row index arrays. Marginal point covariance is deliberately not stored: the
loader derives it from conditional covariance and the sparse prior. No member
contains `joint_gauge_covariance` or `gauge_prior_covariance`.

The writer publishes the gauge-tree prior under a deterministic manifest name:

```text
gauge-tree-prior-<artifact-id>.json
```

The observation manifest binds both the prior artifact ID and the semantic prior
ID. Moving the complete artifact directory therefore preserves identities.

## Writing

```python
from prob4d.provider_v2_factors import (
    write_tree_sparse_observation_artifact,
)

loaded = write_tree_sparse_observation_artifact(
    tree_stacked,
    "outputs/case-a/tree-sparse-observation.json",
    sequence_id="sequence-a",
    case_id="case-a",
    stream_id="prob4d:tree-sparse:camera-panel",
    source_repository="IPS-Stuttgart/Prob4D",
    source_revision=source_revision,
    metadata={"split": "calibration"},
)
```

Publication is create-if-absent and idempotent for identical content. The writer
publishes the prior and all row members before the observation manifest, so an
interrupted run cannot expose a manifest that references missing files. A
competing different writer cannot replace retained content.

## Loading

```python
from prob4d.provider_v2_factors import (
    load_tree_sparse_observation_artifact,
)

loaded = load_tree_sparse_observation_artifact(
    "outputs/case-a/tree-sparse-observation.json"
)
tree_stacked = loaded.factors
```

The loader:

- rejects duplicate JSON keys, non-finite constants, unknown fields, coercive
  schema versions, and artifact-ID drift;
- confines every payload to one relative NPY filename;
- opens files through stable descriptor-bound no-follow reads where supported;
- verifies exact byte counts, SHA-256 values, bounded NPY headers, dtypes, shapes,
  C ordering, content identities, and absence of trailing bytes;
- independently loads and verifies the referenced sparse prior artifact;
- decodes the canonical string tables and rejects invalid indices; and
- reconstructs the execution object through the strict direct tree-sparse
  factory, which revalidates row geometry, identity, timing, probabilities, and
  grouping semantics.

Loading and writing use tree factors and diagonal covariance blocks only. Dense
prior materialization and dense-covariance verification are not part of this
artifact path.

## Compatibility boundary

Schema-v4 factor bundles remain the compatibility representation and still carry
the dense joint covariance. This artifact is a separately versioned portable
execution product for producers and consumers that explicitly opt into the
sparse tree contract. It does not silently reinterpret an existing schema-v4
manifest.

A valid artifact establishes storage integrity, row/prior identity, and exact
sparse replay semantics. It does not establish observation accuracy, covariance
calibration, physical-query identifiability, BayesianPhysTwin benefit, Causal4D
benefit, deployment safety, or state of the art.
