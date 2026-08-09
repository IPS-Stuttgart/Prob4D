# Claim-bearing tree-sparse observation artifacts

The portable tree-sparse observation artifact is a storage and replay contract.
It is not scientific evidence by itself. Claim-bearing use requires a separate
provider-v2 envelope that binds the artifact to the exact causal source,
calibration, provider manifest, and independently verified runtime revision.

## Evidence boundary

`ClaimBearingTreeSparseObservationEnvelopeV1` binds:

- the exact tree-sparse observation manifest bytes and artifact ID;
- the exact sparse gauge-tree prior artifact ID and semantic prior ID;
- sequence, case, stream, source repository, source revision, causal cutoff,
  observation count, and gauge order;
- the complete selected source-window lineage for every gauge;
- the extended provider-v2 manifest ID and complete provider attestation;
- exact gauge- and point-calibration artifact IDs;
- independently matched runtime revision evidence; and
- finite content-addressed metadata.

Every row is checked against the source-window interval of its named gauge. A row
outside that interval fails before observation files are published.

## Extended provider manifest

The ordinary provider-v2 manifest remains unchanged for frozen observation-belief
and schema-v4 factor products. Tree-sparse claim-bearing execution opts into
`prob4d_tree_sparse_provider_manifest`, which adds the capabilities:

```text
content_addressed_tree_sparse_observation_artifacts
strict_claim_bearing_tree_sparse_observation_loading
```

and declares schema version 1 for both:

```text
TreeSparseObservationArtifactV1
ClaimBearingTreeSparseObservationEnvelopeV1
```

The envelope loader requires those exact capabilities and versions. A generic
provider-v2 attestation cannot be relabelled as tree-sparse claim evidence.

## Writing with validated calibration objects

```python
from prob4d.provider_v2_factors import (
    write_claim_bearing_tree_sparse_observation,
)

validated = write_claim_bearing_tree_sparse_observation(
    tree_stacked,
    "outputs/case-a/tree-sparse-claim.json",
    sequence_id="sequence-a",
    case_id="case-a",
    stream_id="prob4d:tree-sparse:camera-panel",
    source_revision=source_revision,
    causal_selection=causal_selection,
    gauge_covariance_calibration=gauge_calibration,
    point_uncertainty_calibration=point_calibration,
    artifact_metadata={"split": "calibration"},
    metadata={"protocol": "tree-sparse-to-bpt-v1"},
)
```

The high-level writer:

1. validates the calibration pair against the selected prediction target;
2. independently verifies the runtime revision;
3. builds the extended provider manifest and calibrated provider attestation;
4. validates row-to-window causal alignment and all metadata before filesystem
   publication;
5. writes the portable observation and sparse-prior artifacts; and
6. publishes the envelope create-if-absent after the referenced manifest hash is
   fixed.

A lower-level `seal_claim_bearing_tree_sparse_observation` entry point accepts an
already constructed provider attestation for controlled tests and external
orchestration. It applies the same strict validation before publication.

## Loading

```python
from prob4d.provider_v2_factors import (
    load_claim_bearing_tree_sparse_observation,
)

validated = load_claim_bearing_tree_sparse_observation(
    "outputs/case-a/tree-sparse-claim.json"
)
```

Loading first validates the envelope and provider attestation, resolves the
observation manifest through a confined relative path, verifies its exact file
hash, strictly loads the tree-sparse observation and prior artifacts, and then
checks every mirrored identity and row/source-window relationship again.

The result exposes the strict observation object, envelope artifact ID, provider
manifest ID, and calibration IDs. A downstream BayesianPhysTwin adapter should
consume this validated object rather than reopening or trusting an unsealed
storage artifact.

## Failure and compatibility semantics

Duplicate or non-finite JSON, Boolean schema aliases, unknown fields, path
traversal, changed artifact bytes, substituted prior, changed calibration IDs,
provider-manifest drift, unverified runtime code, source-window mismatch, and
content-identity tampering fail closed.

The ordinary schema-v4 claim-bearing factor bundle remains supported and
unchanged. This envelope is additive and explicitly selected; it does not
reinterpret historical artifacts or evidence.

A valid envelope establishes provenance, causal information order, calibration
identity, and strict sparse replay. It does not establish provider competence,
uncertainty calibration on a fresh cohort, physical-query benefit,
BayesianPhysTwin safety, Causal4D intervention benefit, deployment safety, or
state of the art.
