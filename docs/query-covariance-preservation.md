# Query-space covariance preservation

`prob4d.query_covariance_preservation` compares full-joint and simplified
covariance treatments after they have been projected into a frozen physical
query. This distinguishes mathematical completeness in observation space from
whether a covariance representation materially affects the quantity consumed by
BayesianPhysTwin.

## Inputs

A `QueryCovariancePreservationCertificateV1` binds:

- the exact BayesianPhysTwin-owned query-definition identity;
- the exact Prob4D observation artifact;
- one positive-trace reference query covariance;
- one or more candidate query covariances;
- a consumer-frozen distortion policy; and
- optional memory and runtime evidence for each candidate.

Candidates can represent, for example:

- the complete explicit joint covariance;
- a rank-capped joint factor;
- a tree-sparse reconstruction;
- block-diagonal gauge covariance;
- marginal-preserving independent rows; or
- conditional-only covariance.

The certificate does not choose among preserved candidates. Computational and
scientific selection remains with the downstream consumer.

## Distortion measures

For reference query covariance `C` and candidate `C_hat`, the certificate reports:

- relative trace distortion;
- relative Frobenius distortion;
- minimum, mean, and maximum directional variance ratios on the numerically
  supported reference subspace;
- maximum absolute directional ratio error; and
- candidate trace placed in a reference-null query direction.

Let

```text
C = V Lambda V^T
```

on the supported reference subspace. Directional ratios are the eigenvalues of

```text
Lambda^(-1/2) V^T C_hat V Lambda^(-1/2).
```

A value below one understates uncertainty in at least one supported direction; a
value above one overstates it. Trace in the orthogonal reference-null subspace is
reported separately instead of being hidden by the supported-direction ratios.

## Example from existing projections

```python
from prob4d.query_covariance_preservation import (
    QueryCovariancePreservationCertificateV1,
    QueryCovariancePreservationPolicyV1,
)

policy = QueryCovariancePreservationPolicyV1(
    relative_rank_tolerance=1e-10,
    maximum_relative_trace_distortion=0.05,
    maximum_relative_frobenius_distortion=0.05,
    minimum_directional_variance_ratio=0.9,
    maximum_directional_variance_ratio=1.1,
    maximum_unsupported_trace_fraction=0.0,
)

certificate = QueryCovariancePreservationCertificateV1.from_projections(
    query_definition_id=query_definition_id,
    observation_artifact_id=observation_artifact_id,
    reference_representation="full-joint",
    reference_projection=full_joint_projection,
    candidate_projections={
        "tree-sparse": tree_sparse_projection,
        "block-diagonal": block_diagonal_projection,
        "independent-rows": independent_rows_projection,
    },
    policy=policy,
)
```

The projection objects can come directly from
`prob4d.query_covariance_relevance`; only their immutable `total_covariance`
matrices are consumed.

## Persistence and replay

```python
from prob4d.query_covariance_preservation import (
    load_query_covariance_preservation,
    write_query_covariance_preservation,
)

write_query_covariance_preservation("query-preservation.json", certificate)
replayed = load_query_covariance_preservation("query-preservation.json")
```

The loader recomputes every eigensystem-derived ratio, distortion, failure reason,
acceptance bit, summary, and content identity. Candidate order is canonicalized
by candidate ID. Covariance matrices must be finite, symmetric, and positive
semidefinite.

## Decision boundary

A preserved candidate is only numerically equivalent under the exact registered
query and tolerances. The result does not prove provider competence, target
calibration, physical-query improvement, safe update admission, Causal4D
intervention benefit, deployment safety, or state of the art.
