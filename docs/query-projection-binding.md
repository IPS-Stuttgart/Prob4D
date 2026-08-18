# Bound query-covariance projection

Prob4D can project its conditional and shared observation covariance into a
caller-owned physical query. A compact projection summary alone, however, does
not prove which Jacobian bytes or which observation-row order produced it.
`prob4d.query_projection_binding` closes that lineage gap without moving query
ownership into Prob4D.

## Responsibility split

The workflow remains one-way:

```text
BayesianPhysTwin
  freezes query semantics, exact Jacobian bytes, and ordered observation rows
        |
        v
Prob4D
  independently validates that binding and projects its covariance
        |
        v
BayesianPhysTwin
  composes the frozen treatment/value decision and retains exact fallback
```

BayesianPhysTwin first creates a content-addressed
`QueryJacobianBindingV1`. The binding records:

- query name, component order, unit, and coordinate frame;
- source observation and provider-manifest identities;
- the exclusive causal frame stop;
- the exact little-endian contiguous `float64` Jacobian shape and SHA-256;
- a SHA-256 over the exact ordered observation-row identifiers; and
- declarations that no target outcome or future frame was used.

Prob4D validates that record independently and does not import
BayesianPhysTwin. It then verifies the actual Jacobian bytes and row roster
before reading them as the projection operator.

## Projection

```python
from prob4d.query_projection_binding import (
    project_bound_joint_covariance_to_query,
    write_bound_query_covariance_projection,
)

receipt = project_bound_joint_covariance_to_query(
    query_jacobian_binding.to_record(),
    query_jacobian,
    ordered_row_ids,
    conditional_covariance_m2,
    shared_low_rank_factor_m,
)

write_bound_query_covariance_projection(
    receipt,
    "bound-query-covariance-projection.json",
)
```

The content-addressed receipt binds:

- the exact query-Jacobian binding ID;
- the source observation and provider manifest;
- the Jacobian-byte and row-order digests;
- complete descriptors of the conditional covariance and shared factor arrays;
- the compact `QueryCovarianceProjectionV1` summary; and
- the fixed scientific claim boundary.

Changing the Jacobian, row order, conditional covariance, shared factor, or
projection summary changes the receipt identity. Reordered rows and changed
Jacobian bytes fail before projection.

The same functionality is exposed through the preview
`prob4d.api.covariance_v1` façade.

## Downstream use

For new BayesianPhysTwin studies, set
`PhysicalQueryV1.jacobian_provider_id` to the
`QueryJacobianBindingV1.artifact_id`. BayesianPhysTwin can then validate the
Prob4D receipt and replace the historical summary-only identity with the exact
bound projection identity when composing the covariance-treatment decision.
Existing frozen summary-only decisions remain unchanged.

## Scientific boundary

This receipt establishes numerical input lineage and cross-repository
interoperability only. It does not establish provider competence, calibrated
uncertainty, physical-query benefit, Causal4D intervention benefit, deployment
safety, or state of the art. BayesianPhysTwin still owns query relevance,
proper-score/value evidence, update admission, and exact fallback.
