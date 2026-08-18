# Provider-neutral held-out promotion lock

Prob4D's original held-out promotion lock was created for the historical
MotionCrafter route. Its version-1 descriptor stores
`motioncrafter_revision` and `model_set_id` directly. Those artifacts remain
strictly readable and retain their original bytes and content identities.

New claim-bearing provider studies should use the additive version-2 lock. It
replaces the two MotionCrafter-specific fields with one strict
`provider_identity` record aligned with `PredictionProviderManifestV1`:

```json
{
  "provider_identity": {
    "schema_name": "prob4d.provider-promotion-identity",
    "schema_version": 1,
    "provider_family": "cut3r",
    "provider_repository": "naver/CUT3R",
    "provider_revision": "<exact 40- or 64-character revision>",
    "model_set_id": "<SHA-256>",
    "loader_id": "<SHA-256>",
    "coordinate_semantics": "sequence-local-sim3",
    "point_semantics": "dense-point-map",
    "flow_semantics": "absent",
    "ray_semantics": "absent",
    "source_dependency_semantics":
      "per-output-exclusive-source-frame-interval-v1"
  }
}
```

The provider family, repository, exact revision, model set, loader, coordinate
and payload semantics, and source-dependency semantics are immutable across the
frozen source, calibration, and target groups. Case-local manifest identities,
manifest-byte digests, and provider run IDs remain in the corresponding
provider-manifest and target-admission artifacts; they are not method invariants
and therefore do not belong in the promotion lock.

Start from the provider-neutral example:

```bash
cp docs/examples/heldout-provider-promotion-config-v2.json protocol.json

prob4d experiment heldout-provider freeze protocol.json \
  --output promotion-lock.json
```

The existing `run`, `verify`, target-admission, and cohort-binding commands accept
both lock versions. For a version-2 lock, target admission fails closed unless
every admitted manifest matches the complete frozen provider contract. A changed
loader, repository, coordinate system, payload meaning, or source-dependency
semantics is rejected even when the provider revision and model-set digest are
unchanged.

Promotion evidence cards are versioned in parallel. Historical lock version 1
continues to produce the historical evidence-card layout with a
`repositories.motioncrafter` record. Lock version 2 produces evidence-card
version 2 with the complete generic provider identity under
`repositories.provider`.

## Compatibility boundary

- Version-1 lock descriptors, IDs, reports, and evidence cards are not rewritten.
- Version-2 is additive and uses the same registered comparison arms, grouped
  bootstrap, provider gate, guarded-query gate, and exact-fallback semantics.
- A configuration cannot mix `provider_identity` with the historical
  `motioncrafter_revision` or top-level `model_set_id` fields.
- The target-provider admission schema remains unchanged because it was already
  provider-neutral; version-2 locks strengthen its lock-consistency check to the
  complete provider contract.

## Scientific boundary

Provider-neutral identities authenticate a frozen method and its information
semantics. They do not establish source support, provider accuracy, uncertainty
calibration, BayesianPhysTwin update value, Causal4D intervention benefit,
deployment safety, or state of the art. No target outcome is needed to create or
validate either lock version.
