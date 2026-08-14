# Provider adapter SDK and conformance

`prob4d.adapter.v1` is the versioned provider-authoring boundary for converting
an external 4-D model into the canonical provider-neutral prediction contract.
The provider retains ownership of model loading, native caches, checkpoints, and
inference. Prob4D owns deterministic conversion, causal lineage, dependence
semantics, no-clobber publication, and the final
`PredictionProviderManifestV1`.

The SDK is intentionally narrower than the stable consumer-facing
`prob4d.api.v2` façade. A successful adapter run establishes interoperability
and provenance only; it does not establish accuracy, calibration, independence,
or downstream physical benefit.

## Contract objects

A provider implements `PredictionProviderAdapterV1`:

```python
from collections.abc import Sequence

from prob4d.adapter.v1 import (
    PredictionProviderAdapterV1,
    ProviderAdapterIdentityV1,
    ProviderAdapterRequestV1,
    ProviderAdapterWindowV1,
)


class MyProviderAdapter(PredictionProviderAdapterV1):
    @property
    def identity(self) -> ProviderAdapterIdentityV1:
        return ProviderAdapterIdentityV1(
            adapter_name="my-provider-adapter",
            adapter_version=1,
            adapter_implementation_id=adapter_source_sha256,
            provider_family="MyProvider",
            provider_repository="owner/provider",
            provider_revision=provider_commit,
            provider_run_id=provider_run_sha256,
            model_set_id=model_set_sha256,
            loader_id=loader_sha256,
            coordinate_semantics="window-local-sim3",
            point_semantics="dense-point-map",
            flow_semantics="forward-point-displacement",
            ray_semantics="camera-ray-unit-vector",
            metadata={"native_format": "my-provider-cache-v1"},
        )

    def produce(
        self,
        request: ProviderAdapterRequestV1,
    ) -> Sequence[ProviderAdapterWindowV1]:
        # Load only the input snapshot bound by request.input_snapshot_id.
        # Every declared source interval must end at or before
        # request.causal_frame_stop.
        return convert_native_cache(request)
```

`ProviderAdapterIdentityV1` binds the adapter implementation, provider source,
model set, loader, run, coordinate convention, point/flow/ray semantics, and the
explicit declaration that truth, protected target outcomes, and downstream
physical innovations were not used.

`ProviderAdapterRequestV1` binds one exact native input family and snapshot, one
sequence, and one exclusive causal cutoff. Requests reject opened target data.

`ProviderAdapterWindowV1` combines a validated `PredictionWindow` with:

- one confined relative output path;
- product role;
- view and stochastic-member identities;
- dependence groups; and
- exact source-frame lineage for every output frame.

Materialize the adapter through:

```python
from prob4d.adapter.v1 import materialize_provider_adapter

manifest = materialize_provider_adapter(
    adapter,
    request,
    "outputs/case-a/provider-neutral.json",
)
```

The materializer canonicalizes adapter output order, writes canonical NPZ
payloads atomically, permits only idempotent same-content reuse, verifies every
payload, and embeds the adapter identity, request identity, input snapshot,
causal cutoff, and request metadata in the manifest.

## Build request artifacts

Create the base request used by the conformance fixture:

```bash
prob4d prediction adapter-conformance build-request \
  --sequence-id fixture-case \
  --causal-frame-stop 25 \
  --input-family-id <sha256> \
  --input-snapshot-id <prefix-snapshot-sha256> \
  --output base-request.json
```

Create a second request for the same input family with a longer causal prefix:

```bash
prob4d prediction adapter-conformance build-request \
  --sequence-id fixture-case \
  --causal-frame-stop 42 \
  --input-family-id <same-sha256> \
  --input-snapshot-id <extended-snapshot-sha256> \
  --output future-request.json
```

Optional finite JSON metadata can be supplied with `--metadata-json`. Verify a
request independently with:

```bash
prob4d prediction adapter-conformance verify-request \
  --artifact base-request.json
```

## Adapter conformance

Expose either an adapter object or a zero-argument factory as `module:attribute`,
then run:

```bash
prob4d prediction adapter-conformance run \
  my_provider.prob4d_adapter:build_adapter \
  --base-request base-request.json \
  --future-request future-request.json \
  --output-dir outputs/adapter-conformance \
  --report outputs/adapter-conformance.json
```

The self-contained conformance artifact checks:

1. manifest identity and exact request binding;
2. adapter identity stability across calls;
3. deterministic repeated production;
4. invariance to adapter output order; and
5. causal-prefix invariance when a longer native input snapshot is supplied.

The last check compares the complete base payload identity sequence with the
payloads from the extended run that are admissible at the original cutoff. A
provider that silently changes an earlier prediction after seeing later input
therefore fails conformance even when every individual NPZ is valid.

Verify the retained artifact with:

```bash
prob4d prediction adapter-conformance verify \
  --artifact outputs/adapter-conformance.json
```

Exit status `0` means every conformance check passed. Exit status `2` is a valid,
replayable conformance failure. Schema, identity, or artifact corruption raises a
validation error instead of being relabelled as a provider failure.

## Readiness-matrix integration

A prospective provider-readiness matrix lock binds both:

- `provider_adapter_identity_id`; and
- `provider_adapter_conformance_id`.

Every later source-only readiness decision must carry those identities together
with the matrix lock and comparison-policy identities. This separates adapter
interoperability from source competence while preventing an unqualified adapter
from being substituted after the comparative provider program is frozen.

## Scientific boundary

Adapter conformance does not show that a provider mean is useful, that its
uncertainty is calibrated, that two providers are independent, or that a
BayesianPhysTwin update should be accepted. Those are later source, covariance,
query-relevance, and exact-fallback gates. Causal4D remains downstream of the
belief selected by BayesianPhysTwin.
