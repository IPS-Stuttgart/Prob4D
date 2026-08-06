# Exact observation-to-bias stream binding

`ObservationFactorStreamV1` and `VisualBiasNuisanceStreamV1` are independently
content-addressed. That separation is useful, but independent validity alone does
not prove that the two retained streams describe the same recursive evidence. A
caller could accidentally pair a valid camera-bias stream with another valid
factor stream, shift an update boundary, or reuse the right number of rows in the
wrong order.

`ObservationBiasStreamBindingV1` closes that integration boundary without
changing either source schema.

## What is checked

For every update, the binding requires exact equality of:

- the observation-factor update ID cited by the visual-bias update;
- the admitted frame start and exclusive causal frame stop;
- the update-local observation count;
- the ordered observation-identity SHA-256;
- the contiguous global row interval; and
- the retained update order and previous-binding-update ID.

The binding also retains the exact factor-bundle manifest and payload hashes, the
visual-bias sidecar artifact ID, the source observation artifact ID, the complete
factor-stream artifact ID, the complete visual-bias-stream artifact ID, and the
shared bias-model ID.

By default, building or replaying a binding revalidates every factor bundle and
payload referenced by the observation stream. The visual-bias stream loader
revalidates its own strict JSON/NPZ snapshot, array descriptors, hashes, byte
counts, update chain, and shared-prior contract.

## Command-line use

Build a binding after both source streams have been written:

```bash
prob4d observation bias-binding build \
  outputs/object-07/observation-factor-stream.json \
  outputs/object-07/visual-bias-stream.json \
  outputs/object-07/observation-bias-binding.json
```

Validate the retained binding structure and content identity:

```bash
prob4d observation bias-binding validate \
  outputs/object-07/observation-bias-binding.json
```

Replay the binding against the current exact source artifacts:

```bash
prob4d observation bias-binding verify \
  outputs/object-07/observation-bias-binding.json \
  outputs/object-07/observation-factor-stream.json \
  outputs/object-07/visual-bias-stream.json
```

The `verify` command rebuilds the binding from the current streams and requires
exact object equality, not merely matching update counts.

## Python use

```python
from prob4d.observation_bias_binding import (
    build_observation_bias_binding,
    verify_observation_bias_binding,
    write_observation_bias_binding,
)

binding = build_observation_bias_binding(
    observation_factor_stream,
    visual_bias_stream,
    metadata={"consumer": "BayesianPhysTwin"},
)
write_observation_bias_binding(
    binding,
    "outputs/object-07/observation-bias-binding.json",
)
verify_observation_bias_binding(
    binding,
    observation_factor_stream,
    visual_bias_stream,
)
```

## Append-only publication

A binding can be republished at the same path only when all retained update
bindings are an exact prefix of the candidate. Rollback, forked update chains,
changed source identity, changed bias model, changed stream key, and changed
metadata are rejected. A same-artifact write is idempotent.

The top-level source-stream artifact IDs are allowed to advance only together
with at least one newly appended update. This prevents a writer from silently
retargeting an unchanged binding to different source manifests.

## Downstream boundary

A BayesianPhysTwin adapter can now require one binding artifact before jointly
using an observation-factor stream and its persistent visual-bias latent. That
establishes structural correspondence only. BayesianPhysTwin still owns the
physical state, action-conditioned transition, observability analysis, regret
guard, calibration, and exact fallback. Causal4D remains downstream of an
accepted, content-bound physical belief.

A valid binding does not establish visual-provider competence, completeness of
the bias basis, target calibration, physical-query benefit, intervention benefit,
deployment safety, or state of the art.
