# Recursive visual-bias nuisance streams

`VisualBiasNuisanceV1` represents coherent camera or provider bias separately
from local point covariance and from the existing `Sim(3)` gauge nuisance. A
recursive Bayesian update must not instantiate an independent copy of that bias
prior at every observation time. Doing so counts the same calibration prior more
than once and removes the cross-update covariance induced by one persistent
camera bias.

`VisualBiasNuisanceStreamV1` binds several causal observation updates to one
shared visual-bias latent state:

```text
z[k, i] = h(x[k], i) + Jg[k, i] delta_g[k] + Bb[k, i] b + epsilon[k, i]
                                                              ^
                                                one shared latent across k
```

The stream is additive. It does not modify observation-factor schema v4 or any
frozen provider-v1/provider-v2 artifact. A downstream estimator must opt in and
bind the stream artifact explicitly.

## What the stream binds

For every update, the append-only record includes:

- the exact `ObservationFactorStreamV1` update ID;
- the exact `VisualBiasNuisanceV1` sidecar artifact ID;
- the observation artifact and ordered-row identity digests;
- the causal frame interval admitted by that update;
- the contiguous row interval in the combined payload;
- the maximum retained gauge projection; and
- the previous visual-bias stream update ID.

All updates must share exactly the same:

- ordered bias-scope IDs;
- ordered basis names;
- joint bias covariance bytes;
- orthogonalization semantics and tolerance; and
- source-calibration model metadata.

The payload concatenates only row mappings and Jacobians. It stores the joint
bias covariance once. `low_rank_factor()` therefore retains cross-update
covariance rather than constructing a block-diagonal copy of the prior for each
measurement.

## Python usage

```python
from prob4d.visual_bias_stream import (
    append_visual_bias_nuisance,
    build_visual_bias_nuisance_stream,
    load_visual_bias_nuisance_stream,
    write_visual_bias_nuisance_stream,
)

stream = build_visual_bias_nuisance_stream(
    stream_key="object-07-camera-0",
    nuisances=(prefix_bias_0, prefix_bias_1),
    observation_stream_update_ids=(factor_update_0, factor_update_1),
    frame_intervals=((100, 110), (110, 120)),
    model_metadata={
        "calibration_artifact_id": calibration_id,
        "scope": "camera-0",
    },
    metadata={"case_id": "object-07"},
)

stream = append_visual_bias_nuisance(
    stream,
    prefix_bias_2,
    observation_stream_update_id=factor_update_2,
    frame_interval=(120, 130),
)

write_visual_bias_nuisance_stream(
    stream,
    "outputs/object-07/visual-bias-stream.json",
)
loaded = load_visual_bias_nuisance_stream(
    "outputs/object-07/visual-bias-stream.json"
)
```

Validate an installed artifact without importing an experiment runner:

```bash
prob4d observation visual-bias-stream validate \
  outputs/object-07/visual-bias-stream.json
```

The writer uses a fail-closed writer lock, atomic temporary files, idempotent
same-artifact reuse, and refuses to replace a different content-addressed
stream. The loader rejects duplicate or non-finite JSON, unknown fields, unsafe
or symlinked payload paths, pickle-enabled payloads, hash/byte-count changes,
array descriptor drift, update-chain forks, overlapping frame intervals, and
row-chain truncation.

## Correct downstream use

The shared latent must be represented once. With stacked bias design `B` and
source-calibrated prior `Sigma_b`, the coherent covariance contribution is

```text
B Sigma_b B^T,
```

where `B` contains rows from every recursive update. Replacing this with one
independent block per update destroys the off-diagonal covariance and can make
repeated observations appear more informative than they are.

The stream does not define physical discrepancy dynamics or an action-conditioned
transition model. BayesianPhysTwin owns the physical state, action-conditioned
prediction, regret guard, and exact fallback. Prob4D owns the observation-side
bias evidence and its causal/source lineage. Causal4D remains downstream of an
accepted, content-bound physical belief.

## Claim boundary

A valid stream establishes only that several observation updates were bound to
one explicit, source-calibrated bias model without silently duplicating its
prior. It does not show that the visual provider is accurate, the bias basis is
complete, target uncertainty is calibrated, a physical update is beneficial,
or an intervention conclusion is valid. Those remain separate held-out gates.
