# Executable provider-neutral prediction runtime

`PredictionProviderManifestV1` records exact provider, model, loader, payload,
source-lineage, coordinate, stochastic-member, and dependence semantics. The
runtime in `prob4d.provider_runtime` turns that portable declaration into an
executable input without reintroducing MotionCrafter-specific assumptions.

## Causal loading boundary

The manifest is parsed and its content identity is validated before any dense
prediction payload is opened. A causal cutoff and optional payload selection are
then applied using manifest metadata only. Prob4D hashes, decodes, and validates
only the selected payloads.

Consequently, a malformed or unavailable post-cutoff archive cannot affect a
valid prefix execution, and the runtime summary records
`future_prediction_payloads_opened: 0`.

```bash
prob4d prediction runtime inspect \
  outputs/sequence/provider-neutral.json \
  --causal-frame-stop 134
```

Selection can be narrowed with repeated `--payload-id` or `--window-id`
arguments. A selected payload that crosses the cutoff fails before its bytes are
opened.

## Coordinate dispatch

The runtime converts the manifest's coordinate semantics into explicit `Sim3`
transforms:

| Manifest semantics | Required runtime input |
| --- | --- |
| `window-local-sim3` | one gauge for every selected payload |
| `sequence-local-sim3` | one shared sequence gauge |
| `camera-local-metric` | one rigid, unit-scale camera-to-world transform |
| `metric-world` | no additional transform |

A sequence-local provider may use the identity gauge only through an explicit
exploratory acknowledgement. A metric-world provider rejects any additional
transform so already metric output cannot be silently re-gauged.

Programmatic users can pass independently calibrated point/flow uncertainty and
gauge covariance to `fuse_prediction_provider`. It calls the existing tiled
`fuse_windows` implementation and therefore retains the established uniform,
precision, and covariance-intersection semantics.

## Dependent alternative constructions

Two complete-sequence outputs from one model execution may have the same frame
grid, view, stochastic member, and dependence groups while differing only in
construction. The two official VGGT point products are the motivating example.
They are alternatives, not independent sensor likelihoods.

The runtime rejects such a joint selection by default. Select one payload, or
pass `--allow-dependent-alternatives` for an explicitly exploratory
covariance-intersection control. Independently decoded overlapping windows keep
their existing admissibility because their product role is
`independent-window` and their unknown correlation is handled by the selected
fusion rule.

VGGT uses the complete supplied sequence. For an earlier causal cutoff, run VGGT
on that prefix and import the resulting prefix manifest. Relabelling a
full-sequence result as causal remains invalid.

## Exploratory executable baseline

The CLI can produce a fused artifact using a declared fixed isotropic covariance:

```bash
prob4d prediction runtime fuse-exploratory \
  outputs/sequence/provider-neutral.json \
  outputs/sequence/provider-runtime-fused.npz \
  --point-standard-deviation-m 0.01 \
  --method covariance_intersection \
  --gauge-config outputs/sequence/provider-gauges.json
```

The gauge file is closed and manifest-bound:

```json
{
  "schema": "prob4d.provider-runtime-gauges",
  "schema_version": 1,
  "manifest_artifact_id": "<64 hex characters>",
  "coordinate_semantics": "window-local-sim3",
  "gauges": {
    "window-000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
}
```

For `sequence-local-sim3`, use the key `__sequence__`. For
`camera-local-metric`, use `__camera_to_world__`. A `metric-world` configuration
contains an empty `gauges` object.

The resulting `FusedPredictionArtifact` records the manifest identity, selected
payloads and dependence groups, coordinate semantics, gauge-config hash, fixed
uncertainty scale, and `claim_bearing: false`. It is suitable for matched
provider-evaluation baselines, not for calibrated BayesianPhysTwin assimilation.

## Claim-bearing use

A claim-bearing path must supply independently fitted conditional uncertainty,
explicit gauge covariance where applicable, source-calibrated reliability, and
the existing provider-promotion evidence. A successful runtime load or
exploratory fusion establishes interoperability and causal admission only. It
does not establish provider competence, calibrated uncertainty, physical-query
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
