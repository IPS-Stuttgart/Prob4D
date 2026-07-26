# Versioned Observation Artifact

Prob4D can export a strict observation product for Bayesian-PhysTwin or another
Bayesian consumer. The artifact records dense positions and optional scene flow,
their marginal covariance, the unresolved global gauge, contributor structure,
source revisions, and the latest source frame that could have influenced every
output frame.

## Export

```bash
prob4d-observation export \
  outputs/sequence/predictions.json \
  outputs/sequence/observation.json \
  --method prob4d_uniform
```

A causal-prefix export must specify the final permitted source frame:

```bash
prob4d-observation export \
  outputs/sequence/predictions.json \
  outputs/sequence/prefix_133.json \
  --method prob4d_uniform \
  --causal-max-frame 133
```

The causal command does not slice an observation fused from the complete video.
It first removes every MotionCrafter window whose maximum source-frame ID exceeds
the boundary, then re-estimates the admissible gauges and re-runs fusion. Export
fails when no complete source window is admissible. The resulting artifact is
validated so that both its output frame IDs and all recorded source dependencies
are at or before the requested boundary.

## Files

An export consists of two files in the same directory:

- `observation.json`: version, gauge status, source-window and frame-level
  provenance, SHA-256, and a compact summary;
- `observation.npz`: dense arrays stored without pickle and with symmetric
  covariance matrices packed into six upper-triangular entries.

The JSON manifest hash-locks the NumPy payload. Moving or renaming the pair is
allowed when the `array_file` entry remains a same-directory relative filename
and its SHA-256 remains unchanged.

## Coordinate and gauge semantics

The standard MotionCrafter export is deliberately marked as:

```text
coordinate_status = gauge_relative
gauge_status      = unresolved
covariance_units  = gauge_unit^2
```

The first selected window names the internal gauge reference, but this is not a
metric anchor. Downstream code must retain the unresolved `Sim(3)` gauge as a
latent variable, supply independent metric information, or abstain from treating
the observations as metric. In particular, gauge-relative covariance must never
be labelled `m^2`.

The in-memory contract also supports a metric artifact, but metric status is
accepted only with an anchored seven-parameter gauge mean and positive
semidefinite gauge covariance.

## Dependence and support

Each source window records its absolute source-frame IDs and a correlation-group
identifier. Each output frame records the source-window IDs that can contribute
to it, while the dense `contributors` array retains the number of valid
predictions at every pixel. The exporter conservatively assigns the maximum
source frame used anywhere in the selected gauge-estimation and fusion run to
every output frame. This prevents a downstream consumer from claiming a tighter
causal boundary merely by slicing a previously generated artifact.

The artifact does not assert conditional independence between pixels or windows.
The correlation group and contributor identities are intended for conservative
prefusion, composite-likelihood grouping, or an explicit shared nuisance model.

## Validation

```bash
prob4d-observation validate outputs/sequence/observation.json
```

Validation checks:

- manifest and array format versions;
- payload SHA-256 and same-directory path confinement;
- array shapes, finite values, contributor consistency, and paired flow fields;
- symmetric positive-semidefinite point, flow, and optional gauge covariance;
- source-window identity and frame containment;
- coordinate units and gauge-status consistency;
- required producer, source-model, method, revision, and input-manifest
  provenance;
- the declared causal boundary, when present.

Validation is intentionally fail-closed. An indefinite covariance, a missing
revision, a future-dependent causal artifact, or a metric label without an
anchored gauge raises an error rather than being silently regularized.
