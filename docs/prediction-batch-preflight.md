# Causal prediction-batch preflight

Individually valid provider archives can still be incompatible with one frozen
scorer batch because frame counts, spatial grids, dtypes, or optional fields
differ. `prob4d.prediction_batch_preflight` detects that boundary before a
scorer calls `stack`.

## Causal loading

The manifest is parsed before any dense payload is opened. With an exclusive
causal cutoff, only payloads admitted by their declared source-frame lineage are
hashed and decoded. Excluded future payloads are never opened, and every result
records `future_prediction_payloads_opened: 0`.

A selected payload with a byte-count, SHA-256, schema, window-ID, frame-lineage,
optional-field, or storage-dtype mismatch raises
`PredictionBatchIntegrityError`. Integrity failure is not relabelled as a
scientific negative.

## Default policy

`PredictionBatchPolicyV1` requires:

- at least one admitted payload;
- a common frame count;
- a common spatial point-grid shape;
- a common point dtype; and
- a common scene-flow/ray presence signature.

A prospectively defined ragged scorer can explicitly relax the relevant
requirement. The preflight never pads, truncates, or rewrites payload bytes.

## Commands

```bash
python -m prob4d.prediction_batch_preflight build \
  outputs/provider/provider-neutral.json \
  outputs/provider/prediction-batch-preflight.json \
  --causal-frame-stop 134
```

Exit status `0` means compatibility. Exit status `2` is a valid persisted
`batch-incompatible` result with stable violations such as
`frame-count-mismatch`, `spatial-shape-mismatch`, `point-dtype-mismatch`, and
`optional-field-mismatch`.

```bash
python -m prob4d.prediction_batch_preflight verify \
  outputs/provider/prediction-batch-preflight.json
```

A stopped protocol can bind this artifact in a
[`ProviderTerminalDecisionV1`](provider-terminal-decision.md).

## Nonclaim

Passing preflight establishes representation compatibility only. It does not
establish provider accuracy, calibration, independence, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
