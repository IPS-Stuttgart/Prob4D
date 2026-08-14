# Causal prediction-batch preflight

A provider run can produce individually valid prediction archives that still
cannot be consumed by one frozen scorer batch. Examples include inconsistent
frame counts, spatial grids, dtypes, or optional scene-flow/ray fields. Such a
mismatch should be discovered before a scorer calls `stack`, and it should not be
misreported as provider-competence evidence.

`prob4d.prediction_batch_preflight` adds a content-addressed infrastructure
boundary for that purpose. It is deliberately separate from provider accuracy,
calibration, BayesianPhysTwin admission, and Causal4D intervention evaluation.

## Causal information boundary

The preflight loads the provider-neutral prediction manifest first and selects
payloads from manifest metadata. With an exclusive cutoff `c`, a payload is
opened only when all of its declared source dependencies satisfy the manifest's
causal-admission rule. Excluded future payloads are never hashed, decoded, or
validated by the batch preflight.

The resulting artifact always records:

```json
{
  "future_prediction_payloads_opened": 0
}
```

A missing or malformed post-cutoff payload therefore cannot alter a valid prefix
preflight. A selected payload with a byte-count, SHA-256, schema, window-ID,
frame-lineage, optional-field, or storage-dtype integrity failure raises
`PredictionBatchIntegrityError`; it is not converted into a scientific negative.

## Default compatibility policy

The default `PredictionBatchPolicyV1` requires:

- at least one causally admitted payload;
- a common frame count;
- a common spatial point-grid shape;
- a common point dtype; and
- a common scene-flow/ray presence signature.

The policy is embedded in the artifact. A prospectively defined ragged scorer can
explicitly relax one or more requirements. Relaxation does not pad, truncate, or
otherwise change payload bytes; it only states that the downstream scorer was
designed for that representation.

## Command-line use

Build a preflight artifact before source or target scoring:

```bash
python -m prob4d.prediction_batch_preflight build \
  outputs/provider/provider-neutral.json \
  outputs/provider/prediction-batch-preflight.json \
  --causal-frame-stop 134
```

Exit status `0` means the admitted payloads satisfy the declared policy. Exit
status `2` is a valid, persisted `batch-incompatible` result. Integrity or schema
errors fail instead of being relabelled as incompatibility.

A scorer with an already frozen ragged spatial representation can declare that
capability explicitly:

```bash
python -m prob4d.prediction_batch_preflight build \
  outputs/provider/provider-neutral.json \
  outputs/provider/prediction-batch-preflight.json \
  --causal-frame-stop 134 \
  --allow-ragged-spatial
```

Verify a persisted artifact with:

```bash
python -m prob4d.prediction_batch_preflight verify \
  outputs/provider/prediction-batch-preflight.json
```

## Structured violations

A mismatch is retained with the exact reference and conflicting payload IDs, the
field, expected value, observed value, and a stable code such as:

- `no-causally-admitted-payloads`;
- `frame-count-mismatch`;
- `spatial-shape-mismatch`;
- `point-dtype-mismatch`; or
- `optional-field-mismatch`.

This makes failures actionable without requiring a downstream NumPy traceback.
It also prevents a software-terminal run from being mistaken for a negative
provider result.

## Terminal evidence

A protocol that stops at this boundary can bind the preflight artifact in a
`ProviderTerminalDecisionV1` with classification `batch-incompatible`. That
decision must authorize no scientific inference and must explicitly forbid
provider competence, calibration, BayesianPhysTwin benefit, Causal4D benefit,
deployment-safety, and state-of-the-art claims. See
[provider terminal decisions](provider-terminal-decision.md).

## Scientific boundary

Passing preflight proves only that the admitted prediction bytes satisfy one
prospectively declared scorer representation. It does not establish accuracy,
calibration, statistical independence, physical-query relevance, guarded update
benefit, intervention benefit, deployment safety, or state of the art.
