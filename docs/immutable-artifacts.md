# Immutable in-memory artifact values

Prob4D validates several values before computing content identities, sealing
causal lineage, or admitting an observation into a claim-bearing workflow. Those
values must not change after validation.

## JSON values

`frozen_finite_json_mapping` returns recursively read-only `Mapping` and
`Sequence` implementations. They do not inherit from `dict` or `list`, so Python
base-class descriptors cannot bypass their mutation guards. Call `plain_json`
when a mutable built-in representation is required for serialization or explicit
export.

The following are rejected:

```python
metadata["changed"] = True
metadata["items"].append("changed")
dict.__setitem__(metadata, "changed", True)
list.append(metadata["items"], "changed")
```

`copy.copy`, `copy.deepcopy`, and the explicit `.copy()` methods return ordinary
mutable JSON containers without changing the frozen source value.

## NumPy values

`PredictionWindow` arrays are defensively copied and rebuilt over immutable
`bytes` storage after validation. Direct assignment fails, and callers cannot
restore writeability with `array.setflags(write=True)`.

This is stronger than clearing the write flag on an owned NumPy allocation. An
owned allocation can normally have that flag restored later, which would allow a
validated value to diverge from the bytes or identity that were audited.

The helper rejects object-dtype arrays. Numeric and Boolean dtypes, scalar and
empty shapes, and C-order values are retained exactly.

## Scope

This contract applies to portable or claim-bearing values whose post-validation
mutation could invalidate provenance or content identity. Large execution-only
buffers, memory-mapped stores, and private fusion ownership paths retain their
separate performance contracts and are not silently copied by this change.

Passing these integrity checks is not evidence of provider competence,
uncertainty calibration, physical-state identifiability, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
