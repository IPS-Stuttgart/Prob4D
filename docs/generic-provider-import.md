# Generic external-provider import

Prob4D can ingest canonical prediction windows from an external 4-D provider
without adding that provider's runtime, model weights, or dependency stack to the
Prob4D package.

The boundary has two steps:

```text
external provider
    -> versioned PredictionWindow NPZ files + strict import specification
    -> PredictionProviderManifestV1
```

The resulting neutral manifest standardizes provenance and causal admission. It
does not establish provider accuracy, calibration, independence, or downstream
BayesianPhysTwin benefit.

## Create a scaffold

```bash
prob4d prediction scaffold-generic \
  outputs/sequence-a/external-provider
```

The command creates a new directory containing:

```text
README.md
provider-import.json
windows/
```

It refuses to replace an existing directory. The generated JSON is intentionally
not admissible: every `REPLACE_WITH_...` value must be replaced with exact
provenance before import.

## Export canonical payloads

Each payload must be a versioned `PredictionWindow` archive. The provider owns
inference and writes these files; Prob4D owns validation and conversion into the
portable manifest.

The specification declares:

- exact provider repository and revision;
- provider-run, model-set, and loader content identities;
- coordinate, point, flow, and ray semantics;
- product role, view, stochastic-member identity, and dependence groups;
- one confined relative payload path; and
- one exclusive causal source-frame interval for every output frame.

A complete static example is retained at
[`examples/provider-neutral-import-spec.json`](examples/provider-neutral-import-spec.json).

## Import and verify

```bash
prob4d prediction import-generic \
  outputs/sequence-a/external-provider/provider-import.json \
  outputs/sequence-a/external-provider/provider-neutral.json

prob4d prediction validate \
  outputs/sequence-a/external-provider/provider-neutral.json \
  --causal-frame-stop 134
```

Prob4D derives the payload SHA-256, byte count, dense storage precision, and
optional-field declarations from the exact NPZ bytes. It parses private byte
snapshots and rechecks the source files before publication.

The importer rejects:

- absolute paths, parent traversal, and symbolic-link traversal;
- duplicate JSON keys, non-finite values, and unknown fields;
- malformed provider, revision, run, model-set, or loader identities;
- caller-controlled importer metadata;
- payload mutation during import;
- window-ID or output-frame lineage mismatch; and
- an output manifest outside the confined payload bundle.

After writing the neutral manifest, the importer immediately performs full
payload verification. A repeated identical publication is idempotent; a different
manifest at the same path is rejected by the canonical manifest writer.

## Python API

```python
from prob4d.prediction_provider_import import (
    import_prediction_provider_specification,
)
from prob4d.prediction_provider_scaffold import (
    scaffold_prediction_provider_import,
)
```

## Downstream boundary

A valid manifest proves byte-level interoperability, declared source lineage,
and causal admission only. Source/calibration-only covariance, reliability,
identity, and bias modelling remain separate steps. BayesianPhysTwin still owns
the guarded physical update and exact fallback, and Causal4D should consume only
the selected BayesianPhysTwin belief rather than raw provider output.
