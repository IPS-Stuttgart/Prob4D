# Provider execution attestation

Prob4D can bind how an external 4-D provider was executed without importing the
provider, its model weights, or its runtime dependencies. The content-addressed
version-1 `prob4d.provider-execution-attestation` sidecar records exact code and
model identities, command arguments, causal declarations, runtime fingerprints,
and input/output bytes.

This contract complements `PredictionProviderManifestV1`:

```text
external provider execution
    -> provider execution attestation
    -> canonical PredictionWindow payloads
    -> PredictionProviderManifestV1
    -> Prob4D readiness gates
```

A valid attestation establishes the integrity of the attestation itself and binds
the declared identities of referenced artifacts and execution lineage. Prob4D
does not reopen those external bytes through this contract. The attestation does
not establish provider accuracy, calibrated uncertainty, BayesianPhysTwin
benefit, Causal4D intervention benefit, target authorization, or deployment
safety.

## Evidence modes

The contract keeps two evidence levels explicit:

- `declarative-only-v1` records a declaration without claiming wrapper-observed
  execution bytes. `execution_evidence_sha256` must be `null`.
- `wrapper-observed-v1` binds the exact output of an execution wrapper or external
  attestation through `execution_evidence_sha256`.

`execution_evidence_complete` is derived rather than caller-controlled. It is
true only when the execution succeeded, wrapper-observed evidence is bound, an
immutable runtime identity is present, at least one input and output artifact is
bound, and the resulting provider-manifest semantic ID is recorded. Completeness
is provenance evidence only; it is not provider admission.

## Privacy-preserving runtime identity

Raw environment-variable values are not stored. Each allow-listed variable is
recorded as a name and SHA-256 digest of its exact value. Do not include secret
names or publish digests where the value has a small guessable domain unless that
risk is acceptable for the execution environment.

Command arguments, artifact names, timestamps, and metadata are retained
verbatim. They must not contain credentials, access tokens, private paths, or
other protected values.

The runtime record binds:

- Python version and implementation;
- platform identity;
- an optional OCI/container digest in `sha256:<digest>` form; and
- an optional environment-lock digest.

At least one container or environment-lock digest is required for complete
wrapper-observed evidence.

## Specification

Start from the checked-in
[example provider-execution specification](examples/provider-execution-attestation-spec.json).

Important fields are:

- exact provider repository and revision;
- provider-run, model-set, and loader content IDs;
- the complete argument vector, without shell interpolation;
- source-order, online-prefix, revisit, global-alignment, and future-processing
  declarations;
- hashed input and output artifact rosters with byte counts;
- execution start and completion times in canonical UTC `Z` form;
- terminal status; and
- the semantic `PredictionProviderManifestV1` ID, when already available.

Artifact and environment-variable records are sorted canonically by name.
Duplicate names, unknown fields, non-finite JSON, coercive Boolean/integer aliases,
invalid revisions, time reversal, and content-ID mismatches fail closed.

## Create and verify

The initial additive command is available through the module entry point so the
stable `prob4d.api.v2` export inventory remains unchanged:

```bash
python -m prob4d.provider_execution_attestation create \
  docs/examples/provider-execution-attestation-spec.json \
  --output outputs/provider-execution-attestation.json

python -m prob4d.provider_execution_attestation verify \
  outputs/provider-execution-attestation.json \
  --require-complete
```

Publication is atomic and no-clobber. Repeating an identical write is idempotent;
a different artifact at the same destination is rejected.

## Recommended wrapper order

For CUT3R or another external provider, the execution wrapper should:

1. snapshot the exact provider revision, checkpoint, input, loader, and runtime
   identities;
2. record the argument vector and causal declarations before inference;
3. execute the provider;
4. hash every retained input, wrapper log, and generated output byte;
5. import or produce the canonical provider-neutral manifest;
6. bind that manifest's semantic ID into the final attestation; and
7. publish the attestation and provider manifest without replacing existing
   different bytes.

Prob4D can verify this sidecar but cannot infer trust in the external wrapper.
Claim-bearing experiments must separately attest the runner or build provenance,
freeze source/calibration gates, and retain exact fallback in BayesianPhysTwin.
