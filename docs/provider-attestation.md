# Provider-v2 attestation contract

Every artifact produced through `prob4d.provider_v2` embeds a versioned
`prob4d.provider-attestation` record in
`metadata.prob4d_provider_attestation`. The observation artifact content address
covers this record, including the complete provider manifest.

The purpose is not to make a self-signed producer statement magically trusted.
It is to make the exact statement complete, immutable, and independently
checkable by Bayesian-PhysTwin or Causal4D without importing Prob4D.

## Schema version 1

The attestation binds:

- provider API version, repository revision, and Python import boundary;
- the complete content-addressed provider-v2 capability manifest;
- calibrated versus exploratory export mode;
- whether prediction/calibration compatibility was checked;
- gauge and point covariance-calibration artifact IDs;
- covariance-root and composition-Jacobian modes; and
- runtime-revision evidence, including source, observed commit, checkout
  cleanliness, match status, and independent-verification status.

Consumers recompute `provider_manifest_id` from the embedded manifest after
removing its own ID field. They then validate the known provider name, API
version, source repository, import boundary, observation schemas, limitations,
and the required provider-v2 capabilities.

## Claim-bearing requirements

A claim-bearing version-1 attestation must declare all of the following:

```text
export_mode = calibrated
claim_bearing = true
calibration_compatibility_validated = true
covariance_root_mode = canonical_eigenspaces
composition_jacobian_mode = analytic
```

Both calibration artifact IDs must be lowercase SHA-256 digests. Runtime evidence
must match the observation's exact Prob4D revision and must have been independently
resolved from installed VCS metadata or a clean source checkout. An environment-only
revision assertion remains recordable for exploratory deployment but cannot satisfy
a claim-bearing attestation.

## Compatibility boundary

Provider-v1 artifacts remain valid for frozen reproduction and do not acquire a
provider-v2 attestation retroactively. Consumers should:

1. continue validating the neutral observation and causal-stream contracts;
2. validate a provider attestation whenever one is present; and
3. explicitly require a claim-bearing provider-v2 attestation for new prospective
   Prob4D-to-Bayesian-PhysTwin evidence.

Adding an undeclared field, changing the embedded manifest, weakening a required
capability, changing a calibration ID, or contradicting runtime evidence changes
the observation artifact ID and fails strict attestation validation.

## Python validation

```python
from prob4d.provider_v2 import validate_provider_attestation

validated = validate_provider_attestation(
    observation.metadata["prob4d_provider_attestation"],
    source_revision=observation.source_revision,
    require_claim_bearing=True,
)
```

Downstream repositories intentionally reimplement the small JSON/hash validator so
they can reject an inconsistent provider artifact before importing or trusting the
producer package.
