# Provider-neutral evaluation identity

Claim-bearing `prob4d evaluate provider` inputs may use any prediction provider
that is represented by `PredictionProviderManifestV1`. A fused prediction archive
binds that provider through the `provider_identity` object inside its artifact
metadata. CUT3R, VGGT, MotionCrafter, and future providers use the same evaluator;
provider-specific field names are not required.

## Version 1 record

```json
{
  "provider_identity": {
    "schema_name": "prob4d.provider-evaluation-provider-identity",
    "schema_version": 1,
    "provider_manifest_id": "<sha256 content identity>",
    "provider_manifest_sha256": "<sha256 exact manifest bytes>",
    "provider_family": "cut3r",
    "provider_repository": "naver/CUT3R",
    "provider_revision": "<exact 40- or 64-character revision>",
    "provider_run_id": "<sha256 run identity>",
    "model_set_id": "<sha256 model-set identity>",
    "loader_id": "<sha256 loader identity>",
    "coordinate_semantics": "sequence-local-sim3",
    "point_semantics": "dense-point-map",
    "flow_semantics": "absent",
    "ray_semantics": "absent",
    "source_dependency_semantics": "per-output-exclusive-source-frame-interval-v1"
  }
}
```

The values and allowed semantics are shared with
`PredictionProviderManifestV1`. The evaluator rejects unknown fields, malformed
revisions or digests, unsupported coordinate/payload semantics, and a changed
source-dependency contract.

Three values are intentionally case-local:

- `provider_manifest_id`;
- `provider_manifest_sha256`;
- `provider_run_id`.

They authenticate the exact source sequence and execution used by one case, so
they normally differ across held-out objects or sessions. The cross-case method
signature instead freezes provider family, repository, revision, model set,
loader, coordinate and payload semantics, source-dependency semantics, Prob4D
revision, fusion/covariance meaning, gauge estimator, and calibration identities.
This separates legitimate case identity changes from method drift.

## Historical MotionCrafter replay

Existing MotionCrafter fused artifacts remain readable without rewriting their
bytes or identifiers. When `provider_identity` is absent, evaluation validates
the historical fields:

- `motioncrafter_revision`;
- `motioncrafter_model_set_sha256`;
- `motioncrafter_seed_policy`;
- `prediction_manifest_sha256`.

This adapter is replay-only. New provider exports should emit the generic record.
A MotionCrafter export may temporarily retain all four historical fields alongside
the generic record, but the revision, model set, and manifest digest must agree
exactly. Partial or contradictory mirrors fail closed.

## Claim boundary

The record authenticates provider and execution semantics for observation-level
evaluation. It does not establish source-mean competence, calibrated uncertainty,
BayesianPhysTwin update acceptance, physical-query benefit, Causal4D intervention
benefit, deployment safety, or state of the art. Those remain separate gates.
