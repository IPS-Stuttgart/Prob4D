# CUT3R common-support source competence v2

`prob4d.cut3r_source_competence_v2` is an additive claim-integrity layer for the
frozen CUT3R source comparison. It preserves the existing comparison lock,
`SourceProviderCompetencePolicyV1`, stable v1 report, and readiness order while
adding two requirements:

1. candidate and baseline primary metrics must use exactly the same frozen
   support; and
2. seam, drift, association, identity retention, and support retention are
   retained and evaluated as paired endpoints.

This layer is source-only. It does not execute CUT3R, open a confirmation object,
authorize a target run, or change the production Prob4D estimator.

## Why exact support identity is required

A scalar error and a retained-count value do not prove that two arms were scored
on the same material points. Equal-sized but different support sets can make an
apparently paired contrast unpaired. Version 2 therefore requires one
`metric_support` object in every frame-level arm record:

```json
{
  "point_support_sha256": "<sha256>",
  "point_support_count": 128,
  "endpoint_support_sha256": "<sha256>",
  "endpoint_support_count": 1,
  "proper_score_support_sha256": "<sha256>",
  "proper_score_dimension": 384,
  "proper_score_semantics": "arm-neutral-fixed-scale-gaussian-score-v1",
  "seam_support_sha256": "<sha256-or-null>",
  "seam_support_count": 64
}
```

For each frozen `(group, case, frame, seed)` tuple, the candidate and baseline
objects must be byte-identical after canonicalization. This binds the row set,
row order, dimension, and proper-score convention. A shifted, filtered, or
arm-specific scoring support fails before aggregation.

The exact canonical row identifiers for the Deform360 execution are frozen in
`protocols/cut3r_deform360_common_support_definition_v2.json`.

## Arm-neutral source-mean proper score

The source-mean gate asks whether the provider mean is useful. It must not fail
only because an otherwise useful mean has a poor arm-specific covariance model.
The v2 lock therefore permits only:

```text
arm-neutral-fixed-scale-gaussian-score-v1
```

The scale or covariance is fitted from development/calibration information and
is byte-identical across both arms. Joint NLL, normalized NEES, coverage, width,
shared-subspace energy, and conditional-subspace energy remain in the later
covariance-localization gate.

## Freeze the additive lock

First build the existing v1 source-competence lock with the exact provider
manifests and the `source_competence_policy` contained in the v2 specification.
Then freeze the common-support supplement before source-evaluation scores are
read:

```bash
python -m prob4d.cut3r_source_competence_v2 freeze \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  protocols/cut3r_deform360_source_competence_v2_spec.json \
  --output outputs/cut3r/common-support-lock-v2.json

python -m prob4d.cut3r_source_competence_v2 verify-lock \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json
```

The supplement binds the existing comparison and v1 lock identities, exact v1
policy, source groups, seeds, contrast, record-definition digest, common-support
definition digest, proper-score semantics, and paired decision margins.

For the frozen Deform360 source comparison it additionally requires:

- all four source-evaluation object sessions to be evaluable;
- zero permitted technical failures;
- no mean point, endpoint, or fixed-scale proper-score regression;
- no mean seam or drift regression and at most 20% worst-group regression;
- mean association, identity, and support loss no worse than 0.02;
- worst-group loss no worse than 0.05; and
- at least three of four groups passing each paired endpoint family.

The four independent source-evaluation groups remain:

- `036-napkin-cloth-episode-0009`;
- `058-roll-napkin-episode-0001`;
- `152-slime-episode-0008`; and
- `198-kneepad-cloth-episode-0002`.

Frames, cameras, seeds, points, and support rows remain nested observations.

## Version-2 record envelope

The v2 record artifact retains the complete v1 record values and adds the
common-support lock and support-definition identities:

```json
{
  "schema": "prob4d.cut3r-source-competence-records",
  "schema_version": 2,
  "comparison_lock_id": "<sha256>",
  "source_competence_lock_id": "<sha256>",
  "common_support_lock_id": "<sha256>",
  "record_definition_sha256": "<sha256>",
  "common_support_definition_sha256": "<sha256>",
  "source_truth_used": true,
  "target_payloads_opened": false,
  "target_outcomes_opened": false,
  "group_failures": [],
  "records": []
}
```

The implementation strips only the v2 support envelope and sends the resulting
records through the stable v1 builder. The v1 builder remains authoritative for
complete-roster checks, arm-neutral reference counts, positive denominators,
hierarchical weighting, and the stable `SourceProviderCompetenceReportV1`.

## Build and verify the report

```bash
python -m prob4d.cut3r_source_competence_v2 report \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-records-v2.json \
  --output outputs/cut3r/source-competence-v2.json \
  --v1-output outputs/cut3r/source-provider-competence-v1.json \
  --require-pass

python -m prob4d.cut3r_source_competence_v2 verify-report \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-records-v2.json \
  outputs/cut3r/source-competence-v2.json \
  --require-pass
```

A valid negative report is written before exit status `3` is returned. The
self-contained v2 report includes the exact stable v1 report, both-arm metrics
for every complete group, paired ratios and deltas, group-level reasons,
aggregate decisions, and all content identities.

## Readiness gates

Emit readiness-compatible source-mean and identity gates with:

```bash
python -m prob4d.cut3r_source_competence_v2 gates \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-records-v2.json \
  outputs/cut3r/source-competence-v2.json
```

Identity is `not-evaluated` after a source-mean failure. A v1 absolute pass cannot
rescue a v2 common-support or paired-endpoint failure.

## Ordered next action

After the lock is frozen, execute exactly the three already registered causal
arms on the retained Deform360 source inputs:

1. `native-continuous`;
2. `restarted-newest`; and
3. `restarted-prob4d-fused`.

The primary Prob4D contrast remains `restarted-prob4d-fused` versus
`restarted-newest`. Stop at the first negative ordered gate. Do not open any of
the twelve confirmation object sessions unless every target-free readiness gate
returns `ready-for-one-target-evaluation`.
