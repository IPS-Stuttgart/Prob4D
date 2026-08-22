# CUT3R source-competence support audit

`prob4d.cut3r_source_competence_audit` is an additive source-only verifier for the
merged CUT3R common-support v2 protocol. Version 2 proves that the candidate and
baseline *declare* identical support envelopes. The audit closes the stronger
integrity gap: it reconstructs those envelopes from retained canonical row
identifiers and compares the result independently with each scored arm.

The audit also binds the exact bytes and semantic artifact identity of the
arm-neutral fixed-scale proper-score reference. The score reference must be fit
using development/calibration information only. A semantics label without an
exact reference identity is not sufficient for a claim-bearing source-mean gate.

Existing v2 locks, records, reports, and valid negative outcomes remain unchanged
and replayable. Once this audit is frozen for an execution, only its audited
source-mean and identity/reliability gates are claim-bearing.

## Evidence chain

The complete source-only chain is:

```text
CUT3R comparison lock
  -> source competence v1 lock
  -> common-support v2 lock
  -> support-audit lock + exact proper-score reference bytes
  -> canonical metric-support manifest
  -> common-support v2 records and report
  -> replay-complete support-audit report
  -> audited source-mean and identity/reliability gates
```

The support-audit lock binds:

- the comparison, v1 source-competence, and v2 common-support lock identities;
- the record and common-support definition identities;
- the exact source object/session roster, seeds, and causal contrast;
- `arm-neutral-fixed-scale-gaussian-score-v1` semantics;
- one exact `proper_score_reference_artifact_id`;
- the SHA-256 digest of the exact proper-score reference file bytes;
- the `development-and-calibration-only` fit scope; and
- the complete-manifest and target-closed claim boundary.

## Freeze before opening source scores

Prepare an audit specification from
`docs/examples/cut3r-source-competence-audit-spec.json`. Replace every example
digest with the exact content identity from the frozen execution. Then freeze the
lock while the proper-score reference file is still source/target separated:

```bash
python -m prob4d.cut3r_source_competence_audit freeze \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-audit-spec.json \
  outputs/cut3r/arm-neutral-score-reference.json \
  --output outputs/cut3r/source-competence-support-audit-lock.json

python -m prob4d.cut3r_source_competence_audit verify-lock \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-support-audit-lock.json \
  outputs/cut3r/arm-neutral-score-reference.json
```

`verify-lock` reads the reference file again and fails if any byte differs. A
symlinked or empty reference is rejected.

## Canonical row manifest

The manifest input contains one arm-neutral entry for every frozen

```text
(group_id, case_id, frame_index, random_seed)
```

pair. Each entry retains four ordered arrays:

```json
{
  "group_id": "object-session-id",
  "case_id": "camera-or-case-id",
  "frame_index": 42,
  "random_seed": 7,
  "point_rows": [],
  "endpoint_rows": [],
  "proper_score_rows": [],
  "seam_rows": []
}
```

The row shapes are frozen by
`protocols/cut3r_deform360_common_support_definition_v2.json`:

```text
point:
  [group_id, case_id, frame_index, material_identity_id, coordinate_frame_id]
endpoint:
  [group_id, case_id, frame_index, endpoint_role,
   material_identity_id, coordinate_frame_id]
proper score:
  [group_id, case_id, frame_index, material_identity_id,
   coordinate_axis, coordinate_frame_id]
seam:
  [group_id, case_id, frame_index, window_left_id, window_right_id,
   material_identity_id, coordinate_frame_id]
```

The auditor rejects duplicate rows, row/envelope disagreement, endpoint or seam
rows outside point support, score rows outside point support, inconsistent
coordinate axes, missing pairs, extra pairs, and target access. Every point must
contribute exactly three distinct scalar coordinate axes to the proper score.
Array order is preserved and hashed; reordering an otherwise identical set is a
support change.

Build and verify the manifest:

```bash
python -m prob4d.cut3r_source_competence_audit manifest \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-support-audit-lock.json \
  outputs/cut3r/metric-support-manifest-input.json \
  --output outputs/cut3r/metric-support-manifest.json

python -m prob4d.cut3r_source_competence_audit verify-manifest \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-support-audit-lock.json \
  outputs/cut3r/metric-support-manifest.json
```

## Build the audited receipt

First produce the ordinary v2 report. Then independently bind the retained rows,
score reference, v2 records, and v2 decision:

```bash
python -m prob4d.cut3r_source_competence_audit report \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-support-audit-lock.json \
  outputs/cut3r/source-competence-records-v2.json \
  outputs/cut3r/metric-support-manifest.json \
  outputs/cut3r/source-competence-v2.json \
  --output outputs/cut3r/source-competence-support-audit-report.json \
  --require-pass

python -m prob4d.cut3r_source_competence_audit verify-report \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-support-audit-lock.json \
  outputs/cut3r/source-competence-records-v2.json \
  outputs/cut3r/metric-support-manifest.json \
  outputs/cut3r/source-competence-v2.json \
  outputs/cut3r/source-competence-support-audit-report.json \
  --require-pass
```

The report is emitted for both positive and valid negative v2 outcomes. A support
or reference mismatch fails before publication because it invalidates the claimed
metric construction. `--require-pass` returns exit status `3` only after a valid,
fully audited negative source-competence report has been retained.

## Audited readiness gates

```bash
python -m prob4d.cut3r_source_competence_audit gates \
  outputs/cut3r/cut3r-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/common-support-lock-v2.json \
  outputs/cut3r/source-competence-support-audit-lock.json \
  outputs/cut3r/source-competence-records-v2.json \
  outputs/cut3r/metric-support-manifest.json \
  outputs/cut3r/source-competence-v2.json \
  outputs/cut3r/source-competence-support-audit-report.json
```

Both evaluated gates use the support-audit report ID as their evidence ID. Their
metadata retains the common-support lock ID, v2 records and report IDs, manifest
and audit-lock IDs, and both proper-score reference identities. Identity remains
`not-evaluated` after a source-mean failure. Downstream readiness must not
substitute the unaudited v2 gate evidence once the audit lock exists.

## Scientific boundary

This audit does not improve CUT3R predictions or estimate another covariance. It
makes the registered source comparison falsifiable at the row and score-reference
level. The next action after a passing audited source result remains the ordered
gauge/dependence, nonlinear-closure, conditional-covariance, physical-query, and
single frozen target gates. A negative result must stop at its first failed gate.
