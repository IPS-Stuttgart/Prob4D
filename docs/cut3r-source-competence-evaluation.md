# CUT3R source competence evaluation

`prob4d prediction cut3r-source-competence` converts complete paired source
records from one frozen CUT3R comparison into the existing
`SourceProviderCompetenceReportV1` decision. It closes the gap between provider
execution and the ordered source-mean and identity/reliability gates without
changing a provider, contrast, cohort, policy, or target-access boundary.

The primary intended contrast is `restarted-prob4d-fused` versus
`restarted-newest`. Both arms use the same restarted CUT3R windows, so the report
isolates the source-side value of Prob4D fusion. A separately frozen lock may use
the registered `native-continuous` versus `restarted-newest` contrast to measure
CUT3R recurrent-state value. Noncausal or disabled contrasts are rejected.

## Freeze before source scores

Build the source-competence lock after the CUT3R comparison lock is final and
before opening source-evaluation scores:

```bash
prob4d prediction cut3r-source-competence freeze \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/source-competence-spec.json \
  --output outputs/cut3r/source-competence-lock.json

prob4d prediction cut3r-source-competence verify-lock \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json
```

The lock binds:

- the exact CUT3R comparison lock and source object/session roster;
- one enabled, causal, claim-eligible registered contrast;
- the candidate and baseline provider-manifest identities;
- the source cohort binding and group definition;
- a SHA-256 identity for the complete record-generation definition;
- the frozen `SourceProviderCompetencePolicyV1`; and
- the hierarchical weighting and target-closed claim boundary.

The candidate and baseline provider manifests must be distinct. The policy may
not require more evaluable groups than the frozen source roster contains.

## Complete paired records

The strict JSON record artifact contains one row for every frozen

```text
(group, case, evaluation frame, random seed, contrast arm)
```

combination. Every nonfailed source group must retain every frozen frame, seed,
and both contrast arms. A row contains:

- point and endpoint error in metres;
- proper score;
- an optional seam error in metres;
- correct and predicted association counts;
- retained and reference identity counts; and
- retained and reference support counts.

The record-definition digest owns the exact geometric correspondence, point and
endpoint definitions, proper-score convention, seam-frame definition, and count
construction. The aggregator does not infer these semantics from field names.
It verifies only their complete, finite, target-closed, paired representation.

Identity and support reference counts are arm neutral and therefore must match
between paired arms. Denominators must be positive after aggregation, and every
case/seed/arm must contain at least one seam observation. Evaluation intervals
must contain at least two frames so drift is identifiable.

A complete source object/session may instead carry one predeclared technical
failure. Such a group contains no scored rows and remains visible to the frozen
technical-failure policy. It is never silently dropped.

## Hierarchical aggregation

Nested records do not become independent evidence. The report computes:

1. equal-frame means inside each frozen seed and case;
2. equal-seed means inside each case;
3. equal-case means inside each complete object/session; and
4. the existing equal-group decision across complete objects/sessions.

Point, endpoint, and seam RMSE are formed from the corresponding hierarchically
averaged squared errors. Drift is the absolute least-squares slope of point error
against source frame index within each case and seed. Association precision,
identity retention, and support retention are first formed from summed counts
within each case and seed, then receive the same equal-seed and equal-case
weighting.

## Build, verify, and summarize

```bash
prob4d prediction cut3r-source-competence report \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/source-competence-records.json \
  --output outputs/cut3r/source-provider-competence.json \
  --require-pass

prob4d prediction cut3r-source-competence verify-report \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/source-competence-records.json \
  outputs/cut3r/source-provider-competence.json \
  --require-pass

prob4d prediction cut3r-source-competence summarize \
  outputs/cut3r/source-comparison-lock.json \
  outputs/cut3r/source-competence-lock.json \
  outputs/cut3r/source-competence-records.json \
  outputs/cut3r/source-provider-competence.json \
  --json
```

A valid negative report is always written. `--require-pass` returns exit status
`3` after publication when either source mean quality or identity/reliability
does not pass. The report remains the ordinary replay-complete
`SourceProviderCompetenceReportV1`, so the existing fresh-provider readiness
logic consumes it without another translation layer.

## Position in the qualification order

This command implements the source-mean and identity/reliability decision in the
registered order:

```text
support
  -> source means and identities
  -> gauge/dependence
  -> linearization closure
  -> conditional point covariance
  -> physical-query relevance
  -> one frozen target evaluation
```

A source-mean negative is terminal for that provider version. An identity
negative may motivate a separately versioned identity method. Neither result may
be repaired by covariance fitting or downstream BayesianPhysTwin performance on
the same opened source groups.

## Claim boundary

This is source-only provider-competence evidence. It does not establish target
transfer, calibrated uncertainty on an independent cohort, BayesianPhysTwin
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
