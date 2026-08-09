# Replay-complete held-out provider evidence

`prob4d.heldout_provider_evidence` composes the complete target-blind selection
record and the held-out Prob4D-to-BayesianPhysTwin promotion record into one
portable evidence-v2 artifact.

The existing artifacts remain authoritative and independently versioned. The
new artifact does not replace them. It verifies that they belong to the same
experiment, group split, selected candidate, provider bytes, target decisions,
bootstrap plan, and deterministic promotion result.

## Bound inputs

A `HeldoutProviderEvidenceV2` embeds and verifies:

- the complete `SelectionEvidenceBundleV2`, including every
  calibration-group-by-candidate metric row, deterministic ordering, thresholds,
  constraints, tie breaks, and target deployment decisions;
- the target-free `HeldoutProviderPromotionLockV1`, including the complete
  development/calibration/target roster and frozen bootstrap seed and resample
  count;
- an explicit binding from the selected candidate and its method identifier to
  the frozen primary promotion arm;
- the exact UTF-8 provider-report JSON bytes, not only a parsed mapping;
- the sealed target group-by-arm `HeldoutPromotionQueryResultsV1`; and
- the retained `HeldoutProviderPromotionReportV1`.

The resulting descriptor has a content identity. Its compact replay receipt has
a separate identity.

## Independent replay

Loading the artifact performs all of the following without importing an
experiment runner:

1. replay the complete calibration candidate order;
2. verify the calibration roster equals the frozen lock;
3. verify deployment decisions cover exactly the frozen target groups;
4. verify the selected candidate binds the declared primary arm and provider or
   query method;
5. parse the embedded provider bytes with duplicate-key and non-finite-value
   rejection;
6. verify their SHA-256 identity against the retained promotion report;
7. match every primary-arm acceptance, deployed artifact, fallback artifact, and
   rejected-update fallback decision to the selection evidence;
8. recompute the provider and guarded-query gates using the frozen bootstrap
   settings; and
9. require byte-for-byte equality with the retained promotion report.

A changed whitespace byte in the embedded provider report changes its digest and
fails replay even when the parsed JSON values are otherwise identical.

## Pack an artifact

The inputs must already have passed their ordinary validators.

```bash
python -m prob4d.heldout_provider_evidence pack \
  --selection-evidence selection-evidence.json \
  --promotion-lock promotion-lock.json \
  --provider-report provider-report.json \
  --query-results query_results.sealed.json \
  --promotion-report promotion_report.json \
  --arm-id identity \
  --method-role provider \
  --output heldout-provider-evidence-v2.json
```

`--method-role provider` requires the selected candidate's `method_id` to equal
the bound arm's `provider_method_id`. Use `query` when calibration selects the
BayesianPhysTwin query method instead.

Publication is no-clobber by default.

## Verify and print the replay receipt

```bash
python -m prob4d.heldout_provider_evidence verify \
  heldout-provider-evidence-v2.json
```

The receipt reports:

- selection and replay identities;
- complete candidate order and selected candidate;
- calibration and target group counts;
- bootstrap resample count and seed;
- accepted, rejected, and exact-fallback counts; and
- provider, query, and overall promotion decisions.

## Claim boundary

A passing replay establishes that the retained target-blind selection and
held-out decision are internally consistent for the exact frozen cohort and
artifact bytes. It does not establish Causal4D intervention benefit, deployment
safety, generalization beyond the cohort, or state of the art.
