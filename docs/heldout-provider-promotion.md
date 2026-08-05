# Held-out provider and BayesianPhysTwin promotion gate

`prob4d experiment heldout-provider` coordinates the decisive real-data gate
without introducing a new estimator. It composes an independently generated
Prob4D provider-competence report with complete guarded BayesianPhysTwin query
outcomes under one target-free, content-addressed protocol.

The statistical unit is a complete physical object or acquisition session. A
frame, point, track, or already-opened output directory is not an independent
held-out unit.

## Three-stage workflow

### 1. Freeze before target access

Start from the documented configuration skeleton and replace every placeholder
revision, digest, group, method, margin, and metadata value:

```bash
cp docs/examples/heldout-provider-promotion-config.json protocol.json
prob4d experiment heldout-provider freeze protocol.json \
  --output promotion-lock.json
```

The lock requires disjoint sorted development, calibration, and target groups.
It binds the exact Prob4D, BayesianPhysTwin, and MotionCrafter revisions; immutable
model and run-spec identities; the provider-evaluation manifest; all calibration,
selection, and guard artifacts; the bootstrap unit and seed; and every decision
margin.

Exactly one arm is required for each registered role:

1. unchanged physical fallback;
2. simple visual baseline;
3. persistent identities with row-wise gauge marginalization;
4. framewise observations with the complete explicit joint gauge nuisance;
5. persistent observations with the complete explicit joint gauge nuisance;
6. cross-window identity marginalization with the complete joint gauge nuisance;
7. an independently anchored, visibly sensor-assisted arm.

Additional diagnostic arms are permitted, but the primary query arm must be a
non-sensor, non-diagnostic candidate. The physical fallback has no provider
method because it is not a visual observation source.

### 2. Run once on the frozen target

First run `prob4d evaluate provider` with a decision-bearing schema-v2 manifest.
The resulting provider report must be schema version 3, use common support, reject
legacy artifacts, use no oracle alignment, and match the frozen target groups,
methods, reference, bootstrap count, seed, and manifest digest exactly.

BayesianPhysTwin then writes one raw row for every target-group/arm pair:

```json
{
  "promotion_lock_id": "<promotion-lock SHA-256>",
  "rows": [
    {
      "group_id": "target-object-01",
      "arm_id": "fallback",
      "query_rmse_mm": 5.1,
      "deployed_artifact_id": "<physical-fallback SHA-256>",
      "fallback_artifact_id": "<physical-fallback SHA-256>",
      "accepted": null,
      "exact_fallback_reproduced": null,
      "accepted_coverage": null,
      "accepted_width_mm": null,
      "technical_failure": false,
      "technical_failure_reason": null,
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

Reference fallback rows use `accepted=null`. Accepted visual updates use
`accepted=true` and may carry coverage and width. Rejected updates use
`accepted=false`, deploy the exact fallback artifact, and set
`exact_fallback_reproduced=true`. Technical failures are retained as rejected
rows with a reason; they are never silently excluded.

Seal and evaluate the complete matrix:

```bash
prob4d experiment heldout-provider run promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results query-results.raw.json \
  --output-dir outputs/promotion \
  --require-pass
```

The command refuses an output directory containing any of its three retained
files, so a repeated invocation cannot silently replace an opened result. It
writes:

- `query_results.sealed.json`, with canonical ordering and a content identity;
- `promotion_report.json`, with both provider and physical-query decisions; and
- `promotion_report.md`, a compact gate table.

Without `--require-pass`, a scientifically valid negative result still returns
exit code 0 after writing all evidence. With it, a valid failed gate returns exit
code 3.

### 3. Replay independently

```bash
prob4d experiment heldout-provider verify promotion-lock.json \
  --provider-report outputs/provider/provider_evaluation.json \
  --query-results outputs/promotion/query_results.sealed.json \
  --report outputs/promotion/promotion_report.json
```

Verification revalidates every artifact and recomputes the deterministic report.
Any changed row, method set, target group, fallback identity, bootstrap setting,
or report field fails closed.

## Conjunctive physical-query gates

The primary candidate passes only when all of the following pass:

- the upper 95% paired target-group bootstrap bound clears the frozen superiority
  margin relative to physical fallback;
- harmful accepted updates do not exceed the frozen count;
- worst-group regression remains within its frozen limit;
- technical failures do not exceed their frozen limit;
- accepted-update mean coverage reaches the frozen threshold when one is set; and
- every rejected update reproduces the exact physical fallback.

Provider competence is a separate conjunctive gate. A good observation score does
not authorize a Bayesian update, and a guarded query result does not repair a
failed provider report.

## Claim boundary

A passing report supports only the exact frozen provider and guarded
BayesianPhysTwin query on the declared independent objects or sessions. It does
not establish general MotionCrafter competence, calibrated uncertainty outside
that cohort, Causal4D intervention benefit, or overall state of the art. A failed
well-powered gate is complete evidence and must not be retuned on the same opened
target cohort.
