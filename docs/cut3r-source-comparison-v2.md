# Prospective CUT3R source-comparison v2

> **Terminally revoked.** The issue-triggered v2 route admitted a command that
> conflicted with the retained v1 no-retry boundary. Its queued empirical job
> was cancelled before runner acceptance, and the duplicate dispatch was
> skipped. The workflow is removed and must not be restored or dispatched.
> No v2 provider output or scientific result exists. The terminal metadata is
> preserved in
> `evidence/cut3r-source-comparison-v2-revocation/terminal_receipt.json`.

Version 1 of the frozen recurrent-online CUT3R source comparison terminated while
initializing Python imports. The exact retained record is
`evidence/cut3r-source-comparison-smoke-v1/summary.json`. It proves that the
attempt stopped before video verification, RGB decoding, CUT3R inference,
prediction writing, source truth or residual access, candidate-reference access,
target access, BayesianPhysTwin, or Causal4D.

The retained failure was therefore informative about execution plumbing but not
about any source case, provider output, metric, uncertainty, or scientific
outcome. Its no-retry rule remains binding for the v1 plan. The remainder of
this document preserves the withdrawn v2 design for provenance only; it is not
an executable protocol.

## Sole implementation change

The localized failure was CUT3R's internal package layout. CUT3R's repository
root was available, but `CUT3R/src` was not exposed as a top-level import path.
The v2 implementation:

- prepends the exact CUT3R repository root and exact `CUT3R/src` directory;
- imports CUT3R consistently through `dust3r.*`; and
- retains runtime-bootstrap failures as explicit zero-progress case records.

No provider revision, checkpoint, input video, source group, camera, frame range,
window, overlap, gauge estimator, alignment rule, point uncertainty, fusion mean,
confidence threshold, storage type, seed, comparator, support rule, failure rule,
or information boundary is selected from data or changed for scientific reasons.

## Prospective one-shot order

The exact issue-49 command admits one execution from merged `main`. The hosted
job first revalidates the byte-exact v1 terminal smoke, v1 plan, zero-information
boundary, and localized repair. Duplicate admission is rejected.

On `workstation2`, the read-only job then performs this fixed order:

1. bind runner identity, Linux/X64, `nvidia-smi`, and physical GPU 1;
2. locate the exact Python/package inventory already frozen by v1;
3. generate a new content-addressed execution plan from the merged v2 bytes,
   before opening any source RGB;
4. rehearse repaired imports and checkpoint/model loading without source RGB;
5. run exactly one smoke on the same frozen development case used by v1;
6. independently rehash and validate the smoke artifact;
7. only for an `ordinary-success` custody receipt, execute source shard 0;
8. independently validate shard 0;
9. only for a passing shard-0 receipt, execute source shard 1; and
10. independently validate shard 1 and publish a terminal content-addressed
    receipt.

The development smoke never counts as a source-shard case. Source shards use a
separate output root and execute every one of their frozen cases under the new
plan. Each v2 case is attempted once. A technical failure stops progression; no
case replacement, retry, alternate GPU, changed runtime, partial-group fit, or
threshold change is authorized.

## Publication and privacy boundary

A successful execution publishes the generated plan, smoke custody, both source
shards, both shard custody receipts, and a complete SHA-256 inventory. A failed
execution publishes only bounded manifests, tracebacks, reports, receipts, and
control-plane diagnostics; decoded RGB and dense partial prediction payloads are
not uploaded in a negative artifact.

This stage authorizes source RGB decoding and source prediction generation only.
It does not open source truth, physical residuals, candidate-reference file
contents, confirmation objects, target payloads or outcomes, BayesianPhysTwin
results, Causal4D results, deployment evidence, or state-of-the-art evidence.

Only a terminal `source-predictions-custody-complete` receipt can support the
separately frozen source-scoring stage. Source competence and downstream guarded
physical-query value remain separate gates.
