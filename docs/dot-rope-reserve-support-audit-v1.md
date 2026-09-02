# Outcome-blind DOT reserve support qualification

Status: **completed support seal; no learned-provider or 3-D outcome result**.

## Purpose

The first CUT3R/DOT held-out confirmation terminated because one frozen
sequence lacked the preregistered provider/truth support. That support-negative
result remains terminal and is not reinterpreted. The untouched R11--R70
reserve therefore received a new support-only qualification before any normal
image, learned-provider prediction, or 3-D coordinate outcome was opened.

## Frozen audit

Protocol `protocols/dot-rope-reserve-support-audit-v1.json` registered:

- sequences R11 through R70;
- cameras `cam001` through `cam010`;
- frames 1 through 7;
- only publisher 2-D marker-coordinate files;
- stable coordinate-row index as the support identity;
- visibility requiring finite, nonnegative coordinates;
- common support as the intersection across all seven frames;
- deterministic camera selection by common count, minimum frame count, mean
  frame count, and finally the lowest camera index;
- qualification requiring at least eight common markers and at least eight
  visible markers in every frame;
- no replacement of unsupported sequences.

All six official reserve archives were resolved through the publisher's
Dataverse metadata, downloaded, and verified against their official byte counts
and MD5 checksums. The audit read exactly 4,200 two-dimensional coordinate
members: 60 sequences times 10 cameras times seven frames. It did not read a
3-D coordinate value, normal or UV image, provider output, residual, covariance
score, BayesianPhysTwin result, or Causal4D result.

## Result

All **60 of 60** reserve sequences satisfied the conservative support rule under
their deterministically selected camera. The complete per-camera table, one
selected camera per sequence, official archive receipts, exact information
boundary, and content-addressed support result are retained in
`evidence/dot-rope-reserve-support-audit-v1/`.

This result says only that a feasible observation/evaluation interface exists.
It does not imply that CUT3R will emit valid predictions at the marker support
or that any uncertainty method will improve a held-out score.

## Frozen learned-provider cohort

Before opening images or running CUT3R, the audit result was used to freeze a
computationally tractable 12-sequence cohort:

1. retain only support-qualified sequences whose deterministic selected camera
   is `cam001`;
2. within each of the six reserve archives, rank sequences by
   `SHA256(support_id:sequence)`;
3. select the first two sequences per archive;
4. reserve the remaining 48 sequences without replacement.

The exact roster, camera map, archive map, selection rule, information order,
and cohort identity are stored in
`evidence/dot-rope-reserve-support-audit-v1/provider-cohort.json`.

The next stage must retain the source-frozen CUT3R code, checkpoint, window
construction, covariance family, and source-selected dependence strength. It
must seal provider outputs and provider-side support before opening 3-D truth.
No target-side tuning or replacement is permitted.

## Claim boundary

This is outcome-blind support qualification and cohort freezing. It is not a
positive learned-provider result and is not part of the primary controlled
Tracking Cloth claim. Its purpose is to make a valid end-to-end confirmation
possible without changing the earlier support-negative outcome or selecting
sequences from provider residuals or 3-D target errors.