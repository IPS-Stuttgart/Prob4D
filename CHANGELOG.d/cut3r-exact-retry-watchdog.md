### Fixed

- Preserve an exact, reviewable retry path for cancelled retained CUT3R
  source-freeze requests instead of requiring a new scientific request or silently
  executing newer development code.
- Reject incomplete retained-runner configuration before privileged scheduling and
  cancel a run when no matching GPU runner accepts the job within the bounded
  queue window.

### Scientific boundary

This is execution-control hardening only. It changes no provider, checkpoint,
cohort, prefix, support criterion, comparison arm, estimator, target-access state,
or scientific claim.
