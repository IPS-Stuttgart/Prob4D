# DOT R04-R10 queued-run recovery v1

This operational helper addresses one retained scheduling state only: workflow
run `33363832286` was authorized from the frozen R04-R10 request, but its
self-hosted provider job remained queued before any provider step started.

The helper does **not** create a new scientific request. It may cancel and rerun
that exact workflow run only when all of the following remain true at execution
time:

- the run is still attempt 1 at head
  `bb2179158b27178c6ebed9be866bee829108b72a`;
- the provider job `Seal marker-free R04-R10 CUT3R predictions` is still queued
  and has no start time or steps;
- the run has published zero artifacts;
- the retained confirmation request is still
  `62d64df1b1b72f2b2aff0b17cf4c7aad245150f9fa1ff67712eedc0f4e109ce6`;
- no scientific input, provider payload, marker payload, or outcome is changed
  or opened by the recovery helper.

If any condition is false, recovery fails closed. In particular, a run that has
started, completed, published evidence, or already been retried is not touched.

The recovery workflow itself runs only on a GitHub-hosted runner and has no DOT
dataset path, GPU access, CUT3R checkout, checkpoint, marker reader, or
BayesianPhysTwin/Causal4D access. Its only write permission is GitHub Actions
control-plane access needed to cancel and rerun the exact retained workflow.

Triggering is separated from implementation. After this helper is reviewed and
merged, exactly one file may be added on `main`:

`protocols/execution_requests/dot_rope_cut3r_heldout_queued_recovery_v1.json`

That file is a content-addressed operational request. The push workflow verifies
that it is the only changed path before touching the retained run.

This recovery changes no scientific claim. The R04-R10 confirmation remains
owned by `protocols/dot-rope-cut3r-heldout-confirmation-v1.json`; R11-R70 remain
closed until the registered prerequisite decision permits otherwise.
