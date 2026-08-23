# CUT3R source-comparison preflight

This capability revives the useful part of the abandoned
`agent/cut3r-source-comparison-preflight` branch on the current `main` line. It
performs an **outcome-blind retained-input check** between publication of the
frozen CUT3R source locks and any source-comparison inference.

It does not select a model, execute CUT3R inference, decode RGB frames, open
source predictions or residuals, inspect source truth values, or access any
confirmation or target object/session.

## What the preflight checks

For the exact request, source-freeze, comparison-specification, and
comparison-lock bytes, the builder:

- recomputes the canonical request, source-freeze, source-case, comparison-spec,
  and comparison-lock identities;
- revalidates the exact source-freeze information boundary, recurrent-online
  execution mode, single revisit, disabled global alignment, and disabled second
  pass;
- regenerates the canonical comparison lock from the retained specification and
  requires exact equality with the published lock;
- resolves exactly 40 frozen source video cases across 10 complete source groups;
- requires every case, group role, camera, video digest, and video byte count to
  agree across the source freeze and comparison lock;
- verifies each video digest and byte count before passing the file to `ffprobe`;
- verifies the three frozen input sidecars for every case by exact path, byte
  count, and SHA-256 digest;
- records container-level `ffprobe` metadata without decoding frames;
- requires the retained CUT3R checkout revision, normalized GitHub repository,
  checkpoint filename, checkpoint digest, and checkpoint byte count to match the
  frozen provider identity;
- requires the complete CUT3R worktree—including untracked files—to be clean
  before any provider Python is executed;
- authorizes executable probes only after all provider and checkpoint identities
  match, then resolves `demo.py` exclusively from Git's tracked-file inventory,
  requires one confined regular file, and checks its `--help` invocation and
  required Python imports; and
- inventories candidate source-reference file names, suffixes, and byte counts
  only inside each explicitly frozen source episode, without opening those file
  contents or traversing sibling target objects.

Untracked or symlinked `demo.py` files are never executed. Untracked Python files
cannot shadow the provider dependency probe because any untracked checkout
content blocks all executable probes. Raw remote URLs, absolute retained paths,
command output, and diagnostic text are not stored in the report. Diagnostic
output is redacted and retained only as a SHA-256 digest and byte count.

The report is content-addressed and returns one of two decisions:

- `source-comparison-preflight-ready`; or
- `technical-preflight-failure`.

A technical failure is retained evidence. It does not authorize changing the
frozen roster, deleting a camera, shortening a causal prefix, substituting a
checkpoint, or opening outcomes.

## Frozen request identity

The checked-in `preflight_request_id` is the SHA-256 digest of canonical JSON
after removing the `preflight_request_id` field itself. Runtime validation and
the hosted contract test both recompute that digest, so any request change
requires an explicit new identity.

The request binds the expected publication paths:

```text
protocols/locks/cut3r_deform360_source_freeze_v2.json
protocols/locks/cut3r_deform360_comparison_spec_v2.json
protocols/locks/cut3r_deform360_comparison_lock_v2.json
```

Those files are produced only after the currently authorized source-freeze
workflow succeeds and its exact-byte publication step completes. Missing,
symlinked, noncanonical, or mutually inconsistent lock bytes fail closed.

## Local or retained-runner invocation

After those lock files are present on the exact reviewed revision:

```bash
python scripts/science/build_cut3r_source_comparison_preflight.py \
  --repository . \
  --request \
    protocols/execution_requests/cut3r_deform360_source_comparison_preflight_v1.json \
  --processed-root "$DEFORM360_PROCESSED_ROOT" \
  --cut3r-checkout "$CUT3R_CHECKOUT" \
  --checkpoint "$CUT3R_CHECKPOINT" \
  --output outputs/cut3r/source-comparison-preflight.json
```

Exit status `0` means the preflight is ready. Exit status `3` means the
predeclared technical preflight failed. Output publication is atomic and
no-clobber: an identical existing report is accepted idempotently, while
different retained bytes are never overwritten.

## Execution boundary in this PR

The workflow added with this capability is intentionally **GitHub-hosted only**.
It validates the implementation, request identity, canonical lock contract,
action pins, and self-hosted-runner policy. It cannot enter a self-hosted runner
and cannot write an issue comment.

The capability can be merged before the retained lock files exist because runtime
validation requires those exact merged files and fails closed when they are
absent. A later exact-main execution route is justified only after the
source-freeze and comparison locks have been published. That route must follow
the repository's current read-only, exact-merged-main authorization pattern
rather than execute pull-request-controlled source on the retained runner.

## Claim boundary

This preflight establishes only that the frozen source files, lock identities,
and installed CUT3R surface can be resolved without opening outcomes. It does not
establish support, provider competence, uncertainty calibration, recurrent-state
recovery, BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state
of the art. Confirmation and target access remain forbidden.
