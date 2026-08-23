# CUT3R source-comparison preflight

This capability revives the useful part of the abandoned
`agent/cut3r-source-comparison-preflight` branch on the current `main` line. It
performs an **outcome-blind retained-input check** between publication of the
frozen CUT3R source locks and any source-comparison inference.

It does not select a model, execute CUT3R, decode RGB frames, open source
predictions or residuals, inspect source truth values, or access any confirmation
or target object/session.

## What the preflight checks

For the exact source-freeze and comparison-lock bytes named by
`protocols/execution_requests/cut3r_deform360_source_comparison_preflight_v1.json`,
the builder:

- resolves the 40 frozen source video cases across 10 complete source groups;
- verifies every retained video byte count and SHA-256 digest;
- records container-level `ffprobe` metadata without decoding frames;
- records the exact CUT3R checkout revision and checkpoint identity;
- verifies that one `demo.py` surface is available and that its `--help`
  invocation and required Python imports succeed; and
- inventories candidate source-reference file names, suffixes, and byte counts
  without opening their contents.

The report is content-addressed and returns one of two decisions:

- `source-comparison-preflight-ready`; or
- `technical-preflight-failure`.

A technical failure is retained evidence. It does not authorize changing the
frozen roster, deleting a camera, shortening a causal prefix, substituting a
checkpoint, or opening outcomes.

## Frozen request identity

The checked-in `preflight_request_id` is the SHA-256 digest of canonical JSON
after removing the `preflight_request_id` field itself. The hosted contract test
recomputes that digest, so any request change requires an explicit new identity.

The request binds the expected publication paths:

```text
protocols/locks/cut3r_deform360_source_freeze_v2.json
protocols/locks/cut3r_deform360_comparison_spec_v2.json
protocols/locks/cut3r_deform360_comparison_lock_v2.json
```

Those files are produced only after the currently authorized source-freeze
workflow succeeds and its exact-byte publication step completes.

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
predeclared technical preflight failed.

## Execution boundary in this PR

The workflow added with this capability is intentionally **GitHub-hosted only**.
It validates the implementation, request identity, action pins, and
self-hosted-runner policy. It cannot enter a self-hosted runner and cannot write
an issue comment.

A later exact-main execution route is justified only after the source-freeze and
comparison locks have been published. That route must follow the repository's
current read-only, exact-merged-main authorization pattern rather than execute
pull-request-controlled source on the retained runner.

## Claim boundary

This preflight establishes only that the frozen source files and installed CUT3R
surface can be resolved without opening outcomes. It does not establish support,
provider competence, uncertainty calibration, recurrent-state recovery,
BayesianPhysTwin benefit, Causal4D benefit, deployment safety, or state of the
art. Confirmation and target access remain forbidden.
