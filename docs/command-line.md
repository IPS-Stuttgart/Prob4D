# Command line

Prob4D 0.5 installs one executable:

```text
prob4d
```

Inspect the canonical registry with:

```bash
prob4d commands list
prob4d commands list --json
prob4d commands describe observation-export-calibrated --json
prob4d commands validate --json
```

Representative grouped routes include:

```text
prob4d ablate
prob4d benchmark
prob4d motioncrafter
prob4d observation export-calibrated
prob4d observation export-exploratory
prob4d observation validate
prob4d evaluate provider
prob4d experiment heldout-provider
prob4d provider manifest
prob4d provider target-admit
prob4d provider target-verify
prob4d prediction adapter-conformance
prob4d prediction readiness-matrix
prob4d phystwin evaluate
prob4d phystwin state
prob4d phystwin uncertainty
prob4d sintel uncertainty
prob4d storage benchmark
prob4d storage materialize
prob4d storage validate
prob4d vggt baseline
```

`prob4d prediction adapter-conformance` builds target-closed adapter requests and
checks exact repeatability, adapter-output order invariance, identity stability,
and causal-prefix invariance. `prob4d prediction readiness-matrix` freezes a
comparative provider program before source execution, composes the later
source-only readiness decisions, and authorizes at most one selected target
route. See [provider adapter SDK](provider-adapter-sdk.md) and
[provider-readiness matrix](provider-readiness-matrix.md).

The bare `prob4d observation export` route prints guidance and deliberately runs
no exporter. Claim-bearing work must select `export-calibrated`; uncalibrated or
alternate-method controls must select `export-exploratory`.

All historical `prob4d-*` aliases, `prob4d commands migrate`, and
`prob4d observation export-v1` were removed in 0.5. Frozen scripts that cannot be
migrated must pin Prob4D 0.4.1.
