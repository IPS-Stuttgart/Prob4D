# Contract portability

Prob4D publishes portable, typed, immutable artifacts for consumers that do not
share its source checkout. The `Contract portability` workflow closes two gaps
that ordinary Linux editable-install tests do not cover.

## Installed-wheel consumer typing

The workflow builds one wheel, installs it into an isolated environment together
with the authoritative MyPy version, and type-checks only the public
`prob4d.api.v2` consumer fixture. The positive fixture runs with strict checking,
`disallow-any-expr`, and `disallow-any-unimported` so a façade that silently
returns `Any` cannot pass merely because its re-export module is annotated.

A separate invalid fixture passes an integer to
`load_claim_bearing_observation_belief`. CI requires that fixture to fail with an
`arg-type` diagnostic. Together the two fixtures establish that the installed
`py.typed` distribution exposes concrete consumer-visible types and rejects an
incorrect call.

The typing fixtures must import only stable installed package surfaces. They must
not depend on repository-only helpers, editable installs, scientific data, or
private implementation modules.

## Immutable publication on supported hosted systems

The second job runs the focused immutable-publication, prediction-window storage,
and memory-mapped prediction-store tests on GitHub-hosted macOS and Windows. The
ordinary complete suite continues to cover Linux and every supported Python
version.

This matrix checks the actual hard-link no-clobber path, explicit atomic
replacement, temporary-file cleanup, NPZ round trips, and read-only memory maps.
A platform failure is treated as a portability failure; the implementation must
not silently weaken create-once semantics on that platform.

Network filesystems and externally mounted volumes may provide weaker semantics
than the hosted local filesystems. Claim-bearing deployments must retain the
existing write-then-reopen verification and should validate their actual storage
backend before treating it as immutable evidence storage.

## Security and scientific boundary

The workflow is read-only, uses pinned GitHub Actions, does not receive secrets,
and never opens provider, calibration, target, or physical-query data.

Passing establishes installed-package typing and local-filesystem portability
only. It does not establish provider accuracy, uncertainty calibration,
BayesianPhysTwin benefit, Causal4D intervention benefit, deployment safety, or
state of the art.
