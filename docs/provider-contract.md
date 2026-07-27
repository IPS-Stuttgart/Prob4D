# Prob4D observation-provider contract

`prob4d provider manifest` exposes the narrow producer capabilities that a
Bayesian estimator may rely on without importing Prob4D implementation modules.
The manifest records the installed package version, an optional exact Git
revision, supported artifact schema versions, positive capabilities, and
explicit nonclaims.

```bash
prob4d provider manifest \
  --provider-revision 0123456789abcdef0123456789abcdef01234567 \
  --output outputs/prob4d-provider.json
```

## Python import boundary

Downstream development code should import `prob4d.provider_v1` instead of
underscore-prefixed modules or experiment helpers. The Python call surface and
the produced stream contract are versioned independently:

- `provider_v1` identifies the stable Python signatures;
- `prob4d_causal_stream_contract_version` identifies the provider-specific
  interpretation of a strict observation artifact.

A breaking Python signature change requires a new provider module. A breaking
gauge, factor-group, or lineage interpretation requires a new stream-contract
version. Exact Git revisions and artifact hashes remain mandatory for frozen
experiments.

## Artifact semantics

The current strict causal stream contract is version 2. It declares that Prob4D
can:

- select independently decoded windows whose complete source interval precedes
  an exclusive causal cutoff;
- produce a content-addressed `ObservationBeliefV1` artifact and append-invariant
  causal-source digest;
- keep local conditional point covariance separate from shared gauge factors;
- propagate the metric-anchor prior and selected relative-gauge constraints into
  one joint cross-window `Sim(3)` covariance;
- export one shared low-rank latent factor whose window blocks preserve that
  covariance, with trace-audited rank reduction;
- retain association probability separately from residual-independent prior
  reliability; and
- bind metric coordinates to a content-addressed anchor that identifies the
  exact first prediction payload, exact external calibration artifact, case, and
  world frame.

A zero anchor covariance is declared as `fixed_external_calibration`. A nonzero
anchor covariance is declared as `propagated_external_prior` and is included in
the same joint latent factor as the relative-gauge uncertainty. The producer and
both consumers reject an explicit v2 artifact when this treatment, calibration
digest, or inclusion flag is absent or inconsistent.

The production default is a causal sequential spanning tree. It preserves the
uncertainty of the selected causal constraints without pretending that redundant
dense alignment edges are independent. Fixed-lag smoothing now carries a
Schur-complement information prior across the moving boundary, but its portable
all-window covariance contains only block-diagonal historical marginals. It
therefore remains an explicit reconstruction control and is not labelled as
strict stream contract v2.

Prob4D 0.2.0 artifacts that already contain canonical
`joint_gauge_latent_####` factors but predate the explicit version field can be
recognized by updated Bayesian-PhysTwin and Causal4D validators. Their validation
report marks the version as inferred. Newly exported production artifacts carry
the version, complete metric-anchor schema, calibration digest, and covariance
treatment explicitly.

The manifest does **not** claim that exported covariance has passed prospective
target calibration, that redundant dense-edge fusion is validated, or that a
valid observation artifact by itself improves a Bayesian physical twin.

Use `prob4d-validate-observation` to reject schema drift, array drift, malformed
archives, and content-address mismatches. Strict causal export writes through a
temporary archive, reloads and validates its content address, then atomically
replaces the requested output. `PredictionWindow` inputs are copied and frozen
after validation so caller-side mutation cannot alter a sealed source object.
The richer `ObservationFactorBundle` schema v3 remains available when the
consumer estimates gauge nuisance variables explicitly.

The grouped `prob4d` command is the preferred discoverable interface. Existing
`prob4d-*` commands remain available so historical run manifests and frozen
experiments retain their exact command lines.
