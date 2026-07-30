# Prob4D observation-provider contract

`prob4d provider manifest` exposes the narrow producer capabilities that a
Bayesian estimator may rely on without importing Prob4D implementation modules.
The manifest records the installed package version, an optional exact Git
revision, supported artifact schema versions, positive capabilities, and
explicit nonclaims.

```bash
prob4d provider manifest \
  --api-version 2 \
  --provider-revision 0123456789abcdef0123456789abcdef01234567 \
  --output outputs/prob4d-provider-v2.json
```

## Python import boundary

Use the provider surface matching the experiment:

- `prob4d.provider_v1` is frozen for exact reproduction and historical artifacts;
- `prob4d.provider_v2` is the safe-by-default surface for new calibrated or
  explicitly exploratory development.

Provider v2 directly exposes its strict claim-bearing loader, explicit export
functions, schema-v4 factor bundle, and append-only factor-stream contract.
Downstream code should not import underscore-prefixed implementation modules or
Prob4D experiment helpers.

The Python call surface and the produced stream contract are versioned
independently. A breaking Python signature requires another provider module. A
breaking gauge, factor-group, covariance, reliability, or lineage interpretation
requires the corresponding artifact or stream-contract version to change. Exact
Git revisions and artifact hashes remain mandatory for frozen experiments.

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
  exact first prediction payload, external calibration artifact, case, and world
  frame.

A zero anchor covariance is declared as `fixed_external_calibration`. A nonzero
anchor covariance is declared as `propagated_external_prior` and is included in
the same joint latent factor as relative-gauge uncertainty. The producer and
both consumers reject an explicit v2 artifact when this treatment, calibration
digest, or inclusion flag is absent or inconsistent.

The production default is a causal sequential spanning tree. It preserves the
uncertainty of selected causal constraints without pretending that redundant
dense alignment edges are independent. Fixed-lag smoothing carries a
Schur-complement information prior across the moving boundary, but its portable
all-window covariance contains only block-diagonal historical marginals. It
therefore remains an explicit reconstruction control and is not labelled as
strict stream contract v2.

## Unfused and incremental observations

Provider v2 exposes `ObservationFactorBundle` schema v4 for consumers that keep
explicit gauge nuisance variables. It stores one ordered joint `7K x 7K` gauge
covariance and distinguishes `joint-cross-window` from `marginal-blocks-only`
semantics. Schema-v2/v3 bundles upgrade conservatively as marginal-only because
missing off-diagonal blocks cannot be reconstructed. The frozen provider-v1
surface continues to advertise and write schema v3.

`ObservationFactorStreamV1` binds several causally disjoint schema-v4 delta
bundles without rewriting old intervals. Update IDs cover bundle and payload
hashes, frame boundaries, observation-identity digests, gauge IDs, and the
previous update ID. See [append-only observation-factor streams](observation-factor-stream.md).

## Validation and nonclaims

Use `prob4d-validate-observation` to reject observation-schema drift, malformed
archives, and content-address mismatches. Use
`load_claim_bearing_observation_belief` from `prob4d.provider_v2` to require the
complete calibrated provider-v2 admission boundary.

The manifest does **not** claim that exported covariance has passed prospective
target calibration, that redundant dense-edge fusion is validated, or that a
valid observation artifact improves a Bayesian physical twin.

The grouped `prob4d` command is the preferred discoverable interface. New callers
must choose `export-calibrated`, `export-exploratory`, or `export-v1`. The bare
`prob4d observation export` route prints migration guidance and runs no exporter.
Existing standalone `prob4d-*` commands remain available for historical run
manifests.
