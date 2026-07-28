# Prob4D observation-provider contract

`prob4d provider manifest` exposes the narrow producer capabilities that a
Bayesian estimator may rely on without importing Prob4D implementation modules.
The manifest records the installed package version, an optional exact Git
revision, supported artifact schema versions, positive capabilities, and explicit
nonclaims.

```bash
prob4d provider manifest \
  --provider-revision 0123456789abcdef0123456789abcdef01234567 \
  --output outputs/prob4d-provider.json
```

## Python import boundary

Existing experiments import `prob4d.provider_v1`. That surface remains frozen for
exact reproduction, including `ObservationFactorBundle` schema v3. New
claim-bearing development imports `prob4d.provider_v2`, which adds explicit
calibrated and exploratory observation exports, runtime/provider attestation, and
`JointObservationFactorBundle` schema v4.

The Python call surface and produced artifact contracts are versioned
independently:

- provider v1 identifies frozen signatures and schema-v3 factor compatibility;
- provider v2 identifies safe-by-default claim-bearing development surfaces;
- `prob4d_causal_stream_contract_version` identifies the provider-specific
  interpretation of a strict observation-belief artifact; and
- each observation-factor manifest declares its own schema version.

A breaking Python signature change requires a new provider module. A breaking
gauge, factor-group, or lineage interpretation requires a new artifact or stream
schema. Exact Git revisions and artifact hashes remain mandatory for frozen
experiments.

## Observation-belief semantics

The current strict causal stream contract is version 2. It declares that Prob4D
can:

- select independently decoded windows whose complete source interval precedes an
  exclusive causal cutoff;
- produce a content-addressed `ObservationBeliefV1` artifact and append-invariant
  causal-source digest;
- keep local conditional point covariance separate from shared gauge factors;
- propagate the metric-anchor prior and selected relative-gauge constraints into
  one joint cross-window `Sim(3)` covariance;
- export one shared low-rank latent factor whose window blocks preserve that
  covariance, with trace-audited rank reduction;
- retain association probability separately from residual-independent prior
  reliability; and
- bind metric coordinates to a content-addressed anchor that identifies the exact
  first prediction payload, external calibration artifact, case, and world frame.

A zero anchor covariance is declared as `fixed_external_calibration`. A nonzero
anchor covariance is declared as `propagated_external_prior` and is included in
the same joint latent factor as relative-gauge uncertainty. Producer and consumers
reject a version-2 artifact when this treatment, calibration digest, or inclusion
flag is absent or inconsistent.

The production default is a causal sequential spanning tree. It preserves the
uncertainty of selected causal constraints without pretending that redundant dense
alignment edges are independent. Fixed-lag smoothing carries a Schur-complement
information prior across the moving boundary, but its portable all-window
covariance contains only block-diagonal historical marginals. It therefore remains
an explicit reconstruction control and is not labelled as strict stream contract
version 2.

## Unfused factor semantics

`ObservationFactorBundle` schema v3 is the frozen provider-v1 representation. It
contains per-window gauge marginals only, so stacking reproduces the historical
block-diagonal gauge prior. It must not be relabelled as preserving cross-window
covariance.

Provider v2 additionally advertises `JointObservationFactorBundle` schema v4 and
the `joint_observation_factor_gauge_covariance` capability. Schema v4 carries one
ordered full `7K x 7K` gauge covariance under semantics
`ordered-full-cross-window-covariance-v1`. The manifest binds gauge order and the
payload checksum. Construction and loading validate:

- exact matrix dimension and finite values;
- symmetry and positive semidefiniteness;
- exact gauge ordering; and
- equality between each joint diagonal block and its `GaugeEstimate` marginal.

A consumer that keeps gauges explicitly uses conditional point covariance,
`gauge_jacobian`, and the full `gauge_prior_covariance`. A consumer that eliminates
gauges uses `R + J P_g J^T`. It must not add row-marginal gauge covariance and an
explicit gauge prior simultaneously.

## Compatibility and validation

Prob4D 0.2.0 observation-belief artifacts that contain canonical
`joint_gauge_latent_####` factors but predate the explicit stream-version field can
be recognized by updated Bayesian-PhysTwin and Causal4D validators. Their report
marks the version as inferred. Newly exported production artifacts carry the
version, complete metric-anchor schema, calibration digest, and covariance
treatment explicitly.

The observation-factor loader accepts schema versions 2, 3, and 4. Schema v2 is
upgraded from its inclusive frame limit and missing reliability fields. Schema v3
remains schema v3 after loading. Only schema v4 claims to carry a joint gauge prior.

The provider manifest does **not** claim that exported covariance has passed
prospective target calibration, that redundant dense-edge fusion is validated, or
that a valid observation artifact by itself improves a Bayesian physical twin.

Use `prob4d-validate-observation` to reject observation-belief schema drift, array
drift, malformed archives, and content-address mismatches. Observation-factor
manifests independently bind their NPZ payload with SHA-256 and disallow pickle.
`PredictionWindow` inputs are copied and frozen after validation so caller-side
mutation cannot alter a sealed source object.

The grouped `prob4d` command is the preferred discoverable interface. Existing
`prob4d-*` commands remain available so historical run manifests and frozen
experiments retain their exact command lines.
