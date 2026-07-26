# Prob4D architecture and repository boundary

Prob4D is an optional 4-D perception provider. It estimates uncertain `Sim(3)`
gauges for independently decoded MotionCrafter windows, calibrates conditional
point uncertainty, and exports observations without performing a physical-state
update.

The intended dependency direction is:

```text
MotionCrafter / alternative 4-D provider
    -> Prob4D provider API
    -> phys4d.observation_belief or ObservationFactorBundle
    -> Bayesian-PhysTwin guarded physical belief
    -> conditional Causal4D intervention analysis
```

Prob4D must not depend on Bayesian-PhysTwin or Causal4D at runtime. Those
repositories may validate neutral artifacts independently, but should not import
Prob4D experiment helpers or underscore-prefixed modules.

## Stable provider surfaces

Downstream Python code should import `prob4d.provider_v1`. Version 1 exposes the
causal source selector, fixed metric-anchor contract, portable observation
belief, strict artifact loader, factor-bundle contract, and provider manifest.
A breaking change requires a new provider module rather than silently changing
version 1. Frozen experiments should still record exact repository commits and
input artifact hashes.

`prob4d provider manifest` provides the corresponding machine-readable
compatibility descriptor. The grouped `prob4d` CLI is the discoverable command
surface; legacy `prob4d-*` commands remain available for frozen run manifests.

## Causal information boundary

Predictive exports use an exclusive `causal_frame_stop`. A selected window must
be independently decoded and every source frame used by that window must satisfy

```text
source_frame < causal_frame_stop.
```

Selection occurs from manifest metadata before payload loading. Alignment, gauge
estimation, disagreement, uncertainty, and prior reliability are recomputed from
only the admitted prefix. Appending future manifest entries must not change the
selected-source digest or exported artifact.

## Covariance boundary

The production `ObservationBeliefV1` path uses a causal sequential gauge tree and
propagates the fixed metric-anchor uncertainty plus selected relative-gauge
uncertainty into one joint `7K x 7K` covariance. One shared low-rank latent factor
preserves the resulting cross-window covariance. Rank reduction is trace-audited
and fails closed below the declared retained-covariance threshold.

`local_covariance_m2` is conditional point covariance and must not include the
gauge contribution again. `ObservationFactorBundle` remains the richer interface
when a downstream estimator keeps explicit gauge nuisance variables.

The legacy fixed-lag covariance is an opt-in reconstruction control because its
current boundary treatment fixes marginalized gauges at posterior means. The
provider makes no prospective calibration or physical-twin-improvement claim.

## Immutable validated inputs

`PredictionWindow` defensively copies and freezes every NumPy field after
validation. Caller-side mutations therefore cannot alter a window after it has
entered a content-addressed workflow. Methods that intentionally provide mutable
values, such as `rays()`, return copies.

## Artifact ownership

Prob4D owns MotionCrafter prediction manifests, decoded-window payloads, gauge
and uncertainty calibration artifacts, portable observation beliefs,
observation-factor bundles, provider manifests, benchmarks, and run manifests.
Bayesian-PhysTwin owns physical priors, guarded updates, fallback behavior, and
accepted twin beliefs. Causal4D owns realized-intervention inference downstream
of an accepted, content-bound twin belief.

The `prob4d-phystwin*` commands are integration experiments, not the stable
provider interface. Paper-facing tables, figures, and sealed result manifests
belong in the corresponding paper or reproduction repository.
