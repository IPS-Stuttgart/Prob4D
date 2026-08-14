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

New integrations use the current `prob4d.api.v2` façade. It exposes the strict
provider-v2 loaders, calibrated and explicitly exploratory export contracts,
portable observation and factor records, sparse gauge priors, project identity,
and the serialization helpers required by downstream consumers. Breaking
changes require another versioned façade rather than silently changing v2.

`prob4d.provider_v1` in Prob4D 0.5 is only a narrow artifact-compatibility bridge
for immutable provider-v1 records, manifests, serializers, validators, and
schema-v3 factor IO retained by frozen evidence. It does not expose a provider-v1
estimator or exporter. Exact provider-v1 execution and reproduction require the
Prob4D 0.4.1 wheel or the corresponding tagged source revision.

Provider v2 separates exploratory and calibrated exports. Its calibrated entry
point validates the prediction manifest against both covariance calibrations
before opening decoded payloads, requires an exact Prob4D source revision, fixes
sequential gauge propagation, and forbids the pointwise covariance fallback. See
[Provider API version 2](provider-v2.md).

`prob4d provider manifest` emits the current provider-v2 capability descriptor.
The grouped `prob4d` CLI is the only installed executable in Prob4D 0.5; the
historical standalone `prob4d-*` commands are available only from frozen older
releases. Use `prob4d commands list` and `prob4d commands describe ...` to inspect
the canonical command surface.

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

Fixed-lag smoothing carries a Schur-complement information prior when gauges
leave the active window, so the moving boundary does not become exact. The
portable all-window covariance still exports only historical marginal blocks and
therefore remains an opt-in reconstruction control. The provider makes no
prospective calibration or physical-twin-improvement claim.

## Provider readiness boundary

Provider support, source mean quality, identity reliability, gauge dependence,
point covariance, downstream query relevance, and exact fallback are distinct,
ordered gates. The canonical grouped commands expose the existing immutable
readiness contracts without changing their scientific meaning:

```bash
prob4d diagnostic provider-support-envelope --help
prob4d diagnostic source-covariance-localization --help
prob4d provider prefix-admission --help
prob4d experiment fresh-provider-readiness --help
```

A richer point-uncertainty model is authorized only when source means and
identities pass, shared gauge/dependence uncertainty is adequate, and the
remaining failure is localized to conditional point covariance. A target
readiness authorization permits exactly one evaluation of the bound unopened
target roster; it is not evidence of provider competence or physical benefit.

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

The grouped `prob4d phystwin ...` commands are integration experiments, not the
stable provider interface. Paper-facing tables, figures, and sealed result
manifests belong in the corresponding paper or reproduction repository.
