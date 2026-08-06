# Version and compatibility boundaries

Prob4D versions its Python providers separately from its portable artifact
contracts. Exact repository revisions and artifact digests remain mandatory for
frozen evidence even when package-version ranges are compatible.

## Prob4D 0.3.x surfaces

| Surface | Version | Intended use |
| --- | ---: | --- |
| `prob4d.source.Windowed4DSourceManifestV1` | 1 | Provider-neutral input normalization |
| `prob4d.contracts` | additive facade | Portable cross-repository contracts |
| `prob4d.provider_v1` | 1 | Frozen reproduction and provider-v1 compatibility |
| `prob4d.provider_v2` | 2 | New calibrated or explicitly exploratory development |
| `phys4d.observation_belief` | 1 | Portable fused observation container |
| Prob4D causal stream contract | 2 | Strict causal lineage and joint-gauge semantics |
| `ObservationFactorBundle` | 4 | Unfused factors with ordered joint gauge covariance |
| `ObservationFactorStreamV1` | 1 | Append-only sequence of causal schema-v4 delta bundles |
| MotionCrafter model identifier | 1 / 2 | Legacy common seed / derived per-call seed semantics |

Provider-v1 behavior and the standalone
`prob4d-export-observation-belief` executable remain available for frozen run
manifests. New work should choose an explicit provider-v2 export mode and use the
strict loader re-exported by `prob4d.provider_v2`.

The historical `prob4d.motioncrafter-model.v1` identifier covers the original
common-seed behavior, whether the manifest omits `seed_policy` or explicitly
uses `legacy-common`. A run using `derived-per-call` receives a
`prob4d.motioncrafter-model.v2` identifier, and its source-bound seed schedule is
validated before claim-bearing calibration compatibility is accepted. See
[the stochastic seed policy](stochastic-seed-policy.md).

## Grouped CLI migration

| Command | Meaning |
| --- | --- |
| `prob4d observation export-calibrated` | Claim-bearing provider-v2 export |
| `prob4d observation export-exploratory` | Labelled provider-v2 control |
| `prob4d observation export-v1` | Frozen provider-v1 grouped compatibility route |
| `prob4d observation export` | Migration guidance only; no exporter is run |

The deliberately ambiguous grouped command fails closed. Historical scripts may
continue using the unchanged standalone provider-v1 executable.

## Repository transfer identity

The canonical repository is `IPS-Stuttgart/Prob4D`, but frozen artifacts may
correctly retain `FlorianPfaff/Prob4D`. Repository owner/name strings must not be
rewritten inside content-addressed provider or observation artifacts.

Use:

```bash
prob4d project identity --compact
```

for the additive stable project descriptor. New orchestration metadata should
bind `github-repository-id:1295794737` as the durable project identity and record
the canonical repository separately for navigation. Existing provider-v1,
causal-stream, and provider-v2 artifact schemas retain their exact historical
repository semantics. See [repository identity](repository-identity.md).

## Companion projects

At the time Prob4D 0.3.0 was prepared, the companion package versions on their
main branches were:

- Bayesian-PhysTwin 0.4.0;
- Causal4D 0.4.1.

These numbers are development reference points, not substitutes for the
three-repository installed-wheel golden path. Claim-bearing runs must bind exact
Prob4D, Bayesian-PhysTwin, and Causal4D commits, wheel hashes, provider
manifests, artifacts, and protocol identifiers.

## Upgrade rules

- A breaking source-window, lineage, coordinate, or identity interpretation
  requires a new windowed 4-D source-contract version.
- A breaking Python signature requires a new provider module.
- A breaking Prob4D-specific interpretation of an observation artifact requires
  a new causal-stream contract version.
- A breaking factor-bundle representation requires a new bundle schema.
- A changed stream hash or interval interpretation requires a new factor-stream
  schema.
- A changed covariance or reliability fitting method requires regenerated,
  content-addressed calibration artifacts.
- A changed stochastic seed derivation requires a new MotionCrafter model
  identifier schema and regenerated calibration artifacts.
- A repository transfer alone does not permit rewriting a frozen artifact's
  source-repository field.

Passing compatibility tests is infrastructure evidence. It does not establish
accuracy, calibration, transfer, intervention benefit, or safety.
