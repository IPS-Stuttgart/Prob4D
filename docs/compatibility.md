# Version and compatibility boundaries

Prob4D versions its Python façades separately from portable artifact and wire
contracts. Exact repository revisions and artifact digests remain mandatory for
frozen evidence even when released package versions are compatible.

## Prob4D 0.4.x surfaces

| Surface | Version | Intended use |
| --- | ---: | --- |
| `prob4d.api.v1` | 1 | Stable frozen provider-v1 downstream façade |
| `prob4d.api.v2` | 2 | Stable claim-bearing provider-v2 downstream façade |
| `prob4d.provider_v1` | 1 | Frozen implementation and reproduction compatibility |
| `prob4d.provider_v2` | 2 | Provider-v2 implementation behind `prob4d.api.v2` |
| `phys4d.observation_belief` | 1 | Portable fused observation container |
| Prob4D causal stream contract | 2 | Strict causal lineage and joint-gauge semantics |
| `ObservationFactorBundle` | 4 | Unfused factors with ordered joint gauge covariance |
| `ObservationFactorStreamV1` | 1 | Append-only sequence of causal schema-v4 delta bundles |
| `GaugeTreeSquareRootPriorV1` | 1 | Portable causal tree transition/innovation prior |
| `TreeSparseObservationArtifactV1` | 1 | Portable tree-sparse factor and prior artifact |
| `ClaimBearingTreeSparseObservationEnvelopeV1` | 1 | Strict tree-sparse admission envelope |
| `prob4d.provider_v2_factors.v1` | 1 | Normative provider-v2 conformance corpus |
| MotionCrafter model identifier | 1 / 2 | Legacy common seed / derived per-call seed semantics |

Provider-v1 behavior and the standalone
`prob4d-export-observation-belief` executable remain available for frozen run
manifests. New work should choose an explicit provider-v2 export mode and import
the strict loaders and contracts through `prob4d.api.v2`. Direct imports from
`prob4d.provider_v2`, `prob4d.provider_v2_factors`, `prob4d.gauge`, or
`prob4d.sim3` are implementation dependencies rather than the supported
ecosystem boundary.

The broad `prob4d` root remains a historical compatibility surface. Its export
inventory is unchanged, but runtime attributes are now loaded lazily and its
complete static typing surface is packaged in `prob4d/__init__.pyi`. The
content-addressed public API manifest records the exact root, `api.v1`, and
`api.v2` inventories for an installed distribution. See
[public API manifest](public-api-manifest.md).

The historical `prob4d.motioncrafter-model.v1` identifier covers the original
common-seed behavior, whether the manifest omits `seed_policy` or explicitly
uses `legacy-common`. A run using `derived-per-call` receives a
`prob4d.motioncrafter-model.v2` identifier, and its source-bound seed schedule is
validated before claim-bearing calibration compatibility is accepted. See
[the stochastic seed policy](stochastic-seed-policy.md).

## Normative cross-repository corpora

Prob4D packages two separate data-only interoperability references:

- `phys4d.observation_belief` version 1 fixes the neutral fused-observation wire
  contract; and
- `prob4d.provider_v2_factors.v1` fixes the explicit-gauge and tree-sparse
  provider-v2 construction boundary.

BayesianPhysTwin and Causal4D may carry byte-identical corpus copies while
retaining independent validators. Exact corpus identity, structural semantics,
and declared numerical tolerances are conformance evidence. A green corpus does
not establish provider accuracy, calibration, physical-query benefit, or
intervention benefit.

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

## Companion-project compatibility

Do not maintain a table of mutable companion-repository branch versions in this
document. Such a table becomes stale whenever Prob4D, BayesianPhysTwin, or
Causal4D merges an unrelated change and can be mistaken for frozen scientific
provenance.

For ordinary development, consume released versioned façades and run the
three-repository installed-wheel compatibility capsule. For claim-bearing runs,
bind the exact Prob4D, BayesianPhysTwin, and Causal4D revisions, wheel hashes,
provider manifests, contract-corpus identities, protocol identifiers, and input
and output artifact digests inside the owning evidence record.

## Upgrade rules

- A breaking stable Python signature requires a new `prob4d.api.vN` façade.
- A breaking provider implementation signature that remains outside the stable
  façade requires a new provider module or an explicit internal migration.
- A breaking Prob4D-specific interpretation of an observation artifact requires
  a new causal-stream contract version.
- A breaking factor-bundle representation requires a new bundle schema.
- A changed tree-prior or tree-sparse representation requires a new artifact
  schema and corresponding provider capability.
- A changed normative corpus requires a new corpus version and bundle identity;
  one repository must not silently edit its local validator or vectors.
- A changed stream hash or interval interpretation requires a new factor-stream
  schema.
- A changed covariance or reliability fitting method requires regenerated,
  content-addressed calibration artifacts.
- A changed stochastic seed derivation requires a new MotionCrafter model
  identifier schema and regenerated calibration artifacts.
- A repository transfer alone does not permit rewriting a frozen artifact's
  source-repository field.

Passing compatibility tests or validating a public API manifest is
infrastructure evidence. It does not establish accuracy, calibration, transfer,
intervention benefit, deployment safety, or state of the art.
