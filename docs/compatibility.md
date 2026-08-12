# Version and compatibility boundaries

Prob4D versions Python façades separately from portable artifact and wire
contracts. Exact repository revisions and artifact digests remain mandatory for
claim-bearing evidence even when released package versions are compatible.

## Prob4D 0.5 surfaces

| Surface | Status | Intended use |
| --- | --- | --- |
| `prob4d` | minimal root | package version only |
| `prob4d.api.v1` | removed | pin Prob4D 0.4.1 for full provider-v1 reproduction |
| `prob4d.api.v2` | current stable façade | BayesianPhysTwin and other provider-v2 consumers |
| `prob4d.provider_v1` | artifact compatibility bridge | historical schemas, manifests, serialization, and validation only |
| `prob4d.provider_v2` | implementation behind v2 | internal implementation and attested manifest identity |
| `phys4d.observation_belief` v1 | portable contract | neutral fused-observation interchange |
| Prob4D causal stream v2 | portable contract | strict causal lineage and joint-gauge semantics |
| `ObservationFactorBundle` v4 | portable contract | unfused factors with ordered joint gauge covariance |
| `ObservationFactorStreamV1` | portable contract | append-only causal schema-v4 updates |
| `GaugeTreeSquareRootPriorV1` | portable contract | causal tree transition/innovation prior |
| `TreeSparseObservationArtifactV1` | portable contract | tree-sparse factor and prior artifact |

New downstream code must use `prob4d.api.v2`. Direct imports from the package
root are no longer possible except for `__version__`.

The narrow `prob4d.provider_v1` module is not an execution surface. It keeps only
immutable v1 record types, schema-v3 factor IO, observation IO, causal binding,
and provider-manifest metadata needed to inspect or round-trip frozen evidence.
It exposes no provider-v1 estimator or export function. Full provider-v1
execution requires the exact Prob4D 0.4.1 distribution.

## Command-line boundary

Prob4D 0.5 installs only:

```text
prob4d
```

All historical `prob4d-*` console-script aliases and the `commands migrate`
operation were removed. Canonical grouped routes are discoverable with:

```bash
prob4d commands list
prob4d commands describe observation-export-calibrated
prob4d commands validate
```

A frozen workflow that requires an old executable name or provider-v1 exporter
must pin the Prob4D 0.4.1 wheel or exact source revision.

## Public API manifest

Manifest schema version 2 records only:

- package-root surface version 2 with `__version__`; and
- `prob4d.api.v2` as the current façade.

Schema-v1 manifests remain immutable evidence about 0.4.x installations. They
must not be rewritten or treated as current 0.5 manifests.

## Repository transfer identity

The canonical repository is `IPS-Stuttgart/Prob4D`, but frozen artifacts may
correctly retain `FlorianPfaff/Prob4D`. Repository owner/name strings must not be
rewritten inside content-addressed provider or observation artifacts.

Use:

```bash
prob4d project identity --compact
```

for the additive stable project descriptor. New orchestration metadata should
bind `github-repository-id:1295794737` and record the canonical repository
separately. Existing artifact schemas retain their exact historical repository
semantics.

## Companion-project compatibility

Ordinary development should consume released versioned façades and run the
three-repository installed-wheel compatibility capsule. Claim-bearing runs must
bind exact Prob4D, BayesianPhysTwin, and Causal4D revisions, wheel hashes,
provider manifests, contract-corpus identities, protocol identifiers, and input
and output artifact digests.

Compatibility tests and public API manifests are infrastructure evidence. They
do not establish accuracy, calibration, physical-query benefit, intervention
benefit, deployment safety, or state of the art.

## Upgrade rules

- A breaking current Python signature requires a new `prob4d.api.vN` façade.
- A breaking artifact interpretation requires a new contract or schema version.
- A changed normative corpus requires a new corpus version and bundle identity.
- A changed calibration method requires regenerated content-addressed artifacts.
- A repository transfer alone does not permit rewriting frozen provenance.
- Historical executables, provider-v1 execution, and broad root exports are not
  reintroduced after 0.5.
