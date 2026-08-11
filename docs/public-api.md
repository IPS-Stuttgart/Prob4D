# Public Python API

Prob4D has two different compatibility concerns:

1. experiment and research modules evolve with the active scientific programme;
2. downstream observation contracts must remain stable for BayesianPhysTwin,
   Causal4D, and independent providers.

Downstream projects should therefore import a versioned façade. Frozen
provider-v1 integrations retain:

```python
from prob4d.api import v1 as prob4d_api
```

New claim-bearing provider-v2 and explicit-gauge integrations should use:

```python
from prob4d.api import v2 as prob4d_api
```

The broad package root and implementation modules remain available for
interactive use and historical compatibility, but they are not supported
new-integration dependency boundaries. A breaking change requires a new module
such as `prob4d.api.v3`; it must not silently change `prob4d.api.v1` or
`prob4d.api.v2`.

## Lazy compatibility root

Importing `prob4d` now records the complete historical root export inventory
without importing calibration, fusion, gauge-graph, observation, prediction
storage, or reliability implementations. Accessing an exported attribute loads
its owning module once and caches the object in the package root.

The export set and object identities are unchanged. `dir(prob4d)`,
`from prob4d import Sim3`, and historical star imports continue to see the same
inventory. The packaged `prob4d/__init__.pyi` retains the complete static typing
surface even though runtime loading is lazy.

This optimization does not promote the broad root into a stable dependency
boundary. New integrations should still use a versioned façade.

## Version 1 boundary

`prob4d.api.v1` is the frozen provider-v1 reproduction surface. It exposes the
historical observation-belief and schema-v3 factor contracts, covariance
calibration artifacts, serialization helpers, and provider-v1 manifest.

## Version 2 boundary

`prob4d.api.v2` is the stable ecosystem surface for new calibrated or explicitly
exploratory provider-v2 work. It exposes:

- strict claim-bearing observation-belief loading and provider attestation;
- schema-v4 explicit-gauge factor bundles and append-only factor streams;
- sparse and tree-sparse execution contracts and strict claim-bearing loaders;
- portable causal gauge-tree priors and artifacts; and
- transfer-safe Prob4D project identity.

Experiment runners, command dispatch, MotionCrafter implementation internals,
provider-evaluation studies, and paper-specific evidence code are deliberately
outside the façade. BayesianPhysTwin may independently revalidate producer
artifacts and apply its own physical-update guards. Causal4D may continue to
validate neutral wire artifacts without importing Prob4D.

## Machine-readable inventory

The exact executing surfaces can be emitted as a content-addressed JSON artifact:

```bash
python -m prob4d.public_api_manifest print

python -m prob4d.public_api_manifest build \
  --output public-api-manifest.json

python -m prob4d.public_api_manifest verify \
  public-api-manifest.json \
  --require-current
```

The artifact records the package version, project identity, root loading mode,
root exports, versioned façade exports, and provider API versions. See
[public API manifest](public-api-manifest.md) for the schema and claim boundary.

## Ownership boundary

Prob4D owns calibrated probabilistic 4D observations, portable observation and
factor contracts, covariance calibration artifacts, provider manifests, and
strict serialization helpers.

BayesianPhysTwin owns physical-prior fusion, guarded state updates, and exact
fallback decisions. Causal4D owns realized-intervention abduction and
counterfactual prediction. Neither downstream repository should reinterpret raw
Prob4D covariance or depend on experiment-specific Prob4D modules.

## Version reporting

Runtime version reporting is resolved from installed distribution metadata. An
uninstalled source tree reports the explicit sentinel `0+unknown` rather than a
second hard-coded release number. Release versions remain declared once in
`pyproject.toml` and are checked against `CITATION.cff` during release validation.
