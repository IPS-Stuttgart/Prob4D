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
